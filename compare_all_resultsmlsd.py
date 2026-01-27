#!/usr/bin/env python3
"""
Multi-seed SEA results consolidator (+ global algorithm ranking).

What it does
------------
Given a parent experiment folder (e.g., node_variations/), which contains multiple
seed subfolders (e.g., node_variations_results_seed1 ... seed5), each with
results_*.pkl files, this script:

1) Loads all results_*.pkl per seed folder (safe unpickling even without torch).
2) Computes per-seed per-(dataset, algorithm) summary stats:
   - mean/std over trials inside that seed folder for each metric.
3) Aggregates across seeds for each (dataset, algorithm):
   - mean/std over the *seed means* for each metric (variation across seeds).
4) Produces a GLOBAL algorithm-level average across all datasets:
   - one row per algorithm (method), sorted by Avg_AUC (best on top).
5) Writes CSV outputs (long/tidy format, plus optional wide format).

Outputs
-------
- seed_level_results.csv
  One row per (seed, dataset, algorithm) with trial-aggregated metrics.

- seed_aggregated_results.csv
  One row per (dataset, algorithm) with mean/std across seeds and n_seeds used.

- seed_aggregated_results_wide.csv (optional)
  One row per dataset, with algorithm-specific columns.

- global_algorithm_ranking.csv
  One row per algorithm, global averages across datasets, sorted by Avg_AUC desc.

Usage examples
--------------
python consolidate_multi_seed_results.py --parent_dir node_variations
python consolidate_multi_seed_results.py --parent_dir node_variations --wide
python consolidate_multi_seed_results.py --parent_dir node_variations --include_methods "RFCI PAG" "FGES"
python consolidate_multi_seed_results.py --parent_dir node_variations --out_dir node_variations/csv_out
"""

import os
import sys
import glob
import pickle
import argparse
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Safe pickle reading (handles torch tensors or missing torch install)
# ---------------------------------------------------------------------
def read_pickle_safe(fp: str):
    """
    Safely read pickle file, handling PyTorch tensors and other compiled extensions.
    Uses a mock torch module to prevent DLL import failures during unpickling.
    """
    original_torch = sys.modules.get("torch", None)
    torch_was_imported = "torch" in sys.modules

    class MockTensor:
        """Mock PyTorch tensor that can be converted to numpy."""
        def __init__(self, *args, **kwargs):
            if args:
                a0 = args[0]
                if isinstance(a0, np.ndarray):
                    self._data = a0.copy()
                elif isinstance(a0, (list, tuple)):
                    self._data = np.array(a0)
                elif hasattr(a0, "numpy"):
                    try:
                        self._data = a0.numpy()
                    except Exception:
                        self._data = np.array([])
                else:
                    try:
                        self._data = np.array(a0)
                    except Exception:
                        self._data = np.array([])
            else:
                self._data = np.array([])

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self._data

        def tolist(self):
            return self._data.tolist()

        def __getstate__(self):
            return {"_data": self._data}

        def __setstate__(self, state):
            self._data = state.get("_data", np.array([]))

    mock_torch = ModuleType("torch")
    mock_torch.Tensor = MockTensor
    mock_torch.FloatTensor = MockTensor
    mock_torch.LongTensor = MockTensor
    mock_torch.IntTensor = MockTensor
    mock_torch.DoubleTensor = MockTensor

    sys.modules["torch"] = mock_torch

    try:
        with open(fp, "rb") as f:
            data = pickle.load(f)

        def convert_tensors(obj):
            if isinstance(obj, dict):
                return {k: convert_tensors(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_tensors(x) for x in obj]
            if isinstance(obj, MockTensor):
                return obj.numpy()
            if hasattr(obj, "numpy") and hasattr(obj, "cpu"):
                try:
                    return obj.numpy()
                except Exception:
                    return obj
            return obj

        return convert_tensors(data)

    except Exception as e:
        class IgnoreTorchUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module.startswith("torch"):
                    class TorchPlaceholder:
                        def __init__(self, *args, **kwargs):
                            self.args = args
                            self.kwargs = kwargs
                            if args and isinstance(args[0], np.ndarray):
                                self.data = args[0]
                            elif args and isinstance(args[0], (list, tuple)):
                                self.data = np.array(args[0])
                            else:
                                self.data = np.array([])

                        def numpy(self):
                            return self.data
                    return TorchPlaceholder
                return super().find_class(module, name)

        try:
            with open(fp, "rb") as f:
                unpickler = IgnoreTorchUnpickler(f)
                data = unpickler.load()

            def convert_placeholders(obj):
                if isinstance(obj, dict):
                    return {k: convert_placeholders(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [convert_placeholders(x) for x in obj]
                if hasattr(obj, "data"):
                    return getattr(obj, "data")
                return obj

            return convert_placeholders(data)
        except Exception as e2:
            raise RuntimeError(f"Failed to load pickle {fp}. Error 1: {e}; Error 2: {e2}") from e2
    finally:
        if torch_was_imported and original_torch is not None:
            sys.modules["torch"] = original_torch
        elif not torch_was_imported and "torch" in sys.modules:
            del sys.modules["torch"]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
DEFAULT_METRICS = [
    "auc", "prc", "time",
    # F1
    "f1_threshold", "f1_oracle_k",
    # SHD
    "shd_threshold", "shd_oracle_k",
    "normalized_shd_threshold", "normalized_shd_oracle_k",
    # NNZ
    "nnz_threshold", "nnz_oracle_k",
    # Precision/Recall
    "precision_threshold", "precision_oracle_k",
    "recall_threshold", "recall_oracle_k",
    # Other
    "m_true",
    "sid", "normalized_sid"
]


def standardize_metrics(dataset_dict: dict):
    """
    Ensure legacy metric names (f1, shd, etc.) are mapped to new threshold names
    if the new names are missing. This provides backward compatibility.
    """
    # Mapping: New Name -> Legacy Name
    # We only copy legacy -> new if new is missing.
    mappings = {
        "f1_threshold": "f1",
        "shd_threshold": "shd",
        "normalized_shd_threshold": "normalized_shd",
        "nnz_threshold": "nnz",
        # We don't map oracle_k because legacy didn't have it (or it was just different)
    }

    for new_key, legacy_key in mappings.items():
        if new_key not in dataset_dict and legacy_key in dataset_dict:
            dataset_dict[new_key] = dataset_dict[legacy_key]
    
    return dataset_dict



def find_result_files(directory: str):
    return sorted(glob.glob(os.path.join(directory, "results_*.pkl")))


def extract_method_name(filename: str) -> str:
    basename = os.path.basename(filename)
    method = basename.replace("results_", "").replace(".pkl", "")
    method = method.upper().replace("_", " ")
    return method


def parse_seed_from_folder(folder_name: str) -> str:
    import re
    lower = folder_name.lower()
    m = re.search(r"(seed[-_ ]?\d+)", lower)
    if m:
        return m.group(1).replace(" ", "")
    return folder_name


def clean_numeric_list(values):
    out = []
    for v in values if isinstance(values, (list, tuple)) else []:
        if v is None:
            continue
        if isinstance(v, (float, np.floating)) and np.isnan(v):
            continue
        try:
            if not np.isfinite(v):
                continue
        except Exception:
            pass
        out.append(v)
    return out


def compute_within_seed_stats(dataset_dict: dict, metrics=DEFAULT_METRICS):
    out = {}
    for m in metrics:
        vals = clean_numeric_list(dataset_dict.get(m, []))
        out[f"{m}_mean"] = float(np.mean(vals)) if vals else np.nan
        out[f"{m}_std"] = float(np.std(vals)) if vals else np.nan
        out[f"{m}_n"] = int(len(vals)) if vals else 0
    return out


def load_seed_folder(seed_dir: str, include_methods=None, exclude_methods=None, verbose=False):
    seed_dir = os.path.abspath(seed_dir)
    seed_label = parse_seed_from_folder(os.path.basename(seed_dir))

    files = find_result_files(seed_dir)
    if not files:
        if verbose:
            print(f"[WARN] No results_*.pkl found in {seed_dir}")
        return []

    rows = []
    for fp in files:
        method = extract_method_name(fp)

        if include_methods is not None and method not in include_methods:
            continue
        if exclude_methods is not None and method in exclude_methods:
            continue

        if verbose:
            print(f"  Loading {os.path.basename(fp)} as method='{method}'")

        try:
            data = read_pickle_safe(fp)
        except Exception as e:
            print(f"[WARN] Failed to load {fp}: {e}")
            continue

        if not isinstance(data, dict):
            print(f"[WARN] Unexpected pickle structure in {fp}: expected dict, got {type(data)}")
            continue

        for dataset_name, dataset_dict in data.items():
            if not isinstance(dataset_dict, dict):
                continue

            # Standardize metrics (handle legacy vs new names)
            dataset_dict = standardize_metrics(dataset_dict)
            
            stats = compute_within_seed_stats(dataset_dict)

            row = {
                "seed": seed_label,
                "seed_dir": seed_dir,
                "method": method,
                "dataset": dataset_name,
            }
            row.update(stats)
            rows.append(row)

    return rows


def aggregate_across_seeds(seed_level_df: pd.DataFrame, metrics=DEFAULT_METRICS):
    agg_rows = []
    for (dataset, method), g in seed_level_df.groupby(["dataset", "method"], dropna=False):
        row = {"dataset": dataset, "method": method}
        row["n_seeds"] = int(g["seed"].nunique())

        for m in metrics:
            means = g[f"{m}_mean"].astype(float).values
            means = means[np.isfinite(means)]
            row[f"{m}_mean_over_seeds"] = float(np.mean(means)) if means.size else np.nan
            row[f"{m}_std_over_seeds"] = float(np.std(means)) if means.size else np.nan

            within_stds = g[f"{m}_std"].astype(float).values
            within_stds = within_stds[np.isfinite(within_stds)]
            row[f"{m}_avg_within_seed_std"] = float(np.mean(within_stds)) if within_stds.size else np.nan

            n_trials = g[f"{m}_n"].astype(int).values
            row[f"{m}_avg_n_trials"] = float(np.mean(n_trials)) if n_trials.size else 0.0

        agg_rows.append(row)

    return pd.DataFrame(agg_rows)


def compute_global_algorithm_ranking(agg_df: pd.DataFrame, metrics=DEFAULT_METRICS):
    """
    Global average across *datasets* for each method, using per-dataset seed-aggregated means.

    For each method:
      Avg_AUC = mean over datasets of auc_mean_over_seeds
      AUC_std_over_datasets = std over datasets of auc_mean_over_seeds
      ... similarly for other metrics.

    Result is sorted by Avg_AUC desc (best on top).
    """
    rows = []
    for method, g in agg_df.groupby("method", dropna=False):
        row = {"Method": method, "Num_Datasets": int(g["dataset"].nunique())}

        for m in metrics:
            col = f"{m}_mean_over_seeds"
            vals = g[col].astype(float).values if col in g.columns else np.array([])
            vals = vals[np.isfinite(vals)]
            row[f"Avg_{m.upper()}"] = float(np.mean(vals)) if vals.size else np.nan
            row[f"{m.upper()}_std_over_datasets"] = float(np.std(vals)) if vals.size else np.nan

        rows.append(row)

    global_df = pd.DataFrame(rows)

    # sort by AUC (descending)
    if "Avg_AUC" in global_df.columns:
        global_df = global_df.sort_values("Avg_AUC", ascending=False, na_position="last").reset_index(drop=True)
    else:
        global_df = global_df.sort_values("Method").reset_index(drop=True)

    return global_df


def to_wide_format(agg_df: pd.DataFrame, metrics=DEFAULT_METRICS):
    df = agg_df.copy().sort_values(["dataset", "method"]).reset_index(drop=True)

    value_cols = []
    for m in metrics:
        value_cols.extend([
            f"{m}_mean_over_seeds",
            f"{m}_std_over_seeds",
            f"{m}_avg_within_seed_std",
            f"{m}_avg_n_trials",
        ])
    value_cols.append("n_seeds")

    wide = df.pivot(index="dataset", columns="method", values=value_cols)
    wide.columns = [f"{method}__{col}" for col, method in wide.columns]
    wide = wide.reset_index()
    return wide


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Consolidate SEA results across multiple seed subfolders (+ global algorithm ranking).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--parent_dir", type=str, required=True,
                    help="Parent directory that contains seed subfolders.")
    ap.add_argument("--seed_glob", type=str, default="*seed*",
                    help="Glob pattern to detect seed subfolders under parent_dir.")
    ap.add_argument("--out_dir", type=str, default=None,
                    help="Output directory for CSVs. Defaults to parent_dir.")
    ap.add_argument("--wide", action="store_true",
                    help="Also write wide-format aggregated CSV.")
    ap.add_argument("--include_methods", nargs="*", default=None,
                    help="Only include these methods (names like 'RFCI PAG', 'FGES').")
    ap.add_argument("--exclude_methods", nargs="*", default=None,
                    help="Exclude these methods.")
    ap.add_argument("--verbose", action="store_true",
                    help="Verbose logging.")
    args = ap.parse_args()

    parent_dir = os.path.abspath(args.parent_dir)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else parent_dir
    os.makedirs(out_dir, exist_ok=True)

    seed_dirs = sorted([p for p in glob.glob(os.path.join(parent_dir, args.seed_glob)) if os.path.isdir(p)])
    if not seed_dirs:
        print(f"[ERROR] No seed directories found under: {parent_dir}")
        print(f"        Using seed_glob: {args.seed_glob}")
        sys.exit(1)

    if args.verbose:
        print(f"Found {len(seed_dirs)} seed directories:")
        for d in seed_dirs:
            print(f"  - {d}")

    include_methods = set(args.include_methods) if args.include_methods else None
    exclude_methods = set(args.exclude_methods) if args.exclude_methods else None

    all_rows = []
    print("=" * 80)
    print("LOADING SEED FOLDERS")
    print("=" * 80)
    for sd in seed_dirs:
        print(f"\nSeed folder: {sd}")
        rows = load_seed_folder(sd, include_methods=include_methods, exclude_methods=exclude_methods, verbose=args.verbose)
        print(f"  -> loaded {len(rows)} (seed,dataset,method) rows")
        all_rows.extend(rows)

    if not all_rows:
        print("[ERROR] No results could be loaded from any seed directory.")
        sys.exit(1)

    seed_level_df = pd.DataFrame(all_rows)
    agg_df = aggregate_across_seeds(seed_level_df)

    seed_level_df = seed_level_df.sort_values(["dataset", "method", "seed"]).reset_index(drop=True)
    agg_df = agg_df.sort_values(["dataset", "method"]).reset_index(drop=True)

    # GLOBAL algorithm ranking across datasets (sorted by AUC)
    global_df = compute_global_algorithm_ranking(agg_df)

    # Write outputs
    seed_level_csv = os.path.join(out_dir, "seed_level_results.csv")
    agg_csv = os.path.join(out_dir, "seed_aggregated_results.csv")
    global_csv = os.path.join(out_dir, "global_algorithm_ranking.csv")

    seed_level_df.to_csv(seed_level_csv, index=False)
    agg_df.to_csv(agg_csv, index=False)
    global_df.to_csv(global_csv, index=False)

    print("\n" + "=" * 80)
    print("WRITTEN OUTPUTS")
    print("=" * 80)
    print(f"Seed-level CSV:            {seed_level_csv}")
    print(f"Aggregated per-dataset CSV:{agg_csv}")
    print(f"Global algorithm ranking:  {global_csv}")

    if args.wide:
        wide_df = to_wide_format(agg_df)
        wide_csv = os.path.join(out_dir, "seed_aggregated_results_wide.csv")
        wide_df.to_csv(wide_csv, index=False)
        print(f"Aggregated wide CSV:       {wide_csv}")

    # Print top/bottom quick view (sorted by AUC)
    print("\n" + "=" * 80)
    print("GLOBAL RANKING (sorted by Avg_AUC desc)")
    print("=" * 80)
    cols = [
        "Method", 
        "Avg_AUC", "AUC_std_over_datasets", 
        "Avg_PRC", 
        "Avg_F1_THRESHOLD", "Avg_F1_ORACLE_K",
        "Avg_SHD_THRESHOLD",
        "Avg_TIME", 
        "Num_Datasets"
    ]
    cols = [c for c in cols if c in global_df.columns]
    if len(global_df) > 0:
        print(global_df[cols].to_string(index=False))
    else:
        print("No global ranking rows produced.")

    print("\nDone.")


if __name__ == "__main__":
    main()
