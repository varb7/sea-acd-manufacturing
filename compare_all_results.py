#!/usr/bin/env python3
"""
Enhanced script to compare all result pickle files from SEA pipeline.

This script automatically finds all results_*.pkl files and provides
a comprehensive comparison of all methods in a single run.
"""


import os
import sys
import glob
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path - handle both running from sea-reproduce and root directory
script_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(script_dir, 'src')
sea_reproduce_src = os.path.join(script_dir, 'sea-reproduce', 'src')

if os.path.exists(src_path):
    sys.path.insert(0, src_path)
elif os.path.exists(sea_reproduce_src):
    sys.path.insert(0, sea_reproduce_src)
else:
    # Try parent directory (if running from sea-reproduce)
    parent_src = os.path.join(os.path.dirname(script_dir), 'src')
    if os.path.exists(parent_src):
        sys.path.insert(0, parent_src)


def read_pickle_safe(fp):
    """
    Safely read pickle file, handling PyTorch tensors and other compiled extensions.
    Uses a mock torch module to prevent DLL import failures.
    """
    import sys
    from types import ModuleType
    
    # Save original torch if it exists
    original_torch = sys.modules.get('torch', None)
    torch_was_imported = 'torch' in sys.modules
    
    # Create a mock torch module to prevent actual import during unpickling
    class MockTensor:
        """Mock PyTorch tensor that can be converted to numpy."""
        def __init__(self, *args, **kwargs):
            # Try to extract data from various formats
            if args:
                if isinstance(args[0], (list, tuple)):
                    self._data = np.array(args[0])
                elif isinstance(args[0], np.ndarray):
                    self._data = args[0].copy()
                elif hasattr(args[0], 'numpy'):
                    try:
                        self._data = args[0].numpy()
                    except:
                        self._data = np.array([])
                else:
                    self._data = np.array(args[0]) if args else np.array([])
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
            return {'_data': self._data}
        
        def __setstate__(self, state):
            self._data = state.get('_data', np.array([]))
    
    # Create mock torch module
    mock_torch = ModuleType('torch')
    mock_torch.Tensor = MockTensor
    mock_torch.FloatTensor = MockTensor
    mock_torch.LongTensor = MockTensor
    mock_torch.IntTensor = MockTensor
    mock_torch.DoubleTensor = MockTensor
    
    # Install mock before loading
    sys.modules['torch'] = mock_torch
    
    try:
        # Now load the pickle - it will use our mock torch
        with open(fp, "rb") as f:
            data = pickle.load(f)
        
        # Convert any mock tensors to numpy arrays recursively
        def convert_tensors(obj):
            if isinstance(obj, dict):
                return {k: convert_tensors(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_tensors(item) for item in obj]
            elif isinstance(obj, MockTensor):
                return obj.numpy()
            elif hasattr(obj, 'numpy') and hasattr(obj, 'cpu'):
                try:
                    return obj.numpy()
                except:
                    return obj
            else:
                return obj
        
        data = convert_tensors(data)
        return data
        
    except Exception as e:
        # If that fails, try with a simpler approach - just ignore torch classes
        try:
            class IgnoreTorchUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    if module.startswith('torch'):
                        # Return a simple class that can hold data
                        class TorchPlaceholder:
                            def __init__(self, *args, **kwargs):
                                self.args = args
                                self.kwargs = kwargs
                                # Try to extract numpy data if present
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
            
            with open(fp, "rb") as f:
                unpickler = IgnoreTorchUnpickler(f)
                data = unpickler.load()
            
            # Convert placeholders to numpy
            def convert_placeholders(obj):
                if isinstance(obj, dict):
                    return {k: convert_placeholders(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_placeholders(item) for item in obj]
                elif hasattr(obj, 'data'):
                    return obj.data if hasattr(obj, 'data') else obj
                else:
                    return obj
            
            data = convert_placeholders(data)
            return data
            
        except Exception as e2:
            raise Exception(f"Failed to load pickle. Error 1: {e}, Error 2: {e2}")
    finally:
        # Restore original torch if it existed
        if torch_was_imported and original_torch is not None:
            sys.modules['torch'] = original_torch
        elif not torch_was_imported and 'torch' in sys.modules:
            # Remove mock if torch wasn't originally imported
            del sys.modules['torch']


def find_result_files(directory="."):
    """Find all result pickle files in the directory."""
    pattern = os.path.join(directory, "results_*.pkl")
    files = glob.glob(pattern)
    return sorted(files)


def extract_method_name(filename):
    """Extract method name from filename (e.g., 'results_rfci_pag.pkl' -> 'RFCI_PAG')."""
    basename = os.path.basename(filename)
    # Remove 'results_' prefix and '.pkl' suffix
    method = basename.replace("results_", "").replace(".pkl", "")
    # Convert to uppercase and replace underscores
    method = method.upper().replace("_", " ")
    return method


def load_all_results(result_files):
    """Load all result files and return a dictionary."""
    results = {}
    method_names = {}
    
    for filepath in result_files:
        method_name = extract_method_name(filepath)
        print(f"Loading: {os.path.basename(filepath)} -> {method_name}")
        try:
            data = read_pickle_safe(filepath)
            results[filepath] = data
            method_names[filepath] = method_name
            print(f"  Successfully loaded {len(data)} datasets")
        except Exception as e:
            print(f"  Warning: Failed to load {filepath}: {e}")
            import traceback
            traceback.print_exc()
    
    return results, method_names


def get_common_datasets(results_dict):
    """Find datasets common to all result files."""
    if not results_dict:
        return set()
    
    # Start with datasets from first file
    common = set(list(results_dict.values())[0].keys())
    
    # Intersect with all other files
    for results in results_dict.values():
        common = common & set(results.keys())
    
    return common


def compute_metrics(data):
    """Compute mean and std for metrics in a dataset."""
    auc_scores = [s for s in data.get("auc", []) if s is not None and not (isinstance(s, float) and np.isnan(s))]
    prc_scores = [s for s in data.get("prc", []) if s is not None and not (isinstance(s, float) and np.isnan(s))]
    f1_scores = [s for s in data.get("f1", []) if s is not None and not (isinstance(s, float) and np.isnan(s))]
    times = [s for s in data.get("time", []) if s is not None and not (isinstance(s, float) and np.isnan(s))]
    
    # New metrics
    shd_scores = [s for s in data.get("shd", []) if s is not None and not (isinstance(s, float) and np.isnan(s))]
    normalized_shd_scores = [s for s in data.get("normalized_shd", []) if s is not None and not (isinstance(s, float) and np.isnan(s))]
    nnz_scores = [s for s in data.get("nnz", []) if s is not None and not (isinstance(s, float) and np.isnan(s))]
    sid_scores = [s for s in data.get("sid", []) if s is not None and not (isinstance(s, float) and np.isnan(s))]
    normalized_sid_scores = [s for s in data.get("normalized_sid", []) if s is not None and not (isinstance(s, float) and np.isnan(s))]
    
    metrics = {
        "auc_mean": np.mean(auc_scores) if auc_scores else None,
        "auc_std": np.std(auc_scores) if auc_scores else None,
        "prc_mean": np.mean(prc_scores) if prc_scores else None,
        "prc_std": np.std(prc_scores) if prc_scores else None,
        "time_mean": np.mean(times) if times else None,
        "time_std": np.std(times) if times else None,
        "n_samples": len(auc_scores) if auc_scores else 0,

        # New metric
        "f1_mean": np.mean(f1_scores) if f1_scores else None,
        "f1_std": np.std(f1_scores) if f1_scores else None,
        
        # New metrics
        "shd_mean": np.mean(shd_scores) if shd_scores else None,
        "shd_std": np.std(shd_scores) if shd_scores else None,
        "normalized_shd_mean": np.mean(normalized_shd_scores) if normalized_shd_scores else None,
        "normalized_shd_std": np.std(normalized_shd_scores) if normalized_shd_scores else None,
        "nnz_mean": np.mean(nnz_scores) if nnz_scores else None,
        "nnz_std": np.std(nnz_scores) if nnz_scores else None,
        "sid_mean": np.mean(sid_scores) if sid_scores else None,
        "sid_std": np.std(sid_scores) if sid_scores else None,
        "normalized_sid_mean": np.mean(normalized_sid_scores) if normalized_sid_scores else None,
        "normalized_sid_std": np.std(normalized_sid_scores) if normalized_sid_scores else None,
    }
    
    return metrics


def create_comparison_table(results_dict, method_names, common_datasets):
    """Create a comprehensive comparison table."""
    comparison_data = []
    
    for dataset in sorted(common_datasets):
        row = {"Dataset": dataset}
        
        for filepath, results in results_dict.items():
            method_name = method_names[filepath]
            data = results[dataset]
            metrics = compute_metrics(data)
            
            # Add metrics with method name prefix
            row[f"{method_name}_AUC"] = metrics["auc_mean"]
            row[f"{method_name}_AUC_std"] = metrics["auc_std"]
            row[f"{method_name}_AUPRC"] = metrics["prc_mean"]
            row[f"{method_name}_AUPRC_std"] = metrics["prc_std"]
            row[f"{method_name}_F1"] = metrics["f1_mean"]
            row[f"{method_name}_F1_std"] = metrics["f1_std"]
            row[f"{method_name}_Time"] = metrics["time_mean"]
            row[f"{method_name}_Time_std"] = metrics["time_std"]
            row[f"{method_name}_N"] = metrics["n_samples"]
            
            # Add new metrics if available
            if metrics["shd_mean"] is not None:
                row[f"{method_name}_SHD"] = metrics["shd_mean"]
                row[f"{method_name}_SHD_std"] = metrics["shd_std"]
            if metrics["normalized_shd_mean"] is not None:
                row[f"{method_name}_Norm_SHD"] = metrics["normalized_shd_mean"]
                row[f"{method_name}_Norm_SHD_std"] = metrics["normalized_shd_std"]
            if metrics["nnz_mean"] is not None:
                row[f"{method_name}_NNZ"] = metrics["nnz_mean"]
                row[f"{method_name}_NNZ_std"] = metrics["nnz_std"]
            if metrics["sid_mean"] is not None:
                row[f"{method_name}_SID"] = metrics["sid_mean"]
                row[f"{method_name}_SID_std"] = metrics["sid_std"]
            if metrics["normalized_sid_mean"] is not None:
                row[f"{method_name}_Norm_SID"] = metrics["normalized_sid_mean"]
                row[f"{method_name}_Norm_SID_std"] = metrics["normalized_sid_std"]
        
        comparison_data.append(row)
    
    return pd.DataFrame(comparison_data)


def create_summary_table(results_dict, method_names, common_datasets):
    """Create a summary table with overall averages."""
    summary_data = []
    
    for filepath, results in results_dict.items():
        method_name = method_names[filepath]
        
        all_auc = []
        all_prc = []
        all_f1 = []
        all_times = []
        all_shd = []
        all_normalized_shd = []
        all_nnz = []
        all_sid = []
        all_normalized_sid = []
        total_samples = 0
        
        for dataset in common_datasets:
            data = results[dataset]
            auc_scores = data.get("auc", [])
            prc_scores = data.get("prc", [])
            f1_scores = data.get("f1", [])
            times = data.get("time", [])
            shd_scores = [s for s in data.get("shd", []) if s is not None]
            normalized_shd_scores = [s for s in data.get("normalized_shd", []) if s is not None]
            nnz_scores = [s for s in data.get("nnz", []) if s is not None]
            sid_scores = [s for s in data.get("sid", []) if s is not None]
            normalized_sid_scores = [s for s in data.get("normalized_sid", []) if s is not None]

            all_auc.extend(auc_scores)
            all_prc.extend(prc_scores)
            all_f1.extend(f1_scores)
            all_times.extend(times)
            all_shd.extend(shd_scores)
            all_normalized_shd.extend(normalized_shd_scores)
            all_nnz.extend(nnz_scores)
            all_sid.extend(sid_scores)
            all_normalized_sid.extend(normalized_sid_scores)
            total_samples += len(auc_scores)
        
        summary_data.append({
            "Method": method_name,
            "Avg_AUC": np.mean(all_auc) if all_auc else None,
            "AUC_std": np.std(all_auc) if all_auc else None,
            "Avg_AUPRC": np.mean(all_prc) if all_prc else None,
            "AUPRC_std": np.std(all_prc) if all_prc else None,
            "Avg_F1": np.mean(all_f1) if all_f1 else None,
            "F1_std": np.std(all_f1) if all_f1 else None,
            "Avg_Time": np.mean(all_times) if all_times else None,
            "Time_std": np.std(all_times) if all_times else None,
            "Avg_SHD": np.mean(all_shd) if all_shd else None,
            "SHD_std": np.std(all_shd) if all_shd else None,
            "Avg_Norm_SHD": np.mean(all_normalized_shd) if all_normalized_shd else None,
            "Norm_SHD_std": np.std(all_normalized_shd) if all_normalized_shd else None,
            "Avg_NNZ": np.mean(all_nnz) if all_nnz else None,
            "NNZ_std": np.std(all_nnz) if all_nnz else None,
            "Avg_SID": np.mean(all_sid) if all_sid else None,
            "SID_std": np.std(all_sid) if all_sid else None,
            "Avg_Norm_SID": np.mean(all_normalized_sid) if all_normalized_sid else None,
            "Norm_SID_std": np.std(all_normalized_sid) if all_normalized_sid else None,
            "Total_Samples": total_samples,
            "Num_Datasets": len(common_datasets)
        })
    
    return pd.DataFrame(summary_data)


def print_detailed_comparison(df, method_names):
    """Print detailed per-dataset comparison."""
    print("\n" + "="*80)
    print("DETAILED PER-DATASET COMPARISON")
    print("="*80)
    
    # Create a simplified view for readability
    methods = list(method_names.values())
    
    # Print AUC comparison
    print("\n--- AUC Scores (Mean ± Std) ---")
    auc_cols = ["Dataset"] + [f"{m}_AUC" for m in methods]
    if all(col in df.columns for col in auc_cols):
        auc_df = df[auc_cols].copy()
        # Format values
        for method in methods:
            col_mean = f"{method}_AUC"
            col_std = f"{method}_AUC_std"
            if col_mean in df.columns and col_std in df.columns:
                auc_df[method] = df[col_mean].apply(lambda x: f"{x:.3f}") + " ± " + df[col_std].apply(lambda x: f"{x:.3f}")
        display_cols = ["Dataset"] + methods
        print(auc_df[display_cols].to_string(index=False))
    
    # Print AUPRC comparison
    print("\n--- AUPRC Scores (Mean ± Std) ---")
    prc_cols = ["Dataset"] + [f"{m}_AUPRC" for m in methods]
    if all(col in df.columns for col in prc_cols):
        prc_df = df[prc_cols].copy()
        for method in methods:
            col_mean = f"{method}_AUPRC"
            col_std = f"{method}_AUPRC_std"
            if col_mean in df.columns and col_std in df.columns:
                prc_df[method] = df[col_mean].apply(lambda x: f"{x:.3f}") + " ± " + df[col_std].apply(lambda x: f"{x:.3f}")
        display_cols = ["Dataset"] + methods
        print(prc_df[display_cols].to_string(index=False))
    
    # Print F1 comparison
    print("\n--- F1 Scores (Mean ± Std) ---")
    f1_cols = ["Dataset"] + [f"{m}_F1" for m in methods]
    if all(col in df.columns for col in f1_cols):
        f1_df = df[f1_cols].copy()
        for method in methods:
            col_mean = f"{method}_F1"
            col_std = f"{method}_F1_std"
            if col_mean in df.columns and col_std in df.columns:
                def fmt(v):
                    return "N/A" if pd.isna(v) or v is None else f"{v:.3f}"
                f1_df[method] = df[col_mean].apply(fmt) + " ± " + df[col_std].apply(fmt)
        display_cols = ["Dataset"] + methods
        print(f1_df[display_cols].to_string(index=False))
    
    # Print Time comparison
    print("\n--- Time (seconds, Mean ± Std) ---")
    time_cols = ["Dataset"] + [f"{m}_Time" for m in methods]
    if all(col in df.columns for col in time_cols):
        time_df = df[time_cols].copy()
        for method in methods:
            col_mean = f"{method}_Time"
            col_std = f"{method}_Time_std"
            if col_mean in df.columns and col_std in df.columns:
                time_df[method] = df[col_mean].apply(lambda x: f"{x:.2f}") + " ± " + df[col_std].apply(lambda x: f"{x:.2f}")
        display_cols = ["Dataset"] + methods
        print(time_df[display_cols].to_string(index=False))
    
    # Print SHD comparison
    print("\n--- SHD (Mean ± Std, lower is better) ---")
    shd_cols = ["Dataset"] + [f"{m}_SHD" for m in methods]
    shd_available = any(f"{m}_SHD" in df.columns for m in methods)
    if shd_available:
        shd_df = df[["Dataset"]].copy()
        for method in methods:
            col_mean = f"{method}_SHD"
            if col_mean in df.columns:
                shd_df[method] = df[col_mean].apply(lambda x: f"{x:.1f}" if pd.notna(x) and x is not None else "N/A")
        print(shd_df.to_string(index=False))
    
    # Print Normalized SHD comparison
    print("\n--- Normalized SHD (Mean ± Std, lower is better) ---")
    norm_shd_cols = ["Dataset"] + [f"{m}_Norm_SHD" for m in methods]
    norm_shd_available = any(f"{m}_Norm_SHD" in df.columns for m in methods)
    if norm_shd_available:
        norm_shd_df = df[["Dataset"]].copy()
        for method in methods:
            col_mean = f"{method}_Norm_SHD"
            if col_mean in df.columns:
                norm_shd_df[method] = df[col_mean].apply(lambda x: f"{x:.3f}" if pd.notna(x) and x is not None else "N/A")
        print(norm_shd_df.to_string(index=False))
    
    # Print NNZ comparison
    print("\n--- NNZ (Number of Non-Zero Edges, Mean ± Std) ---")
    nnz_cols = ["Dataset"] + [f"{m}_NNZ" for m in methods]
    nnz_available = any(f"{m}_NNZ" in df.columns for m in methods)
    if nnz_available:
        nnz_df = df[["Dataset"]].copy()
        for method in methods:
            col_mean = f"{method}_NNZ"
            if col_mean in df.columns:
                nnz_df[method] = df[col_mean].apply(lambda x: f"{x:.1f}" if pd.notna(x) and x is not None else "N/A")
        print(nnz_df.to_string(index=False))
    
    # Print SID comparison (if available)
    print("\n--- SID (Mean ± Std, lower is better) ---")
    sid_cols = ["Dataset"] + [f"{m}_SID" for m in methods]
    sid_available = any(f"{m}_SID" in df.columns for m in methods)
    if sid_available:
        sid_df = df[["Dataset"]].copy()
        for method in methods:
            col_mean = f"{method}_SID"
            if col_mean in df.columns:
                sid_df[method] = df[col_mean].apply(lambda x: f"{x:.1f}" if pd.notna(x) and x is not None else "N/A")
        print(sid_df.to_string(index=False))


def print_summary(summary_df):
    """Print overall summary statistics."""
    print("\n" + "="*80)
    print("OVERALL SUMMARY (Average Across All Datasets)")
    print("="*80)
    
    # Format the summary table
    formatted_df = summary_df.copy()
    
    # Format AUC with NaN handling
    def format_with_std(mean_col, std_col, decimals=3):
        def formatter(row):
            mean_val = row[mean_col]
            std_val = row[std_col]
            if pd.isna(mean_val) or mean_val is None:
                return "N/A"
            if pd.isna(std_val) or std_val is None:
                std_val = 0.0
            return f"{mean_val:.{decimals}f} ± {std_val:.{decimals}f}"
        return formatter
    
    formatted_df["AUC"] = formatted_df.apply(format_with_std("Avg_AUC", "AUC_std", 3), axis=1)
    formatted_df["AUPRC"] = formatted_df.apply(format_with_std("Avg_AUPRC", "AUPRC_std", 3), axis=1)
    formatted_df["F1"] = formatted_df.apply(format_with_std("Avg_F1", "F1_std", 3), axis=1)
    formatted_df["Time"] = formatted_df.apply(format_with_std("Avg_Time", "Time_std", 2), axis=1)
    
    # Format new metrics (handle None values)
    def format_metric(mean_col, std_col):
        def formatter(row):
            if pd.isna(row[mean_col]) or row[mean_col] is None:
                return "N/A"
            mean_val = row[mean_col]
            std_val = row[std_col] if not pd.isna(row[std_col]) and row[std_col] is not None else 0.0
            if "SHD" in mean_col or "SID" in mean_col:
                return f"{mean_val:.2f} ± {std_val:.2f}"
            else:
                return f"{mean_val:.1f} ± {std_val:.1f}"
        return formatter
    
    formatted_df["SHD"] = formatted_df.apply(format_metric("Avg_SHD", "SHD_std"), axis=1)
    formatted_df["Norm_SHD"] = formatted_df.apply(format_metric("Avg_Norm_SHD", "Norm_SHD_std"), axis=1)
    formatted_df["NNZ"] = formatted_df.apply(format_metric("Avg_NNZ", "NNZ_std"), axis=1)
    formatted_df["SID"] = formatted_df.apply(format_metric("Avg_SID", "SID_std"), axis=1)
    formatted_df["Norm_SID"] = formatted_df.apply(format_metric("Avg_Norm_SID", "Norm_SID_std"), axis=1)
    
    display_cols = ["Method", "AUC", "AUPRC", "F1", "Time", "SHD", "Norm_SHD", "NNZ"]
    if formatted_df["SID"].notna().any():
        display_cols.extend(["SID", "Norm_SID"])
    display_cols.extend(["Total_Samples", "Num_Datasets"])
    print(formatted_df[display_cols].to_string(index=False))
    
    # Find best performers
    print("\n" + "-"*80)
    print("BEST PERFORMERS:")
    print("-"*80)
    
    # Handle NaN values - only find best if there are valid values
    if summary_df["Avg_AUC"].notna().any():
        best_auc_idx = summary_df["Avg_AUC"].idxmax()
        print(f"Best AUC:  {summary_df.loc[best_auc_idx, 'Method']} ({summary_df.loc[best_auc_idx, 'Avg_AUC']:.3f})")
    else:
        print("Best AUC:  N/A (no valid data)")
    
    if summary_df["Avg_AUPRC"].notna().any():
        best_prc_idx = summary_df["Avg_AUPRC"].idxmax()
        print(f"Best AUPRC: {summary_df.loc[best_prc_idx, 'Method']} ({summary_df.loc[best_prc_idx, 'Avg_AUPRC']:.3f})")
    else:
        print("Best AUPRC: N/A (no valid data)")
    
    if summary_df["Avg_Time"].notna().any():
        fastest_idx = summary_df["Avg_Time"].idxmin()
        print(f"Fastest:   {summary_df.loc[fastest_idx, 'Method']} ({summary_df.loc[fastest_idx, 'Avg_Time']:.2f}s)")
    else:
        print("Fastest:   N/A (no valid data)")
    
    # Best performers for new metrics (lower is better for SHD/SID)
    if summary_df["Avg_SHD"].notna().any():
        best_shd_idx = summary_df["Avg_SHD"].idxmin()
        print(f"Best SHD (lowest): {summary_df.loc[best_shd_idx, 'Method']} ({summary_df.loc[best_shd_idx, 'Avg_SHD']:.2f})")
    if summary_df["Avg_Norm_SHD"].notna().any():
        best_norm_shd_idx = summary_df["Avg_Norm_SHD"].idxmin()
        print(f"Best Norm SHD (lowest): {summary_df.loc[best_norm_shd_idx, 'Method']} ({summary_df.loc[best_norm_shd_idx, 'Avg_Norm_SHD']:.3f})")
    if summary_df["Avg_SID"].notna().any():
        best_sid_idx = summary_df["Avg_SID"].idxmin()
        print(f"Best SID (lowest): {summary_df.loc[best_sid_idx, 'Method']} ({summary_df.loc[best_sid_idx, 'Avg_SID']:.2f})")


def print_rankings(summary_df):
    """Print rankings for each metric."""
    print("\n" + "="*80)
    print("RANKINGS")
    print("="*80)
    
    # Rank by AUC (higher is better)
    summary_df["AUC_Rank"] = summary_df["Avg_AUC"].rank(ascending=False, method="min")
    # Rank by AUPRC (higher is better)
    summary_df["AUPRC_Rank"] = summary_df["Avg_AUPRC"].rank(ascending=False, method="min")
    # Rank by F1 (higher is better)
    summary_df["F1_Rank"] = summary_df["Avg_F1"].rank(ascending=False, method="min")
    # Rank by Time (lower is better)
    summary_df["Time_Rank"] = summary_df["Avg_Time"].rank(ascending=True, method="min")
    
    # Rank by new metrics (lower is better for SHD/SID)
    if summary_df["Avg_SHD"].notna().any():
        summary_df["SHD_Rank"] = summary_df["Avg_SHD"].rank(ascending=True, method="min")
    if summary_df["Avg_Norm_SHD"].notna().any():
        summary_df["Norm_SHD_Rank"] = summary_df["Avg_Norm_SHD"].rank(ascending=True, method="min")
    if summary_df["Avg_SID"].notna().any():
        summary_df["SID_Rank"] = summary_df["Avg_SID"].rank(ascending=True, method="min")
    
    ranking_cols = ["Method", "AUC_Rank", "AUPRC_Rank", "F1_Rank", "Time_Rank"]
    if "SHD_Rank" in summary_df.columns:
        ranking_cols.append("SHD_Rank")
    if "Norm_SHD_Rank" in summary_df.columns:
        ranking_cols.append("Norm_SHD_Rank")
    if "SID_Rank" in summary_df.columns:
        ranking_cols.append("SID_Rank")
    
    ranking_df = summary_df[ranking_cols].copy()
    ranking_df = ranking_df.sort_values("AUC_Rank")
    
    print("\nRanking by AUC (1 = best):")
    print(ranking_df[["Method", "AUC_Rank"]].to_string(index=False))
    
    print("\nRanking by AUPRC (1 = best):")
    print(ranking_df[["Method", "AUPRC_Rank"]].to_string(index=False))
    
    print("\nRanking by Time (1 = fastest):")
    print(ranking_df[["Method", "Time_Rank"]].to_string(index=False))
    
    if "SHD_Rank" in ranking_df.columns:
        print("\nRanking by SHD (1 = best, lowest):")
        print(ranking_df[["Method", "SHD_Rank"]].to_string(index=False))
    
    if "Norm_SHD_Rank" in ranking_df.columns:
        print("\nRanking by Normalized SHD (1 = best, lowest):")
        print(ranking_df[["Method", "Norm_SHD_Rank"]].to_string(index=False))
    
    if "SID_Rank" in ranking_df.columns:
        print("\nRanking by SID (1 = best, lowest):")
        print(ranking_df[["Method", "SID_Rank"]].to_string(index=False))


def main(directory="."):
    """Main function to compare all result files."""
    
    print("="*80)
    print("COMPREHENSIVE RESULTS COMPARISON")
    print("="*80)
    
    # Convert to absolute path
    directory = os.path.abspath(directory)
    print(f"Searching in directory: {directory}")
    
    # Find all result files
    result_files = find_result_files(directory)
    
    if not result_files:
        print(f"\nNo result files found in {directory}")
        print("Looking for files matching pattern: results_*.pkl")
        return
    
    print(f"\nFound {len(result_files)} result file(s):")
    for f in result_files:
        print(f"  - {os.path.basename(f)}")
    
    # Load all results
    print("\nLoading results...")
    results_dict, method_names = load_all_results(result_files)
    
    if not results_dict:
        print("No valid result files loaded!")
        return
    
    # Find common datasets
    common_datasets = get_common_datasets(results_dict)
    
    if not common_datasets:
        print("\nNo common datasets found across all result files!")
        print("Available datasets per file:")
        for filepath, results in results_dict.items():
            print(f"  {method_names[filepath]}: {list(results.keys())}")
        return
    
    print(f"\nFound {len(common_datasets)} common dataset(s):")
    for ds in sorted(common_datasets):
        print(f"  - {ds}")
    
    # Create comparison tables
    comparison_df = create_comparison_table(results_dict, method_names, common_datasets)
    summary_df = create_summary_table(results_dict, method_names, common_datasets)
    
    # Print results
    print_summary(summary_df)
    print_rankings(summary_df)
    print_detailed_comparison(comparison_df, method_names)
    
    # Save to CSV
    output_file = os.path.join(directory, "comparison_results.csv")
    print(f"\n" + "="*80)
    print(f"Saving detailed comparison to: {output_file}")
    comparison_df.to_csv(output_file, index=False)
    
    summary_file = os.path.join(directory, "comparison_summary.csv")
    print(f"Saving summary to: {summary_file}")
    summary_df.to_csv(summary_file, index=False)
    
    print("\nComparison complete!")


if __name__ == "__main__":
    # Allow specifying directory as command line argument
    # Default: look in parent directory (where pickle files usually are)
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        # Default to parent directory if running from sea-reproduce folder
        # Otherwise use current directory
        if os.path.basename(os.getcwd()) == "sea-reproduce":
            directory = ".."
        else:
            directory = "."
    main(directory)