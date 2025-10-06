#!/usr/bin/env python3
# convert_observational_to_dcdi.py
"""
Convert observational data and DAG to DCDI format.

This script takes observational data and a corresponding DAG, then converts them
to the DCDI (Discrete Causal Discovery with Interventions) format by creating
the necessary files and updating an Excel index.

Usage
-----
python data_dcdi.py  \
        --data   path/to/data.npy     \  # N × d samples  (or .csv → auto-detected)
        --dag    path/to/dag.npy      \  # d × d adjacency
        --outdir datasets             \  # root folder for graph_### sub-dirs
        --index  data/index.xlsx      \  # master Excel file (created if absent)
        --split  test                   # tag: train / val / test

Example
-------
python data_dcdi.py --data sergio_8000.csv --dag dag.npy --outdir datasets --split train
"""

import argparse
import os
import re
import numpy as np
import pandas as pd
import pathlib
from typing import Tuple

def next_graph_id(base: pathlib.Path) -> str:
    """Return the next unused 'graph_###' folder name."""
    existing = [int(m.group(1))
                for m in map(lambda p: re.match(r"graph_(\d+)", p.name),
                             base.glob("graph_*"))
                if m]
    return f"graph_{max(existing)+1:03d}" if existing else "graph_000"

def validate_data_and_dag(data_arr: np.ndarray, dag_arr: np.ndarray) -> None:
    """Validate that data and DAG are compatible."""
    if data_arr.ndim != 2:
        raise ValueError(f"Data must be 2D array, got shape {data_arr.shape}")
    
    if dag_arr.ndim != 2:
        raise ValueError(f"DAG must be 2D array, got shape {dag_arr.shape}")
    
    if dag_arr.shape[0] != dag_arr.shape[1]:
        raise ValueError(f"DAG must be square, got shape {dag_arr.shape}")
    
    if dag_arr.shape[0] != data_arr.shape[1]:
        raise ValueError(f"DAG dimensions ({dag_arr.shape[0]}) must match data columns ({data_arr.shape[1]})")
    
    # Check for NaN values
    if np.isnan(data_arr).any():
        raise ValueError("Data contains NaN values")
    
    if np.isnan(dag_arr).any():
        raise ValueError("DAG contains NaN values")

def save_dcdi_bundle(folder: pathlib.Path, data_arr: np.ndarray, dag_arr: np.ndarray) -> None:
    """
    Write DCDI format files: data.npy, regimes.csv, interventions.csv, graph.npy.
    
    For observational data:
    - regimes.csv: all zeros (observational regime)
    - interventions.csv: all -1 (no interventions)
    """
    folder.mkdir(parents=True, exist_ok=False)
    
    # Save data and graph
    np.save(folder / "data.npy", data_arr.astype(np.float32))
    np.save(folder / "graph.npy", dag_arr.astype(np.int8))

    N = data_arr.shape[0]
    
    # For observational data: all samples are in regime 0 (observational)
    pd.Series(np.zeros(N, dtype=int)).to_csv(
        folder / "regimes.csv", index=False, header=False
    )
    
    # For observational data: no interventions (all -1)
    pd.Series(-np.ones(N, dtype=int)).to_csv(
        folder / "interventions.csv", index=False, header=False
    )

def append_to_excel(index_fp: pathlib.Path, row: dict) -> None:
    """Append one row to an Excel sheet, creating it if necessary."""
    if index_fp.exists():
        df = pd.read_excel(index_fp)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    
    # Keep a stable column order
    cols = ["fp_data", "fp_graph", "fp_regime", "fp_intervention", "split", "n_samples", "n_variables"]
    df = df[[c for c in cols if c in df.columns] + [c for c in df.columns if c not in cols]]
    
    with pd.ExcelWriter(index_fp, engine="openpyxl", mode="w") as xls:
        df.to_excel(xls, index=False)

def load_matrix(path: pathlib.Path) -> np.ndarray:
    """Load matrix from various file formats."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    if path.suffix == ".npy":
        return np.load(path)
    elif path.suffix in {".csv", ".txt"}:
        return pd.read_csv(path, header=None if path.suffix == ".txt" else 0).to_numpy()
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}. Supported: .npy, .csv, .txt")

def main():
    ap = argparse.ArgumentParser(
        description="Convert observational data and DAG to DCDI format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    ap.add_argument("--data", required=True, type=pathlib.Path,
                   help="Path to data file (.npy, .csv, .txt)")
    ap.add_argument("--dag", required=True, type=pathlib.Path,
                   help="Path to DAG file (.npy, .csv, .txt)")
    ap.add_argument("--outdir", default="datasets", type=pathlib.Path,
                   help="Root folder for graph_### sub-directories")
    ap.add_argument("--index", default="data/index.xlsx", type=pathlib.Path,
                   help="Master Excel index file (created if absent)")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"],
                   help="Data split tag")
    
    args = ap.parse_args()

    try:
        # 1. Load arrays
        print(f"📁 Loading data from: {args.data}")
        data_arr = load_matrix(args.data)
        print(f"📁 Loading DAG from: {args.dag}")
        dag_arr = load_matrix(args.dag)
        
        # 2. Validate compatibility
        print("🔍 Validating data and DAG compatibility...")
        validate_data_and_dag(data_arr, dag_arr)
        print(f"✅ Data shape: {data_arr.shape}, DAG shape: {dag_arr.shape}")

        # 3. Create new graph folder
        graph_id = next_graph_id(args.outdir)
        graph_dir = args.outdir / graph_id
        print(f"📂 Creating DCDI bundle in: {graph_dir}")
        save_dcdi_bundle(graph_dir, data_arr, dag_arr)

        # 4. Append to Excel sheet
        row = dict(
            fp_data=str(graph_dir / "data.npy"),
            fp_graph=str(graph_dir / "graph.npy"),
            fp_regime=str(graph_dir / "regimes.csv"),
            fp_intervention=str(graph_dir / "interventions.csv"),
            split=args.split,
            n_samples=data_arr.shape[0],
            n_variables=data_arr.shape[1]
        )
        
        args.index.parent.mkdir(parents=True, exist_ok=True)
        append_to_excel(args.index, row)

        print(f"✅ Successfully converted to DCDI format!")
        print(f"   Graph ID: {graph_id}")
        print(f"   Files created: data.npy, graph.npy, regimes.csv, interventions.csv")
        print(f"   Index updated: {args.index}")
        print(f"   Split: {args.split}")

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
