#!/usr/bin/env python3
"""
Enhanced script to compare all result pickle files from SEA pipeline.

This script automatically finds all results_*.pkl files and provides
a comprehensive comparison of all methods in a single run.
"""


import os
import sys
import glob
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

from utils import read_pickle


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
            data = read_pickle(filepath)
            results[filepath] = data
            method_names[filepath] = method_name
        except Exception as e:
            print(f"  Warning: Failed to load {filepath}: {e}")
    
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
    auc_scores = data.get("auc", [])
    prc_scores = data.get("prc", [])
    times = data.get("time", [])
    
    # New metrics
    shd_scores = [s for s in data.get("shd", []) if s is not None]
    normalized_shd_scores = [s for s in data.get("normalized_shd", []) if s is not None]
    nnz_scores = [s for s in data.get("nnz", []) if s is not None]
    sid_scores = [s for s in data.get("sid", []) if s is not None]
    normalized_sid_scores = [s for s in data.get("normalized_sid", []) if s is not None]
    
    metrics = {
        "auc_mean": np.mean(auc_scores) if auc_scores else 0.0,
        "auc_std": np.std(auc_scores) if auc_scores else 0.0,
        "prc_mean": np.mean(prc_scores) if prc_scores else 0.0,
        "prc_std": np.std(prc_scores) if prc_scores else 0.0,
        "time_mean": np.mean(times) if times else 0.0,
        "time_std": np.std(times) if times else 0.0,
        "n_samples": len(auc_scores) if auc_scores else 0,
        
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
            times = data.get("time", [])
            shd_scores = [s for s in data.get("shd", []) if s is not None]
            normalized_shd_scores = [s for s in data.get("normalized_shd", []) if s is not None]
            nnz_scores = [s for s in data.get("nnz", []) if s is not None]
            sid_scores = [s for s in data.get("sid", []) if s is not None]
            normalized_sid_scores = [s for s in data.get("normalized_sid", []) if s is not None]
            
            all_auc.extend(auc_scores)
            all_prc.extend(prc_scores)
            all_times.extend(times)
            all_shd.extend(shd_scores)
            all_normalized_shd.extend(normalized_shd_scores)
            all_nnz.extend(nnz_scores)
            all_sid.extend(sid_scores)
            all_normalized_sid.extend(normalized_sid_scores)
            total_samples += len(auc_scores)
        
        summary_data.append({
            "Method": method_name,
            "Avg_AUC": np.mean(all_auc) if all_auc else 0.0,
            "AUC_std": np.std(all_auc) if all_auc else 0.0,
            "Avg_AUPRC": np.mean(all_prc) if all_prc else 0.0,
            "AUPRC_std": np.std(all_prc) if all_prc else 0.0,
            "Avg_Time": np.mean(all_times) if all_times else 0.0,
            "Time_std": np.std(all_times) if all_times else 0.0,
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
    formatted_df["AUC"] = formatted_df["Avg_AUC"].apply(lambda x: f"{x:.3f}") + " ± " + formatted_df["AUC_std"].apply(lambda x: f"{x:.3f}")
    formatted_df["AUPRC"] = formatted_df["Avg_AUPRC"].apply(lambda x: f"{x:.3f}") + " ± " + formatted_df["AUPRC_std"].apply(lambda x: f"{x:.3f}")
    formatted_df["Time"] = formatted_df["Avg_Time"].apply(lambda x: f"{x:.2f}") + " ± " + formatted_df["Time_std"].apply(lambda x: f"{x:.2f}")
    
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
    
    display_cols = ["Method", "AUC", "AUPRC", "Time", "SHD", "Norm_SHD", "NNZ"]
    if formatted_df["SID"].notna().any():
        display_cols.extend(["SID", "Norm_SID"])
    display_cols.extend(["Total_Samples", "Num_Datasets"])
    print(formatted_df[display_cols].to_string(index=False))
    
    # Find best performers
    print("\n" + "-"*80)
    print("BEST PERFORMERS:")
    print("-"*80)
    
    best_auc_idx = summary_df["Avg_AUC"].idxmax()
    best_prc_idx = summary_df["Avg_AUPRC"].idxmax()
    fastest_idx = summary_df["Avg_Time"].idxmin()
    
    print(f"Best AUC:  {summary_df.loc[best_auc_idx, 'Method']} ({summary_df.loc[best_auc_idx, 'Avg_AUC']:.3f})")
    print(f"Best AUPRC: {summary_df.loc[best_prc_idx, 'Method']} ({summary_df.loc[best_prc_idx, 'Avg_AUPRC']:.3f})")
    print(f"Fastest:   {summary_df.loc[fastest_idx, 'Method']} ({summary_df.loc[fastest_idx, 'Avg_Time']:.2f}s)")
    
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
    # Rank by Time (lower is better)
    summary_df["Time_Rank"] = summary_df["Avg_Time"].rank(ascending=True, method="min")
    
    # Rank by new metrics (lower is better for SHD/SID)
    if summary_df["Avg_SHD"].notna().any():
        summary_df["SHD_Rank"] = summary_df["Avg_SHD"].rank(ascending=True, method="min")
    if summary_df["Avg_Norm_SHD"].notna().any():
        summary_df["Norm_SHD_Rank"] = summary_df["Avg_Norm_SHD"].rank(ascending=True, method="min")
    if summary_df["Avg_SID"].notna().any():
        summary_df["SID_Rank"] = summary_df["Avg_SID"].rank(ascending=True, method="min")
    
    ranking_cols = ["Method", "AUC_Rank", "AUPRC_Rank", "Time_Rank"]
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