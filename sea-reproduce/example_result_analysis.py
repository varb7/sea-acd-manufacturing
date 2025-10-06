#!/usr/bin/env python3
"""
Example: How to interpret RFCI results from SEA pipeline.

This shows the built-in result interpretation capabilities.
"""

import os
import sys
import numpy as np
import pandas as pd

# Add src to path
sys.path.append('src')
from utils import read_pickle

def simple_results_analysis(results_file):
    """Simple analysis of results file."""
    
    # Load results
    print(f"Loading results from: {results_file}")
    results = read_pickle(results_file)
    
    print(f"\nFound {len(results)} datasets")
    print("="*50)
    
    # Analyze each dataset
    for key, data in results.items():
        print(f"\nDataset: {key}")
        print("-" * 30)
        
        # Extract metrics
        auc_scores = data.get("auc", [])
        prc_scores = data.get("prc", [])
        times = data.get("time", [])
        
        if auc_scores:
            print(f"AUC: {np.mean(auc_scores):.3f} ± {np.std(auc_scores):.3f}")
            print(f"AUPRC: {np.mean(prc_scores):.3f} ± {np.std(prc_scores):.3f}")
            print(f"Time: {np.mean(times):.2f} ± {np.std(times):.2f} seconds")
            print(f"Samples: {len(auc_scores)}")
        else:
            print("No metrics available")

def compare_algorithms(results_file1, results_file2, name1="Method 1", name2="Method 2"):
    """Compare two result files."""
    
    print(f"\nComparing {name1} vs {name2}")
    print("="*50)
    
    # Load both results
    results1 = read_pickle(results_file1)
    results2 = read_pickle(results_file2)
    
    # Find common datasets
    common_keys = set(results1.keys()) & set(results2.keys())
    
    if not common_keys:
        print("No common datasets found!")
        return
    
    print(f"Found {len(common_keys)} common datasets")
    
    # Compare metrics
    comparison_data = []
    
    for key in common_keys:
        data1 = results1[key]
        data2 = results2[key]
        
        auc1 = np.mean(data1.get("auc", [0]))
        auc2 = np.mean(data2.get("auc", [0]))
        prc1 = np.mean(data1.get("prc", [0]))
        prc2 = np.mean(data2.get("prc", [0]))
        time1 = np.mean(data1.get("time", [0]))
        time2 = np.mean(data2.get("time", [0]))
        
        comparison_data.append({
            "Dataset": key,
            f"{name1}_AUC": auc1,
            f"{name2}_AUC": auc2,
            f"{name1}_AUPRC": prc1,
            f"{name2}_AUPRC": prc2,
            f"{name1}_Time": time1,
            f"{name2}_Time": time2,
            "AUC_Diff": auc2 - auc1,
            "AUPRC_Diff": prc2 - prc1,
            "Time_Diff": time2 - time1
        })
    
    # Create comparison table
    df = pd.DataFrame(comparison_data)
    print("\nComparison Results:")
    print(df.to_string(index=False))
    
    # Summary statistics
    avg_auc_diff = df["AUC_Diff"].mean()
    avg_prc_diff = df["AUPRC_Diff"].mean()
    avg_time_diff = df["Time_Diff"].mean()
    
    print(f"\nSummary:")
    print(f"Average AUC difference ({name2} - {name1}): {avg_auc_diff:.3f}")
    print(f"Average AUPRC difference ({name2} - {name1}): {avg_prc_diff:.3f}")
    print(f"Average time difference ({name2} - {name1}): {avg_time_diff:.2f} seconds")
    
    return df

def extract_predictions(results_file, dataset_key=None):
    """Extract predictions and true graphs for detailed analysis."""
    
    results = read_pickle(results_file)
    
    if dataset_key and dataset_key in results:
        data = results[dataset_key]
        true_graphs = data.get("true", [])
        pred_graphs = data.get("pred", [])
        
        print(f"\nDataset: {dataset_key}")
        print(f"Number of samples: {len(true_graphs)}")
        
        if true_graphs and pred_graphs:
            print("True and predicted graphs available for analysis")
            return true_graphs, pred_graphs
        else:
            print("No graph data available")
            return None, None
    else:
        print("Available datasets:")
        for key in results.keys():
            print(f"  - {key}")
        return None, None

def main():
    """Example usage."""
    
    # Example 1: Analyze RFCI results
    rfci_results = "results_rfci.pkl"  # Your RFCI results file
    
    if os.path.exists(rfci_results):
        print("Example 1: Analyzing RFCI results")
        simple_results_analysis(rfci_results)
    else:
        print(f"RFCI results file not found: {rfci_results}")
    
    # Example 2: Compare RFCI with FCI baseline
    fci_results = "results_fci.pkl"  # FCI baseline results
    
    if os.path.exists(rfci_results) and os.path.exists(fci_results):
        print("\nExample 2: Comparing RFCI vs FCI")
        compare_algorithms(rfci_results, fci_results, "FCI", "RFCI")
    else:
        print("Cannot compare - missing result files")
    
    # Example 3: Extract predictions for detailed analysis
    if os.path.exists(rfci_results):
        print("\nExample 3: Extracting predictions")
        true_graphs, pred_graphs = extract_predictions(rfci_results)
        
        if true_graphs and pred_graphs:
            print("You can now analyze individual predictions:")
            print("- Compare predicted vs true adjacency matrices")
            print("- Compute edge-wise accuracy")
            print("- Analyze specific edge predictions")
            print("- Visualize graph structures")

if __name__ == "__main__":
    main()
