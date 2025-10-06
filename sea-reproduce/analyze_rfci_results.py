#!/usr/bin/env python3
"""
Analyze RFCI results from SEA pipeline.

This script loads pickle results files and provides comprehensive analysis
including metrics, visualizations, and comparisons.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.metrics import roc_auc_score, average_precision_score

# Add src to path for utils
sys.path.append('src')
from utils import read_pickle

def load_results(results_file):
    """Load results from pickle file."""
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    print(f"Loading results from: {results_file}")
    results = read_pickle(results_file)
    print(f"Found {len(results)} datasets")
    return results

def compute_additional_metrics(true, pred, threshold=0.5):
    """Compute SHD and edge accuracy metrics."""
    def shd_metric(pred_bin, true_labels):
        """Structural Hamming Distance."""
        diff = true_labels - pred_bin
        rev = (((diff + diff.T) == 0) & (diff != 0)).sum() / 2
        fn = (diff == 1).sum() - rev
        fp = (diff == -1).sum() - rev
        return fn + fp + rev
    
    def to_2d(a):
        """Convert 1D edge list to 2D adjacency matrix."""
        options = [10, 11, 20, 100, 200, 300, 400, 500, 1000]
        for n in options:
            if len(a) == n*(n-1):
                break
        
        mask = np.tri(n, k=-1, dtype=bool)
        halfway = n*(n-1)//2
        
        g1 = np.zeros((n, n))
        g2 = np.zeros((n, n))
        g1[mask] = a[:halfway]
        g2[mask] = a[halfway:]
        
        g = g1 + g2.T
        return g, a[:halfway], a[halfway:]
    
    def to_1d(a):
        """Convert 2D adjacency to 1D edge lists."""
        n = a.shape[0]
        mask = np.tri(n, k=-1, dtype=bool)
        forward = a[mask]
        backward = a.T[mask]
        return forward, backward
    
    # Convert to 2D and get aligned edges
    true, pred = np.array(true, dtype=int), np.array(pred)
    if true.ndim == 1:
        true, true_f, true_b = to_2d(true)
        pred, pred_f, pred_b = to_2d(pred)
    else:
        true_f, true_b = to_1d(true)
        pred_f, pred_b = to_1d(pred)
    
    pred_bin = (pred > threshold).astype(int)
    
    # Compute SHD
    shd = shd_metric(pred_bin, true)
    
    # Compute edge direction accuracy
    true_mask = (true_f + true_b) > 0
    true_direction = (true_f[true_mask] > true_b[true_mask])
    pred_forward = (pred_f[true_mask] > pred_b[true_mask])
    pred_backward = (pred_f[true_mask] < pred_b[true_mask])
    
    edge_acc = (true_direction == pred_forward)[true_direction].sum() + \
               (~true_direction == pred_backward)[~true_direction].sum()
    edge_acc = edge_acc / len(true_direction) if len(true_direction) > 0 else 0.0
        
    return shd, edge_acc

def parse_dataset_key(key):
    """Parse dataset key to extract metadata."""
    if "sachs" in key:
        return {"nodes": 11, "edges": 17, "mechanism": "sachs", "type": "real"}
    else:
        # Parse synthetic dataset keys like "p100e100linear"
        parts = key.split("_")[0]  # Remove any suffixes
        if parts.startswith("p") and "e" in parts:
            nodes = int(parts.split("e")[0][1:])  # Extract number after 'p'
            edges = int(parts.split("e")[1])       # Extract number after 'e'
            mechanism = "unknown"
            return {"nodes": nodes, "edges": edges, "mechanism": mechanism, "type": "synthetic"}
    return {"nodes": "unknown", "edges": "unknown", "mechanism": "unknown", "type": "unknown"}

def analyze_results(results):
    """Comprehensive analysis of results."""
    print("\n" + "="*60)
    print("RFCI RESULTS ANALYSIS")
    print("="*60)
    
    # Collect metrics for each dataset
    dataset_metrics = {}
    
    for key, data in results.items():
        print(f"\nDataset: {key}")
        print("-" * 40)
        
        # Parse dataset info
        info = parse_dataset_key(key)
        print(f"Nodes: {info['nodes']}, Edges: {info['edges']}")
        print(f"Mechanism: {info['mechanism']}, Type: {info['type']}")
        
        # Extract metrics
        auc_scores = data.get("auc", [])
        prc_scores = data.get("prc", [])
        times = data.get("time", [])
        true_graphs = data.get("true", [])
        pred_graphs = data.get("pred", [])
        
        if not auc_scores:
            print("No metrics found for this dataset")
            continue
        
        # Basic statistics
        print(f"AUC: {np.mean(auc_scores):.3f} ± {np.std(auc_scores):.3f}")
        print(f"AUPRC: {np.mean(prc_scores):.3f} ± {np.std(prc_scores):.3f}")
        print(f"Time: {np.mean(times):.2f} ± {np.std(times):.2f} seconds")
        
        # Additional metrics if we have predictions
        if true_graphs and pred_graphs:
            shd_scores = []
            edge_acc_scores = []
            
            for true, pred in zip(true_graphs, pred_graphs):
                try:
                    shd, edge_acc = compute_additional_metrics(true, pred)
                    shd_scores.append(shd)
                    edge_acc_scores.append(edge_acc)
                except Exception as e:
                    print(f"Warning: Could not compute additional metrics: {e}")
            
            if shd_scores:
                print(f"SHD: {np.mean(shd_scores):.1f} ± {np.std(shd_scores):.1f}")
                print(f"Edge Accuracy: {np.mean(edge_acc_scores):.3f} ± {np.std(edge_acc_scores):.3f}")
        
        # Store for summary
        dataset_metrics[key] = {
            "info": info,
            "auc_mean": np.mean(auc_scores),
            "auc_std": np.std(auc_scores),
            "prc_mean": np.mean(prc_scores),
            "prc_std": np.std(prc_scores),
            "time_mean": np.mean(times),
            "time_std": np.std(times),
            "n_samples": len(auc_scores)
        }
    
    return dataset_metrics

def create_summary_table(dataset_metrics):
    """Create a summary table of all results."""
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    
    # Create DataFrame
    rows = []
    for key, metrics in dataset_metrics.items():
        info = metrics["info"]
        row = {
            "Dataset": key,
            "Nodes": info["nodes"],
            "Edges": info["edges"],
            "Type": info["type"],
            "AUC": f"{metrics['auc_mean']:.3f} ± {metrics['auc_std']:.3f}",
            "AUPRC": f"{metrics['prc_mean']:.3f} ± {metrics['prc_std']:.3f}",
            "Time (s)": f"{metrics['time_mean']:.1f} ± {metrics['time_std']:.1f}",
            "Samples": metrics["n_samples"]
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    
    return df

def create_visualizations(results, dataset_metrics):
    """Create visualizations of the results."""
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Performance by dataset size
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Extract data for plotting
    nodes = []
    edges = []
    aucs = []
    prcs = []
    times = []
    
    for key, metrics in dataset_metrics.items():
        info = metrics["info"]
        if info["nodes"] != "unknown" and info["edges"] != "unknown":
            nodes.append(info["nodes"])
            edges.append(info["edges"])
            aucs.append(metrics["auc_mean"])
            prcs.append(metrics["prc_mean"])
            times.append(metrics["time_mean"])
    
    if nodes:
        # AUC vs Nodes
        axes[0,0].scatter(nodes, aucs, alpha=0.7)
        axes[0,0].set_xlabel("Number of Nodes")
        axes[0,0].set_ylabel("AUC")
        axes[0,0].set_title("AUC vs Graph Size")
        
        # AUPRC vs Nodes
        axes[0,1].scatter(nodes, prcs, alpha=0.7, color='orange')
        axes[0,1].set_xlabel("Number of Nodes")
        axes[0,1].set_ylabel("AUPRC")
        axes[0,1].set_title("AUPRC vs Graph Size")
        
        # Time vs Nodes
        axes[1,0].scatter(nodes, times, alpha=0.7, color='green')
        axes[1,0].set_xlabel("Number of Nodes")
        axes[1,0].set_ylabel("Time (seconds)")
        axes[1,0].set_title("Runtime vs Graph Size")
        axes[1,0].set_yscale('log')
        
        # AUC vs AUPRC
        axes[1,1].scatter(aucs, prcs, alpha=0.7, color='red')
        axes[1,1].set_xlabel("AUC")
        axes[1,1].set_ylabel("AUPRC")
        axes[1,1].set_title("AUC vs AUPRC")
        
        plt.tight_layout()
        plt.savefig("rfci_results_analysis.png", dpi=300, bbox_inches='tight')
        print("Saved visualization: rfci_results_analysis.png")
        plt.show()
    
    # 2. Distribution of metrics across all datasets
    all_aucs = []
    all_prcs = []
    all_times = []
    
    for key, data in results.items():
        all_aucs.extend(data.get("auc", []))
        all_prcs.extend(data.get("prc", []))
        all_times.extend(data.get("time", []))
    
    if all_aucs:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].hist(all_aucs, bins=20, alpha=0.7, color='blue')
        axes[0].set_xlabel("AUC")
        axes[0].set_ylabel("Frequency")
        axes[0].set_title("Distribution of AUC Scores")
        
        axes[1].hist(all_prcs, bins=20, alpha=0.7, color='orange')
        axes[1].set_xlabel("AUPRC")
        axes[1].set_ylabel("Frequency")
        axes[1].set_title("Distribution of AUPRC Scores")
        
        axes[2].hist(all_times, bins=20, alpha=0.7, color='green')
        axes[2].set_xlabel("Time (seconds)")
        axes[2].set_ylabel("Frequency")
        axes[2].set_title("Distribution of Runtime")
        axes[2].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig("rfci_metrics_distribution.png", dpi=300, bbox_inches='tight')
        print("Saved visualization: rfci_metrics_distribution.png")
        plt.show()

def compare_with_baselines(dataset_metrics, baseline_file=None):
    """Compare RFCI results with baseline methods if available."""
    if not baseline_file or not os.path.exists(baseline_file):
        print("\nNo baseline file provided or found. Skipping comparison.")
        return
    
    print("\n" + "="*60)
    print("COMPARISON WITH BASELINES")
    print("="*60)
    
    try:
        baseline_results = read_pickle(baseline_file)
        print(f"Loaded baseline results from: {baseline_file}")
        
        # Simple comparison for common datasets
        common_datasets = set(dataset_metrics.keys()) & set(baseline_results.keys())
        
        if not common_datasets:
            print("No common datasets found between RFCI and baseline results.")
            return
        
        print(f"Found {len(common_datasets)} common datasets for comparison")
        
        comparison_data = []
        for dataset in common_datasets:
            rfci_auc = dataset_metrics[dataset]["auc_mean"]
            rfci_prc = dataset_metrics[dataset]["prc_mean"]
            
            baseline_data = baseline_results[dataset]
            baseline_auc = np.mean(baseline_data.get("auc", [0]))
            baseline_prc = np.mean(baseline_data.get("prc", [0]))
            
            comparison_data.append({
                "Dataset": dataset,
                "RFCI_AUC": rfci_auc,
                "Baseline_AUC": baseline_auc,
                "RFCI_AUPRC": rfci_prc,
                "Baseline_AUPRC": baseline_prc,
                "AUC_Improvement": rfci_auc - baseline_auc,
                "AUPRC_Improvement": rfci_prc - baseline_prc
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print("\nComparison Results:")
        print(comparison_df.to_string(index=False))
        
        # Summary statistics
        avg_auc_improvement = comparison_df["AUC_Improvement"].mean()
        avg_prc_improvement = comparison_df["AUPRC_Improvement"].mean()
        
        print(f"\nAverage AUC improvement: {avg_auc_improvement:.3f}")
        print(f"Average AUPRC improvement: {avg_prc_improvement:.3f}")
        
    except Exception as e:
        print(f"Error loading baseline results: {e}")

def main():
    """Main analysis function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze RFCI results from SEA pipeline")
    parser.add_argument("--results_file", required=True, help="Path to results pickle file")
    parser.add_argument("--baseline_file", help="Path to baseline results for comparison")
    parser.add_argument("--output_dir", default=".", help="Directory to save outputs")
    
    args = parser.parse_args()
    
    # Change to output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.chdir(args.output_dir)
    
    try:
        # Load results
        results = load_results(args.results_file)
        
        # Analyze results
        dataset_metrics = analyze_results(results)
        
        # Create summary table
        summary_df = create_summary_table(dataset_metrics)
        
        # Create visualizations
        create_visualizations(results, dataset_metrics)
        
        # Compare with baselines if provided
        compare_with_baselines(dataset_metrics, args.baseline_file)
        
        # Save summary to CSV
        summary_df.to_csv("rfci_results_summary.csv", index=False)
        print(f"\nSaved summary to: rfci_results_summary.csv")
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
