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


def get_all_unique_datasets(results_dict):
    """Find all unique datasets across result files."""
    if not results_dict:
        return set()
    
    all_datasets = set()
    for results in results_dict.values():
        all_datasets.update(results.keys())
    
    return all_datasets


def create_long_format_table(results_dict, method_names, all_datasets):
    """Create a long-format table for dataset-wise deep dive."""
    data_rows = []
    
    for dataset in sorted(all_datasets):
        for filepath, results in results_dict.items():
            method_name = method_names[filepath]
            
            # Use safe get
            if dataset not in results:
                continue
                
            data = results[dataset]
            metrics = compute_metrics(data)
            
            row = {
                "Dataset": dataset,
                "Method": method_name,
                "AUC_Mean": metrics.get("auc_mean"),
                "AUC_Std": metrics.get("auc_std"),
                "AUPRC_Mean": metrics.get("prc_mean"),
                "AUPRC_Std": metrics.get("prc_std"),
                # F1 - both threshold and oracle_k
                "F1_Threshold_Mean": metrics.get("f1_threshold_mean"),
                "F1_Threshold_Std": metrics.get("f1_threshold_std"),
                "F1_OracleK_Mean": metrics.get("f1_oracle_k_mean"),
                "F1_OracleK_Std": metrics.get("f1_oracle_k_std"),
                # Time
                "Time_Mean": metrics.get("time_mean"),
                "Time_Std": metrics.get("time_std"),
                # SHD - both threshold and oracle_k
                "SHD_Threshold_Mean": metrics.get("shd_threshold_mean"),
                "SHD_Threshold_Std": metrics.get("shd_threshold_std"),
                "SHD_OracleK_Mean": metrics.get("shd_oracle_k_mean"),
                "SHD_OracleK_Std": metrics.get("shd_oracle_k_std"),
                # Normalized SHD
                "Norm_SHD_Threshold_Mean": metrics.get("normalized_shd_threshold_mean"),
                "Norm_SHD_Threshold_Std": metrics.get("normalized_shd_threshold_std"),
                "Norm_SHD_OracleK_Mean": metrics.get("normalized_shd_oracle_k_mean"),
                "Norm_SHD_OracleK_Std": metrics.get("normalized_shd_oracle_k_std"),
                # NNZ
                "NNZ_Threshold_Mean": metrics.get("nnz_threshold_mean"),
                "NNZ_Threshold_Std": metrics.get("nnz_threshold_std"),
                "NNZ_OracleK_Mean": metrics.get("nnz_oracle_k_mean"),
                "NNZ_OracleK_Std": metrics.get("nnz_oracle_k_std"),
                # Precision/Recall
                "Precision_Threshold_Mean": metrics.get("precision_threshold_mean"),
                "Recall_Threshold_Mean": metrics.get("recall_threshold_mean"),
                "Precision_OracleK_Mean": metrics.get("precision_oracle_k_mean"),
                "Recall_OracleK_Mean": metrics.get("recall_oracle_k_mean"),
                # Other
                "SID_Mean": metrics.get("sid_mean"),
                "SID_Std": metrics.get("sid_std"),
                "Norm_SID_Mean": metrics.get("normalized_sid_mean"),
                "Norm_SID_Std": metrics.get("normalized_sid_std"),
                "M_True_Mean": metrics.get("m_true_mean"),
                "N_Samples": metrics.get("n_samples")
            }
            data_rows.append(row)
            
    return pd.DataFrame(data_rows)


def compute_metrics(data):
    """Compute mean and std for metrics in a dataset.
    
    Supports both new metric names (f1_threshold, f1_oracle_k, shd_threshold, shd_oracle_k)
    and legacy names (f1, shd) for backward compatibility.
    """
    def safe_filter(values):
        """Filter out None and NaN values."""
        return [s for s in values if s is not None and not (isinstance(s, float) and np.isnan(s))]
    
    auc_scores = safe_filter(data.get("auc", []))
    prc_scores = safe_filter(data.get("prc", []))
    times = safe_filter(data.get("time", []))
    
    # F1 metrics - support both new and legacy names
    # New: f1_threshold, f1_oracle_k | Legacy: f1
    f1_threshold = safe_filter(data.get("f1_threshold", []))
    f1_oracle_k = safe_filter(data.get("f1_oracle_k", []))
    f1_legacy = safe_filter(data.get("f1", []))
    
    # SHD metrics - support both new and legacy names
    shd_threshold = safe_filter(data.get("shd_threshold", []))
    shd_oracle_k = safe_filter(data.get("shd_oracle_k", []))
    shd_legacy = safe_filter(data.get("shd", []))
    
    normalized_shd_threshold = safe_filter(data.get("normalized_shd_threshold", []))
    normalized_shd_oracle_k = safe_filter(data.get("normalized_shd_oracle_k", []))
    normalized_shd_legacy = safe_filter(data.get("normalized_shd", []))
    
    # NNZ metrics
    nnz_threshold = safe_filter(data.get("nnz_threshold", []))
    nnz_oracle_k = safe_filter(data.get("nnz_oracle_k", []))
    nnz_legacy = safe_filter(data.get("nnz", []))
    
    # Precision/Recall
    precision_threshold = safe_filter(data.get("precision_threshold", []))
    precision_oracle_k = safe_filter(data.get("precision_oracle_k", []))
    recall_threshold = safe_filter(data.get("recall_threshold", []))
    recall_oracle_k = safe_filter(data.get("recall_oracle_k", []))
    
    # Other metrics
    sid_scores = safe_filter(data.get("sid", []))
    normalized_sid_scores = safe_filter(data.get("normalized_sid", []))
    m_true = safe_filter(data.get("m_true", []))
    
    metrics = {
        "auc_mean": np.mean(auc_scores) if auc_scores else None,
        "auc_std": np.std(auc_scores) if auc_scores else None,
        "prc_mean": np.mean(prc_scores) if prc_scores else None,
        "prc_std": np.std(prc_scores) if prc_scores else None,
        "time_mean": np.mean(times) if times else None,
        "time_std": np.std(times) if times else None,
        "n_samples": len(auc_scores) if auc_scores else 0,
        
        # F1 THRESHOLD (use new if available, fallback to legacy)
        "f1_threshold_mean": np.mean(f1_threshold) if f1_threshold else (np.mean(f1_legacy) if f1_legacy else None),
        "f1_threshold_std": np.std(f1_threshold) if f1_threshold else (np.std(f1_legacy) if f1_legacy else None),
        
        # F1 ORACLE-K (only available in new format)
        "f1_oracle_k_mean": np.mean(f1_oracle_k) if f1_oracle_k else None,
        "f1_oracle_k_std": np.std(f1_oracle_k) if f1_oracle_k else None,
        
        # SHD THRESHOLD (use new if available, fallback to legacy)
        "shd_threshold_mean": np.mean(shd_threshold) if shd_threshold else (np.mean(shd_legacy) if shd_legacy else None),
        "shd_threshold_std": np.std(shd_threshold) if shd_threshold else (np.std(shd_legacy) if shd_legacy else None),
        
        # SHD ORACLE-K (only available in new format)
        "shd_oracle_k_mean": np.mean(shd_oracle_k) if shd_oracle_k else None,
        "shd_oracle_k_std": np.std(shd_oracle_k) if shd_oracle_k else None,
        
        # Normalized SHD
        "normalized_shd_threshold_mean": np.mean(normalized_shd_threshold) if normalized_shd_threshold else (np.mean(normalized_shd_legacy) if normalized_shd_legacy else None),
        "normalized_shd_threshold_std": np.std(normalized_shd_threshold) if normalized_shd_threshold else (np.std(normalized_shd_legacy) if normalized_shd_legacy else None),
        "normalized_shd_oracle_k_mean": np.mean(normalized_shd_oracle_k) if normalized_shd_oracle_k else None,
        "normalized_shd_oracle_k_std": np.std(normalized_shd_oracle_k) if normalized_shd_oracle_k else None,
        
        # NNZ
        "nnz_threshold_mean": np.mean(nnz_threshold) if nnz_threshold else (np.mean(nnz_legacy) if nnz_legacy else None),
        "nnz_threshold_std": np.std(nnz_threshold) if nnz_threshold else (np.std(nnz_legacy) if nnz_legacy else None),
        "nnz_oracle_k_mean": np.mean(nnz_oracle_k) if nnz_oracle_k else None,
        "nnz_oracle_k_std": np.std(nnz_oracle_k) if nnz_oracle_k else None,
        
        # Precision/Recall
        "precision_threshold_mean": np.mean(precision_threshold) if precision_threshold else None,
        "precision_threshold_std": np.std(precision_threshold) if precision_threshold else None,
        "precision_oracle_k_mean": np.mean(precision_oracle_k) if precision_oracle_k else None,
        "precision_oracle_k_std": np.std(precision_oracle_k) if precision_oracle_k else None,
        "recall_threshold_mean": np.mean(recall_threshold) if recall_threshold else None,
        "recall_threshold_std": np.std(recall_threshold) if recall_threshold else None,
        "recall_oracle_k_mean": np.mean(recall_oracle_k) if recall_oracle_k else None,
        "recall_oracle_k_std": np.std(recall_oracle_k) if recall_oracle_k else None,
        
        # Other metrics
        "sid_mean": np.mean(sid_scores) if sid_scores else None,
        "sid_std": np.std(sid_scores) if sid_scores else None,
        "normalized_sid_mean": np.mean(normalized_sid_scores) if normalized_sid_scores else None,
        "normalized_sid_std": np.std(normalized_sid_scores) if normalized_sid_scores else None,
        "m_true_mean": np.mean(m_true) if m_true else None,
    }
    
    return metrics


def create_comparison_table(results_dict, method_names, common_datasets):
    """Create a comprehensive comparison table."""
    comparison_data = []
    
    for dataset in sorted(common_datasets):
        row = {"Dataset": dataset}
        
        for filepath, results in results_dict.items():
            method_name = method_names[filepath]
            
            # Handle missing datasets
            if dataset not in results:
                continue
                
            data = results[dataset]
            metrics = compute_metrics(data)
            
            # Add metrics with method name prefix
            row[f"{method_name}_AUC"] = metrics["auc_mean"]
            row[f"{method_name}_AUC_std"] = metrics["auc_std"]
            row[f"{method_name}_AUPRC"] = metrics["prc_mean"]
            row[f"{method_name}_AUPRC_std"] = metrics["prc_std"]
            
            # F1 - threshold and oracle_k
            row[f"{method_name}_F1_Thresh"] = metrics["f1_threshold_mean"]
            row[f"{method_name}_F1_Thresh_std"] = metrics["f1_threshold_std"]
            row[f"{method_name}_F1_OracleK"] = metrics["f1_oracle_k_mean"]
            row[f"{method_name}_F1_OracleK_std"] = metrics["f1_oracle_k_std"]
            
            row[f"{method_name}_Time"] = metrics["time_mean"]
            row[f"{method_name}_Time_std"] = metrics["time_std"]
            row[f"{method_name}_N"] = metrics["n_samples"]
            
            # SHD - threshold and oracle_k
            if metrics["shd_threshold_mean"] is not None:
                row[f"{method_name}_SHD_Thresh"] = metrics["shd_threshold_mean"]
                row[f"{method_name}_SHD_Thresh_std"] = metrics["shd_threshold_std"]
            if metrics["shd_oracle_k_mean"] is not None:
                row[f"{method_name}_SHD_OracleK"] = metrics["shd_oracle_k_mean"]
                row[f"{method_name}_SHD_OracleK_std"] = metrics["shd_oracle_k_std"]
            
            # Normalized SHD
            if metrics["normalized_shd_threshold_mean"] is not None:
                row[f"{method_name}_Norm_SHD_Thresh"] = metrics["normalized_shd_threshold_mean"]
            if metrics["normalized_shd_oracle_k_mean"] is not None:
                row[f"{method_name}_Norm_SHD_OracleK"] = metrics["normalized_shd_oracle_k_mean"]
            
            # NNZ
            if metrics["nnz_threshold_mean"] is not None:
                row[f"{method_name}_NNZ_Thresh"] = metrics["nnz_threshold_mean"]
            if metrics["nnz_oracle_k_mean"] is not None:
                row[f"{method_name}_NNZ_OracleK"] = metrics["nnz_oracle_k_mean"]
            
            # Precision/Recall
            if metrics["precision_threshold_mean"] is not None:
                row[f"{method_name}_Prec_Thresh"] = metrics["precision_threshold_mean"]
                row[f"{method_name}_Recall_Thresh"] = metrics["recall_threshold_mean"]
            if metrics["precision_oracle_k_mean"] is not None:
                row[f"{method_name}_Prec_OracleK"] = metrics["precision_oracle_k_mean"]
                row[f"{method_name}_Recall_OracleK"] = metrics["recall_oracle_k_mean"]
            
            # M_True (ground truth edge count)
            if metrics["m_true_mean"] is not None:
                row[f"{method_name}_M_True"] = metrics["m_true_mean"]
            
            # SID
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
    
    def safe_filter(values):
        """Filter out None and NaN values."""
        return [s for s in values if s is not None and not (isinstance(s, float) and np.isnan(s))]
    
    for filepath, results in results_dict.items():
        method_name = method_names[filepath]
        
        all_auc = []
        all_prc = []
        all_f1_threshold = []
        all_f1_oracle_k = []
        all_times = []
        all_shd_threshold = []
        all_shd_oracle_k = []
        all_normalized_shd_threshold = []
        all_normalized_shd_oracle_k = []
        all_nnz_threshold = []
        all_nnz_oracle_k = []
        all_sid = []
        all_normalized_sid = []
        total_samples = 0
        
        for dataset in common_datasets:
            # Handle missing datasets
            if dataset not in results:
                continue
                
            data = results[dataset]
            all_auc.extend(safe_filter(data.get("auc", [])))
            all_prc.extend(safe_filter(data.get("prc", [])))
            all_times.extend(safe_filter(data.get("time", [])))
            
            # F1: new names first, then legacy fallback
            f1_thresh = safe_filter(data.get("f1_threshold", []))
            f1_oracle = safe_filter(data.get("f1_oracle_k", []))
            f1_legacy = safe_filter(data.get("f1", []))
            all_f1_threshold.extend(f1_thresh if f1_thresh else f1_legacy)
            all_f1_oracle_k.extend(f1_oracle)
            
            # SHD: new names first, then legacy fallback
            shd_thresh = safe_filter(data.get("shd_threshold", []))
            shd_oracle = safe_filter(data.get("shd_oracle_k", []))
            shd_legacy = safe_filter(data.get("shd", []))
            all_shd_threshold.extend(shd_thresh if shd_thresh else shd_legacy)
            all_shd_oracle_k.extend(shd_oracle)
            
            # Normalized SHD
            norm_shd_thresh = safe_filter(data.get("normalized_shd_threshold", []))
            norm_shd_oracle = safe_filter(data.get("normalized_shd_oracle_k", []))
            norm_shd_legacy = safe_filter(data.get("normalized_shd", []))
            all_normalized_shd_threshold.extend(norm_shd_thresh if norm_shd_thresh else norm_shd_legacy)
            all_normalized_shd_oracle_k.extend(norm_shd_oracle)
            
            # NNZ
            nnz_thresh = safe_filter(data.get("nnz_threshold", []))
            nnz_oracle = safe_filter(data.get("nnz_oracle_k", []))
            nnz_legacy = safe_filter(data.get("nnz", []))
            all_nnz_threshold.extend(nnz_thresh if nnz_thresh else nnz_legacy)
            all_nnz_oracle_k.extend(nnz_oracle)
            
            # Other metrics
            all_sid.extend(safe_filter(data.get("sid", [])))
            all_normalized_sid.extend(safe_filter(data.get("normalized_sid", [])))
            total_samples += len(safe_filter(data.get("auc", [])))
        
        summary_data.append({
            "Method": method_name,
            "Avg_AUC": np.mean(all_auc) if all_auc else None,
            "AUC_std": np.std(all_auc) if all_auc else None,
            "Avg_AUPRC": np.mean(all_prc) if all_prc else None,
            "AUPRC_std": np.std(all_prc) if all_prc else None,
            # F1 - threshold and oracle_k
            "Avg_F1_Thresh": np.mean(all_f1_threshold) if all_f1_threshold else None,
            "F1_Thresh_std": np.std(all_f1_threshold) if all_f1_threshold else None,
            "Avg_F1_OracleK": np.mean(all_f1_oracle_k) if all_f1_oracle_k else None,
            "F1_OracleK_std": np.std(all_f1_oracle_k) if all_f1_oracle_k else None,
            # Time
            "Avg_Time": np.mean(all_times) if all_times else None,
            "Time_std": np.std(all_times) if all_times else None,
            # SHD - threshold and oracle_k
            "Avg_SHD_Thresh": np.mean(all_shd_threshold) if all_shd_threshold else None,
            "SHD_Thresh_std": np.std(all_shd_threshold) if all_shd_threshold else None,
            "Avg_SHD_OracleK": np.mean(all_shd_oracle_k) if all_shd_oracle_k else None,
            "SHD_OracleK_std": np.std(all_shd_oracle_k) if all_shd_oracle_k else None,
            # Normalized SHD
            "Avg_Norm_SHD_Thresh": np.mean(all_normalized_shd_threshold) if all_normalized_shd_threshold else None,
            "Norm_SHD_Thresh_std": np.std(all_normalized_shd_threshold) if all_normalized_shd_threshold else None,
            "Avg_Norm_SHD_OracleK": np.mean(all_normalized_shd_oracle_k) if all_normalized_shd_oracle_k else None,
            "Norm_SHD_OracleK_std": np.std(all_normalized_shd_oracle_k) if all_normalized_shd_oracle_k else None,
            # NNZ
            "Avg_NNZ_Thresh": np.mean(all_nnz_threshold) if all_nnz_threshold else None,
            "NNZ_Thresh_std": np.std(all_nnz_threshold) if all_nnz_threshold else None,
            "Avg_NNZ_OracleK": np.mean(all_nnz_oracle_k) if all_nnz_oracle_k else None,
            "NNZ_OracleK_std": np.std(all_nnz_oracle_k) if all_nnz_oracle_k else None,
            # Other
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
    
    # Helper function to format values with None/NaN handling
    def fmt(v, decimals=3):
        if pd.isna(v) or v is None:
            return "N/A"
        return f"{v:.{decimals}f}"
    
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
                auc_df[method] = df[col_mean].apply(lambda x: fmt(x)) + " ± " + df[col_std].apply(lambda x: fmt(x))
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
                prc_df[method] = df[col_mean].apply(lambda x: fmt(x)) + " ± " + df[col_std].apply(lambda x: fmt(x))
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
                f1_df[method] = df[col_mean].apply(lambda x: fmt(x)) + " ± " + df[col_std].apply(lambda x: fmt(x))
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
                time_df[method] = df[col_mean].apply(lambda x: fmt(x, 2)) + " ± " + df[col_std].apply(lambda x: fmt(x, 2))
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
            if mean_col not in row or pd.isna(row[mean_col]) or row[mean_col] is None:
                return "N/A"
            mean_val = row[mean_col]
            std_val = row.get(std_col, 0.0)
            if pd.isna(std_val) or std_val is None:
                std_val = 0.0
            return f"{mean_val:.{decimals}f} ± {std_val:.{decimals}f}"
        return formatter
    
    formatted_df["AUC"] = formatted_df.apply(format_with_std("Avg_AUC", "AUC_std", 3), axis=1)
    formatted_df["AUPRC"] = formatted_df.apply(format_with_std("Avg_AUPRC", "AUPRC_std", 3), axis=1)
    formatted_df["F1_Thresh"] = formatted_df.apply(format_with_std("Avg_F1_Thresh", "F1_Thresh_std", 3), axis=1)
    formatted_df["F1_OracleK"] = formatted_df.apply(format_with_std("Avg_F1_OracleK", "F1_OracleK_std", 3), axis=1)
    formatted_df["Time"] = formatted_df.apply(format_with_std("Avg_Time", "Time_std", 2), axis=1)
    
    # Format SHD metrics
    formatted_df["SHD_Thresh"] = formatted_df.apply(format_with_std("Avg_SHD_Thresh", "SHD_Thresh_std", 1), axis=1)
    formatted_df["SHD_OracleK"] = formatted_df.apply(format_with_std("Avg_SHD_OracleK", "SHD_OracleK_std", 1), axis=1)
    formatted_df["Norm_SHD_Thresh"] = formatted_df.apply(format_with_std("Avg_Norm_SHD_Thresh", "Norm_SHD_Thresh_std", 3), axis=1)
    formatted_df["NNZ_Thresh"] = formatted_df.apply(format_with_std("Avg_NNZ_Thresh", "NNZ_Thresh_std", 1), axis=1)
    formatted_df["SID"] = formatted_df.apply(format_with_std("Avg_SID", "SID_std", 1), axis=1)
    formatted_df["Norm_SID"] = formatted_df.apply(format_with_std("Avg_Norm_SID", "Norm_SID_std", 3), axis=1)
    
    # Primary metrics display
    display_cols = ["Method", "AUC", "AUPRC", "F1_Thresh", "F1_OracleK", "Time", "SHD_Thresh", "SHD_OracleK"]
    optional_cols = ["Norm_SHD_Thresh", "NNZ_Thresh"]
    for col in optional_cols:
        if col in formatted_df.columns and formatted_df[col].apply(lambda x: x != "N/A").any():
            display_cols.append(col)
    display_cols.extend(["Total_Samples", "Num_Datasets"])
    
    # Only include columns that exist
    display_cols = [c for c in display_cols if c in formatted_df.columns]
    print(formatted_df[display_cols].to_string(index=False))
    
    # Find best performers
    print("\n" + "-"*80)
    print("BEST PERFORMERS:")
    print("-"*80)
    
    # Handle NaN values - only find best if there are valid values
    if "Avg_AUC" in summary_df.columns and summary_df["Avg_AUC"].notna().any():
        best_auc_idx = summary_df["Avg_AUC"].idxmax()
        print(f"Best AUC:  {summary_df.loc[best_auc_idx, 'Method']} ({summary_df.loc[best_auc_idx, 'Avg_AUC']:.3f})")
    else:
        print("Best AUC:  N/A (no valid data)")
    
    if "Avg_AUPRC" in summary_df.columns and summary_df["Avg_AUPRC"].notna().any():
        best_prc_idx = summary_df["Avg_AUPRC"].idxmax()
        print(f"Best AUPRC: {summary_df.loc[best_prc_idx, 'Method']} ({summary_df.loc[best_prc_idx, 'Avg_AUPRC']:.3f})")
    else:
        print("Best AUPRC: N/A (no valid data)")
    
    # F1 best performers
    if "Avg_F1_Thresh" in summary_df.columns and summary_df["Avg_F1_Thresh"].notna().any():
        best_f1_idx = summary_df["Avg_F1_Thresh"].idxmax()
        print(f"Best F1 (Threshold): {summary_df.loc[best_f1_idx, 'Method']} ({summary_df.loc[best_f1_idx, 'Avg_F1_Thresh']:.3f})")
    if "Avg_F1_OracleK" in summary_df.columns and summary_df["Avg_F1_OracleK"].notna().any():
        best_f1_oracle_idx = summary_df["Avg_F1_OracleK"].idxmax()
        print(f"Best F1 (Oracle-K): {summary_df.loc[best_f1_oracle_idx, 'Method']} ({summary_df.loc[best_f1_oracle_idx, 'Avg_F1_OracleK']:.3f})")
    
    if "Avg_Time" in summary_df.columns and summary_df["Avg_Time"].notna().any():
        fastest_idx = summary_df["Avg_Time"].idxmin()
        print(f"Fastest:   {summary_df.loc[fastest_idx, 'Method']} ({summary_df.loc[fastest_idx, 'Avg_Time']:.2f}s)")
    else:
        print("Fastest:   N/A (no valid data)")
    
    # Best performers for SHD (lower is better)
    if "Avg_SHD_Thresh" in summary_df.columns and summary_df["Avg_SHD_Thresh"].notna().any():
        best_shd_idx = summary_df["Avg_SHD_Thresh"].idxmin()
        print(f"Best SHD (Threshold, lowest): {summary_df.loc[best_shd_idx, 'Method']} ({summary_df.loc[best_shd_idx, 'Avg_SHD_Thresh']:.1f})")
    if "Avg_SHD_OracleK" in summary_df.columns and summary_df["Avg_SHD_OracleK"].notna().any():
        best_shd_oracle_idx = summary_df["Avg_SHD_OracleK"].idxmin()
        print(f"Best SHD (Oracle-K, lowest): {summary_df.loc[best_shd_oracle_idx, 'Method']} ({summary_df.loc[best_shd_oracle_idx, 'Avg_SHD_OracleK']:.1f})")


def print_rankings(summary_df):
    """Print rankings for each metric."""
    print("\n" + "="*80)
    print("RANKINGS")
    print("="*80)
    
    # Rank by AUC (higher is better)
    if "Avg_AUC" in summary_df.columns:
        summary_df["AUC_Rank"] = summary_df["Avg_AUC"].rank(ascending=False, method="min")
    # Rank by AUPRC (higher is better)
    if "Avg_AUPRC" in summary_df.columns:
        summary_df["AUPRC_Rank"] = summary_df["Avg_AUPRC"].rank(ascending=False, method="min")
    # Rank by F1 Threshold (higher is better)
    if "Avg_F1_Thresh" in summary_df.columns:
        summary_df["F1_Thresh_Rank"] = summary_df["Avg_F1_Thresh"].rank(ascending=False, method="min")
    # Rank by F1 Oracle-K (higher is better)
    if "Avg_F1_OracleK" in summary_df.columns:
        summary_df["F1_OracleK_Rank"] = summary_df["Avg_F1_OracleK"].rank(ascending=False, method="min")
    # Rank by Time (lower is better)
    if "Avg_Time" in summary_df.columns:
        summary_df["Time_Rank"] = summary_df["Avg_Time"].rank(ascending=True, method="min")
    
    # Rank by SHD (lower is better)
    if "Avg_SHD_Thresh" in summary_df.columns and summary_df["Avg_SHD_Thresh"].notna().any():
        summary_df["SHD_Thresh_Rank"] = summary_df["Avg_SHD_Thresh"].rank(ascending=True, method="min")
    if "Avg_SHD_OracleK" in summary_df.columns and summary_df["Avg_SHD_OracleK"].notna().any():
        summary_df["SHD_OracleK_Rank"] = summary_df["Avg_SHD_OracleK"].rank(ascending=True, method="min")
    if "Avg_SID" in summary_df.columns and summary_df["Avg_SID"].notna().any():
        summary_df["SID_Rank"] = summary_df["Avg_SID"].rank(ascending=True, method="min")
    
    ranking_cols = ["Method"]
    if "AUC_Rank" in summary_df.columns:
        ranking_cols.append("AUC_Rank")
    if "AUPRC_Rank" in summary_df.columns:
        ranking_cols.append("AUPRC_Rank")
    if "F1_Thresh_Rank" in summary_df.columns:
        ranking_cols.append("F1_Thresh_Rank")
    if "F1_OracleK_Rank" in summary_df.columns:
        ranking_cols.append("F1_OracleK_Rank")
    if "Time_Rank" in summary_df.columns:
        ranking_cols.append("Time_Rank")
    if "SHD_Thresh_Rank" in summary_df.columns:
        ranking_cols.append("SHD_Thresh_Rank")
    if "SHD_OracleK_Rank" in summary_df.columns:
        ranking_cols.append("SHD_OracleK_Rank")
    if "SID_Rank" in summary_df.columns:
        ranking_cols.append("SID_Rank")
    
    ranking_df = summary_df[ranking_cols].copy()
    if "AUC_Rank" in ranking_df.columns:
        ranking_df = ranking_df.sort_values("AUC_Rank")
    
    print("\nRanking by AUC (1 = best):")
    if "AUC_Rank" in ranking_df.columns:
        print(ranking_df[["Method", "AUC_Rank"]].to_string(index=False))
    
    print("\nRanking by AUPRC (1 = best):")
    if "AUPRC_Rank" in ranking_df.columns:
        print(ranking_df[["Method", "AUPRC_Rank"]].to_string(index=False))
    
    print("\nRanking by F1 Threshold (1 = best):")
    if "F1_Thresh_Rank" in ranking_df.columns:
        print(ranking_df[["Method", "F1_Thresh_Rank"]].to_string(index=False))
    
    print("\nRanking by F1 Oracle-K (1 = best):")
    if "F1_OracleK_Rank" in ranking_df.columns:
        print(ranking_df[["Method", "F1_OracleK_Rank"]].to_string(index=False))
    
    print("\nRanking by Time (1 = fastest):")
    if "Time_Rank" in ranking_df.columns:
        print(ranking_df[["Method", "Time_Rank"]].to_string(index=False))
    
    if "SHD_Thresh_Rank" in ranking_df.columns:
        print("\nRanking by SHD Threshold (1 = best, lowest):")
        print(ranking_df[["Method", "SHD_Thresh_Rank"]].to_string(index=False))
    
    if "SHD_OracleK_Rank" in ranking_df.columns:
        print("\nRanking by SHD Oracle-K (1 = best, lowest):")
        print(ranking_df[["Method", "SHD_OracleK_Rank"]].to_string(index=False))
    
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
    
    # Find all unique datasets
    all_datasets = get_all_unique_datasets(results_dict)
    
    if not all_datasets:
        print("\nNo datasets found across result files!")
        return
    
    print(f"\nFound {len(all_datasets)} unique dataset(s):")
    for ds in sorted(all_datasets):
        print(f"  - {ds}")
    
    # Create tables (using all datasets)
    comparison_df = create_comparison_table(results_dict, method_names, all_datasets)
    summary_df = create_summary_table(results_dict, method_names, all_datasets)
    dataset_perf_df = create_long_format_table(results_dict, method_names, all_datasets)
    
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

    perf_file = os.path.join(directory, "dataset_performance.csv")
    print(f"Saving dataset performance (long format) to: {perf_file}")
    dataset_perf_df.to_csv(perf_file, index=False)
    
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