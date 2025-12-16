"""
Main inference file. Call via inference.sh
"""
import os
import sys
import yaml
import random
from collections import defaultdict

import numpy as np
import torch
import pytorch_lightning as pl

from args import parse_args
from data import InferenceDataModule, BaselineDataModule
from model import load_model
from helpers import printt, get_suffix, save_pickle


# TODO why not?
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def set_seed(seed):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Make cudnn deterministic (may slow down training slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_structural_metrics(pred_list, true_list):
    """
    Compute SHD, NNZ, SID from prediction and true edge lists.
    
    Args:
        pred_list: list of predicted edge probabilities (flattened)
        true_list: list of true edge labels (flattened)
    
    Returns:
        shd: Structural Hamming Distance
        normalized_shd: SHD normalized by number of possible edges
        nnz: Number of non-zero (predicted) edges
        sid: Structural Intervention Distance (placeholder)
        normalized_sid: Normalized SID (placeholder)
    """
    if not pred_list or not true_list:
        return None, None, None, None, None
    
    n_edges = len(pred_list)
    if n_edges == 0:
        return 0, 0.0, 0, None, None
    
    # Convert to numpy arrays
    pred_arr = np.array(pred_list)
    true_arr = np.array(true_list)
    
    # Threshold predictions at 0.5 to get binary predictions
    pred_binary = (pred_arr > 0.5).astype(int)
    true_binary = (true_arr > 0).astype(int)
    
    # SHD = number of differing edges (mismatches)
    shd = int(np.sum(pred_binary != true_binary))
    
    # Normalized SHD (by total number of edge positions)
    # n_edges represents the upper triangle entries for directed edges
    # For n_vars nodes: n_edges = n_vars * (n_vars - 1)
    # Solve: n^2 - n = n_edges => n = (1 + sqrt(1 + 4*n_edges)) / 2
    n_vars = int((1 + np.sqrt(1 + 4 * n_edges)) / 2)
    max_edges = n_vars * (n_vars - 1)
    normalized_shd = shd / max_edges if max_edges > 0 else 0.0
    
    # NNZ = number of predicted edges (non-zero in prediction)
    nnz = int(np.sum(pred_binary))
    
    # SID (Structural Intervention Distance) - requires DAG comparison
    # This is more complex and typically requires specialized libraries
    # For now, we'll leave it as None (can be added later if needed)
    sid = None
    normalized_sid = None
    
    return shd, normalized_shd, nnz, sid, normalized_sid


def main():
    printt("Starting...")
    with open("data/goodluck.txt") as f:
        for line in f:
            print(line, end="")

    args = parse_args()
    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.set_float32_matmul_precision("medium")

    # data loaders
    if args.model == "baseline":
        data = BaselineDataModule(args)
    else:
        data = InferenceDataModule(args)
    printt("Finished loading raw data.")

    # setup
    set_seed(args.seed)
    model = load_model(args)
    printt("Finished loading model.")

    # inference
    kwargs = {
        "accelerator": "gpu" if args.gpu >= 0 else "cpu"
    }
    if args.gpu >= 0:
        kwargs["devices"] = [args.gpu]
    tester = pl.Trainer(num_nodes=1,
                        enable_checkpointing=False,
                        logger=False,
                        **kwargs)

    best_path = args.checkpoint_path
    printt(f"DEBUG: About to run prediction with checkpoint: {best_path}")
    printt(f"DEBUG: Data module loaded: {data}")
    printt(f"DEBUG: Model loaded: {model}")
    
    if os.path.exists(best_path):
        printt("DEBUG: Checkpoint exists, running prediction...")
        try:
            results = tester.predict(model, data, ckpt_path=best_path)
            printt(f"DEBUG: Prediction completed. Results type: {type(results)}")
            if results is None:
                printt("ERROR: tester.predict() returned None!")
            else:
                printt(f"DEBUG: Results length: {len(list(results)) if results else 'None'}")
        except Exception as e:
            printt(f"ERROR during prediction: {e}")
            import traceback
            traceback.print_exc()
            return
    # baselines only
    else:
        printt("Inference with NO checkpoint")
        try:
            results = tester.predict(model, data)
            printt(f"DEBUG: Baseline prediction completed. Results type: {type(results)}")
        except Exception as e:
            printt(f"ERROR during baseline prediction: {e}")
            import traceback
            traceback.print_exc()
            return
    
    printt("DEBUG: About to process results...")
    if results is None:
        printt("ERROR: Results is None, cannot process!")
        return
        
    # post-process results for dispatcher
    results_dict = defaultdict(list)
    try:
        for batch in results:
            printt(f"DEBUG: Processing batch: {type(batch)}")
            # Skip None batches (from skipped datasets)
            if batch is None:
                printt("DEBUG: Skipping None batch (dataset was skipped)")
                continue
            for k, v in batch.items():
                if type(v) is list:
                    results_dict[k].extend(v)
                else:
                    results_dict[k].append(v)
    except Exception as e:
        printt(f"ERROR processing results: {e}")
        import traceback
        traceback.print_exc()
        return
    # organize by data setting
    key_to_metrics = defaultdict(lambda: defaultdict(list))
    auc = results_dict["auroc"]
    prc = results_dict["auprc"]
    time_list = results_dict["time"]
    true = results_dict["true"]
    pred = results_dict["pred"]
    keys = results_dict["key"]
    
    # Validate list lengths match
    n_keys = len(keys)
    if n_keys == 0:
        printt("WARNING: No valid results to process (all datasets may have been skipped)")
        save_pickle(args.results_file, {})
        printt("All done. Exiting.")
        return
    
    printt(f"DEBUG: Processing {n_keys} results. List lengths: auc={len(auc)}, prc={len(prc)}, time={len(time_list)}, true={len(true)}, pred={len(pred)}")
    
    for i, key in enumerate(keys):
        if i < len(auc):
            key_to_metrics[key]["auc"].append(auc[i])
        if i < len(prc):
            key_to_metrics[key]["prc"].append(prc[i])
        if "f1" in results_dict and i < len(results_dict["f1"]):
            key_to_metrics[key]["f1"].append(results_dict["f1"][i])
        if i < len(time_list):
            key_to_metrics[key]["time"].append(time_list[i])
        if i < len(true):
            key_to_metrics[key]["true"].append(true[i])
        if i < len(pred):
            key_to_metrics[key]["pred"].append(pred[i])
        
        # Compute structural metrics from predictions and ground truth
        if i < len(pred) and i < len(true) and pred[i] and true[i]:
            shd, norm_shd, nnz, sid, norm_sid = compute_structural_metrics(
                pred[i], true[i]
            )
            if shd is not None:
                key_to_metrics[key]["shd"].append(shd)
            if norm_shd is not None:
                key_to_metrics[key]["normalized_shd"].append(norm_shd)
            if nnz is not None:
                key_to_metrics[key]["nnz"].append(nnz)
            if sid is not None:
                key_to_metrics[key]["sid"].append(sid)
            if norm_sid is not None:
                key_to_metrics[key]["normalized_sid"].append(norm_sid)
    
    key_to_metrics = dict(key_to_metrics)
    save_pickle(args.results_file, key_to_metrics)
    printt("All done. Exiting.")


if __name__ == "__main__":
    main()

