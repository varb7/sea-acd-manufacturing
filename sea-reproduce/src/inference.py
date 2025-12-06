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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


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
    time = results_dict["time"]
    true = results_dict["true"]
    pred = results_dict["pred"]
    for i, key in enumerate(results_dict["key"]):
        key_to_metrics[key]["auc"].append(auc[i])
        key_to_metrics[key]["prc"].append(prc[i])
        key_to_metrics[key]["time"].append(time[i])
        key_to_metrics[key]["true"].append(true[i])
        key_to_metrics[key]["pred"].append(pred[i])
    key_to_metrics = dict(key_to_metrics)
    save_pickle(args.results_file, key_to_metrics)
    printt("All done. Exiting.")


if __name__ == "__main__":
    main()

