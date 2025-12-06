"""
Dataset objects

-   InterventionalDataset and ObservationalDataset are individual
    "datasets" with a single graph / set of interventions

-   MetaDataset descendents are datasets of datasets which
    sample individual InterventionalDataset and ObservationalDataset
    objects depending on the traditioanl algorithm selected
"""

import os
import time
import pickle
import glob
from collections import defaultdict
from contextlib import redirect_stdout

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from . import samplers
from .utils import (
    run_fci,
    run_ges,
    run_gies,
    run_grasp,
    run_rfci,
    run_fges_tetrad,
    run_cfci,
    run_fci_max,
    run_gfci,
)
from .utils import convert_to_graphs, convert_to_item
from utils import read_csv


# ======== Start of individual datasets ========


class InterventionalDataset(Dataset):
    def __init__(self, fp_data, fp_graph, fp_regime, algorithm):
        super().__init__()
        # read raw data
        self.key = fp_graph.split("/")[-2]
        self.data = np.load(fp_data)
        self.graph = torch.from_numpy(np.load(fp_graph)).long()
        self.num_vars = self.data.shape[1]
        self.num_edges = self.graph.sum()
        self.algorithm = algorithm
        self.time = 0  # placeholder for later

        # read regimes (intervened nodes)
        with open(fp_regime) as f:
            # if >1 node intervened, formatted as a list
            lines = [line.strip() for line in f.readlines()]
        regimes = [tuple(sorted(int(x) for x in line.split(",")))
                if len(line) > 0 else () for line in lines]
        assert len(regimes) == len(self.data)

        # get unique and map to nodes
        unique_regimes = sorted(set(regimes))  # first is obs
        self.idx_to_regime = {i: reg for i, reg in enumerate(unique_regimes)}
        self.regime_to_idx = {reg: i for i, reg in enumerate(unique_regimes)}
        self.num_regimes = len(self.idx_to_regime)

        # map regimes to dataset
        self.regimes = defaultdict(list)
        for i, reg in enumerate(regimes):
            self.regimes[self.regime_to_idx[reg]].append(i)
        self.regimes = {reg: np.array(idx, dtype=int) for reg, idx in
                self.regimes.items()}  # convert to np.ndarray

        # map from nodes to regimes
        self.node_to_regime = defaultdict(list)
        for i, regime in self.idx_to_regime.items():
            for node in regime:
                self.node_to_regime[node].append(i)
        self.node_to_regime = dict(self.node_to_regime)
        # for Sachs
        for node in range(self.num_vars):
            if node not in self.node_to_regime:
                self.node_to_regime[node] = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class ObservationalDataset(Dataset):
    def __init__(self, fp_data, fp_graph, algorithm, fp_metadata=None):
        super().__init__()
        # read raw data
        self.key = fp_graph.split("/")[-2]
        self.data = np.load(fp_data)
        self.graph = torch.from_numpy(np.load(fp_graph)).long()
        self.num_vars = self.data.shape[1]
        self.num_edges = self.graph.sum()
        self.algorithm = algorithm
        self.time = 0
        
        # Load metadata for prior knowledge
        self.metadata = None
        self.variable_names = None
        self._load_metadata(fp_data, fp_metadata)
    
    def _load_metadata(self, fp_data, fp_metadata=None):
        """
        Load metadata from file. Tries multiple strategies:
        1. Explicit fp_metadata path if provided
        2. Auto-discover: look for *_meta.pkl in same directory as data
        3. Auto-discover: look for metadata.pkl in same directory as data
        """
        metadata_path = None
        
        # Strategy 1: Explicit path
        if fp_metadata and os.path.exists(fp_metadata):
            metadata_path = fp_metadata
        else:
            # Strategy 2 & 3: Auto-discover in data directory
            data_dir = os.path.dirname(fp_data)
            if data_dir:
                # Look for *_meta.pkl files
                meta_patterns = [
                    os.path.join(data_dir, "*_meta.pkl"),
                    os.path.join(data_dir, "metadata.pkl"),
                    os.path.join(data_dir, "meta.pkl"),
                ]
                for pattern in meta_patterns:
                    matches = glob.glob(pattern)
                    if matches:
                        metadata_path = matches[0]
                        break
                
                # Also try replacing data file extension with _meta.pkl
                if not metadata_path:
                    base_name = os.path.splitext(os.path.basename(fp_data))[0]
                    potential_meta = os.path.join(data_dir, f"{base_name}_meta.pkl")
                    if os.path.exists(potential_meta):
                        metadata_path = potential_meta
        
        # Load metadata if found
        if metadata_path and os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
                print(f"[METADATA] Loaded metadata from: {metadata_path}")
                
                # Extract variable names if available
                if isinstance(self.metadata, dict):
                    # Try common keys for variable names
                    for key in ['variable_names', 'variables', 'columns', 'node_names', 'temporal_order']:
                        if key in self.metadata:
                            names = self.metadata[key]
                            if isinstance(names, (list, tuple)) and len(names) == self.num_vars:
                                self.variable_names = list(names)
                                print(f"[METADATA] Found {len(self.variable_names)} variable names")
                                break
                            elif isinstance(names, (list, tuple)) and len(names) > 0:
                                # temporal_order might have all names even if subsetted
                                self.variable_names = list(names)[:self.num_vars] if len(names) >= self.num_vars else None
                                if self.variable_names:
                                    print(f"[METADATA] Using first {len(self.variable_names)} variable names from '{key}'")
                                    break
            except Exception as e:
                print(f"[METADATA] Warning: Could not load metadata from {metadata_path}: {e}")
                self.metadata = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ======== Start of meta datasets ========


class MetaDataset(Dataset):
    """
        Dataset of datasets
    """
    def __init__(self, data_file, args, splits_to_load=None):
        super().__init__()
        # read raw data
        self.args = args
        self.data_file = data_file
        data_to_load = read_csv(self.data_file)
        self.splits = defaultdict(list)
        self.data = []
        # create individual Dataset objects
        for item in tqdm(data_to_load, ncols=40):
            split = item["split"]
            if splits_to_load is not None and split not in splits_to_load:
                continue
            self.splits[split].append(len(self.data))
            if args.algorithm == "gies":
                self.data.append(InterventionalDataset(item["fp_data"],
                                                       item["fp_graph"],
                                                       item["fp_regime"],
                                                       args.algorithm))
            else:
                # Get metadata path from CSV if available, otherwise None (auto-discover)
                fp_metadata = item.get("fp_metadata", None)
                self.data.append(ObservationalDataset(item["fp_data"],
                                                      item["fp_graph"],
                                                      args.algorithm,
                                                      fp_metadata=fp_metadata))
            if args.debug and len(self.data) > 100:
                break
        # initialize per-class
        self.sampler_classes = None
        self._run_alg = get_run_alg(args.algorithm)

    def _sample_batches(self, dataset, num_batches):
        # this must be initialized per-class
        if self.sampler_classes is None:
            raise Exception("MetaDataset did not initialize sampler_classes")
        # sample batches per sampler
        kwargs = {
            "num_batches": num_batches // len(self.sampler_classes),
            "batch_size": self.args.fci_batch_size,
            "num_vars_batch": self.args.fci_vars,
        }
        for i, create_sampler in enumerate(self.sampler_classes):
            if i == 0:
                sampler = create_sampler(self.args, dataset,
                                         run_alg=self.run_alg)
                batches, feats = sampler.sample_batches(**kwargs)
                # save outputs of traditional algorithms
                if self.args.use_learned_sampler:
                    self.graphs = sampler.graphs
                    self.orders = sampler.orders
            else:
                sampler = create_sampler(self.args, dataset, visit_counts,
                                         run_alg=self.run_alg)
                # no need to replace feats
                batches.extend(sampler.sample_batches(**kwargs)[0])
            # update counts if necessary
            visit_counts = sampler.visit_counts
        return batches, feats

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        raise Exception("Not implemented")


class TrainDataset(MetaDataset):
    """
        Sample varying # of batches per individual dataset
    """
    def __init__(self, data_file, args, splits_to_load=None):
        super().__init__(data_file, args, splits_to_load)

    def __getitem__(self, idx):
        dataset = self.data[idx]
        num_batches = np.random.randint(self.args.fci_batches,
                                        self.args.fci_batches * 5, 1).item()
        batches, corrs = self._sample_batches(dataset, num_batches)
        results = self.run_alg(batches, dataset=dataset)
        graphs, orders = convert_to_graphs(results, dataset)
        if graphs is None:
            return {}
        return convert_to_item(dataset, corrs, graphs, orders)


class TestDataset(MetaDataset):
    """
        Sample fixed # of batches per individual dataset
    """
    def __init__(self, data_file, args, splits_to_load=None):
        super().__init__(data_file, args, splits_to_load)

    def __getitem__(self, idx):
        dataset = self.data[idx]
        print(f"DEBUG: Loading dataset idx={idx}, key={dataset.key}, num_vars={dataset.num_vars}, data_shape={dataset.data.shape}")
        
        # Validate dataset for problematic data before processing
        if np.isnan(dataset.data).any():
            print(f"SKIP: Dataset idx={idx}, key={dataset.key} contains NaN values - skipping")
            return {}
        
        if np.isinf(dataset.data).any():
            print(f"SKIP: Dataset idx={idx}, key={dataset.key} contains Inf values - skipping")
            return {}
        
        num_batches = self.args.fci_batches_inference
        start = time.time()  # keep track of CPU time
        try:
            batches, corrs = self._sample_batches(dataset, num_batches)
        except Exception as e:
            print(f"SKIP: Dataset idx={idx}, key={dataset.key} failed in _sample_batches: {e}")
            return {}
        # learned sampler = we already ran the algorithms
        if self.args.use_learned_sampler:
            graphs, orders = self.graphs, self.orders
        else:
            results = self.run_alg(batches, dataset=dataset)
            graphs, orders = convert_to_graphs(results, dataset)
        end = time.time()  # keep track of CPU time
        dataset.time = end - start
        if graphs is None:
            return {}
        return convert_to_item(dataset, corrs, graphs, orders)


class BaselineDataset(MetaDataset):
    """
        Used for running baseline algorithms only. Samples all variables.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # only use RandomSampler since we sample ALL nodes for baselines
        is_obs = (self.args.algorithm != "gies")
        if is_obs:
            batch_sampler = samplers.ObservationalSampler
        else:
            batch_sampler = samplers.InterventionalSampler
        score_sampler = samplers.RandomSampler
        class Sampler(batch_sampler, score_sampler):
            pass
        self.create_sampler = Sampler

    def __getitem__(self, idx):
        dataset = self.data[idx]
        num_batches = self.args.fci_batches_inference
        start = time.time()  # keep track of CPU time
        batches, corrs = self._sample_batches(dataset, num_batches)
        results = self.run_alg(batches)
        graphs, orders = convert_to_graphs(results, dataset)
        end = time.time()  # keep track of CPU time
        dataset.time = end - start
        if graphs is None:
            return {}
        return convert_to_item(dataset, corrs, graphs, orders)

    def _sample_batches(self, dataset, num_batches):
        # sample all nodes every single time
        sampler = self.create_sampler(self.args, dataset,
                                      run_alg=self.run_alg)
        batches = sampler.sample_batches(
                num_batches=num_batches,
                batch_size=self.args.fci_batch_size,
                # note this line!
                num_vars_batch=dataset.num_vars)
        return batches


class MetaObservationalDataset(MetaDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampler_classes = get_samplers(is_obs=True,
                                    is_learned=self.args.use_learned_sampler)

    def run_alg(self, batches, dataset=None):
        """
        batches: tuples (batch, order) output of sample_batches
        dataset: Optional dataset object to extract prior knowledge from
        """
        results = []
        
        dataset_name = getattr(dataset, 'key', 'unknown')
        prior_enabled = getattr(self.args, 'use_prior_knowledge', True)
        prior = None
        columns = None
        prior_message_printed = False

        # Extract prior knowledge from dataset metadata if available
        if dataset is not None and prior_enabled:
            if hasattr(dataset, 'metadata') and dataset.metadata is not None:
                try:
                    from utils.tetrad_prior_knowledge import (
                        format_prior_knowledge_for_algorithm,
                        log_prior_knowledge_summary,
                    )
                    prior = format_prior_knowledge_for_algorithm(dataset.metadata, self.args.algorithm)
                    if prior:
                        prior_message_printed = True
                        try:
                            print(f"[PRIOR] Using prior knowledge for dataset '{dataset_name}' with algorithm '{self.args.algorithm}'.")
                            log_prior_knowledge_summary(prior, dataset_name=dataset_name)
                        except Exception as log_exc:
                            print(f"[PRIOR] Could not log prior knowledge summary for '{dataset_name}': {log_exc}")
                    else:
                        print(f"[PRIOR] Dataset '{dataset_name}' metadata yielded no constraints.")
                        prior_message_printed = True
                except Exception as e:
                    print(f"Warning: Could not extract prior knowledge for dataset '{dataset_name}': {e}")
            
            # Get variable names from dataset if available
            if hasattr(dataset, 'variable_names') and dataset.variable_names:
                columns = dataset.variable_names
        elif not prior_message_printed:
            if not prior_enabled:
                print(f"[PRIOR] Prior knowledge disabled via arguments; running dataset '{dataset_name}' unconstrained.")
            else:
                print(f"[PRIOR] Dataset '{dataset_name}' missing metadata; running unconstrained.")
        
        for batch, order in batches:
            # For Tetrad algorithms, pass prior knowledge and columns
            if self.args.algorithm in ["fges", "cfci", "fcimax", "gfci", "rfci"]:
                # Use actual variable names if available, otherwise use v0, v1, etc.
                if columns is None:
                    k = batch.shape[1]
                    columns = [f"v{i}" for i in range(k)]
                
                # Check if wrapper function accepts prior and columns
                try:
                    import inspect
                    sig = inspect.signature(self._run_alg)
                    if 'prior' in sig.parameters and 'columns' in sig.parameters:
                        G = self._run_alg(batch, prior=prior, columns=columns)
                    elif 'prior' in sig.parameters:
                        G = self._run_alg(batch, prior=prior)
                    else:
                        G = self._run_alg(batch)
                except:
                    # Fallback: try with prior if it's a Tetrad algorithm
                    G = self._run_alg(batch, prior=prior) if prior is not None else self._run_alg(batch)
            else:
                G = self._run_alg(batch)
            
            if G is None:
                continue
            order = torch.from_numpy(order).long()
            results.append((G, order))
        return results


class MetaInterventionalDataset(MetaDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampler_classes = get_samplers(is_obs=False,
                                    is_learned=self.args.use_learned_sampler)

    def run_alg(self, batches):
        """
        batches: tuples (batch, order, regime) output of sample_batches
        """
        results = []
        for batch, order, regime in batches:
            graph = self._run_alg(batch, regime)
            if graph is None:
                continue
            order = torch.from_numpy(order).long()
            results.append((graph, order))
        return results


def get_samplers(is_obs, is_learned):
    # observational vs. interventional determines whether regimes
    # are sampled for each batch
    if is_obs:
        batch_sampler = samplers.ObservationalSampler
    else:
        batch_sampler = samplers.InterventionalSampler
    # fixed vs. learned determines we score nodes based on features/random
    # or the outputs of a trained model
    if is_learned:
        score_samplers = [samplers.LearnedSampler]
    else:
        score_samplers = [samplers.RandomSampler,
                          samplers.CorrelationSampler]
    # combine
    sampler_cls = []
    for score_sampler in score_samplers:
        class Sampler(batch_sampler, score_sampler):
            pass
        sampler_cls.append(Sampler)
    return sampler_cls


def get_run_alg(algorithm):
    if algorithm == "fci":
        return run_fci
    elif algorithm == "rfci":
        return run_rfci
    elif algorithm == "cfci":
        return run_cfci
    elif algorithm == "ges":
        return run_ges
    elif algorithm == "fges":
        return run_fges_tetrad
    elif algorithm == "grasp":
        return run_grasp
    elif algorithm == "gies":
        return run_gies
    elif algorithm == "fcimax":
        return run_fci_max
    elif algorithm == "gfci":
        return run_gfci
    else:
        raise Exception("Unsupported algorithm", algorithm)

