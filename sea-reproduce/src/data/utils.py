import os
from contextlib import redirect_stdout

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import default_collate
from torch.nn.utils.rnn import pad_sequence

from causallearn.search.ConstraintBased.FCI import fci
from causallearn.search.ScoreBased.GES import ges
from causallearn.search.PermutationBased.GRaSP import grasp
from gies import fit_bic
import pandas as pd
# Import Tetrad modules
try:
    from rfci_module import run_rfci as tetrad_run_rfci
except ImportError:
    # Fallback: if RFCI module is not found
    print("Warning: RFCI module not found. Please ensure rfci_module.py is accessible.")
    tetrad_run_rfci = None

try:
    from fges_module import run_fges as tetrad_run_fges
except ImportError:
    print("Warning: FGES module not found. Please ensure fges_module.py is accessible.")
    tetrad_run_fges = None

try:
    from cfci_module import run_cfci as tetrad_run_cfci
except ImportError:
    print("Warning: CFCI module not found. Please ensure cfci_module.py is accessible.")
    tetrad_run_cfci = None

try:
    from fcimax_module import run_fci_max as tetrad_run_fci_max
except ImportError:
    print("Warning: FCIMax module not found. Please ensure fcimax_module.py is accessible.")
    tetrad_run_fci_max = None

try:
    from gfci_module import run_gfci as tetrad_run_gfci
except ImportError:
    print("Warning: GFCI module not found. Please ensure gfci_module.py is accessible.")
    tetrad_run_gfci = None

try:
    from pc_module import run_pc as tetrad_run_pc
except ImportError:
    print("Warning: PC module not found. Please ensure pc_module.py is accessible.")
    tetrad_run_pc = None

try:
    from cpc_module import run_cpc as tetrad_run_cpc
except ImportError:
    print("Warning: CPC module not found. Please ensure cpc_module.py is accessible.")
    tetrad_run_cpc = None

try:
    from boss_fci_module import run_boss_fci as tetrad_run_boss_fci
except ImportError:
    print("Warning: BOSS-FCI module not found. Please ensure boss_fci_module.py is accessible.")
    tetrad_run_boss_fci = None

try:
    from grasp_fci_module import run_grasp_fci as tetrad_run_grasp_fci
except ImportError:
    print("Warning: GRaSP-FCI module not found. Please ensure grasp_fci_module.py is accessible.")
    tetrad_run_grasp_fci = None

try:
    from tetrad_fci_module import run_tetrad_fci as tetrad_run_tetrad_fci
except ImportError:
    print("Warning: Tetrad FCI module not found. Please ensure tetrad_fci_module.py is accessible.")
    tetrad_run_tetrad_fci = None


edge_map_fci = {
    # 0 reserved for padding
    (2, 2): 1,  # (0, 0) no edge but not padded
    (1, 3): 2,  # (-1, 1)
    (3, 1): 3,  # (1, -1)
    (3, 3): 4,  # (1, 1)
    (3, 4): 5,  # (1, 2)
    (4, 3): 6,  # (2, 1)
    (4, 4): 7   # (2, 2)
}


edge_map_ges = {
    # 0 reserved for padding
    ( 0,  0): 1,  # (0, 0) no edge but not padded
    (-1,  1): 2,  # forward
    ( 1, -1): 3,  # backward
    (-1, -1): 4,  #  confused edge
}


edge_map_gies = {
    (0, 0): 0,  # reserved for padding
    (1, 1): 1,  # (0, 0) no edge but not padded
    (2, 1): 2,  # (1, 0)
    (1, 2): 3,  # (0, 1)
    (2, 2): 4,  # (1, 1) not DAG but exists (?)
}


def convert_pytetrad_pag_to_causallearn(adj):
    """
    Convert PyTetrad PAG format to causal-learn PAG format.
    
    PyTetrad uses semantic edge encoding:
        -1 = backward edge (<-)
         0 = no edge
         1 = undirected edge (-)
         2 = forward edge (->)
    
    causal-learn uses endpoint marker encoding:
        -1 = tail (-)
         0 = no edge
         1 = arrowhead (>)
         2 = circle (o)
    
    This conversion ensures PyTetrad outputs are compatible with the original
    edge_map_fci which expects causal-learn format.
    
    Args:
        adj: Adjacency matrix in PyTetrad format (n_vars, n_vars)
    
    Returns:
        Adjacency matrix in causal-learn format (n_vars, n_vars)
    """
    n = adj.shape[0]
    adj_cl = np.zeros((n, n), dtype=int)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            
            ij = adj[i, j]
            ji = adj[j, i]
            
            # No edge: both are 0
            if ij == 0 and ji == 0:
                adj_cl[i, j] = 0
                adj_cl[j, i] = 0
            
            # Directed edge i→j: PyTetrad (2, -1)
            # Convert to causal-learn: tail at i (-1), arrow at j (1)
            elif ij == 2 and ji == -1:
                adj_cl[i, j] = -1  # tail at j from i's perspective
                adj_cl[j, i] = 1   # arrow at i from j's perspective
            
            # Directed edge i←j: PyTetrad (-1, 2)
            # Convert to causal-learn: arrow at i (1), tail at j (-1)
            elif ij == -1 and ji == 2:
                adj_cl[i, j] = 1   # arrow at j from i's perspective
                adj_cl[j, i] = -1  # tail at i from j's perspective
            
            # Undirected edge i-j: PyTetrad (1, 1)
            # Convert to causal-learn: tail-tail (-1, -1)
            elif ij == 1 and ji == 1:
                adj_cl[i, j] = -1
                adj_cl[j, i] = -1
            
            # Bidirected edge i↔j: PyTetrad would be (2, 2) if it occurs
            # Convert to causal-learn: arrow-arrow (1, 1)
            elif ij == 2 and ji == 2:
                adj_cl[i, j] = 1
                adj_cl[j, i] = 1
            
            # Edge cases: partial orientations or mixed types
            # These shouldn't occur in standard PyTetrad output, but handle gracefully
            else:
                # Default: preserve as-is (might need adjustment based on actual data)
                adj_cl[i, j] = ij
                adj_cl[j, i] = ji
    
    return adj_cl

def convert_to_item(dataset, feats, graphs, orders):
    """
        Convert to batchable item format
    """
    # view + cat for speed > stack
    feats = [torch.from_numpy(f).float() for f in feats]
    feats = torch.cat([f.view(1, *f.size()) for f in feats], dim=0)

    # NOTE inverse is NOT the global ordering, which is given by unique,
    # but it is consistent within this single item.
    # unique is (B*N, 2), inverse is (B*N)
    unique, inverse = torch.unique(orders, dim=0,
            return_inverse=True)

    # get true edges
    labels = dataset.graph
    item = {
        "key": dataset.key,
        "label": labels,
        "input": graphs,
        "feats": feats,
        "index": inverse.reshape(len(graphs), -1),  # shape is (T, k*k)
        "unique": unique + 1,  # (num unique, 2)  NEED TO 1 INDEX FOR PADDING
        "time": dataset.time  # CPU time elapsed so far
    }
    return item


def convert_result_to_lg(g, edge_map):
    edge_attr = []
    for i in range(len(g)):
        for j in range(len(g)):
            # eliminate diagonal from the source
            if i == j:
                continue
            ij = g[i,j]
            ji = g[j,i]
            edge_attr.append(edge_map[(ij, ji)])
    return edge_attr


def convert_to_graphs(results, dataset):
    """
        Convert to PyG line graphs
    """
    graphs = []
    orders = []
    for G, order in results:
        if dataset.algorithm == "fci":
            # G includes {-1, 0, 1, 2} for FCI
            # to include padding, we should map this to {1, 2, 3, 4}
            graphs.append(convert_result_to_lg(G + 2, edge_map_fci))
        elif dataset.algorithm in ["pc", "cpc"]:
            # PC and CPC output PAG format {-1, 0, 1, 2} for FCI aggregator
            # Values: -1 (backward), 0 (no edge), 1 (undirected), 2 (forward)
            # Shift by +2 to map to {1, 2, 3, 4} for edge_map_fci
            graphs.append(convert_result_to_lg(G + 2, edge_map_fci))
        elif dataset.algorithm in ["ges", "grasp", "fges"]:
            # G includes {-1, 0, 1} for GES, GRaSP, FGES (CPDAG-based)
            graphs.append(convert_result_to_lg(G, edge_map_ges))
        elif dataset.algorithm in ["rfci", "cfci", "fcimax", "gfci", "bossfci", "graspfci", "tetradfci"]:
            # PAG-based algorithms return FCI-compatible values {-1,0,1,2}; shift by +2
            graphs.append(convert_result_to_lg(G + 2, edge_map_fci))
        else:
            # GIES: G includes {0, 1}
            # to include padding, we should map this to {1, 2}
            graphs.append(convert_result_to_lg(G + 1, edge_map_gies))
        orders.append(torch.cartesian_prod(order, order))
    if len(graphs) == 0:
        return None, None
    # (T, k*k)
    graphs = torch.tensor(graphs, dtype=torch.long)
    orders = torch.cat(orders, dim=0)
    # remove diagonal
    orders = orders[orders[:,0] != orders[:,1]]
    return graphs, orders


def run_fci(batch):
    try:
        with open("dummy", "w") as f:
            with redirect_stdout(f):
                G, edges = fci(batch,
                               independence_test_method="fisherz",
                               alpha=0.05,  # default
                               depth=-1,  # no max, fine if only 5-10 vars
                               max_path_length=-1,  # no max
                               verbose=False,
                               show_progress=False)
    # sometimes, very rarely, FCI fails...
    except:
        return
    return G.graph


def run_ges(batch):
    try:
        output = ges(batch,
                     score_func="local_score_BIC")
        return output["G"].graph
    except:
        return


def run_grasp(batch):
    try:
        with open("dummy", "w") as f:
            with redirect_stdout(f):
                output = grasp(batch,
                           score_func="local_score_BIC")
        return output.graph
    except:
        return


def run_gies(batch, regime):
    try:
        graph, score = fit_bic(data=batch, I=regime, A0=None,
                phases=["forward", "backward", "turning"],
                iterate=True, debug=0)  # a real verbose flag!
    except:
        return
    return graph.astype(int)


def run_rfci(batch, alpha=0.05, depth=-1, include_undirected=True, count_partial=False,
              prior=None, columns=None):
    """
    Tetrad RFCI wrapper for SEA pipeline.
    
    Args:
        batch: np.ndarray with shape (n_samples, k_vars)
        alpha: significance level for independence tests
        depth: max conditioning set size (-1 = unlimited)
        include_undirected: whether to include undirected edges
        count_partial: whether to count partial orientations
        prior: Optional prior knowledge dictionary
        columns: Optional list of column names (defaults to v0, v1, ...)
    
    Returns:
        np.ndarray: binary adjacency matrix (k_vars, k_vars) with dtype=int
    """
    if tetrad_run_rfci is None:
        print("Warning: RFCI module not available, skipping batch")
        return None
    
    try:
        k = batch.shape[1]
        # Create column names for the variables if not provided
        if columns is None:
            columns = [f"v{i}" for i in range(k)]
        
        # Call the RFCI function with numpy array and column names
        adj = tetrad_run_rfci(
            batch,
            columns=columns,
            alpha=alpha,
            depth=depth,
            include_undirected=include_undirected,
            count_partial=count_partial,
            prior=prior
        )
        
        # Convert PyTetrad PAG format to causal-learn PAG format
        # This ensures compatibility with the original edge_map_fci
        adj = convert_pytetrad_pag_to_causallearn(adj)
        
        # Ensure int np.ndarray
        return adj.astype(int)
    except Exception as e:
        print(f"RFCI failed for batch: {e}")
        return None


def run_fges_tetrad(batch, penalty_discount=1.0, max_degree=-1, parallel=False, 
                     equivalent_sample_size=10.0, orient_cpdag_to_dag=False, prior=None, columns=None):
    """
    Tetrad FGES wrapper for SEA pipeline.
    Returns GES-compatible format {-1, 0, 1} for use with edge_map_ges.
    
    NOTE: orient_cpdag_to_dag=False preserves CPDAG format to match causal-learn GES output.
    This is critical for SEA aggregator compatibility - the aggregator expects edge type 4
    (undirected/confused edges) which are lost when converting CPDAG to DAG.
    
    Args:
        batch: np.ndarray with shape (n_samples, k_vars)
        penalty_discount: score complexity penalty (1.0 = standard BIC, matches causal-learn GES)
        max_degree: limit degree per node (-1 = unlimited)
        parallel: try to use multiple threads
        equivalent_sample_size: for BDeu score on discrete data
        orient_cpdag_to_dag: convert CPDAG to DAG (False recommended for SEA compatibility)
        prior: Optional prior knowledge dictionary
        columns: Optional list of column names (defaults to v0, v1, ...)
    
    Returns:
        np.ndarray: adjacency matrix (k_vars, k_vars) with values {-1, 0, 1}
    """
    if tetrad_run_fges is None:
        print("Warning: FGES module not available, skipping batch")
        return None
    
    try:
        k = batch.shape[1]
        if columns is None:
            columns = [f"v{i}" for i in range(k)]
        
        adj = tetrad_run_fges(
            batch,
            columns=columns,
            penalty_discount=penalty_discount,
            max_degree=max_degree,
            parallel=parallel,
            equivalent_sample_size=equivalent_sample_size,
            orient_cpdag_to_dag=orient_cpdag_to_dag,
            prior=prior
        )
        
        return adj.astype(int)
    except Exception as e:
        print(f"FGES failed for batch: {e}")
        return None


def run_cfci(batch, alpha=0.01, depth=-1, include_undirected=True, prior=None, columns=None):
    """
    Tetrad CFCI wrapper for SEA pipeline.
    Returns FCI-compatible adjacency matrix directly from the module.
    
    Args:
        batch: np.ndarray with shape (n_samples, k_vars)
        alpha: significance level for independence tests
        depth: max conditioning set size (-1 = unlimited)
        include_undirected: whether to include undirected edges
        prior: Optional prior knowledge dictionary
        columns: Optional list of column names (defaults to v0, v1, ...)
    
    Returns:
        np.ndarray: FCI-compatible adjacency matrix (k_vars, k_vars) with values {-1, 0, 1, 2}
    """
    if tetrad_run_cfci is None:
        print("Warning: CFCI module not available, skipping batch")
        return None
    
    try:
        k = batch.shape[1]
        if columns is None:
            columns = [f"v{i}" for i in range(k)]
        
        # Module returns PyTetrad PAG format {-1, 0, 1, 2}
        adj_fci = tetrad_run_cfci(
            batch,
            columns=columns,
            alpha=alpha,
            depth=depth,
            include_undirected=include_undirected,
            prior=prior
        )
        
        # Convert PyTetrad PAG format to causal-learn PAG format
        adj_fci = convert_pytetrad_pag_to_causallearn(adj_fci)
        
        return adj_fci.astype(int)
    except Exception as e:
        print(f"CFCI failed for batch: {e}")
        return None


def run_fci_max(batch, alpha=0.01, depth=-1, include_undirected=True, prior=None, columns=None):
    """
    Tetrad FCI-Max wrapper for SEA pipeline.
    Returns FCI-compatible adjacency matrix directly from the module.
    
    Args:
        batch: np.ndarray with shape (n_samples, k_vars)
        alpha: significance level for independence tests
        depth: max conditioning set size (-1 = unlimited)
        include_undirected: whether to include undirected edges
        prior: Optional prior knowledge dictionary
        columns: Optional list of column names (defaults to v0, v1, ...)
    
    Returns:
        np.ndarray: FCI-compatible adjacency matrix (k_vars, k_vars) with values {-1, 0, 1, 2}
    """
    if tetrad_run_fci_max is None:
        print("Warning: FCIMax module not available, skipping batch")
        return None
    
    try:
        k = batch.shape[1]
        if columns is None:
            columns = [f"v{i}" for i in range(k)]
        
        # Module returns PyTetrad PAG format {-1, 0, 1, 2}
        adj_fci = tetrad_run_fci_max(
            batch,
            columns=columns,
            alpha=alpha,
            depth=depth,
            include_undirected=include_undirected,
            prior=prior
        )
        
        # Convert PyTetrad PAG format to causal-learn PAG format
        adj_fci = convert_pytetrad_pag_to_causallearn(adj_fci)
        
        return adj_fci.astype(int)
    except Exception as e:
        print(f"FCI-Max failed for batch: {e}")
        return None


def run_gfci(batch, alpha=0.01, depth=-1, penalty_discount=2.0, include_undirected=True, prior=None, columns=None):
    """
    Tetrad GFCI wrapper for SEA pipeline.
    Returns FCI-compatible adjacency matrix directly from the module.
    
    Args:
        batch: np.ndarray with shape (n_samples, k_vars)
        alpha: significance level for independence tests
        depth: max conditioning set size (-1 = unlimited)
        penalty_discount: score complexity penalty
        include_undirected: whether to include undirected edges
        prior: Optional prior knowledge dictionary
        columns: Optional list of column names (defaults to v0, v1, ...)
    
    Returns:
        np.ndarray: FCI-compatible adjacency matrix (k_vars, k_vars) with values {-1, 0, 1, 2}
    """
    if tetrad_run_gfci is None:
        print("Warning: GFCI module not available, skipping batch")
        return None
    
    try:
        k = batch.shape[1]
        if columns is None:
            columns = [f"v{i}" for i in range(k)]
        
        # Module returns PyTetrad PAG format {-1, 0, 1, 2}
        adj_fci = tetrad_run_gfci(
            batch,
            columns=columns,
            alpha=alpha,
            depth=depth,
            penalty_discount=penalty_discount,
            include_undirected=include_undirected,
            prior=prior
        )
        
        # Convert PyTetrad PAG format to causal-learn PAG format
        adj_fci = convert_pytetrad_pag_to_causallearn(adj_fci)
        
        return adj_fci.astype(int)
    except Exception as e:
        print(f"GFCI failed for batch: {e}")
        return None


def run_pc(batch, alpha=0.05, depth=-1, include_undirected=True, prior=None, columns=None, output_format="ges"):
    """
    Tetrad PC wrapper for SEA pipeline.
    Returns adjacency matrix in specified format (GES or PAG).
    
    Args:
        batch: np.ndarray with shape (n_samples, k_vars)
        alpha: significance level for independence tests
        depth: max conditioning set size (-1 = unlimited)
        include_undirected: whether to include undirected edges
        prior: Optional prior knowledge dictionary
        columns: Optional list of column names (defaults to v0, v1, ...)
        output_format: "ges" for GIES aggregator (default) or "pag" for FCI aggregator
    
    Returns:
        np.ndarray: Adjacency matrix (k_vars, k_vars)
            - If output_format="ges": GES-compatible values {-1, 0, 1}
            - If output_format="pag": PAG-compatible values {-1, 0, 1, 2} for FCI aggregator
    """
    if tetrad_run_pc is None:
        print("Warning: PC module not available, skipping batch")
        return None
    
    try:
        k = batch.shape[1]
        if columns is None:
            columns = [f"v{i}" for i in range(k)]
        
        # Module returns format based on output_format parameter
        adj = tetrad_run_pc(
            batch,
            columns=columns,
            alpha=alpha,
            depth=depth,
            include_undirected=include_undirected,
            prior=prior,
            output_format=output_format
        )
        
        return adj.astype(int)
    except Exception as e:
        print(f"PC failed for batch: {e}")
        return None


def run_cpc(batch, alpha=0.05, depth=-1, include_undirected=True, prior=None, columns=None, output_format="ges"):
    """
    Tetrad CPC (Conservative PC) wrapper for SEA pipeline.
    Returns adjacency matrix in specified format (GES or PAG).
    
    Args:
        batch: np.ndarray with shape (n_samples, k_vars)
        alpha: significance level for independence tests
        depth: max conditioning set size (-1 = unlimited)
        include_undirected: whether to include undirected edges
        prior: Optional prior knowledge dictionary
        columns: Optional list of column names (defaults to v0, v1, ...)
        output_format: "ges" for GIES aggregator (default) or "pag" for FCI aggregator
    
    Returns:
        np.ndarray: Adjacency matrix (k_vars, k_vars)
            - If output_format="ges": GES-compatible values {-1, 0, 1}
            - If output_format="pag": PAG-compatible values {-1, 0, 1, 2} for FCI aggregator
    """
    if tetrad_run_cpc is None:
        print("Warning: CPC module not available, skipping batch")
        return None
    
    try:
        k = batch.shape[1]
        if columns is None:
            columns = [f"v{i}" for i in range(k)]
        
        # Module returns format based on output_format parameter
        adj = tetrad_run_cpc(
            batch,
            columns=columns,
            alpha=alpha,
            depth=depth,
            include_undirected=include_undirected,
            prior=prior,
            output_format=output_format
        )
        
        return adj.astype(int)
    except Exception as e:
        print(f"CPC failed for batch: {e}")
        return None


def run_boss_fci(batch, alpha=0.01, depth=-1, penalty_discount=2.0, include_undirected=True, prior=None, columns=None):
    """
    Tetrad BOSS-FCI wrapper for SEA pipeline.
    Returns FCI-compatible adjacency matrix directly from the module.
    
    BOSS-FCI is a hybrid algorithm that combines BOSS (Best Order Score Search)
    with FCI orientation rules to produce a PAG.
    
    Args:
        batch: np.ndarray with shape (n_samples, k_vars)
        alpha: significance level for independence tests
        depth: max conditioning set size (-1 = unlimited)
        penalty_discount: score complexity penalty
        include_undirected: whether to include undirected edges
        prior: Optional prior knowledge dictionary
        columns: Optional list of column names (defaults to v0, v1, ...)
    
    Returns:
        np.ndarray: FCI-compatible adjacency matrix (k_vars, k_vars) with values {-1, 0, 1, 2}
    """
    if tetrad_run_boss_fci is None:
        print("Warning: BOSS-FCI module not available, skipping batch")
        return None
    
    try:
        k = batch.shape[1]
        if columns is None:
            columns = [f"v{i}" for i in range(k)]
        
        # Module returns PyTetrad PAG format {-1, 0, 1, 2}
        adj_fci = tetrad_run_boss_fci(
            batch,
            columns=columns,
            alpha=alpha,
            depth=depth,
            penalty_discount=penalty_discount,
            include_undirected=include_undirected,
            prior=prior
        )
        
        # Convert PyTetrad PAG format to causal-learn PAG format
        adj_fci = convert_pytetrad_pag_to_causallearn(adj_fci)
        
        return adj_fci.astype(int)
    except Exception as e:
        print(f"BOSS-FCI failed for batch: {e}")
        return None


def run_grasp_fci(batch, alpha=0.01, depth=-1, penalty_discount=2.0, include_undirected=True, prior=None, columns=None):
    """
    Tetrad GRaSP-FCI wrapper for SEA pipeline.
    Returns FCI-compatible adjacency matrix directly from the module.
    
    GRaSP-FCI is a hybrid algorithm that combines GRaSP (Greedy Relaxations of
    Sparsest Permutation) with FCI orientation rules to produce a PAG.
    
    Args:
        batch: np.ndarray with shape (n_samples, k_vars)
        alpha: significance level for independence tests
        depth: max conditioning set size (-1 = unlimited)
        penalty_discount: score complexity penalty
        include_undirected: whether to include undirected edges
        prior: Optional prior knowledge dictionary
        columns: Optional list of column names (defaults to v0, v1, ...)
    
    Returns:
        np.ndarray: FCI-compatible adjacency matrix (k_vars, k_vars) with values {-1, 0, 1, 2}
    """
    if tetrad_run_grasp_fci is None:
        print("Warning: GRaSP-FCI module not available, skipping batch")
        return None
    
    try:
        k = batch.shape[1]
        if columns is None:
            columns = [f"v{i}" for i in range(k)]
        
        # Module returns PyTetrad PAG format {-1, 0, 1, 2}
        adj_fci = tetrad_run_grasp_fci(
            batch,
            columns=columns,
            alpha=alpha,
            depth=depth,
            penalty_discount=penalty_discount,
            include_undirected=include_undirected,
            prior=prior
        )
        
        # Convert PyTetrad PAG format to causal-learn PAG format
        adj_fci = convert_pytetrad_pag_to_causallearn(adj_fci)
        
        return adj_fci.astype(int)
    except Exception as e:
        print(f"GRaSP-FCI failed for batch: {e}")
        return None


def run_tetrad_fci(batch, alpha=0.01, depth=-1, max_path_length=-1, include_undirected=True, prior=None, columns=None):
    """
    Tetrad standard FCI wrapper for SEA pipeline.
    Returns FCI-compatible adjacency matrix directly from the module.
    
    This is the standard FCI algorithm implemented in Tetrad, which handles
    latent confounders and selection bias to produce a PAG.
    
    Args:
        batch: np.ndarray with shape (n_samples, k_vars)
        alpha: significance level for independence tests
        depth: max conditioning set size (-1 = unlimited)
        max_path_length: max path length for discriminating path rule (-1 = unlimited)
        include_undirected: whether to include undirected edges
        prior: Optional prior knowledge dictionary
        columns: Optional list of column names (defaults to v0, v1, ...)
    
    Returns:
        np.ndarray: FCI-compatible adjacency matrix (k_vars, k_vars) with values {-1, 0, 1, 2}
    """
    if tetrad_run_tetrad_fci is None:
        print("Warning: Tetrad FCI module not available, skipping batch")
        return None
    
    try:
        k = batch.shape[1]
        if columns is None:
            columns = [f"v{i}" for i in range(k)]
        
        # Module returns PyTetrad PAG format {-1, 0, 1, 2}
        adj_fci = tetrad_run_tetrad_fci(
            batch,
            columns=columns,
            alpha=alpha,
            depth=depth,
            max_path_length=max_path_length,
            include_undirected=include_undirected,
            prior=prior
        )
        
        # Convert PyTetrad PAG format to causal-learn PAG format
        adj_fci = convert_pytetrad_pag_to_causallearn(adj_fci)
        
        return adj_fci.astype(int)
    except Exception as e:
        print(f"Tetrad FCI failed for batch: {e}")
        return None


def collate(args, batch):
    """
        Overwrite default_collate for jagged tensors
    """
    # initialize new batch
    # and skip invalid items haha
    keys = ["label", "input", "key", "index", "feats", "unique", "time"]
    batch = {key:[item[key] for item in batch if key in item] for key in keys}
    
    # Check if batch is empty (all datasets were skipped)
    if all(len(val) == 0 for val in batch.values()):
        print("WARNING: Entire batch is empty (all datasets skipped), returning None")
        return None
    
    new_batch = {}
    for key, val in batch.items():
        # Skip empty values (happens when datasets are skipped)
        if len(val) == 0:
            continue
            
        if not torch.is_tensor(val[0]) or val[0].ndim == 0:
            new_batch[key] = default_collate(val)
        # don't collate this; adjust based on train/val
        elif "order" in key:
            offset = []
            for i, v in enumerate(val):
                if i == 0:
                    offset.extend([0] * len(v))
                else:
                    offset.extend([offset[-1] + len(val[i-1])] * len(v))
            new_batch[f"{key}_len"] = torch.tensor(offset)
            new_batch[key] = [v.clone() for v in val]
        elif key in ["feats", "label"]:
            # each is [N, N] so require 2D padding
            max_nodes = max([len(v) for v in val])
            for i, v in enumerate(val):
                pad = max_nodes - len(v)
                if pad > 0:
                    val[i] = F.pad(v, (0, pad, 0, pad))
            new_batch[key] = torch.stack(val, dim=0)
        else:
            new_batch[f"{key}_len"] = torch.tensor([len(v) for v in val])
            # dimension = 1 is now time
            # each of these should be (length, )
            padded = pad_sequence(val, batch_first=True)
            new_batch[key] = padded
    return new_batch


def get_mask(lens, max_len=None):
    # mask where 0 is padding and 1 is token
    if max_len is None:
        max_len = lens.max()
    mask = torch.arange(max_len)[None, :] < lens[:, None]
    return mask

