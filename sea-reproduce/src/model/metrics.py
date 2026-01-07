"""
Correct metric computation for directed edge sets.

Implements the guide's edge-set based F1 formula (Equation 26):
- Separates edge existence from edge direction
- Builds explicit directed edge sets Ê and E*
- Computes set-based precision, recall, and F1

Enhanced with:
- Oracle Top-K edge selection (k = number of true edges)
- Threshold-based edge selection
- Clean pair-state SHD computation
"""

import torch
import torch.nn.functional as F
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision


# =============================================================================
# B1. Compute probabilities and existence scores
# =============================================================================

def compute_edge_probs(logits, temperature=1.0):
    """
    Compute edge probabilities and direction predictions from logits.
    
    Args:
        logits: [N, 3] tensor with logits [z_no_edge, z_forward, z_backward]
        temperature: Temperature scaling factor (T>1 = less confident)
    
    Returns:
        probs: [N, 3] softmax probabilities
        p_edge: [N] edge existence probability = P(forward) + P(backward)
        dir_pred: [N] predicted direction {1=forward, 2=backward} (matches label scheme)
    """
    # Apply temperature scaling and softmax
    probs = F.softmax(logits / temperature, dim=-1)
    
    # Edge existence probability: P(edge) = P(forward) + P(backward)
    p_edge = probs[:, 1] + probs[:, 2]
    
    # Predicted direction: 1 + argmax(P(forward), P(backward))
    # argmax returns 0 for forward, 1 for backward
    # Adding 1 gives us {1=forward, 2=backward} matching label scheme
    dir_pred = 1 + torch.argmax(probs[:, 1:], dim=1)
    
    return probs, p_edge, dir_pred


# =============================================================================
# B2. Edge selection policy
# =============================================================================

def select_edges(p_edge, labels, mode="threshold", threshold=0.5):
    """
    Select which edges to predict based on selection policy.
    
    Args:
        p_edge: [N] edge existence probabilities
        labels: [N] true labels {0=no_edge, 1=forward, 2=backward}
        mode: "threshold" or "oracle_k"
        threshold: threshold for edge selection (only used in threshold mode)
    
    Returns:
        selected_idx: indices of selected edges
        k: number of edges selected (for logging)
    """
    if mode == "oracle_k":
        # Oracle mode: select top-k edges where k = number of true edges
        k = (labels > 0).sum().item()
        if k == 0:
            # No true edges - return empty selection
            selected_idx = torch.tensor([], dtype=torch.long, device=p_edge.device)
        else:
            # Select top k edges by probability
            _, topk_idx = torch.topk(p_edge, k=min(k, len(p_edge)))
            selected_idx = topk_idx
    elif mode == "threshold":
        # Threshold mode: select edges with p_edge >= threshold
        selected_idx = torch.where(p_edge >= threshold)[0]
        k = len(selected_idx)
    else:
        raise ValueError(f"Unknown edge selection mode: {mode}")
    
    return selected_idx, k


# =============================================================================
# B3. Build directed edge sets
# =============================================================================

def build_edge_sets(selected_idx, dir_pred, labels):
    """
    Build predicted and true directed edge sets.
    
    Args:
        selected_idx: indices of selected edges
        dir_pred: [N] predicted direction {1=forward, 2=backward}
        labels: [N] true labels {0=no_edge, 1=forward, 2=backward}
    
    Returns:
        E_hat: set of (idx, direction) tuples for predicted edges
        E_star: set of (idx, direction) tuples for true edges
    """
    # Build predicted edge set
    E_hat = set()
    for idx in selected_idx.tolist():
        direction = dir_pred[idx].item()
        E_hat.add((idx, direction))
    
    # Build true edge set
    E_star = set()
    for idx in range(len(labels)):
        label = labels[idx].item()
        if label > 0:  # 1=forward, 2=backward
            E_star.add((idx, label))
    
    return E_hat, E_star


# =============================================================================
# C1. Directed Edge-Set F1 (Equation 26)
# =============================================================================

def compute_directed_f1(E_hat, E_star):
    """
    Compute F1 on directed edge sets.
    
    F1 = 2 * |E_hat ∩ E_star| / (|E_hat| + |E_star|)
    
    Args:
        E_hat: set of (idx, direction) tuples for predicted edges
        E_star: set of (idx, direction) tuples for true edges
    
    Returns:
        f1, precision, recall, tp, fp, fn
    """
    intersection = E_hat & E_star
    
    tp = len(intersection)
    fp = len(E_hat) - tp
    fn = len(E_star) - tp
    
    precision = tp / len(E_hat) if len(E_hat) > 0 else 0.0
    recall = tp / len(E_star) if len(E_star) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return f1, precision, recall, tp, fp, fn


# =============================================================================
# C2. SHD on pair states (clean formulation)
# =============================================================================

def compute_pair_state_shd(selected_idx, dir_pred, labels):
    """
    Compute SHD using pair-state representation.
    
    Each pair has a state in {0=no_edge, 1=forward, 2=backward}.
    SHD = number of pairs where pred_state != true_state.
    
    Reversal costs 1 (standard SHD definition).
    
    Args:
        selected_idx: indices of selected edges
        dir_pred: [N] predicted direction {1=forward, 2=backward}
        labels: [N] true labels {0=no_edge, 1=forward, 2=backward}
    
    Returns:
        shd: Structural Hamming Distance
        normalized_shd: SHD / N (normalized by number of pairs)
        m_true: number of true edges
    """
    N = len(labels)
    
    # Build predicted state array (0 = no edge, 1 = forward, 2 = backward)
    pred_state = torch.zeros(N, dtype=torch.long, device=labels.device)
    for idx in selected_idx.tolist():
        pred_state[idx] = dir_pred[idx]
    
    # True state is just the labels
    true_state = labels.long()
    
    # SHD = count of mismatches
    shd = (pred_state != true_state).sum().item()
    
    # Normalize by number of pairs
    normalized_shd = shd / N if N > 0 else 0.0
    
    # Count true edges
    m_true = (labels > 0).sum().item()
    
    return shd, normalized_shd, m_true


# =============================================================================
# C3. AUROC/AUPRC on edge existence
# =============================================================================

def compute_edge_existence_metrics(p_edge, labels):
    """
    Compute AUROC/AUPRC on edge existence (threshold-independent).
    
    Args:
        p_edge: [N] edge existence probabilities
        labels: [N] true labels {0=no_edge, 1=forward, 2=backward}
    
    Returns:
        auroc: AUROC score for edge existence
        auprc: AUPRC score for edge existence
    """
    # Binary labels: does any edge exist? (long type for torchmetrics)
    t_edge = (labels > 0).long()
    
    # Compute metrics
    auroc_metric = BinaryAUROC()
    auprc_metric = BinaryAveragePrecision()
    
    auroc = auroc_metric(p_edge, t_edge)
    auprc = auprc_metric(p_edge, t_edge)
    
    return auroc, auprc


# =============================================================================
# Main entry point: compute all metrics
# =============================================================================

def compute_all_edge_metrics(logits, labels, temperature=1.0, 
                              edge_select_mode="threshold", edge_threshold=0.5):
    """
    Compute all edge metrics using the specified selection policy.
    
    This is the main entry point for edge-set based metrics.
    
    Args:
        logits: [N, 3] tensor with logits [z_no_edge, z_forward, z_backward]
        labels: [N] tensor with values {0=no_edge, 1=forward, 2=backward}
        temperature: Temperature scaling factor
        edge_select_mode: "threshold" or "oracle_k"
        edge_threshold: Threshold for edge selection (threshold mode only)
    
    Returns:
        dict with all metrics:
            - auroc, auprc: edge existence metrics (threshold-independent)
            - f1, precision, recall: directed edge-set metrics
            - tp, fp, fn: counts for debugging
            - shd, normalized_shd: pair-state SHD
            - nnz_pred: number of predicted edges
            - m_true: number of true edges
            - p_edge_stats: dict with min/mean/max of p_edge
    """
    # B1: Compute probabilities
    probs, p_edge, dir_pred = compute_edge_probs(logits, temperature)
    
    # C3: AUROC/AUPRC (threshold-independent)
    auroc, auprc = compute_edge_existence_metrics(p_edge, labels)
    
    # B2: Select edges
    selected_idx, k = select_edges(p_edge, labels, edge_select_mode, edge_threshold)
    
    # B3: Build edge sets
    E_hat, E_star = build_edge_sets(selected_idx, dir_pred, labels)
    
    # C1: Directed F1
    f1, precision, recall, tp, fp, fn = compute_directed_f1(E_hat, E_star)
    
    # C2: Pair-state SHD
    shd, normalized_shd, m_true = compute_pair_state_shd(selected_idx, dir_pred, labels)
    
    # Debug stats
    p_edge_stats = {
        "min": p_edge.min().item(),
        "mean": p_edge.mean().item(),
        "max": p_edge.max().item()
    }
    
    return {
        # Threshold-independent metrics
        "auroc": auroc.item() if hasattr(auroc, 'item') else auroc,
        "auprc": auprc.item() if hasattr(auprc, 'item') else auprc,
        
        # Directed edge-set metrics
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        
        # SHD metrics
        "shd": shd,
        "normalized_shd": normalized_shd,
        
        # Counts for debugging
        "nnz_pred": len(E_hat),
        "m_true": m_true,
        
        # Selection info
        "edge_select_mode": edge_select_mode,
        "edge_threshold": edge_threshold if edge_select_mode == "threshold" else None,
        
        # Debug stats
        "p_edge_stats": p_edge_stats
    }


# =============================================================================
# Legacy compatibility wrappers (to be removed after verification)
# =============================================================================

def compute_directed_edge_set_f1(probs, labels, threshold=0.5, temperature=1.0):
    """Legacy wrapper for compute_all_edge_metrics - F1 portion."""
    result = compute_all_edge_metrics(probs, labels, temperature, "threshold", threshold)
    return result["f1"], result["precision"], result["recall"], result["nnz_pred"]


def compute_directed_edge_set_shd(probs, labels, threshold=0.5, temperature=1.0):
    """Legacy wrapper for compute_all_edge_metrics - SHD portion."""
    result = compute_all_edge_metrics(probs, labels, temperature, "threshold", threshold)
    return result["shd"], result["normalized_shd"]
