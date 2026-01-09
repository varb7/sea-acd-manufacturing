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


def compute_legacy_auc(probs, labels):
    """
    Legacy AUROC/AUPRC calculation for comparison with old results.
    
    OLD BEHAVIOR: Flatten forward/backward probabilities into a single vector
    and compare against flattened labels. This evaluates P(forward) and P(backward)
    separately rather than P(edge).
    
    Args:
        probs: [N, 3] softmax probabilities [p_no_edge, p_forward, p_backward]
        labels: [N] true labels {0=no_edge, 1=forward, 2=backward}
    
    Returns:
        auroc: AUROC score (legacy method)
        auprc: AUPRC score (legacy method)
    """
    # OLD: p = F.softmax(p, dim=-1)[:,1:].t().reshape(-1)
    # This takes columns 1,2 (forward, backward), transposes, and flattens
    # Result: [P(fwd)_1, P(fwd)_2, ..., P(bwd)_1, P(bwd)_2, ...]
    p_flat = probs[:, 1:].t().reshape(-1)  # Shape: [2N]
    
    # OLD: t was already flattened from torch.cat([forward_i, backward_i])
    # For legacy mode, we need to reconstruct this format
    # forward_label = 1 if label == 1, else 0
    # backward_label = 1 if label == 2, else 0
    forward_labels = (labels == 1).long()
    backward_labels = (labels == 2).long()
    t_flat = torch.cat([forward_labels, backward_labels])  # Shape: [2N]
    
    # Compute metrics
    auroc_metric = BinaryAUROC()
    auprc_metric = BinaryAveragePrecision()
    
    auroc = auroc_metric(p_flat, t_flat)
    auprc = auprc_metric(p_flat, t_flat)
    
    return auroc, auprc


# =============================================================================
# Main entry point: compute all metrics
# =============================================================================

def compute_all_edge_metrics(logits, labels, temperature=1.0, edge_threshold=0.5):
    """
    Compute all edge metrics using BOTH threshold and oracle-K selection policies.
    
    This is the main entry point for edge-set based metrics.
    Always uses legacy AUC (flattened forward/backward probs) for consistency with original SEA.
    
    Args:
        logits: [N, 3] tensor with logits [z_no_edge, z_forward, z_backward]
        labels: [N] tensor with values {0=no_edge, 1=forward, 2=backward}
        temperature: Temperature scaling factor
        edge_threshold: Threshold for edge selection in threshold mode (default=0.5)
    
    Returns:
        dict with all metrics:
            - auroc, auprc: legacy AUC metrics (threshold-independent)
            - f1_threshold, precision_threshold, recall_threshold: threshold-based metrics
            - f1_oracle_k, precision_oracle_k, recall_oracle_k: oracle-K based metrics
            - shd_threshold, shd_oracle_k: SHD for each mode
            - tp, fp, fn counts for both modes
            - nnz_pred: number of predicted edges (threshold mode)
            - m_true: number of true edges
            - p_edge_stats: dict with min/mean/max of p_edge
    """
    # B1: Compute probabilities
    probs, p_edge, dir_pred = compute_edge_probs(logits, temperature)
    
    # C3: AUROC/AUPRC - Always use legacy (flattened forward/backward) for consistency
    auroc, auprc = compute_legacy_auc(probs, labels)
    
    # ========== THRESHOLD MODE ==========
    selected_idx_thresh, k_thresh = select_edges(p_edge, labels, "threshold", edge_threshold)
    E_hat_thresh, E_star = build_edge_sets(selected_idx_thresh, dir_pred, labels)
    f1_thresh, prec_thresh, rec_thresh, tp_thresh, fp_thresh, fn_thresh = compute_directed_f1(E_hat_thresh, E_star)
    shd_thresh, norm_shd_thresh, m_true = compute_pair_state_shd(selected_idx_thresh, dir_pred, labels)
    
    # ========== ORACLE-K MODE ==========
    selected_idx_oracle, k_oracle = select_edges(p_edge, labels, "oracle_k", edge_threshold)
    E_hat_oracle, _ = build_edge_sets(selected_idx_oracle, dir_pred, labels)
    f1_oracle, prec_oracle, rec_oracle, tp_oracle, fp_oracle, fn_oracle = compute_directed_f1(E_hat_oracle, E_star)
    shd_oracle, norm_shd_oracle, _ = compute_pair_state_shd(selected_idx_oracle, dir_pred, labels)
    
    # Debug stats
    p_edge_stats = {
        "min": p_edge.min().item(),
        "mean": p_edge.mean().item(),
        "max": p_edge.max().item()
    }
    
    return {
        # Threshold-independent metrics (legacy AUC)
        "auroc": auroc.item() if hasattr(auroc, 'item') else auroc,
        "auprc": auprc.item() if hasattr(auprc, 'item') else auprc,
        
        # ===== THRESHOLD-BASED METRICS =====
        "f1_threshold": f1_thresh,
        "precision_threshold": prec_thresh,
        "recall_threshold": rec_thresh,
        "tp_threshold": tp_thresh,
        "fp_threshold": fp_thresh,
        "fn_threshold": fn_thresh,
        "shd_threshold": shd_thresh,
        "normalized_shd_threshold": norm_shd_thresh,
        
        # ===== ORACLE-K BASED METRICS =====
        "f1_oracle_k": f1_oracle,
        "precision_oracle_k": prec_oracle,
        "recall_oracle_k": rec_oracle,
        "tp_oracle_k": tp_oracle,
        "fp_oracle_k": fp_oracle,
        "fn_oracle_k": fn_oracle,
        "shd_oracle_k": shd_oracle,
        "normalized_shd_oracle_k": norm_shd_oracle,
        
        # Counts for debugging
        "nnz_pred_threshold": len(E_hat_thresh),
        "nnz_pred_oracle_k": len(E_hat_oracle),
        "m_true": m_true,
        "edge_threshold": edge_threshold,
        
        # Debug stats
        "p_edge_stats": p_edge_stats
    }


# =============================================================================
# Legacy compatibility wrappers
# =============================================================================

def compute_directed_edge_set_f1(probs, labels, threshold=0.5, temperature=1.0):
    """Legacy wrapper for compute_all_edge_metrics - F1 portion (threshold mode)."""
    result = compute_all_edge_metrics(probs, labels, temperature, threshold)
    return result["f1_threshold"], result["precision_threshold"], result["recall_threshold"], result["nnz_pred_threshold"]


def compute_directed_edge_set_shd(probs, labels, threshold=0.5, temperature=1.0):
    """Legacy wrapper for compute_all_edge_metrics - SHD portion (threshold mode)."""
    result = compute_all_edge_metrics(probs, labels, temperature, threshold)
    return result["shd_threshold"], result["normalized_shd_threshold"]
