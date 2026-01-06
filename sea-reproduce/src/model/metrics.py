"""
Correct metric computation for directed edge sets.

Implements the guide's edge-set based F1 formula (Equation 26):
- Separates edge existence from edge direction
- Builds explicit directed edge sets Ê and E*
- Computes set-based precision, recall, and F1
"""

import torch
import torch.nn.functional as F
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision


def compute_edge_existence_metrics(probs, labels, temperature=1.0):
    """
    Compute AUROC/AUPRC on edge existence (not direction).
    
    This separates "does an edge exist?" from "what is the direction?",
    which is the correct way to evaluate edge detection performance.
    
    Args:
        probs: [N, 3] tensor with logits [z_no_edge, z_forward, z_backward]
        labels: [N] tensor with values {0=no_edge, 1=forward, 2=backward}
        temperature: Temperature scaling factor (T>1 = less confident)
    
    Returns:
        auroc: AUROC score for edge existence
        auprc: AUPRC score for edge existence
    """
    # Apply temperature scaling and softmax
    probs_scaled = F.softmax(probs / temperature, dim=-1)
    
    # Edge existence probability: P(edge) = P(forward) + P(backward) = 1 - P(no_edge)
    p_edge = probs_scaled[:, 1] + probs_scaled[:, 2]
    
    # Binary labels: does any edge exist?
    t_edge = (labels > 0).float()
    
    # Compute metrics
    auroc_metric = BinaryAUROC()
    auprc_metric = BinaryAveragePrecision()
    
    auroc = auroc_metric(p_edge, t_edge)
    auprc = auprc_metric(p_edge, t_edge)
    
    return auroc, auprc


def compute_directed_edge_set_f1(probs, labels, threshold=0.5, temperature=1.0):
    """
    Compute F1 on directed edge sets (matches guide's Equation 26).
    
    This implements the correct edge-set based F1:
    1. Decide if edge exists: p_edge = P(forward) + P(backward) >= threshold
    2. If yes, pick direction: argmax(P(forward), P(backward))
    3. Build predicted set Ê and true set E*
    4. Compute F1 = 2|Ê∩E*| / (|Ê| + |E*|)
    
    Args:
        probs: [N, 3] tensor with logits [z_no_edge, z_forward, z_backward]
        labels: [N] tensor with values {0=no_edge, 1=forward, 2=backward}
        threshold: Threshold for edge existence decision (default=0.5)
        temperature: Temperature scaling factor
    
    Returns:
        f1: F1 score on directed edge sets
        precision: Precision = |Ê∩E*| / |Ê|
        recall: Recall = |Ê∩E*| / |E*|
        nnz: Number of predicted edges (|Ê|)
    """
    # Apply temperature scaling and softmax
    probs_scaled = F.softmax(probs / temperature, dim=-1)
    
    # Step 1: Compute edge existence scores
    p_edge = probs_scaled[:, 1] + probs_scaled[:, 2]
    
    # Step 2: Decide which edges exist
    edge_exists = p_edge >= threshold
    
    # Step 3: For existing edges, pick direction
    # direction: 0 = forward, 1 = backward
    direction = torch.argmax(probs_scaled[:, 1:], dim=1)
    
    # Step 4: Build predicted directed edge set Ê
    E_hat = set()
    for idx in range(len(probs)):
        if edge_exists[idx]:
            # Use tuple (index, direction_type) to represent directed edge
            if direction[idx] == 0:
                E_hat.add((idx.item(), 'forward'))
            else:
                E_hat.add((idx.item(), 'backward'))
    
    # Step 5: Build true directed edge set E*
    E_star = set()
    for idx in range(len(labels)):
        label = labels[idx].item()
        if label == 1:  # forward edge exists
            E_star.add((idx, 'forward'))
        elif label == 2:  # backward edge exists
            E_star.add((idx, 'backward'))
    
    # Step 6: Compute set-based metrics
    intersection = E_hat & E_star
    
    tp = len(intersection)
    fp = len(E_hat) - tp
    fn = len(E_star) - tp
    
    # Precision, Recall, F1
    precision = tp / len(E_hat) if len(E_hat) > 0 else 0.0
    recall = tp / len(E_star) if len(E_star) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    nnz = len(E_hat)  # Number of predicted edges
    
    return f1, precision, recall, nnz
