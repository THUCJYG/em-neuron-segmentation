"""Targets are generated AFTER spatial augmentation; channel 0=right, 1=down."""
import numpy as np
import torch
import torch.nn.functional as F


def make_affinities(labels, distance=1):
    """Compare endpoints at the given rightward/downward pixel distance.

    Label 0 means unannotated/ignored. An edge is supervised only when BOTH
    endpoints have positive labels. The last 'distance' columns/rows
    have no endpoint. Different positive endpoint IDs are negatives.
    """
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError("Expected a 2D instance-ID image")
    if distance < 1:
        raise ValueError("distance must be positive")
    target = np.zeros((2, *labels.shape), dtype=np.float32)
    valid = np.zeros_like(target, dtype=bool)
    for channel, (start, end) in enumerate((
        (np.s_[:, :-distance], np.s_[:, distance:]),
        (np.s_[:-distance, :], np.s_[distance:, :]),
    )):
        valid[channel][start] = (labels[start] > 0) & (labels[end] > 0)
        target[channel][start] = labels[start] == labels[end]
    target[~valid] = 0
    return target, valid


def affinity_loss(logits, target, valid, kind="balanced_bce"):
    """Baseline: equal total weights for positive and negative valid edges."""
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    valid = valid.to(loss.dtype)
    if kind == "bce":
        return (loss * valid).sum() / valid.sum().clamp_min(1)
    if kind != "balanced_bce":
        raise ValueError(f"Unknown loss: {kind}")
    pos, neg = valid * target, valid * (1 - target)
    np_, nn_ = pos.sum(), neg.sum()
    active = (np_ > 0).to(loss.dtype) + (nn_ > 0).to(loss.dtype)
    return ((loss * pos).sum() / np_.clamp_min(1)
            + (loss * neg).sum() / nn_.clamp_min(1)) / active.clamp_min(1)


def boundary_dice_loss(logits, target, valid):
    prob = torch.sigmoid(-logits) * valid
    truth = (1 - target) * valid
    return 1 - (2 * (prob * truth).sum() + 1) / (prob.sum() + truth.sum() + 1)


def coarse_boundary_loss(logits, target, valid):
    """Predict whether any annotated right/down edge in a coarse cell is a boundary."""
    size = logits.shape[-2:]
    boundary = ((target < 0.5) & valid.bool()).any(dim=1, keepdim=True)
    observed = valid.bool().any(dim=1, keepdim=True)
    occupied = F.adaptive_max_pool2d(boundary.float(), size)
    mask = F.adaptive_max_pool2d(observed.float(), size)
    return affinity_loss(logits, occupied, mask)


def boundary_counts(probabilities, target, valid):
    """Boundary-edge counts (tp, fp, fn) over valid edges.

    The positive class for F1 is a BOUNDARY edge: target 0 (two different
    instances). A prediction p < 0.5 means "boundary". An always-one predictor
    therefore has F1 = 0, whatever the class imbalance.
    """
    valid = np.asarray(valid, dtype=bool)
    pred = (np.asarray(probabilities) < 0.5) & valid
    truth = (np.asarray(target) < 0.5) & valid
    tp = int((pred & truth).sum())
    return tp, int((pred & ~truth).sum()), int((~pred & truth).sum())


def f1_score(tp, fp, fn):
    """F1 = 2TP / (2TP + FP + FN); None when there are no boundaries at all."""
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else None
