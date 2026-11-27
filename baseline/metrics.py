"""Single-class instance F1 at IoU 0.50 and 0.75, using Hungarian matching.

Per image, predicted and GT instances are matched one-to-one by the Hungarian
algorithm (maximum total IoU over pairs with IoU >= t). TP/FP/FN are pooled
over the whole split. All sizes are scored; GT ID 0 is ignored.
No confidence scores are used. This is NOT pixel-level F1 or Dice.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def instance_iou(gt, pred):
    gt, pred = np.asarray(gt), np.asarray(pred)
    if gt.shape != pred.shape or gt.ndim != 2:
        raise ValueError("Expected equally sized 2D GT/predicted instance maps")
    if np.any(gt < 0) or np.any(pred < 0):
        raise ValueError("IDs must be nonnegative; 0 is ignored in GT")
    valid = gt > 0
    gt_ids, gi = np.unique(gt[valid], return_inverse=True)
    pred_all, pi = np.unique(pred[valid], return_inverse=True)
    if not len(gt_ids):
        return gt_ids, np.array([], dtype=np.int64), np.zeros((0, 0))
    table = np.bincount(gi * len(pred_all) + pi,
                        minlength=len(gt_ids) * len(pred_all)).reshape(len(gt_ids), -1)
    gt_area = table.sum(axis=1, keepdims=True)
    positive = pred_all > 0
    overlap = table[:, positive]
    pred_area = overlap.sum(axis=0, keepdims=True)
    union = gt_area + pred_area - overlap
    return gt_ids, pred_all[positive], overlap / np.maximum(union, 1)


def hungarian_matches(iou, threshold):
    """Number of one-to-one matches with IoU >= threshold (maximum total IoU)."""
    if iou.size == 0:
        return 0
    weight = np.where(iou >= threshold, iou, 0.0)
    rows, cols = linear_sum_assignment(weight, maximize=True)
    return int((iou[rows, cols] >= threshold).sum())


def f1_from_counts(tp, n_pred, n_gt):
    fp, fn = n_pred - tp, n_gt - tp
    return {"precision": tp / n_pred if n_pred else None,
            "recall": tp / n_gt if n_gt else None,
            "F1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None}


class InstanceF1:
    def __init__(self, thresholds=(0.5, 0.75)):
        self.thresholds = tuple(thresholds)
        self.total_gt = 0
        self.total_predictions = 0
        self.tp = {t: 0 for t in self.thresholds}

    def update(self, gt, pred):
        _, ids, iou = instance_iou(gt, pred)
        self.total_gt += iou.shape[0]
        self.total_predictions += len(ids)
        for t in self.thresholds:
            self.tp[t] += hungarian_matches(iou, t)

    def compute(self):
        result = {}
        for t in self.thresholds:
            tag = int(round(t * 100))
            scores = f1_from_counts(self.tp[t], self.total_predictions, self.total_gt)
            result.update({f"F1_{tag}": scores["F1"], f"precision_{tag}": scores["precision"],
                           f"recall_{tag}": scores["recall"], f"TP_{tag}": self.tp[t]})
        result.update(gt_instances=self.total_gt, predicted_instances=self.total_predictions)
        return result
