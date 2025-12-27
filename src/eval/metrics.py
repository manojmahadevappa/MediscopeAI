"""Evaluation metrics: accuracy, AUC, dice, concordance index (stubs)."""
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np


def accuracy(preds, targets):
    return accuracy_score(targets, preds)


def auc_score(prob, targets):
    try:
        return roc_auc_score(targets, prob)
    except Exception:
        return float('nan')


def dice_score(pred_mask, true_mask, eps=1e-6):
    p = (pred_mask > 0.5).astype(np.float32)
    t = (true_mask > 0.5).astype(np.float32)
    intersect = (p * t).sum()
    return (2. * intersect + eps) / (p.sum() + t.sum() + eps)


def concordance_index(event_times, predicted_scores, event_observed):
    # Placeholder: prefer lifelines.utils.concordance_index
    try:
        from lifelines.utils import concordance_index
        return concordance_index(event_times, -predicted_scores, event_observed)
    except Exception:
        return float('nan')
