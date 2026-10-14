from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def top1_from_full_logits(logits, labels):
    has_label = labels.sum(axis=1) > 0
    if not has_label.any():
        return float("nan")
    predictions = logits.argmax(axis=1)
    correct = labels[np.arange(len(labels)), predictions] > 0
    return float(correct[has_label].mean())


def multilabel_metrics(logits, labels):
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -80, 80)))
    aucs, aps = [], []
    for c in range(labels.shape[1]):
        if labels[:, c].min() == labels[:, c].max():
            continue
        aucs.append(roc_auc_score(labels[:, c], probabilities[:, c]))
        aps.append(average_precision_score(labels[:, c], probabilities[:, c]))
    return {
        "macro_auroc": float(np.mean(aucs)) if aucs else float("nan"),
        "cmAP": float(np.mean(aps)) if aps else float("nan"),
        "top1": top1_from_full_logits(logits, labels),
        "n_valid_classes": len(aucs),
        "n_samples": int(len(labels)),
    }
