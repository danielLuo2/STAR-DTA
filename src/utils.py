"""Seeding and evaluation metrics."""

import json
import random

import numpy as np
from scipy import stats


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rmse(y_true, y_pred):
    return float(np.sqrt(((y_true - y_pred) ** 2).mean(axis=0)))


def mse(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean(axis=0)


def pearson(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0, 1]


def spearman(y_true, y_pred):
    return stats.spearmanr(y_true, y_pred)[0]


def ci(y_true, y_pred, chunk=2048):
    """Concordance index as in DeepDTA / GraphDTA.

    Over all pairs with y_i > y_j: 1 if f_i > f_j, 0.5 if f_i == f_j, else 0. Pairs with tied labels
    are skipped. Computed block-wise; equal to the reference double loop.
    """
    y = np.asarray(y_true, dtype=np.float64).ravel()
    f = np.asarray(y_pred, dtype=np.float64).ravel()
    z = 0
    s2 = 0  # twice the concordant score, kept as an integer
    for a in range(0, len(y), chunk):
        yi, fi = y[a:a + chunk, None], f[a:a + chunk, None]
        pair = yi > y[None, :]
        z += int(pair.sum())
        s2 += 2 * int((pair & (fi > f[None, :])).sum()) + int((pair & (fi == f[None, :])).sum())
    return s2 / 2 / z if z > 0 else 0.0


def load_region_labels(path):
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
