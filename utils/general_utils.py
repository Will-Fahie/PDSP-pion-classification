import numpy as np
import pickle
from sklearn.metrics import confusion_matrix


def load_file(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _count_in(seq, allowed):
    return sum(1 for x in seq if x in allowed)


def _confusion_count(y_pred, y_true, pred_set, true_set):
    if not isinstance(y_pred, list):
        y_pred = list(y_pred)
    if not isinstance(y_true, list):
        y_true = list(y_true)
    return sum(1 for p, t in zip(y_pred, y_true) if p in pred_set and t in true_set)


def _binomial_unc(k, n):
    p = k / n
    return np.sqrt(p * (1 - p) / n)


def purity(y_pred, y_true, pred_set, true_set, return_uncertainty=False):
    matched = _confusion_count(y_pred, y_true, pred_set, true_set)
    n_pred = _count_in(y_pred, pred_set)
    if n_pred == 0:
        return (0, 0) if return_uncertainty else 0
    val = matched / n_pred
    return (val, _binomial_unc(matched, n_pred)) if return_uncertainty else val


def efficiency(y_pred, y_true, pred_set, true_set, return_uncertainty=False):
    matched = _confusion_count(y_pred, y_true, pred_set, true_set)
    n_true = _count_in(y_true, true_set)
    if n_true == 0:
        return (0, 0) if return_uncertainty else 0
    val = matched / n_true
    return (val, _binomial_unc(matched, n_true)) if return_uncertainty else val


def create_confusion_matrix(y_true, y_pred):
    classes = sorted({*y_true, *y_pred})
    classes = [str(c) for c in classes]
    y_true = [str(x) for x in y_true]
    y_pred = [str(x) for x in y_pred]

    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cells = []
    for i, t in enumerate(classes):
        for j, p in enumerate(classes):
            pur, dp = purity(y_pred, y_true, [p], [t], return_uncertainty=True)
            eff, de = efficiency(y_pred, y_true, [p], [t], return_uncertainty=True)
            cells.append(
                f"{cm[i, j]}\n{100*pur:.1f} ± {100*dp:.1f}%\n{100*eff:.1f} ± {100*de:.1f}%"
            )

    info = np.asarray(cells).reshape(cm.shape)
    return cm[::-1], info[::-1], classes
