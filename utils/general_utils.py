import numpy as np
import pickle
from sklearn.metrics import confusion_matrix
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # utils/general_utils.py -> repo root

SPECIES_COLORS = {
    r'$\pi^{\pm}$':       '#1f77b4',
    r'$\pi^{\pm}$:2nd':   '#4a90d9',
    r'$p$':               '#ff7f0e',
    r'$\gamma$':          '#1b9e77',
    r'$\gamma$:other':    '#8c564b',   # was '#3cb371' -- too close to signal gamma
    r'$\mu^{\pm}$':       '#9467bd',
    r'$e^{+}$':           '#d1495b',
    'other':              '#6b7280',
}

SPECIES_ALL = [r'$\pi^{\pm}$', r'$\pi^{\pm}$:2nd', '$p$', r'$\gamma$',
               r'$\gamma$:other', r'$\mu^{\pm}$', '$e^{+}$', 'other']

SELECTION_CUTS = {
    "pion": [
        ("chi²/ndof_p > 61.2",      lambda p: p['track_chi2/ndof_proton'] > 61.2),
        ("track length > 27.1",     lambda p: p['track_length'] > 27.1),
        ("track score > 0.5",       lambda p: p['track_score'] > 0.5),
        ("1.6 < dEdX median < 2.8", lambda p: 1.6 < p['dEdX_median'] < 2.8),
    ],
    "photon": [
        ("chi²/ndof_p > 61.2", lambda p: p['track_chi2/ndof_proton'] > 61.2),
        ("track score < 0.45", lambda p: p['track_score'] < 0.45),
        ("n_hits > 80",        lambda p: p['n_hits'] > 80),
        ("b < 20",             lambda p: p['b'] < 20),
        ("3 < d < 90",         lambda p: 3 < p['d'] < 90),
    ],
}

def species_label(t):
    """Split photons into beam-pi0 signal vs everything else."""
    if t['particle'] == r'$\gamma$':
        return r'$\gamma$' if t.get('is_gamma_from_beam_pi0') else r'$\gamma$:other'
    return t['particle']

def target_config(target):
    assert target in ("pion", "photon")
    suffix = f"_{target}"
    label_key = f"labels_{target}"
    if target == "pion":
        signal = lambda t: (t['particle'] == r'$\pi^{\pm}$')
        name, colour, bkg = r'$\pi^{\pm}$', 'steelblue', r'$\gamma$'
        ts_keeps = 'above'          # pions are high track score
    else:
        signal = lambda t: (t['particle'] == r'$\gamma$' and t['is_gamma_from_beam_pi0'])
        name, colour, bkg = r'$\gamma$ (beam $\pi^0$)', 'forestgreen', r'$\gamma$:other'
        ts_keeps = 'below'          # photons are low track score
    return dict(suffix=suffix, label_key=label_key, is_signal=signal,
                signal_name=name, signal_colour=colour, main_bkg=bkg,
                ts_cut_keeps=ts_keeps)

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

def binomial_unc(k, n):
    if n == 0:
        return 0.0
    p = k / n
    return np.sqrt(p * (1 - p) / n)


def binomial_unc_pct(k, n):
    """Binomial uncertainty on k/n, expressed in percent.

    Most call sites report purity and efficiency as percentages, so returning
    the fractional value there means every caller writes `100 * binomial_unc(...)`.
    """
    return 100 * binomial_unc(k, n)

def purity(y_pred, y_true, pred_set, true_set, return_uncertainty=False):
    matched = _confusion_count(y_pred, y_true, pred_set, true_set)
    n_pred = _count_in(y_pred, pred_set)
    if n_pred == 0:
        return (0, 0) if return_uncertainty else 0
    val = matched / n_pred
    return (val, binomial_unc(matched, n_pred)) if return_uncertainty else val


def efficiency(y_pred, y_true, pred_set, true_set, return_uncertainty=False):
    matched = _confusion_count(y_pred, y_true, pred_set, true_set)
    n_true = _count_in(y_true, true_set)
    if n_true == 0:
        return (0, 0) if return_uncertainty else 0
    val = matched / n_true
    return (val, binomial_unc(matched, n_true)) if return_uncertainty else val


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
