"""Evaluation and plotting helpers for the PFO classifiers.

Every plotting function takes an optional `save_path`. Omit it and the figure
is only displayed; give it a path and the figure is written there as well.
Figures are saved as whatever the extension says -- the project uses PDF.
"""

import os
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from utils.general_utils import (purity, efficiency, create_confusion_matrix,
                                 binomial_unc_pct)


def _save(fig_path):
    """Save the current figure if a path was given.

    Kept as a helper so every plotting function handles the optional path
    identically. No dpi: the project saves vector PDFs, where it has no effect
    except on rasterised elements.
    """
    if fig_path is None:
        return
    fig_path = Path(fig_path)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, bbox_inches='tight')
    print(f"  saved {fig_path}")


# ---------------------------------------------------------------------------
# Training diagnostics
# ---------------------------------------------------------------------------

def plot_training_curves(history, title='', save_path=None):
    """Loss, purity and efficiency against epoch.

    Purity and efficiency are evaluated at a fixed threshold of 0.5, not at the
    optimised operating point, so they are a rough guide only. The quantity
    training actually minimises is the validation loss in the left panel.
    """
    prefix = f'{title} - ' if title else ''
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].set_title(f'{prefix}Loss')
    axes[0].grid(True, alpha=0.3)

    for ax, key, name in [(axes[1], 'purity', 'Purity'),
                          (axes[2], 'efficiency', 'Efficiency')]:
        ax.plot([100 * v for v in history[key]])
        ax.set_xlabel('Epoch')
        ax.set_ylabel('%')
        ax.set_title(f'{prefix}{name} (threshold=0.5)')
        ax.set_ylim([0, 100])
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(save_path)
    plt.show()


# ---------------------------------------------------------------------------
# Operating point
# ---------------------------------------------------------------------------

def optimise_threshold(probs, labels, label='Model', color='steelblue',
                       save_path=None):
    """Pick the threshold maximising purity x efficiency.

    Plots the scan, prints the metrics at the chosen threshold, returns it.
    """
    thresholds = np.arange(0.05, 0.96, 0.01)
    purs = np.array([purity((probs >= t).astype(int), labels, [1], [1])
                     for t in thresholds])
    effs = np.array([efficiency((probs >= t).astype(int), labels, [1], [1])
                     for t in thresholds])
    product = purs * effs
    best = thresholds[np.argmax(product)]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, 100 * purs, color=color, label='Purity')
    ax.plot(thresholds, 100 * effs, color=color, ls='--', label='Efficiency')
    ax.plot(thresholds, 100 * product, color='grey', ls=':',
            label='Purity × Efficiency')
    ax.axvline(best, color='red', ls=':', lw=1.5,
               label=f'Optimal threshold ({best:.2f})')
    ax.set_xlabel('Classification Threshold', fontsize=14)
    ax.set_ylabel('%', fontsize=14)
    ax.set_title(f'{label} - Threshold Optimisation', fontsize=15)
    ax.legend(fontsize=13)
    ax.tick_params(labelsize=12)
    ax.set_xlim([0.05, 0.95])
    ax.set_ylim([0, 100])
    plt.tight_layout()
    _save(save_path)
    plt.show()

    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    preds = (probs >= best).astype(int)
    pur, dpur = purity(preds, labels, [1], [1], return_uncertainty=True)
    eff, deff = efficiency(preds, labels, [1], [1], return_uncertainty=True)

    print(f"=== {label} (threshold={best:.2f}) ===")
    print(f"  AUC:        {roc_auc:.3f}")
    print(f"  Purity:     {100*pur:.1f}% ± {100*dpur:.1f}%")
    print(f"  Efficiency: {100*eff:.1f}% ± {100*deff:.1f}%")
    print(f"  Product:    {100*pur*eff:.1f}%")

    return best


def _op_point(probs, labels, threshold):
    """(efficiency, false positive rate, purity) at one threshold."""
    preds = (probs >= threshold).astype(int)
    tp = int(np.sum((preds == 1) & (labels == 1)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    safe = lambda a, b: a / b if b else 0.0
    return safe(tp, tp + fn), safe(fp, fp + tn), safe(tp, tp + fp)


# ---------------------------------------------------------------------------
# ROC and purity-efficiency
# ---------------------------------------------------------------------------

def plot_roc_and_purity_efficiency(results_list, title='', save_path=None):
    """ROC on the left, purity against efficiency on the right.

    results_list : list of dicts with keys probs, labels, threshold, color, label.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for r in results_list:
        probs, labels = r['probs'], r['labels']
        color, name, t_op = r['color'], r['label'], r['threshold']

        fpr, tpr, _ = roc_curve(labels, probs)
        axes[0].plot(fpr, tpr, color=color, lw=2,
                     label=f"{name} (AUC = {auc(fpr, tpr):.3f})")

        ts = np.linspace(0.01, 0.99, 200)
        pe = [_op_point(probs, labels, t) for t in ts]
        effs, _, purs = zip(*pe)
        axes[1].plot(effs, purs, color=color, lw=2, label=name)

        eff_op, fpr_op, pur_op = _op_point(probs, labels, t_op)
        axes[0].scatter(fpr_op, eff_op, color=color, marker='o', s=80, zorder=5,
                        label=f'  operating point (t={t_op:.2f})')
        axes[1].scatter(eff_op, pur_op, color=color, marker='o', s=80, zorder=5,
                        label=f'  operating point (t={t_op:.2f})')

    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.4, lw=1)
    axes[0].set_xlabel('False Positive Rate', fontsize=12)
    axes[0].set_ylabel('True Positive Rate (Efficiency)', fontsize=12)
    axes[0].set_title('ROC Curve', fontsize=12)
    axes[0].legend(fontsize=9)
    axes[0].set_xlim([0, 1]); axes[0].set_ylim([0, 1])

    axes[1].set_xlabel('Efficiency (Recall)', fontsize=12)
    axes[1].set_ylabel('Purity (Precision)', fontsize=12)
    axes[1].set_title('Purity vs Efficiency', fontsize=12)
    axes[1].legend(fontsize=9)
    axes[1].set_xlim([0, 1]); axes[1].set_ylim([0, 1])

    if title:
        plt.suptitle(title, fontsize=13, y=1.01)
    plt.tight_layout()
    _save(save_path)
    plt.show()


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(labels, preds, threshold, title='π± Classification',
                          figsize=(6, 5), save_path=None):
    """Binary confusion matrix with count, purity and efficiency per cell.

    create_confusion_matrix returns rows reversed relative to the class list it
    also returns, so the reversal below puts them back in class order and the
    axis labels line up. Verified against a known case.
    """
    cm, info, raw_names = create_confusion_matrix(labels, preds)
    cm = cm[::-1]
    info = info[::-1]

    # cells come as count/purity/efficiency; swap the last two
    def _swap(s):
        a, b, c = s.split('\n')
        return f'{a}\n{c}\n{b}'
    info = np.vectorize(_swap)(info)

    rename = {'0': 'not pion', '1': 'pion'}
    names = [rename.get(n, n) for n in raw_names]
    col_totals = cm.sum(axis=0)
    row_totals = cm.sum(axis=1)

    fig, ax = plt.subplots(figsize=figsize)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       fill=False, edgecolor='black', lw=1))
            ax.text(j, i, info[i][j], ha='center', va='center', fontsize=9)

    ax.set_xlim(-0.5, cm.shape[1] - 0.5)
    ax.set_ylim(-0.5, cm.shape[0] - 0.5)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels([f'{names[j]}\n({col_totals[j]:,})'
                        for j in range(len(names))],
                       rotation=30, ha='right', fontsize=10)
    ax.set_yticklabels([f'{names[i]}\n({row_totals[i]:,})'
                        for i in range(len(names))],
                       rotation=30, ha='right', va='center', fontsize=10)
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('True', fontsize=11)
    ax.set_title(f'{title} (threshold={threshold:.2f})', fontsize=12)
    plt.tight_layout()
    _save(save_path)
    plt.show()


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_results(probs, labels, threshold, model_name, save_path):
    """Write probs, labels, threshold and the headline metrics to a pickle.

    This is what every analysis notebook reads, so the keys here are the
    contract between the classifiers and everything downstream.
    """
    preds = (probs >= threshold).astype(int)
    pur, _ = purity(preds, labels, [1], [1], return_uncertainty=True)
    eff, _ = efficiency(preds, labels, [1], [1], return_uncertainty=True)
    fpr, tpr, _ = roc_curve(labels, probs)

    out = {
        "model_name": model_name,
        "probs": probs,
        "labels": labels,
        "threshold": threshold,
        "purity": pur,
        "efficiency": eff,
        "auc": auc(fpr, tpr),
    }

    d = os.path.dirname(save_path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(out, f)
    print(f"Saved to {save_path}")
    return out


# ---------------------------------------------------------------------------
# Cut flow
# ---------------------------------------------------------------------------

def cut_table(pfos, cut_list, is_signal, label='', print_table=True):
    """Cut-flow table: purity and efficiency after each cut is added.

    Parameters
    ----------
    pfos : list of PFO dicts
    cut_list : list of (label, fn) where fn(pfo) -> bool, applied cumulatively
               in order. This is the format of general_utils.SELECTION_CUTS.
    is_signal : fn(pfo) -> bool, the truth definition of signal.
    label : name for the table header, e.g. 'pi+' or 'gamma'.

    Returns
    -------
    list of dicts, one per stage (stage 0 = no cuts), each with
    cut, purity, d_purity, efficiency, d_efficiency, selected, TP, FP, FN, TN.
    Returning the rows means callers can plot or save the cut flow rather than
    only read it off the terminal.
    """
    rows = []
    for n in range(len(cut_list) + 1):
        TP = FP = FN = TN = 0
        for pfo in pfos:
            sig = is_signal(pfo)
            sel = all(fn(pfo) for _, fn in cut_list[:n]) if n > 0 else True
            if   sig and     sel: TP += 1
            elif sig and not sel: FN += 1
            elif not sig and sel: FP += 1
            else:                 TN += 1
        rows.append({
            "cut": "(no cuts)" if n == 0 else cut_list[n - 1][0],
            "purity":       100 * TP / (TP + FP) if (TP + FP) else 0.0,
            "d_purity":     binomial_unc_pct(TP, TP + FP),
            "efficiency":   100 * TP / (TP + FN) if (TP + FN) else 0.0,
            "d_efficiency": binomial_unc_pct(TP, TP + FN),
            "selected":     TP + FP,
            "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        })

    if print_table:
        print(f"\n  {label} cuts")
        print(f"  {'Cut added':<28} {'Purity':>16}  {'Efficiency':>16}  {'Selected':>10}")
        print(f"  {'-'*78}")
        for r in rows:
            print(f"  {r['cut']:<28} {r['purity']:.1f} ± {r['d_purity']:.1f}% "
                  f"   {r['efficiency']:.1f} ± {r['d_efficiency']:.1f}%    "
                  f"{r['selected']:>8,}")
        last = rows[-1]
        print(f"  {'-'*78}")
        print(f"  Purity × Efficiency = {last['purity']:.1f}% × "
              f"{last['efficiency']:.1f}% = "
              f"{last['purity']*last['efficiency']/100:.2f}%")

    return rows


# ---------------------------------------------------------------------------
# Combining figures
# ---------------------------------------------------------------------------

def combine_pdfs(plots_dir, out_name="all_plots.pdf", pattern="*.pdf",
                 order=None, strict=False):
    """Merge PDFs in `plots_dir` into a single file.

    order  : optional list of filename stems. Anything listed comes first, in
             that order; everything else follows alphabetically.
    strict : if True, merge ONLY the files named in `order`. Use this when the
             directory may hold figures from an earlier run with a different
             TARGET, which would otherwise be pulled in silently.

    Requires pypdf:  pip install --user pypdf
    """
    from pypdf import PdfWriter

    plots_dir = Path(plots_dir)
    out_path = plots_dir / out_name

    files = sorted(p for p in plots_dir.glob(pattern) if p.name != out_name)

    if order and strict:
        by_stem = {p.stem: p for p in files}
        missing = [s for s in order if s not in by_stem]
        files = [by_stem[s] for s in order if s in by_stem]
        if missing:
            print(f"not found (skipped): {missing}")
    elif order:
        rank = {name: i for i, name in enumerate(order)}
        files.sort(key=lambda p: (rank.get(p.stem, len(order)), p.stem))

    if not files:
        print(f"no PDFs found in {plots_dir}")
        return None

    writer = PdfWriter()
    for f in files:
        writer.append(str(f))
    with open(out_path, "wb") as fh:
        writer.write(fh)

    print(f"merged {len(files)} figures -> {out_path}")
    for f in files:
        print(f"    {f.name}")
    return out_path