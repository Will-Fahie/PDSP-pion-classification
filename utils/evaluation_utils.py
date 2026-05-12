import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from general_utils import purity, efficiency, create_confusion_matrix


def plot_training_curves(history, title=''):
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
    plt.show()


def optimise_threshold(probs, labels, label='Model', color='steelblue'):
    """Pick threshold that maximises purity x efficiency. Plots the scan, prints metrics."""
    thresholds = np.arange(0.05, 0.96, 0.01)
    purs = np.array([purity((probs >= t).astype(int), labels, [1], [1]) for t in thresholds])
    effs = np.array([efficiency((probs >= t).astype(int), labels, [1], [1]) for t in thresholds])
    product = purs * effs
    best = thresholds[np.argmax(product)]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, 100 * purs, color=color, label='Purity')
    ax.plot(thresholds, 100 * effs, color=color, ls='--', label='Efficiency')
    ax.plot(thresholds, 100 * product, color='grey', ls=':', label='Purity × Efficiency')
    ax.axvline(best, color='red', ls=':', lw=1.5, label=f'Optimal threshold ({best:.2f})')
    ax.set_xlabel('Classification Threshold', fontsize=14)
    ax.set_ylabel('%', fontsize=14)
    ax.set_title(f'{label} - Threshold Optimisation', fontsize=15)
    ax.legend(fontsize=13)
    ax.tick_params(labelsize=12)
    ax.set_xlim([0.05, 0.95])
    ax.set_ylim([0, 100])
    plt.tight_layout()
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
    preds = (probs >= threshold).astype(int)
    tp = int(np.sum((preds == 1) & (labels == 1)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    safe = lambda a, b: a / b if b else 0.0
    return safe(tp, tp + fn), safe(fp, fp + tn), safe(tp, tp + fp)  # eff, fpr, pur


def plot_roc_and_purity_efficiency(results_list, title=''):
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
    plt.show()


def plot_confusion_matrix(labels, preds, threshold, title='π± Classification', figsize=(6, 5)):
    cm, info, raw_names = create_confusion_matrix(labels, preds)
    cm = cm[::-1]
    info = info[::-1]

    # create_confusion_matrix returns count/purity/efficiency per cell; swap last two
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
    ax.set_xticklabels([f'{names[j]}\n({col_totals[j]:,})' for j in range(len(names))],
                       rotation=30, ha='right', fontsize=10)
    ax.set_yticklabels([f'{names[i]}\n({row_totals[i]:,})' for i in range(len(names))],
                       rotation=30, ha='right', va='center', fontsize=10)
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('True', fontsize=11)
    ax.set_title(f'{title} (threshold={threshold:.2f})', fontsize=12)
    plt.tight_layout()
    plt.show()


def save_results(probs, labels, threshold, model_name, save_path):
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
