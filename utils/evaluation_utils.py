import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from general_utils import purity, efficiency, create_confusion_matrix


def plot_training_curves(history, title=''):
    """Plot train/val loss, purity, and efficiency over training epochs.

    Args:
        history: dict with keys 'train_loss', 'val_loss', 'purity', 'efficiency'
        title:   optional prefix for subplot titles
    """
    prefix = f'{title} — ' if title else ''
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'],   label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].set_title(f'{prefix}Loss')

    axes[1].plot([100 * p for p in history['purity']])
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('%')
    axes[1].set_title(f'{prefix}Purity (threshold=0.5)')
    axes[1].set_ylim([0, 100])

    axes[2].plot([100 * e for e in history['efficiency']])
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('%')
    axes[2].set_title(f'{prefix}Efficiency (threshold=0.5)')
    axes[2].set_ylim([0, 100])

    plt.tight_layout()
    plt.show()


def optimise_threshold(probs, labels, label='Model', color='steelblue'):
    """Scan thresholds from 0.05 to 0.95, find the one maximising purity × efficiency.

    Plots threshold vs purity and efficiency with a vertical line at the optimum.
    Prints final metrics with uncertainties.

    Args:
        probs:  1-D array of predicted probabilities
        labels: 1-D array of true binary labels
        label:  model name used in the plot title and printed output
        color:  line colour for the purity and efficiency curves

    Returns:
        best_threshold (float)
    """
    thresholds = np.arange(0.05, 0.96, 0.01)
    purities_scan, effs_scan = [], []

    for t in thresholds:
        preds = (probs >= t).astype(int)
        purities_scan.append(purity(preds, labels, [1], [1]))
        effs_scan.append(efficiency(preds, labels, [1], [1]))

    purities_scan = np.array(purities_scan)
    effs_scan     = np.array(effs_scan)
    product       = purities_scan * effs_scan

    best_idx       = np.argmax(product)
    best_threshold = thresholds[best_idx]

    # Plot threshold vs purity and efficiency
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, 100 * purities_scan, color=color,        label='Purity')
    ax.plot(thresholds, 100 * effs_scan,     color=color, ls='--', label='Efficiency')
    ax.plot(thresholds, 100 * product,       color='grey',  ls=':', label='Purity \u00d7 Efficiency')
    ax.axvline(best_threshold, color='red', ls=':', lw=1.5,
               label=f'Optimal threshold ({best_threshold:.2f})')
    ax.set_xlabel('Classification Threshold', fontsize=12)
    ax.set_ylabel('%', fontsize=12)
    ax.set_title(f'{label} \u2014 Threshold Optimisation', fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim([0.05, 0.95])
    ax.set_ylim([0, 100])
    plt.tight_layout()
    plt.show()

    # Compute and print final metrics
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    final_preds = (probs >= best_threshold).astype(int)
    pur_val, pur_unc = purity(final_preds, labels, [1], [1], return_uncertainty=True)
    eff_val, eff_unc = efficiency(final_preds, labels, [1], [1], return_uncertainty=True)

    print(f"=== {label} (threshold={best_threshold:.2f}) ===")
    print(f"  AUC:        {roc_auc:.3f}")
    print(f"  Purity:     {100 * pur_val:.1f}% \u00b1 {100 * pur_unc:.1f}%")
    print(f"  Efficiency: {100 * eff_val:.1f}% \u00b1 {100 * eff_unc:.1f}%")
    print(f"  Product:    {100 * pur_val * eff_val:.1f}%")

    return best_threshold


def plot_roc_and_purity_efficiency(results_list, title=''):
    """Plot ROC curve and purity-vs-efficiency curve for one or more models.

    Args:
        results_list: list of dicts, each with keys:
                      'probs', 'labels', 'threshold', 'color', 'label'
        title:        optional overall figure suptitle
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for r in results_list:
        probs_r   = r['probs']
        labels_r  = r['labels']
        threshold = r['threshold']
        color     = r['color']
        label_r   = r['label']

        # ROC curve
        fpr, tpr, _ = roc_curve(labels_r, probs_r)
        roc_auc = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, color=color, lw=2,
                     label=f"{label_r} (AUC = {roc_auc:.3f})")

        # Purity vs Efficiency curve (dense threshold scan)
        thresholds = np.linspace(0.01, 0.99, 200)
        purities, efficiencies = [], []
        for t in thresholds:
            preds = (probs_r >= t).astype(int)
            tp = np.sum((preds == 1) & (labels_r == 1))
            fp = np.sum((preds == 1) & (labels_r == 0))
            fn = np.sum((preds == 0) & (labels_r == 1))
            p  = tp / (tp + fp) if (tp + fp) > 0 else 0
            e  = tp / (tp + fn) if (tp + fn) > 0 else 0
            purities.append(p)
            efficiencies.append(e)
        axes[1].plot(efficiencies, purities, color=color, lw=2, label=label_r)

        # Operating point
        preds = (probs_r >= threshold).astype(int)
        tp = np.sum((preds == 1) & (labels_r == 1))
        fp = np.sum((preds == 1) & (labels_r == 0))
        fn = np.sum((preds == 0) & (labels_r == 1))
        tn = np.sum((preds == 0) & (labels_r == 0))
        tpr_op = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr_op = fp / (fp + tn) if (fp + tn) > 0 else 0
        pur_op = tp / (tp + fp) if (tp + fp) > 0 else 0
        axes[0].scatter(fpr_op, tpr_op, color=color, marker='o', s=80, zorder=5,
                        label=f'  operating point (t={threshold:.2f})')
        axes[1].scatter(tpr_op, pur_op, color=color, marker='o', s=80, zorder=5,
                        label=f'  operating point (t={threshold:.2f})')

    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.4, lw=1)
    axes[0].set_xlabel('False Positive Rate', fontsize=12)
    axes[0].set_ylabel('True Positive Rate (Efficiency)', fontsize=12)
    axes[0].set_title('ROC Curve', fontsize=12)
    axes[0].legend(fontsize=9)
    axes[0].set_xlim([0, 1])
    axes[0].set_ylim([0, 1])

    axes[1].set_xlabel('Efficiency (Recall)', fontsize=12)
    axes[1].set_ylabel('Purity (Precision)', fontsize=12)
    axes[1].set_title('Purity vs Efficiency', fontsize=12)
    axes[1].legend(fontsize=9)
    axes[1].set_xlim([0, 1])
    axes[1].set_ylim([0, 1])

    if title:
        plt.suptitle(title, fontsize=13, y=1.01)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(labels, preds, threshold, title='Pion Classification'):
    """Plot a labelled confusion matrix.

    Args:
        labels:    1-D array of true binary labels
        preds:     1-D array of predicted binary labels
        threshold: the classification threshold used (shown in the plot title)
        title:     base title for the plot
    """
    cm, info, label_names = create_confusion_matrix(labels, preds)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, info[i][j], ha='center', va='center', fontsize=9)
    ax.set_xticks(range(len(label_names)))
    ax.set_yticks(range(len(label_names)))
    ax.set_xticklabels(label_names)
    ax.set_yticklabels(label_names[::-1])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'{title} (threshold={threshold:.2f})')
    plt.colorbar(im)
    plt.tight_layout()
    plt.show()


def save_results(probs, labels, threshold, model_name, save_path):
    """Save evaluation results to a pickle file.

    Args:
        probs:      1-D array of predicted probabilities
        labels:     1-D array of true binary labels
        threshold:  optimal classification threshold
        model_name: descriptive name stored in the results dict
        save_path:  path to the output .pkl file

    Returns:
        results dict
    """
    final_preds = (probs >= threshold).astype(int)
    pur_val, _ = purity(final_preds, labels, [1], [1], return_uncertainty=True)
    eff_val, _ = efficiency(final_preds, labels, [1], [1], return_uncertainty=True)

    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    results = {
        "model_name": model_name,
        "probs":      probs,
        "labels":     labels,
        "threshold":  threshold,
        "purity":     pur_val,
        "efficiency": eff_val,
        "auc":        roc_auc,
    }

    dir_name = os.path.dirname(save_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved to {save_path}")
    return results
