"""
Figures for event-level topology reconstruction.

Kept separate from evaluation_utils because those functions evaluate PFO
classifiers (ROC curves, thresholds, confusion matrices for a binary tag),
whereas these draw 4x4 topology matrices and pi0 selection scans.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from utils.general_utils import binomial_unc_pct
from utils.topology_utils import (TRUE_ORDER, RECO_ORDER, TRUE_LABELS, RECO_LABELS,
                                  DIAG_CELLS, get_matrix, best_row)


def _cell_text(n, row_total, col_total):
    """count / column-purity / row-efficiency, each with binomial uncertainty."""
    row_pct = 100 * n / row_total if row_total else 0.0
    col_pct = 100 * n / col_total if col_total else 0.0
    return (f'{n:,}\n{col_pct:.1f}±{binomial_unc_pct(n, col_total):.1f}%'
            f'\n{row_pct:.1f}±{binomial_unc_pct(n, row_total):.1f}%')


def _draw_matrix(ax, joint, mode, vmax, fs_cell, fs_axis,
                 show_ylabels=True, show_xlabel=True):
    matrix, colour, row_totals, col_totals = get_matrix(joint, mode=mode)
    im = ax.imshow(colour, cmap='cool', aspect='auto', vmin=0, vmax=vmax)

    for (i, j) in DIAG_CELLS:
        ax.add_patch(mpatches.Rectangle((j - 0.5, i - 0.5), 1, 1, linewidth=2.5,
                                        edgecolor='black', facecolor='none', zorder=3))

    for i in range(len(TRUE_ORDER)):
        for j in range(len(RECO_ORDER)):
            ax.text(j, i, _cell_text(matrix[i, j], row_totals[i], col_totals[j]),
                    ha='center', va='center', fontsize=fs_cell)

    ax.set_xticks(range(len(RECO_ORDER)))
    ax.set_xticklabels([f'{lbl}\n({col_totals[j]:,})' for j, lbl in enumerate(RECO_LABELS)],
                       rotation=30, ha='right', fontsize=fs_axis)
    if show_xlabel:
        ax.set_xlabel('Reconstructed topology', fontsize=fs_axis)

    ax.set_yticks(range(len(TRUE_ORDER)))
    if show_ylabels:
        ax.set_yticklabels([f'{lbl}\n({row_totals[i]:,})' for i, lbl in enumerate(TRUE_LABELS)],
                           fontsize=fs_axis)
        ax.set_ylabel('True topology', fontsize=fs_axis)
    else:
        ax.set_yticklabels([])
    return im


def plot_topology_grid(panels, out_path, ncols=2, mode='purity',
                       fs=(20, 16, 18, 20), figsize=None, dpi=300):
    """Grid of topology confusion matrices, one panel per selector.

    panels : list of (joint_counter, title, subtitle)
    Colour scale is shared across panels so they are directly comparable.
    """
    n = len(panels)
    nrows = int(np.ceil(n / ncols))
    fs_title, fs_cell, fs_axis, fs_cbar = fs
    figsize = figsize or (12 * ncols, 9 * nrows)

    vmax = max(get_matrix(j, mode=mode)[1].max() for j, *_ in panels)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi, squeeze=False)
    fig.subplots_adjust(right=0.85, top=0.93, hspace=0.45, wspace=0.15,
                        left=0.14, bottom=0.08)

    im = None
    for idx, ax in enumerate(axes.flat):
        if idx >= n:
            ax.axis('off')
            continue
        joint, title, subtitle = panels[idx]
        im = _draw_matrix(ax, joint, mode, vmax, fs_cell, fs_axis,
                          show_ylabels=(idx % ncols == 0),
                          show_xlabel=(idx >= n - ncols))
        ax.set_title(f'{title}\n{subtitle}', fontsize=fs_title, linespacing=1.6)

    cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.65])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=fs_cbar)
    label = 'Purity' if mode == 'purity' else 'Purity × Efficiency (%)'
    cbar.set_label(label, fontsize=fs_cbar, labelpad=-100 if mode == 'purity' else -140)
    cbar.ax.yaxis.set_label_position('left')

    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.show()
    print(f'Saved {out_path}')


def plot_topology_single(joint, title, subtitle, out_path, mode='pxe',
                         vmax=None, fs=(26, 21, 23, 26), dpi=300):
    """One topology matrix, sized for a slide."""
    fs_title, fs_cell, fs_axis, fs_cbar = fs
    if vmax is None:
        vmax = get_matrix(joint, mode=mode)[1].max()

    fig, ax = plt.subplots(figsize=(14, 10), dpi=dpi)
    fig.subplots_adjust(right=0.85, top=0.90, left=0.18, bottom=0.18)
    im = _draw_matrix(ax, joint, mode, vmax, fs_cell, fs_axis)
    ax.set_title(f'{title}\n{subtitle}', fontsize=fs_title, linespacing=1.6)

    cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.65])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=fs_cbar)
    label = 'Purity' if mode == 'purity' else 'Purity × Efficiency (%)'
    cbar.set_label(label, fontsize=fs_cbar, labelpad=-100 if mode == 'purity' else -140)
    cbar.ax.yaxis.set_label_position('left')

    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.show()
    print(f'Saved {out_path}')


def plot_pi0_scan(scans, baselines, out_path, pfo_threshold=None, dpi=130):
    """pi0 purity / efficiency / PxE against the photon classifier threshold.

    scans     : {pairing: rows}
    baselines : {pairing: metrics dict for the cut-based selection}
    Solid = 'best' pairing, dashed = 'exact'; horizontal dotted lines are the
    corresponding cut-based baselines.
    """
    styles = {"best": "-", "exact": "--"}
    fig, axL = plt.subplots(figsize=(10, 6), dpi=dpi)
    axR = axL.twinx()
    handles = []

    for pairing, rows in scans.items():
        ls = styles.get(pairing, "-")
        t   = np.array([r["threshold"] for r in rows])
        eff = np.array([r["efficiency"] for r in rows])
        pxe = np.array([r["pxe"] for r in rows])
        pur = np.array([r["purity"] for r in rows])

        h1, = axL.plot(t, eff, color='forestgreen', ls=ls, lw=2,
                       label=f'efficiency ({pairing})')
        h2, = axL.plot(t, pxe, color='crimson', ls=ls, lw=2,
                       label=f'purity × efficiency ({pairing})')
        h3, = axR.plot(t, pur, color='steelblue', ls=ls, lw=2,
                       label=f'purity ({pairing})')
        handles += [h1, h2, h3]

        b = best_row(rows)
        axL.scatter([b["threshold"]], [b["pxe"]], color='crimson', s=45, zorder=5)

    for pairing, base in baselines.items():
        ls = styles.get(pairing, "-")
        axL.axhline(base["efficiency"], color='forestgreen', ls=':', lw=1.2, alpha=0.7)
        axL.axhline(base["pxe"], color='crimson', ls=':', lw=1.2, alpha=0.7)
        axR.axhline(base["purity"], color='steelblue', ls=':', lw=1.2, alpha=0.7)

    if pfo_threshold is not None:
        axL.axvline(pfo_threshold, color='grey', ls='-.', lw=1.5)

    axL.set_xlabel('photon classifier threshold')
    axL.set_ylabel('pi0 efficiency / purity × efficiency [%]')
    axR.set_ylabel('pi0 purity [%]')
    axL.set_title('pi0 reconstruction vs photon threshold\n'
                  'dotted = cut-based baseline, dash-dot = PFO-level optimum')
    axL.set_xlim(0.05, 0.75)
    axL.legend(handles=handles, fontsize=8, loc='upper left',
               bbox_to_anchor=(1.08, 1), ncol=1)
    axL.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.show()
    print(f'Saved {out_path}')


def plot_table(rows, col_labels, out_path, highlight=None, dpi=150, figsize=(13, None)):
    """Render a small results table as a figure, styled like the model-comparison
    purity table. `highlight` is a row index (0-based, excluding the header)."""
    w, h = figsize
    h = h or 0.55 * (len(rows) + 1) + 0.6
    fig, ax = plt.subplots(figsize=(w, h), dpi=dpi)
    ax.axis('off')
    tbl = ax.table(cellText=rows, colLabels=col_labels, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(13)
    tbl.scale(1, 2.2)

    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor('#2c3e50')
        tbl[0, j].set_text_props(color='white', fontweight='bold')
    for i in range(1, len(rows) + 1):
        bg = '#f5f5f5' if i % 2 == 0 else 'white'
        for j in range(len(col_labels)):
            tbl[i, j].set_facecolor(bg)
    if highlight is not None:
        for j in range(len(col_labels)):
            tbl[highlight + 1, j].set_facecolor('#d6ecd6')

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.show()
    print(f'Saved {out_path}')