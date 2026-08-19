"""Histogram helpers: raw-count distributions with optional lower panels.

Two functions:
  stacked_by_species -- stacked raw-count histograms, for composition
  hist_ratio_grid    -- overlaid raw-count histograms with a lower panel,
                        either a signal fraction or a plain ratio
"""

import numpy as np
import matplotlib.pyplot as plt


def _bins_for(v, ranges, n, nbins):
    lo, hi = ranges.get(n, (np.nanpercentile(v, 0.5), np.nanpercentile(v, 99.5)))
    return np.linspace(lo, hi, nbins)


def stacked_by_species(data, species, show, colours, ranges, ncols=3, nbins=50,
                       figsize_per=(6, 4.5), dpi=130, logy=False,
                       suptitle=None, out_path=None):
    """Stacked raw-count histograms broken down by species.

    data    : {variable name: values array}
    species : (n_pfo,) array of species labels
    show    : species to include, in stacking order
    colours : {species: colour}
    ranges  : {variable name: (lo, hi)}. Missing entries auto-scale to the
              0.5-99.5 percentile range.
    """
    names = list(data)
    nrows = int(np.ceil(len(names) / ncols))
    w, h = figsize_per
    fig, ax = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows), dpi=dpi)
    ax = np.atleast_1d(ax).ravel()
    counts = {sp: int((species == sp).sum()) for sp in show}

    for j, n in enumerate(names):
        v = data[n]
        bins = _bins_for(v, ranges, n, nbins)
        arrs = [v[(species == sp) & np.isfinite(v)] for sp in show]
        ax[j].hist(arrs, bins=bins, stacked=True,
                   color=[colours[sp] for sp in show],
                   label=[f"{sp} (N={counts[sp]:,})" for sp in show],
                   edgecolor='none')
        if logy:
            ax[j].set_yscale('log')
        ax[j].set_xlabel(n)
        ax[j].set_ylabel('PFOs per bin')
        ax[j].grid(alpha=0.25)

    for j in range(len(names), len(ax)):
        ax[j].axis('off')
    ax[0].legend(fontsize=8)

    if suptitle:
        fig.suptitle(suptitle, fontsize=13, y=1.00)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.show()
    if out_path:
        print(f"Saved {out_path}")


def hist_ratio_grid(data, groups, ranges, ncols=3, nbins=50,
                    ratio_pair=None, ratio_mode='fraction',
                    figsize_per=(6, 4.5), dpi=130,
                    ylabel='PFOs per bin', logy=False, suptitle=None,
                    out_path=None):
    """Overlaid raw-count histograms with a lower panel per variable.

    groups     : list of (label, mask, colour), one histogram each
    ratio_pair : (numerator_label, denominator_label)
    ratio_mode : 'fraction' -> num / (num + den), the per-bin signal fraction.
                   Use for signal vs background: it reads as purity and spans
                   0-1. A plain ratio is uninformative when the two samples
                   differ greatly in size, since every bin sits near the
                   overall ratio.
                 'ratio'    -> num / den. Use for data vs MC, where 1 is the
                   expectation.

    Each variable gets its own nested gridspec so the main panel and its lower
    panel are tight against each other while variables stay well separated.
    """
    names = list(data)
    nrows = int(np.ceil(len(names) / ncols))
    w, h = figsize_per

    fig = plt.figure(figsize=(w * ncols, (h + 1.4) * nrows), dpi=dpi)
    outer = fig.add_gridspec(nrows, ncols, hspace=0.42, wspace=0.30)

    for j, n in enumerate(names):
        r, c = divmod(j, ncols)
        inner = outer[r, c].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
        ax = fig.add_subplot(inner[0])
        axr = fig.add_subplot(inner[1], sharex=ax)

        v_all = data[n]
        bins = _bins_for(v_all, ranges, n, nbins)
        centres = 0.5 * (bins[:-1] + bins[1:])

        counts = {}
        for label, mask, colour in groups:
            v = v_all[mask & np.isfinite(v_all)]
            counts[label], _ = np.histogram(v, bins=bins)
            ax.step(centres, counts[label], where='mid', lw=1.8, color=colour,
                    label=f"{label} (N={len(v):,})")

        if logy:
            ax.set_yscale('log')
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(labelbottom=False)
        ax.grid(alpha=0.25)

        num, den = counts[ratio_pair[0]], counts[ratio_pair[1]]
        if ratio_mode == 'fraction':
            tot = num + den
            y = np.divide(num, tot, out=np.full(len(num), np.nan, float), where=tot > 0)
            err = np.sqrt(np.divide(y * (1 - y), np.maximum(tot, 1), where=tot > 0))
            axr.axhline(np.nansum(num) / max(np.nansum(tot), 1),
                        color='grey', lw=1, ls='--')
            axr.set_ylim(0, 1)
            axr.set_ylabel('signal\nfraction', fontsize=8)
        else:
            y = np.divide(num, den, out=np.full(len(num), np.nan, float), where=den > 0)
            err = y * np.sqrt(np.divide(1.0, np.maximum(num, 1), where=num > 0) +
                             np.divide(1.0, np.maximum(den, 1), where=den > 0))
            axr.axhline(1.0, color='grey', lw=1, ls='--')
            axr.set_ylim(0, 2)
            axr.set_ylabel('ratio', fontsize=8)

        axr.errorbar(centres, y, yerr=err, fmt='k.', ms=4, lw=0.8)
        axr.set_xlabel(n, fontsize=10)
        axr.grid(alpha=0.25)

        if j == 0:
            ax.legend(fontsize=8)

    if suptitle:
        fig.suptitle(suptitle, fontsize=13, y=0.995)
    if out_path:
        plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.show()
    if out_path:
        print(f"Saved {out_path}")