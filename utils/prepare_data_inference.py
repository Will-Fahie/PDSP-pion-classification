"""
Prepare the REAL DATA PFOs for inference with an MC-trained classifier.

This is the data counterpart of data-preparation.ipynb, stripped to what data
actually needs:
  * builds the input features (sequences, summary, masks) for every data PFO
  * does NOT build labels or a train/test split -- data has no truth, so it is
    only ever fed to a model, never trained on
  * normalises with the MC train_all statistics, loaded from
    prepared/prepared-mc/norm_stats.pkl (written by data-preparation.ipynb).
    Using the MC stats is essential: the model was trained on MC-scaled
    inputs, so the data must be put on the same scale.

Output: prepared/prepared-data/data_inference.pkl
  { 'sequences': (N,222,2) float32,   # z-scored, padded positions zeroed
    'summary':   (N,F)     float32,   # z-scored; F and the column order are recorded in norm_stats.pkl
    'masks':     (N,222)   float32,   # 1 = real hit, 0 = padding
    'event_number': (N,) int,         # to map predictions back to events
    'PFO_ID':       (N,) int }        # per-PFO id from add_derived_fields
"""

import pickle
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # utils/prepare_data_inference.py -> repo root

DATA_IN = PROJECT_ROOT / "extracted" / "extracted-data" / "data_new.pkl"
OUT_DIR  = PROJECT_ROOT / "prepared" / "prepared-data"
OUT_PATH = OUT_DIR / "data_inference.pkl"
NORM_STATS_PATH = PROJECT_ROOT / "prepared" / "prepared-mc" / "norm_stats.pkl"

MAX_LEN = 222

with open(NORM_STATS_PATH, "rb") as f:
    _stats = pickle.load(f)
s_mean, s_std   = _stats["s_mean"], _stats["s_std"]
sm_mean, sm_std = _stats["sm_mean"], _stats["sm_std"]

# Taken from the stats file, not repeated here: MC and data must build the
# summary array from the same columns in the same order, or the model sees
# data on a scale it was never trained on. Falls back to the original four for
# stats files written before this change.
SUMMARY_FEATURES = _stats.get("summary_features", [
    'track_chi2/ndof_proton', 'track_length', 'track_score', 'dEdX_median',
])
print(f"summary features from norm_stats: {SUMMARY_FEATURES}")


def keep(pfo):
    """Same quality cut used for the MC training set: drop zero-hit PFOs and any
    with a NaN dEdX_median. The model was never trained on these, so they can't
    be meaningfully classified."""
    if pfo['sequence_length'] == 0:
        return False
    m = pfo.get('dEdX_median')
    return not (isinstance(m, float) and np.isnan(m))


def main():
    with open(DATA_IN, "rb") as f:
        raw = pickle.load(f)
    raw = raw.to_dict("records") if hasattr(raw, "to_dict") else list(raw)
    print(f"Loaded {len(raw):,} data PFOs")

    n0 = len(raw)
    raw = [p for p in raw if keep(p)]
    print(f"Kept {len(raw):,} after dropping zero-hit/NaN PFOs ({n0 - len(raw):,} removed)")

    sequences = np.stack(
        [np.stack([d['dEdX_sequence'], d['residual_range_sequence']], axis=-1) for d in raw],
        axis=0,
    ).astype(np.float32)

    missing = [k for k in SUMMARY_FEATURES if k not in raw[0]]
    if missing:
        raise SystemExit(
            f"data PFOs are missing summary features {missing}.\n"
            "Re-run extract_data.py and add_derived_fields.py on the data sample."
        )

    summary = np.array(
        [[float(d[k]) for k in SUMMARY_FEATURES] for d in raw],
        dtype=np.float32,
    )

    masks = np.zeros((len(raw), MAX_LEN), dtype=np.float32)
    for i, d in enumerate(raw):
        masks[i, :min(d['sequence_length'], MAX_LEN)] = 1.0

    seq_n  = np.where(masks[:, :, None], (sequences - s_mean) / s_std, 0.0).astype(np.float32)
    summ_n = ((summary - sm_mean) / sm_std).astype(np.float32)
    summ_n = np.nan_to_num(summ_n, nan=0.0, posinf=0.0, neginf=0.0)   # as in data-preparation

    real = masks.astype(bool)
    print(f"sequences {seq_n.shape}, summary {summ_n.shape}, masks {masks.shape}")
    print(f"  norm seq (real hits)  mean={seq_n[real].mean(0)}  std={seq_n[real].std(0)}")
    print(f"  norm summ             mean={summ_n.mean(0)}  std={summ_n.std(0)}")

    event_number = np.array([d['event_number'] for d in raw])
    pfo_id       = np.array([d['PFO_ID'] for d in raw])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "wb") as f:
        pickle.dump({
            "sequences": seq_n,
            "summary": summ_n,
            "masks": masks,
            "event_number": event_number,
            "PFO_ID": pfo_id,
        }, f)
    print(f"Saved {seq_n.shape[0]:,} data PFOs to {OUT_PATH}")


if __name__ == "__main__":
    main()