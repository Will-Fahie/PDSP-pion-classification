#!/usr/bin/env python3
"""
Add derived per-PFO fields to the output of extract_data.py.

    extracted/extracted-mc/extracted_mc.pkl     -> mc_new.pkl
    extracted/extracted-data/extracted_data.pkl -> data_new.pkl

Fields added:
  PFO_ID       1-indexed integer per PFO
  b            impact parameter, cm
  d            photon travel distance, cm
  dEdX_median  median dEdX over real hits (zero-padding excluded)
  E_c          corrected shower energy, only if 'shower_energy' present

Energy correction (CORSIKA/LArSoft calibration):
  E_c = E / (1 + C),  C = P0*ln(E - P1) + P2

NOTE (2026-07): an earlier attempt converted each '*_sequence' field to a
float32 numpy array to "save memory". On this data that was a mistake -- the
sequences are stored efficiently (shared backing buffer), so pickle keeps the
input small (~1.2 GB); forcing a separate dense array per PFO destroyed that
sharing and exploded the output to ~175 GB. This version leaves the sequences
exactly as they are and only adds the scalar derived fields.

Usage
-----
    python add_derived_fields.py --sample mc
    python add_derived_fields.py --sample data
    python add_derived_fields.py --sample both
"""

import argparse
import gc
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # utils/add_derived_fields.py -> repo root

P0, P1, P2 = 0.1566, 26.0, -1.073

SAMPLES = {
    "mc": {
        "in": PROJECT_ROOT / "extracted" / "extracted-mc" / "extracted_mc.pkl",
        "out": PROJECT_ROOT / "extracted" / "extracted-mc" / "mc_new.pkl",
    },
    "data": {
        "in": PROJECT_ROOT / "extracted" / "extracted-data" / "extracted_data.pkl",
        "out": PROJECT_ROOT / "extracted" / "extracted-data" / "data_new.pkl",
    },
}


def xyz(rec):
    return np.array([float(rec['x']), float(rec['y']), float(rec['z'])], dtype=np.float64)


def impact_parameter(beam_end, shower_start, shower_dir):
    v = beam_end - shower_start
    norm = np.linalg.norm(shower_dir)
    d_hat = shower_dir / norm if norm > 0 else shower_dir
    return float(np.linalg.norm(np.cross(v, d_hat)))


def travel_distance(beam_end, shower_start):
    return float(np.linalg.norm(beam_end - shower_start))

PION_MASS_MEV = 139.57039


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def angle_between(u, v):
    """Angle in degrees between two vectors. NaN if either has zero length."""
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return np.nan
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))))


def beam_direction(pfo):
    """Beam direction for this event.

    Prefers beam_inst_dir, which is measured by the beam instrumentation
    upstream of the TPC. Falls back to the chord from beam_start_pos to
    beam_end_pos, which is the reconstructed track and so already affected by
    scattering in the argon. Returns None if neither is usable.
    """
    if 'beam_inst_dir' in pfo:
        v = xyz(pfo['beam_inst_dir'])
        if np.linalg.norm(v) > 0 and np.all(np.isfinite(v)):
            return unit(v)
    if 'beam_start_pos' in pfo:
        v = xyz(pfo['beam_end_pos']) - xyz(pfo['beam_start_pos'])
        if np.linalg.norm(v) > 0 and np.all(np.isfinite(v)):
            return unit(v)
    return None


def beam_kinetic_energy(P):
    """KE [MeV] of a charged pion from its momentum [MeV]."""
    if not np.isfinite(P) or P <= 0:
        return np.nan
    return float(np.sqrt(P * P + PION_MASS_MEV ** 2) - PION_MASS_MEV)

# The correction is E_c = E / (1 + C) with C = P0*ln(E - P1) + P2, so the
# denominator is 1 + C = P0*ln(E - P1) + (1 + P2). With these constants that
# crosses zero at E = 27.594 MeV: below it E_c is negative, at it E_c diverges
# (values down to -100 GeV were observed), and it stays non-monotonic up to
# E = 44 MeV, where two different reconstructed energies map onto the same
# corrected value. The fit is only meaningful above that, so do not apply it
# below E_C_MIN.
#
# 50 MeV is provisional -- replace with whatever range the fit was derived over
# once confirmed.
E_C_MIN = 50.0


def corrected_energy(E):
    if not np.isfinite(E) or E < E_C_MIN:
        return np.nan
    return float(E / (1 + P0 * np.log(E - P1) + P2))


def process(sample, verbose=True):
    paths = SAMPLES[sample]
    data_in, data_out = paths["in"], paths["out"]

    if not data_in.exists():
        print(f"[{sample}] input not found: {data_in} -- skipping")
        print(f"[{sample}] run extract_data.py --sample {sample} first")
        return

    size_mb = data_in.stat().st_size / 1e6
    print(f"\n[{sample}] loading {data_in} ({size_mb:.1f} MB)...")
    with open(data_in, 'rb') as f:
        raw = pickle.load(f)

    pfos = raw.to_dict('records') if isinstance(raw, pd.DataFrame) else raw
    if pfos is not raw:
        del raw
        gc.collect()
    print(f"[{sample}] loaded {len(pfos):,} PFOs")

    has_energy = 'shower_energy' in pfos[0]
    total = len(pfos)
    t0 = time.time()

    for pid, pfo in enumerate(pfos, start=1):
        # ROOT CAUSE FIX: beam_end_pos / shower_start_pos / shower_direction are
        # awkward-array Record objects. Each one drags its parent array's backing
        # buffers into the pickle once the shared layout is broken by a load->save
        # round-trip, which balloons the output ~8x (to ~9 GB+). Replace them with
        # plain {x, y, z} float dicts so the saved file stays compact (~1.2 GB).
        # Access pattern rec['x'] is unchanged for anything downstream.
        # Positions and directions are already plain dicts from pfo_extraction,
        # but older extractions may still hold awkward Records; converting is
        # cheap and idempotent.
        for _name in ('beam_end_pos', 'beam_start_pos', 'beam_inst_dir',
                      'shower_start_pos', 'shower_direction'):
            if _name in pfo:
                _r = pfo[_name]
                pfo[_name] = {'x': float(_r['x']), 'y': float(_r['y']), 'z': float(_r['z'])}

        pfo['PFO_ID'] = pid
        L = pfo['sequence_length']
        pfo['dEdX_median'] = float(np.median(pfo['dEdX_sequence'][:L])) if L > 0 else np.nan

        beam = xyz(pfo['beam_end_pos'])
        s_pos = xyz(pfo['shower_start_pos'])
        s_dir = xyz(pfo['shower_direction'])
        pfo['b'] = impact_parameter(beam, s_pos, s_dir)
        pfo['d'] = travel_distance(beam, s_pos)

        # Angle between this PFO's direction and the beam. Small means the
        # object points the way the beam was going.
        bdir = beam_direction(pfo)
        pfo['beam_angle'] = angle_between(s_dir, bdir) if bdir is not None else np.nan

        # Angle between this PFO's direction and the vertex -> shower-start
        # vector. Independent of the beam direction, so useful as a cross-check.
        pfo['vertex_angle'] = angle_between(s_dir, s_pos - beam)

        if 'beam_inst_P' in pfo:
            pfo['beam_KE'] = beam_kinetic_energy(float(pfo['beam_inst_P']))

        if has_energy:
            pfo['E_c'] = corrected_energy(float(pfo['shower_energy']))

    data_out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{sample}] saving to {data_out}...")
    with open(data_out, 'wb') as f:
        pickle.dump(pfos, f, protocol=pickle.HIGHEST_PROTOCOL)

    added = "PFO_ID, dEdX_median, b, d, beam_angle, vertex_angle"
    if 'beam_inst_P' in pfos[0]:
        added += ", beam_KE"
    if has_energy:
        added += ", E_c"
    print(f"[{sample}] done. Updated {len(pfos):,} PFOs with: {added}")


def main():
    parser = argparse.ArgumentParser(description="Add derived per-PFO fields to extracted pkl files.")
    parser.add_argument(
        "-s", "--sample", dest="sample", choices=["mc", "data", "both"], default="both",
        help="Which sample(s) to process. Default: both.",
    )
    parser.add_argument(
        "-q", "--quiet", dest="quiet", action="store_true", help="Suppress per-1000 progress lines.",
    )
    args = parser.parse_args()

    samples = ["mc", "data"] if args.sample == "both" else [args.sample]
    for sample in samples:
        process(sample, verbose=not args.quiet)


if __name__ == '__main__':
    main()
