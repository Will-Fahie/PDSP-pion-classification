"""
Adds derived fields to extracted_data.pkl -> data_new.pkl:

  PFO_ID       1-indexed integer per PFO
  b            impact parameter, cm
  d            photon travel distance, cm
  dEdX_median  median dEdX over real hits (zero-padding excluded)
  E_c          corrected shower energy, only if 'shower_energy' present

Energy correction (CORSIKA/LArSoft calibration):
  E_c = E / (1 + C),  C = P0*ln(E - P1) + P2
"""

import os
import pickle
import time

import numpy as np
import pandas as pd


HERE = os.path.dirname(__file__)
DATA_IN  = os.path.join(HERE, '..', 'extracted-data', 'extracted_data.pkl')
DATA_OUT = os.path.join(HERE, '..', 'extracted-data', 'data_new.pkl')

P0, P1, P2 = 0.1566, 26.0, -1.073


def xyz(rec):
    return np.array([float(rec['x']), float(rec['y']), float(rec['z'])], dtype=np.float64)


def impact_parameter(beam_end, shower_start, shower_dir):
    v = beam_end - shower_start
    norm = np.linalg.norm(shower_dir)
    d_hat = shower_dir / norm if norm > 0 else shower_dir
    return float(np.linalg.norm(np.cross(v, d_hat)))


def travel_distance(beam_end, shower_start):
    return float(np.linalg.norm(beam_end - shower_start))


def corrected_energy(E):
    if np.isnan(E) or E <= P1:
        return np.nan
    C = P0 * np.log(E - P1) + P2
    return float(E / (1 + C))


def main():
    size_mb = os.path.getsize(DATA_IN) / 1e6
    print(f"Loading {DATA_IN} ({size_mb:.1f} MB)...")
    with open(DATA_IN, 'rb') as f:
        raw = pickle.load(f)

    pfos = raw.to_dict('records') if isinstance(raw, pd.DataFrame) else raw
    print(f"Loaded {len(pfos):,} PFOs")

    has_energy = 'shower_energy' in pfos[0]
    total = len(pfos)
    t0 = time.time()

    for pid, pfo in enumerate(pfos, start=1):
        pfo['PFO_ID'] = pid
        L = pfo['sequence_length']
        pfo['dEdX_median'] = float(np.median(pfo['dEdX_sequence'][:L]))

        beam  = xyz(pfo['beam_end_pos'])
        s_pos = xyz(pfo['shower_start_pos'])
        s_dir = xyz(pfo['shower_direction'])
        pfo['b'] = impact_parameter(beam, s_pos, s_dir)
        pfo['d'] = travel_distance(beam, s_pos)

        if has_energy:
            pfo['E_c'] = corrected_energy(float(pfo['shower_energy']))

        if pid % 1000 == 0 or pid == total:
            dt = time.time() - t0
            rate = pid / dt
            eta = (total - pid) / rate
            print(f"  {pid:,}/{total:,}  ({100*pid/total:.1f}%)  {rate:.0f} PFO/s  ~{eta:.0f}s left")

    print(f"Saving to {DATA_OUT}...")
    with open(DATA_OUT, 'wb') as f:
        pickle.dump(pfos, f)

    added = "PFO_ID, dEdX_median, b, d" + (", E_c" if has_energy else "")
    print(f"Done. Updated {len(pfos):,} PFOs with: {added}")
    if not has_energy:
        print("  Note: 'shower_energy' not in data; E_c skipped.")


if __name__ == '__main__':
    main()
