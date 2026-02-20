"""
Add derived fields to ALL_DATA.pkl:

  track_ID  1-indexed integer identifier for each track
  b         impact parameter, float64, cm
              b = ||(beam_end_pos - shower_start_pos) x shower_direction||
  d         photon travel distance, float64, cm
              d = ||beam_end_pos - shower_start_pos||
  dEdX_median  median dEdX over actual hits (excludes zero-padding), float64
  E_c          corrected shower energy, float64
               Only added when 'shower_energy' exists in the data.
               E_c = E / (1 + C),  C = P0*ln(E - P1) + P2

Energy correction parameters (CORSIKA/LArSoft calibration):
  P0 = 0.1566,  P1 = 26.0 MeV,  P2 = -1.073

Usage:
  python utils/add_derived_fields.py          (from project root)
  python add_derived_fields.py                (from utils/)
"""

import os
import pickle
import numpy as np

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'extracted-data', 'ALL_DATA.pkl')

# Energy correction parameters
P0, P1, P2 = 0.1566, 26.0, -1.073


def xyz(record):
    """Extract (x, y, z) from an awkward Record into a float64 numpy array."""
    return np.array([float(record['x']), float(record['y']), float(record['z'])],
                    dtype=np.float64)


def impact_parameter(beam_end, shower_start, shower_dir):
    """b = ||(beam_end - shower_start) x shower_dir_normalised||"""
    v = beam_end - shower_start
    norm = np.linalg.norm(shower_dir)
    d_hat = shower_dir / norm if norm > 0 else shower_dir
    return float(np.linalg.norm(np.cross(v, d_hat)))


def travel_distance(beam_end, shower_start):
    """d = ||beam_end - shower_start||"""
    return float(np.linalg.norm(beam_end - shower_start))


def corrected_energy(E):
    """E_c = E / (1 + C);  C = P0*ln(E - P1) + P2.  Returns nan if E <= P1."""
    if np.isnan(E) or E <= P1:
        return np.nan
    C = P0 * np.log(E - P1) + P2
    return float(E / (1 + C))


def main():
    print(f"Loading {DATA_PATH} ...")
    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded {len(data):,} tracks")

    has_energy = 'shower_energy' in data[0]

    for track_id, track in enumerate(data, start=1):
        track['track_ID'] = track_id

        L = track['sequence_length']
        track['dEdX_median'] = float(np.median(track['dEdX_sequence'][:L]))

        beam  = xyz(track['beam_end_pos'])
        s_pos = xyz(track['shower_start_pos'])
        s_dir = xyz(track['shower_direction'])

        track['b'] = impact_parameter(beam, s_pos, s_dir)
        track['d'] = travel_distance(beam, s_pos)

        if has_energy:
            track['E_c'] = corrected_energy(float(track['shower_energy']))

    print("Saving ...")
    with open(DATA_PATH, 'wb') as f:
        pickle.dump(data, f)

    added = "track_ID, dEdX_median, b, d" + (", E_c" if has_energy else "")
    print(f"Done — updated {len(data):,} tracks with: {added}")
    if not has_energy:
        print("  Note: 'shower_energy' not found in data — E_c was not computed.")


if __name__ == '__main__':
    main()
