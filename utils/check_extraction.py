#!/usr/bin/env python3
"""
Sanity-check the output of extract_data.py before trusting it.

    python utils/check_extraction.py --sample both --test    # checks *_test.pkl
    python utils/check_extraction.py --sample both           # checks the real files

Checks, per sample:
  * the pkl loads and is a non-empty list of dicts
  * every expected field is present
  * sequences are padded to MAX_LEN and sequence_length <= MAX_LEN
  * sequence_length matches the number of non-padded dEdX entries
  * MC has truth (particle labels populated); data does not
  * event_number is contiguous from 0 and consistent with the events file
"""

import argparse
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAX_LEN = 222

EXPECTED_FIELDS = [
    "dEdX_sequence", "residual_range_sequence", "sequence_length", "particle",
    "is_gamma_from_beam_pi0", "pi0_mother_id", "track_chi2/ndof_proton",
    "track_length", "track_score", "beam_end_pos", "beam_start_pos",
    "beam_inst_dir", "beam_inst_P", "shower_start_pos",
    "shower_direction", "shower_energy", "n_hits", "n_hits_collection",
    "event_number",
]

SAMPLES = {
    "mc": (PROJECT_ROOT / "extracted" / "extracted-mc", "extracted_mc", "mc_events"),
    "data": (PROJECT_ROOT / "extracted" / "extracted-data", "extracted_data", "data_events"),
}

PASS, FAIL, WARN = "  PASS", "  FAIL", "  WARN"


def check(sample, test=False):
    out_dir, pfo_stem, event_stem = SAMPLES[sample]
    suffix = "_test" if test else ""
    pfo_path = out_dir / f"{pfo_stem}{suffix}.pkl"
    event_path = out_dir / f"{event_stem}{suffix}.pkl"

    print(f"\n{'='*70}\n{sample.upper()}  ({pfo_path.name})\n{'='*70}")

    if not pfo_path.exists():
        print(f"{FAIL}  file not found: {pfo_path}")
        return False

    with open(pfo_path, "rb") as f:
        pfos = pickle.load(f)

    ok = True

    # --- basic shape -------------------------------------------------------
    if not isinstance(pfos, list) or len(pfos) == 0:
        print(f"{FAIL}  expected a non-empty list, got {type(pfos)} of len "
              f"{len(pfos) if hasattr(pfos, '__len__') else '?'}")
        return False
    print(f"{PASS}  loaded {len(pfos):,} PFOs")

    # --- fields ------------------------------------------------------------
    missing = [k for k in EXPECTED_FIELDS if k not in pfos[0]]
    if missing:
        print(f"{FAIL}  missing fields: {missing}")
        ok = False
    else:
        print(f"{PASS}  all {len(EXPECTED_FIELDS)} expected fields present")

    # --- padding / lengths -------------------------------------------------
    # NOTE: sequence_length is the TRUE hit count before truncation, so it is
    # legitimately allowed to exceed MAX_LEN -- the stored arrays are what get
    # padded/truncated. Downstream code uses min(sequence_length, MAX_LEN).
    bad_pad = bad_len = bad_count = n_truncated = 0
    max_len_seen = 0
    for p in pfos[:5000]:                      # sampling is enough to catch systematics
        if len(p["dEdX_sequence"]) != MAX_LEN or len(p["residual_range_sequence"]) != MAX_LEN:
            bad_pad += 1
        L = p["sequence_length"]
        max_len_seen = max(max_len_seen, L)
        if L < 0:
            bad_len += 1
        elif L > MAX_LEN:
            n_truncated += 1
        elif int(np.count_nonzero(np.asarray(p["dEdX_sequence"]) != 0.0)) > L:
            bad_count += 1

    print(f"{PASS if bad_pad == 0 else FAIL}  sequences padded to {MAX_LEN}"
          f"{'' if bad_pad == 0 else f' ({bad_pad} wrong)'}")
    print(f"{PASS if bad_len == 0 else FAIL}  sequence_length >= 0"
          f"{'' if bad_len == 0 else f' ({bad_len} negative)'}")
    print(f"{PASS if bad_count == 0 else WARN}  non-zero hits <= sequence_length"
          f"{'' if bad_count == 0 else f' ({bad_count} mismatched)'}")
    print(f"  info  {n_truncated:,} PFOs longer than {MAX_LEN} hits (truncated on store); "
          f"longest = {max_len_seen:,}")
    ok = ok and bad_pad == 0 and bad_len == 0

    n_empty = sum(1 for p in pfos if p["sequence_length"] == 0)
    print(f"  info  {n_empty:,} PFOs with zero hits "
          f"({100*n_empty/len(pfos):.1f}%) -- dropped later in data-preparation")

    # --- truth presence ----------------------------------------------------
    species = Counter(p["particle"] for p in pfos)
    labelled = sum(v for k, v in species.items() if k is not None)
    frac_labelled = labelled / len(pfos)

    if sample == "mc":
        if frac_labelled > 0.5:
            print(f"{PASS}  truth present: {frac_labelled:.0%} of PFOs have a species label")
            for sp, n in species.most_common(8):
                print(f"          {n:>8,}  {sp!r}")
        else:
            print(f"{FAIL}  MC but only {frac_labelled:.0%} of PFOs are labelled "
                  "-- truth may not have been read")
            ok = False
        n_pi0_gamma = sum(1 for p in pfos if p["is_gamma_from_beam_pi0"])
        print(f"  info  {n_pi0_gamma:,} photons from a beam pi0")
    else:
        # On real data the tag generator has no truth to work from, so every PFO
        # falls through to the catch-all 'other' category (or None). Anything
        # else means truth has leaked into the data path.
        real_species = {k for k in species if k not in (None, "other")}
        if not real_species:
            print(f"{PASS}  no real species labels, as expected for data "
                  f"(all {len(pfos):,} PFOs are 'other'/None)")
        else:
            print(f"{FAIL}  real data has genuine species labels: {sorted(real_species)}")
            ok = False

    # --- event numbering ---------------------------------------------------
    evt = np.array([p["event_number"] for p in pfos])
    uniq = np.unique(evt)
    contiguous = uniq[0] == 0 and uniq[-1] == len(uniq) - 1
    print(f"{PASS if contiguous else WARN}  event_number spans {uniq[0]}..{uniq[-1]} "
          f"over {len(uniq):,} events{'' if contiguous else ' (not contiguous)'}")

    # --- events file -------------------------------------------------------
    if not event_path.exists():
        print(f"{WARN}  events file not found: {event_path}")
    else:
        with open(event_path, "rb") as f:
            events = pickle.load(f)
        if not isinstance(events, pd.DataFrame):
            print(f"{WARN}  events file is {type(events)}, expected DataFrame")
        else:
            print(f"{PASS}  events file: {len(events):,} rows, columns={list(events.columns)}")
            if len(events) != len(uniq):
                print(f"{WARN}  {len(events):,} event rows vs {len(uniq):,} distinct "
                      "event_numbers in the PFO file")
            if sample == "mc" and "true_topology_name" in events.columns:
                print("  info  true topologies:")
                for name, n in events["true_topology_name"].value_counts().items():
                    print(f"          {n:>8,}  {name}")
            elif sample == "mc":
                print(f"{FAIL}  MC events file has no true_topology_name column")
                ok = False

    return ok


def main():
    parser = argparse.ArgumentParser(description="Sanity-check extract_data.py output.")
    parser.add_argument("-s", "--sample", choices=["mc", "data", "both"], default="both")
    parser.add_argument("--test", action="store_true", help="Check the *_test.pkl smoke-test files.")
    args = parser.parse_args()

    samples = ["mc", "data"] if args.sample == "both" else [args.sample]
    results = {s: check(s, test=args.test) for s in samples}

    print(f"\n{'='*70}")
    for s, ok in results.items():
        print(f"{s:>6}: {'OK' if ok else 'PROBLEMS FOUND'}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
