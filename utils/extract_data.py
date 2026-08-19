#!/usr/bin/env python3
"""
Unified extraction of PDSP ntuples into the per-PFO / per-event pkl files used
by this project. Replaces extract_data_mc.ipynb (Will's) and
extract_data_real.py, which did the same job by two different routes.

Why one script:
  The MC path used to apply a *fiducial* mask (masks/fiducial_selection_masks.dill)
  that the data path never applied, so MC and data were not selected the same
  way -- a likely source of the data/MC discrepancies. Both now go through
  exactly the same selection, differing only in the `is_mc` flag.

Selection is done by Shyam's `BeamPionSelection`, not by hand-loading .dill
masks. That function:
  * skips the fiducial cut entirely when no fiducial masks are configured
    (`if "fiducial" in masks and len(masks["fiducial"]) > 0`), so a missing
    fiducial_masks.dill is a non-event rather than a crash;
  * picks mc_arguments vs data_arguments off the same config via `is_mc`;
  * runs the cuts live from BEAM_PARTICLE_SELECTION when no SELECTION_MASKS
    block is present, so no precomputed .dill files are needed at all.

Truth handling is automatic: real data has an empty trueParticlesBT, so the
truth-only fields (particle labels, pi0 provenance, FSI topology) are simply
not filled. No separate data-only code path.

Usage
-----
    python extract_data.py --config extraction_config.json --sample mc
    python extract_data.py --config extraction_config.json --sample data
    python extract_data.py --config extraction_config.json --sample both

Outputs (per sample)
--------------------
    extracted/extracted-mc/extracted_mc.pkl      per-PFO   (MC)
    extracted/extracted-mc/mc_events.pkl         per-event (MC)
    extracted/extracted-data/extracted_data.pkl  per-PFO   (data)
    extracted/extracted-data/data_events.pkl     per-event (data)

Next step: run add_derived_fields.py on the per-PFO pkl to add
PFO_ID / dEdX_median / b / d / E_c.
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

# --- make Shyam's analysis tree importable ---------------------------------
# Needs to resolve both `python.analysis.*` and `apps.*`, which both live
# under <ANALYSIS_ROOT>.
#
# Resolution order:
#   1. $XS_ANALYSIS_ROOT, if set (use this if your clone lives somewhere else)
#   2. ~/xs_analysis/pion-argon-xs-analysis/analysis
DEFAULT_ANALYSIS_ROOT = Path.home() / "xs_analysis" / "pion-argon-xs-analysis" / "analysis"
ANALYSIS_ROOT = Path(
    os.environ.get("XS_ANALYSIS_ROOT", DEFAULT_ANALYSIS_ROOT)
).expanduser().resolve()

if not (ANALYSIS_ROOT / "python" / "analysis").is_dir():
    raise SystemExit(
        f"Could not find the analysis tree at:\n  {ANALYSIS_ROOT}\n\n"
        f"Expected to see {ANALYSIS_ROOT / 'python' / 'analysis'}.\n"
        "Point XS_ANALYSIS_ROOT at your clone, e.g.\n"
        "  export XS_ANALYSIS_ROOT=/path/to/pion-argon-xs-analysis/analysis"
    )

if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

import pandas as pd  # noqa: E402

from python.analysis import Master, Application  # noqa: E402
from apps.cex_analysis_input import BeamPionSelection  # noqa: E402

from pfo_extraction import extract_pfo_data, extract_event_data, has_truth  # noqa: E402
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # utils/extract_data.py -> repo root

MAX_SEQUENCE_LENGTH = 222

OUTPUTS = {
    "mc": {
        "dir": PROJECT_ROOT / "extracted" / "extracted-mc",
        "pfos": "extracted_mc.pkl",
        "events": "mc_events.pkl",
    },
    "data": {
        "dir": PROJECT_ROOT / "extracted" / "extracted-data",
        "pfos": "extracted_data.pkl",
        "events": "data_events.pkl",
    },
}


def select_events(file_descriptors, args, is_mc, verbose=True, n_events=None):
    """
    Load each ntuple and apply the beam pion selection.

    Returns a list of (filename, Data) tuples, which is what extract_pfo_data
    and extract_event_data expect.

    n_events limits how many events are read per file -- use it for smoke tests,
    leave it None for the real run.
    """
    selected = []
    total_before = 0
    total_after = 0

    for fd in file_descriptors:
        if n_events is None:
            events = Master.Data(fd, verbose=verbose)
        else:
            events = Master.Data(fd, n_events, verbose=verbose)
        n_before = len(events.eventNum)

        events_selected = BeamPionSelection(events, args, is_mc)
        n_after = len(events_selected.eventNum)

        total_before += n_before
        total_after += n_after

        if verbose:
            print("-" * 85)
            print(f"{events.filename}")
            print(f"  events before selection: {n_before:,}")
            print(f"  events after  selection: {n_after:,}")
            print(f"  truth information:       {'yes' if has_truth(events_selected) else 'no'}")

        selected.append((events.filename, events_selected))

    if verbose:
        print("-" * 85)
        print(f"Totals: {total_before:,} -> {total_after:,} events "
              f"across {len(file_descriptors)} file(s)")

    return selected


def run_sample(sample, args, verbose=True, n_events=None, suffix=""):
    """Extract one sample ('mc' or 'data') end to end."""
    is_mc = (sample == "mc")

    file_descriptors = args.ntuple_files.get(sample)
    if not file_descriptors:
        print(f"[{sample}] no ntuple files configured under NTUPLE_FILES.{sample} -- skipping")
        return

    print(f"\n{'='*85}\n{sample.upper()}: selecting events\n{'='*85}")
    selected = select_events(file_descriptors, args, is_mc, verbose=verbose, n_events=n_events)

    out_dir = OUTPUTS[sample]["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[{sample}] extracting per-PFO data...")
    pfo_data, _ = extract_pfo_data(
        selected, max_sequence_length=MAX_SEQUENCE_LENGTH, verbose=verbose
    )
    pfo_path = out_dir / OUTPUTS[sample]["pfos"].replace(".pkl", f"{suffix}.pkl")
    with open(pfo_path, "wb") as f:
        pickle.dump(pfo_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[{sample}] saved {len(pfo_data):,} PFOs to {pfo_path}")

    print(f"\n[{sample}] extracting per-event data...")
    event_data = extract_event_data(selected, verbose=verbose)
    event_path = out_dir / OUTPUTS[sample]["events"].replace(".pkl", f"{suffix}.pkl")
    with open(event_path, "wb") as f:
        pickle.dump(pd.DataFrame(event_data), f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[{sample}] saved {len(event_data):,} event rows to {event_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract PDSP ntuples into per-PFO / per-event pkl files."
    )
    Application.ApplicationArguments.Config(parser, required=True)
    parser.add_argument(
        "-s", "--sample", dest="sample", choices=["mc", "data", "both"], default="both",
        help="Which sample(s) to extract. Default: both.",
    )
    parser.add_argument(
        "-q", "--quiet", dest="quiet", action="store_true", help="Less verbose output.",
    )
    parser.add_argument(
        "-n", "--events", dest="n_events", type=int, default=None,
        help="Only read this many events per file (smoke test). Writes to *_test.pkl "
             "so it cannot clobber a real extraction.",
    )
    # override_out=False: this script writes to fixed project paths, so we do
    # not want ResolveArgs inventing an output directory from the ntuple name.
    args = Application.ApplicationArguments.ResolveArgs(parser.parse_args(), override_out=False)

    verbose = not getattr(args, "quiet", False)
    n_events = getattr(args, "n_events", None)
    suffix = "_test" if n_events is not None else ""

    if n_events is not None:
        print(f"SMOKE TEST: reading at most {n_events:,} events per file, "
              f"writing to *_test.pkl (real outputs untouched).")

    samples = ["mc", "data"] if args.sample == "both" else [args.sample]
    for sample in samples:
        run_sample(sample, args, verbose=verbose, n_events=n_events, suffix=suffix)

    if n_events is not None:
        print("\nSmoke test done. Re-run without -n to produce the real files.")
    else:
        print("\nDone. Next: run add_derived_fields.py on the per-PFO pkl(s).")


if __name__ == "__main__":
    main()
