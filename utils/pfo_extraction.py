"""
Per-PFO and per-event extraction from a selected PDSP ntuple.

This is Will's original utils.py with the mask-handling code removed:
`get_mc_masks`, `combine_fiducial_and_beam_masks` and `select_mc_events` are
gone, because beam/fiducial selection is now done by Shyam's
`BeamPionSelection` (see extract_data.py). What remains is the part that has no
equivalent in Shyam's repo -- turning selected events into the flat per-PFO
dicts the classifiers consume.

Everything here is truth-optional: on real data `trueParticlesBT` is empty, so
`particle` comes back as None/'other' and the pi0 truth fields take their
defaults.
"""

import awkward as ak
import numpy as np

from python.analysis import Master
from python.analysis.Tags import GenerateTrueParticleTagsPiPlus


# Hits above this dE/dx are detector artefacts, not real energy deposits.
DEDX_OUTLIER_THRESHOLD = 813.9

# 4 FSI topologies (Bhuller Table 4.1), charged pions merged (no B-field).
TOPO_NAMES = {
    0: "absorption",
    1: "charge_exchange",
    2: "single_pi",
    3: "pion_production",
    -1: "non_signal",
}


def has_truth(events) -> bool:
    """True if this sample carries truth information (i.e. it is MC)."""
    return bool(ak.count(events.trueParticlesBT.pdg) > 0)


def _as_data_list(selected):
    """Accept a Data, a list of Data, or a list of (filename, Data) tuples."""
    if isinstance(selected, Master.Data):
        return [selected]
    if isinstance(selected, list):
        if len(selected) == 0:
            return []
        if isinstance(selected[0], tuple):
            return [d for (_, d) in selected]
        return selected
    raise TypeError(
        "selected must be a Master.Data, a list of Master.Data, "
        "or a list of (filename, Master.Data) tuples."
    )


def find_particle_from_tags(tags, event, track):
    """Find particle type from tags for a given event and track."""
    for k, v in tags.items():
        if v.mask[event][track]:
            return k


def _plain_xyz(rec):
    """Convert an awkward position/direction Record to a plain {x, y, z} dict.

    This matters a lot: awkward Records pickle via ak_from_buffers, so a file
    full of them takes many minutes to *load* (and drags each Record's parent
    buffers along, inflating the file). Converting at extraction time keeps the
    pkl compact and fast to read. The rec['x'] access pattern is unchanged for
    everything downstream.
    """
    return {"x": float(rec["x"]), "y": float(rec["y"]), "z": float(rec["z"])}


def _extract_summary_statistics(mc, event, track, tags):
    """Extract scalar summary statistics for a PFO.

    Beam quantities are per-event, so they are indexed by [event] alone and
    repeated on every PFO in that event. All four are reco, not truth, so they
    exist on real data as well as MC:

      beam_end_pos   reco_beam_endX/Y/Z      -- the interaction vertex
      beam_start_pos reco_beam_startX/Y/Z    -- where the beam track begins
      beam_inst_dir  beam_inst_dirX/Y/Z      -- direction from the beam
                                                 instrumentation, upstream of
                                                 the TPC
      beam_inst_P    beam_inst_P             -- instrumented momentum, MeV
                                                 (Master applies the scale)
    """
    return {
        "track_chi2/ndof_proton": (
            mc.recoParticles.track_chi2_proton[event][track]
            / mc.recoParticles.track_chi2_proton_ndof[event][track]
        ),
        "track_length": float(mc.recoParticles.track_len[event][track]),
        "track_score": float(mc.recoParticles.track_score[event][track]),
        "beam_end_pos": _plain_xyz(mc.recoParticles.beam_endPos[event]),
        "beam_start_pos": _plain_xyz(mc.recoParticles.beam_startPos[event]),
        "beam_inst_dir": _plain_xyz(mc.recoParticles.beam_inst_dir[event]),
        "beam_inst_P": float(mc.recoParticles.beam_inst_P[event]),
        "shower_start_pos": _plain_xyz(mc.recoParticles.shower_start_pos[event][track]),
        "shower_direction": _plain_xyz(mc.recoParticles.shower_direction[event][track]),
        "shower_energy": float(mc.recoParticles.shower_energy[event][track]),
        "n_hits": int(mc.recoParticles.n_hits[event][track]),
        "n_hits_collection": int(mc.recoParticles.n_hits_collection[event][track]),
        "particle": find_particle_from_tags(tags, event, track),
    }


def _extract_sequences(mc, event, track):
    """Extract dEdX and residual-range sequences for a PFO (beam-end first)."""
    dEdX_seq = np.array(mc.recoParticles.track_dEdX[event][track], dtype=np.float32)
    rr_seq = np.array(mc.recoParticles.residual_range[event][track], dtype=np.float32)
    return dEdX_seq[::-1], rr_seq[::-1]


def _pad_single_sequence(sequence, max_length, pad_value=0.0):
    """Pad or truncate a sequence to a fixed length."""
    seq_len = len(sequence)
    if seq_len < max_length:
        padded = np.pad(sequence, (0, max_length - seq_len),
                        mode="constant", constant_values=pad_value)
    else:
        padded = sequence[:max_length]
    return padded.astype(sequence.dtype)


def extract_pfo_data(selected, max_sequence_length=222, verbose=False):
    """
    Extract per-PFO observables from already-selected samples.

    Returns (list_of_pfo_dicts, stats). `event_number` is a global counter
    across all files, and matches the one extract_event_data assigns.
    """
    data_list = _as_data_list(selected)
    if not data_list:
        return [], {"n_pfos_before_errors": 0, "n_pfos_skipped_error": 0, "n_pfos_final": 0}

    all_pfos = []
    n_pfos_before_errors = 0
    n_pfos_skipped_error = 0
    event_num = -1

    for mc in data_list:
        tags = GenerateTrueParticleTagsPiPlus(mc)
        has_truth_info = has_truth(mc)
        n_events_file = len(mc.recoParticles.track_chi2_proton)

        for event in range(n_events_file):
            if verbose and event > 0 and event % 10_000 == 0:
                print(f"    {event:,}/{n_events_file:,} events "
                      f"({100*event/n_events_file:.0f}%), {len(all_pfos):,} PFOs so far",
                      flush=True)

            n_pfos_event = len(mc.recoParticles.track_chi2_proton[event])
            n_pfos_before_errors += n_pfos_event
            event_num += 1

            for pfo_index in range(n_pfos_event):
                try:
                    dEdX_seq, rr_seq = _extract_sequences(mc, event, pfo_index)
                    dEdX_seq = dEdX_seq.astype(np.float32, copy=True)
                    rr_seq = rr_seq.astype(np.float32, copy=True)
                    if len(dEdX_seq) != len(rr_seq):
                        raise ValueError("dEdX and residual range sequences have different lengths")

                    if len(dEdX_seq) > 0:
                        keep = dEdX_seq <= DEDX_OUTLIER_THRESHOLD
                        dEdX_seq = dEdX_seq[keep]
                        rr_seq = rr_seq[keep]

                    summary_info = _extract_summary_statistics(mc, event, pfo_index, tags)

                    # Truth-only fields; stay at their defaults on real data.
                    is_gamma_from_beam_pi0 = False
                    pi0_mother_id = -1
                    if has_truth_info:
                        pdg = mc.trueParticlesBT.pdg[event][pfo_index]
                        is_beam_pi0 = mc.trueParticlesBT.is_beam_pi0[event][pfo_index]
                        is_gamma_from_beam_pi0 = bool(pdg == 22 and is_beam_pi0)
                        if is_gamma_from_beam_pi0:
                            pi0_mother_id = int(mc.trueParticlesBT.mother[event][pfo_index])

                    all_pfos.append({
                        "dEdX_sequence": _pad_single_sequence(dEdX_seq, max_sequence_length),
                        "residual_range_sequence": _pad_single_sequence(rr_seq, max_sequence_length),
                        "sequence_length": len(dEdX_seq),
                        "particle": summary_info["particle"],
                        "is_gamma_from_beam_pi0": is_gamma_from_beam_pi0,
                        "pi0_mother_id": pi0_mother_id,
                        "track_chi2/ndof_proton": summary_info["track_chi2/ndof_proton"],
                        "track_length": summary_info["track_length"],
                        "track_score": summary_info["track_score"],
                        "beam_end_pos": summary_info["beam_end_pos"],
                        "beam_end_pos": summary_info["beam_end_pos"],
                        "beam_start_pos": summary_info["beam_start_pos"],
                        "beam_inst_dir": summary_info["beam_inst_dir"],
                        "beam_inst_P": summary_info["beam_inst_P"],
                        "shower_start_pos": summary_info["shower_start_pos"],
                        "shower_direction": summary_info["shower_direction"],
                        "shower_energy": summary_info["shower_energy"],
                        "n_hits": summary_info["n_hits"],
                        "n_hits_collection": summary_info["n_hits_collection"],
                        "event_number": event_num,
                    })

                except Exception:
                    n_pfos_skipped_error += 1
                    continue

    stats = {
        "n_pfos_before_errors": n_pfos_before_errors,
        "n_pfos_skipped_error": n_pfos_skipped_error,
        "n_pfos_final": len(all_pfos),
    }

    if verbose:
        # NB: outlier rejection removes individual HITS within a PFO, never a
        # whole PFO -- so "seen" and "extracted" only differ when an error is hit.
        print(f"  PFOs seen:                  {n_pfos_before_errors:,}")
        print(f"  PFOs skipped due to errors: {n_pfos_skipped_error:,}")
        print(f"  PFOs extracted:             {stats['n_pfos_final']:,}")

    return all_pfos, stats


def _fsi_topology(n_pipm, n_pi0):
    """4 FSI topologies (Bhuller Table 4.1), charged pions merged (no B-field)."""
    if n_pipm == 0 and n_pi0 == 0:
        return 0   # absorption
    if n_pipm == 0 and n_pi0 == 1:
        return 1   # charge exchange
    if n_pipm == 1 and n_pi0 == 0:
        return 2   # single-pion production
    return 3       # pion production (multi)


def extract_event_data(selected, verbose=False):
    """
    One row per event, keyed by the SAME global event_number extract_pfo_data
    uses. Truth columns are only present for MC.
    """
    data_list = _as_data_list(selected)
    events_out = []
    event_num = -1

    for mc in data_list:
        has_truth_info = has_truth(mc)
        tp = mc.trueParticles

        for event in range(len(mc.recoParticles.track_chi2_proton)):
            event_num += 1
            row = {
                "event_number": event_num,
                "n_pfos": int(len(mc.recoParticles.track_chi2_proton[event])),
            }
            if has_truth_info:
                n_pipm = int(tp.nPiPlus[event]) + int(tp.nPiMinus[event])
                n_pi0 = int(tp.nPi0[event])
                end = str(tp.true_beam_endProcess[event])
                signal = (end == "pi+Inelastic")
                topo = _fsi_topology(n_pipm, n_pi0) if signal else -1
                row.update({
                    "beam_KE_front_face": float(tp.beam_KE_front_face[event]),
                    "true_beam_endProcess": end,
                    "nPiPlus": int(tp.nPiPlus[event]),
                    "nPiMinus": int(tp.nPiMinus[event]),
                    "nPi0": n_pi0,
                    "nProton": int(tp.nProton[event]),
                    "true_topology": topo,
                    "true_topology_name": TOPO_NAMES[topo],
                })
            events_out.append(row)

    if verbose:
        from collections import Counter
        print(f"  events extracted: {len(events_out):,}")
        if events_out and "true_topology_name" in events_out[0]:
            print(f"  {Counter(e['true_topology_name'] for e in events_out)}")

    return events_out