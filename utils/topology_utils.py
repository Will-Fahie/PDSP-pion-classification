"""
Event-level topology reconstruction for the PDSP pion analysis.

Everything here operates one level *above* the PFO classifiers: it takes
per-PFO tags (from cuts or from a classifier) and reconstructs what happened
in the event -- how many charged pions, how many photons, whether a pi0 can be
built from a photon pair, and hence which interaction topology occurred.

Contents
--------
Pi0Reconstructor : builds pi0 candidates from tagged photons and scores them
classify_topology: maps (n_pi+, n_gamma, n_pi0) onto an interaction topology
get_matrix       : joint (true, reco) counter -> normalised confusion matrix
save_scan/load_scan : persist threshold scans with provenance metadata

Why a class for the pi0 side: scoring a pi0 selection needs the event index,
the per-event PFO lists, the truth flags and a pair-kinematics cache. Passing
five parallel structures between functions was the main source of clutter in
topology-selection.ipynb, so they live together instead.
"""

import csv
import itertools
from datetime import datetime

import numpy as np

from utils.general_utils import binomial_unc_pct

# ---------------------------------------------------------------------------
# Physics constants
# ---------------------------------------------------------------------------

PION_LABEL  = r'$\pi^{\pm}$'
GAMMA_LABEL = r'$\gamma$'

PI0_MASS = 135.0   # MeV, PDG
MASS_CUT = 225.0   # MeV,     thesis Table 4.16: m_gammagamma < 225
PHI_CUT  = 60.0    # degrees, thesis Table 4.16: opening angle < 60


# ---------------------------------------------------------------------------
# Shower geometry
# ---------------------------------------------------------------------------

def shower_dir_vec(pfo):
    """Unit-ish direction vector of a shower, as a plain numpy array."""
    sd = pfo['shower_direction']
    return np.array([float(sd['x']), float(sd['y']), float(sd['z'])])


def pi0_candidate(pfo1, pfo2):
    """Invariant mass [MeV] and opening angle [deg] of a two-photon candidate.

    Uses the corrected shower energies E_c. Returns (None, None) if either
    energy is missing or NaN, which happens for showers whose calorimetric
    correction could not be computed.
    """
    E1, E2 = pfo1.get('E_c'), pfo2.get('E_c')
    if E1 is None or E2 is None or np.isnan(E1) or np.isnan(E2):
        return None, None
    d1, d2 = shower_dir_vec(pfo1), shower_dir_vec(pfo2)
    cos_phi = float(np.clip(np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2)), -1.0, 1.0))
    phi  = float(np.degrees(np.arccos(cos_phi)))
    mass = float(np.sqrt(max(0.0, 2 * E1 * E2 * (1 - cos_phi))))
    return mass, phi


# ---------------------------------------------------------------------------
# pi0 reconstruction
# ---------------------------------------------------------------------------

class Pi0Reconstructor:
    """Builds and scores pi0 candidates from a set of photon-tagged PFOs.

    Construction indexes the PFO list by event once; after that, scoring a new
    photon selection is cheap, so threshold scans are practical.

    Pairing modes
    -------------
    "exact" : require exactly 2 tagged photons in the event (the thesis
              requirement). A third tagged shower kills the event outright,
              which means the requirement gets *harder* to satisfy as the
              photon selection gets more efficient.
    "best"  : allow >= 2 tagged photons and choose the pair whose invariant
              mass is closest to the pi0 mass. Recovers events that "exact"
              discards.

    Note on "best": choosing the pair by proximity to 135 MeV biases the
    reconstructed mass spectrum toward 135 MeV. That is acceptable when
    *counting* pi0, but the selected-pair mass distribution must not then be
    used to fit or measure the pi0 mass.
    """

    def __init__(self, pfos, mass_cut=MASS_CUT, phi_cut=PHI_CUT,
                 gamma_label=GAMMA_LABEL):
        self.pfos = pfos
        self.mass_cut = mass_cut
        self.phi_cut = phi_cut

        # Event indexing: ev_idx[i] is the compact event index of PFO i.
        evt_numbers = np.array([p['event_number'] for p in pfos])
        self.event_numbers, self.ev_idx = np.unique(evt_numbers, return_inverse=True)
        self.n_ev = len(self.event_numbers)

        self.idx_by_event = [[] for _ in range(self.n_ev)]
        for i, e in enumerate(self.ev_idx):
            self.idx_by_event[e].append(i)

        # Truth: an event has a true pi0 if >= 1 of its PFOs is a true photon
        # from a beam pi0. Deliberately not "== 2": events where only one decay
        # photon was reconstructed as a PFO still contained a real pi0.
        self.true_pi0 = np.zeros(self.n_ev, dtype=bool)
        for i, p in enumerate(pfos):
            if p['particle'] == gamma_label and p['is_gamma_from_beam_pi0']:
                self.true_pi0[self.ev_idx[i]] = True
        self.n_true_pi0 = int(self.true_pi0.sum())

        self._pair_cache = {}

    # -- internals ----------------------------------------------------------

    def _pair(self, i, j):
        """Cached (mass, phi) for a PFO pair. The same pairs recur across every
        threshold in a scan, so caching turns the scan from minutes to seconds."""
        k = (i, j) if i < j else (j, i)
        if k not in self._pair_cache:
            self._pair_cache[k] = pi0_candidate(self.pfos[k[0]], self.pfos[k[1]])
        return self._pair_cache[k]

    def both_signal(self, i, j):
        """True if both PFOs are real photons from the *same* beam pi0.
        This is the signal definition: a pair of unrelated real photons is
        still a wrong pi0."""
        a, b = self.pfos[i], self.pfos[j]
        return (a['is_gamma_from_beam_pi0'] and b['is_gamma_from_beam_pi0']
                and a['pi0_mother_id'] == b['pi0_mother_id'])

    def select(self, mask, pairing="best"):
        """Choose one pi0 candidate pair per event.

        Parameters
        ----------
        mask : (n_pfo,) bool -- which PFOs are tagged as photons
        pairing : "exact" | "best"

        Returns
        -------
        chosen : {event_index: (i, j)} for events with a candidate passing
                 the mass and phi cuts
        counts : (n_ev,) int -- tagged photons per event
        """
        if pairing not in ("exact", "best"):
            raise ValueError(f"pairing must be 'exact' or 'best', got {pairing!r}")

        counts = np.bincount(self.ev_idx[mask], minlength=self.n_ev)
        todo = np.where(counts == 2)[0] if pairing == "exact" else np.where(counts >= 2)[0]

        chosen = {}
        for e in todo:
            cands = [k for k in self.idx_by_event[e] if mask[k]]
            best, best_d = None, float('inf')
            for i, j in itertools.combinations(cands, 2):
                mass, phi = self._pair(i, j)
                if mass is None or mass >= self.mass_cut or phi >= self.phi_cut:
                    continue
                d = abs(mass - PI0_MASS)
                if d < best_d:
                    best_d, best = d, (i, j)
            if best is not None:
                chosen[e] = best
        return chosen, counts

    # -- scoring ------------------------------------------------------------

    def metrics(self, mask, pairing="best"):
        """Purity and efficiency of the resulting pi0 selection.

        TP/FP/FN follow the convention of the original notebook cell: an event
        that yields a candidate whose two photons are not from the same pi0 is
        counted as FP only, never also as FN.
        """
        chosen, counts = self.select(mask, pairing)

        TP = FP = n_pass_true = 0
        for e, (i, j) in chosen.items():
            if self.true_pi0[e]:
                n_pass_true += 1
            if self.both_signal(i, j):
                TP += 1
            else:
                FP += 1
        FN = self.n_true_pi0 - n_pass_true

        pur = TP / (TP + FP) if (TP + FP) else 0.0
        eff = TP / (TP + FN) if (TP + FN) else 0.0
        return {
            "TP": TP, "FP": FP, "FN": FN,
            "purity": 100 * pur,
            "efficiency": 100 * eff,
            "pxe": 100 * pur * eff,
            "d_purity": binomial_unc_pct(TP, TP + FP),
            "d_efficiency": binomial_unc_pct(TP, TP + FN),
            "ev_eq2": int((counts == 2).sum()),
            "ev_ge3": int((counts >= 3).sum()),
        }

    def pi0_flags(self, mask, pairing="best"):
        """(n_ev,) int array: 1 if a pi0 candidate was reconstructed.
        Use this to feed classify_topology."""
        chosen, _ = self.select(mask, pairing)
        flags = np.zeros(self.n_ev, dtype=int)
        for e in chosen:
            flags[e] = 1
        return flags

    def gamma_counts(self, mask):
        """(n_ev,) int array: tagged photons per event."""
        return np.bincount(self.ev_idx[mask], minlength=self.n_ev)

    # -- scanning -----------------------------------------------------------

    def scan(self, probs, thresholds, pairing="best"):
        """Score the pi0 selection across classifier thresholds.

        Returns a list of dicts, one per threshold, each carrying the full
        metrics dict plus the threshold itself.
        """
        rows = []
        for t in thresholds:
            m = self.metrics(probs >= t, pairing)
            m["threshold"] = float(t)
            rows.append(m)
        return rows


def best_row(rows, key="pxe"):
    """Row with the largest value of `key` (default: purity x efficiency)."""
    return max(rows, key=lambda r: r[key])


def print_scan(rows, pairing, baseline=None):
    """Print a scan as a table. `baseline` is a metrics dict for the cut-based
    selection, appended as a final row for comparison."""
    print(f"pairing = {pairing}")
    print(f"{'thr':>6}{'TP':>8}{'FP':>8}{'purity':>10}{'eff':>9}"
          f"{'P x E':>9}{'ev==2':>9}{'ev>=3':>9}")
    print("-" * 68)
    for r in rows:
        print(f"{r['threshold']:>6.2f}{r['TP']:>8,}{r['FP']:>8,}"
              f"{r['purity']:>9.1f}%{r['efficiency']:>8.2f}%{r['pxe']:>8.2f}%"
              f"{r['ev_eq2']:>9,}{r['ev_ge3']:>9,}")
    b = best_row(rows)
    print(f"\nbest pi0 P x E: t={b['threshold']:.2f}  purity {b['purity']:.1f}%  "
          f"efficiency {b['efficiency']:.2f}%  P x E {b['pxe']:.2f}%")
    if baseline is not None:
        print(f"cut-based     : t=  --  purity {baseline['purity']:.1f}%  "
              f"efficiency {baseline['efficiency']:.2f}%  P x E {baseline['pxe']:.2f}%")


# ---------------------------------------------------------------------------
# Persistence (with provenance, so two scans can be compared safely)
# ---------------------------------------------------------------------------

SCAN_FIELDS = ["threshold", "TP", "FP", "FN", "purity", "efficiency", "pxe",
               "ev_eq2", "ev_ge3"]

# Metadata keys that must agree before two scans may be compared.
PROVENANCE_KEYS = ["photon_predictions", "pfo_level_threshold", "mass_cut_MeV",
                   "phi_cut_deg", "n_true_pi0", "n_events", "n_pfos"]


def save_scan(rows, pairing, meta, out_dir, baseline=None):
    """Write a scan to `out_dir/pi0_scan_{pairing}.csv`.

    Metadata rows are written first as `# key, value`, so a later comparison
    can verify both scans describe the same models, cuts and sample. Without
    this, retraining the photon classifier would silently produce two tables
    that look comparable but are not.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"pi0_scan_{pairing}.csv"
    meta = {"written": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "pairing": pairing, **meta}

    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        for k, v in meta.items():
            wr.writerow([f"# {k}", v])
        wr.writerow(SCAN_FIELDS)
        for r in rows:
            wr.writerow([f"{r[k]:.6g}" if isinstance(r[k], float) else r[k]
                         for k in SCAN_FIELDS])
        if baseline is not None:
            wr.writerow(["cut_baseline"] +
                        [f"{baseline[k]:.6g}" if isinstance(baseline[k], float) else baseline[k]
                         for k in SCAN_FIELDS[1:]])
    return path


def load_scan(pairing, out_dir):
    """Read back a saved scan. Returns (meta, rows, baseline_or_None)."""
    path = out_dir / f"pi0_scan_{pairing}.csv"
    meta, rows, baseline = {}, [], None
    with open(path) as f:
        for r in csv.reader(f):
            if not r:
                continue
            if r[0].startswith("#"):
                meta[r[0].lstrip("# ").strip()] = r[1]
            elif r[0] == SCAN_FIELDS[0]:
                continue                                   # header
            elif r[0] == "cut_baseline":
                baseline = dict(zip(SCAN_FIELDS[1:], [float(x) for x in r[1:]]))
            else:
                rows.append(dict(zip(SCAN_FIELDS, [float(x) for x in r])))
    return meta, rows, baseline


def check_provenance(meta_a, meta_b, name_a="a", name_b="b"):
    """Warn if two scans describe different setups. Returns True if consistent."""
    bad = [k for k in PROVENANCE_KEYS if meta_a.get(k) != meta_b.get(k)]
    if bad:
        print("WARNING - these scans do not describe the same setup:")
        for k in bad:
            print(f"  {k}: {name_a}={meta_a.get(k)!r}  {name_b}={meta_b.get(k)!r}")
        print("  Re-run both before comparing.\n")
        return False
    return True


# ---------------------------------------------------------------------------
# Topology classification
# ---------------------------------------------------------------------------

def classify_topology(pi_plus, gamma, pi_zero):
    """Map reconstructed particle counts onto an interaction topology.

    Definitions follow the thesis:
      absorption             : no charged pions, no photons
      charge exchange        : no charged pions, and either a reconstructed pi0
                               from two photons, or a single photon (one decay
                               photon was missed)
      single pion production : exactly one charged pion, no photons
      pion production        : anything with extra charged pions or photons

    The `gamma == 1` branch matters: with a low pi0 efficiency most charge
    exchange is tagged through it rather than through a real pi0, so the CEX
    row is largely insensitive to pi0 reconstruction quality until that
    efficiency improves.
    """
    if pi_plus == 0 and gamma == 0:
        return 'absorption'
    if pi_plus == 0 and ((gamma == 2 and pi_zero == 1) or gamma == 1):
        return 'charge_exchange'
    if pi_plus == 1 and gamma == 0:
        return 'single_pion_production'
    if pi_plus > 1 or (pi_plus > 0 and gamma == 2 and pi_zero == 1) or gamma > 2:
        return 'pion_production'
    return 'other'


TRUE_ORDER  = ['pion_production', 'single_pion_production', 'charge_exchange', 'absorption']
RECO_ORDER  = ['absorption', 'charge_exchange', 'single_pion_production', 'pion_production']
TRUE_LABELS = ['pion production', 'single pion\nproduction', 'charge exchange', 'absorption']
RECO_LABELS = ['absorption', 'charge exchange', 'single pion\nproduction', 'pion production']

# Cells to outline in black: the (true, reco) pairs that agree.
DIAG_CELLS = [(i, RECO_ORDER.index(t)) for i, t in enumerate(TRUE_ORDER) if t in RECO_ORDER]


def get_matrix(joint, mode='purity'):
    """Turn a Counter of (true_topo, reco_topo) -> count into a matrix.

    mode='purity' : colour by column-normalised value (of everything tagged X,
                    how much really was X)
    mode='pxe'    : colour by purity x efficiency, i.e. column-norm * row-norm

    Returns (raw counts, colour values, row totals, column totals).
    """
    matrix = np.array([[joint[(t, r)] for r in RECO_ORDER] for t in TRUE_ORDER])
    col_totals = matrix.sum(axis=0)
    row_totals = np.array([sum(v for (t, _), v in joint.items() if t == to)
                           for to in TRUE_ORDER])
    col_norm = matrix / col_totals[None, :].clip(min=1)
    if mode == 'purity':
        return matrix, col_norm, row_totals, col_totals
    row_norm = matrix / row_totals[:, None].clip(min=1)
    return matrix, col_norm * row_norm * 100, row_totals, col_totals