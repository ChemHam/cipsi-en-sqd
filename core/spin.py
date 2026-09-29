
import numpy as np
from pyscf import fci

MULTIPLICITY = {
    # tag: (target S^2, window, label prefix)
    "singlet": (0.00, 0.4, "S"),
    "triplet": (2.00, 0.5, "T"),
    "doublet": (0.75, 0.4, "D"),
    "quartet": (3.75, 0.5, "Q"),
}

# Which two a system uses. bind() sets this from the input file: a closed
# shell is labelled S0/T1, an open shell D0/Q1, and testing the wrong pair
# against S^2 puts every root in "other".
SPIN_TAGS = ("doublet", "quartet")


def classify_spin(s2):
    for tag in SPIN_TAGS:
        target, window, _ = MULTIPLICITY[tag]
        if abs(s2 - target) < window:
            return tag
    return "other"

def get_alpha_coeffs(c_full, strsa, basis, ci_strs):
    c = np.asarray(c_full)
    coeffs = np.zeros(len(basis))
    if c.ndim == 2:
        sm = {int(s): i for i, s in enumerate(ci_strs[0])}
        for bi, bs in enumerate(basis):
            ia = sm.get(int(bs))
            if ia is not None and ia < c.shape[0]:
                coeffs[bi] = np.max(np.abs(c[ia, :]))
    return coeffs


def get_beta_coeffs(c_full, strsb, basis_b, ci_strs):
    c = np.asarray(c_full)
    coeffs = np.zeros(len(basis_b))
    if c.ndim == 2:
        sm = {int(s): i for i, s in enumerate(ci_strs[1])}
        for bi, bs in enumerate(basis_b):
            ib = sm.get(int(bs))
            if ib is not None and ib < c.shape[1]:
                coeffs[bi] = np.max(np.abs(c[:, ib]))
    return coeffs


def compute_state_marginals(ci_vec, sub_a, sub_b, full_a_map, full_b_map,
                             n_alpha_full, n_beta_full):
    c = np.asarray(ci_vec, float)
    if c.ndim != 2: return None, None
    pa_sub = np.sum(c**2, axis=1)
    pb_sub = np.sum(c**2, axis=0)
    pa_full = np.zeros(n_alpha_full)
    pb_full = np.zeros(n_beta_full)
    for i_sub in range(min(len(sub_a), len(pa_sub))):
        i_full = full_a_map.get(int(sub_a[i_sub]))
        if i_full is not None: pa_full[i_full] = pa_sub[i_sub]
    for i_sub in range(min(len(sub_b), len(pb_sub))):
        i_full = full_b_map.get(int(sub_b[i_sub]))
        if i_full is not None: pb_full[i_full] = pb_sub[i_sub]
    return pa_full, pb_full
