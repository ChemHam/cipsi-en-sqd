import numpy as np
from pyscf.fci import selected_ci

from .config import bind
from .spin import MULTIPLICITY, SPIN_TAGS, classify_spin


# Scan parameters, bound from the input file before compute_point() runs.
# The function bodies below read them as module globals, exactly as they
# did in the notebook, so that nothing inside them had to be rewritten.
MAX_DOUBLET = None
MAX_QUARTET = None
MAX_LOW = None
MAX_HIGH = None


def label_states_fci(solver, e_list, c_list, norb, nelec, e_core):
    low_tag, high_tag = SPIN_TAGS
    low_pre, high_pre = MULTIPLICITY[low_tag][2], MULTIPLICITY[high_tag][2]
    labeled, s2_dict, idx_map = {}, {}, {}
    n_low, n_high = 0, 0
    for k in range(len(e_list)):
        s2, _ = solver.spin_square(c_list[k], norb, nelec)
        s2 = float(s2); tag = classify_spin(s2)
        if tag == low_tag and n_low < MAX_LOW:
            lb = f"{low_pre}{n_low}"; n_low += 1
        elif tag == high_tag and n_high < MAX_HIGH:
            lb = f"{high_pre}{n_high+1}"; n_high += 1
        else: continue
        labeled[lb] = float(e_core + e_list[k]); s2_dict[lb] = s2; idx_map[lb] = k
        if n_low >= MAX_LOW and n_high >= MAX_HIGH: break
    return labeled, s2_dict, idx_map



def _project_var_to_full_indices(var_strs, full_str_to_idx):
    """Map each variational-space string to its full-space index (-1 if absent)."""
    out = np.full(len(var_strs), -1, dtype=np.int64)
    for i, s in enumerate(var_strs):
        j = full_str_to_idx.get(int(s))
        if j is not None:
            out[i] = j
    return out


def _overlap_var_with_fci(c_var, c_fci_full, a_idx, b_idx):
    """<FCI|var> inner product. If all variational indices are valid, single advanced-indexing pass."""
    if c_var is None: return 0.0
    cv = np.asarray(c_var, float)
    if cv.ndim != 2: return 0.0
    na_var, nb_var = cv.shape
    if na_var > len(a_idx) or nb_var > len(b_idx): return 0.0
    aix = a_idx[:na_var]; bix = b_idx[:nb_var]
    if (aix < 0).any() or (bix < 0).any():
        # Only some are valid: extract the valid sub-block
        va = np.where(aix >= 0)[0]; vb = np.where(bix >= 0)[0]
        if len(va) == 0 or len(vb) == 0: return 0.0
        sub_fci = c_fci_full[np.ix_(aix[va], bix[vb])]
        sub_var = cv[np.ix_(va, vb)]
        return float(np.sum(sub_fci * sub_var))
    sub_fci = c_fci_full[np.ix_(aix, bix)]
    return float(np.sum(sub_fci * cv))


def label_states_by_overlap(c_var_list, e_var_list, ci_strs_var,
                             c_fci_list, fci_idx_map,
                             strsa_full, strsb_full,
                             norb, nelec, e_core, target_states,
                             overlap_floor=0.30, verbose=False):
    var_a, var_b = ci_strs_var
    full_a_map = {int(s): i for i, s in enumerate(strsa_full)}
    full_b_map = {int(s): i for i, s in enumerate(strsb_full)}
    a_idx = _project_var_to_full_indices(var_a, full_a_map)
    b_idx = _project_var_to_full_indices(var_b, full_b_map)

    pairs = []  # (overlap_sq, label, k)
    for lb in target_states:
        fi = fci_idx_map.get(lb)
        if fi is None or fi >= len(c_fci_list): continue
        c_fci = np.asarray(c_fci_list[fi], float)
        if c_fci.ndim != 2: continue
        if c_fci.shape != (len(strsa_full), len(strsb_full)): continue
        for k, c_var in enumerate(c_var_list):
            ov = _overlap_var_with_fci(c_var, c_fci, a_idx, b_idx)
            pairs.append((ov*ov, lb, k))
    pairs.sort(reverse=True)

    labeled, s2_dict, idx_map = {}, {}, {}
    used_var, used_lb = set(), set()
    for ov_sq, lb, k in pairs:
        if lb in used_lb or k in used_var: continue
        if ov_sq < overlap_floor: continue
        labeled[lb] = float(e_core + e_var_list[k])
        idx_map[lb] = k
        used_lb.add(lb); used_var.add(k)
        try:
            s2 = float(selected_ci.spin_square(c_var_list[k], norb, nelec)[0])
        except Exception:
            s2 = -1.0
        s2_dict[lb] = s2
        if verbose:
            print(f"      [overlap-label] {lb} → var root {k}, "
                  f"|<FCI|var>|² = {ov_sq:.4f}, E = {labeled[lb]:.6f} Ha, S² = {s2:.4f}")
    if verbose:
        unmatched = [lb for lb in target_states if lb not in labeled]
        if unmatched:
            print(f"      [overlap-label] WARNING unmatched: {unmatched} "
                  f"(no var root with overlap² ≥ {overlap_floor})")
    return labeled, s2_dict, idx_map


def debug_overlap_matrix(c_var_list, ci_strs_var, c_fci_list, fci_idx_map,
                          strsa_full, strsb_full, target_states):
    """Return all (label x var root) overlap-squared values as a dict (for debug/SI)."""
    var_a, var_b = ci_strs_var
    full_a_map = {int(s): i for i, s in enumerate(strsa_full)}
    full_b_map = {int(s): i for i, s in enumerate(strsb_full)}
    a_idx = _project_var_to_full_indices(var_a, full_a_map)
    b_idx = _project_var_to_full_indices(var_b, full_b_map)
    out = {}
    for lb in target_states:
        fi = fci_idx_map.get(lb)
        if fi is None or fi >= len(c_fci_list): continue
        c_fci = np.asarray(c_fci_list[fi], float)
        if c_fci.ndim != 2: continue
        out[lb] = []
        for c_var in c_var_list:
            ov = _overlap_var_with_fci(c_var, c_fci, a_idx, b_idx)
            out[lb].append(ov*ov)
    return out


def label_states_sci(myci, e_list, c_list, norb, nelec, e_core):
    low_tag, high_tag = SPIN_TAGS
    low_pre = MULTIPLICITY[low_tag][2]
    high_pre = MULTIPLICITY[high_tag][2]
    labeled, s2_dict, idx_map = {}, {}, {}
    n_low, n_high = 0, 0
    for k in range(len(e_list)):
        try:
            s2 = float(selected_ci.spin_square(c_list[k], norb, nelec)[0])
        except Exception:
            s2 = -1.0
        tag = classify_spin(s2)
        if tag == low_tag and n_low < MAX_LOW:
            lb = f"{low_pre}{n_low}"; n_low += 1
        elif tag == high_tag and n_high < MAX_HIGH:
            lb = f"{high_pre}{n_high + 1}"; n_high += 1
        else:
            continue
        labeled[lb] = float(e_core + e_list[k])
        s2_dict[lb] = s2
        idx_map[lb] = k
        if n_low >= MAX_LOW and n_high >= MAX_HIGH:
            break
    return labeled, s2_dict, idx_map

def assign_states(mode, myci, **kw):
    if mode == "overlap":
        return label_states_by_overlap(
            kw["c_var_list"], kw["e_var_list"], kw["ci_strs_var"],
            kw["c_fci_list"], kw["fci_idx_map"],
            kw["strsa_full"], kw["strsb_full"],
            kw["norb"], kw["nelec"], kw["e_core"], kw["target_states"],
            kw.get("overlap_floor", 0.30), kw.get("verbose", False))
    if mode == "spin":
        return label_states_sci(myci, kw["e_var_list"], kw["c_var_list"],
                                kw["norb"], kw["nelec"], kw["e_core"])
    raise ValueError(f"unknown labeling {mode!r}")
