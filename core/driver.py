import time

import numpy as np
from pyscf import fci
from pyscf.fci import selected_ci
from qiskit_addon_sqd.fermion import (
    bitstring_matrix_to_ci_strs,
    enlarge_batch_from_transitions,
    postselect_by_hamming_right_and_left,
    recover_configurations,
    solve_fermion,
    subsample,
)

from .config import bind
from .arrays import det_strings, fix_ci_strs, safe_array
from .cipsi_en import build_sd_transitions, cipsi_extend
from . import integrals
from .labeling import (assign_states, debug_overlap_matrix,
                       label_states_by_overlap, label_states_fci)
from .sampling import sample_bsm as sample_ucj_bsm
from .solver import kernel_safe
from .spin import (compute_state_marginals, get_alpha_coeffs,
                   get_beta_coeffs)


# Scan parameters, bound from the input file before compute_point() runs.
# The function bodies below read them as module globals, exactly as they
# did in the notebook, so that nothing inside them had to be rewritten.
NORB = None
NELEC = None
TARGET_STATES = None
N_SHOTS = None
N_SEEDS = None
NOISE_LEVEL = None
S_CORE_ITER = None
N_BATCHES = None
SAMPLES_PER_BATCH = None
MERGE_TOP_K = None
NROOTS_INT = None
NROOTS_GROW = None
OVERLAP_FLOOR = None
CIPSI_ACUT = None
CIPSI_BROAD_EPS = None
CIPSI_EN_EPS = None
CIPSI_MAX_ITER = None
CIPSI_EXT_ROOTS = None
CIPSI_BROAD_MULT = None
CIPSI_BROAD_FRAC = None
LABELING = None
OPEN_SHELL = None
CFG = None


def compute_point(R, system, REF_NROOTS=None):
    # system supplies the geometry; everything else comes from the input
    # file, bound into this module by config.bind().
    if REF_NROOTS is None:
        REF_NROOTS = NROOTS_INT
    print(f"\n  R = {R:.2f} A", end="", flush=True)
    t0_all = time.time()
    na, nb = NELEC

    _built = integrals.build(system, R, CFG)
    h1, eri, e_core = _built["h1"], _built["eri"], _built["e_core"]
    t1, t2 = _built["t1"], _built["t2"]
    strsa, strsb = det_strings(NORB, NELEC)
    n_alpha_full, n_beta_full = len(strsa), len(strsb)
    full_a_map = {int(s): i for i, s in enumerate(strsa)}
    full_b_map = {int(s): i for i, s in enumerate(strsb)}

    # --- Reference (FCI) ---
    t0 = time.time()
    solver = fci.direct_spin1.FCI()
    solver.conv_tol = 1e-10; solver.max_cycle = 200; solver.max_memory = 8000
    e_fci, c_fci = solver.kernel(h1, eri, NORB, NELEC, nroots=REF_NROOTS)
    if np.isscalar(e_fci):
        e_fci, c_fci = [float(e_fci)], [c_fci]
    fci_labeled, fci_s2, fci_idx_map = label_states_fci(
        solver, e_fci, c_fci, NORB, NELEC, e_core)
    dt_fci = time.time() - t0

    # === SQD Pipeline ===
    t0 = time.time()
    bsm_parts, probs_parts = [], []
    for s_idx in range(N_SEEDS):
        bsm_s, probs_s = sample_ucj_bsm(
            NORB, NELEC, t1, t2, N_SHOTS, NOISE_LEVEL, seed=42+s_idx*1000)
        bsm_parts.append(bsm_s); probs_parts.append(probs_s)
    bsm_raw = np.vstack(bsm_parts)
    probs_raw = np.concatenate(probs_parts); probs_raw /= probs_raw.sum()

    bsm_ps, probs_ps = postselect_by_hamming_right_and_left(
        bsm_raw, probs_raw, hamming_right=na, hamming_left=nb)
    for it in range(S_CORE_ITER):
        energy_sqd, sci_state, avg_occs, spin_sq = solve_fermion(
            bsm_ps, hcore=h1, eri=eri, open_shell=OPEN_SHELL)
        bsm_ps, probs_ps = recover_configurations(
            bsm_ps, probs_ps, avg_occs, na, nb, rand_seed=42+it)

    spb = min(SAMPLES_PER_BATCH, len(bsm_ps))
    batches = subsample(bsm_ps, probs_ps, spb, N_BATCHES, rand_seed=42)
    batch_energies = []
    for i, batch in enumerate(batches):
        e_b, _, _, _ = solve_fermion(batch, hcore=h1, eri=eri, open_shell=OPEN_SHELL)
        batch_energies.append((e_b, i))
    batch_energies.sort()
    merged_rows = []
    for _, idx in batch_energies[:MERGE_TOP_K]:
        merged_rows.append(batches[idx])
    merged_batch = np.vstack(merged_rows)
    _, unique_idx = np.unique(merged_batch, axis=0, return_index=True)
    merged_batch = merged_batch[np.sort(unique_idx)]

    sqd_a, sqd_b = bitstring_matrix_to_ci_strs(merged_batch, open_shell=OPEN_SHELL)
    sqd_a = fix_ci_strs(sqd_a, NORB); sqd_b = fix_ci_strs(sqd_b, NORB)
    dt_sqd = time.time() - t0

    # === Ext-SQD ===
    t0 = time.time()
    ops = build_sd_transitions(NORB, NELEC)
    ext_bsm = enlarge_batch_from_transitions(merged_batch, ops)
    ext_a, ext_b = bitstring_matrix_to_ci_strs(ext_bsm, open_shell=OPEN_SHELL)
    ext_a = fix_ci_strs(ext_a, NORB); ext_b = fix_ci_strs(ext_b, NORB)

    myci = selected_ci.SelectedCI()
    myci.conv_tol = 1e-10; myci.max_cycle = 200; myci.max_memory = 8000
    ci_ext = (ext_a, ext_b)
    nr_ext = min(NROOTS_INT, len(ext_a) * len(ext_b))
    el_ext, cl_ext = kernel_safe(myci, h1, eri, NORB, NELEC, ci_ext, nr_ext)

    # === [PATCH 4a] Ext-SQD labeling: energy-order -> FCI overlap-based ===
    ext_labeled, ext_s2, ext_idx_map = assign_states(
        LABELING, myci,
        
        c_var_list    = cl_ext,
        e_var_list    = el_ext,
        ci_strs_var   = (ext_a, ext_b),
        c_fci_list    = c_fci,
        fci_idx_map   = fci_idx_map,
        strsa_full    = strsa,
        strsb_full    = strsb,
        norb          = NORB,
        nelec         = NELEC,
        e_core        = e_core,
        target_states = TARGET_STATES,
        overlap_floor = OVERLAP_FLOOR,
        verbose       = True,
    )
    dt_ext = time.time() - t0

    # === CIPSI: seed → broad → EN ===
    t0 = time.time()
    seed_a = np.array(sorted(set(int(x) for x in sqd_a)), np.int64)
    seed_b = np.array(sorted(set(int(x) for x in sqd_b)), np.int64)
    basis_a, basis_b = seed_a.copy(), seed_b.copy()
    cipsi_convergence = []

    ci_seed = (seed_a, seed_b)
    e_seed, c_seed = kernel_safe(myci, h1, eri, NORB, NELEC, ci_seed,
                                  min(NROOTS_GROW, len(seed_a)*len(seed_b)), loose=True)

    all_a = set(int(x) for x in seed_a)
    all_b = set(int(x) for x in seed_b)
    if e_seed:
        for cv in c_seed[:min(CIPSI_EXT_ROOTS, len(c_seed))]:
            sca = get_alpha_coeffs(cv, strsa, seed_a, ci_seed)
            ext2_a = cipsi_extend(seed_a, sca, NORB, na, CIPSI_ACUT, 0, h1, eri,
                               CIPSI_BROAD_EPS*CIPSI_BROAD_MULT, use_coeffs=False, broad_frac=CIPSI_BROAD_FRAC)
            all_a |= set(int(x) for x in ext2_a)
            scb = get_beta_coeffs(cv, strsb, seed_b, ci_seed)
            ext2_b = cipsi_extend(seed_b, scb, NORB, nb, CIPSI_ACUT, 0, h1, eri,
                               CIPSI_BROAD_EPS*CIPSI_BROAD_MULT, use_coeffs=False, broad_frac=CIPSI_BROAD_FRAC)
            all_b |= set(int(x) for x in ext2_b)
    basis_a = np.array(sorted(all_a), np.int64)
    basis_b = np.array(sorted(all_b), np.int64)
    cipsi_convergence.append({"iter": 0, "D_a": len(basis_a), "D_b": len(basis_b), "mode": "seed-broad"})

    d_broad_a, d_broad_b = len(basis_a), len(basis_b)
    ci_broad = (basis_a, basis_b)
    nr_broad = min(NROOTS_INT, len(basis_a)*len(basis_b))
    el_broad, cl_broad = kernel_safe(myci, h1, eri, NORB, NELEC, ci_broad, nr_broad)

    # === Broad-stage labeling also overlap-based (accurate progress output) ===
    broad_labeled, broad_s2, _ = assign_states(
        LABELING, myci,
        
        c_var_list    = cl_broad,
        e_var_list    = el_broad,
        ci_strs_var   = (basis_a, basis_b),
        c_fci_list    = c_fci,
        fci_idx_map   = fci_idx_map,
        strsa_full    = strsa,
        strsb_full    = strsb,
        norb          = NORB,
        nelec         = NELEC,
        e_core        = e_core,
        target_states = TARGET_STATES,
        overlap_floor = OVERLAP_FLOOR,
        verbose       = True,
    )

    _tr = cipsi_convergence[-1] if cipsi_convergence else None
    if _tr is not None and (_tr["D_a"], _tr["D_b"]) == (len(basis_a), len(basis_b)):
        _tr["energies"] = broad_labeled
        _tr["s2"] = broad_s2

    print(f"\n    [BROAD] D=({len(basis_a)},{len(basis_b)})")
    for lb in TARGET_STATES:
        eb, er = broad_labeled.get(lb), fci_labeled.get(lb)
        if eb is not None and er is not None:
            print(f"      {lb}: ΔE={((eb-er)*1000):+.4f} mHa")

    for it in range(1, CIPSI_MAX_ITER):
        ci_s2 = (basis_a, basis_b)
        nr2 = min(NROOTS_GROW, len(basis_a)*len(basis_b))
        e2, c2 = kernel_safe(myci, h1, eri, NORB, NELEC, ci_s2, nr2, loose=True)
        if not e2: break

        old_a, old_b = len(basis_a), len(basis_b)
        all_a = set(int(x) for x in basis_a)
        all_b = set(int(x) for x in basis_b)
        n_roots_use = min(CIPSI_EXT_ROOTS, len(c2))
        e0_list = [float(e2[ri]) for ri in range(n_roots_use)]

        ac_list = [get_alpha_coeffs(c2[ri], strsa, basis_a, ci_s2) for ri in range(n_roots_use)]
        ext2_a = cipsi_extend(basis_a, ac_list, NORB, na, CIPSI_ACUT, 0, h1, eri,
                           CIPSI_EN_EPS, use_coeffs=True, broad_frac=CIPSI_BROAD_FRAC, en_mode=True, e0=e0_list)
        all_a |= set(int(x) for x in ext2_a)

        bc_list = [get_beta_coeffs(c2[ri], strsb, basis_b, ci_s2) for ri in range(n_roots_use)]
        ext2_b = cipsi_extend(basis_b, bc_list, NORB, nb, CIPSI_ACUT, 0, h1, eri,
                           CIPSI_EN_EPS, use_coeffs=True, broad_frac=CIPSI_BROAD_FRAC, en_mode=True, e0=e0_list)
        all_b |= set(int(x) for x in ext2_b)

        # Broad supplement on top configs
        max_coeff_a = np.zeros(len(basis_a))
        max_coeff_b = np.zeros(len(basis_b))
        for ri in range(n_roots_use):
            max_coeff_a = np.maximum(max_coeff_a, np.abs(get_alpha_coeffs(c2[ri], strsa, basis_a, ci_s2)))
            max_coeff_b = np.maximum(max_coeff_b, np.abs(get_beta_coeffs(c2[ri], strsb, basis_b, ci_s2)))

        for top_basis, top_coeffs, all_set in [(basis_a, max_coeff_a, all_a), (basis_b, max_coeff_b, all_b)]:
            top_n = max(10, int(len(top_basis)*CIPSI_BROAD_FRAC))
            top_idx = np.argsort(-top_coeffs)[:top_n]
            for s in top_basis[top_idx]:
                s_int = int(s)
                occ = [i for i in range(NORB) if (s_int >> i) & 1]
                vir = [i for i in range(NORB) if not ((s_int >> i) & 1)]
                for i in occ:
                    for a in vir:
                        all_set.add(s_int ^ (1 << i) ^ (1 << a))

        _tr = cipsi_convergence[-1] if cipsi_convergence else None
        if it > 1 and _tr is not None and \
                (_tr["D_a"], _tr["D_b"]) == (len(basis_a), len(basis_b)):
            try:
                _lab, _s2, _ = assign_states(
                    LABELING, myci,
                    c_var_list    = c2,
                    e_var_list    = e2,
                    ci_strs_var   = ci_s2,
                    c_fci_list    = c_fci,
                    fci_idx_map   = fci_idx_map,
                    strsa_full    = strsa,
                    strsb_full    = strsb,
                    norb          = NORB,
                    nelec         = NELEC,
                    e_core        = e_core,
                    target_states = TARGET_STATES,
                    overlap_floor = OVERLAP_FLOOR,
                    verbose       = False,
                )
                _tr["energies"] = _lab
                _tr["s2"] = _s2
            except Exception as exc:
                print(f"    [trace] iter {it}: labeling skipped "
                      f"({type(exc).__name__}: {exc})", flush=True)

        basis_a = np.array(sorted(all_a), np.int64)
        basis_b = np.array(sorted(all_b), np.int64)
        cipsi_convergence.append({"iter": it, "D_a": len(basis_a), "D_b": len(basis_b), "mode": "EN+broad"})
        if len(basis_a) == old_a and len(basis_b) == old_b: break

    ci_cipsi = (basis_a, basis_b)
    nr_cipsi = min(NROOTS_INT, len(basis_a)*len(basis_b))
    el_cipsi, cl_cipsi = kernel_safe(myci, h1, eri, NORB, NELEC, ci_cipsi, nr_cipsi)

    # === CIPSI final labeling: energy-order -> FCI overlap-based ===
    cipsi_labeled, cipsi_s2, cipsi_idx_map = assign_states(
        LABELING, myci,
        
        c_var_list    = cl_cipsi,
        e_var_list    = el_cipsi,
        ci_strs_var   = (basis_a, basis_b),
        c_fci_list    = c_fci,
        fci_idx_map   = fci_idx_map,
        strsa_full    = strsa,
        strsb_full    = strsb,
        norb          = NORB,
        nelec         = NELEC,
        e_core        = e_core,
        target_states = TARGET_STATES,
        overlap_floor = OVERLAP_FLOOR,
        verbose       = True,         # Verbose output for all labeling stages
    )
    _tr = cipsi_convergence[-1] if cipsi_convergence else None
    if _tr is not None and (_tr["D_a"], _tr["D_b"]) == (len(basis_a), len(basis_b)):
        _tr["energies"] = cipsi_labeled
        _tr["s2"] = cipsi_s2

    dt_cipsi = time.time() - t0
    cipsi_convergence.append({"iter": "final", "D_a": len(basis_a), "D_b": len(basis_b),
                              "mode": "final", "energies": cipsi_labeled, "s2": cipsi_s2})

    print(f"    [EN] Broad D=({d_broad_a},{d_broad_b}) → Final D=({len(basis_a)},{len(basis_b)})")
    for lb in TARGET_STATES:
        eb, ef, er = broad_labeled.get(lb), cipsi_labeled.get(lb), fci_labeled.get(lb)
        if eb is not None and ef is not None and er is not None:
            print(f"      {lb}: broad={((eb-er)*1000):+.4f} → final={((ef-er)*1000):+.4f} mHa")
        elif ef is None and er is not None:
            print(f"      {lb}: UNLABELED (overlap² < {OVERLAP_FLOOR})")

    # === [PATCH 5] Debug: all variational root energies + (label, k) overlap matrix ===
    all_var_roots = {
        "ext_energies":      [float(e + e_core) for e in el_ext],
        "cipsi_energies":    [float(e + e_core) for e in el_cipsi],
        "ext_overlap_sq":    debug_overlap_matrix(
            cl_ext, (ext_a, ext_b), c_fci, fci_idx_map,
            strsa, strsb, TARGET_STATES),
        "cipsi_overlap_sq":  debug_overlap_matrix(
            cl_cipsi, (basis_a, basis_b), c_fci, fci_idx_map,
            strsa, strsb, TARGET_STATES),
    }

    # === Natural orbital occupation (GS) ===
    nat_occ = {}
    try:
        if len(c_fci) > 0:
            rdm1_ref = fci.direct_spin1.make_rdm1(c_fci[0], NORB, NELEC)
            nat_occ["ref"] = np.sort(np.linalg.eigvalsh(rdm1_ref))[::-1].tolist()
        if len(cl_ext) > 0:
            myci._strs = ci_ext
            rdm1_ext = myci.make_rdm1(cl_ext[0], NORB, NELEC)
            nat_occ["ext"] = np.sort(np.linalg.eigvalsh(rdm1_ext))[::-1].tolist()
        if len(cl_cipsi) > 0:
            myci._strs = ci_cipsi
            rdm1_cipsi = myci.make_rdm1(cl_cipsi[0], NORB, NELEC)
            nat_occ["cipsi"] = np.sort(np.linalg.eigvalsh(rdm1_cipsi))[::-1].tolist()
    except Exception as e:
        print(f"\n    [NO] Failed: {e}")

    # === Diagnostics ===
    ext_set_a = set(int(x) for x in ext_a)
    cipsi_set_a = set(int(x) for x in basis_a)
    missing_a = sorted(ext_set_a - cipsi_set_a)
    diag = {"n_missing_a": len(missing_a), "n_ext_a": len(ext_set_a), "n_cipsi_a": len(cipsi_set_a),
            "n_ext_b": len(ext_b), "n_cipsi_b": len(basis_b)}

    # WF analysis (GS)
    wf_analysis = {}
    cr_ext = None; ext_str_map_a = {}
    if len(cl_ext) > 0:
        cr_ext = np.asarray(cl_ext[0], float)
        if cr_ext.ndim == 2:
            ext_str_map_a = {int(s): i for i, s in enumerate(ext_a)}
            alpha_weights = {}
            for ia, s in enumerate(ext_a):
                if ia < cr_ext.shape[0]:
                    alpha_weights[int(s)] = float(np.sum(cr_ext[ia, :]**2))
            total_weight = sum(alpha_weights.values())
            cipsi_weight = sum(alpha_weights.get(s, 0) for s in cipsi_set_a)
            ext_only_s = ext_set_a - cipsi_set_a
            ext_only_w = sum(alpha_weights.get(s, 0) for s in ext_only_s)
            ext_only_max = max((alpha_weights.get(s, 0) for s in ext_only_s), default=0)
            wf_analysis = {
                "ext_recovery_pct": 100.0, "cipsi_recovery_pct": 100.0*cipsi_weight/max(total_weight, 1e-20),
                "ext_per_config": total_weight/max(len(ext_set_a), 1),
                "cipsi_per_config": cipsi_weight/max(len(cipsi_set_a), 1),
                "categories": {"ext_only": {"n": len(ext_only_s), "pct": 100.0*ext_only_w/max(total_weight, 1e-20), "max": ext_only_max}},
            }
            if wf_analysis["ext_per_config"] > 0:
                wf_analysis["efficiency_ratio"] = wf_analysis["cipsi_per_config"]/wf_analysis["ext_per_config"]
            avg_ext_only = ext_only_w/max(len(ext_only_s), 1)
            if avg_ext_only > 0:
                wf_analysis["selectivity_ratio"] = wf_analysis["cipsi_per_config"]/avg_ext_only

            def cum_curve(config_set, weights, max_pts=100):
                ws = sorted([weights.get(s, 0) for s in config_set], reverse=True)
                cum = np.cumsum(ws)
                if len(cum) == 0: return [], []
                cum_pct = (cum/max(total_weight, 1e-20)*100).tolist()
                if len(cum_pct) <= max_pts: return list(range(1, len(cum_pct)+1)), cum_pct
                step = max(1, len(cum_pct)//max_pts)
                idx = list(range(0, len(cum_pct), step))
                if idx[-1] != len(cum_pct)-1: idx.append(len(cum_pct)-1)
                return [i+1 for i in idx], [cum_pct[i] for i in idx]
            full_set_a = set(int(s) for s in strsa)
            wf_analysis["opt_curve"] = dict(zip(["x","y"], cum_curve(full_set_a, alpha_weights)))
            wf_analysis["ext_curve"] = dict(zip(["x","y"], cum_curve(ext_set_a, alpha_weights)))
            wf_analysis["cipsi_curve"] = dict(zip(["x","y"], cum_curve(cipsi_set_a, alpha_weights)))

    # === State-resolved alpha/beta marginals ===
    wf_amps = {}
    try:
        for lb in TARGET_STATES:
            if lb not in fci_labeled: continue
            entry = {}
            fi = fci_idx_map.get(lb)
            if fi is not None and fi < len(c_fci):
                c_r = np.asarray(c_fci[fi], float)
                if c_r.ndim == 2:
                    pa_r, pb_r = np.sum(c_r**2, axis=1), np.sum(c_r**2, axis=0)
                    if len(pa_r) == n_alpha_full and len(pb_r) == n_beta_full:
                        entry["ref_a"], entry["ref_b"] = pa_r.tolist(), pb_r.tolist()
                    else:
                        pa_full, pb_full = np.zeros(n_alpha_full), np.zeros(n_beta_full)
                        pa_full[:len(pa_r)] = pa_r; pb_full[:len(pb_r)] = pb_r
                        entry["ref_a"], entry["ref_b"] = pa_full.tolist(), pb_full.tolist()
            ei = ext_idx_map.get(lb)
            if ei is not None and ei < len(cl_ext):
                pa_e, pb_e = compute_state_marginals(cl_ext[ei], ext_a, ext_b, full_a_map, full_b_map, n_alpha_full, n_beta_full)
                if pa_e is not None: entry["ext_a"], entry["ext_b"] = pa_e.tolist(), pb_e.tolist()
            ci_i = cipsi_idx_map.get(lb)
            if ci_i is not None and ci_i < len(cl_cipsi):
                pa_c, pb_c = compute_state_marginals(cl_cipsi[ci_i], basis_a, basis_b, full_a_map, full_b_map, n_alpha_full, n_beta_full)
                if pa_c is not None: entry["cipsi_a"], entry["cipsi_b"] = pa_c.tolist(), pb_c.tolist()
            if entry: wf_amps[lb] = entry
        if wf_amps:
            print(f"\n    [AMP] States: {list(wf_amps.keys())}, α:{n_alpha_full}, β:{n_beta_full}")
    except Exception as e:
        print(f"\n    [AMP] Failed: {e}")

    wf_amps_meta = {"n_alpha_full": n_alpha_full, "n_beta_full": n_beta_full, "states_stored": list(wf_amps.keys())}

    # Important missing
    n_important, max_missing_w = 0, 0.0
    if cr_ext is not None and cr_ext.ndim == 2:
        for s in missing_a:
            ia = ext_str_map_a.get(s)
            if ia is not None and ia < cr_ext.shape[0]:
                w = float(np.max(np.abs(cr_ext[ia, :])))
                max_missing_w = max(max_missing_w, w)
                if w > 0.01: n_important += 1
    diag["n_important_missing"] = n_important; diag["max_missing_weight"] = max_missing_w

    if len(missing_a) > 0:
        print(f"\n    [DIAG] Missing alpha: {len(missing_a)}/{len(ext_set_a)} (important: {n_important})")
    conv_str = " → ".join(f"({c['D_a']},{c['D_b']})" for c in cipsi_convergence)
    print(f"\n    [CONV] {conv_str}")
    if wf_analysis:
        eo = wf_analysis.get("categories", {}).get("ext_only", {})
        print(f"    [WF] Ext: {wf_analysis['ext_recovery_pct']:.2f}% | CIPSI: {wf_analysis['cipsi_recovery_pct']:.2f}% | "
              f"eff: {wf_analysis.get('efficiency_ratio', 0):.1f}x | ext-only: {eo.get('n', 0)} ({eo.get('pct', 0):.4f}%)")
    dim_ext = len(ext_a) * len(ext_b)
    dim_cipsi = len(basis_a) * len(basis_b)
    print(f"  Ref:{dt_fci:.0f}s  SQD({len(sqd_a)},{len(sqd_b)}):{dt_sqd:.0f}s  "
          f"Ext({len(ext_a)},{len(ext_b)}):{dt_ext:.0f}s  CIPSI({len(basis_a)},{len(basis_b)}):{dt_cipsi:.0f}s  "
          f"[{time.time()-t0_all:.0f}s]", flush=True)

    return {
        "R": R, "fci": fci_labeled, "ext_sqd": ext_labeled, "cipsi": cipsi_labeled,
        "fci_s2": fci_s2, "ext_s2": ext_s2, "cipsi_s2": cipsi_s2,
        "dims": {"sqd_a": len(sqd_a), "sqd_b": len(sqd_b),
                 "ext_a": len(ext_a), "ext_b": len(ext_b), "ext": dim_ext,
                 "cipsi_a": len(basis_a), "cipsi_b": len(basis_b), "cipsi": dim_cipsi,
                 "shots": N_SHOTS*N_SEEDS},
        "diag": diag, "cipsi_convergence": cipsi_convergence, "wf_analysis": wf_analysis,
        "nat_occ": nat_occ, "wf_amps": wf_amps, "wf_amps_meta": wf_amps_meta,
        "all_var_roots": all_var_roots,                                   
        "strings": {"sqd_a": [int(x) for x in sqd_a],
                    "sqd_b": [int(x) for x in sqd_b],
                    "ext_a": [int(x) for x in ext_a],
                    "ext_b": [int(x) for x in ext_b],
                    "cipsi_a": [int(x) for x in basis_a],
                    "cipsi_b": [int(x) for x in basis_b]},
    }
