"""Perturbative selection and the single-and-double subspace expansion.

Extracted verbatim from the production notebook pipeline. The function bodies
are byte-identical to the versions that produced the published scans; nothing
in this file has been retyped. Deliberate changes are marked FIX and listed in
CHANGES.md.
"""

import numpy as np
import itertools

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

if HAS_NUMBA:
    @njit(cache=True)
    def _in_sorted(val, arr):
        """Binary search in sorted int64 array."""
        lo, hi = 0, len(arr) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if arr[mid] == val: return True
            elif arr[mid] < val: lo = mid + 1
            else: hi = mid - 1
        return False

    @njit(cache=True)
    def _compute_diag_jit(s, norb, h1, eri):
        """JIT diagonal H element."""
        occ = np.empty(norb, dtype=np.int64); nocc = 0
        for i in range(norb):
            if (s >> i) & 1:
                occ[nocc] = i; nocc += 1
        e = 0.0
        for ii in range(nocc):
            i = occ[ii]; e += h1[i, i]
        for ii in range(nocc):
            i = occ[ii]
            for jj in range(ii+1, nocc):
                j = occ[jj]; e += eri[i, i, j, j] - eri[i, j, j, i]
        return e

    @njit(cache=True)
    def _en_couplings_one_config(s, ck, norb, h1, eri, bs_sorted):
        """JIT: compute S/D couplings from one source config.
        Returns (candidate_configs, coupling_values) arrays."""
        occ = np.empty(norb, dtype=np.int64); nocc = 0
        vir = np.empty(norb, dtype=np.int64); nvir = 0
        for i in range(norb):
            if (s >> i) & 1: occ[nocc] = i; nocc += 1
            else: vir[nvir] = i; nvir += 1

        max_n = nocc * nvir + (nocc * (nocc-1) // 2) * (nvir * (nvir-1) // 2)
        cands = np.empty(max_n, dtype=np.int64)
        coups = np.empty(max_n, dtype=np.float64)
        n = 0

        # Singles
        for ii in range(nocc):
            i = occ[ii]
            for aa in range(nvir):
                a = vir[aa]
                ns = s ^ (1 << i) ^ (1 << a)
                if _in_sorted(ns, bs_sorted): continue
                v = h1[a, i]
                for jj in range(nocc):
                    j_occ = occ[jj]
                    v += eri[a, j_occ, i, j_occ] - eri[a, j_occ, j_occ, i]
                cands[n] = ns; coups[n] = v * ck; n += 1

        # Doubles
        for ii in range(nocc):
            i = occ[ii]
            for jj in range(ii+1, nocc):
                j = occ[jj]
                for aa in range(nvir):
                    a = vir[aa]
                    for bb in range(aa+1, nvir):
                        b = vir[bb]
                        ns = s ^ (1 << i) ^ (1 << j) ^ (1 << a) ^ (1 << b)
                        if _in_sorted(ns, bs_sorted): continue
                        v = eri[a, i, b, j] - eri[a, j, b, i]
                        cands[n] = ns; coups[n] = v * ck; n += 1
        return cands[:n], coups[:n]

    @njit(cache=True)
    def _broad_one_config(s, norb, eri, eps, bs_sorted):
        """JIT: broad mode S/D from one config. Singles always added, doubles eps-filtered."""
        occ = np.empty(norb, dtype=np.int64); nocc = 0
        vir = np.empty(norb, dtype=np.int64); nvir = 0
        for i in range(norb):
            if (s >> i) & 1: occ[nocc] = i; nocc += 1
            else: vir[nvir] = i; nvir += 1

        max_n = nocc * nvir + (nocc * (nocc-1) // 2) * (nvir * (nvir-1) // 2)
        cands = np.empty(max_n, dtype=np.int64)
        n = 0

        for ii in range(nocc):
            i = occ[ii]
            for aa in range(nvir):
                a = vir[aa]
                ns = s ^ (1 << i) ^ (1 << a)
                if not _in_sorted(ns, bs_sorted):
                    cands[n] = ns; n += 1

        for ii in range(nocc):
            i = occ[ii]
            for jj in range(ii+1, nocc):
                j = occ[jj]
                for aa in range(nvir):
                    a = vir[aa]
                    for bb in range(aa+1, nvir):
                        b = vir[bb]
                        ns = s ^ (1 << i) ^ (1 << j) ^ (1 << a) ^ (1 << b)
                        if _in_sorted(ns, bs_sorted): continue
                        if eps > 0:
                            v = abs(eri[a, i, b, j] - eri[a, j, b, i])
                            if v < eps: continue
                        cands[n] = ns; n += 1
        return cands[:n]

    @njit(cache=True)
    def _pairwise_one_config(s, ck, norb, h1, eri, eps, bs_sorted):
        """JIT: standard pair-wise |H_ij * c_j| > eps from one config."""
        occ = np.empty(norb, dtype=np.int64); nocc = 0
        vir = np.empty(norb, dtype=np.int64); nvir = 0
        for i in range(norb):
            if (s >> i) & 1: occ[nocc] = i; nocc += 1
            else: vir[nvir] = i; nvir += 1

        max_n = nocc * nvir + (nocc * (nocc-1) // 2) * (nvir * (nvir-1) // 2)
        cands = np.empty(max_n, dtype=np.int64)
        n = 0

        for ii in range(nocc):
            i = occ[ii]
            for aa in range(nvir):
                a = vir[aa]
                ns = s ^ (1 << i) ^ (1 << a)
                if _in_sorted(ns, bs_sorted): continue
                v = h1[a, i]
                for jj in range(nocc):
                    j = occ[jj]
                    v += eri[a, j, i, j] - eri[a, j, j, i]
                if abs(v) * ck < eps: continue
                cands[n] = ns; n += 1

        for ii in range(nocc):
            i = occ[ii]
            for jj in range(ii+1, nocc):
                j = occ[jj]
                for aa in range(nvir):
                    a = vir[aa]
                    for bb in range(aa+1, nvir):
                        b = vir[bb]
                        ns = s ^ (1 << i) ^ (1 << j) ^ (1 << a) ^ (1 << b)
                        if _in_sorted(ns, bs_sorted): continue
                        v = abs(eri[a, i, b, j] - eri[a, j, b, i])
                        if v * ck < eps: continue
                        cands[n] = ns; n += 1
        return cands[:n]

    print(f"Numba JIT: enabled (4 functions compiled on first call)")

else:
    print("Numba: not available, using pure Python (slower)")


def _compute_diag(s, norb, h1, eri):
    if HAS_NUMBA:
        return _compute_diag_jit(s, norb, h1, eri)
    occ = [i for i in range(norb) if (s >> i) & 1]
    e = 0.0
    for i in occ:
        e += h1[i, i]
    for ii, i in enumerate(occ):
        for j in occ[ii+1:]:
            e += eri[i, i, j, j] - eri[i, j, j, i]
    return e


def cipsi_extend(basis, ci_coeffs, norb, ne, acut, maxd, h1, eri, eps,
              use_coeffs=True, broad_frac=0.01, en_mode=False, e0=0.0):
    bs = set(int(x) for x in basis)
    bs_sorted = np.array(sorted(bs), dtype=np.int64)  # for JIT binary search

    if use_coeffs:
        if isinstance(ci_coeffs, list):
            imp_ref = np.abs(ci_coeffs[0]) >= acut
            if imp_ref.sum() < 5:
                imp_ref[np.argsort(-np.abs(ci_coeffs[0]))[:5]] = True
        else:
            imp = np.abs(ci_coeffs) >= acut
            if imp.sum() < 5:
                imp[np.argsort(-np.abs(ci_coeffs))[:5]] = True
    else:
        cmax = np.max(np.abs(ci_coeffs)) if len(ci_coeffs) > 0 else 0
        threshold = cmax * broad_frac
        imp = np.abs(ci_coeffs) >= threshold
        if imp.sum() < 5:
            imp[np.argsort(-np.abs(ci_coeffs))[:min(5, len(basis))]] = True

    # === BROAD MODE ===
    if not use_coeffs:
        imp_idx = np.where(imp)[0]
        cands = set()
        if HAS_NUMBA:
            for k in imp_idx:
                new_configs = _broad_one_config(int(basis[k]), norb, eri, eps, bs_sorted)
                for ci in range(len(new_configs)):
                    cands.add(int(new_configs[ci]))
        else:
            for k in imp_idx:
                s = int(basis[k])
                occ = [i for i in range(norb) if (s >> i) & 1]
                vir = [i for i in range(norb) if not ((s >> i) & 1)]
                for i in occ:
                    for a in vir:
                        ns = s ^ (1 << i) ^ (1 << a)
                        if ns not in bs: cands.add(ns)
                for ii, i in enumerate(occ):
                    for j in occ[ii+1:]:
                        for aa, a in enumerate(vir):
                            for b in vir[aa+1:]:
                                ns = s ^ (1<<i) ^ (1<<j) ^ (1<<a) ^ (1<<b)
                                if ns in bs: continue
                                if eps > 0:
                                    v = abs(eri[a,i,b,j] - eri[a,j,b,i])
                                    if v < eps: continue
                                cands.add(ns)
        r = np.array(sorted(bs | cands), np.int64)
        return r[:maxd] if maxd and len(r) > maxd else r

    # === EN MODE ===
    if en_mode:
        if isinstance(e0, (list, np.ndarray)):
            e0_list = list(e0); coeffs_list = list(ci_coeffs)
        else:
            e0_list = [e0]; coeffs_list = [ci_coeffs]

        all_couplings = []
        for ri, (e0_r, coeffs_r) in enumerate(zip(e0_list, coeffs_list)):
            imp_r = np.abs(coeffs_r) >= acut
            if imp_r.sum() < 5:
                imp_r[np.argsort(-np.abs(coeffs_r))[:5]] = True
            imp_idx_r = np.where(imp_r)[0]

            coupling = {}
            if HAS_NUMBA:
                for k in imp_idx_r:
                    s = int(basis[k]); ck = float(coeffs_r[k])
                    new_c, new_v = _en_couplings_one_config(s, ck, norb, h1, eri, bs_sorted)
                    for ci in range(len(new_c)):
                        ns = int(new_c[ci])
                        coupling[ns] = coupling.get(ns, 0.0) + new_v[ci]
            else:
                from collections import defaultdict
                coupling = defaultdict(float)
                for k in imp_idx_r:
                    s = int(basis[k]); ck = float(coeffs_r[k])
                    occ = [i for i in range(norb) if (s >> i) & 1]
                    vir = [i for i in range(norb) if not ((s >> i) & 1)]
                    for i in occ:
                        for a in vir:
                            ns = s ^ (1 << i) ^ (1 << a)
                            if ns in bs: continue
                            v = h1[a, i]
                            for j_occ in occ:
                                v += eri[a, j_occ, i, j_occ] - eri[a, j_occ, j_occ, i]
                            coupling[ns] += v * ck
                    for ii, i in enumerate(occ):
                        for j in occ[ii+1:]:
                            for aa, a in enumerate(vir):
                                for b in vir[aa+1:]:
                                    ns = s ^ (1<<i) ^ (1<<j) ^ (1<<a) ^ (1<<b)
                                    if ns in bs: continue
                                    v = eri[a,i,b,j] - eri[a,j,b,i]
                                    coupling[ns] += v * ck
            all_couplings.append((coupling, e0_r))

        all_candidates = set()
        for coupling, _ in all_couplings:
            all_candidates |= set(coupling.keys())
        cands = set()
        for ns in all_candidates:
            h_ii = _compute_diag(ns, norb, h1, eri)
            for coupling, e0_r in all_couplings:
                coup = coupling.get(ns, 0.0)
                if abs(coup) < 1e-15: continue
                denom = abs(e0_r - h_ii)
                if denom < 1e-12: denom = 1e-12
                delta_e = coup * coup / denom
                if delta_e > eps:
                    cands.add(ns); break
        r = np.array(sorted(bs | cands), np.int64)
        return r[:maxd] if maxd and len(r) > maxd else r

    # === STANDARD PAIR-WISE MODE ===
    imp_idx = np.where(imp)[0]
    cands = set()
    if HAS_NUMBA:
        for k in imp_idx:
            new_configs = _pairwise_one_config(int(basis[k]), abs(float(ci_coeffs[k])),
                                                norb, h1, eri, eps, bs_sorted)
            for ci in range(len(new_configs)):
                cands.add(int(new_configs[ci]))
    else:
        for k in imp_idx:
            s = int(basis[k]); ck = abs(ci_coeffs[k])
            occ = [i for i in range(norb) if (s >> i) & 1]
            vir = [i for i in range(norb) if not ((s >> i) & 1)]
            for i in occ:
                for a in vir:
                    ns = s ^ (1 << i) ^ (1 << a)
                    if ns in bs: continue
                    v = h1[a, i]
                    for j in occ:
                        v += eri[a, j, i, j] - eri[a, j, j, i]
                    if abs(v) * ck < eps: continue
                    cands.add(ns)
            for ii, i in enumerate(occ):
                for j in occ[ii+1:]:
                    for aa, a in enumerate(vir):
                        for b in vir[aa+1:]:
                            ns = s ^ (1<<i) ^ (1<<j) ^ (1<<a) ^ (1<<b)
                            if ns in bs: continue
                            v = abs(eri[a,i,b,j] - eri[a,j,b,i])
                            if v * ck < eps: continue
                            cands.add(ns)
    r = np.array(sorted(bs | cands), np.int64)
    return r[:maxd] if maxd and len(r) > maxd else r


def build_sd_transitions(norb, nelec):
    na, nb = nelec; ops = []
    for i in range(na):
        for a in range(na, norb):
            op = np.full(2*norb, 'I', dtype='U1'); op[norb+a]='+'; op[norb+i]='-'; ops.append(op)
    for i in range(na):
        for j in range(i+1, na):
            for a in range(na, norb):
                for b in range(a+1, norb):
                    op = np.full(2*norb, 'I', dtype='U1'); op[norb+a]='+'; op[norb+b]='+'; op[norb+i]='-'; op[norb+j]='-'; ops.append(op)
    for i in range(nb):
        for a in range(nb, norb):
            op = np.full(2*norb, 'I', dtype='U1'); op[a]='+'; op[i]='-'; ops.append(op)
    for i in range(nb):
        for j in range(i+1, nb):
            for a in range(nb, norb):
                for b in range(a+1, norb):
                    op = np.full(2*norb, 'I', dtype='U1'); op[a]='+'; op[b]='+'; op[i]='-'; op[j]='-'; ops.append(op)
    return np.array(ops)
