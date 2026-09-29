"""Active-space integrals and CCSD amplitudes, shared by every system."""

import numpy as np
from pyscf import ao2mo, cc, gto, mcscf, mp, scf


def safe_array(x):
    return np.nan_to_num(np.asarray(x, float))


def molecule(atom, cfg, spin):
    return gto.M(atom=atom, basis=cfg.basis, charge=cfg.charge, spin=spin,
                 unit="Angstrom", verbose=0, max_memory=cfg.max_memory)


def scf_reference(mol, spin):
    """RHF for a closed shell, ROHF otherwise."""
    mf = (scf.ROHF if spin else scf.RHF)(mol).run(conv_tol=1e-12)
    if not mf.converged:
        raise RuntimeError("SCF did not converge")
    return mf


def active_space(mf, cfg):
    """CASCI projection onto the active orbitals."""
    mc = mcscf.CASCI(mf, ncas=cfg.norb, nelecas=cfg.nelec)
    mc.ncore = cfg.ncore
    mc.mo_coeff = mf.mo_coeff
    h1, e_core = mc.get_h1eff()
    return h1, ao2mo.restore(1, mc.get_h2eff(), cfg.norb), float(e_core)


def amplitudes(mf, cfg, spin):
    na, nb = cfg.nelec
    mycc = cc.CCSD(mf, frozen=cfg.ncore)
    mycc.conv_tol = cfg.ccsd_conv_tol
    mycc.max_cycle = cfg.ccsd_max_cycle
    mycc.verbose = 0
    source = "ccsd"
    try:
        mycc.kernel()
        if not mycc.converged:
            raise RuntimeError(f"not converged in {mycc.max_cycle} cycles")
        _t2 = mycc.t2
        _mx = max(float(np.abs(np.asarray(b)).max())
                  for b in (_t2 if isinstance(_t2, (tuple, list)) else (_t2,)))
        if _mx > 0.5:
            raise RuntimeError(f"t2 reaches {_mx:.2f}")
        t1_raw, t2_raw, e_corr = mycc.t1, mycc.t2, mycc.e_corr
    except Exception as exc:
        pt = mp.MP2(mf, frozen=cfg.ncore)
        pt.verbose = 0
        pt.kernel()
        t2_raw, e_corr = pt.t2, pt.e_corr
        t1_raw = None
        source = "mp2"
        print(f"    [amp] CCSD failed ({type(exc).__name__}: {exc}); "
              f"using MP2", flush=True)

    if not spin:
        nvir = cfg.norb - na
        t1 = None if t1_raw is None else safe_array(t1_raw[:na, :nvir])
        t2 = safe_array(t2_raw[:na, :na, :nvir, :nvir])
        _check("t2", t2, cfg.norb, closed=True)
        return _report(t1, t2, source, e_corr)

    nva, nvb = cfg.norb - na, cfg.norb - nb
    if not isinstance(t2_raw, (tuple, list)):
        raise RuntimeError("expected spin-resolved amplitudes from ROHF, got "
                           f"one array of {np.shape(t2_raw)}")
    # t1 is not used: the UCJ ansatz comes from a double factorization of T2,
    # and t1 would only initialize a final orbital rotation, which this
    # ansatz does not have. MP2 has no t1 at all.
    t1 = (None if t1_raw is None else
          (safe_array(t1_raw[0][:na, :nva]), safe_array(t1_raw[1][:nb, :nvb])))
    t2 = (safe_array(t2_raw[0][:na, :na, :nva, :nva]),
          safe_array(t2_raw[1][:na, :nb, :nva, :nvb]),
          safe_array(t2_raw[2][:nb, :nb, :nvb, :nvb]))
    _check("t2 aa", t2[0], cfg.norb, closed=True)
    _check("t2 bb", t2[2], cfg.norb, closed=True)
    return _report(t1, t2, source, e_corr)


def _report(t1, t2, source, e_corr):
    blocks = t2 if isinstance(t2, tuple) else (t2,)
    mx = max(float(np.abs(b).max()) for b in blocks)
    print(f"    [amp] {source}  E_corr={e_corr:.8f}  max|t2|={mx:.4f}",
          flush=True)
    if mx > 0.8:
        raise RuntimeError(
            f"t2 amplitudes reach {mx:.2f}; the LUCJ operator would be built "
            f"from a diverged expansion")
    return t1, t2


def _check(label, arr, norb, closed=False):
    """The LUCJ operator takes its orbital count from these shapes."""
    span = (arr.shape[0] + arr.shape[2]) if closed else (arr.shape[0] + arr.shape[1])
    if span != norb:
        raise RuntimeError(f"{label} {arr.shape} spans {span} orbitals, "
                           f"not {norb}")


def build(system, x, cfg):
    """Everything one geometry needs."""
    mol = molecule(system.geometry(x, cfg), cfg, system.SPIN)
    mf = scf_reference(mol, system.SPIN)
    h1, eri, e_core = active_space(mf, cfg)
    t1, t2 = amplitudes(mf, cfg, system.SPIN)
    return dict(h1=h1, eri=eri, e_core=e_core, t1=t1, t2=t2, mf=mf, mol=mol)
