"""Circuit sampling of the seed configurations."""

import numpy as np
import ffsim
from qiskit import QuantumCircuit

from .arrays import det_strings, safe_array


def lucj_operator(t2, ts, n_reps, open_shell):
    """The LUCJ operator at one amplitude scaling.

    Open shell wants t1 = (a, b) and t2 = (aa, ab, bb); closed shell wants
    one array of each. from_t_amplitudes reads the orbital count off their
    shapes, so a window of the wrong width builds a circuit of the wrong
    width and qc.append raises.
    """
    if open_shell:
        return ffsim.UCJOpSpinUnbalanced.from_t_amplitudes(
            t2=tuple(x * ts for x in t2), n_reps=(n_reps, n_reps))
    return ffsim.UCJOpSpinBalanced.from_t_amplitudes(t2=t2 * ts, n_reps=n_reps)


def sample_bsm(norb, nelec, t1, t2, n_shots, noise_level=0.1, seed=42,
               scalings=(0.5, 1.0, 1.5, 2.0), n_reps=8, report=True):
    """Sample the seed configurations from the LUCJ state.

    The circuit is averaged over the amplitude scalings, then ten percent
    uniform noise is mixed in and the distribution is sampled.

    A scaling that raises stops the run. The notebook caught them with a bare
    except and fell back to a delta on Hartree-Fock, which still yields
    configurations and still produces energies, so a scan that never built a
    circuit looked like one that did. That is how the open-shell scans ran.
    """
    # t2 decides, not t1: the ansatz is built from t2 alone, and t1 is None
    # whenever the amplitudes came from MP2.
    open_shell = isinstance(t2, (tuple, list))
    t2a = tuple(safe_array(x) for x in t2) if open_shell else safe_array(t2)

    rng = np.random.default_rng(seed)
    strsa, strsb = det_strings(norb, nelec)
    n_b = len(strsb)

    probs_merged, n_merged = None, 0
    for ts in scalings:
        qc = QuantumCircuit(2 * norb)
        qc.append(ffsim.qiskit.PrepareHartreeFockJW(norb, nelec),
                  list(range(2 * norb)))
        qc.barrier()
        ucj = lucj_operator(t2a, ts, n_reps, open_shell)
        jw = (ffsim.qiskit.UCJOpSpinUnbalancedJW if open_shell
              else ffsim.qiskit.UCJOpSpinBalancedJW)
        qc.append(jw(ucj), list(range(2 * norb)))
        vec = ffsim.qiskit.final_state_vector(qc, norb=norb, nelec=nelec)
        p = np.abs(np.asarray(vec, np.complex128)) ** 2
        probs_merged = p.copy() if probs_merged is None else probs_merged + p
        n_merged += 1
    if n_merged != len(scalings):
        raise RuntimeError(f"only {n_merged} of {len(scalings)} LUCJ scalings "
                           f"produced a state vector")
    if report:
        print(f"    [lucj] {n_merged}/{len(scalings)} scalings merged",
              flush=True)

    probs_best = probs_merged / n_merged
    probs_noisy = (1 - noise_level) * probs_best + noise_level / len(probs_best)
    probs_noisy /= probs_noisy.sum()
    indices = rng.choice(len(probs_noisy), size=n_shots, p=probs_noisy)

    bsm = np.zeros((n_shots, 2 * norb), dtype=bool)
    shot_probs = np.zeros(n_shots)
    for k, idx in enumerate(indices):
        ia, ib = idx // n_b, idx % n_b
        a, b = int(strsa[ia]), int(strsb[ib])
        for i in range(norb):
            if (b >> i) & 1:
                bsm[k, i] = True
            if (a >> i) & 1:
                bsm[k, norb + i] = True
        shot_probs[k] = probs_noisy[idx]
    shot_probs /= shot_probs.sum()
    return bsm, shot_probs


# The driver calls it by the open-shell name.
sample_ucj_bsm = sample_bsm
