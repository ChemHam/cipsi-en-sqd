
import numpy as np
from pyscf import fci
from pyscf.fci import selected_ci

from .config import bind


# Scan parameters, bound from the input file before compute_point() runs.
# The function bodies below read them as module globals, exactly as they
# did in the notebook, so that nothing inside them had to be rewritten.
GROW_CONV_TOL = None
GROW_MAX_CYCLE = None


def kernel_safe(myci, h1, eri, norb, nelec, ci_strs, nroots, loose=False):
    """Diagonalize in a fixed determinant space."""
    try:
        if loose:
            myci.conv_tol = GROW_CONV_TOL; myci.max_cycle = GROW_MAX_CYCLE
        else:
            myci.conv_tol = 1e-10; myci.max_cycle = 200
        e, c = selected_ci.kernel_fixed_space(
            myci, h1, eri, norb, nelec, ci_strs=ci_strs, nroots=nroots)
        if np.isscalar(e): return [float(e)], [c]
        return [float(x) for x in e], list(c)
    except Exception as exc:
        # FIX. This returned empty lists, and the caller went on without the
        # states it had asked for, so a deviation was then taken over whatever
        # roots happened to survive. A diagonalization that fails is not a
        # result; see CHANGES.md.
        raise RuntimeError(
            f"diagonalization failed for nroots={nroots}, loose={loose}: "
            f"{type(exc).__name__}: {exc}") from exc
