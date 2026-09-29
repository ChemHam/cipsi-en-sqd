"""Scan parameters, read from an input file and bound into the library."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


# Parameters every system shares unless its input file overrides them. The
# values are those of the published scans; changing one here changes every
# system, which is why the input files carry the per-system ones.
@dataclass
class Config:
    # --- system ---
    name: str
    norb: int
    nelec: tuple            # (n_alpha, n_beta)
    ncore: int
    system: str = ""
    basis: str = "cc-pvdz"
    max_memory: int = 8000
    charge: int = 0
    spin: int = 0
    target_states: tuple = ("D0", "D1", "D2", "Q1")

    # --- scan ---
    geometries: tuple = ()
    geometry_kwargs: dict = field(default_factory=dict)

    # --- sampling ---
    n_shots: int = 100_000
    n_seeds: int = 1
    noise_level: float = 0.1
    s_core_iter: int = 5
    n_batches: int = 10
    samples_per_batch: int = 500
    merge_top_k: int = 1
    ccsd_conv_tol: float = 1e-10
    ccsd_max_cycle: int = 300
    lucj_reps: int = 8
    lucj_scalings: tuple = (0.5, 1.0, 1.5, 2.0)
    seed: int = 42

    # --- selection ---
    cipsi_acut: float = 3e-3
    cipsi_broad_eps: float = 5e-4
    cipsi_en_eps: float = 1e-6
    cipsi_max_iter: int = 10
    cipsi_ext_roots: int = 10
    cipsi_broad_mult: int = 10
    cipsi_broad_frac: float = 0.01

    # --- eigensolver ---
    nroots_grow: int = 10
    nroots_int: int = 10
    grow_conv_tol: float = 1e-4
    grow_max_cycle: int = 30
    final_conv_tol: float = 1e-10
    final_max_cycle: int = 200

    # --- state assignment ---
    labeling: str = "overlap"      # "overlap" or "spin"
    overlap_floor: float = 0.30
    max_doublet: int = 3
    max_quartet: int = 1
    # Which two multiplicities the spin labeller looks for, and how many of
    # each. A closed shell is S0/T1, an open shell D0/Q1; testing the wrong
    # pair against S^2 leaves every root unlabelled.
    open_shell: bool = True
    spin_tags: tuple = ("doublet", "quartet")
    max_low: int = 3
    max_high: int = 1

    def __post_init__(self):
        self.nelec = tuple(self.nelec)
        self.target_states = tuple(self.target_states)
        self.spin_tags = tuple(self.spin_tags)
        self.geometries = tuple(self.geometries)
        self.lucj_scalings = tuple(self.lucj_scalings)
        if self.labeling not in ("overlap", "spin"):
            raise ValueError(f"labeling must be 'overlap' or 'spin', "
                             f"not {self.labeling!r}")
        na, nb = self.nelec
        if na < nb:
            raise ValueError(f"{self.name}: nelec is (alpha, beta) and alpha "
                             f"must not be the smaller, got {self.nelec}")
        if na > self.norb or nb > self.norb:
            raise ValueError(f"{self.name}: more electrons of one spin than "
                             f"orbitals, nelec={self.nelec} norb={self.norb}")

    # The names the notebook's function bodies look up. Keeping the mapping
    # explicit means a typo raises here rather than leaving a global unset.
    _GLOBALS = {
        "NORB": "norb", "NELEC": "nelec", "NCORE": "ncore",
        "TARGET_STATES": "target_states",
        "N_SHOTS": "n_shots", "N_SEEDS": "n_seeds",
        "NOISE_LEVEL": "noise_level", "S_CORE_ITER": "s_core_iter",
        "N_BATCHES": "n_batches", "SAMPLES_PER_BATCH": "samples_per_batch",
        "MERGE_TOP_K": "merge_top_k",
        "NROOTS_INT": "nroots_int", "NROOTS_GROW": "nroots_grow",
        "OVERLAP_FLOOR": "overlap_floor",
        "MAX_DOUBLET": "max_doublet", "MAX_QUARTET": "max_quartet",
        "CIPSI_ACUT": "cipsi_acut", "CIPSI_BROAD_EPS": "cipsi_broad_eps",
        "CIPSI_EN_EPS": "cipsi_en_eps", "CIPSI_MAX_ITER": "cipsi_max_iter",
        "CIPSI_EXT_ROOTS": "cipsi_ext_roots",
        "CIPSI_BROAD_MULT": "cipsi_broad_mult",
        "CIPSI_BROAD_FRAC": "cipsi_broad_frac",
        "GROW_CONV_TOL": "grow_conv_tol", "GROW_MAX_CYCLE": "grow_max_cycle",
        "FINAL_CONV_TOL": "final_conv_tol",
        "FINAL_MAX_CYCLE": "final_max_cycle",
        "LABELING": "labeling",
        "SPIN_TAGS": "spin_tags",
        "OPEN_SHELL": "open_shell",
        "MAX_LOW": "max_low",
        "MAX_HIGH": "max_high",
    }

    def as_globals(self) -> dict:
        return {g: getattr(self, a) for g, a in self._GLOBALS.items()}

    def to_dict(self) -> dict:
        return asdict(self)


def load(path) -> Config:
    """Read an input file. JSON, or YAML if PyYAML is installed."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        import yaml                      # optional; JSON needs nothing
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    known = {f.name for f in fields(Config)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
    return Config(**data)


_BOUND = None


def bind(cfg: Config) -> None:
    """Bind the parameters into every module that reads them as globals.

    Called once, before any geometry is computed. A second call with a
    different configuration raises: a process runs one system, and silently
    rebinding halfway through would make the results of a scan depend on the
    order its geometries happened to be visited in.
    """
    global _BOUND
    if _BOUND is not None and _BOUND.to_dict() != cfg.to_dict():
        raise RuntimeError(
            f"configuration already bound to {_BOUND.name!r}; start a new "
            f"process for {cfg.name!r}")

    from . import driver, labeling, solver, spin
    values = cfg.as_globals()
    for module in (driver, labeling, solver, spin):
        for name in list(vars(module)):
            if name in values:
                setattr(module, name, values[name])
    driver.CFG = cfg
    _BOUND = cfg


def bound() -> Config:
    if _BOUND is None:
        raise RuntimeError("no configuration bound; call sqdlib.config.bind()")
    return _BOUND
