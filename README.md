# CIPSI-EN-SQD

**Epstein–Nesbet perturbative selection for sample-based quantum diagonalization**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Data: CSV](https://img.shields.io/badge/data-CSV-blue.svg)](data/)

CIPSI-EN-SQD augments sample-based quantum diagonalization (SQD) with a two-stage
configuration selection: a broad coupling-based expansion followed by an
Epstein–Nesbet perturbative screening. This repository contains the numerical data
underlying all figures and tables of the manuscript.

---

## Systems

| Directory | System | Active space | Scan coordinate |
|---|---|---|---|
| [`data/N2`](data/N2) | N₂ | (10e, 16o) | Bond length |
| [`data/OH`](data/OH) | OH | (7e, 18o) | Bond length |
| [`data/CN`](data/CN) | CN | (9e, 16o) | Bond length |
| [`data/LiO2`](data/LiO2) | LiO₂ | (13e, 14o) | Bond length |
| [`data/C2H4`](data/C2H4) | C₂H₄ | (12e, 14o) | Torsional angle |

All calculations use the cc-pVDZ basis set. Reference values (`FCI`) are obtained by
exact diagonalization within the stated active space.

**State labelling.** Variational roots are assigned to reference states by maximum
squared overlap with the corresponding FCI eigenvector, *not* by energy ordering.
The full overlap matrices are provided in `*_rootoverlap.csv` (OH and CN).

---

## File descriptions

### `*_pes.csv` — energies and spin expectation values

One row per geometry × electronic state. This is the primary data behind the potential
energy curves reported in the manuscript.

| Column | Description |
|---|---|
| `R_angstrom` | Bond length (Å). For C₂H₄ this column is `twist_angle_deg`, the torsional angle in degrees. |
| `state` | Electronic state label (`S0`, `T1`, … for closed-shell systems; `D0`, `D1`, `D2`, `Q1` for open-shell systems). |
| `E_FCI_Ha` | Reference energy from exact diagonalization in the active space (Hartree). |
| `E_ExtSQD_Ha` | Extended-SQD energy (Hartree). |
| `E_CIPSI_EN_SQD_Ha` | CIPSI-EN-SQD energy (Hartree). |
| `err_ExtSQD_mHa` | `E_ExtSQD − E_FCI`, in millihartree. |
| `err_CIPSI_EN_SQD_mHa` | `E_CIPSI_EN_SQD − E_FCI`, in millihartree. |
| `S2_FCI`, `S2_ExtSQD`, `S2_CIPSI_EN_SQD` | Spin expectation value ⟨S²⟩ for each method. |

### `*_dims.csv` — subspace dimensions and diagnostics

One row per geometry.

| Column | Description |
|---|---|
| `dim_SQD_alpha`, `dim_SQD_beta` | Number of α / β determinants in the SQD subspace. |
| `dim_ExtSQD_alpha`, `dim_ExtSQD_beta` | Same, after the Ext-SQD expansion. |
| `dim_CIPSI_EN_SQD_alpha`, `dim_CIPSI_EN_SQD_beta` | Same, after CIPSI-EN selection. |
| `shots`, `n_seeds` | Sampling parameters. |
| `n_missing` | Determinants present in the Ext-SQD space but discarded by CIPSI-EN selection. |
| `n_important_missing` | Discarded determinants whose weight exceeds the significance threshold. |
| `max_missing_weight` | Largest weight among discarded determinants. |
| `ext_recovery_pct`, `cipsi_recovery_pct` | Fraction of the FCI wavefunction weight recovered (%). |
| `efficiency_ratio` | Recovered weight per determinant, relative to Ext-SQD. |
| `selectivity_ratio` | Ratio of retained to discarded weight. |

### `*_natocc.csv` — natural orbital occupations

Long format: one row per geometry × method × orbital.

| Column | Description |
|---|---|
| `method` | `ref` (SQD), `ext` (Ext-SQD), or `cipsi` (CIPSI-EN-SQD). |
| `orbital_index` | Natural orbital index, ordered by decreasing occupation. |
| `occupation` | Natural orbital occupation number. |

### `*_convergence.csv` — CIPSI iteration trajectory

| Column | Description |
|---|---|
| `iteration` | Iteration index, or `final` for the converged result. |
| `n_determinants` | Subspace dimension at that iteration. |
| `mode` | Selection mode: `seed-broad` (initial broad expansion) or `EN+broad` (combined). |

### `*_recovery.csv` — cumulative wavefunction recovery

| Column | Description |
|---|---|
| `n_configs` | Number of determinants included, sorted by decreasing weight. |
| `cumulative_weight` | Cumulative squared amplitude recovered. |

### `*_rootoverlap.csv` — root-to-state assignment (OH, CN only)

Evidence for the overlap-based state labelling described above.

| Column | Description |
|---|---|
| `method` | `ExtSQD` or `CIPSI_EN_SQD`. |
| `root_index` | Index of the variational root, ordered by energy. |
| `E_root_Ha` | Energy of that root (Hartree). |
| `overlap_sq_<state>` | Squared overlap between the root and each FCI reference state. |

---

## Computational details

| Setting | Value |
|---|---|
| Basis set | cc-pVDZ |
| CIPSI amplitude cutoff (`ACUT`) | 3 × 10⁻³ |
| Broad selection threshold (`eps_broad`) | 5 × 10⁻⁴ |
| Epstein–Nesbet threshold (`eps_EN`) | 1 × 10⁻⁶ |
| Software | PySCF 2.12.1, ffsim, qiskit-addon-sqd |

Per-geometry sampling parameters (`shots`, `n_seeds`) are recorded in `*_dims.csv`.

---

## Usage

```python
import pandas as pd

pes = pd.read_csv("data/N2/n2_16o_pes.csv")
s0 = pes[pes.state == "S0"]
print(s0[["R_angstrom", "err_ExtSQD_mHa", "err_CIPSI_EN_SQD_mHa"]])
```

---

## License

Released under the [MIT License](LICENSE).

## Funding

Supported by the Basic Science Research Program through the National Research
Foundation of Korea (NRF), funded by the Ministry of Education
(No. RS-2019-NR040081).
