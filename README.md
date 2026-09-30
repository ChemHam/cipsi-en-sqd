# CIPSI-EN-SQD

Code and data for

> H. Ham, C. G. Kumar, M. Kim, J. Y. Lee, *Compact and accurate quantum-centric excited-state simulations via perturbative configuration selection: closed- and open-shell molecular benchmarks* (submitted to J. Chem. Theory Comput.).

CIPSI-EN-SQD grows the subspace of extended sample-based quantum diagonalization (Ext-SQD) iteratively and admits a candidate configuration only if an Epstein–Nesbet-type estimate of its contribution to one of the target states exceeds a threshold. The method, its parameters and the analysis of the data are described in the paper; this file only explains how to run the code and how the data are organized.

## Requirements

```
pyscf==2.10.0
qiskit==1.4.4
ffsim==0.0.60
qiskit-addon-sqd==0.12.0
numpy==2.2.6
```

These are the versions used for every calculation in the paper.

## Usage

OH, CN and LiO<sub>2</sub> were computed with `run_scan.py`:

```bash
python run_scan.py inputs/oh_18o.json                          # whole scan
python run_scan.py inputs/cn_16o.json --geometries 2.90        # one geometry
python run_scan.py inputs/lio2_14o.json --geometries 1.4:2.0   # a range
```

Each geometry is written to `results/<name>/R_<value>.json` as soon as it finishes, together with `results/<name>/meta.json` (package versions and all settings). A scan that is stopped resumes where it left off; `--redo` recomputes existing geometries and `--threads` sets the number of PySCF threads (default 24, overridden by `OMP_NUM_THREADS`). All settings of a system (active space, geometries, target states, sampling and selection parameters) are in its input file under `inputs/`.

N<sub>2</sub> and C<sub>2</sub>H<sub>4</sub> were computed with an earlier notebook implementation of the same procedure; the files in `data/N2/` and `data/C2H4/` are those results, one file per geometry.

## Data

`data/<System>/` holds the results reported in the paper, one zip archive per system (LiO<sub>2</sub> in two parts). Each archive contains one JSON file per geometry:

| System | Active space | Scan | Files |
|---|---|---|---|
| N<sub>2</sub> | (10e, 16o) | *R* = 0.8–3.0 Å, 31 points | `R_0.800.json` … |
| C<sub>2</sub>H<sub>4</sub> | (12e, 14o) | *θ* = 0–180°, 19 points | `theta_0.000.json` … |
| OH | (7e, 18o) | *R* = 0.7–3.0 Å, 32 points | `R_0.700.json` … |
| CN | (9e, 16o) | *R* = 1.0–3.0 Å, 26 points | `R_1.000.json` … |
| LiO<sub>2</sub> | (13e, 14o) | *R*(Li–O<sub>2</sub> midpoint) = 1.4–5.0 Å, 43 points | `R_1.400.json` … |

All use cc-pVDZ with frozen 1s cores. Unzip the archives of a system into one folder before use.

Main fields of each JSON file (energies in hartree, total energies):

| Key | Content |
|---|---|
| `R` | scan coordinate (the dihedral angle in degrees for C<sub>2</sub>H<sub>4</sub>) |
| `fci`, `ext_sqd`, `cipsi` | energy of each target state from FCI, Ext-SQD and CIPSI-EN-SQD |
| `fci_s2`, `ext_s2`, `cipsi_s2` | ⟨S²⟩ of each target state |
| `dims` | numbers of α and β strings of the SQD seed (`sqd_a`, `sqd_b`), Ext-SQD (`ext_a`, `ext_b`) and CIPSI-EN-SQD (`cipsi_a`, `cipsi_b`); the subspace dimension is the product |
| `wf_amps` | for each target state, the α- and β-string weights over the full string space from FCI (`ref_a`, `ref_b`), Ext-SQD (`ext_a`, `ext_b`) and CIPSI-EN-SQD (`cipsi_a`, `cipsi_b`); `wf_amps_meta` gives the full string counts |
| `cipsi_convergence` | subspace size, and energies where stored, at each iteration of the selection |
| `all_var_roots` | (OH, CN, LiO<sub>2</sub>) energies of all variational roots and their squared overlaps with each FCI target state |
| `strings` | (OH, CN, LiO<sub>2</sub>) the retained α and β strings of both methods |

The target states are S<sub>0</sub>, T<sub>1</sub>, T<sub>2</sub> for N<sub>2</sub>; S<sub>0</sub>, S<sub>1</sub>, S<sub>2</sub>, T<sub>1</sub>, T<sub>2</sub> for C<sub>2</sub>H<sub>4</sub>; and D<sub>0</sub>, D<sub>1</sub>, D<sub>2</sub>, Q<sub>1</sub> for the radicals. The quantities in Table S1 (exclusive-region weight, uncovered reference weight) and in Figures 6 and S2–S4 are computed from `wf_amps` as defined in the paper; the `wf_analysis` field is an earlier diagnostic referenced to the lowest Ext-SQD root and is not what the paper reports.

**N<sub>2</sub> at *R* = 2.00 Å.** This point is the reference run of Table S2 (`"source"` field in the file), which replaces the point of the original scan so that Table S1 and Table S2 refer to the same calculation. The N<sub>2</sub> pipeline is not bit-reproducible at a fixed random seed.

## License and funding

MIT License. This work was supported by the Basic Science Research Program through the National Research Foundation of Korea (NRF) funded by the Ministry of Education (No. RS-2019-NR040081).
