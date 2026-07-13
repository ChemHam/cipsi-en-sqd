# cipsi-en-sqd
CIPSI-EN-SQD: Epstein-Nesbet perturbative selection for sample-based quantum diagonalization

CIPSI-EN-SQD augments sample-based quantum diagonalization (SQD) with a two-stage configuration selection: a broad coupling-based expansion followed by an Epstein–Nesbet perturbative screening. This repository contains the numerical data underlying all figures and tables of the manuscript.
File descriptions

*_pes.csv — energies and spin expectation values

One row per geometry × electronic state. This is the primary data behind the potential energy curves reported in the manuscript.

ColumnDescriptionR_angstromBond length (Å). For C2H4 this column is twist_angle_deg, the torsional angle in degrees.stateElectronic state label (S0, T1, … for closed-shell systems; D0, D1, D2, Q1 for open-shell systems).E_FCI_HaReference energy from exact diagonalization in the active space (Hartree).E_ExtSQD_HaExtended-SQD energy (Hartree).E_CIPSI_EN_SQD_HaCIPSI-EN-SQD energy (Hartree).err_ExtSQD_mHaE_ExtSQD − E_FCI, in millihartree.err_CIPSI_EN_SQD_mHaE_CIPSI_EN_SQD − E_FCI, in millihartree.S2_FCI, S2_ExtSQD, S2_CIPSI_EN_SQDSpin expectation value ⟨S²⟩ for each method.

*_dims.csv — subspace dimensions and diagnostics

One row per geometry.

ColumnDescriptiondim_SQD_alpha, dim_SQD_betaNumber of α / β determinants in the SQD subspace.dim_ExtSQD_alpha, dim_ExtSQD_betaSame, after the Ext-SQD expansion.dim_CIPSI_EN_SQD_alpha, dim_CIPSI_EN_SQD_betaSame, after CIPSI-EN selection.shots, n_seedsSampling parameters.n_missingDeterminants present in the Ext-SQD space but discarded by CIPSI-EN selection.n_important_missingDiscarded determinants whose weight exceeds the significance threshold.max_missing_weightLargest weight among discarded determinants.ext_recovery_pct, cipsi_recovery_pctFraction of the FCI wavefunction weight recovered (%).efficiency_ratioRecovered weight per determinant, relative to Ext-SQD.selectivity_ratioRatio of retained to discarded weight.

*_natocc.csv — natural orbital occupations

Long format: one row per geometry × method × orbital.

ColumnDescriptionmethodref (SQD), ext (Ext-SQD), or cipsi (CIPSI-EN-SQD).orbital_indexNatural orbital index, ordered by decreasing occupation.occupationNatural orbital occupation number.

*_convergence.csv — CIPSI iteration trajectory

ColumnDescriptioniterationIteration index, or final for the converged result.n_determinantsSubspace dimension at that iteration.modeSelection mode: seed-broad (initial broad expansion) or EN+broad (combined).

*_recovery.csv — cumulative wavefunction recovery

ColumnDescriptionn_configsNumber of determinants included, sorted by decreasing weight.cumulative_weightCumulative squared amplitude recovered.

*_rootoverlap.csv — root-to-state assignment (OH, CN only)

Evidence for the overlap-based state labelling described above.

ColumnDescriptionmethodExtSQD or CIPSI_EN_SQD.root_indexIndex of the variational root, ordered by energy.E_root_HaEnergy of that root (Hartree).overlap_sq_<state>Squared overlap |⟨Ψ_FCI(state)|Ψ_root⟩|² between the root and each FCI reference state.
