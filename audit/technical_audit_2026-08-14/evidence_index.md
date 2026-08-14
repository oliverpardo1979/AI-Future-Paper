# Evidence index for the technical audit

Audit date: 2026-08-14
Audited commit: `158a0ed86256de59a77ae8320c8c0d1a03b7ad86`

## Scope

The audit covers the A×M manuscript compiled from `main_axm.tex`, its six source
sections, the rendered 50-page PDF, the active A×M simulation and audit scripts,
the canonical numerical outputs under `numerical_axm`, the bibliography, and
primary external sources used to check empirical and literature claims. It does
not alter the manuscript.

## Manuscript sources

- `main_axm.tex`
- `sections_axm/01_introduction.tex`
- `sections_axm/02_literature.tex`
- `sections_axm/03_model.tex`
- `sections_axm/04_growth_regimes.tex`
- `sections_axm/06_conclusion.tex`
- `sections_axm/appendix.tex`
- `references.bib`
- `build_axm/main_axm.pdf`

All 50 rendered pages were inspected through five ten-page contact sheets. No
material clipping, overflow, broken cross-reference, or unreadable figure was
found at that review scale.

## Numerical sources

### Unit elasticity in final production

- `scripts/simulate_axm_equilibrium.py`
- `scripts/audit_axm_model.py`
- `numerical_axm/equilibrium_transition_paths.csv`
- `numerical_axm/equilibrium_transition_summary.csv`
- `numerical_axm/equilibrium_horizon_paths.csv`
- `numerical_axm/equilibrium_horizon_robustness.csv`
- `numerical_axm/audit_report.csv`
- `numerical_axm/equilibrium_equation_map.csv`

The audit report contains 121 passing rows. Across the six horizon paths, the
largest stored static residual is 4.7407411e-11, the largest independently
reconstructed static error is 1.4362893e-9, and the largest stored dynamic
residual is 1.6724797e-5. The dynamic audit uses solver-stored collocation
residuals rather than an independent differentiation of the saved paths.

### Gross complements in final production

- `scripts/simulate_axm_complements_equilibrium.py`
- `scripts/audit_axm_complements.py`
- `numerical_axm/complements_transition_paths.csv`
- `numerical_axm/complements_transition_summary.csv`
- `numerical_axm/complements_horizon_paths.csv`
- `numerical_axm/complements_horizon_robustness.csv`
- `numerical_axm/complements_acceptance_report.csv`
- `numerical_axm/complements_equation_residuals.csv`
- `numerical_axm/complements_audit_manifest.json`

The acceptance report contains 130 passing rows. On the two primary paths, the
largest static reconstruction error is 2.9162450e-11, the largest independent
finite-difference dynamic error is 5.6802017e-7, and the largest solver RMS is
9.4277121e-7.

### Gross substitutes in final production

- `scripts/simulate_axm_high_sigma_equilibrium.py`
- `scripts/audit_axm_high_sigma.py`
- `numerical_axm/high_sigma_sigma150_validated_boundary_paths.csv`
- `numerical_axm/high_sigma_sigma150_z128_validated_boundary_paths.csv`
- `numerical_axm/high_sigma_sigma150_validated_free_continuation.csv`
- `numerical_axm/high_sigma_sigma150_z128_validated_free_continuation.csv`
- `numerical_axm/high_sigma_sigma150_validated_acceptance_report.csv`
- `numerical_axm/high_sigma_sigma150_validated_equation_residuals.csv`
- `numerical_axm/high_sigma_sigma150_validated_boundary_convergence.csv`
- `numerical_axm/high_sigma_sigma150_validated_audit_manifest.json`

The acceptance report contains 76 passing rows and covers 11,747 saved
observations. The largest reconstructed static first-order-condition error is
1.5606538e-10, the largest stored dynamic residual is 2.2619986e-5, and the
largest independent finite-difference dynamic error on the compact audited
window is 3.3274823e-5. The independent derivative window excludes the final 50
years near the artificial boundary.

## Primary external sources checked

- Davidson, Halperin, Houlden, and Korinek (2026), NBER Working Paper 35155:
  https://www.nber.org/papers/w35155
- Korinek and McKelvey (2026), Bank of Canada Staff Working Paper 2026-20:
  https://www.bankofcanada.ca/2026/06/staff-working-paper-2026-20/
- International Energy Agency, Energy and AI (2025):
  https://www.iea.org/reports/energy-and-ai/executive-summary
- International Energy Agency, Key Questions on Energy and AI (2026):
  https://www.iea.org/reports/key-questions-on-energy-and-ai/executive-summary
- METR RE-Bench report (2024):
  https://metr.org/blog/2024-11-22-evaluating-r-d-capabilities-of-llms/
- METR Claude 3.7 report (2025):
  https://metr.org/evaluations/claude-3-7-report/
- Aum and Shin (2024), NBER Working Paper 32591:
  https://www.nber.org/papers/w32591
- Aghion, Jones, and Jones, chapter revision on AI and growth:
  https://www.nber.org/system/files/chapters/c14015/revisions/c14015.rev1.pdf

## Evidence classifications used

- **Demonstrated:** follows from a proposition whose hypotheses are explicit and
  whose proof was checked.
- **Conditionally derived:** algebraically correct on a postulated branch, but
  branch existence, reachability, or global optimality is not established.
- **Numerically supported:** appears in the reported finite-horizon or
  finite-boundary computations and passes the stated numerical checks.
- **Not verified as equilibrium:** the computation satisfies dated necessary
  conditions but an admissible infinite-horizon continuation, transversality, or
  global optimality is not established.

## Chart map

The report contains one compact bar chart, “Evidentiary status of 12 central
results.” It answers whether the paper's main weakness is incorrect economic
results or excessive scope. The four mutually exclusive categories are counts
from the result-validation matrix: 8 supported within their stated scope, 2
partial or conditional, 1 unsupported in its strong formulation, and 1
contradicted by the closest literature. Exact claims and caveats remain in the
table because the chart is only a summary.

The three detailed report tables are queried from
`audit/technical_audit_2026-08-14/audit_tables.sqlite`. The reviewed snapshot is
rebuilt from `artifact.json` by `build_audit_database.py`.
