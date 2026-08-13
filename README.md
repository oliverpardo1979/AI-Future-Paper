# AI Future Paper

Draft paper on endogenous automation of AI research, market structure, and regulation.

## Structure

- `main.tex`: manuscript entry point.
- `sections/01_introduction.tex`: motivation and contribution.
- `sections/02_literature.tex`: related literature.
- `sections/03_toy_model.tex`: baseline dynamic model and analytical foundations.
- `sections/04_extended_model.tex`: roadmap for the quantitative general-equilibrium model.
- `sections/05_numerical_experiments.tex`: four growth regimes, each pairing analytical results with equilibrium transitions, plus numerical validation.
- `sections/05_regulation.tex`: market-regulation scenarios.
- `sections/06_conclusion.tex`: conclusions.
- `sections/appendix.tex`: proofs.
- `references.bib`: bibliography.
- `empirical/`: source-specific CSV snapshots and documentation for the empirical
  motivation figures.
- `literature/literature_browser.html`: searchable literature database with
  abstracts or explicitly labeled editorial summaries and document links.
- `scripts/build_empirical_figures.py`: reproducibly builds the two empirical
  motivation figures without modifying the underlying data snapshots.
- `scripts/build_literature_database.py`: reproducible literature updater and
  coverage validator.
- `scripts/simulate_equilibrium.py`: perfect-foresight equilibrium solver for the
  finite-terminal regimes $\sigma_{XL}<1$ and $\sigma_{XL}=1$, including the
  human-essential $\sigma_{HM}=1$ research benchmark.
- `scripts/simulate_high_sigma_equilibrium.py`: free-boundary equilibrium solver
  for $1<\sigma_{XL}<1/\alpha$.
- `scripts/audit_equilibrium_outputs.py`: equation-by-equation audit of every
  published canonical equilibrium path.
- `scripts/check_equilibrium_horizon_convergence.py`: terminal-horizon stability
  test for the $\sigma_{XL}\leq1$ paths.
- `numerical/equilibrium_transition_paths.csv`: simulated equilibrium paths and
  equation residuals.
- `numerical/equilibrium_transition_summary.csv`: terminal statistics and numerical
  validation by regime.
- `numerical/high_sigma_equilibrium_paths.csv`: published gross-substitution
  equilibrium paths.
- `numerical/high_sigma_equilibrium_summary.csv`: singularity estimates and
  equilibrium residuals for the gross-substitution paths.
- `numerical/equilibrium_system_audit_summary.csv`: algebraic, dynamic, static,
  optimality-margin, and endpoint diagnostics for all six published paths.
- `docs/`: static GitHub Pages equilibrium laboratory. Readers can load audited
  benchmarks, change parameters and initial conditions, solve a new canonical
  branch locally in the browser, inspect diagnostics, and download its path.
- `scripts/build_web_benchmarks.py`: creates the browser-ready benchmark dataset
  from the published canonical CSV outputs.

The older proportional-allocation simulations are retained as transparent mechanism
exercises. They impose investment or research-spending shares and therefore should not
be interpreted as decentralized equilibrium paths.

## Build

Compile `main.tex` with a LaTeX engine and BibTeX:

```text
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Interactive equilibrium laboratory

The public simulator is deployed from `docs/` by
`.github/workflows/pages.yml`. For a local preview, serve the folder over HTTP:

```text
python -m http.server 8000 --directory docs
```

The custom solver runs in the reader's browser. Convergence means that the
reported canonical equations and numerical checks pass; it does not by itself
prove global dynamic optimality, existence, or uniqueness.
