# AI Future Paper

Draft paper on endogenous automation of AI research, market structure, and regulation.

## Structure

- `main.tex`: manuscript entry point.
- `sections/01_introduction.tex`: motivation and contribution.
- `sections/02_literature.tex`: related literature.
- `sections/03_toy_model.tex`: baseline dynamic model.
- `sections/04_extended_model.tex`: roadmap for the quantitative general-equilibrium model.
- `sections/05_regulation.tex`: market-regulation scenarios.
- `sections/06_conclusion.tex`: conclusions.
- `sections/appendix.tex`: proofs.
- `references.bib`: bibliography.
- `literature/literature_browser.html`: searchable literature database with
  abstracts or explicitly labeled editorial summaries and document links.
- `scripts/build_literature_database.py`: reproducible literature updater and
  coverage validator.

## Build

Compile `main.tex` with a LaTeX engine and BibTeX:

```text
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
