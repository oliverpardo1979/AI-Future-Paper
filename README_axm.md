# Parallel A-M paper

This part of the project leaves `main.tex` and the original model unchanged.
The parallel manuscript is `main_axm.tex`. Its baseline research technology is

$$
\dot A=\chi\left[
\omega_H H^{\varrho_{HM}}+
\omega_M(AM)^{\varrho_{HM}}
\right]^{\eta/\varrho_{HM}},
\qquad
\varrho_{HM}=\frac{\sigma_{HM}-1}{\sigma_{HM}},
$$

with no separate $A^\phi$ term. The constant-returns effective-research index inside this
equation is denoted by $E$ when useful, so the same law can be written
$\dot A=\chi E^\eta$. The index $E$ is an accounting device, not a second
production stage. The baseline maintains $0<\eta<1$; $\eta=1$ is the
constant-returns boundary and is not covered by the current propositions.
At $\sigma_{HM}=1$, the continuous limit is
$\dot A=\chi H^{\eta\omega_H}(AM)^{\eta\omega_M}$.

Inference and research compute share a constant unit cost. The manuscript
measures both inputs in resource-expenditure units and normalizes that cost to
one; capability and research productivity are correspondingly rescaled.

## Reproduce the verified figures and audit

Run `python scripts/simulate_axm_equilibrium.py` and then
`python scripts/audit_axm_model.py`. The Davidson-style feedback diagnostic is
generated separately, without rerunning the solver:

```
python scripts/plot_axm_feedback_diagnostic.py
```

The first command solves the two reported perfect-foresight path approximations and writes
their data to `numerical_axm/` and figures to `figures_axm/`. The second checks
the analytical benchmark calculations and rejects numerical paths whose market,
first-order, feasibility, second-order, terminal-target, or dynamic residuals
exceed the documented tolerances. It also stores and audits all six paths used in
the terminal-horizon robustness exercise. Proposition
`prop:axm-equilibrium-sufficiency` proves that any admissible exact
infinite-horizon path satisfying the full system and transversality conditions is
globally optimal under both reported unit-elasticity parameterizations. The numerical audit
verifies its finite-horizon discretization and terminal proxies; it cannot by
itself prove an infinite-horizon transversality limit or existence of the
continued path.

The audit also reconstructs the new research specification independently from
every saved observation: `AM = A*M`, the CES input index, the consolidated
capability law, both research first-order conditions, and the capability-feedback
term in the costate equation. Files in `numerical_axm/` and `figures_axm/` are the
only model-simulation artifacts used by `main_axm.tex`.
The equation-by-equation paper/code map is
`numerical_axm/equilibrium_equation_map.csv`, and the six complete robustness
paths are in `numerical_axm/equilibrium_horizon_paths.csv`.
The feedback script reads the canonical unit-elasticity paths and evaluates the
plug-in balanced-growth measure $G(t)=\eta s_M(t)(1+\nu)$ at each dated
allocation. It is not the transition Jacobian or independent numerical evidence.
Its line $G=1$ is a balanced-growth reference, not a stand-alone explosion
condition away from the limiting system.

The final-production complementarity cases use a separate runner and an
independent auditor:

```
python scripts/simulate_axm_complements_equilibrium.py
python scripts/audit_axm_complements.py
```

The runner solves the new $A\times M$ specification at
$\sigma_{XL}=0.75$, for $\sigma_{HM}=1$ and $2$, from cold starts at terminal
dates 1,600, 1,800, and 2,000. The primary paths use the 2,000-year boundary.
Only $C/Y$ and $X/(qA^2)$ are imposed at that boundary; growth rates, factor
shares, $X/L$, the research allocation, resource shares, and the interest rate
are reported rather than imposed. The independent audit imports no solver
functions, reconstructs every static equation, computes separate seven-point
dynamic residuals, checks horizon stability and finite-date TVC proxies, and
hashes all four canonical inputs. Its acceptance label is limited to
"finite-horizon candidate paths satisfying the dated equilibrium system and
the stated terminal conditions." It is not an existence theorem for an exact
infinite-horizon equilibrium.

The web-based AI Growth Lab in `docs/`, the model-simulation artifacts in
`numerical/` and `figures/`, and simulation scripts without `_axm` in their names
belong to the original specification. They must not be used as model evidence
for the parallel A-M manuscript until revalidated. The empirical evidence figure
`figures/empirical_ai_scale.png` is shared across the papers and is not a model
simulation.

`scripts/simulate_axm_high_sigma_equilibrium.py` contains the separate
free-boundary solver for $\sigma_{XL}>1$. The reported case uses
$\sigma_{XL}=1.5$ and $\sigma_{HM}=2$ and continues the terminal boundary through
$Y/K=16,32,64,128$. Reproduce the staged continuation and its audit with

```
python scripts/simulate_axm_high_sigma_equilibrium.py --assemble-published
python scripts/audit_axm_high_sigma.py
python scripts/plot_axm_high_sigma_validated.py
```

The first command starts from the unit-elasticity, $\sigma_{HM}=2$ solution,
continues first in $\sigma_{XL}$, then in the fixed horizon, and finally in the
free terminal boundary. The preliminary seed, parameter, horizon, and boundary
continuations use tolerance $3\times10^{-5}$. Starting from the resulting coarse
$Y/K=64$ solution, a second pass solves $Y/K=16,32,64,128$ sequentially at
tolerance $10^{-6}$. Only those four refined path/continuation files are written.
Use the cheap `--assemble-published --dry-run` command to inspect the exact
sequence without solving or changing any files. The full route contains 38
boundary-value solves and can take tens of minutes or longer. Run
`python scripts/audit_axm_high_sigma.py` after solving.
The independent audit reconstructs the dated equations, checks interiority and
the monopoly second-order condition, and requires convergence of the initial
jumps, paths on common pre-singular windows, terminal ratios, and estimated
singularity dates. Passing this gate licenses only the description "convergent
finite-boundary approximations to a conditional pre-singular branch satisfying
dated necessary conditions." It does not
establish an infinite-horizon equilibrium, transversality at infinity, or global
intertemporal optimality.

Compile the manuscript from the repository root with
`tectonic --keep-logs --outdir build_axm main_axm.tex`.
