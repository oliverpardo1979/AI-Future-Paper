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
`python scripts/audit_axm_model.py`.

The first command solves the two reported perfect-foresight candidate paths and writes
their data to `numerical_axm/` and figures to `figures_axm/`. The second checks
the analytical benchmark calculations and rejects numerical paths whose market,
first-order, feasibility, second-order, terminal-target, or dynamic residuals
exceed the documented tolerances. These checks verify the stated system of
necessary equilibrium conditions used in the manuscript; they do not establish
global concavity of the developer's intertemporal problem.

`scripts/simulate_axm_high_sigma_equilibrium.py` contains the separate
free-boundary solver for $\sigma_{XL}>1$. The paper does not yet report its
curves because continuation across increasingly distant terminal boundaries has
not passed the convergence test. This is an intentional safeguard, not a missing
figure.

Compile the manuscript from the repository root with
`tectonic --keep-logs --outdir build_axm main_axm.tex`.
