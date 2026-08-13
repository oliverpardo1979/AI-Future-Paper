"""Check finite-horizon convergence of the non-singular equilibrium BVPs."""

from __future__ import annotations

import csv
import math
from dataclasses import replace
from pathlib import Path

import numpy as np

import simulate_equilibrium as equilibrium


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    baseline, analytical = equilibrium.mechanism.analytical_calibration(
        equilibrium.Parameters()
    )
    initial_capital = equilibrium.mechanism.calibrate_initial_capital(
        baseline, analytical["capital_output_ratio"]
    )
    initial_state = (initial_capital, 1.0, 1.0)
    baseline = equilibrium.mechanism.calibrate_research_productivity(
        baseline, initial_state, analytical["capability_growth"]
    )
    rows = []
    for scenario, sigma_xl, sigma_hm, horizons in (
        (
            "equilibrium_sigma_0_75",
            0.75,
            2.00,
            (400.0, 600.0, 900.0),
        ),
        (
            "equilibrium_sigma_1_00",
            1.00,
            2.00,
            (1000.0, 1500.0, 2000.0),
        ),
        (
            "equilibrium_sigma_1_00_hm_1_00",
            1.00,
            1.00,
            (1000.0, 1500.0, 2000.0),
        ),
    ):
        parameters = replace(
            baseline,
            sigma_xl=sigma_xl,
            sigma_hm=sigma_hm,
        )
        for horizon in horizons:
            solution, targets = equilibrium.solve_equilibrium(
                parameters,
                initial_state,
                horizon=horizon,
            )
            if not solution.success:
                raise RuntimeError(
                    f"sigma={sigma_xl}, horizon={horizon}: {solution.message}"
                )
            initial = solution.sol(0.0)
            terminal = solution.sol(horizon)
            terminal_rates, terminal_block = equilibrium.equilibrium_rates(
                horizon, terminal, parameters
            )
            rows.append(
                {
                    "scenario": scenario,
                    "sigma_xl": sigma_xl,
                    "sigma_hm": sigma_hm,
                    "horizon": horizon,
                    "initial_log_consumption": float(initial[2]),
                    "initial_log_shadow_value": float(initial[3]),
                    "initial_consumption_share": math.exp(
                        float(initial[2])
                        - equilibrium.equilibrium_static_block(
                            float(initial[0]),
                            float(initial[1]),
                            0.0,
                            float(initial[3]),
                            parameters,
                        )["log_output"]
                    ),
                    "terminal_capital_growth": float(terminal_rates[0]),
                    "terminal_capability_growth": float(terminal_rates[1]),
                    "terminal_consumption_growth": float(terminal_rates[2]),
                    "target_aggregate_growth": targets["aggregate_growth"],
                    "target_capability_growth": targets["capability_growth"],
                    "terminal_consumption_share": math.exp(
                        float(terminal[2]) - terminal_block["log_output"]
                    ),
                    "target_consumption_share": targets["consumption_share"],
                    "mesh_nodes": solution.x.size,
                    "max_rms_residual": float(
                        np.max(solution.rms_residuals)
                    ),
                }
            )
    output = ROOT / "numerical" / "equilibrium_horizon_convergence.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
