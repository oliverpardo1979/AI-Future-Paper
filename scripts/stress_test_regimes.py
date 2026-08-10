"""Long-horizon falsification tests for the fixed-share numerical closure.

The script does not solve the optimizing equilibrium.  It varies the production
elasticity, the research-feedback denominator, and initial conditions to test
whether the illustrative closure behaves consistently with the paper's analytical
propositions.  A failure of a numerical test can reject an implementation or expose
a missing assumption; passing a test is not a proof of an equilibrium proposition.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import simulate_model as model  # noqa: E402


def feedback_denominators(parameters: model.Parameters) -> tuple[float, float]:
    upsilon = parameters.omega_x / (1.0 - parameters.omega_x)
    d_cd = (
        1.0
        - parameters.phi
        - parameters.eta * parameters.omega_m * upsilon
    )
    d_ai = 1.0 - parameters.phi - parameters.eta * upsilon
    return d_cd, d_ai


def tail_rows(
    rows: list[dict[str, float | str]], window: float = 100.0
) -> list[dict[str, float | str]]:
    final_time = float(rows[-1]["time"])
    selected = [row for row in rows if float(row["time"]) >= final_time - window]
    return selected if len(selected) >= 2 else rows[-min(2, len(rows)) :]


def average(rows: list[dict[str, float | str]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def endpoint_slope(rows: list[dict[str, float | str]], key: str) -> float:
    if len(rows) < 2:
        return math.nan
    elapsed = float(rows[-1]["time"]) - float(rows[0]["time"])
    if elapsed <= 0:
        return math.nan
    return (float(rows[-1][key]) - float(rows[0][key])) / elapsed


def summarize(
    scenario: str,
    feedback_case: str,
    initial_case: str,
    parameters: model.Parameters,
    initial_state: tuple[float, float, float],
    requested_horizon: float,
    step: float,
    rows: list[dict[str, float | str]],
) -> dict[str, float | str]:
    d_cd, d_ai = feedback_denominators(parameters)
    final = rows[-1]
    tail = tail_rows(rows)
    sigma_xl = parameters.sigma_xl
    low_sigma_ai_limit = (
        (1.0 - sigma_xl) / (1.0 - parameters.alpha * sigma_xl)
        if sigma_xl < 1.0
        else math.nan
    )
    low_sigma_fixed_share_capability_limit = (
        parameters.eta * parameters.n / (1.0 - parameters.phi)
        if sigma_xl < 1.0
        else math.nan
    )
    low_sigma_optimizing_capability_limit = (
        parameters.eta
        * parameters.n
        / (1.0 - parameters.phi + parameters.eta)
        if sigma_xl < 1.0
        else math.nan
    )
    low_sigma_optimizing_research_input_limit = (
        (1.0 - parameters.phi)
        * parameters.n
        / (1.0 - parameters.phi + parameters.eta)
        if sigma_xl < 1.0
        else math.nan
    )
    q = (1.0 - parameters.alpha) / parameters.alpha
    log_scaled_output_capital = (
        float(final["log_output"])
        - float(final["log_capital"])
        - q * float(final["log_capability"])
    )
    return {
        "scenario": scenario,
        "feedback_case": feedback_case,
        "initial_case": initial_case,
        "sigma_xl": sigma_xl,
        "sigma_hm": parameters.sigma_hm,
        "phi": parameters.phi,
        "eta": parameters.eta,
        "d_cd": d_cd,
        "d_ai": d_ai,
        "initial_capital": initial_state[0],
        "initial_capability": initial_state[1],
        "requested_horizon": requested_horizon,
        "step": step,
        "last_year": float(final["time"]),
        "stop_reason": str(final.get("stop_reason", "unreported")),
        "terminal_capability_growth": float(final["capability_growth"]),
        "terminal_output_per_capita_growth": float(
            final["output_per_capita_growth"]
        ),
        "terminal_consumption_per_capita_growth": float(
            final["consumption_per_capita_growth"]
        ),
        "terminal_wage_growth": float(final["wage_growth"]),
        "terminal_capital_growth": float(final["capital_growth"]),
        "terminal_net_capital_return": float(final["net_capital_return"]),
        "terminal_euler_gap": float(final["euler_gap"]),
        "terminal_ai_production_share": float(final["ai_share"]),
        "terminal_automated_research_share": float(
            final["automated_research_share"]
        ),
        "tail_capability_growth": average(tail, "capability_growth"),
        "tail_output_per_capita_growth": average(tail, "output_per_capita_growth"),
        "tail_consumption_per_capita_growth": average(
            tail, "consumption_per_capita_growth"
        ),
        "tail_wage_growth": average(tail, "wage_growth"),
        "tail_capital_growth": average(tail, "capital_growth"),
        "tail_net_capital_return": average(tail, "net_capital_return"),
        "tail_euler_gap": average(tail, "euler_gap"),
        "tail_ai_production_share": average(tail, "ai_share"),
        "tail_automated_research_share": average(
            tail, "automated_research_share"
        ),
        "tail_human_research_population_share": average(
            tail, "human_research_share"
        ),
        "tail_consumption_output_ratio": average(tail, "consumption_share"),
        "tail_capability_growth_slope": endpoint_slope(
            tail, "capability_growth"
        ),
        "tail_output_per_capita_growth_slope": endpoint_slope(
            tail, "output_per_capita_growth"
        ),
        "tail_net_capital_return_slope": endpoint_slope(
            tail, "net_capital_return"
        ),
        "low_sigma_theoretical_ai_share": low_sigma_ai_limit,
        "low_sigma_ai_share_gap": (
            float(final["ai_share"]) - low_sigma_ai_limit
            if sigma_xl < 1.0
            else math.nan
        ),
        "low_sigma_fixed_share_capability_growth": (
            low_sigma_fixed_share_capability_limit
        ),
        "low_sigma_fixed_share_capability_growth_gap": (
            float(final["capability_growth"])
            - low_sigma_fixed_share_capability_limit
            if sigma_xl < 1.0
            else math.nan
        ),
        "low_sigma_optimizing_candidate_capability_growth": (
            low_sigma_optimizing_capability_limit
        ),
        "low_sigma_optimizing_candidate_research_input_growth": (
            low_sigma_optimizing_research_input_limit
        ),
        "log_scaled_output_capital": log_scaled_output_capital,
        "max_abs_monopoly_foc_log_error": max(
            abs(float(row["monopoly_foc_log_error"])) for row in rows
        ),
        "max_abs_research_mix_log_error": max(
            abs(float(row["research_mix_log_error"])) for row in rows
        ),
        "minimum_consumption_output_ratio": min(
            float(row["consumption_share"]) for row in rows
        ),
        "max_abs_euler_gap": max(abs(float(row["euler_gap"])) for row in rows),
    }


def run_case(
    scenario: str,
    feedback_case: str,
    initial_case: str,
    parameters: model.Parameters,
    initial_state: tuple[float, float, float],
    horizon: float,
    step: float,
    acceleration_cutoff: float,
    max_log_state: float,
) -> dict[str, float | str]:
    rows = model.simulate(
        scenario,
        parameters,
        initial_state,
        horizon=horizon,
        step=step,
        acceleration_cutoff=acceleration_cutoff,
        max_log_state=max_log_state,
        catch_numerical_errors=True,
        report_stop_reason=True,
    )
    return summarize(
        scenario,
        feedback_case,
        initial_case,
        parameters,
        initial_state,
        horizon,
        step,
        rows,
    )


def build_cases(
    baseline: model.Parameters,
    initial_state: tuple[float, float, float],
    target_capability_growth: float,
) -> list[
    tuple[
        str,
        str,
        str,
        model.Parameters,
        tuple[float, float, float],
    ]
]:
    feedback_cases = {
        "stable": (0.6500, 0.45),
        "near_ai_boundary": (0.8675, 0.45),
        "ai_boundary": (0.8875, 0.45),
        "negative_ai_denominator": (0.9075, 0.45),
    }
    central_sigmas = (0.75, 0.99, 1.00, 1.01, 1.05, 1.10, 1.50, 2.00)
    feedback_sigmas = (1.00, 1.05, 2.00)
    cases: list[
        tuple[
            str,
            str,
            str,
            model.Parameters,
            tuple[float, float, float],
        ]
    ] = []

    calibrated: dict[tuple[str, float], model.Parameters] = {}
    for feedback_case, (phi, eta) in feedback_cases.items():
        sigmas = central_sigmas if feedback_case == "stable" else feedback_sigmas
        for sigma_xl in sigmas:
            parameters = replace(
                baseline, sigma_xl=sigma_xl, phi=phi, eta=eta
            )
            parameters = model.calibrate_research_productivity(
                parameters, initial_state, target_capability_growth
            )
            calibrated[(feedback_case, sigma_xl)] = parameters
            scenario = f"{feedback_case}_sigma_{sigma_xl:.3f}_baseline"
            cases.append(
                (
                    scenario,
                    feedback_case,
                    "baseline",
                    parameters,
                    initial_state,
                )
            )

    capital, capability, population = initial_state
    initial_variants = {
        "low_capital": (0.50 * capital, capability, population),
        "high_capital": (2.00 * capital, capability, population),
        "low_capability": (capital, 0.50 * capability, population),
        "high_capability": (capital, 2.00 * capability, population),
    }
    for sigma_xl in (0.75, 1.05, 2.00):
        parameters = calibrated[("stable", sigma_xl)]
        for initial_case, state in initial_variants.items():
            scenario = f"stable_sigma_{sigma_xl:.3f}_{initial_case}"
            cases.append(
                (scenario, "stable", initial_case, parameters, state)
            )
    return cases


def step_convergence_rows(
    baseline: model.Parameters,
    initial_state: tuple[float, float, float],
    target_capability_growth: float,
    acceleration_cutoff: float,
    max_log_state: float,
) -> list[dict[str, float | str]]:
    results: list[dict[str, float | str]] = []
    for sigma_xl in (0.75, 0.99, 1.00, 1.01, 1.05, 1.10, 2.00):
        parameters = replace(baseline, sigma_xl=sigma_xl)
        parameters = model.calibrate_research_productivity(
            parameters, initial_state, target_capability_growth
        )
        reference: dict[str, float | str] | None = None
        for step in (0.50, 1.00, 2.00, 5.00):
            scenario = f"step_sigma_{sigma_xl:.3f}_{step:.2f}"
            summary = run_case(
                scenario,
                "stable",
                "baseline",
                parameters,
                initial_state,
                horizon=200.0,
                step=step,
                acceleration_cutoff=acceleration_cutoff,
                max_log_state=max_log_state,
            )
            if reference is None:
                reference = summary
            results.append(
                {
                    "sigma_xl": sigma_xl,
                    "step": step,
                    "last_year": summary["last_year"],
                    "stop_reason": summary["stop_reason"],
                    "tail_capability_growth": summary[
                        "tail_capability_growth"
                    ],
                    "tail_output_per_capita_growth": summary[
                        "tail_output_per_capita_growth"
                    ],
                    "tail_net_capital_return": summary[
                        "tail_net_capital_return"
                    ],
                    "capability_growth_difference_from_half_year": (
                        float(summary["tail_capability_growth"])
                        - float(reference["tail_capability_growth"])
                    ),
                    "output_growth_difference_from_half_year": (
                        float(summary["tail_output_per_capita_growth"])
                        - float(reference["tail_output_per_capita_growth"])
                    ),
                    "net_return_difference_from_half_year": (
                        float(summary["tail_net_capital_return"])
                        - float(reference["tail_net_capital_return"])
                    ),
                }
            )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=float, default=2000.0)
    parser.add_argument("--step", type=float, default=2.0)
    parser.add_argument("--acceleration-cutoff", type=float, default=0.50)
    parser.add_argument("--max-log-state", type=float, default=200.0)
    parser.add_argument("--skip-step-convergence", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline, analytical = model.analytical_calibration(model.Parameters())
    initial_capital = model.calibrate_initial_capital(
        baseline, analytical["capital_output_ratio"]
    )
    initial_state = (initial_capital, 1.0, 1.0)
    baseline = model.calibrate_research_productivity(
        baseline, initial_state, analytical["capability_growth"]
    )
    cases = build_cases(
        baseline, initial_state, analytical["capability_growth"]
    )

    summaries: list[dict[str, float | str]] = []
    for index, case in enumerate(cases, start=1):
        scenario, feedback_case, initial_case, parameters, state = case
        print(f"[{index:02d}/{len(cases):02d}] {scenario}", flush=True)
        summaries.append(
            run_case(
                scenario,
                feedback_case,
                initial_case,
                parameters,
                state,
                args.horizon,
                args.step,
                args.acceleration_cutoff,
                args.max_log_state,
            )
        )
    model.write_rows(ROOT / "numerical" / "regime_stress_summary.csv", summaries)

    if not args.skip_step_convergence:
        print("Running step-size convergence checks", flush=True)
        convergence = step_convergence_rows(
            baseline,
            initial_state,
            analytical["capability_growth"],
            args.acceleration_cutoff,
            args.max_log_state,
        )
        model.write_rows(
            ROOT / "numerical" / "regime_stress_step_convergence.csv",
            convergence,
        )


if __name__ == "__main__":
    main()
