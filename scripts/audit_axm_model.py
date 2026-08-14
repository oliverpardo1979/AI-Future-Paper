"""Reproducible algebraic and numerical audit for the A*M paper."""

from __future__ import annotations

import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "numerical_axm" / "equilibrium_transition_summary.csv"
HORIZON_AUDIT = ROOT / "numerical_axm" / "equilibrium_horizon_robustness.csv"
REPORT = ROOT / "numerical_axm" / "audit_report.csv"


def analytical_checks() -> list[dict[str, str | float]]:
    alpha = 0.33
    omega_x = 0.20
    omega_m = 0.35
    eta = 0.45
    population_growth = 0.012
    discount_rate = 0.04
    nu = omega_x / (1.0 - omega_x)
    theta = (1.0 - alpha) / alpha
    d_cd = 1.0 - eta * omega_m * (1.0 + nu)
    d_ai = 1.0 - eta * (1.0 + nu)
    delta = 1.0 + theta - eta
    h = eta * alpha / delta
    inference_share = (1.0 - alpha) ** 2
    investment_share = alpha - theta * h
    research_share = eta * inference_share / delta
    consumption_share = (
        1.0 - inference_share - investment_share - research_share
    )
    threshold = 1.0 / (alpha * eta)
    sigma_hm = 2.0
    ces_power = (sigma_hm - 1.0) / sigma_hm
    capability = 2.0
    human_research = 0.8
    research_compute = 0.3
    original_compute_cost = 1.7
    original_inference_compute = 0.4
    original_research_productivity = 0.02
    normalized_capability = capability / original_compute_cost
    normalized_inference_compute = (
        original_compute_cost * original_inference_compute
    )
    normalized_research_productivity = (
        original_research_productivity / original_compute_cost
    )
    service_before_normalization = capability * original_inference_compute
    service_after_normalization = (
        normalized_capability * normalized_inference_compute
    )
    auxiliary_research_input_index = 0.6
    normalized_capability_flow = (
        normalized_research_productivity
        * auxiliary_research_input_index**eta
    )
    rescaled_original_capability_flow = (
        original_research_productivity
        * auxiliary_research_input_index**eta
        / original_compute_cost
    )

    def log_capability_flow(log_capability: float) -> float:
        current_capability = math.exp(log_capability)
        human_term = (1.0 - omega_m) * human_research**ces_power
        machine_term = omega_m * (
            current_capability * research_compute
        ) ** ces_power
        log_research_input_index = math.log(
            (human_term + machine_term) ** (1.0 / ces_power)
        )
        return eta * log_research_input_index

    finite_difference_step = 1e-6
    log_capability = math.log(capability)
    numerical_elasticity = (
        log_capability_flow(log_capability + finite_difference_step)
        - log_capability_flow(log_capability - finite_difference_step)
    ) / (2.0 * finite_difference_step)
    human_term = (1.0 - omega_m) * human_research**ces_power
    machine_term = omega_m * (
        capability * research_compute
    ) ** ces_power
    research_input_index = (human_term + machine_term) ** (1.0 / ces_power)
    capability_flow_via_index = (
        original_research_productivity * research_input_index**eta
    )
    capability_flow_direct = original_research_productivity * (
        human_term + machine_term
    ) ** (eta / ces_power)
    automated_contribution = machine_term / (human_term + machine_term)
    envelope_elasticity = eta * automated_contribution
    assertions = {
        "D_CD_positive": d_cd > 0.0,
        "D_AI_positive": d_ai > 0.0,
        "singular_investment_share_positive": investment_share > 0.0,
        "singular_research_share_positive": research_share > 0.0,
        "singular_consumption_share_positive": consumption_share > 0.0,
        "singular_resource_shares_sum_to_one": math.isclose(
            inference_share
            + investment_share
            + research_share
            + consumption_share,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "wage_threshold_above_one": threshold > 1.0,
        "capability_envelope_derivative": math.isclose(
            numerical_elasticity,
            envelope_elasticity,
            rel_tol=1e-8,
            abs_tol=1e-8,
        ),
        "generalized_ces_single_equation": math.isclose(
            capability_flow_via_index,
            capability_flow_direct,
            rel_tol=0.0,
            abs_tol=1e-14,
        ),
        "compute_cost_normalization": math.isclose(
            service_before_normalization,
            service_after_normalization,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "compute_cost_normalization_capability_law": math.isclose(
            normalized_capability_flow,
            rescaled_original_capability_flow,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "transversality_discount_positive": (
            discount_rate - population_growth > 0.0
        ),
    }
    if not all(assertions.values()):
        raise AssertionError(assertions)
    return [
        {"object": "D_CD", "value": d_cd, "status": "pass"},
        {"object": "D_AI", "value": d_ai, "status": "pass"},
        {"object": "singular_h", "value": h, "status": "pass"},
        {
            "object": "singular_inference_share",
            "value": inference_share,
            "status": "pass",
        },
        {
            "object": "singular_investment_share",
            "value": investment_share,
            "status": "pass",
        },
        {
            "object": "singular_research_share",
            "value": research_share,
            "status": "pass",
        },
        {
            "object": "singular_consumption_share",
            "value": consumption_share,
            "status": "pass",
        },
        {
            "object": "wage_sign_threshold_sigma_XL",
            "value": threshold,
            "status": "pass",
        },
        {
            "object": "capability_envelope_derivative_error",
            "value": abs(numerical_elasticity - envelope_elasticity),
            "status": "pass",
        },
        {
            "object": "generalized_ces_single_equation_error",
            "value": abs(
                capability_flow_via_index - capability_flow_direct
            ),
            "status": "pass",
        },
        {
            "object": "compute_cost_normalization_service_error",
            "value": abs(
                service_before_normalization
                - service_after_normalization
            ),
            "status": "pass",
        },
        {
            "object": "compute_cost_normalization_capability_law_error",
            "value": abs(
                normalized_capability_flow
                - rescaled_original_capability_flow
            ),
            "status": "pass",
        },
        {
            "object": "balanced_growth_transversality_rate",
            "value": discount_rate - population_growth,
            "status": "pass",
        },
    ]


def numerical_checks() -> list[dict[str, str | float]]:
    with SUMMARY.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 2:
        raise AssertionError(f"Expected two verified paths, found {len(rows)}.")
    report: list[dict[str, str | float]] = []
    for row in rows:
        name = row["scenario"]
        tests = {
            "resource": float(row["max_abs_resource_residual"]) < 1e-10,
            "technologies": float(row["max_abs_technology_log_error"])
            < 1e-10,
            "monopoly_foc": float(
                row["max_abs_monopoly_foc_log_error"]
            )
            < 1e-9,
            "research_compute_foc": float(
                row["max_abs_research_compute_foc_log_error"]
            )
            < 1e-9,
            "research_human_foc": float(
                row["max_abs_research_human_foc_log_error"]
            )
            < 1e-8,
            "labor_market": float(row["max_abs_labor_market_error"]) < 1e-10,
            "dynamic_path": max(
                float(row["max_abs_capital_law_residual"]),
                float(row["max_abs_capability_law_residual"]),
                float(row["max_abs_consumption_path_residual"]),
                float(row["max_abs_shadow_costate_residual"]),
            )
            < 2e-5,
            "positive_consumption": float(row["minimum_consumption_share"]) > 0,
            "monopoly_second_order": float(
                row["minimum_monopoly_soc_margin"]
            )
            > 0,
            "terminal_capability_growth_target": abs(
                float(row["terminal_capability_growth"])
                - float(row["target_capability_growth"])
            )
            < 5e-5,
            "terminal_output_growth_target": abs(
                float(row["terminal_output_per_capita_growth"])
                - (
                    float(row["target_aggregate_growth"])
                    - 0.012
                )
            )
            < 5e-5,
        }
        if not all(tests.values()):
            raise AssertionError({name: tests})
        for test, passed in tests.items():
            report.append(
                {
                    "object": f"{name}:{test}",
                    "value": float(passed),
                    "status": "pass",
                }
            )
    return report


def horizon_checks() -> list[dict[str, str | float]]:
    with HORIZON_AUDIT.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["sigma_hm"], []).append(row)
    if set(grouped) != {"1.0", "2.0"}:
        raise AssertionError(f"Unexpected horizon groups: {sorted(grouped)}")

    report: list[dict[str, str | float]] = []
    for sigma_hm, group in grouped.items():
        if len(group) != 3:
            raise AssertionError(
                f"Expected three horizons for sigma_HM={sigma_hm}."
            )
        consumptions = [
            float(row["initial_log_consumption"]) for row in group
        ]
        shadows = [
            float(row["initial_log_shadow_value"]) for row in group
        ]
        residuals = [float(row["max_rms_residual"]) for row in group]
        values = {
            "initial_consumption_range": max(consumptions) - min(consumptions),
            "initial_shadow_range": max(shadows) - min(shadows),
            "maximum_solver_residual": max(residuals),
        }
        tests = {
            "initial_consumption_range": values[
                "initial_consumption_range"
            ]
            < 2e-6,
            "initial_shadow_range": values["initial_shadow_range"] < 2e-6,
            "maximum_solver_residual": values[
                "maximum_solver_residual"
            ]
            < 2e-5,
        }
        if not all(tests.values()):
            raise AssertionError({f"sigma_HM={sigma_hm}": values})
        for test, passed in tests.items():
            report.append(
                {
                    "object": f"sigma_HM={sigma_hm}:horizon_{test}",
                    "value": values[test],
                    "status": "pass" if passed else "fail",
                }
            )
    return report


def main() -> None:
    rows = analytical_checks() + numerical_checks() + horizon_checks()
    with REPORT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["object", "value", "status"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} checks passed; wrote {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
