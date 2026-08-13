"""Audit the published numerical paths against the equilibrium system.

The script reads the paths produced directly by the boundary-value solvers and
recomputes the algebraic identities in equation (22), the dynamic residuals of
the four endogenous states, the static first-order conditions, the monopoly
second-order margin, and terminal-value diagnostics.  It writes a scenario
summary and a long equation-by-equation audit table.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NUMERICAL = ROOT / "numerical"

ALPHA = 0.33
OMEGA_X = 0.20
OMEGA_M = 0.35
POPULATION_GROWTH = 0.012
DEPRECIATION = 0.05
DISCOUNT = 0.04
PHI = 0.65
ETA = 0.45

SIGMA_BY_SCENARIO = {
    "equilibrium_sigma_0_75": 0.75,
    "equilibrium_sigma_1_00": 1.00,
    "equilibrium_sigma_1_00_hm_1_00": 1.00,
    "equilibrium_sigma_1_35": 1.35,
    "equilibrium_sigma_1_50": 1.50,
    "equilibrium_sigma_2_00": 2.00,
}

SIGMA_HM_BY_SCENARIO = {
    "equilibrium_sigma_1_00_hm_1_00": 1.00,
}

HIGH_SIGMA_TERMINAL_Z = {1.35: 50.0, 1.50: 50.0, 2.00: 100.0}


def logsumexp(left: float, right: float) -> float:
    maximum = max(left, right)
    return maximum + math.log(
        math.exp(left - maximum) + math.exp(right - maximum)
    )


def ces_log_aggregate(
    log_left: float,
    log_right: float,
    left_weight: float,
    elasticity: float,
) -> float:
    if abs(elasticity - 1.0) <= 1e-12:
        return left_weight * log_left + (1.0 - left_weight) * log_right
    power = (elasticity - 1.0) / elasticity
    return logsumexp(
        math.log(left_weight) + power * log_left,
        math.log1p(-left_weight) + power * log_right,
    ) / power


def load_paths() -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for path in (
        NUMERICAL / "equilibrium_transition_paths.csv",
        NUMERICAL / "high_sigma_equilibrium_paths.csv",
    ):
        with path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                grouped[row["scenario"]].append(row)
    return dict(grouped)


def maximum_absolute(rows: list[dict[str, str]], field: str) -> float:
    return max(abs(float(row[field])) for row in rows)


def minimum_value(rows: list[dict[str, str]], field: str) -> float:
    return min(float(row[field]) for row in rows)


def maximum_value(rows: list[dict[str, str]], field: str) -> float:
    return max(float(row[field]) for row in rows)


def full_domain_monopoly_margin(sigma_xl: float) -> float:
    """Minimum normalized -MR' over the economically relevant share domain."""

    if abs(sigma_xl - 1.0) <= 1e-12:
        shares = [OMEGA_X]
    else:
        shares = [index / 100_000.0 for index in range(1, 100_000)]
    margins = []
    for share in shares:
        elasticity = (1.0 - share) / sigma_xl + ALPHA * share
        if not 0.0 < elasticity < 1.0:
            continue
        elasticity_derivative = (
            (ALPHA - 1.0 / sigma_xl)
            * (1.0 - 1.0 / sigma_xl)
            * share
            * (1.0 - share)
        )
        margins.append(
            elasticity * (1.0 - elasticity) + elasticity_derivative
        )
    if not margins:
        raise ValueError(f"No feasible monopoly share for sigma={sigma_xl}.")
    return min(margins)


def trapezoid_integral(times: list[float], values: list[float]) -> float:
    return sum(
        0.5 * (values[index] + values[index - 1])
        * (times[index] - times[index - 1])
        for index in range(1, len(times))
    )


def audit_scenario(
    scenario: str,
    rows: list[dict[str, str]],
) -> tuple[dict[str, float | str], list[dict[str, float | str]]]:
    sigma_xl = SIGMA_BY_SCENARIO[scenario]
    sigma_hm = SIGMA_HM_BY_SCENARIO.get(scenario, 2.00)
    production_errors = []
    research_errors = []
    service_errors = []
    share_errors = []
    inverse_elasticity_errors = []
    capital_price_errors = []
    wage_errors = []
    ai_price_errors = []
    final_firm_zero_profit_errors = []
    household_budget_errors = []
    monopoly_rent_errors = []
    population_errors = []
    per_capita_consumption_errors = []

    initial_time = float(rows[0]["time"])
    initial_log_population = float(rows[0]["log_population"])

    for row in rows:
        log_capital = float(row["log_capital"])
        log_capability = float(row["log_capability"])
        log_output = float(row["log_output"])
        log_labor = float(row["log_production_labor"])
        log_human_research = float(row["log_human_research"])
        log_ai_services = float(row["log_ai_services"])
        log_inference_compute = float(row["log_inference_compute"])
        log_automated_research = float(row["log_automated_research"])
        log_effective_research = float(row["log_effective_research"])
        time = float(row["time"])

        population_errors.append(
            float(row["log_population"])
            - initial_log_population
            - POPULATION_GROWTH * (time - initial_time)
        )
        per_capita_consumption_errors.append(
            float(row["log_consumption_per_capita"])
            - float(row["log_consumption"])
            + float(row["log_population"])
        )

        log_service_composite = ces_log_aggregate(
            log_labor,
            log_ai_services,
            1.0 - OMEGA_X,
            sigma_xl,
        )
        production_errors.append(
            log_output
            - ALPHA * log_capital
            - (1.0 - ALPHA) * log_service_composite
        )
        log_research_composite = ces_log_aggregate(
            log_human_research,
            log_automated_research,
            1.0 - OMEGA_M,
            sigma_hm,
        )
        research_errors.append(
            log_effective_research - log_research_composite
        )
        service_errors.append(
            log_ai_services - log_capability - log_inference_compute
        )

        if abs(sigma_xl - 1.0) <= 1e-12:
            predicted_share = OMEGA_X
        else:
            power = (sigma_xl - 1.0) / sigma_xl
            log_ai_term = math.log(OMEGA_X) + power * log_ai_services
            denominator = logsumexp(
                math.log1p(-OMEGA_X) + power * log_labor,
                log_ai_term,
            )
            predicted_share = math.exp(log_ai_term - denominator)
        share = float(row["ai_share"])
        share_errors.append(share - predicted_share)
        predicted_inverse_elasticity = (
            (1.0 - share) / sigma_xl + ALPHA * share
        )
        inverse_elasticity_errors.append(
            float(row["inverse_demand_elasticity"])
            - predicted_inverse_elasticity
        )

        log_capital_return = math.log(float(row["gross_capital_return"]))
        capital_price_errors.append(
            log_capital_return
            - math.log(ALPHA)
            - log_output
            + log_capital
        )
        predicted_log_wage = (
            math.log1p(-ALPHA)
            + math.log1p(-share)
            + log_output
            - log_labor
        )
        wage_errors.append(float(row["log_wage"]) - predicted_log_wage)
        predicted_log_price = (
            math.log1p(-ALPHA)
            + math.log(share)
            + log_output
            - log_ai_services
        )
        ai_price_errors.append(
            float(row["log_ai_price"]) - predicted_log_price
        )
        capital_income_share = float(row["gross_capital_return"]) * math.exp(
            log_capital - log_output
        )
        production_labor_income_share = math.exp(
            float(row["log_wage"]) + log_labor - log_output
        )
        ai_revenue_share = (1.0 - ALPHA) * share
        final_firm_zero_profit_errors.append(
            capital_income_share
            + production_labor_income_share
            + ai_revenue_share
            - 1.0
        )
        profit_share = ai_revenue_share - float(row["inference_share"])
        monopoly_rent_errors.append(
            profit_share - predicted_inverse_elasticity * ai_revenue_share
        )
        capital_output_ratio = math.exp(log_capital - log_output)
        household_budget_errors.append(
            float(row["capital_growth"]) * capital_output_ratio
            - (
                (float(row["gross_capital_return"]) - DEPRECIATION)
                * capital_output_ratio
                + float(row["aggregate_labor_share"])
                + profit_share
                - float(row["aggregate_labor_share"])
                * float(row["human_research_share"])
                - float(row["research_resource_share"])
                - float(row["consumption_share"])
            )
        )

    algebraic = {
        "population_log_law": max(abs(value) for value in population_errors),
        "per_capita_consumption_identity": max(
            abs(value) for value in per_capita_consumption_errors
        ),
        "production_log_identity": max(abs(value) for value in production_errors),
        "research_ces_log_identity": max(abs(value) for value in research_errors),
        "ai_service_log_identity": max(abs(value) for value in service_errors),
        "ai_share_identity": max(abs(value) for value in share_errors),
        "inverse_elasticity_identity": max(
            abs(value) for value in inverse_elasticity_errors
        ),
        "capital_price_log_identity": max(
            abs(value) for value in capital_price_errors
        ),
        "wage_log_identity": max(abs(value) for value in wage_errors),
        "ai_price_log_identity": max(abs(value) for value in ai_price_errors),
        "final_firm_zero_profit": max(
            abs(value) for value in final_firm_zero_profit_errors
        ),
        "household_budget": max(abs(value) for value in household_budget_errors),
        "monopoly_rent_identity": max(
            abs(value) for value in monopoly_rent_errors
        ),
    }
    dynamic = {
        "capital_law": maximum_absolute(rows, "capital_law_residual"),
        "capability_law": maximum_absolute(rows, "capability_law_residual"),
        "consumption_euler_path": maximum_absolute(
            rows, "consumption_euler_path_residual"
        ),
        "shadow_costate": maximum_absolute(rows, "shadow_costate_residual"),
    }
    static = {
        "resource_constraint": max(
            abs(float(row["resource_share_sum"]) - 1.0) for row in rows
        ),
        "labor_market": maximum_absolute(rows, "labor_market_error"),
        "monopoly_foc_log": maximum_absolute(rows, "monopoly_foc_log_error"),
        "human_research_foc_log": maximum_absolute(
            rows, "research_human_foc_log_error"
        ),
        "machine_research_foc_log": maximum_absolute(
            rows, "research_compute_foc_log_error"
        ),
        "euler_formula": maximum_absolute(rows, "euler_residual"),
    }

    times = [float(row["time"]) for row in rows]
    net_returns = [
        float(row["gross_capital_return"]) - DEPRECIATION for row in rows
    ]
    initial = rows[0]
    final = rows[-1]
    household_tvc_log_change = (
        -(DISCOUNT - POPULATION_GROWTH) * times[-1]
        + float(final["log_capital"])
        - float(final["log_consumption"])
        - float(initial["log_capital"])
        + float(initial["log_consumption"])
    )
    developer_tvc_log_change = (
        -trapezoid_integral(times, net_returns)
        + float(final["log_shadow_value"])
        + float(final["log_capability"])
        - float(initial["log_shadow_value"])
        - float(initial["log_capability"])
    )

    if sigma_xl < 1.0:
        target_consumption_share = 1.0 - ALPHA * (
            POPULATION_GROWTH + DEPRECIATION
        ) / (DISCOUNT + DEPRECIATION)
        research_input_growth = (
            (1.0 - PHI) * POPULATION_GROWTH / (1.0 - PHI + ETA)
        )
        target_profit_derivative_to_shadow = DISCOUNT - (
            1.0 - ETA
        ) * research_input_growth
        terminal_conditions = {
            "terminal_consumption_share": abs(
                float(final["consumption_share"])
                - target_consumption_share
            ),
            "terminal_profit_derivative_to_shadow": abs(
                math.exp(
                    float(final["log_ai_services"])
                    - 2.0 * float(final["log_capability"])
                    - float(final["log_shadow_value"])
                )
                - target_profit_derivative_to_shadow
            ),
        }
    elif abs(sigma_xl - 1.0) <= 1e-12:
        beta = (1.0 - ALPHA) * OMEGA_X
        output_feedback = beta / (1.0 - ALPHA - beta)
        research_feedback_weight = OMEGA_M if sigma_hm == 1.0 else 1.0
        capability_growth = ETA * POPULATION_GROWTH / (
            1.0
            - PHI
            - ETA * research_feedback_weight * output_feedback
        )
        per_capita_growth = output_feedback * capability_growth
        research_resource_share = (
            beta**2
            * ETA
            * research_feedback_weight
            * capability_growth
            / (
                DISCOUNT
                - POPULATION_GROWTH
                + (1.0 - PHI) * capability_growth
            )
        )
        investment_share = ALPHA * (
            POPULATION_GROWTH + per_capita_growth + DEPRECIATION
        ) / (DISCOUNT + DEPRECIATION + per_capita_growth)
        target_consumption_share = (
            1.0 - investment_share - beta**2 - research_resource_share
        )
        target_shadow_capability_to_output = research_resource_share / (
            ETA * research_feedback_weight * capability_growth
        )
        terminal_conditions = {
            "terminal_consumption_share": abs(
                float(final["consumption_share"])
                - target_consumption_share
            ),
            "terminal_shadow_capability_to_output": abs(
                math.exp(
                    float(final["log_shadow_value"])
                    + float(final["log_capability"])
                    - float(final["log_output"])
                )
                - target_shadow_capability_to_output
            ),
        }
    else:
        capability_exponent = (1.0 - ALPHA) / ALPHA
        inference_share_limit = (1.0 - ALPHA) ** 2
        capability_growth_to_z = ETA * ALPHA / (
            1.0 - PHI + capability_exponent
        )
        research_resource_limit = (
            ETA
            * inference_share_limit
            / (1.0 - PHI + capability_exponent)
        )
        investment_limit = ALPHA - (
            capability_exponent * capability_growth_to_z
        )
        target_consumption_share = (
            1.0
            - inference_share_limit
            - research_resource_limit
            - investment_limit
        )
        target_shadow_capability_to_capital = inference_share_limit / (
            ETA * ALPHA
        )
        terminal_conditions = {
            "terminal_output_capital_ratio": abs(
                math.exp(
                    float(final["log_output"])
                    - float(final["log_capital"])
                )
                - HIGH_SIGMA_TERMINAL_Z[sigma_xl]
            ),
            "terminal_consumption_share": abs(
                float(final["consumption_share"])
                - target_consumption_share
            ),
            "terminal_shadow_capability_to_capital": abs(
                math.exp(
                    float(final["log_shadow_value"])
                    + float(final["log_capability"])
                    - float(final["log_capital"])
                )
                - target_shadow_capability_to_capital
            ),
        }

    summary: dict[str, float | str] = {
        "scenario": scenario,
        "sigma_xl": sigma_xl,
        "sigma_hm": sigma_hm,
        "observations": len(rows),
        "max_algebraic_identity_residual": max(algebraic.values()),
        "max_dynamic_path_residual": max(dynamic.values()),
        "max_static_equilibrium_residual": max(static.values()),
        "max_terminal_condition_residual": max(terminal_conditions.values()),
        "minimum_monopoly_soc_margin_on_path": minimum_value(
            rows, "monopoly_soc_margin"
        ),
        "minimum_monopoly_soc_margin_full_domain": full_domain_monopoly_margin(
            sigma_xl
        ),
        "minimum_consumption_share": minimum_value(rows, "consumption_share"),
        "minimum_investment_share": minimum_value(rows, "investment_share"),
        "minimum_inference_share": minimum_value(rows, "inference_share"),
        "minimum_research_resource_share": minimum_value(
            rows, "research_resource_share"
        ),
        "minimum_ai_share": minimum_value(rows, "ai_share"),
        "maximum_ai_share": maximum_value(rows, "ai_share"),
        "minimum_human_research_share": minimum_value(
            rows, "human_research_share"
        ),
        "maximum_human_research_share": maximum_value(
            rows, "human_research_share"
        ),
        "maximum_inverse_demand_elasticity": maximum_value(
            rows, "inverse_demand_elasticity"
        ),
        "household_tvc_log_change": household_tvc_log_change,
        "developer_tvc_log_change": developer_tvc_log_change,
        "terminal_capital_to_consumption": math.exp(
            float(final["log_capital"]) - float(final["log_consumption"])
        ),
        "terminal_shadow_capability_to_consumption": math.exp(
            float(final["log_shadow_value"])
            + float(final["log_capability"])
            - float(final["log_consumption"])
        ),
    }

    long_rows: list[dict[str, float | str]] = []
    for group, metrics, tolerance in (
        ("algebraic identity", algebraic, 1e-9),
        ("dynamic equation", dynamic, 2e-5),
        ("static equilibrium condition", static, 1e-8),
    ):
        for equation, value in metrics.items():
            long_rows.append(
                {
                    "scenario": scenario,
                    "sigma_xl": sigma_xl,
                    "group": group,
                    "equation": equation,
                    "maximum_absolute_residual": value,
                    "tolerance": tolerance,
                    "passes": value <= tolerance,
                }
            )
    long_rows.extend(
        [
            {
                "scenario": scenario,
                "sigma_xl": sigma_xl,
                "group": "optimality inequality",
                "equation": "monopoly_soc_on_path",
                "maximum_absolute_residual": summary[
                    "minimum_monopoly_soc_margin_on_path"
                ],
                "tolerance": 0.0,
                "passes": summary["minimum_monopoly_soc_margin_on_path"] > 0.0,
            },
            {
                "scenario": scenario,
                "sigma_xl": sigma_xl,
                "group": "optimality inequality",
                "equation": "monopoly_soc_full_domain",
                "maximum_absolute_residual": summary[
                    "minimum_monopoly_soc_margin_full_domain"
                ],
                "tolerance": 0.0,
                "passes": summary[
                    "minimum_monopoly_soc_margin_full_domain"
                ]
                > 0.0,
            },
            {
                "scenario": scenario,
                "sigma_xl": sigma_xl,
                "group": "interiority",
                "equation": "positive_consumption_share",
                "maximum_absolute_residual": summary[
                    "minimum_consumption_share"
                ],
                "tolerance": 0.0,
                "passes": summary["minimum_consumption_share"] > 0.0,
            },
            {
                "scenario": scenario,
                "sigma_xl": sigma_xl,
                "group": "interiority",
                "equation": "positive_investment_share",
                "maximum_absolute_residual": summary[
                    "minimum_investment_share"
                ],
                "tolerance": 0.0,
                "passes": summary["minimum_investment_share"] > 0.0,
            },
            {
                "scenario": scenario,
                "sigma_xl": sigma_xl,
                "group": "interiority",
                "equation": "positive_compute_uses",
                "maximum_absolute_residual": min(
                    summary["minimum_inference_share"],
                    summary["minimum_research_resource_share"],
                ),
                "tolerance": 0.0,
                "passes": min(
                    summary["minimum_inference_share"],
                    summary["minimum_research_resource_share"],
                )
                > 0.0,
            },
            {
                "scenario": scenario,
                "sigma_xl": sigma_xl,
                "group": "interiority",
                "equation": "interior_ai_share",
                "maximum_absolute_residual": min(
                    summary["minimum_ai_share"],
                    1.0 - summary["maximum_ai_share"],
                ),
                "tolerance": 0.0,
                "passes": summary["minimum_ai_share"] > 0.0
                and summary["maximum_ai_share"] < 1.0,
            },
            {
                "scenario": scenario,
                "sigma_xl": sigma_xl,
                "group": "interiority",
                "equation": "interior_human_research_share",
                "maximum_absolute_residual": min(
                    summary["minimum_human_research_share"],
                    1.0 - summary["maximum_human_research_share"],
                ),
                "tolerance": 0.0,
                "passes": summary["minimum_human_research_share"] > 0.0
                and summary["maximum_human_research_share"] < 1.0,
            },
            {
                "scenario": scenario,
                "sigma_xl": sigma_xl,
                "group": "interiority",
                "equation": "inverse_demand_elasticity_below_one",
                "maximum_absolute_residual": 1.0
                - summary["maximum_inverse_demand_elasticity"],
                "tolerance": 0.0,
                "passes": summary["maximum_inverse_demand_elasticity"] < 1.0,
            },
        ]
    )
    for equation, value in terminal_conditions.items():
        long_rows.append(
            {
                "scenario": scenario,
                "sigma_xl": sigma_xl,
                "group": "terminal condition",
                "equation": equation,
                "maximum_absolute_residual": value,
                "tolerance": 1e-6,
                "passes": value <= 1e-6,
            }
        )
    return summary, long_rows


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_boundary_convergence() -> None:
    sources = (
        (
            1.35,
            NUMERICAL
            / "high_sigma_equilibrium_1_35_boundary_audit_free_continuation.csv",
        ),
        (
            1.50,
            NUMERICAL / "high_sigma_equilibrium_1_5_free_continuation.csv",
        ),
        (
            2.00,
            NUMERICAL / "high_sigma_equilibrium_2_extreme_free_continuation.csv",
        ),
    )
    capability_exponent = (1.0 - ALPHA) / ALPHA
    capability_growth_to_z = ETA * ALPHA / (
        1.0 - PHI + capability_exponent
    )
    singularity_rate = capability_exponent * capability_growth_to_z
    rows: list[dict[str, float | str]] = []
    for sigma_xl, path in sources:
        with path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                terminal_z = float(row["terminal_output_capital_ratio"])
                duration = float(row["duration"])
                rows.append(
                    {
                        "sigma_xl": sigma_xl,
                        "terminal_output_capital_ratio": terminal_z,
                        "duration": duration,
                        "estimated_singularity_time": duration
                        + 1.0 / (singularity_rate * terminal_z),
                        "initial_log_consumption": row[
                            "initial_log_consumption"
                        ],
                        "initial_log_shadow_value": row[
                            "initial_log_shadow_value"
                        ],
                        "mesh_nodes": row["mesh_nodes"],
                        "max_rms_residual": row["max_rms_residual"],
                    }
                )
    write_csv(NUMERICAL / "high_sigma_boundary_convergence.csv", rows)


def main() -> None:
    summaries = []
    audit_rows = []
    for scenario, rows in sorted(load_paths().items()):
        summary, scenario_audit = audit_scenario(scenario, rows)
        summaries.append(summary)
        audit_rows.extend(scenario_audit)
    write_csv(NUMERICAL / "equilibrium_system_audit_summary.csv", summaries)
    write_csv(NUMERICAL / "equilibrium_system_audit.csv", audit_rows)
    write_boundary_convergence()
    failed = [row for row in audit_rows if row["passes"] is False]
    if failed:
        raise SystemExit(f"Equilibrium audit failed for {len(failed)} conditions.")
    print(f"Audited {len(summaries)} equilibrium paths; all checks passed.")


if __name__ == "__main__":
    main()
