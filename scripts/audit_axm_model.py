"""Reproducible algebraic and numerical audit for the A*M paper."""

from __future__ import annotations

import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "numerical_axm" / "equilibrium_transition_summary.csv"
HORIZON_AUDIT = ROOT / "numerical_axm" / "equilibrium_horizon_robustness.csv"
PATHS = ROOT / "numerical_axm" / "equilibrium_transition_paths.csv"
HORIZON_PATHS = ROOT / "numerical_axm" / "equilibrium_horizon_paths.csv"
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
    singular_denominator = 1.0 + theta - eta
    h = eta * alpha / singular_denominator
    inference_share = (1.0 - alpha) ** 2
    investment_share = alpha - theta * h
    research_share = eta * inference_share / singular_denominator
    consumption_share = (
        1.0 - inference_share - investment_share - research_share
    )
    threshold = 1.0 / (alpha * eta)
    beta = (1.0 - alpha) * omega_x
    cd_research_concavity_sum = eta * (1.0 + omega_m)
    sigma_two_research_curvature = 2.0 * eta
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
        "reported_monopoly_capability_concavity": beta <= 0.5,
        "reported_cd_research_concavity": cd_research_concavity_sum <= 1.0,
        "reported_sigma_two_research_concavity": (
            sigma_two_research_curvature <= 1.0
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
            "object": "wage_sign_threshold_min_sigma",
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
        {
            "object": "reported_monopoly_beta",
            "value": beta,
            "status": "pass",
        },
        {
            "object": "reported_cd_research_concavity_sum",
            "value": cd_research_concavity_sum,
            "status": "pass",
        },
        {
            "object": "reported_sigma_two_research_curvature",
            "value": sigma_two_research_curvature,
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
            "factor_prices": float(row["max_abs_factor_price_error"])
            < 1e-10,
            "share_definitions": float(
                row["max_abs_share_definition_error"]
            )
            < 1e-10,
            "research_duality": float(row["max_abs_research_dual_error"])
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


def path_specification_checks() -> list[dict[str, str | float]]:
    """Reconstruct the consolidated A*M research block from every saved row."""

    with PATHS.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with HORIZON_PATHS.open(newline="", encoding="utf-8") as handle:
        horizon_rows = list(csv.DictReader(handle))
    if not rows:
        raise AssertionError("The saved equilibrium-path file is empty.")
    if not horizon_rows:
        raise AssertionError("The saved horizon-path file is empty.")
    if len(rows) != 1402:
        raise AssertionError(f"Expected 1,402 baseline rows, found {len(rows)}.")
    if len(horizon_rows) != 4206:
        raise AssertionError(
            f"Expected 4,206 horizon-audit rows, found {len(horizon_rows)}."
        )

    scenario_sigma_hm = {
        "axm_sigma_xl_1_hm_1": 1.0,
        "axm_sigma_xl_1_hm_2": 2.0,
    }
    if {row["scenario"] for row in rows} != set(scenario_sigma_hm):
        raise AssertionError("Unexpected scenarios in the saved A*M paths.")
    expected_horizon_scenarios = {
        "axm_sigma_xl_1_hm_1_T_1000",
        "axm_sigma_xl_1_hm_1_T_1200",
        "axm_sigma_xl_1_hm_1_T_1400",
        "axm_sigma_xl_1_hm_2_T_1400",
        "axm_sigma_xl_1_hm_2_T_1600",
        "axm_sigma_xl_1_hm_2_T_1800",
    }
    if {row["scenario"] for row in horizon_rows} != expected_horizon_scenarios:
        raise AssertionError("Unexpected scenarios in the saved horizon paths.")
    expected_grids = {
        "axm_sigma_xl_1_hm_1": 1200.0,
        "axm_sigma_xl_1_hm_2": 1600.0,
        "axm_sigma_xl_1_hm_1_T_1000": 1000.0,
        "axm_sigma_xl_1_hm_1_T_1200": 1200.0,
        "axm_sigma_xl_1_hm_1_T_1400": 1400.0,
        "axm_sigma_xl_1_hm_2_T_1400": 1400.0,
        "axm_sigma_xl_1_hm_2_T_1600": 1600.0,
        "axm_sigma_xl_1_hm_2_T_1800": 1800.0,
    }
    structural_rows = rows + horizon_rows
    structural_report: list[dict[str, str | float]] = []
    for scenario, horizon in expected_grids.items():
        group = [row for row in structural_rows if row["scenario"] == scenario]
        expected_count = int(horizon / 2.0) + 1
        times = [float(row["time"]) for row in group]
        finite = all(
            math.isfinite(float(value))
            for row in group
            for key, value in row.items()
            if key != "scenario"
        )
        valid = (
            len(group) == expected_count
            and len(set(times)) == expected_count
            and times == [2.0 * index for index in range(expected_count)]
            and finite
        )
        if not valid:
            raise AssertionError(f"Invalid saved grid for {scenario}.")
        structural_report.append(
            {
                "object": f"{scenario}:complete_finite_time_grid",
                "value": float(expected_count),
                "status": "pass",
            }
        )
    scenario_sigma_hm.update(
        {
            scenario: 1.0 if "_hm_1_" in scenario else 2.0
            for scenario in expected_horizon_scenarios
        }
    )
    rows.extend(horizon_rows)

    omega_m = 0.35
    omega_h = 1.0 - omega_m
    eta = 0.45
    chi = 0.01
    max_errors = {
        "inference_service_X_equals_AU": 0.0,
        "automated_research_service_equals_AM": 0.0,
        "competitive_factor_prices": 0.0,
        "net_interest_rate": 0.0,
        "household_budget_equivalence": 0.0,
        "final_ces_share": 0.0,
        "research_unit_expenditure": 0.0,
        "research_conditional_demands": 0.0,
        "research_expenditure_share": 0.0,
        "research_scale_foc": 0.0,
        "effective_research_index": 0.0,
        "consolidated_capability_law": 0.0,
        "research_compute_foc": 0.0,
        "human_research_foc": 0.0,
        "costate_with_capability_feedback": 0.0,
    }

    for row in rows:
        sigma_hm = scenario_sigma_hm[row["scenario"]]
        log_capability = float(row["log_capability"])
        log_inference_compute = float(row["log_inference_compute"])
        log_ai_services = float(row["log_ai_services"])
        log_research_compute = float(row["log_automated_research"])
        log_automated_services = float(
            row["log_automated_research_services"]
        )
        log_human_research = float(row["log_human_research"])
        log_reported_index = float(row["log_effective_research"])
        log_shadow = float(row["log_shadow_value"])
        log_wage = float(row["log_wage"])
        log_output = float(row["log_output"])
        log_capital = float(row["log_capital"])
        log_production_labor = float(row["log_production_labor"])
        log_ai_price = float(row["log_ai_price"])
        log_research_price = float(row["log_research_price"])
        alpha = 0.33
        omega_x = 0.20

        factor_price_errors = (
            float(row["gross_capital_return"])
            - alpha * math.exp(log_output - log_capital),
            log_wage
            - math.log1p(-alpha)
            - math.log1p(-float(row["ai_share"]))
            - log_output
            + log_production_labor,
            log_ai_price
            - math.log1p(-alpha)
            - math.log(float(row["ai_share"]))
            - log_output
            + log_ai_services,
        )
        max_errors["competitive_factor_prices"] = max(
            max_errors["competitive_factor_prices"],
            *(abs(error) for error in factor_price_errors),
        )
        max_errors["net_interest_rate"] = max(
            max_errors["net_interest_rate"],
            abs(
                float(row["net_capital_return"])
                - alpha * math.exp(log_output - log_capital)
                + 0.05
            ),
        )
        capital_output_ratio = math.exp(log_capital - log_output)
        human_wage_share = math.exp(
            log_wage + log_human_research - log_output
        )
        developer_profit_share = (
            (1.0 - alpha) * float(row["ai_share"])
            - float(row["inference_share"])
            - human_wage_share
            - float(row["research_resource_share"])
        )
        household_budget_rhs = (
            float(row["net_capital_return"]) * capital_output_ratio
            + float(row["aggregate_labor_share"])
            + developer_profit_share
            - float(row["consumption_share"])
        )
        household_budget_lhs = (
            float(row["capital_growth"]) * capital_output_ratio
        )
        max_errors["household_budget_equivalence"] = max(
            max_errors["household_budget_equivalence"],
            abs(household_budget_lhs - household_budget_rhs),
        )
        max_errors["final_ces_share"] = max(
            max_errors["final_ces_share"],
            abs(float(row["ai_share"]) - omega_x),
        )

        max_errors["inference_service_X_equals_AU"] = max(
            max_errors["inference_service_X_equals_AU"],
            abs(log_ai_services - log_capability - log_inference_compute),
        )
        max_errors["automated_research_service_equals_AM"] = max(
            max_errors["automated_research_service_equals_AM"],
            abs(
                log_automated_services
                - log_capability
                - log_research_compute
            ),
        )

        if math.isclose(sigma_hm, 1.0):
            log_reconstructed_index = (
                omega_h * log_human_research
                + omega_m * log_automated_services
            )
            automated_contribution = omega_m
            log_reconstructed_research_price = (
                omega_h * (log_wage - math.log(omega_h))
                + omega_m * (-log_capability - math.log(omega_m))
            )
        else:
            ces_power = (sigma_hm - 1.0) / sigma_hm
            human_term = math.log(omega_h) + ces_power * log_human_research
            machine_term = (
                math.log(omega_m) + ces_power * log_automated_services
            )
            anchor = max(human_term, machine_term)
            log_sum = anchor + math.log(
                math.exp(human_term - anchor)
                + math.exp(machine_term - anchor)
            )
            log_reconstructed_index = log_sum / ces_power
            automated_contribution = math.exp(machine_term - log_sum)
            log_human_price_term = (
                sigma_hm * math.log(omega_h)
                + (1.0 - sigma_hm) * log_wage
            )
            log_machine_price_term = (
                sigma_hm * math.log(omega_m)
                + (1.0 - sigma_hm) * (-log_capability)
            )
            price_anchor = max(log_human_price_term, log_machine_price_term)
            log_reconstructed_research_price = (
                price_anchor
                + math.log(
                    math.exp(log_human_price_term - price_anchor)
                    + math.exp(log_machine_price_term - price_anchor)
                )
            ) / (1.0 - sigma_hm)

        max_errors["research_unit_expenditure"] = max(
            max_errors["research_unit_expenditure"],
            abs(log_research_price - log_reconstructed_research_price),
        )
        reconstructed_log_human_demand = (
            sigma_hm * math.log(omega_h)
            + sigma_hm * (log_research_price - log_wage)
            + log_reported_index
        )
        reconstructed_log_automated_service_demand = (
            sigma_hm * math.log(omega_m)
            + sigma_hm * (log_research_price + log_capability)
            + log_reported_index
        )
        max_errors["research_conditional_demands"] = max(
            max_errors["research_conditional_demands"],
            abs(log_human_research - reconstructed_log_human_demand),
            abs(
                log_automated_services
                - reconstructed_log_automated_service_demand
            ),
        )
        reconstructed_expenditure_share = 1.0 / (
            1.0
            + math.exp(
                log_wage + log_human_research - log_research_compute
            )
        )
        max_errors["research_expenditure_share"] = max(
            max_errors["research_expenditure_share"],
            abs(
                float(row["automated_research_share"])
                - reconstructed_expenditure_share
            ),
        )
        scale_foc_error = (
            log_shadow
            + math.log(chi)
            + math.log(eta)
            + (eta - 1.0) * log_reported_index
            - log_research_price
        )
        max_errors["research_scale_foc"] = max(
            max_errors["research_scale_foc"], abs(scale_foc_error)
        )

        max_errors["effective_research_index"] = max(
            max_errors["effective_research_index"],
            abs(log_reported_index - log_reconstructed_index),
        )
        capability_growth = float(row["capability_growth"])
        if capability_growth <= 0.0:
            raise AssertionError("Capability growth must be positive on these paths.")
        log_capability_flow = math.log(capability_growth) + log_capability
        reconstructed_flow = math.log(chi) + eta * log_reconstructed_index
        max_errors["consolidated_capability_law"] = max(
            max_errors["consolidated_capability_law"],
            abs(log_capability_flow - reconstructed_flow),
        )

        log_research_compute_foc = (
            log_shadow
            + math.log(eta)
            + math.log(automated_contribution)
            + log_capability_flow
            - log_research_compute
        )
        log_human_research_foc = (
            log_shadow
            + math.log(eta)
            + math.log1p(-automated_contribution)
            + log_capability_flow
            - log_human_research
            - log_wage
        )
        max_errors["research_compute_foc"] = max(
            max_errors["research_compute_foc"],
            abs(log_research_compute_foc),
        )
        max_errors["human_research_foc"] = max(
            max_errors["human_research_foc"],
            abs(log_human_research_foc),
        )

        operating_profit_derivative_over_q = math.exp(
            log_ai_services - log_shadow - 2.0 * log_capability
        )
        reconstructed_shadow_growth = (
            float(row["net_capital_return"])
            - operating_profit_derivative_over_q
            - eta * automated_contribution * capability_growth
        )
        max_errors["costate_with_capability_feedback"] = max(
            max_errors["costate_with_capability_feedback"],
            abs(float(row["shadow_growth"]) - reconstructed_shadow_growth),
        )

    tolerances = {
        "inference_service_X_equals_AU": 1e-10,
        "automated_research_service_equals_AM": 1e-10,
        "competitive_factor_prices": 1e-10,
        "net_interest_rate": 1e-10,
        "household_budget_equivalence": 1e-10,
        "final_ces_share": 1e-10,
        "research_unit_expenditure": 1e-10,
        "research_conditional_demands": 1e-8,
        "research_expenditure_share": 1e-10,
        "research_scale_foc": 1e-9,
        "effective_research_index": 1e-10,
        "consolidated_capability_law": 1e-10,
        "research_compute_foc": 1e-9,
        "human_research_foc": 1e-8,
        "costate_with_capability_feedback": 1e-10,
    }
    failed = {
        name: error
        for name, error in max_errors.items()
        if error >= tolerances[name]
    }
    if failed:
        raise AssertionError({"A*M path reconstruction failures": failed})
    return structural_report + [
        {
            "object": f"all_saved_paths:{name}",
            "value": error,
            "status": "pass",
        }
        for name, error in max_errors.items()
    ]


def horizon_checks() -> list[dict[str, str | float]]:
    with HORIZON_AUDIT.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with HORIZON_PATHS.open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))
    with SUMMARY.open(newline="", encoding="utf-8") as handle:
        baseline_rows = list(csv.DictReader(handle))
    initial_capital = float(baseline_rows[0]["initial_capital_stock"])
    initial_capability = float(baseline_rows[0]["initial_capability_stock"])
    initial_population = float(baseline_rows[0]["initial_population"])

    def terminal_targets(sigma_hm: float) -> tuple[float, float]:
        alpha = 0.33
        omega_x = 0.20
        omega_m = 0.35
        eta = 0.45
        beta = (1.0 - alpha) * omega_x
        upsilon = beta / (1.0 - alpha - beta)
        research_weight = omega_m if math.isclose(sigma_hm, 1.0) else 1.0
        denominator = 1.0 - eta * research_weight * (1.0 + upsilon)
        capability_growth = eta * 0.012 / denominator
        per_capita_growth = upsilon * capability_growth
        research_share = (
            beta**2
            * eta
            * research_weight
            * capability_growth
            / (
                0.04
                - 0.012
                + (1.0 - eta * research_weight) * capability_growth
            )
        )
        investment_share = (
            alpha
            * (0.012 + per_capita_growth + 0.05)
            / (0.04 + 0.05 + per_capita_growth)
        )
        consumption_share = 1.0 - investment_share - beta**2 - research_share
        shadow_target = research_share / (
            eta * research_weight * capability_growth
        )
        return consumption_share, shadow_target

    def raw_metrics(
        group: list[dict[str, str]], sigma_hm: float
    ) -> dict[str, float]:
        ordered = sorted(group, key=lambda item: float(item["time"]))

        def max_abs(*fields: str) -> float:
            return max(
                abs(float(item[field])) for item in ordered for field in fields
            )

        discounted_interest = sum(
            0.5
            * (float(right["time"]) - float(left["time"]))
            * (
                float(left["net_capital_return"])
                + float(right["net_capital_return"])
            )
            for left, right in zip(ordered[:-1], ordered[1:])
        )
        first = ordered[0]
        terminal = ordered[-1]
        consumption_target, shadow_target = terminal_targets(sigma_hm)
        terminal_shadow_ratio = math.exp(
            float(terminal["log_shadow_value"])
            + float(terminal["log_capability"])
            - float(terminal["log_output"])
        )
        return {
            "initial_capital_log_error": abs(
                float(first["log_capital"]) - math.log(initial_capital)
            ),
            "initial_capability_log_error": abs(
                float(first["log_capability"]) - math.log(initial_capability)
            ),
            "initial_population_log_error": abs(
                float(first["log_population"]) - math.log(initial_population)
            ),
            "max_abs_euler_residual": max_abs("euler_residual"),
            "max_abs_resource_residual": max(
                abs(float(item["resource_share_sum"]) - 1.0)
                for item in ordered
            ),
            "max_abs_monopoly_foc_log_error": max_abs(
                "monopoly_foc_log_error"
            ),
            "max_abs_factor_price_error": max_abs(
                "capital_price_error", "wage_foc_log_error", "ai_price_foc_log_error"
            ),
            "max_abs_share_definition_error": max_abs(
                "ai_share_definition_error", "automated_share_definition_error"
            ),
            "max_abs_research_dual_error": max_abs(
                "research_price_log_error",
                "human_conditional_demand_log_error",
                "automated_service_demand_log_error",
                "research_scale_foc_log_error",
            ),
            "max_abs_technology_log_error": max_abs(
                "final_production_log_error",
                "inference_identity_log_error",
                "research_ces_log_error",
            ),
            "max_abs_research_compute_foc_log_error": max_abs(
                "research_compute_foc_log_error"
            ),
            "max_abs_research_human_foc_log_error": max_abs(
                "research_human_foc_log_error"
            ),
            "max_abs_labor_market_error": max_abs("labor_market_error"),
            "max_abs_capital_law_residual": max_abs("capital_law_residual"),
            "max_abs_capability_law_residual": max_abs(
                "capability_law_residual"
            ),
            "max_abs_consumption_path_residual": max_abs(
                "consumption_euler_path_residual"
            ),
            "max_abs_shadow_costate_residual": max_abs(
                "shadow_costate_residual"
            ),
            "minimum_consumption_share": min(
                float(item["consumption_share"]) for item in ordered
            ),
            "minimum_investment_share": min(
                float(item["investment_share"]) for item in ordered
            ),
            "minimum_inference_share": min(
                float(item["inference_share"]) for item in ordered
            ),
            "minimum_research_resource_share": min(
                float(item["research_resource_share"]) for item in ordered
            ),
            "minimum_human_research_share": min(
                float(item["human_research_share"]) for item in ordered
            ),
            "minimum_production_labor_share": min(
                float(item["production_labor_population_share"])
                for item in ordered
            ),
            "minimum_monopoly_soc_margin": min(
                float(item["monopoly_soc_margin"]) for item in ordered
            ),
            "terminal_consumption_target_error": abs(
                float(terminal["consumption_share"]) - consumption_target
            ),
            "terminal_shadow_target_error": abs(
                terminal_shadow_ratio - shadow_target
            ),
            "terminal_household_tvc_log_proxy": (
                -0.04 * float(terminal["time"])
                + float(terminal["log_population"])
                + float(terminal["log_capital"])
                - float(terminal["log_consumption"])
            ),
            "terminal_developer_tvc_log_proxy": (
                -discounted_interest
                + float(terminal["log_shadow_value"])
                + float(terminal["log_capability"])
            ),
        }
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["sigma_hm"], []).append(row)
    if set(grouped) != {"1.0", "2.0"}:
        raise AssertionError(f"Unexpected horizon groups: {sorted(grouped)}")

    report: list[dict[str, str | float]] = []
    expected_horizons = {
        "1.0": {1000.0, 1200.0, 1400.0},
        "2.0": {1400.0, 1600.0, 1800.0},
    }
    for sigma_hm, group in grouped.items():
        if len(group) != 3:
            raise AssertionError(
                f"Expected three horizons for sigma_HM={sigma_hm}."
            )
        if {float(row["horizon"]) for row in group} != expected_horizons[sigma_hm]:
            raise AssertionError(f"Unexpected horizons for sigma_HM={sigma_hm}.")
        for row in group:
            scenario = (
                f"axm_sigma_xl_1_hm_{float(sigma_hm):g}_T_"
                f"{float(row['horizon']):g}"
            )
            raw_group = [item for item in raw_rows if item["scenario"] == scenario]
            reconstructed = raw_metrics(raw_group, float(sigma_hm))
            stale = {
                key: (float(row[key]), value)
                for key, value in reconstructed.items()
                if not math.isclose(
                    float(row[key]), value, rel_tol=1e-12, abs_tol=1e-12
                )
            }
            if stale:
                raise AssertionError({f"stale summary {scenario}": stale})
            path_tests = {
                "initial_states": max(
                    float(row["initial_capital_log_error"]),
                    float(row["initial_capability_log_error"]),
                    float(row["initial_population_log_error"]),
                )
                < 1e-10,
                "static_equations": max(
                    float(row["max_abs_resource_residual"]),
                    float(row["max_abs_technology_log_error"]),
                    float(row["max_abs_monopoly_foc_log_error"]),
                    float(row["max_abs_factor_price_error"]),
                    float(row["max_abs_share_definition_error"]),
                    float(row["max_abs_research_dual_error"]),
                    float(row["max_abs_research_compute_foc_log_error"]),
                    float(row["max_abs_research_human_foc_log_error"]),
                    float(row["max_abs_labor_market_error"]),
                )
                < 1e-8,
                "dynamic_equations": max(
                    float(row["max_abs_capital_law_residual"]),
                    float(row["max_abs_capability_law_residual"]),
                    float(row["max_abs_consumption_path_residual"]),
                    float(row["max_abs_shadow_costate_residual"]),
                )
                < 2e-5,
                "interiority": min(
                    float(row["minimum_consumption_share"]),
                    float(row["minimum_investment_share"]),
                    float(row["minimum_inference_share"]),
                    float(row["minimum_research_resource_share"]),
                    float(row["minimum_human_research_share"]),
                    float(row["minimum_production_labor_share"]),
                )
                > 0.0,
                "monopoly_second_order": float(
                    row["minimum_monopoly_soc_margin"]
                )
                > 0.0,
                "terminal_conditions": max(
                    float(row["terminal_consumption_target_error"]),
                    float(row["terminal_shadow_target_error"]),
                )
                < 1e-6,
                "finite_terminal_tvc_proxies": max(
                    float(row["terminal_household_tvc_log_proxy"]),
                    float(row["terminal_developer_tvc_log_proxy"]),
                )
                < -20.0,
            }
            if not all(path_tests.values()):
                raise AssertionError(
                    {
                        f"sigma_HM={sigma_hm},T={row['horizon']}": path_tests
                    }
                )
            for test, passed in path_tests.items():
                report.append(
                    {
                        "object": (
                            f"sigma_HM={sigma_hm},T={row['horizon']}:"
                            f"{test}"
                        ),
                        "value": float(passed),
                        "status": "pass",
                    }
                )
            report.append(
                {
                    "object": f"{scenario}:summary_matches_raw_path",
                    "value": 1.0,
                    "status": "pass",
                }
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
    rows = (
        analytical_checks()
        + numerical_checks()
        + path_specification_checks()
        + horizon_checks()
    )
    with REPORT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["object", "value", "status"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} checks passed; wrote {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
