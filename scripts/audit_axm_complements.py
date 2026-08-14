"""Independent acceptance audit for the A*M gross-complements paths.

The audit deliberately does not import ``simulate_axm_equilibrium`` or any
solver function.  It reads the four canonical ``complements_*`` CSV files,
reconstructs the equilibrium equations from primitives, differentiates the
saved state paths independently, and applies the ex-ante acceptance gates.

The finite-horizon terminal conditions impose only C/Y and X/(q A^2).  All
reported long-run growth, interest-rate, CES-share, and vanishing-resource
checks are therefore non-imposed diagnostics.  Rejected Newton trial states
are not observable in the saved files: the clipping check certifies the
accepted/evaluated paths, while the two saved fallback flags certify the
static maps evaluated on those paths.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_DIR = ROOT / "numerical_axm"
INPUT_NAMES = (
    "complements_transition_paths.csv",
    "complements_transition_summary.csv",
    "complements_horizon_paths.csv",
    "complements_horizon_robustness.csv",
)
REPORT_NAME = "complements_acceptance_report.csv"
RESIDUAL_NAME = "complements_equation_residuals.csv"
MANIFEST_NAME = "complements_audit_manifest.json"


@dataclass(frozen=True)
class Parameters:
    alpha: float = 0.33
    omega_x: float = 0.20
    sigma_xl: float = 0.75
    population_growth: float = 0.012
    delta: float = 0.05
    discount: float = 0.04
    omega_m: float = 0.35
    eta: float = 0.45
    chi: float = 0.01


@dataclass(frozen=True)
class Gates:
    solver_rms: float = 1e-5
    static_equations: float = 1e-8
    independent_dynamics: float = 2e-5
    monopoly_foc_log: float = 1e-9
    imposed_terminal: float = 1e-7
    nonimposed_terminal: float = 2e-5
    limiting_ai_share: float = 5e-4
    hm2_machine_gap: float = 2e-3
    inference_output_share: float = 1e-4
    research_output_share: float = 1e-4
    human_population_share: float = 1e-5
    initial_jump_range: float = 2e-6
    tvc_log_proxy: float = -20.0
    clipping_lower: float = -700.0
    clipping_upper: float = 60.0


PARAMETERS = Parameters()
GATES = Gates()
SIGMA_HM_VALUES = (1.0, 2.0)
ROBUSTNESS_HORIZONS = (1600.0, 1800.0, 2000.0)
PRIMARY_HORIZON = 2000.0
STATE_LOG_FIELDS = (
    "log_capital",
    "log_capability",
    "log_consumption",
    "log_shadow_value",
)
RATE_FIELDS = (
    "capital_growth",
    "capability_growth",
    "consumption_growth",
    "shadow_growth",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independently audit sigma_XL<1 A*M equilibrium paths."
    )
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument(
        "--wait-seconds",
        type=float,
        default=900.0,
        help="Wait for all runner outputs to exist and stop changing.",
    )
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument(
        "--no-fail-exit",
        action="store_true",
        help="Write a rejected report but return exit status zero.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wait_for_stable_inputs(
    paths: Sequence[Path], wait_seconds: float, poll_seconds: float
) -> None:
    """Wait until every input exists, is nonempty, and has two stable polls."""

    deadline = time.monotonic() + max(wait_seconds, 0.0)
    previous: tuple[tuple[int, int], ...] | None = None
    stable_polls = 0
    while True:
        if all(path.is_file() and path.stat().st_size > 0 for path in paths):
            signature = tuple(
                (path.stat().st_size, path.stat().st_mtime_ns) for path in paths
            )
            if signature == previous:
                stable_polls += 1
            else:
                stable_polls = 0
                previous = signature
            if stable_polls >= 2:
                return
        if time.monotonic() >= deadline:
            missing = [str(path) for path in paths if not path.is_file()]
            raise TimeoutError(
                "Complements outputs did not become stable before timeout. "
                f"Missing: {missing}"
            )
        time.sleep(max(poll_seconds, 0.1))


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Input CSV has no data rows: {path}")
    return rows


def write_rows(path: Path, rows: Iterable[dict[str, object]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"Refusing to write an empty table: {path}")
    fieldnames = list(
        dict.fromkeys(key for row in materialized for key in row.keys())
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialized)


def number(row: dict[str, str], field: str) -> float:
    if field not in row or row[field] == "":
        raise KeyError(f"Missing required numeric field {field!r}.")
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {field!r}: {row[field]!r}")
    return value


def logsumexp(values: Sequence[float]) -> float:
    anchor = max(values)
    return anchor + math.log(sum(math.exp(value - anchor) for value in values))


def log_ces_and_right_share(
    log_left: float,
    log_right: float,
    right_weight: float,
    elasticity: float,
) -> tuple[float, float]:
    if abs(elasticity - 1.0) <= 1e-12:
        return (
            (1.0 - right_weight) * log_left + right_weight * log_right,
            right_weight,
        )
    power = (elasticity - 1.0) / elasticity
    terms = (
        math.log1p(-right_weight) + power * log_left,
        math.log(right_weight) + power * log_right,
    )
    denominator = logsumexp(terms)
    return denominator / power, math.exp(terms[1] - denominator)


def log_research_unit_cost(
    log_wage: float, log_capability: float, sigma_hm: float
) -> float:
    log_machine_price = -log_capability
    if abs(sigma_hm - 1.0) <= 1e-12:
        return (
            (1.0 - PARAMETERS.omega_m)
            * (log_wage - math.log1p(-PARAMETERS.omega_m))
            + PARAMETERS.omega_m
            * (log_machine_price - math.log(PARAMETERS.omega_m))
        )
    terms = (
        sigma_hm * math.log1p(-PARAMETERS.omega_m)
        + (1.0 - sigma_hm) * log_wage,
        sigma_hm * math.log(PARAMETERS.omega_m)
        + (1.0 - sigma_hm) * log_machine_price,
    )
    return logsumexp(terms) / (1.0 - sigma_hm)


def scenario_name(sigma_hm: float, horizon: float | None = None) -> str:
    base = f"axm_complements_sigma_xl_075_hm_{sigma_hm:g}"
    return base if horizon is None else f"{base}_T_{horizon:g}"


def target_values(sigma_hm: float) -> dict[str, float]:
    p = PARAMETERS
    research_weight = p.omega_m if math.isclose(sigma_hm, 1.0) else 1.0
    capability_growth = (
        p.eta
        * p.population_growth
        / (1.0 + p.eta * (1.0 - research_weight))
    )
    shadow_growth = p.population_growth - 2.0 * capability_growth
    capital_output_ratio = p.alpha / (p.discount + p.delta)
    investment_share = (
        p.population_growth + p.delta
    ) * capital_output_ratio
    consumption_share = 1.0 - investment_share
    profit_shadow_ratio = (
        p.discount
        - p.population_growth
        + (2.0 - p.eta * research_weight) * capability_growth
    )
    limiting_ai_share = (1.0 - p.sigma_xl) / (
        1.0 - p.alpha * p.sigma_xl
    )
    return {
        "aggregate_growth": p.population_growth,
        "capability_growth": capability_growth,
        "shadow_growth": shadow_growth,
        "capital_output_ratio": capital_output_ratio,
        "investment_share": investment_share,
        "consumption_share": consumption_share,
        "profit_shadow_ratio": profit_shadow_ratio,
        "limiting_ai_share": limiting_ai_share,
        "net_interest_rate": p.discount,
        "research_weight": research_weight,
    }


def reconstruct_row(
    row: dict[str, str], sigma_hm: float
) -> dict[str, float]:
    """Reconstruct every dated equation without invoking solver code."""

    p = PARAMETERS
    log_k = number(row, "log_capital")
    log_a = number(row, "log_capability")
    log_n = number(row, "log_population")
    log_c = number(row, "log_consumption")
    log_q = number(row, "log_shadow_value")
    log_y = number(row, "log_output")
    log_w = number(row, "log_wage")
    log_px = number(row, "log_ai_price")
    log_pe = number(row, "log_research_price")
    log_x = number(row, "log_ai_services")
    log_u = number(row, "log_inference_compute")
    log_h = number(row, "log_human_research")
    log_l = number(row, "log_production_labor")
    log_m = number(row, "log_automated_research")
    log_am = number(row, "log_automated_research_services")
    log_e = number(row, "log_effective_research")

    log_z, s_x = log_ces_and_right_share(
        log_l, log_x, p.omega_x, p.sigma_xl
    )
    log_e_reconstructed, s_m = log_ces_and_right_share(
        log_h, log_am, p.omega_m, sigma_hm
    )
    log_pe_reconstructed = log_research_unit_cost(log_w, log_a, sigma_hm)
    gross_return = p.alpha * math.exp(log_y - log_k)
    net_return = gross_return - p.delta
    human_share = math.exp(log_h - log_n)
    labor_share = math.exp(log_l - log_n)
    consumption_share = math.exp(log_c - log_y)
    inference_share = math.exp(log_u - log_y)
    research_share = math.exp(log_m - log_y)
    investment_share = number(row, "investment_share")
    reported_sx = number(row, "ai_share")
    reported_sm = number(row, "automated_research_share")

    inverse_elasticity = (
        (1.0 - s_x) / p.sigma_xl + p.alpha * s_x
    )
    markup_term = 1.0 - inverse_elasticity
    if markup_term <= 0.0:
        monopoly_foc = math.inf
    else:
        monopoly_foc = log_px + math.log(markup_term) + log_a
    inverse_elasticity_derivative = (
        (p.alpha - 1.0 / p.sigma_xl)
        * (1.0 - 1.0 / p.sigma_xl)
        * s_x
        * (1.0 - s_x)
    )
    monopoly_soc = (
        inverse_elasticity * (1.0 - inverse_elasticity)
        + inverse_elasticity_derivative
    )

    log_h_demand = (
        sigma_hm * math.log1p(-p.omega_m)
        + sigma_hm * (log_pe - log_w)
        + log_e
    )
    log_am_demand = (
        sigma_hm * math.log(p.omega_m)
        + sigma_hm * (log_pe + log_a)
        + log_e
    )
    expenditure_sm = 1.0 / (
        1.0 + math.exp(log_w + log_h - log_m)
    )

    log_fm = (
        math.log(p.chi)
        + math.log(p.eta)
        + (p.eta - 1.0) * log_e
        + math.log(p.omega_m)
        + log_a
        + (log_e - log_am) / sigma_hm
    )
    log_fh = (
        math.log(p.chi)
        + math.log(p.eta)
        + (p.eta - 1.0) * log_e
        + math.log1p(-p.omega_m)
        + (log_e - log_h) / sigma_hm
    )

    capability_growth = math.exp(
        math.log(p.chi) + p.eta * log_e - log_a
    )
    capital_growth = (
        math.exp(log_y - log_k)
        - math.exp(log_c - log_k)
        - math.exp(log_u - log_k)
        - math.exp(log_m - log_k)
        - p.delta
    )
    consumption_growth = (
        p.population_growth + gross_return - p.delta - p.discount
    )
    profit_shadow_ratio = math.exp(log_x - 2.0 * log_a - log_q)
    shadow_growth = (
        gross_return
        - p.delta
        - p.eta * s_m * capability_growth
        - profit_shadow_ratio
    )

    clipping_arguments = (
        math.log(p.chi) + p.eta * log_e - log_a,
        log_y - log_k,
        log_u - log_y,
        log_m - log_y,
        log_x - 2.0 * log_a,
        log_c - log_k,
        log_u - log_k,
        log_m - log_k,
        -log_q,
    )
    clipping_count = sum(
        value < GATES.clipping_lower or value > GATES.clipping_upper
        for value in clipping_arguments
    )

    static_errors = {
        "population_path_error": (
            log_n - p.population_growth * number(row, "time")
        ),
        "labor_market_error_independent": human_share + labor_share - 1.0,
        "final_production_log_error_independent": (
            log_y - p.alpha * log_k - (1.0 - p.alpha) * log_z
        ),
        "inference_identity_log_error_independent": log_x - log_a - log_u,
        "automated_service_log_error_independent": log_am - log_a - log_m,
        "research_ces_log_error_independent": log_e - log_e_reconstructed,
        "capability_law_log_error_independent": (
            math.log(number(row, "capability_growth"))
            + log_a
            - math.log(p.chi)
            - p.eta * log_e
        ),
        "capital_price_error_independent": (
            number(row, "gross_capital_return") - gross_return
        ),
        "net_interest_error_independent": (
            number(row, "net_capital_return") - net_return
        ),
        "wage_foc_log_error_independent": (
            log_w
            - math.log1p(-p.alpha)
            - math.log1p(-s_x)
            - log_y
            + log_l
        ),
        "ai_price_foc_log_error_independent": (
            log_px
            - math.log1p(-p.alpha)
            - math.log(s_x)
            - log_y
            + log_x
        ),
        "ai_share_error_independent": reported_sx - s_x,
        "research_price_log_error_independent": log_pe - log_pe_reconstructed,
        "human_demand_log_error_independent": log_h - log_h_demand,
        "automated_service_demand_log_error_independent": (
            log_am - log_am_demand
        ),
        "automated_share_quantity_error_independent": reported_sm - s_m,
        "automated_share_expenditure_error_independent": (
            reported_sm - expenditure_sm
        ),
        "research_scale_foc_log_error_independent": (
            log_q
            + math.log(p.chi)
            + math.log(p.eta)
            + (p.eta - 1.0) * log_e
            - log_pe
        ),
        "research_compute_foc_log_error_independent": log_q + log_fm,
        "research_human_foc_log_error_independent": log_q + log_fh - log_w,
        "resource_constraint_error_independent": (
            consumption_share
            + investment_share
            + inference_share
            + research_share
            - 1.0
        ),
        "profit_shadow_ratio_error_independent": (
            number(row, "profit_shadow_ratio") - profit_shadow_ratio
        ),
    }
    static_max = max(abs(value) for value in static_errors.values())
    return {
        **static_errors,
        "static_max_abs_error": static_max,
        "monopoly_foc_log_error_independent": monopoly_foc,
        "monopoly_soc_margin_independent": monopoly_soc,
        "rhs_capital_growth": capital_growth,
        "rhs_capability_growth": capability_growth,
        "rhs_consumption_growth": consumption_growth,
        "rhs_shadow_growth": shadow_growth,
        "stored_capital_rate_error": (
            number(row, "capital_growth") - capital_growth
        ),
        "stored_capability_rate_error": (
            number(row, "capability_growth") - capability_growth
        ),
        "stored_consumption_rate_error": (
            number(row, "consumption_growth") - consumption_growth
        ),
        "stored_shadow_rate_error": (
            number(row, "shadow_growth") - shadow_growth
        ),
        "ai_share_independent": s_x,
        "automated_research_share_independent": s_m,
        "inference_output_share_independent": inference_share,
        "research_output_share_independent": research_share,
        "human_population_share_independent": human_share,
        "profit_shadow_ratio_independent": profit_shadow_ratio,
        "net_interest_rate_independent": net_return,
        "bounded_exp_min_argument": min(clipping_arguments),
        "bounded_exp_max_argument": max(clipping_arguments),
        "bounded_exp_clipping_count": float(clipping_count),
    }


def lagrange_derivative(
    x0: float, xs: Sequence[float], ys: Sequence[float]
) -> float:
    """Derivative of the local interpolation polynomial at x0."""

    shifted = [value - x0 for value in xs]
    derivative = 0.0
    for j, xj in enumerate(shifted):
        denominator = math.prod(
            xj - xk for k, xk in enumerate(shifted) if k != j
        )
        numerator_derivative = 0.0
        for m in range(len(shifted)):
            if m == j:
                continue
            numerator_derivative += math.prod(
                -xk
                for k, xk in enumerate(shifted)
                if k != j and k != m
            )
        derivative += ys[j] * numerator_derivative / denominator
    return derivative


def independent_derivatives(
    rows: Sequence[dict[str, str]], field: str
) -> dict[int, float]:
    """Centered seven-point derivatives; endpoints are intentionally omitted."""

    if len(rows) < 7:
        raise ValueError("At least seven saved dates are required.")
    times = [number(row, "time") for row in rows]
    values = [number(row, field) for row in rows]
    return {
        index: lagrange_derivative(
            times[index],
            times[index - 3 : index + 4],
            values[index - 3 : index + 4],
        )
        for index in range(3, len(rows) - 3)
    }


def endpoint_derivative(rows: Sequence[dict[str, str]], field: str) -> float:
    """Seven-point backward diagnostic, kept separate from the dynamics gate."""

    subset = rows[-7:]
    return lagrange_derivative(
        number(subset[-1], "time"),
        [number(row, "time") for row in subset],
        [number(row, field) for row in subset],
    )


def group_paths(rows: Iterable[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["scenario"], []).append(row)
    for name, group in grouped.items():
        group.sort(key=lambda item: number(item, "time"))
        times = [number(item, "time") for item in group]
        if times[0] != 0.0 or any(
            right <= left for left, right in zip(times[:-1], times[1:])
        ):
            raise ValueError(f"Invalid or unordered time grid for {name}.")
        if len(set(times)) != len(times):
            raise ValueError(f"Duplicate saved dates for {name}.")
    return grouped


def report_row(
    category: str,
    metric: str,
    value: float,
    relation: str,
    threshold: float | str,
    passed: bool,
    scenario: str = "all",
    sigma_hm: float | str = "",
    horizon: float | str = "",
    details: str = "",
) -> dict[str, object]:
    return {
        "category": category,
        "scenario": scenario,
        "sigma_xl": PARAMETERS.sigma_xl,
        "sigma_hm": sigma_hm,
        "horizon": horizon,
        "metric": metric,
        "value": value,
        "relation": relation,
        "threshold": threshold,
        "status": "pass" if passed else "fail",
        "details": details,
    }


def trapezoid_integral(rows: Sequence[dict[str, str]], field: str) -> float:
    return sum(
        0.5
        * (number(right, "time") - number(left, "time"))
        * (number(left, field) + number(right, field))
        for left, right in zip(rows[:-1], rows[1:])
    )


def audit(
    transition_paths: list[dict[str, str]],
    transition_summary: list[dict[str, str]],
    horizon_paths: list[dict[str, str]],
    horizon_summary: list[dict[str, str]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], bool]:
    expected_primary = {scenario_name(value) for value in SIGMA_HM_VALUES}
    expected_horizon = {
        scenario_name(value, horizon)
        for value in SIGMA_HM_VALUES
        for horizon in ROBUSTNESS_HORIZONS
    }
    primary_groups = group_paths(transition_paths)
    horizon_groups = group_paths(horizon_paths)
    if set(primary_groups) != expected_primary:
        raise ValueError(f"Unexpected primary scenarios: {sorted(primary_groups)}")
    if set(horizon_groups) != expected_horizon:
        raise ValueError(f"Unexpected horizon scenarios: {sorted(horizon_groups)}")
    if {row["scenario"] for row in transition_summary} != expected_primary:
        raise ValueError("Transition summary scenario set does not match paths.")
    if {row["scenario"] for row in horizon_summary} != expected_horizon:
        raise ValueError("Horizon summary scenario set does not match paths.")

    summaries = {
        row["scenario"]: row for row in transition_summary + horizon_summary
    }
    all_groups = {**primary_groups, **horizon_groups}
    report: list[dict[str, object]] = []
    residual_rows: list[dict[str, object]] = []
    path_metrics: dict[str, dict[str, float]] = {}

    for name, rows in all_groups.items():
        summary = summaries[name]
        sigma_hm = number(rows[0], "sigma_hm")
        horizon = number(rows[0], "horizon")
        if not math.isclose(number(rows[-1], "time"), horizon, abs_tol=1e-10):
            raise ValueError(f"Saved path does not reach its horizon: {name}")
        if not math.isclose(number(rows[0], "sigma_xl"), PARAMETERS.sigma_xl):
            raise ValueError(f"Wrong sigma_XL in {name}.")
        if not all(
            math.isclose(number(row, "sigma_hm"), sigma_hm)
            and math.isclose(number(row, "horizon"), horizon)
            for row in rows
        ):
            raise ValueError(f"Inconsistent path metadata in {name}.")

        reconstructed = [reconstruct_row(row, sigma_hm) for row in rows]
        derivatives = {
            field: independent_derivatives(rows, field)
            for field in STATE_LOG_FIELDS
        }
        independent_dynamic_max = 0.0
        stored_rate_max = 0.0
        static_max = 0.0
        monopoly_foc_max = 0.0
        monopoly_soc_min = math.inf
        fallback_max = 0.0
        clipping_count = 0.0
        clipping_min = math.inf
        clipping_max = -math.inf

        for index, (raw, rebuilt) in enumerate(zip(rows, reconstructed)):
            output: dict[str, object] = {
                "scenario": name,
                "sigma_xl": PARAMETERS.sigma_xl,
                "sigma_hm": sigma_hm,
                "horizon": horizon,
                "time": number(raw, "time"),
                **rebuilt,
            }
            for state, rate, rhs in zip(
                STATE_LOG_FIELDS,
                RATE_FIELDS,
                (
                    "rhs_capital_growth",
                    "rhs_capability_growth",
                    "rhs_consumption_growth",
                    "rhs_shadow_growth",
                ),
            ):
                if index in derivatives[state]:
                    derivative = derivatives[state][index]
                    residual = derivative - rebuilt[rhs]
                    output[f"fd_{rate}"] = derivative
                    output[f"fd_{rate}_residual"] = residual
                    independent_dynamic_max = max(
                        independent_dynamic_max, abs(residual)
                    )
                else:
                    output[f"fd_{rate}"] = ""
                    output[f"fd_{rate}_residual"] = ""
            residual_rows.append(output)
            static_max = max(static_max, rebuilt["static_max_abs_error"])
            monopoly_foc_max = max(
                monopoly_foc_max,
                abs(rebuilt["monopoly_foc_log_error_independent"]),
            )
            monopoly_soc_min = min(
                monopoly_soc_min,
                rebuilt["monopoly_soc_margin_independent"],
            )
            stored_rate_max = max(
                stored_rate_max,
                *(abs(rebuilt[field]) for field in (
                    "stored_capital_rate_error",
                    "stored_capability_rate_error",
                    "stored_consumption_rate_error",
                    "stored_shadow_rate_error",
                )),
            )
            fallback_max = max(
                fallback_max,
                number(raw, "monopoly_root_fallback"),
                number(raw, "labor_root_fallback"),
            )
            clipping_count += rebuilt["bounded_exp_clipping_count"]
            clipping_min = min(clipping_min, rebuilt["bounded_exp_min_argument"])
            clipping_max = max(clipping_max, rebuilt["bounded_exp_max_argument"])

        terminal = rows[-1]
        rebuilt_terminal = reconstructed[-1]
        targets = target_values(sigma_hm)
        imposed_consumption_error = abs(
            number(terminal, "consumption_share")
            - targets["consumption_share"]
        )
        imposed_shadow_error = abs(
            rebuilt_terminal["profit_shadow_ratio_independent"]
            - targets["profit_shadow_ratio"]
        )
        discounted_interest = trapezoid_integral(rows, "net_capital_return")
        household_tvc = (
            -PARAMETERS.discount * horizon
            + number(terminal, "log_population")
            + number(terminal, "log_capital")
            - number(terminal, "log_consumption")
        )
        developer_tvc = (
            -discounted_interest
            + number(terminal, "log_shadow_value")
            + number(terminal, "log_capability")
        )
        terminal_output_growth = endpoint_derivative(rows, "log_output")
        terminal_wage_growth = endpoint_derivative(rows, "log_wage")
        path_metrics[name] = {
            "static_max": static_max,
            "stored_rate_max": stored_rate_max,
            "independent_dynamic_max": independent_dynamic_max,
            "monopoly_foc_max": monopoly_foc_max,
            "monopoly_soc_min": monopoly_soc_min,
            "fallback_max": fallback_max,
            "clipping_count": clipping_count,
            "clipping_min": clipping_min,
            "clipping_max": clipping_max,
            "imposed_terminal_max": max(
                imposed_consumption_error, imposed_shadow_error
            ),
            "household_tvc": household_tvc,
            "developer_tvc": developer_tvc,
            "terminal_output_growth_independent": terminal_output_growth,
            "terminal_wage_growth_independent": terminal_wage_growth,
        }

        checks = (
            (
                "solver",
                "solver_success_flag",
                number(summary, "solver_success"),
                "=",
                1.0,
                number(summary, "solver_success") == 1.0,
                "",
            ),
            (
                "solver",
                "cold_start_flag",
                number(summary, "cold_start"),
                "=",
                1.0,
                number(summary, "cold_start") == 1.0,
                "no continuation from another horizon",
            ),
            (
                "solver",
                "solver_tolerance",
                number(summary, "solver_tolerance"),
                "<=",
                GATES.solver_rms,
                number(summary, "solver_tolerance") <= GATES.solver_rms,
                "declared ex-ante numerical tolerance",
            ),
            (
                "solver",
                "maximum_rms_residual",
                number(summary, "max_rms_residual"),
                "<",
                GATES.solver_rms,
                number(summary, "max_rms_residual") < GATES.solver_rms,
                "cold-start solve",
            ),
            (
                "equations",
                "maximum_static_equation_error",
                static_max,
                "<",
                GATES.static_equations,
                static_max < GATES.static_equations,
                "independent reconstruction",
            ),
            (
                "equations",
                "maximum_saved_rate_equation_error",
                stored_rate_max,
                "<",
                GATES.independent_dynamics,
                stored_rate_max < GATES.independent_dynamics,
                "saved ODE rates versus independently reconstructed RHS",
            ),
            (
                "equations",
                "maximum_independent_dynamic_error",
                independent_dynamic_max,
                "<",
                GATES.independent_dynamics,
                independent_dynamic_max < GATES.independent_dynamics,
                "centered seven-point derivatives on interior dates",
            ),
            (
                "monopoly",
                "maximum_monopoly_foc_log_error",
                monopoly_foc_max,
                "<",
                GATES.monopoly_foc_log,
                monopoly_foc_max < GATES.monopoly_foc_log,
                "dimensionless log FOC",
            ),
            (
                "monopoly",
                "minimum_monopoly_soc_margin",
                monopoly_soc_min,
                ">",
                0.0,
                monopoly_soc_min > 0.0,
                "",
            ),
            (
                "terminal_imposed",
                "maximum_imposed_terminal_error",
                path_metrics[name]["imposed_terminal_max"],
                "<",
                GATES.imposed_terminal,
                path_metrics[name]["imposed_terminal_max"]
                < GATES.imposed_terminal,
                "C/Y and X/(q A^2) only",
            ),
            (
                "transversality",
                "household_tvc_log_proxy",
                household_tvc,
                "<",
                GATES.tvc_log_proxy,
                household_tvc < GATES.tvc_log_proxy,
                "finite-horizon diagnostic",
            ),
            (
                "transversality",
                "developer_tvc_log_proxy",
                developer_tvc,
                "<",
                GATES.tvc_log_proxy,
                developer_tvc < GATES.tvc_log_proxy,
                "finite-horizon diagnostic",
            ),
            (
                "implementation",
                "maximum_static_fallback_flag",
                fallback_max,
                "=",
                0.0,
                fallback_max == 0.0,
                "monopoly and labor root flags on accepted paths",
            ),
            (
                "implementation",
                "accepted_path_bounded_exp_clipping_count",
                clipping_count,
                "=",
                0.0,
                clipping_count == 0.0,
                (
                    f"reconstructed argument range [{clipping_min:.6g}, "
                    f"{clipping_max:.6g}]; rejected solver trials unobservable"
                ),
            ),
        )
        for category, metric, value, relation, threshold, passed, details in checks:
            report.append(
                report_row(
                    category,
                    metric,
                    float(value),
                    relation,
                    float(threshold),
                    bool(passed),
                    name,
                    sigma_hm,
                    horizon,
                    details,
                )
            )

    # The non-imposed asymptotic gates apply only to the canonical T=2000
    # primary paths.  T=1600 and T=1800 identify horizon stability.
    for sigma_hm in SIGMA_HM_VALUES:
        name = scenario_name(sigma_hm)
        rows = primary_groups[name]
        terminal = rows[-1]
        rebuilt = reconstruct_row(terminal, sigma_hm)
        targets = target_values(sigma_hm)
        g_y = path_metrics[name]["terminal_output_growth_independent"]
        limit_errors = {
            "capital_growth": abs(
                number(terminal, "capital_growth")
                - targets["aggregate_growth"]
            ),
            "consumption_growth": abs(
                number(terminal, "consumption_growth")
                - targets["aggregate_growth"]
            ),
            "output_growth_independent": abs(
                g_y - targets["aggregate_growth"]
            ),
            "capability_growth": abs(
                number(terminal, "capability_growth")
                - targets["capability_growth"]
            ),
            "shadow_growth": abs(
                number(terminal, "shadow_growth")
                - targets["shadow_growth"]
            ),
            "net_interest_rate": abs(
                rebuilt["net_interest_rate_independent"]
                - targets["net_interest_rate"]
            ),
        }
        maximum_limit_error = max(limit_errors.values())
        report.append(
            report_row(
                "terminal_nonimposed",
                "maximum_nonimposed_growth_or_interest_error",
                maximum_limit_error,
                "<",
                GATES.nonimposed_terminal,
                maximum_limit_error < GATES.nonimposed_terminal,
                name,
                sigma_hm,
                PRIMARY_HORIZON,
                "; ".join(
                    f"{key}={value:.6g}" for key, value in limit_errors.items()
                ),
            )
        )
        sx_error = abs(
            rebuilt["ai_share_independent"] - targets["limiting_ai_share"]
        )
        report.append(
            report_row(
                "terminal_nonimposed",
                "limiting_ai_share_error",
                sx_error,
                "<",
                GATES.limiting_ai_share,
                sx_error < GATES.limiting_ai_share,
                name,
                sigma_hm,
                PRIMARY_HORIZON,
                "s_X=(1-sigma_XL)/(1-alpha*sigma_XL)",
            )
        )
        if math.isclose(sigma_hm, 2.0):
            machine_gap = 1.0 - rebuilt["automated_research_share_independent"]
            report.append(
                report_row(
                    "terminal_nonimposed",
                    "hm2_one_minus_machine_share",
                    machine_gap,
                    "<",
                    GATES.hm2_machine_gap,
                    machine_gap < GATES.hm2_machine_gap,
                    name,
                    sigma_hm,
                    PRIMARY_HORIZON,
                )
            )
        for metric, key, threshold in (
            (
                "terminal_inference_output_share",
                "inference_output_share_independent",
                GATES.inference_output_share,
            ),
            (
                "terminal_research_output_share",
                "research_output_share_independent",
                GATES.research_output_share,
            ),
            (
                "terminal_human_population_share",
                "human_population_share_independent",
                GATES.human_population_share,
            ),
        ):
            value = rebuilt[key]
            report.append(
                report_row(
                    "terminal_nonimposed",
                    metric,
                    value,
                    "<",
                    threshold,
                    value < threshold,
                    name,
                    sigma_hm,
                    PRIMARY_HORIZON,
                )
            )
        report.append(
            report_row(
                "terminal_diagnostic",
                "terminal_wage_growth_independent",
                path_metrics[name]["terminal_wage_growth_independent"],
                "diagnostic",
                "",
                True,
                name,
                sigma_hm,
                PRIMARY_HORIZON,
                "seven-point backward derivative; not an acceptance gate",
            )
        )

    for sigma_hm in SIGMA_HM_VALUES:
        group_names = [
            scenario_name(sigma_hm, horizon)
            for horizon in ROBUSTNESS_HORIZONS
        ]
        first_rows = [horizon_groups[name][0] for name in group_names]
        for metric, field in (
            ("initial_log_consumption_range", "log_consumption"),
            ("initial_log_shadow_range", "log_shadow_value"),
        ):
            values = [number(row, field) for row in first_rows]
            spread = max(values) - min(values)
            report.append(
                report_row(
                    "horizon_stability",
                    metric,
                    spread,
                    "<",
                    GATES.initial_jump_range,
                    spread < GATES.initial_jump_range,
                    f"sigma_HM={sigma_hm:g}:T=1600,1800,2000",
                    sigma_hm,
                    "1600-2000",
                    "every horizon is a cold-start solve",
                )
            )

    failed = [row for row in report if row["status"] == "fail"]
    report.append(
        report_row(
            "overall",
            "all_ex_ante_acceptance_gates",
            float(len(failed)),
            "=",
            0.0,
            not failed,
            details="number of failed component gates",
        )
    )
    return report, residual_rows, not failed


def main() -> None:
    args = parse_arguments()
    result_dir = args.result_dir.resolve()
    input_paths = [result_dir / name for name in INPUT_NAMES]
    wait_for_stable_inputs(input_paths, args.wait_seconds, args.poll_seconds)

    transition_paths = read_rows(input_paths[0])
    transition_summary = read_rows(input_paths[1])
    horizon_paths = read_rows(input_paths[2])
    horizon_summary = read_rows(input_paths[3])
    report, residuals, accepted = audit(
        transition_paths,
        transition_summary,
        horizon_paths,
        horizon_summary,
    )

    report_path = result_dir / REPORT_NAME
    residual_path = result_dir / RESIDUAL_NAME
    manifest_path = result_dir / MANIFEST_NAME
    write_rows(report_path, report)
    write_rows(residual_path, residuals)

    script_path = Path(__file__).resolve()
    manifest = {
        "audit": "A*M gross-complements independent acceptance audit",
        "audit_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "accepted": accepted,
        "path_label": (
            "finite-horizon equilibrium candidate paths satisfying all "
            "dated equations and the stated terminal conditions"
        ),
        "scope_note": (
            "No solver functions are imported. Clipping is certified on the "
            "accepted/evaluated paths; rejected Newton trial states are not "
            "observable in the canonical CSV outputs."
        ),
        "parameters": asdict(PARAMETERS),
        "ex_ante_gates": asdict(GATES),
        "canonical_scenarios": {
            "primary": [scenario_name(value) for value in SIGMA_HM_VALUES],
            "horizon_robustness": [
                scenario_name(value, horizon)
                for value in SIGMA_HM_VALUES
                for horizon in ROBUSTNESS_HORIZONS
            ],
        },
        "terminal_condition_provenance": {
            "imposed": ["C/Y", "X/(q A^2)"],
            "nonimposed": [
                "gK",
                "gC",
                "gY",
                "gA",
                "gq",
                "net interest rate",
                "sX",
                "sM",
                "U/Y",
                "M/Y",
                "H/N",
            ],
        },
        "numerical_derivative": (
            "centered seven-point local-polynomial derivative on interior "
            "saved dates; separate seven-point backward terminal diagnostics"
        ),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
        },
        "files": {
            "inputs": [
                {
                    "path": path.relative_to(ROOT).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in input_paths
            ],
            "audit_script": {
                "path": script_path.relative_to(ROOT).as_posix(),
                "bytes": script_path.stat().st_size,
                "sha256": sha256_file(script_path),
            },
            "outputs": [
                {
                    "path": path.relative_to(ROOT).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in (report_path, residual_path)
            ],
        },
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(
        f"Complements audit {'ACCEPTED' if accepted else 'REJECTED'}; "
        f"report={report_path}; residuals={residual_path}; "
        f"manifest={manifest_path}"
    )
    if not accepted and not args.no_fail_exit:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
