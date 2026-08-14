"""Independent audit of the A*M high-sigma free-boundary paths.

This script deliberately does not import either equilibrium solver.  It reads
each canonical grouped ``boundary_paths.csv``, reconstructs the model from its
primitive equations, and
checks whether the files describe a converged *pre-singular necessary-condition
path*.  Passing this audit is not evidence of an infinite-horizon equilibrium:
the free boundary uses conditional singular asymptotics and no transversality or
global-optimality condition at infinity is verified here.

The default canonical inputs are ``high_sigma_sigma150_validated`` and the
separately refined ``high_sigma_sigma150_z128_validated`` prefix.  No similarly
named files are discovered from the directory; additional prefixes must be
listed explicitly.  The required convergence sequence remains z = 16, 32, 64.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "numerical_axm"
DEFAULT_PREFIX = "high_sigma_sigma150_validated"
DEFAULT_EXTRA_PREFIX = "high_sigma_sigma150_z128_validated"
PATH_LABEL = (
    "finite-boundary approximations to a conditional pre-singular branch; "
    "not an infinite-horizon equilibrium"
)


@dataclass(frozen=True)
class Parameters:
    """Primitive parameters used to reconstruct the reported experiment."""

    alpha: float = 0.33
    omega_x: float = 0.20
    sigma_xl: float = 1.50
    n: float = 0.012
    delta: float = 0.05
    discount: float = 0.04
    omega_m: float = 0.35
    sigma_hm: float = 2.00
    eta: float = 0.45
    chi: float = 0.01


@dataclass(frozen=True)
class Tolerances:
    """Ex ante gates; values are recorded verbatim in the manifest."""

    equation_log: float = 5e-8
    equation_level: float = 5e-8
    foc_log: float = 5e-7
    stored_dynamic: float = 3e-5
    independent_dynamic: float = 5e-5
    solver_rms: float = 2e-5
    boundary_log: float = 5e-7
    group_boundary_spread: float = 1e-9
    duration_endpoint: float = 1e-8
    jump_final: float = 1e-7
    tstar_final: float = 2e-2
    tstar_contraction: float = 0.75
    saved_tstar: float = 1e-9
    common_window_log: float = 2e-4
    common_window_level: float = 2e-4
    common_window_terminal_buffer: float = 50.0
    event_time: float = 2e-2
    event_window_log: float = 2e-3
    event_window_level: float = 5e-3
    terminal_ratio_error: float = 5e-3
    clipping_lower: float = -700.0
    clipping_upper: float = 60.0


REQUIRED_BOUNDARIES = (16.0, 32.0, 64.0)
STATE_LOG_FIELDS = (
    "log_capital",
    "log_capability",
    "log_consumption",
    "log_shadow_value",
)
COMMON_LOG_FIELDS = STATE_LOG_FIELDS + ("log_output", "log_wage")
COMMON_LEVEL_FIELDS = (
    "ai_share",
    "automated_research_share",
    "capital_growth",
    "capability_growth",
    "consumption_growth",
)
STORED_DYNAMIC_FIELDS = (
    "capital_law_residual",
    "capability_law_residual",
    "consumption_euler_path_residual",
    "shadow_costate_residual",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit conditional high-sigma A*M free-boundary paths."
    )
    parser.add_argument("--input-prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    parser.add_argument(
        "--required-boundaries",
        default="16,32,64",
        help="Comma-separated free boundaries required for acceptance.",
    )
    parser.add_argument(
        "--extra-prefixes",
        default=None,
        help=(
            "Comma-separated extra canonical prefixes. By default the published "
            "input prefix adds only high_sigma_sigma150_z128_validated; custom "
            "input prefixes add no extras unless listed explicitly."
        ),
    )
    parser.add_argument(
        "--no-auto-extra-boundaries",
        action="store_true",
        help=(
            "Deprecated compatibility flag: suppress the one canonical default "
            "extra prefix. No filesystem glob discovery is performed."
        ),
    )
    parser.add_argument("--alpha", type=float, default=0.33)
    parser.add_argument("--omega-x", type=float, default=0.20)
    parser.add_argument("--sigma-xl", type=float, default=1.50)
    parser.add_argument("--population-growth", type=float, default=0.012)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--discount", type=float, default=0.04)
    parser.add_argument("--omega-m", type=float, default=0.35)
    parser.add_argument("--sigma-hm", type=float, default=2.00)
    parser.add_argument("--eta", type=float, default=0.45)
    parser.add_argument("--chi", type=float, default=0.01)
    parser.add_argument(
        "--no-fail-exit",
        action="store_true",
        help="Write a rejected report without returning a nonzero exit code.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Input file has no data rows: {path}")
    return rows


def write_rows(path: Path, rows: Iterable[dict[str, object]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"Refusing to write an empty audit table: {path}")
    fieldnames = list(
        dict.fromkeys(key for row in materialized for key in row.keys())
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialized)


def as_float(row: dict[str, str], field: str) -> float:
    if field not in row or row[field] == "":
        raise KeyError(f"Missing numeric field {field!r}.")
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError(f"Non-finite value in field {field!r}: {row[field]!r}")
    return value


def close_boundary(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-8)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def logsumexp(left: float, right: float) -> float:
    maximum = max(left, right)
    return maximum + math.log(
        math.exp(left - maximum) + math.exp(right - maximum)
    )


def log_ces(
    log_left: float,
    log_right: float,
    right_weight: float,
    elasticity: float,
) -> tuple[float, float]:
    """Return log CES quantity and the right input's technological share."""

    if not 0.0 < right_weight < 1.0:
        raise ValueError("CES weights must be strictly between zero and one.")
    if elasticity <= 0.0:
        raise ValueError("CES elasticities must be strictly positive.")
    if abs(elasticity - 1.0) <= 1e-12:
        log_quantity = (
            (1.0 - right_weight) * log_left
            + right_weight * log_right
        )
        return log_quantity, right_weight
    power = (elasticity - 1.0) / elasticity
    log_left_term = math.log1p(-right_weight) + power * log_left
    log_right_term = math.log(right_weight) + power * log_right
    log_denominator = logsumexp(log_left_term, log_right_term)
    return (
        log_denominator / power,
        math.exp(log_right_term - log_denominator),
    )


def singular_targets(parameters: Parameters) -> dict[str, float]:
    if parameters.sigma_xl <= 1.0 or parameters.sigma_hm <= 1.0:
        raise ValueError("This audit is only for sigma_XL>1 and sigma_HM>1.")
    kappa = (1.0 - parameters.alpha) / parameters.alpha
    denominator = 1.0 + kappa - parameters.eta
    capability_growth_to_z = parameters.eta * parameters.alpha / denominator
    inference_share = (1.0 - parameters.alpha) ** 2
    investment_share = (
        parameters.alpha - kappa * capability_growth_to_z
    )
    research_share = (
        parameters.eta * inference_share / denominator
    )
    consumption_share = (
        1.0 - inference_share - investment_share - research_share
    )
    return {
        "kappa": kappa,
        "denominator": denominator,
        "capability_growth_to_z": capability_growth_to_z,
        "inference_share": inference_share,
        "investment_share": investment_share,
        "research_share": research_share,
        "consumption_share": consumption_share,
        "shadow_capability_to_capital": (
            inference_share
            / (parameters.eta * parameters.alpha)
        ),
        "singularity_rate": kappa * capability_growth_to_z,
    }


def declared_prefixes(
    base_prefix: str,
    extra_prefixes: Sequence[str],
) -> list[str]:
    """Return only the explicitly declared canonical source prefixes.

    Deliberately do not inspect the directory for similarly named files.  A
    stale ``z256`` experiment must never alter the acceptance gates for the
    four paths plotted and discussed in the manuscript.
    """

    return list(dict.fromkeys([base_prefix, *extra_prefixes]))


def group_combined_paths(
    rows: Sequence[dict[str, str]],
) -> dict[float, list[dict[str, str]]]:
    grouped: dict[float, list[dict[str, str]]] = {}
    for row in rows:
        boundary = as_float(row, "terminal_boundary_z")
        key = round(boundary, 10)
        grouped.setdefault(key, []).append(row)
    return grouped


def load_inputs(
    result_dir: Path,
    prefixes: Sequence[str],
) -> tuple[
    dict[float, dict[str, str]],
    dict[float, list[dict[str, str]]],
    list[dict[str, object]],
    list[Path],
]:
    continuation_by_boundary: dict[float, dict[str, str]] = {}
    paths_by_boundary: dict[float, list[dict[str, str]]] = {}
    grouping_checks: list[dict[str, object]] = []
    inputs: list[Path] = []

    for prefix in prefixes:
        continuation_path = result_dir / f"{prefix}_free_continuation.csv"
        if not continuation_path.exists():
            if prefix == prefixes[0]:
                raise FileNotFoundError(continuation_path)
            continue
        continuation_rows = read_rows(continuation_path)
        inputs.append(continuation_path)
        combined_path = result_dir / f"{prefix}_boundary_paths.csv"
        if not combined_path.exists():
            raise FileNotFoundError(
                f"Canonical grouped path file is missing: {combined_path}"
            )
        combined_groups = group_combined_paths(read_rows(combined_path))
        inputs.append(combined_path)
        continuation_boundaries = {
            round(as_float(row, "terminal_output_capital_ratio"), 10)
            for row in continuation_rows
        }
        group_boundaries = set(combined_groups)
        boundary_sets_match = continuation_boundaries == group_boundaries

        for continuation in continuation_rows:
            boundary = round(
                as_float(continuation, "terminal_output_capital_ratio"), 10
            )
            if boundary in continuation_by_boundary:
                incumbent = continuation_by_boundary[boundary]
                if as_float(continuation, "max_rms_residual") >= as_float(
                    incumbent, "max_rms_residual"
                ):
                    continue
            continuation = dict(continuation)
            continuation["source_prefix"] = prefix
            continuation_by_boundary[boundary] = continuation
            if boundary not in combined_groups:
                continue
            rows = combined_groups[boundary]
            paths_by_boundary[boundary] = rows
            times = [as_float(row, "time") for row in rows]
            boundary_values = [
                as_float(row, "terminal_boundary_z") for row in rows
            ]
            scenarios = {row.get("scenario", "") for row in rows}
            endpoint = max(rows, key=lambda row: as_float(row, "time"))
            duration = as_float(continuation, "duration")
            grouping_checks.append(
                {
                    "prefix": prefix,
                    "combined_file": combined_path.name,
                    "boundary_z": boundary,
                    "boundary_sets_match": int(boundary_sets_match),
                    "group_rows": len(rows),
                    "scenario_count": len(scenarios),
                    "boundary_value_spread": (
                        max(boundary_values) - min(boundary_values)
                    ),
                    "time_grid_finite": int(
                        all(math.isfinite(value) for value in times)
                    ),
                    "time_grid_unique": int(len(times) == len(set(times))),
                    "time_grid_strictly_ordered": int(
                        all(right > left for left, right in zip(times, times[1:]))
                    ),
                    "duration_endpoint_error": abs(max(times) - duration),
                    "endpoint_output_capital_log_error": abs(
                        math.log(
                            as_float(endpoint, "output_capital_ratio")
                            / boundary
                        )
                    ),
                }
            )

    return continuation_by_boundary, paths_by_boundary, grouping_checks, sorted(
        set(inputs)
    )


def reconstruct_row(
    row: dict[str, str],
    parameters: Parameters,
) -> dict[str, object]:
    """Rebuild technologies, prices, FOCs, resources, and ODE right sides."""

    alpha = parameters.alpha
    omega_x = parameters.omega_x
    omega_m = parameters.omega_m
    log_k = as_float(row, "log_capital")
    log_a = as_float(row, "log_capability")
    log_n = as_float(row, "log_population")
    log_c = as_float(row, "log_consumption")
    log_q = as_float(row, "log_shadow_value")
    log_l = as_float(row, "log_production_labor")
    log_h = as_float(row, "log_human_research")
    log_u = as_float(row, "log_inference_compute")
    log_m = as_float(row, "log_automated_research")
    log_x = as_float(row, "log_ai_services")
    log_am = as_float(row, "log_automated_research_services")

    log_z, ai_share = log_ces(
        log_l, log_x, omega_x, parameters.sigma_xl
    )
    log_y = alpha * log_k + (1.0 - alpha) * log_z
    log_e, machine_share = log_ces(
        log_h, log_am, omega_m, parameters.sigma_hm
    )
    log_dot_a = math.log(parameters.chi) + parameters.eta * log_e
    capability_growth = math.exp(log_dot_a - log_a)

    log_wage = (
        math.log1p(-alpha)
        + math.log1p(-ai_share)
        + log_y
        - log_l
    )
    log_ai_price = (
        math.log1p(-alpha)
        + math.log(ai_share)
        + log_y
        - log_x
    )
    gross_return = alpha * math.exp(log_y - log_k)
    inverse_elasticity = (
        (1.0 - ai_share) / parameters.sigma_xl
        + alpha * ai_share
    )
    inverse_elasticity_derivative = (
        (alpha - 1.0 / parameters.sigma_xl)
        * (1.0 - 1.0 / parameters.sigma_xl)
        * ai_share
        * (1.0 - ai_share)
    )
    monopoly_soc_margin = (
        inverse_elasticity * (1.0 - inverse_elasticity)
        + inverse_elasticity_derivative
    )

    log_f_m = (
        math.log(parameters.chi)
        + math.log(parameters.eta)
        + (parameters.eta - 1.0) * log_e
        + math.log(omega_m)
        + log_a
        + (log_e - log_am) / parameters.sigma_hm
    )
    log_f_h = (
        math.log(parameters.chi)
        + math.log(parameters.eta)
        + (parameters.eta - 1.0) * log_e
        + math.log1p(-omega_m)
        + (log_e - log_h) / parameters.sigma_hm
    )

    z_yk = math.exp(log_y - log_k)
    consumption_share = math.exp(log_c - log_y)
    inference_share = math.exp(log_u - log_y)
    research_share = math.exp(log_m - log_y)
    labor_h_share = math.exp(log_h - log_n)
    labor_l_share = math.exp(log_l - log_n)
    machine_cost_share = math.exp(log_m) / (
        math.exp(log_m) + math.exp(log_wage + log_h)
    )
    production_labor_income_share = (1.0 - alpha) * (1.0 - ai_share)
    aggregate_labor_income_share = math.exp(log_wage + log_n - log_y)
    ai_operating_profit_share = (
        (1.0 - alpha) * ai_share - inference_share
    )
    integrated_profit_share = (
        ai_operating_profit_share
        - math.exp(log_wage + log_h - log_y)
        - research_share
    )

    saved_capital_growth = as_float(row, "capital_growth")
    saved_consumption_growth = as_float(row, "consumption_growth")
    saved_shadow_growth = as_float(row, "shadow_growth")
    investment_share = (saved_capital_growth + parameters.delta) / z_yk
    resource_residual = (
        consumption_share
        + inference_share
        + research_share
        + investment_share
        - 1.0
    )
    capital_growth_rhs = (
        z_yk
        - math.exp(log_c - log_k)
        - math.exp(log_u - log_k)
        - math.exp(log_m - log_k)
        - parameters.delta
    )
    consumption_growth_rhs = (
        parameters.n
        + gross_return
        - parameters.delta
        - parameters.discount
    )
    profit_shadow_ratio = math.exp(log_x - 2.0 * log_a - log_q)
    shadow_growth_rhs = (
        gross_return
        - parameters.delta
        - parameters.eta * machine_share * capability_growth
        - profit_shadow_ratio
    )
    household_budget_residual = (
        saved_capital_growth / z_yk
        - (
            alpha
            - parameters.delta / z_yk
            + aggregate_labor_income_share
            + integrated_profit_share
            - consumption_share
        )
    )

    bounded_arguments = (
        log_dot_a - log_a,
        log_y - log_k,
        log_c - log_k,
        log_u - log_k,
        log_m - log_k,
        log_u - log_y,
        log_m - log_y,
        log_x - 2.0 * log_a - log_q,
    )
    saved_ai_share = as_float(row, "ai_share")
    saved_human_share = as_float(row, "human_research_share")
    share_clipping = (
        abs(saved_ai_share - 1e-14) <= 1e-16
        or abs(saved_ai_share - (1.0 - 1e-14)) <= 1e-16
        or abs(saved_human_share - 1e-14) <= 1e-16
        or abs(saved_human_share - (1.0 - 1e-14)) <= 1e-16
    )
    positive = all(
        math.isfinite(value)
        for value in (
            log_k,
            log_a,
            log_n,
            log_c,
            log_q,
            log_l,
            log_h,
            log_u,
            log_m,
            log_x,
            log_am,
            log_z,
            log_y,
            log_e,
            log_wage,
            log_ai_price,
        )
    ) and min(
        consumption_share,
        inference_share,
        research_share,
        investment_share,
        labor_h_share,
        labor_l_share,
        ai_share,
        1.0 - ai_share,
        machine_share,
        1.0 - machine_share,
        monopoly_soc_margin,
    ) > 0.0

    result: dict[str, object] = {
        "scenario": row.get("scenario", ""),
        "terminal_boundary_z": as_float(row, "terminal_boundary_z"),
        "time": as_float(row, "time"),
        "path_label": PATH_LABEL,
        "population_law_log_residual": (
            log_n - parameters.n * as_float(row, "time")
        ),
        "final_ces_log_residual": (
            as_float(row, "log_service_composite") - log_z
        ),
        "final_output_log_residual": as_float(row, "log_output") - log_y,
        "ai_share_residual": as_float(row, "ai_share") - ai_share,
        "inference_identity_log_residual": log_x - log_a - log_u,
        "research_service_identity_log_residual": log_am - log_a - log_m,
        "research_ces_log_residual": (
            as_float(row, "log_effective_research") - log_e
        ),
        "capability_law_level_residual": (
            as_float(row, "capability_growth") - capability_growth
        ),
        "wage_log_residual": as_float(row, "log_wage") - log_wage,
        "ai_price_log_residual": (
            as_float(row, "log_ai_price") - log_ai_price
        ),
        "ai_marginal_cost_log_residual": (
            as_float(row, "log_ai_marginal_cost") + log_a
        ),
        "gross_return_residual": (
            as_float(row, "gross_capital_return") - gross_return
        ),
        "net_return_residual": (
            as_float(row, "net_capital_return")
            - (gross_return - parameters.delta)
        ),
        "inverse_elasticity_residual": (
            as_float(row, "inverse_demand_elasticity")
            - inverse_elasticity
        ),
        "monopoly_foc_log_residual": (
            log_ai_price
            + math.log(1.0 - inverse_elasticity)
            + log_a
        ),
        "research_compute_foc_log_residual": log_q + log_f_m,
        "research_human_foc_log_residual": log_q + log_f_h - log_wage,
        "labor_market_residual": labor_l_share + labor_h_share - 1.0,
        "resource_constraint_residual": resource_residual,
        "household_budget_residual": household_budget_residual,
        "final_firm_zero_profit_residual": (
            alpha
            + (1.0 - alpha) * (1.0 - ai_share)
            + (1.0 - alpha) * ai_share
            - 1.0
        ),
        "research_machine_technology_share_residual": (
            as_float(row, "automated_research_share") - machine_share
        ),
        "research_machine_cost_share_residual": (
            machine_cost_share - machine_share
        ),
        "inference_resource_share_residual": (
            as_float(row, "inference_share") - inference_share
        ),
        "research_resource_share_residual": (
            as_float(row, "research_resource_share") - research_share
        ),
        "investment_share_residual": (
            as_float(row, "investment_share") - investment_share
        ),
        "consumption_share_residual": (
            as_float(row, "consumption_share") - consumption_share
        ),
        "resource_share_sum_residual": (
            as_float(row, "resource_share_sum") - 1.0
        ),
        "production_labor_income_share_residual": (
            as_float(row, "production_labor_share")
            - production_labor_income_share
        ),
        "aggregate_labor_income_share_residual": (
            as_float(row, "aggregate_labor_share")
            - aggregate_labor_income_share
        ),
        "ai_operating_profit_share_residual": (
            as_float(row, "ai_profit_share") - ai_operating_profit_share
        ),
        "ai_markup_residual": (
            as_float(row, "ai_markup")
            - math.exp(log_ai_price + log_a)
        ),
        "shadow_capability_to_capital_residual": (
            as_float(row, "shadow_capability_to_capital")
            - math.exp(log_q + log_a - log_k)
        ),
        "capital_rhs_residual": saved_capital_growth - capital_growth_rhs,
        "capability_rhs_residual": (
            as_float(row, "capability_growth") - capability_growth
        ),
        "consumption_euler_rhs_residual": (
            saved_consumption_growth - consumption_growth_rhs
        ),
        "shadow_costate_rhs_residual": (
            saved_shadow_growth - shadow_growth_rhs
        ),
        "reconstructed_ai_share": ai_share,
        "reconstructed_research_machine_share": machine_share,
        "reconstructed_output_capital_ratio": z_yk,
        "reconstructed_capability_growth": capability_growth,
        "reconstructed_monopoly_soc_margin": monopoly_soc_margin,
        "minimum_positive_share_or_soc": min(
            consumption_share,
            inference_share,
            research_share,
            investment_share,
            labor_h_share,
            labor_l_share,
            ai_share,
            1.0 - ai_share,
            machine_share,
            1.0 - machine_share,
            monopoly_soc_margin,
        ),
        "positive_and_interior": int(positive),
        "share_clipping_detected": int(share_clipping),
        "bounded_exp_clipping_detected": int(
            min(bounded_arguments) <= -700.0
            or max(bounded_arguments) >= 60.0
        ),
        "minimum_bounded_exp_argument": min(bounded_arguments),
        "maximum_bounded_exp_argument": max(bounded_arguments),
    }
    for field in STORED_DYNAMIC_FIELDS:
        result[f"stored_{field}"] = as_float(row, field)
    return result


def maximum_absolute(
    rows: Sequence[dict[str, object]], fields: Sequence[str]
) -> float:
    return max(abs(float(row[field])) for row in rows for field in fields)


def independent_dynamic_error(
    continuation: dict[float, dict[str, str]],
    paths: dict[float, list[dict[str, str]]],
    terminal_buffer: float,
) -> float:
    """Finite-difference the saved states away from the artificial boundary.

    This does not reuse the BVP's derivative residual columns.  A centered
    quarter-year difference is compared with each saved right-hand side on an
    annual audit grid.  The terminal buffer avoids treating the singular tail's
    interpolation error as an equilibrium residual.
    """

    state_rhs = (
        ("log_capital", "capital_growth"),
        ("log_capability", "capability_growth"),
        ("log_consumption", "consumption_growth"),
        ("log_shadow_value", "shadow_growth"),
    )
    step = 0.25
    maximum = 0.0
    for boundary, path_rows in paths.items():
        end = as_float(continuation[boundary], "duration") - terminal_buffer
        if end <= 2.0:
            raise ValueError("Not enough nonterminal path for a dynamic audit.")
        for state_field, rhs_field in state_rhs:
            state_times, state_values = interpolation_series(
                path_rows, state_field
            )
            rhs_times, rhs_values = interpolation_series(path_rows, rhs_field)
            time = 1.0
            while time < end:
                derivative = (
                    interpolate(state_times, state_values, time + step)
                    - interpolate(state_times, state_values, time - step)
                ) / (2.0 * step)
                rhs = interpolate(rhs_times, rhs_values, time)
                maximum = max(maximum, abs(derivative - rhs))
                time += 1.0
    return maximum


def interpolation_series(
    rows: Sequence[dict[str, str]], field: str
) -> tuple[list[float], list[float]]:
    ordered = sorted(rows, key=lambda row: as_float(row, "time"))
    times: list[float] = []
    values: list[float] = []
    for row in ordered:
        time = as_float(row, "time")
        value = as_float(row, field)
        if times and time <= times[-1]:
            if math.isclose(time, times[-1], rel_tol=0.0, abs_tol=1e-12):
                times[-1] = time
                values[-1] = value
                continue
            raise ValueError("Path times are not strictly increasing.")
        times.append(time)
        values.append(value)
    return times, values


def interpolate(times: Sequence[float], values: Sequence[float], time: float) -> float:
    if time < times[0] - 1e-10 or time > times[-1] + 1e-10:
        raise ValueError(f"Interpolation time {time} lies outside the path.")
    if time <= times[0]:
        return values[0]
    if time >= times[-1]:
        return values[-1]
    upper = bisect.bisect_right(times, time)
    lower = upper - 1
    weight = (time - times[lower]) / (times[upper] - times[lower])
    return values[lower] + weight * (values[upper] - values[lower])


def regular_grid(start: float, stop: float, count: int = 1001) -> list[float]:
    if stop <= start:
        return [start]
    return [start + (stop - start) * index / (count - 1) for index in range(count)]


def window_spread(
    selected: dict[float, list[dict[str, str]]],
    field: str,
    grid: Sequence[float],
    shifts: dict[float, float] | None = None,
) -> float:
    series = {
        boundary: interpolation_series(rows, field)
        for boundary, rows in selected.items()
    }
    maximum = 0.0
    for point in grid:
        values = []
        for boundary, (times, observations) in series.items():
            evaluation_time = point + (shifts or {}).get(boundary, 0.0)
            values.append(interpolate(times, observations, evaluation_time))
        maximum = max(maximum, max(values) - min(values))
    return maximum


def crossing_time(rows: Sequence[dict[str, str]], event_z: float) -> tuple[float, int]:
    times, values = interpolation_series(rows, "output_capital_ratio")
    crossings: list[float] = []
    for index in range(1, len(times)):
        left = values[index - 1] - event_z
        right = values[index] - event_z
        if left == 0.0:
            crossings.append(times[index - 1])
        elif left * right < 0.0 or right == 0.0:
            fraction = -left / (right - left)
            crossings.append(
                times[index - 1]
                + fraction * (times[index] - times[index - 1])
            )
    terminal_is_event = math.isclose(
        values[-1], event_z, rel_tol=1e-9, abs_tol=1e-8
    )
    if terminal_is_event and (
        not crossings or abs(crossings[-1] - times[-1]) > 1e-10
    ):
        crossings.append(times[-1])
    if not crossings:
        raise ValueError(f"Path does not cross event z={event_z:g}.")
    return crossings[-1], len(crossings)


def add_convergence_row(
    rows: list[dict[str, object]],
    category: str,
    comparison: str,
    variable: str,
    value: float,
    threshold: float | None,
    status: str,
    details: str = "",
) -> None:
    rows.append(
        {
            "category": category,
            "comparison": comparison,
            "variable": variable,
            "value": value,
            "threshold": "" if threshold is None else threshold,
            "status": status,
            "details": details,
            "path_label": PATH_LABEL,
        }
    )


def convergence_checks(
    continuation: dict[float, dict[str, str]],
    paths: dict[float, list[dict[str, str]]],
    required: Sequence[float],
    parameters: Parameters,
    tolerances: Tolerances,
) -> tuple[list[dict[str, object]], dict[str, bool]]:
    rows: list[dict[str, object]] = []
    gates: dict[str, bool] = {}
    available = sorted(boundary for boundary in continuation if boundary in paths)
    targets = singular_targets(parameters)

    for field in ("initial_log_consumption", "initial_log_shadow_value"):
        deltas: list[float] = []
        for lower, upper in zip(available[:-1], available[1:]):
            delta = abs(
                as_float(continuation[upper], field)
                - as_float(continuation[lower], field)
            )
            deltas.append(delta)
            add_convergence_row(
                rows,
                "initial_jump",
                f"z={lower:g} to z={upper:g}",
                field,
                delta,
                tolerances.jump_final if upper == available[-1] else None,
                (
                    "pass"
                    if upper != available[-1] or delta <= tolerances.jump_final
                    else "fail"
                ),
            )
        if deltas:
            gates[f"jump_{field}"] = deltas[-1] <= tolerances.jump_final

    tstars: dict[float, float] = {}
    saved_tstar_errors: list[float] = []
    for boundary in available:
        duration = as_float(continuation[boundary], "duration")
        tstar = duration + 1.0 / (targets["singularity_rate"] * boundary)
        tstars[boundary] = tstar
        saved = continuation[boundary].get("estimated_singularity_time", "")
        saved_error = (
            abs(float(saved) - tstar) if saved not in ("", None) else math.inf
        )
        saved_tstar_errors.append(saved_error)
        add_convergence_row(
            rows,
            "singularity_time",
            f"z={boundary:g}",
            "saved_T_star_error",
            saved_error,
            tolerances.saved_tstar,
            "pass" if saved_error <= tolerances.saved_tstar else "fail",
            f"recomputed_T_star={tstar:.17g}",
        )
    gates["saved_T_star_matches_recomputed"] = bool(saved_tstar_errors) and all(
        error <= tolerances.saved_tstar for error in saved_tstar_errors
    )
    tstar_deltas: list[float] = []
    for lower, upper in zip(available[:-1], available[1:]):
        delta = abs(tstars[upper] - tstars[lower])
        tstar_deltas.append(delta)
        add_convergence_row(
            rows,
            "singularity_time_increment",
            f"z={lower:g} to z={upper:g}",
            "abs_delta_T_star",
            delta,
            tolerances.tstar_final if upper == available[-1] else None,
            (
                "pass"
                if upper != available[-1] or delta <= tolerances.tstar_final
                else "fail"
            ),
        )
    if tstar_deltas:
        gates["T_star_final_difference"] = (
            tstar_deltas[-1] <= tolerances.tstar_final
        )
    if len(tstar_deltas) >= 2 and tstar_deltas[-2] > 1e-14:
        contraction = tstar_deltas[-1] / tstar_deltas[-2]
        gates["T_star_contraction"] = contraction <= tolerances.tstar_contraction
        add_convergence_row(
            rows,
            "singularity_time_contraction",
            f"last {len(tstar_deltas) + 1} boundaries",
            "last_delta_over_previous_delta",
            contraction,
            tolerances.tstar_contraction,
            "pass" if gates["T_star_contraction"] else "fail",
        )

    required_available = [boundary for boundary in required if boundary in paths]
    if len(required_available) >= 2:
        common_stop = min(
            as_float(continuation[boundary], "duration")
            for boundary in required_available
        ) - tolerances.common_window_terminal_buffer
        if common_stop <= 0.0:
            raise ValueError("The common-window terminal buffer is too large.")
        selected = {boundary: paths[boundary] for boundary in required_available}
        grid = regular_grid(0.0, common_stop)
        for field in COMMON_LOG_FIELDS + COMMON_LEVEL_FIELDS:
            spread = window_spread(selected, field, grid)
            is_log = field in COMMON_LOG_FIELDS
            threshold = (
                tolerances.common_window_log
                if is_log
                else tolerances.common_window_level
            )
            gate = spread <= threshold
            gates[f"common_window_{field}"] = gate
            add_convergence_row(
                rows,
                "common_calendar_window",
                ",".join(f"z={value:g}" for value in required_available),
                field,
                spread,
                threshold,
                "pass" if gate else "fail",
                (
                    f"calendar_window=[0,{common_stop:.9g}]; excludes the last "
                    f"{tolerances.common_window_terminal_buffer:g} years before "
                    "the nearest artificial boundary"
                ),
            )

    for event_z in required:
        eligible = {
            boundary: paths[boundary]
            for boundary in available
            if boundary + 1e-8 >= 2.0 * event_z
        }
        event_is_gated = len(eligible) >= 2
        if not event_is_gated:
            # A single outer path cannot establish convergence.  Still compute
            # the requested event window using the path terminating at the
            # event and the next outer path, but label it explicitly as a
            # boundary-contaminated diagnostic rather than an acceptance gate.
            eligible = {
                boundary: paths[boundary]
                for boundary in available
                if boundary + 1e-8 >= event_z
            }
        event_times: dict[float, float] = {}
        crossing_counts: dict[float, int] = {}
        for boundary, path_rows in eligible.items():
            try:
                event_times[boundary], crossing_counts[boundary] = crossing_time(
                    path_rows, event_z
                )
            except ValueError:
                continue
        if not event_times:
            add_convergence_row(
                rows,
                "event_window",
                f"event z={event_z:g}",
                "event_available",
                0.0,
                1.0,
                "fail",
            )
            gates[f"event_{event_z:g}_available"] = False
            continue
        time_spread = max(event_times.values()) - min(event_times.values())
        time_status = (
            "not_testable"
            if len(event_times) == 1
            else (
                ("pass" if time_spread <= tolerances.event_time else "fail")
                if event_is_gated
                else "not_gated"
            )
        )
        add_convergence_row(
            rows,
            "event_time",
            f"event z={event_z:g}",
            "crossing_time_spread",
            time_spread,
            tolerances.event_time if len(event_times) > 1 else None,
            time_status,
            "; ".join(
                f"boundary {boundary:g}: t={event_times[boundary]:.9g}, "
                f"crossings={crossing_counts[boundary]}"
                for boundary in sorted(event_times)
            ),
        )
        if len(event_times) > 1:
            if event_is_gated:
                gates[f"event_{event_z:g}_time"] = (
                    time_spread <= tolerances.event_time
                )
            lead = min(10.0, min(event_times.values()))
            offsets = regular_grid(-lead, 0.0, 201)
            # Convergence at an event must not compare the path whose artificial
            # terminal condition is imposed at that event.  Use the two most
            # distant available boundaries, i.e. the latest continuation pair.
            comparison_boundaries = sorted(event_times)[-2:]
            selected = {
                boundary: paths[boundary]
                for boundary in comparison_boundaries
            }
            comparison_times = {
                boundary: event_times[boundary]
                for boundary in comparison_boundaries
            }
            for field in COMMON_LOG_FIELDS + COMMON_LEVEL_FIELDS:
                spread = window_spread(
                    selected,
                    field,
                    offsets,
                    shifts=comparison_times,
                )
                is_log = field in COMMON_LOG_FIELDS
                threshold = (
                    tolerances.event_window_log
                    if is_log
                    else tolerances.event_window_level
                )
                gate = spread <= threshold
                if event_is_gated:
                    gates[f"event_{event_z:g}_{field}"] = gate
                add_convergence_row(
                    rows,
                    "event_aligned_window",
                    f"event z={event_z:g}",
                    field,
                    spread,
                    threshold,
                    (
                        ("pass" if gate else "fail")
                        if event_is_gated
                        else "not_gated"
                    ),
                    (
                        f"aligned_window=[-{lead:g},0] years; latest outer "
                        "boundaries="
                        + ",".join(
                            f"{boundary:g}"
                            for boundary in comparison_boundaries
                        )
                        + (
                            "; includes a path whose artificial boundary is "
                            "the event, so this row is diagnostic only"
                            if not event_is_gated
                            else ""
                        )
                    ),
                )
        else:
            add_convergence_row(
                rows,
                "event_aligned_window",
                f"event z={event_z:g}",
                "cross_boundary_comparison",
                1.0,
                None,
                "not_testable",
                "Only one path reaches this event; recorded but not gated.",
            )

    terminal_metrics = {
        "capability_growth_to_output_capital": targets[
            "capability_growth_to_z"
        ],
        "inference_share": targets["inference_share"],
        "research_resource_share": targets["research_share"],
        "investment_share": targets["investment_share"],
        "consumption_share": targets["consumption_share"],
        "shadow_capability_to_capital": targets[
            "shadow_capability_to_capital"
        ],
        "ai_share": 1.0,
        "automated_research_share": 1.0,
    }
    terminal_errors: dict[str, list[tuple[float, float]]] = {
        field: [] for field in terminal_metrics
    }
    for boundary in available:
        terminal_row = max(paths[boundary], key=lambda row: as_float(row, "time"))
        for field, target in terminal_metrics.items():
            error = abs(as_float(terminal_row, field) - target)
            terminal_errors[field].append((boundary, error))
            add_convergence_row(
                rows,
                "terminal_asymptotic_ratio",
                f"z={boundary:g}",
                field,
                error,
                (
                    tolerances.terminal_ratio_error
                    if boundary == available[-1]
                    else None
                ),
                (
                    "pass"
                    if boundary != available[-1]
                    or error <= tolerances.terminal_ratio_error
                    else "fail"
                ),
                f"absolute error from target {target:.12g}",
            )
    for field, observations in terminal_errors.items():
        if observations:
            final_error = observations[-1][1]
            gates[f"terminal_{field}"] = (
                final_error <= tolerances.terminal_ratio_error
            )
            if len(observations) >= 2:
                gates[f"terminal_{field}_nonworsening"] = (
                    observations[-1][1] <= observations[-2][1] + 1e-10
                )
                add_convergence_row(
                    rows,
                    "terminal_asymptotic_convergence",
                    f"z={observations[-2][0]:g} to z={observations[-1][0]:g}",
                    field,
                    observations[-1][1] - observations[-2][1],
                    1e-10,
                    (
                        "pass"
                        if gates[f"terminal_{field}_nonworsening"]
                        else "fail"
                    ),
                    "A nonpositive change means the target error did not worsen.",
                )
    return rows, gates


def gate_row(
    group: str,
    gate: str,
    value: float | int | str,
    comparison: str,
    threshold: float | int | str,
    passed: bool,
    details: str = "",
) -> dict[str, object]:
    return {
        "gate_group": group,
        "gate": gate,
        "value": value,
        "comparison": comparison,
        "threshold": threshold,
        "status": "pass" if passed else "fail",
        "details": details,
        "path_label": PATH_LABEL,
    }


def build_acceptance_report(
    required: Sequence[float],
    continuation: dict[float, dict[str, str]],
    paths: dict[float, list[dict[str, str]]],
    grouping_checks: Sequence[dict[str, object]],
    residuals: Sequence[dict[str, object]],
    convergence_gates: dict[str, bool],
    parameters: Parameters,
    tolerances: Tolerances,
) -> tuple[list[dict[str, object]], bool]:
    report: list[dict[str, object]] = []
    missing = [
        boundary
        for boundary in required
        if boundary not in continuation or boundary not in paths
    ]
    report.append(
        gate_row(
            "inputs",
            "required_boundaries_present",
            len(required) - len(missing),
            "==",
            len(required),
            not missing,
            "missing=" + ",".join(f"{value:g}" for value in missing),
        )
    )
    required_grouping = [
        row
        for row in grouping_checks
        if any(
            close_boundary(float(row["boundary_z"]), boundary)
            for boundary in required
        )
    ]
    sets_match = bool(grouping_checks) and all(
        int(row["boundary_sets_match"]) == 1 for row in grouping_checks
    )
    report.append(
        gate_row(
            "inputs",
            "continuation_matches_canonical_groups",
            int(sets_match),
            "==",
            1,
            sets_match,
            f"canonical_required_groups={len(required_grouping)}",
        )
    )
    boundary_spread = max(
        (float(row["boundary_value_spread"]) for row in grouping_checks),
        default=math.inf,
    )
    report.append(
        gate_row(
            "inputs",
            "terminal_boundary_constant_within_group",
            boundary_spread,
            "<=",
            tolerances.group_boundary_spread,
            boundary_spread <= tolerances.group_boundary_spread,
        )
    )
    one_scenario = bool(grouping_checks) and all(
        int(row["scenario_count"]) == 1 for row in grouping_checks
    )
    report.append(
        gate_row(
            "inputs",
            "one_scenario_per_boundary_group",
            int(one_scenario),
            "==",
            1,
            one_scenario,
        )
    )
    valid_time_grids = bool(grouping_checks) and all(
        int(row["time_grid_finite"]) == 1
        and int(row["time_grid_unique"]) == 1
        and int(row["time_grid_strictly_ordered"]) == 1
        for row in grouping_checks
    )
    report.append(
        gate_row(
            "inputs",
            "finite_unique_strictly_ordered_time_grids",
            int(valid_time_grids),
            "==",
            1,
            valid_time_grids,
        )
    )
    duration_error = max(
        (float(row["duration_endpoint_error"]) for row in grouping_checks),
        default=math.inf,
    )
    report.append(
        gate_row(
            "inputs",
            "continuation_duration_matches_group_endpoint",
            duration_error,
            "<=",
            tolerances.duration_endpoint,
            duration_error <= tolerances.duration_endpoint,
        )
    )
    grouped_endpoint_error = max(
        (
            float(row["endpoint_output_capital_log_error"])
            for row in grouping_checks
        ),
        default=math.inf,
    )
    report.append(
        gate_row(
            "inputs",
            "canonical_group_endpoint_matches_boundary",
            grouped_endpoint_error,
            "<=",
            tolerances.boundary_log,
            grouped_endpoint_error <= tolerances.boundary_log,
        )
    )
    sigma_errors = [
        abs(as_float(row, "sigma_xl") - parameters.sigma_xl)
        for row in continuation.values()
    ]
    report.append(
        gate_row(
            "inputs",
            "sigma_XL_matches_audit_parameters",
            max(sigma_errors, default=math.inf),
            "<=",
            1e-12,
            bool(sigma_errors) and max(sigma_errors) <= 1e-12,
        )
    )
    solver_rms = max(
        (as_float(row, "max_rms_residual") for row in continuation.values()),
        default=math.inf,
    )
    report.append(
        gate_row(
            "solver",
            "free_boundary_solver_rms",
            solver_rms,
            "<=",
            tolerances.solver_rms,
            solver_rms <= tolerances.solver_rms,
        )
    )

    terminal_boundary_errors = []
    for boundary, path_rows in paths.items():
        terminal = max(path_rows, key=lambda row: as_float(row, "time"))
        terminal_boundary_errors.append(
            abs(
                math.log(
                    as_float(terminal, "output_capital_ratio") / boundary
                )
            )
        )
    maximum_boundary_error = max(terminal_boundary_errors, default=math.inf)
    report.append(
        gate_row(
            "boundary",
            "terminal_output_capital_target",
            maximum_boundary_error,
            "<=",
            tolerances.boundary_log,
            maximum_boundary_error <= tolerances.boundary_log,
        )
    )

    log_equations = (
        "population_law_log_residual",
        "final_ces_log_residual",
        "final_output_log_residual",
        "inference_identity_log_residual",
        "research_service_identity_log_residual",
        "research_ces_log_residual",
        "wage_log_residual",
        "ai_price_log_residual",
        "ai_marginal_cost_log_residual",
    )
    level_equations = (
        "ai_share_residual",
        "capability_law_level_residual",
        "gross_return_residual",
        "net_return_residual",
        "inverse_elasticity_residual",
        "labor_market_residual",
        "resource_constraint_residual",
        "household_budget_residual",
        "final_firm_zero_profit_residual",
        "research_machine_technology_share_residual",
        "research_machine_cost_share_residual",
        "inference_resource_share_residual",
        "research_resource_share_residual",
        "investment_share_residual",
        "consumption_share_residual",
        "resource_share_sum_residual",
        "production_labor_income_share_residual",
        "aggregate_labor_income_share_residual",
        "ai_operating_profit_share_residual",
        "ai_markup_residual",
        "shadow_capability_to_capital_residual",
    )
    foc_equations = (
        "monopoly_foc_log_residual",
        "research_compute_foc_log_residual",
        "research_human_foc_log_residual",
    )
    rhs_equations = (
        "capital_rhs_residual",
        "capability_rhs_residual",
        "consumption_euler_rhs_residual",
        "shadow_costate_rhs_residual",
    )
    for name, fields, tolerance in (
        ("technologies_and_prices", log_equations, tolerances.equation_log),
        ("shares_and_resources", level_equations, tolerances.equation_level),
        ("static_first_order_conditions", foc_equations, tolerances.foc_log),
        ("dynamic_rhs_reconstruction", rhs_equations, tolerances.equation_level),
    ):
        maximum = maximum_absolute(residuals, fields)
        report.append(
            gate_row(
                "equations",
                name,
                maximum,
                "<=",
                tolerance,
                maximum <= tolerance,
                "fields=" + ",".join(fields),
            )
        )
    stored_dynamic = maximum_absolute(
        residuals,
        tuple(f"stored_{field}" for field in STORED_DYNAMIC_FIELDS),
    )
    report.append(
        gate_row(
            "dynamics",
            "saved_path_derivative_residuals",
            stored_dynamic,
            "<=",
            tolerances.stored_dynamic,
            stored_dynamic <= tolerances.stored_dynamic,
            "Audits the residual columns saved by the BVP evaluator.",
        )
    )
    independently_differenced = independent_dynamic_error(
        continuation,
        paths,
        tolerances.common_window_terminal_buffer,
    )
    report.append(
        gate_row(
            "dynamics",
            "independently_differenced_compact_path",
            independently_differenced,
            "<=",
            tolerances.independent_dynamic,
            independently_differenced <= tolerances.independent_dynamic,
            (
                "Centered quarter-year differences on an annual grid, "
                f"excluding the last "
                f"{tolerances.common_window_terminal_buffer:g} years."
            ),
        )
    )
    min_positive = min(
        float(row["minimum_positive_share_or_soc"]) for row in residuals
    )
    all_positive = all(int(row["positive_and_interior"]) == 1 for row in residuals)
    report.append(
        gate_row(
            "interiority",
            "positive_allocations_and_monopoly_SOC",
            min_positive,
            ">",
            0.0,
            all_positive and min_positive > 0.0,
        )
    )
    clipping_count = sum(
        int(row["share_clipping_detected"])
        + int(row["bounded_exp_clipping_detected"])
        for row in residuals
    )
    report.append(
        gate_row(
            "interiority",
            "no_share_or_bounded_exp_clipping",
            clipping_count,
            "==",
            0,
            clipping_count == 0,
        )
    )
    for name, passed in sorted(convergence_gates.items()):
        report.append(
            gate_row(
                "boundary_convergence",
                name,
                int(passed),
                "==",
                1,
                passed,
            )
        )

    parameter_gate = (
        0.0 < parameters.alpha < 1.0
        and 0.0 < parameters.omega_x < 1.0
        and 0.0 < parameters.omega_m < 1.0
        and parameters.sigma_xl > 1.0
        and parameters.sigma_hm > 1.0
        and 0.0 < parameters.eta < 1.0
        and parameters.chi > 0.0
    )
    report.append(
        gate_row(
            "scope",
            "high_sigma_A_times_M_parameter_domain",
            int(parameter_gate),
            "==",
            1,
            parameter_gate,
        )
    )
    passed = all(row["status"] == "pass" for row in report)
    report.insert(
        0,
        gate_row(
            "overall",
            "accepted_as_convergent_finite_boundary_approximations",
            int(passed),
            "==",
            1,
            passed,
            (
                "Acceptance does not establish an infinite-horizon equilibrium, "
                "a TVC, global optimality, or reachability from arbitrary states."
            ),
        ),
    )
    return report, passed


def main() -> None:
    arguments = parse_arguments()
    result_dir = arguments.result_dir.resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    required = tuple(
        float(value) for value in arguments.required_boundaries.split(",")
    )
    parameters = Parameters(
        alpha=arguments.alpha,
        omega_x=arguments.omega_x,
        sigma_xl=arguments.sigma_xl,
        n=arguments.population_growth,
        delta=arguments.delta,
        discount=arguments.discount,
        omega_m=arguments.omega_m,
        sigma_hm=arguments.sigma_hm,
        eta=arguments.eta,
        chi=arguments.chi,
    )
    tolerances = Tolerances()
    if arguments.no_auto_extra_boundaries:
        extra_prefixes: tuple[str, ...] = ()
    elif arguments.extra_prefixes is None:
        extra_prefixes = (
            (DEFAULT_EXTRA_PREFIX,)
            if arguments.input_prefix == DEFAULT_PREFIX
            else ()
        )
    else:
        extra_prefixes = tuple(
            prefix.strip()
            for prefix in arguments.extra_prefixes.split(",")
            if prefix.strip()
        )
    prefixes = declared_prefixes(arguments.input_prefix, extra_prefixes)
    continuation, paths, grouping_checks, input_files = load_inputs(
        result_dir, prefixes
    )

    residual_rows: list[dict[str, object]] = []
    for boundary in sorted(paths):
        previous_time = -math.inf
        for row in paths[boundary]:
            time = as_float(row, "time")
            if time <= previous_time:
                raise ValueError(
                    f"Non-increasing time grid in boundary z={boundary:g}."
                )
            previous_time = time
            residual_rows.append(reconstruct_row(row, parameters))

    convergence_rows, convergence_gates = convergence_checks(
        continuation,
        paths,
        required,
        parameters,
        tolerances,
    )
    for row in grouping_checks:
        boundary = float(row["boundary_z"])
        checks = (
            (
                "boundary_value_spread",
                float(row["boundary_value_spread"]),
                tolerances.group_boundary_spread,
            ),
            (
                "duration_endpoint_error",
                float(row["duration_endpoint_error"]),
                tolerances.duration_endpoint,
            ),
            (
                "endpoint_output_capital_log_error",
                float(row["endpoint_output_capital_log_error"]),
                tolerances.boundary_log,
            ),
        )
        for variable, value, threshold in checks:
            add_convergence_row(
                convergence_rows,
                "canonical_grouping",
                f"z={boundary:g}",
                variable,
                value,
                threshold,
                "pass" if value <= threshold else "fail",
                f"canonical_file={row['combined_file']}",
            )
        structural = (
            "boundary_sets_match",
            "scenario_count",
            "time_grid_finite",
            "time_grid_unique",
            "time_grid_strictly_ordered",
        )
        for variable in structural:
            value = int(row[variable])
            add_convergence_row(
                convergence_rows,
                "canonical_grouping",
                f"z={boundary:g}",
                variable,
                float(value),
                1.0,
                "pass" if value == 1 else "fail",
                f"canonical_file={row['combined_file']}",
            )

    acceptance_rows, accepted = build_acceptance_report(
        required,
        continuation,
        paths,
        grouping_checks,
        residual_rows,
        convergence_gates,
        parameters,
        tolerances,
    )

    output_stem = result_dir / arguments.input_prefix
    acceptance_path = Path(f"{output_stem}_acceptance_report.csv")
    residual_path = Path(f"{output_stem}_equation_residuals.csv")
    convergence_path = Path(f"{output_stem}_boundary_convergence.csv")
    manifest_path = Path(f"{output_stem}_audit_manifest.json")
    write_rows(acceptance_path, acceptance_rows)
    write_rows(residual_path, residual_rows)
    write_rows(convergence_path, convergence_rows)

    manifest = {
        "audit_version": 3,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "path_label": PATH_LABEL,
        "overall_accepted": accepted,
        "accepted_claim": (
            "The saved finite-boundary paths satisfy the audited necessary "
            "conditions and the specified boundary-convergence gates."
        ),
        "claims_not_established": [
            "existence of an infinite-horizon equilibrium",
            "transversality at infinity",
            "global optimality of the household or developer program",
            "reachability of the conditional branch from arbitrary initial states",
        ],
        "independence": (
            "All reported model equations were reconstructed in this script; "
            "no equilibrium_static_block or solver equation function was called."
        ),
        "canonical_input_policy": (
            "Only the explicitly declared source prefixes are ingested; no "
            "similarly named files are discovered. For each prefix, "
            "free_continuation.csv supplies boundary metadata and "
            "boundary_paths.csv is the sole path-level input. Per-boundary "
            "and final-boundary path copies are not ingested."
        ),
        "parameters": asdict(parameters),
        "tolerances": asdict(tolerances),
        "required_boundaries": list(required),
        "audited_boundaries": sorted(paths),
        "source_prefixes": prefixes,
        "row_counts": {
            "equation_residuals": len(residual_rows),
            "boundary_convergence": len(convergence_rows),
            "acceptance_gates": len(acceptance_rows),
        },
        "inputs": [
            {
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in input_files
        ],
        "outputs": [
            {
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in (acceptance_path, residual_path, convergence_path)
        ],
        "audit_script": {
            "path": str(Path(__file__).resolve().relative_to(ROOT)),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    status = "PASS" if accepted else "FAIL"
    print(
        f"{status}: {PATH_LABEL}; "
        f"boundaries={','.join(f'{value:g}' for value in sorted(paths))}; "
        f"rows={len(residual_rows)}"
    )
    print(f"Acceptance report: {acceptance_path}")
    print(f"Manifest: {manifest_path}")
    if not accepted and not arguments.no_fail_exit:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
