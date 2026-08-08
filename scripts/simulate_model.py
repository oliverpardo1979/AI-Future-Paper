"""Numerical transition experiments for the AI research-automation model.

The experiments retain the static CES production block, monopoly AI-service
supply, the cost-minimizing CES research mix, capital accumulation, population
growth, and the ideas-production function. To isolate the technological feedbacks,
investment and automated-research spending are held at constant output shares.
Those shares are calibrated to the analytical Cobb--Douglas balanced-growth path.

This is an illustrative transition system, not yet the full saddle-path solution
for household consumption and the developer's shadow value of capability.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "figures"
RESULT_DIR = ROOT / "numerical"


@dataclass(frozen=True)
class Parameters:
    alpha: float = 0.33
    omega: float = 0.20
    sigma: float = 1.00
    n: float = 0.012
    delta: float = 0.05
    discount: float = 0.04
    xi: float = 1.00
    nu: float = 0.35
    theta: float = 2.00
    phi: float = 0.65
    eta: float = 0.45
    chi: float = 0.01
    investment_share: float = 0.22
    research_compute_share: float = 0.005


def logistic(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def bisect_root(
    function: Callable[[float], float],
    lower: float,
    upper: float,
    tolerance: float = 1e-6,
    iterations: int = 55,
) -> float:
    f_lower = function(lower)
    f_upper = function(upper)
    if not (math.isfinite(f_lower) and math.isfinite(f_upper)):
        raise ValueError("Root endpoints must be finite.")
    if f_lower == 0:
        return lower
    if f_upper == 0:
        return upper
    if f_lower * f_upper > 0:
        raise ValueError(f"Root is not bracketed: {f_lower=}, {f_upper=}.")
    for _ in range(iterations):
        middle = 0.5 * (lower + upper)
        f_middle = function(middle)
        if abs(f_middle) < tolerance or abs(upper - lower) < tolerance:
            return middle
        if f_lower * f_middle <= 0:
            upper = middle
            f_upper = f_middle
        else:
            lower = middle
            f_lower = f_middle
    return 0.5 * (lower + upper)


def bracket_root(
    function: Callable[[float], float],
    grid: Iterable[float],
    descending_crossing: bool = False,
) -> tuple[float, float]:
    previous_x: float | None = None
    previous_value: float | None = None
    fallback: tuple[float, float] | None = None
    for x_value in grid:
        try:
            value = function(float(x_value))
        except (ArithmeticError, ValueError, OverflowError):
            continue
        if not math.isfinite(value):
            continue
        if previous_x is not None and previous_value is not None:
            if previous_value == 0:
                return previous_x, previous_x
            if previous_value * value <= 0:
                candidate = (previous_x, float(x_value))
                if descending_crossing and previous_value > 0 >= value:
                    return candidate
                if fallback is None:
                    fallback = candidate
        previous_x = float(x_value)
        previous_value = value
    if fallback is not None:
        return fallback
    raise ValueError("Could not bracket a root on the supplied grid.")


def production_from_ratio(
    log_capital: float,
    log_labor: float,
    log_ai_labor_ratio: float,
    parameters: Parameters,
) -> dict[str, float]:
    alpha = parameters.alpha
    omega = parameters.omega
    sigma = parameters.sigma

    if abs(sigma - 1.0) < 1e-9:
        log_z_per_worker = omega * log_ai_labor_ratio
        ai_share = omega
    else:
        rho = (sigma - 1.0) / sigma
        log_human_term = math.log1p(-omega)
        log_ai_term = math.log(omega) + rho * log_ai_labor_ratio
        log_denominator = float(np.logaddexp(log_human_term, log_ai_term))
        log_z_per_worker = log_denominator / rho
        ai_share = math.exp(log_ai_term - log_denominator)

    log_output = (
        alpha * log_capital
        + (1.0 - alpha) * (log_labor + log_z_per_worker)
    )
    return {
        "log_output": log_output,
        "ai_share": min(max(ai_share, 1e-14), 1.0 - 1e-14),
    }


def monopoly_ai_ratio(
    log_capital: float,
    log_labor: float,
    log_capability: float,
    parameters: Parameters,
) -> float:
    """Solve the monopolist's static first-order condition for log(X/L_Y)."""

    def residual(log_ratio: float) -> float:
        production = production_from_ratio(
            log_capital, log_labor, log_ratio, parameters
        )
        ai_share = production["ai_share"]
        inverse_demand_elasticity = (
            (1.0 - ai_share) / parameters.sigma
            + parameters.alpha * ai_share
        )
        markup_term = 1.0 - inverse_demand_elasticity
        if markup_term <= 0:
            return -1e6
        log_ai_services = log_labor + log_ratio
        log_price = (
            math.log1p(-parameters.alpha)
            + math.log(ai_share)
            + production["log_output"]
            - log_ai_services
        )
        return (
            log_price
            + math.log(markup_term)
            - math.log(parameters.xi)
            + log_capability
        )

    lower, upper = -45.0, 45.0
    return bisect_root(residual, lower, upper)


def research_aggregator(
    log_human_research: float,
    log_automated_research: float,
    parameters: Parameters,
) -> tuple[float, float]:
    theta = parameters.theta
    rho = (theta - 1.0) / theta
    log_human_term = math.log1p(-parameters.nu) + rho * log_human_research
    log_machine_term = math.log(parameters.nu) + rho * log_automated_research
    log_sum = float(np.logaddexp(log_human_term, log_machine_term))
    log_effective_research = log_sum / rho
    automated_share = math.exp(log_machine_term - log_sum)
    return log_effective_research, automated_share


def static_block(
    log_capital: float,
    log_capability: float,
    log_population: float,
    parameters: Parameters,
) -> dict[str, float]:
    """Solve labor allocation and monopoly deployment conditional on the states."""

    def labor_residual(logit_human_share: float) -> float:
        human_share = min(max(logistic(logit_human_share), 1e-12), 1.0 - 1e-12)
        log_human_research = log_population + math.log(human_share)
        log_production_labor = log_population + math.log1p(-human_share)
        log_ai_ratio = monopoly_ai_ratio(
            log_capital, log_production_labor, log_capability, parameters
        )
        production = production_from_ratio(
            log_capital, log_production_labor, log_ai_ratio, parameters
        )
        log_output = production["log_output"]
        ai_share = production["ai_share"]
        log_wage = (
            math.log1p(-parameters.alpha)
            + math.log1p(-ai_share)
            + log_output
            - log_production_labor
        )
        log_automated_research = (
            math.log(parameters.research_compute_share)
            + log_output
            - math.log(parameters.xi)
        )
        log_actual_ratio = log_human_research - log_automated_research
        log_target_ratio = parameters.theta * (
            math.log1p(-parameters.nu)
            - math.log(parameters.nu)
            + math.log(parameters.xi)
            - log_wage
        )
        return log_actual_ratio - log_target_ratio

    lower, upper = -32.0, 12.0
    logit_human_share = bisect_root(labor_residual, lower, upper)
    human_share = min(max(logistic(logit_human_share), 1e-12), 1.0 - 1e-12)
    log_human_research = log_population + math.log(human_share)
    log_production_labor = log_population + math.log1p(-human_share)
    log_ai_ratio = monopoly_ai_ratio(
        log_capital, log_production_labor, log_capability, parameters
    )
    production = production_from_ratio(
        log_capital, log_production_labor, log_ai_ratio, parameters
    )
    log_output = production["log_output"]
    ai_share = production["ai_share"]
    log_ai_services = log_production_labor + log_ai_ratio
    log_wage = (
        math.log1p(-parameters.alpha)
        + math.log1p(-ai_share)
        + log_output
        - log_production_labor
    )
    log_automated_research = (
        math.log(parameters.research_compute_share)
        + log_output
        - math.log(parameters.xi)
    )
    log_effective_research, automated_research_share = research_aggregator(
        log_human_research, log_automated_research, parameters
    )
    inverse_demand_elasticity = (
        (1.0 - ai_share) / parameters.sigma + parameters.alpha * ai_share
    )
    log_inference_compute = log_ai_services - log_capability
    inference_share = math.exp(
        math.log(parameters.xi) + log_inference_compute - log_output
    )
    consumption_share = (
        1.0
        - parameters.investment_share
        - parameters.research_compute_share
        - inference_share
    )
    log_price = (
        math.log1p(-parameters.alpha)
        + math.log(ai_share)
        + log_output
        - log_ai_services
    )
    monopoly_foc_log_error = (
        log_price
        + math.log(1.0 - inverse_demand_elasticity)
        - math.log(parameters.xi)
        + log_capability
    )
    research_mix_log_error = (
        log_human_research
        - log_automated_research
        - parameters.theta
        * (
            math.log1p(-parameters.nu)
            - math.log(parameters.nu)
            + math.log(parameters.xi)
            - log_wage
        )
    )
    automated_cost_share = 1.0 / (
        1.0
        + math.exp(
            log_wage
            + log_human_research
            - math.log(parameters.xi)
            - log_automated_research
        )
    )
    return {
        "log_output": log_output,
        "log_wage": log_wage,
        "log_ai_services": log_ai_services,
        "log_inference_compute": log_inference_compute,
        "log_human_research": log_human_research,
        "log_production_labor": log_production_labor,
        "log_automated_research": log_automated_research,
        "log_effective_research": log_effective_research,
        "human_research_share": human_share,
        "ai_share": ai_share,
        "automated_research_share": automated_research_share,
        "inverse_demand_elasticity": inverse_demand_elasticity,
        "inference_share": inference_share,
        "consumption_share": consumption_share,
        "monopoly_foc_log_error": monopoly_foc_log_error,
        "research_mix_log_error": research_mix_log_error,
        "automation_share_cost_error": automated_research_share
        - automated_cost_share,
    }


def state_growth_rates(
    state: np.ndarray, parameters: Parameters
) -> tuple[np.ndarray, dict[str, float]]:
    log_capital, log_capability, log_population = map(float, state)
    block = static_block(log_capital, log_capability, log_population, parameters)
    capital_growth = (
        parameters.investment_share
        * math.exp(block["log_output"] - log_capital)
        - parameters.delta
    )
    capability_growth = (
        parameters.chi
        * math.exp(
            (parameters.phi - 1.0) * log_capability
            + parameters.eta * block["log_effective_research"]
        )
    )
    rates = np.array([capital_growth, capability_growth, parameters.n], dtype=float)
    block["capital_growth"] = capital_growth
    block["capability_growth"] = capability_growth
    return rates, block


def rk4_step(
    state: np.ndarray,
    step: float,
    parameters: Parameters,
    first_rate: np.ndarray | None = None,
) -> np.ndarray:
    k1 = first_rate
    if k1 is None:
        k1, _ = state_growth_rates(state, parameters)
    k2, _ = state_growth_rates(state + 0.5 * step * k1, parameters)
    k3, _ = state_growth_rates(state + 0.5 * step * k2, parameters)
    k4, _ = state_growth_rates(state + step * k3, parameters)
    return state + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def simulate(
    name: str,
    parameters: Parameters,
    initial_state: tuple[float, float, float],
    horizon: float = 160.0,
    step: float = 0.50,
    acceleration_cutoff: float = 0.50,
) -> list[dict[str, float | str]]:
    state = np.log(np.asarray(initial_state, dtype=float))
    rows: list[dict[str, float | str]] = []
    time = 0.0

    while time <= horizon + 1e-10:
        rates, block = state_growth_rates(state, parameters)
        row: dict[str, float | str] = {
            "scenario": name,
            "time": time,
            "log_capital": float(state[0]),
            "log_capability": float(state[1]),
            "log_population": float(state[2]),
            **block,
        }
        rows.append(row)
        if (
            block["capability_growth"] >= acceleration_cutoff
            or block["capital_growth"] >= acceleration_cutoff
            or max(state) > 100.0
            or block["consumption_share"] <= 0
        ):
            break
        state = rk4_step(state, step, parameters, rates)
        time += step

    times = np.asarray([float(row["time"]) for row in rows])
    log_output = np.asarray([float(row["log_output"]) for row in rows])
    log_wage = np.asarray([float(row["log_wage"]) for row in rows])
    log_production_labor = np.asarray(
        [float(row["log_production_labor"]) for row in rows]
    )
    log_ai_services = np.asarray(
        [float(row["log_ai_services"]) for row in rows]
    )
    log_human_research = np.asarray(
        [float(row["log_human_research"]) for row in rows]
    )
    log_automated_research = np.asarray(
        [float(row["log_automated_research"]) for row in rows]
    )
    log_population = np.asarray(
        [float(row["log_population"]) for row in rows]
    )
    human_research_population_share = np.asarray(
        [float(row["human_research_share"]) for row in rows]
    )
    production_labor_income_share = np.asarray(
        [
            (1.0 - parameters.alpha) * (1.0 - float(row["ai_share"]))
            for row in rows
        ]
    )
    aggregate_labor_income_share = np.asarray(
        [
            production_share / (1.0 - float(row["human_research_share"]))
            for production_share, row in zip(
                production_labor_income_share, rows
            )
        ]
    )
    if len(rows) > 2:
        output_growth = np.gradient(log_output, times)
        wage_growth = np.gradient(log_wage, times)
        production_labor_growth = np.gradient(log_production_labor, times)
        ai_service_growth = np.gradient(log_ai_services, times)
        production_labor_share_growth = np.gradient(
            np.log(production_labor_income_share), times
        )
        aggregate_labor_share_growth = np.gradient(
            np.log(aggregate_labor_income_share), times
        )
        human_research_population_share_growth = np.gradient(
            np.log(human_research_population_share), times
        )
    else:
        output_growth = np.full(len(rows), np.nan)
        wage_growth = np.full(len(rows), np.nan)
        production_labor_growth = np.full(len(rows), np.nan)
        ai_service_growth = np.full(len(rows), np.nan)
        production_labor_share_growth = np.full(len(rows), np.nan)
        aggregate_labor_share_growth = np.full(len(rows), np.nan)
        human_research_population_share_growth = np.full(len(rows), np.nan)
    for index, row in enumerate(rows):
        row["output_growth"] = float(output_growth[index])
        row["output_per_capita_growth"] = float(
            output_growth[index] - parameters.n
        )
        row["wage_growth"] = float(wage_growth[index])
        row["production_labor_growth"] = float(production_labor_growth[index])
        row["ai_service_growth"] = float(ai_service_growth[index])
        row["capital_deepening_wage_contribution"] = parameters.alpha * (
            float(row["capital_growth"]) - float(production_labor_growth[index])
        )
        row["ai_substitution_wage_contribution"] = (
            float(row["ai_share"])
            * (1.0 / parameters.sigma - parameters.alpha)
            * (
                float(ai_service_growth[index])
                - float(production_labor_growth[index])
            )
        )
        row["wage_growth_identity"] = (
            float(row["capital_deepening_wage_contribution"])
            + float(row["ai_substitution_wage_contribution"])
        )
        row["production_labor_income_share"] = float(
            production_labor_income_share[index]
        )
        row["aggregate_labor_income_share"] = float(
            aggregate_labor_income_share[index]
        )
        row["production_labor_share_growth"] = float(
            production_labor_share_growth[index]
        )
        row["aggregate_labor_share_growth"] = float(
            aggregate_labor_share_growth[index]
        )
        row["ai_displacement_share_contribution"] = (
            -(1.0 - 1.0 / parameters.sigma)
            * float(row["ai_share"])
            * (
                float(ai_service_growth[index])
                - float(production_labor_growth[index])
            )
        )
        row["research_reallocation_share_contribution"] = (
            parameters.n - float(production_labor_growth[index])
        )
        row["production_labor_share_growth_identity"] = float(
            row["ai_displacement_share_contribution"]
        )
        row["aggregate_labor_share_growth_identity"] = (
            float(row["ai_displacement_share_contribution"])
            + float(row["research_reallocation_share_contribution"])
        )
        row["log_output_per_capita"] = (
            float(row["log_output"]) - float(row["log_population"])
        )
        row["log_capital_per_production_worker"] = (
            float(row["log_capital"]) - float(row["log_production_labor"])
        )
        row["log_ai_services_per_production_worker"] = (
            float(row["log_ai_services"]) - float(row["log_production_labor"])
        )
        row["log_human_to_automated_research_ratio"] = float(
            log_human_research[index] - log_automated_research[index]
        )
        row["log_automated_research_per_capita"] = float(
            log_automated_research[index] - log_population[index]
        )
        row["human_research_population_share_growth"] = float(
            human_research_population_share_growth[index]
        )
        row["human_research_population_share_growth_identity"] = float(
            output_growth[index]
            - parameters.n
            - parameters.theta * wage_growth[index]
        )
    return rows


def analytical_calibration(parameters: Parameters) -> tuple[Parameters, dict[str, float]]:
    beta = (1.0 - parameters.alpha) * parameters.omega
    upsilon = beta / (1.0 - parameters.alpha - beta)
    denominator = 1.0 - parameters.phi - parameters.eta * upsilon
    if denominator <= 0:
        raise ValueError("The baseline calibration must have a stable CD growth path.")
    capability_growth = parameters.eta * parameters.n / denominator
    per_capita_growth = upsilon * capability_growth
    research_share = (
        beta**2
        * parameters.eta
        * capability_growth
        / (
            parameters.discount
            - parameters.n
            + (1.0 - parameters.phi) * capability_growth
        )
    )
    investment_share = (
        parameters.alpha
        * (parameters.n + per_capita_growth + parameters.delta)
        / (parameters.discount + parameters.delta + per_capita_growth)
    )
    calibrated = replace(
        parameters,
        investment_share=investment_share,
        research_compute_share=research_share,
    )
    return calibrated, {
        "beta": beta,
        "upsilon": upsilon,
        "stability_denominator": denominator,
        "capability_growth": capability_growth,
        "per_capita_growth": per_capita_growth,
        "research_compute_share": research_share,
        "investment_share": investment_share,
        "capital_output_ratio": parameters.alpha
        / (parameters.discount + parameters.delta + per_capita_growth),
    }


def calibrate_initial_capital(
    parameters: Parameters,
    target_capital_output_ratio: float,
    initial_capability: float = 1.0,
    initial_population: float = 1.0,
) -> float:
    log_capability = math.log(initial_capability)
    log_population = math.log(initial_population)

    def residual(log_capital: float) -> float:
        block = static_block(
            log_capital, log_capability, log_population, parameters
        )
        return log_capital - block["log_output"] - math.log(
            target_capital_output_ratio
        )

    lower, upper = bracket_root(residual, np.linspace(-20.0, 20.0, 161))
    log_capital = lower if lower == upper else bisect_root(residual, lower, upper)
    return math.exp(log_capital)


def calibrate_research_productivity(
    parameters: Parameters,
    initial_state: tuple[float, float, float],
    target_capability_growth: float,
) -> Parameters:
    log_capital, log_capability, log_population = map(math.log, initial_state)
    block = static_block(
        log_capital, log_capability, log_population, parameters
    )
    log_chi = (
        math.log(target_capability_growth)
        - (parameters.phi - 1.0) * log_capability
        - parameters.eta * block["log_effective_research"]
    )
    return replace(parameters, chi=math.exp(log_chi))


def calibrate_research_weight(
    parameters: Parameters,
    initial_state: tuple[float, float, float],
    target_automated_share: float,
) -> Parameters:
    """Choose nu so theta experiments share the same initial automation share."""

    log_capital, log_capability, log_population = map(math.log, initial_state)

    def residual(logit_weight: float) -> float:
        weight = logistic(logit_weight)
        candidate = replace(parameters, nu=weight)
        block = static_block(
            log_capital, log_capability, log_population, candidate
        )
        return block["automated_research_share"] - target_automated_share

    lower, upper = bracket_root(residual, np.linspace(-8.0, 8.0, 129))
    logit_weight = (
        lower if lower == upper else bisect_root(residual, lower, upper)
    )
    return replace(parameters, nu=logistic(logit_weight))


COLORS = {
    "blue": "#205493",
    "gold": "#C69214",
    "orange": "#D2601A",
    "olive": "#667A2C",
    "pink": "#A44870",
    "ink": "#22272E",
    "muted": "#66717E",
    "grid": "#D9DEE5",
    "light": "#F5F7FA",
}


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def nice_ticks(lower: float, upper: float, count: int = 5) -> list[float]:
    if not math.isfinite(lower) or not math.isfinite(upper):
        return [0.0, 1.0]
    if abs(upper - lower) < 1e-12:
        return [lower]
    raw_step = (upper - lower) / max(count - 1, 1)
    magnitude = 10.0 ** math.floor(math.log10(abs(raw_step)))
    normalized = raw_step / magnitude
    if normalized <= 1.0:
        step = 1.0 * magnitude
    elif normalized <= 2.0:
        step = 2.0 * magnitude
    elif normalized <= 5.0:
        step = 5.0 * magnitude
    else:
        step = 10.0 * magnitude
    start = math.floor(lower / step) * step
    end = math.ceil(upper / step) * step
    ticks = []
    value = start
    while value <= end + 0.5 * step and len(ticks) < 20:
        if value >= lower - 1e-12 and value <= upper + 1e-12:
            ticks.append(value)
        value += step
    return ticks


def draw_multiplot(
    output_path: Path,
    title: str,
    subtitle: str,
    panels: list[dict],
    series: dict[str, list[dict[str, float | str]]],
    labels: dict[str, str],
    palette: dict[str, str],
    markers: dict[str, str] | None = None,
) -> None:
    width, height = 2400, 1600
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(48, bold=True)
    subtitle_font = load_font(28)
    panel_title_font = load_font(30, bold=True)
    axis_font = load_font(23)
    legend_font = load_font(24)

    draw.text((120, 65), title, fill=COLORS["ink"], font=title_font)
    draw.text((120, 130), subtitle, fill=COLORS["muted"], font=subtitle_font)

    panel_boxes = [
        (120, 245, 1150, 850),
        (1270, 245, 2300, 850),
        (120, 930, 1150, 1535),
        (1270, 930, 2300, 1535),
    ]
    for panel, box in zip(panels, panel_boxes):
        left, top, right, bottom = box
        plot_left, plot_top = left + 120, top + 75
        plot_right, plot_bottom = right - 35, bottom - 85
        draw.text(
            (left, top),
            panel["title"],
            fill=COLORS["ink"],
            font=panel_title_font,
        )

        all_x: list[float] = []
        all_y: list[float] = []
        transformed: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for key, rows in series.items():
            x_values = np.asarray([float(row["time"]) for row in rows])
            y_values = np.asarray([float(row[panel["field"]]) for row in rows])
            if panel.get("transform") is not None:
                y_values = panel["transform"](rows, y_values)
            valid = np.isfinite(x_values) & np.isfinite(y_values)
            x_values, y_values = x_values[valid], y_values[valid]
            transformed[key] = (x_values, y_values)
            all_x.extend(x_values.tolist())
            all_y.extend(y_values.tolist())

        x_min, x_max = panel.get("xlim", (min(all_x), max(all_x)))
        if panel.get("ylim") is not None:
            y_min, y_max = panel["ylim"]
        else:
            y_min, y_max = min(all_y), max(all_y)
            padding = 0.08 * max(y_max - y_min, 1e-8)
            y_min, y_max = y_min - padding, y_max + padding

        for tick in nice_ticks(y_min, y_max, 5):
            y_pixel = plot_bottom - (tick - y_min) / (y_max - y_min) * (
                plot_bottom - plot_top
            )
            draw.line(
                (plot_left, y_pixel, plot_right, y_pixel),
                fill=COLORS["grid"],
                width=2,
            )
            label = panel.get("format", lambda value: f"{value:.2f}")(tick)
            bbox = draw.textbbox((0, 0), label, font=axis_font)
            draw.text(
                (plot_left - 15 - (bbox[2] - bbox[0]), y_pixel - 12),
                label,
                fill=COLORS["muted"],
                font=axis_font,
            )

        for tick in nice_ticks(x_min, x_max, 5):
            x_pixel = plot_left + (tick - x_min) / (x_max - x_min) * (
                plot_right - plot_left
            )
            draw.line(
                (x_pixel, plot_bottom, x_pixel, plot_bottom + 8),
                fill=COLORS["ink"],
                width=2,
            )
            label = f"{tick:.0f}"
            bbox = draw.textbbox((0, 0), label, font=axis_font)
            draw.text(
                (x_pixel - (bbox[2] - bbox[0]) / 2, plot_bottom + 14),
                label,
                fill=COLORS["muted"],
                font=axis_font,
            )

        draw.line(
            (plot_left, plot_top, plot_left, plot_bottom),
            fill=COLORS["ink"],
            width=3,
        )
        draw.line(
            (plot_left, plot_bottom, plot_right, plot_bottom),
            fill=COLORS["ink"],
            width=3,
        )
        draw.text(
            ((plot_left + plot_right) / 2 - 35, plot_bottom + 50),
            "Years",
            fill=COLORS["muted"],
            font=axis_font,
        )

        for key, (x_values, y_values) in transformed.items():
            points = []
            for x_value, y_value in zip(x_values, y_values):
                x_pixel = plot_left + (x_value - x_min) / (x_max - x_min) * (
                    plot_right - plot_left
                )
                y_clipped = min(max(y_value, y_min), y_max)
                y_pixel = plot_bottom - (y_clipped - y_min) / (y_max - y_min) * (
                    plot_bottom - plot_top
                )
                points.append((x_pixel, y_pixel))
            if len(points) >= 2:
                draw.line(points, fill=palette[key], width=6, joint="curve")
            if markers is not None and points:
                marker_step = max(1, len(points) // 9)
                marker_points = points[::marker_step]
                if marker_points[-1] != points[-1]:
                    marker_points.append(points[-1])
                for x_pixel, y_pixel in marker_points:
                    draw_marker(
                        draw,
                        x_pixel,
                        y_pixel,
                        palette[key],
                        markers[key],
                        radius=7,
                    )

    legend_x, legend_y = 130, 190
    for key in series:
        draw.line(
            (legend_x, legend_y + 13, legend_x + 48, legend_y + 13),
            fill=palette[key],
            width=6,
        )
        if markers is not None:
            draw_marker(
                draw,
                legend_x + 24,
                legend_y + 13,
                palette[key],
                markers[key],
                radius=7,
            )
        draw.text(
            (legend_x + 62, legend_y),
            labels[key],
            fill=COLORS["ink"],
            font=legend_font,
        )
        legend_x += 62 + draw.textlength(labels[key], font=legend_font) + 55

    image.save(output_path, dpi=(220, 220))


def blend_color(start: str, end: str, weight: float) -> tuple[int, int, int]:
    weight = min(max(weight, 0.0), 1.0)
    start_rgb = tuple(int(start[index : index + 2], 16) for index in (1, 3, 5))
    end_rgb = tuple(int(end[index : index + 2], 16) for index in (1, 3, 5))
    return tuple(
        round(start_value + weight * (end_value - start_value))
        for start_value, end_value in zip(start_rgb, end_rgb)
    )


def draw_marker(
    draw: ImageDraw.ImageDraw,
    x_value: float,
    y_value: float,
    color: str,
    shape: str,
    radius: int = 8,
) -> None:
    if shape == "square":
        draw.rectangle(
            (
                x_value - radius,
                y_value - radius,
                x_value + radius,
                y_value + radius,
            ),
            fill=color,
            outline="white",
            width=2,
        )
    elif shape == "triangle":
        draw.polygon(
            [
                (x_value, y_value - radius - 2),
                (x_value - radius, y_value + radius),
                (x_value + radius, y_value + radius),
            ],
            fill=color,
            outline="white",
        )
    else:
        draw.ellipse(
            (
                x_value - radius,
                y_value - radius,
                x_value + radius,
                y_value + radius,
            ),
            fill=color,
            outline="white",
            width=2,
        )


def draw_wage_frontier_figure(
    output_path: Path,
    grid_rows: list[dict[str, float | str]],
    sigma_rows: list[dict[str, float | str]],
) -> None:
    width, height = 2400, 1150
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(48, bold=True)
    subtitle_font = load_font(28)
    panel_title_font = load_font(30, bold=True)
    axis_font = load_font(23)
    cell_font = load_font(20, bold=True)
    legend_font = load_font(23)

    draw.text(
        (120, 60),
        "Numerical frontier for real-wage declines",
        fill=COLORS["ink"],
        font=title_font,
    )
    draw.text(
        (120, 128),
        "Minimum percentage change relative to date zero over the first 80 years",
        fill=COLORS["muted"],
        font=subtitle_font,
    )

    left_box = (120, 235, 1150, 1080)
    right_box = (1270, 235, 2300, 1080)

    # Heatmap: population growth by investment share at sigma = 4.
    left, top, right, bottom = left_box
    draw.text(
        (left, top),
        "Population growth and investment (sigma = 4)",
        fill=COLORS["ink"],
        font=panel_title_font,
    )
    plot_left, plot_top = left + 155, top + 85
    plot_right, plot_bottom = right - 35, bottom - 120
    investment_values = sorted(
        {float(row["investment_share"]) for row in grid_rows}
    )
    population_values = sorted(
        {float(row["population_growth"]) for row in grid_rows}
    )
    value_lookup = {
        (float(row["investment_share"]), float(row["population_growth"])): float(
            row["minimum_wage_change_pct"]
        )
        for row in grid_rows
    }
    minimum_value = min(value_lookup.values())
    cell_width = (plot_right - plot_left) / len(investment_values)
    cell_height = (plot_bottom - plot_top) / len(population_values)

    for y_index, population_growth in enumerate(reversed(population_values)):
        for x_index, investment_share in enumerate(investment_values):
            value = value_lookup[(investment_share, population_growth)]
            intensity = 0.0 if minimum_value == 0 else value / minimum_value
            fill = blend_color("#FFF8EF", COLORS["orange"], intensity)
            x0 = plot_left + x_index * cell_width
            y0 = plot_top + y_index * cell_height
            x1, y1 = x0 + cell_width, y0 + cell_height
            draw.rectangle((x0, y0, x1, y1), fill=fill, outline="white", width=3)
            label = "0%" if abs(value) < 0.5 else f"{value:.0f}%"
            label_color = "white" if intensity > 0.58 else COLORS["ink"]
            bbox = draw.textbbox((0, 0), label, font=cell_font)
            draw.text(
                (
                    (x0 + x1 - (bbox[2] - bbox[0])) / 2,
                    (y0 + y1 - (bbox[3] - bbox[1])) / 2 - 2,
                ),
                label,
                fill=label_color,
                font=cell_font,
            )

    for x_index, investment_share in enumerate(investment_values):
        x_pixel = plot_left + (x_index + 0.5) * cell_width
        label = f"{100 * investment_share:.1f}%"
        bbox = draw.textbbox((0, 0), label, font=axis_font)
        draw.text(
            (x_pixel - (bbox[2] - bbox[0]) / 2, plot_bottom + 14),
            label,
            fill=COLORS["muted"],
            font=axis_font,
        )
    for y_index, population_growth in enumerate(reversed(population_values)):
        y_pixel = plot_top + (y_index + 0.5) * cell_height
        label = f"{100 * population_growth:.1f}%"
        bbox = draw.textbbox((0, 0), label, font=axis_font)
        draw.text(
            (plot_left - 15 - (bbox[2] - bbox[0]), y_pixel - 12),
            label,
            fill=COLORS["muted"],
            font=axis_font,
        )
    draw.text(
        ((plot_left + plot_right) / 2 - 95, plot_bottom + 62),
        "Investment share",
        fill=COLORS["muted"],
        font=axis_font,
    )
    # Line chart: production elasticity at selected investment shares.
    left, top, right, bottom = right_box
    draw.text(
        (left, top),
        "Production elasticity at n = 3%",
        fill=COLORS["ink"],
        font=panel_title_font,
    )
    plot_left, plot_top = left + 120, top + 85
    plot_right, plot_bottom = right - 35, bottom - 120
    sigma_values = sorted({float(row["sigma"]) for row in sigma_rows})
    grouped_investment = sorted(
        {float(row["investment_share"]) for row in sigma_rows}
    )
    y_values = [float(row["minimum_wage_change_pct"]) for row in sigma_rows]
    y_min = 5.0 * math.floor(min(y_values) / 5.0)
    y_max = 2.0
    x_min, x_max = min(sigma_values), max(sigma_values)

    for tick in nice_ticks(y_min, y_max, 6):
        y_pixel = plot_bottom - (tick - y_min) / (y_max - y_min) * (
            plot_bottom - plot_top
        )
        draw.line(
            (plot_left, y_pixel, plot_right, y_pixel),
            fill=COLORS["grid"],
            width=2,
        )
        label = f"{tick:.0f}%"
        bbox = draw.textbbox((0, 0), label, font=axis_font)
        draw.text(
            (plot_left - 15 - (bbox[2] - bbox[0]), y_pixel - 12),
            label,
            fill=COLORS["muted"],
            font=axis_font,
        )
    zero_y = plot_bottom - (0.0 - y_min) / (y_max - y_min) * (
        plot_bottom - plot_top
    )
    draw.line(
        (plot_left, zero_y, plot_right, zero_y),
        fill=COLORS["ink"],
        width=3,
    )
    for sigma in sigma_values:
        x_pixel = plot_left + (sigma - x_min) / (x_max - x_min) * (
            plot_right - plot_left
        )
        draw.line(
            (x_pixel, plot_bottom, x_pixel, plot_bottom + 8),
            fill=COLORS["ink"],
            width=2,
        )
        label = f"{sigma:g}"
        bbox = draw.textbbox((0, 0), label, font=axis_font)
        draw.text(
            (x_pixel - (bbox[2] - bbox[0]) / 2, plot_bottom + 14),
            label,
            fill=COLORS["muted"],
            font=axis_font,
        )
    draw.line(
        (plot_left, plot_top, plot_left, plot_bottom),
        fill=COLORS["ink"],
        width=3,
    )
    draw.line(
        (plot_left, plot_bottom, plot_right, plot_bottom),
        fill=COLORS["ink"],
        width=3,
    )
    draw.text(
        ((plot_left + plot_right) / 2 - 80, plot_bottom + 62),
        "Production sigma",
        fill=COLORS["muted"],
        font=axis_font,
    )

    palette = [COLORS["blue"], COLORS["gold"], COLORS["orange"]]
    shapes = ["circle", "square", "triangle"]
    for series_index, investment_share in enumerate(grouped_investment):
        rows = sorted(
            (
                row
                for row in sigma_rows
                if abs(float(row["investment_share"]) - investment_share) < 1e-10
            ),
            key=lambda row: float(row["sigma"]),
        )
        points = []
        for row in rows:
            sigma = float(row["sigma"])
            value = float(row["minimum_wage_change_pct"])
            x_pixel = plot_left + (sigma - x_min) / (x_max - x_min) * (
                plot_right - plot_left
            )
            y_pixel = plot_bottom - (value - y_min) / (y_max - y_min) * (
                plot_bottom - plot_top
            )
            points.append((x_pixel, y_pixel))
        draw.line(points, fill=palette[series_index], width=6, joint="curve")
        for x_pixel, y_pixel in points:
            draw_marker(
                draw,
                x_pixel,
                y_pixel,
                palette[series_index],
                shapes[series_index],
            )

    legend_x, legend_y = left + 100, top + 45
    for series_index, investment_share in enumerate(grouped_investment):
        draw_marker(
            draw,
            legend_x,
            legend_y + 10,
            palette[series_index],
            shapes[series_index],
            radius=7,
        )
        label = f"sI = {100 * investment_share:.1f}%"
        draw.text(
            (legend_x + 18, legend_y - 2),
            label,
            fill=COLORS["ink"],
            font=legend_font,
        )
        legend_x += 170 + draw.textlength(label, font=legend_font)

    image.save(output_path, dpi=(220, 220))


def index_transform(
    rows: list[dict[str, float | str]], values: np.ndarray
) -> np.ndarray:
    return np.exp(np.clip(values - values[0], -30.0, 30.0))


def log10_index_transform(
    rows: list[dict[str, float | str]], values: np.ndarray
) -> np.ndarray:
    return (values - values[0]) / math.log(10.0)


def log10_level_transform(
    rows: list[dict[str, float | str]], values: np.ndarray
) -> np.ndarray:
    del rows
    return values / math.log(10.0)


def percent_transform(
    rows: list[dict[str, float | str]], values: np.ndarray
) -> np.ndarray:
    return 100.0 * values


def final_average(rows: list[dict[str, float | str]], field: str, years: float = 10.0) -> float:
    final_time = float(rows[-1]["time"])
    values = [
        float(row[field])
        for row in rows
        if float(row["time"]) >= final_time - years and math.isfinite(float(row[field]))
    ]
    return float(np.mean(values))


def write_rows(path: Path, rows: list[dict[str, float | str]]) -> None:
    if not rows:
        return
    fieldnames = list(
        dict.fromkeys(key for row in rows for key in row.keys())
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    FIGURE_DIR.mkdir(exist_ok=True)
    RESULT_DIR.mkdir(exist_ok=True)

    baseline, analytical = analytical_calibration(Parameters())
    initial_capital = calibrate_initial_capital(
        baseline, analytical["capital_output_ratio"]
    )
    initial_state = (initial_capital, 1.0, 1.0)
    baseline = calibrate_research_productivity(
        baseline, initial_state, analytical["capability_growth"]
    )

    regimes: dict[str, Parameters] = {}
    for key, sigma in [
        ("complements", 0.75),
        ("cobb_douglas", 1.00),
        ("substitutes", 2.00),
        ("strong_substitutes", 4.00),
    ]:
        candidate = replace(baseline, sigma=sigma)
        regimes[key] = calibrate_research_productivity(
            candidate, initial_state, analytical["capability_growth"]
        )

    high_feedback = replace(
        baseline, sigma=1.0, phi=0.86, eta=0.62
    )
    high_feedback = calibrate_research_productivity(
        high_feedback, initial_state, analytical["capability_growth"]
    )

    regime_rows: dict[str, list[dict[str, float | str]]] = {}
    for key, parameters in regimes.items():
        regime_rows[key] = simulate(key, parameters, initial_state)
    feedback_rows = simulate(
        "high_feedback", high_feedback, initial_state, horizon=600.0, step=2.0
    )

    theta_rows: dict[str, list[dict[str, float | str]]] = {}
    initial_baseline_block = static_block(
        math.log(initial_state[0]),
        math.log(initial_state[1]),
        math.log(initial_state[2]),
        baseline,
    )
    target_automated_share = initial_baseline_block["automated_research_share"]
    for key, theta in [("theta_low", 1.25), ("theta_base", 2.0), ("theta_high", 4.0)]:
        candidate = replace(baseline, theta=theta)
        candidate = calibrate_research_weight(
            candidate, initial_state, target_automated_share
        )
        candidate = calibrate_research_productivity(
            candidate, initial_state, analytical["capability_growth"]
        )
        theta_rows[key] = simulate(
            key, candidate, initial_state, horizon=400.0, step=2.0
        )

    # Research-industry employment experiments. These paths isolate the
    # distinction between automation intensity, H/M, and the scale of human
    # research employment, H/N, as production substitution changes.
    research_labor_parameters: dict[str, Parameters] = {}
    research_labor_rows: dict[str, list[dict[str, float | str]]] = {}
    for key, sigma in [
        ("research_sigma_cd", 1.0),
        ("research_sigma_substitutes", 2.0),
        ("research_sigma_high", 2.25),
    ]:
        candidate = replace(baseline, sigma=sigma)
        candidate = calibrate_research_productivity(
            candidate, initial_state, analytical["capability_growth"]
        )
        research_labor_parameters[key] = candidate
        research_labor_rows[key] = simulate(
            key,
            candidate,
            initial_state,
            horizon=180.0,
            step=1.0,
        )

    initial_rows: dict[str, list[dict[str, float | str]]] = {}
    for key, capability in [("a_low", 0.50), ("a_base", 1.00), ("a_high", 2.00)]:
        initial_rows[key] = simulate(
            key,
            baseline,
            (initial_capital, capability, 1.0),
            horizon=800.0,
            step=2.0,
        )
    common_initial_output = float(
        initial_rows["a_base"][0]["log_output_per_capita"]
    )
    for rows in initial_rows.values():
        for row in rows:
            row["log10_output_per_capita_common_index"] = (
                float(row["log_output_per_capita"]) - common_initial_output
            ) / math.log(10.0)

    # Wage-decline experiments. Population growth remains below the discount rate.
    wage_case_specs = {
        "wage_reference": {
            "n": 0.012,
            "sigma": 4.0,
            "investment_share": baseline.investment_share,
        },
        "wage_high_population": {
            "n": 0.030,
            "sigma": 4.0,
            "investment_share": baseline.investment_share,
        },
        "wage_high_substitution": {
            "n": 0.030,
            "sigma": 10.0,
            "investment_share": baseline.investment_share,
        },
        "wage_low_investment": {
            "n": 0.030,
            "sigma": 10.0,
            "investment_share": 0.150,
        },
    }
    wage_parameters: dict[str, Parameters] = {}
    wage_paths: dict[str, list[dict[str, float | str]]] = {}
    for key, specification in wage_case_specs.items():
        candidate = replace(
            baseline,
            n=float(specification["n"]),
            sigma=float(specification["sigma"]),
            investment_share=float(specification["investment_share"]),
        )
        candidate = calibrate_research_productivity(
            candidate, initial_state, analytical["capability_growth"]
        )
        wage_parameters[key] = candidate
        wage_paths[key] = simulate(
            key,
            candidate,
            initial_state,
            horizon=80.0,
            step=0.5,
            acceleration_cutoff=0.80,
        )

    wage_summary_rows: list[dict[str, float | str]] = []
    for key, rows in wage_paths.items():
        parameters = wage_parameters[key]
        initial_log_wage = float(rows[0]["log_wage"])
        wage_indices = [
            math.exp(float(row["log_wage"]) - initial_log_wage)
            for row in rows
        ]
        minimum_index = min(wage_indices)
        minimum_index_position = wage_indices.index(minimum_index)
        final_row = rows[-1]
        negative_growth_times = [
            float(row["time"])
            for row in rows
            if float(row["wage_growth"]) < 0.0
        ]
        wage_summary_rows.append(
            {
                "scenario": key,
                "population_growth": parameters.n,
                "sigma": parameters.sigma,
                "investment_share": parameters.investment_share,
                "minimum_wage_index": minimum_index,
                "minimum_wage_change_pct": 100.0 * (minimum_index - 1.0),
                "year_of_minimum_wage": float(
                    rows[minimum_index_position]["time"]
                ),
                "minimum_wage_growth_pct": 100.0
                * min(float(row["wage_growth"]) for row in rows),
                "first_negative_wage_growth_year": (
                    negative_growth_times[0] if negative_growth_times else ""
                ),
                "final_wage_index": wage_indices[-1],
                "final_wage_growth_pct": 100.0
                * float(final_row["wage_growth"]),
                "final_capital_deepening_contribution_pct": 100.0
                * float(final_row["capital_deepening_wage_contribution"]),
                "final_ai_substitution_contribution_pct": 100.0
                * float(final_row["ai_substitution_wage_contribution"]),
                "final_ai_share_pct": 100.0 * float(final_row["ai_share"]),
                "minimum_consumption_share_pct": 100.0
                * min(float(row["consumption_share"]) for row in rows),
            }
        )
    write_rows(RESULT_DIR / "wage_sensitivity_paths.csv", [
        row for rows in wage_paths.values() for row in rows
    ])
    write_rows(RESULT_DIR / "wage_sensitivity_summary.csv", wage_summary_rows)

    wage_grid_rows: list[dict[str, float | str]] = []
    population_grid = (0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.038)
    investment_grid = (
        0.100,
        0.125,
        0.150,
        0.175,
        0.200,
        0.225,
        baseline.investment_share,
        0.250,
        0.275,
    )
    sigma_four = replace(baseline, sigma=4.0)
    sigma_four = calibrate_research_productivity(
        sigma_four, initial_state, analytical["capability_growth"]
    )
    for population_growth in population_grid:
        for investment_share in investment_grid:
            candidate = replace(
                sigma_four,
                n=population_growth,
                investment_share=investment_share,
            )
            rows = simulate(
                "wage_grid",
                candidate,
                initial_state,
                horizon=80.0,
                step=2.0,
                acceleration_cutoff=0.80,
            )
            initial_log_wage = float(rows[0]["log_wage"])
            minimum_change = min(
                math.exp(float(row["log_wage"]) - initial_log_wage) - 1.0
                for row in rows
            )
            wage_grid_rows.append(
                {
                    "population_growth": population_growth,
                    "sigma": 4.0,
                    "investment_share": investment_share,
                    "minimum_wage_change_pct": 100.0 * minimum_change,
                    "minimum_wage_growth_pct": 100.0
                    * min(float(row["wage_growth"]) for row in rows),
                }
            )
    write_rows(RESULT_DIR / "wage_decline_grid.csv", wage_grid_rows)

    wage_sigma_rows: list[dict[str, float | str]] = []
    for investment_share in (0.150, 0.200, baseline.investment_share):
        for sigma in (3.1, 4.0, 6.0, 8.0, 10.0):
            candidate = replace(
                baseline,
                n=0.030,
                sigma=sigma,
                investment_share=investment_share,
            )
            candidate = calibrate_research_productivity(
                candidate, initial_state, analytical["capability_growth"]
            )
            rows = simulate(
                "wage_sigma",
                candidate,
                initial_state,
                horizon=80.0,
                step=2.0,
                acceleration_cutoff=0.80,
            )
            initial_log_wage = float(rows[0]["log_wage"])
            minimum_change = min(
                math.exp(float(row["log_wage"]) - initial_log_wage) - 1.0
                for row in rows
            )
            wage_sigma_rows.append(
                {
                    "population_growth": 0.030,
                    "sigma": sigma,
                    "investment_share": investment_share,
                    "minimum_wage_change_pct": 100.0 * minimum_change,
                    "minimum_wage_growth_pct": 100.0
                    * min(float(row["wage_growth"]) for row in rows),
                }
            )
    write_rows(RESULT_DIR / "wage_sigma_frontier.csv", wage_sigma_rows)

    all_rows = [
        row
        for collection in [
            *regime_rows.values(),
            feedback_rows,
            *theta_rows.values(),
            *research_labor_rows.values(),
            *initial_rows.values(),
        ]
        for row in collection
    ]
    write_rows(RESULT_DIR / "transition_paths.csv", all_rows)
    write_rows(
        RESULT_DIR / "research_labor_paths.csv",
        [row for rows in research_labor_rows.values() for row in rows],
    )

    research_labor_summary_rows: list[dict[str, float | str]] = []
    for key, rows in research_labor_rows.items():
        initial_row = rows[0]
        final_row = rows[-1]
        minimum_row = min(
            rows, key=lambda row: float(row["human_research_share"])
        )
        research_labor_summary_rows.append(
            {
                "scenario": key,
                "sigma": research_labor_parameters[key].sigma,
                "last_year": float(final_row["time"]),
                "initial_human_to_automated_ratio": math.exp(
                    float(initial_row["log_human_to_automated_research_ratio"])
                ),
                "final_human_to_automated_ratio": math.exp(
                    float(final_row["log_human_to_automated_research_ratio"])
                ),
                "initial_automated_research_per_capita": math.exp(
                    float(initial_row["log_automated_research_per_capita"])
                ),
                "final_automated_research_per_capita": math.exp(
                    float(final_row["log_automated_research_per_capita"])
                ),
                "initial_human_research_population_pct": 100.0
                * float(initial_row["human_research_share"]),
                "minimum_human_research_population_pct": 100.0
                * float(minimum_row["human_research_share"]),
                "minimum_human_research_population_year": float(
                    minimum_row["time"]
                ),
                "final_human_research_population_pct": 100.0
                * float(final_row["human_research_share"]),
                "final_capability_growth_pct": 100.0
                * float(final_row["capability_growth"]),
            }
        )
    write_rows(
        RESULT_DIR / "research_labor_summary.csv",
        research_labor_summary_rows,
    )

    scenario_parameters = {**regimes, "high_feedback": high_feedback}
    scenario_data = {**regime_rows, "high_feedback": feedback_rows}
    summary_rows: list[dict[str, float | str]] = []
    beta = analytical["beta"]
    upsilon = analytical["upsilon"]
    for key, rows in scenario_data.items():
        parameters = scenario_parameters[key]
        stability_denominator = (
            1.0 - parameters.phi - parameters.eta * upsilon
        )
        summary_rows.append(
            {
                "scenario": key,
                "sigma": parameters.sigma,
                "theta": parameters.theta,
                "phi": parameters.phi,
                "eta": parameters.eta,
                "cd_stability_denominator": stability_denominator,
                "last_year": float(rows[-1]["time"]),
                "initial_capability_growth_pct": 100.0
                * float(rows[0]["capability_growth"]),
                "final_capability_growth_pct": 100.0
                * final_average(rows, "capability_growth"),
                "final_output_per_capita_growth_pct": 100.0
                * final_average(rows, "output_per_capita_growth"),
                "final_wage_growth_pct": 100.0
                * final_average(rows, "wage_growth"),
                "final_ai_production_share_pct": 100.0
                * final_average(rows, "ai_share"),
                "final_automated_research_share_pct": 100.0
                * final_average(rows, "automated_research_share"),
                "final_human_research_population_pct": 100.0
                * final_average(rows, "human_research_share"),
                "final_consumption_share_pct": 100.0
                * final_average(rows, "consumption_share"),
            }
        )
    write_rows(RESULT_DIR / "scenario_summary.csv", summary_rows)

    labor_share_summary_rows: list[dict[str, float | str]] = []
    for key, rows in regime_rows.items():
        initial_row = rows[0]
        final_row = rows[-1]
        initial_production_share = float(
            initial_row["production_labor_income_share"]
        )
        final_production_share = float(
            final_row["production_labor_income_share"]
        )
        initial_aggregate_share = float(
            initial_row["aggregate_labor_income_share"]
        )
        final_aggregate_share = float(
            final_row["aggregate_labor_income_share"]
        )
        labor_share_summary_rows.append(
            {
                "scenario": key,
                "sigma": regimes[key].sigma,
                "last_year": float(final_row["time"]),
                "initial_production_labor_share_pct": 100.0
                * initial_production_share,
                "final_production_labor_share_pct": 100.0
                * final_production_share,
                "production_labor_share_change_pct": 100.0
                * (
                    final_production_share / initial_production_share
                    - 1.0
                ),
                "initial_aggregate_labor_share_pct": 100.0
                * initial_aggregate_share,
                "final_aggregate_labor_share_pct": 100.0
                * final_aggregate_share,
                "aggregate_labor_share_change_pct": 100.0
                * (final_aggregate_share / initial_aggregate_share - 1.0),
                "initial_human_research_population_pct": 100.0
                * float(initial_row["human_research_share"]),
                "final_human_research_population_pct": 100.0
                * float(final_row["human_research_share"]),
                "final_ai_displacement_contribution_pct": 100.0
                * float(final_row["ai_displacement_share_contribution"]),
                "final_research_reallocation_contribution_pct": 100.0
                * float(
                    final_row["research_reallocation_share_contribution"]
                ),
                "final_aggregate_labor_share_growth_pct": 100.0
                * float(final_row["aggregate_labor_share_growth"]),
            }
        )
    write_rows(
        RESULT_DIR / "labor_share_summary.csv",
        labor_share_summary_rows,
    )

    validation_rows = []
    for key, rows in {
        **scenario_data,
        **wage_paths,
        **research_labor_rows,
    }.items():
        validation_rows.append(
            {
                "scenario": key,
                "max_abs_monopoly_foc_log_error": max(
                    abs(float(row["monopoly_foc_log_error"])) for row in rows
                ),
                "max_abs_research_mix_log_error": max(
                    abs(float(row["research_mix_log_error"])) for row in rows
                ),
                "max_abs_automation_share_cost_error": max(
                    abs(float(row["automation_share_cost_error"])) for row in rows
                ),
                "max_abs_wage_growth_identity_error": max(
                    abs(
                        float(row["wage_growth"])
                        - float(row["wage_growth_identity"])
                    )
                    for row in rows
                ),
                "max_abs_production_labor_share_growth_identity_error": max(
                    abs(
                        float(row["production_labor_share_growth"])
                        - float(
                            row[
                                "production_labor_share_growth_identity"
                            ]
                        )
                    )
                    for row in rows
                ),
                "max_abs_aggregate_labor_share_growth_identity_error": max(
                    abs(
                        float(row["aggregate_labor_share_growth"])
                        - float(
                            row[
                                "aggregate_labor_share_growth_identity"
                            ]
                        )
                    )
                    for row in rows
                ),
                "max_abs_human_research_population_growth_identity_error": max(
                    abs(
                        float(row["human_research_population_share_growth"])
                        - float(
                            row[
                                "human_research_population_share_growth_identity"
                            ]
                        )
                    )
                    for row in rows
                ),
                "minimum_consumption_share": min(
                    float(row["consumption_share"]) for row in rows
                ),
            }
        )
    write_rows(RESULT_DIR / "validation.csv", validation_rows)

    calibration_rows = [
        {"parameter": key, "value": value}
        for key, value in {
            **analytical,
            "alpha": baseline.alpha,
            "omega": baseline.omega,
            "n": baseline.n,
            "delta": baseline.delta,
            "discount": baseline.discount,
            "xi": baseline.xi,
            "nu": baseline.nu,
            "theta": baseline.theta,
            "phi": baseline.phi,
            "eta": baseline.eta,
            "chi": baseline.chi,
            "initial_capital": initial_capital,
        }.items()
    ]
    write_rows(RESULT_DIR / "calibration.csv", calibration_rows)

    regime_palette = {
        "complements": COLORS["blue"],
        "cobb_douglas": COLORS["gold"],
        "substitutes": COLORS["orange"],
        "strong_substitutes": COLORS["olive"],
    }
    draw_multiplot(
        FIGURE_DIR / "numerical_production_regimes.png",
        "Numerical transitions across production elasticities",
        "Common initial states, spending rules, and capability growth; monopoly service supply",
        [
            {
                "title": "Output per capita (log10 index)",
                "field": "log_output_per_capita",
                "transform": log10_index_transform,
                "format": lambda value: f"{value:.1f}",
            },
            {
                "title": "Capability growth",
                "field": "capability_growth",
                "transform": percent_transform,
                "format": lambda value: f"{value:.0f}%",
                "ylim": (0.0, 12.0),
            },
            {
                "title": "Real wage (log10 index)",
                "field": "log_wage",
                "transform": log10_index_transform,
                "format": lambda value: f"{value:.1f}",
            },
            {
                "title": "Production labor income share",
                "field": "production_labor_income_share",
                "transform": percent_transform,
                "format": lambda value: f"{value:.0f}%",
                "ylim": (0.0, 70.0),
            },
        ],
        regime_rows,
        {
            "complements": "sigma = 0.75",
            "cobb_douglas": "sigma = 1",
            "substitutes": "sigma = 2",
            "strong_substitutes": "sigma = 4",
        },
        regime_palette,
    )

    draw_multiplot(
        FIGURE_DIR / "numerical_labor_share_paths.png",
        "Production and aggregate labor-income shares",
        "Common initial capability growth; aggregate share includes human researchers",
        [
            {
                "title": "Production labor income share",
                "field": "production_labor_income_share",
                "transform": percent_transform,
                "format": lambda value: f"{value:.0f}%",
                "ylim": (0.0, 75.0),
            },
            {
                "title": "Aggregate labor income share",
                "field": "aggregate_labor_income_share",
                "transform": percent_transform,
                "format": lambda value: f"{value:.0f}%",
                "ylim": (0.0, 75.0),
            },
            {
                "title": "AI expenditure share in service composite",
                "field": "ai_share",
                "transform": percent_transform,
                "format": lambda value: f"{value:.0f}%",
                "ylim": (0.0, 100.0),
            },
            {
                "title": "Human researchers as share of population",
                "field": "human_research_share",
                "transform": percent_transform,
                "format": lambda value: f"{value:.0f}%",
                "ylim": (0.0, 100.0),
            },
        ],
        regime_rows,
        {
            "complements": "sigma = 0.75",
            "cobb_douglas": "sigma = 1",
            "substitutes": "sigma = 2",
            "strong_substitutes": "sigma = 4",
        },
        regime_palette,
    )

    wage_palette = {
        "wage_reference": COLORS["blue"],
        "wage_high_population": COLORS["gold"],
        "wage_high_substitution": COLORS["orange"],
        "wage_low_investment": COLORS["olive"],
    }
    draw_multiplot(
        FIGURE_DIR / "numerical_wage_decline_paths.png",
        "Numerical real-wage transitions",
        "Common initial states and capability growth; rho = 4% and n < rho",
        [
            {
                "title": "Real wage index",
                "field": "log_wage",
                "transform": index_transform,
                "format": lambda value: f"{value:.2f}",
            },
            {
                "title": "Real-wage growth",
                "field": "wage_growth",
                "transform": percent_transform,
                "format": lambda value: f"{value:.1f}%",
            },
            {
                "title": "Capital per production worker (log10 index)",
                "field": "log_capital_per_production_worker",
                "transform": log10_index_transform,
                "format": lambda value: f"{value:.2f}",
            },
            {
                "title": "AI services per production worker (log10 index)",
                "field": "log_ai_services_per_production_worker",
                "transform": log10_index_transform,
                "format": lambda value: f"{value:.1f}",
            },
        ],
        wage_paths,
        {
            "wage_reference": "n=1.2%, sigma=4, sI=23.3%",
            "wage_high_population": "n=3%, sigma=4, sI=23.3%",
            "wage_high_substitution": "n=3%, sigma=10, sI=23.3%",
            "wage_low_investment": "n=3%, sigma=10, sI=15%",
        },
        wage_palette,
    )
    draw_wage_frontier_figure(
        FIGURE_DIR / "numerical_wage_decline_frontier.png",
        wage_grid_rows,
        wage_sigma_rows,
    )

    theta_palette = {
        "theta_low": COLORS["blue"],
        "theta_base": COLORS["gold"],
        "theta_high": COLORS["orange"],
    }
    draw_multiplot(
        FIGURE_DIR / "numerical_research_automation.png",
        "Numerical research-automation transitions",
        "Cobb--Douglas production; common initial automation share and capability growth",
        [
            {
                "title": "Automated share of effective research",
                "field": "automated_research_share",
                "transform": percent_transform,
                "format": lambda value: f"{value:.0f}%",
                "ylim": (0.0, 100.0),
            },
            {
                "title": "Human researchers as share of population",
                "field": "human_research_share",
                "transform": percent_transform,
                "format": lambda value: f"{value:.1f}%",
            },
            {
                "title": "Capability growth",
                "field": "capability_growth",
                "transform": percent_transform,
                "format": lambda value: f"{value:.1f}%",
            },
            {
                "title": "Real wage (log10 index)",
                "field": "log_wage",
                "transform": log10_index_transform,
                "format": lambda value: f"{value:.2f}",
            },
        ],
        theta_rows,
        {
            "theta_low": "theta = 1.25",
            "theta_base": "theta = 2",
            "theta_high": "theta = 4",
        },
        theta_palette,
    )

    research_labor_palette = {
        "research_sigma_cd": COLORS["blue"],
        "research_sigma_substitutes": COLORS["orange"],
        "research_sigma_high": COLORS["olive"],
    }
    research_labor_markers = {
        "research_sigma_cd": "circle",
        "research_sigma_substitutes": "square",
        "research_sigma_high": "triangle",
    }
    draw_multiplot(
        FIGURE_DIR / "numerical_research_labor_dynamics.png",
        "Human employment during AI research automation",
        "theta=2; common initial states and capability growth; fixed investment and research-compute shares",
        [
            {
                "title": "Human-to-automated research ratio, H/M (log scale)",
                "field": "log_human_to_automated_research_ratio",
                "transform": log10_level_transform,
                "format": lambda value: f"{10.0 ** value:.2g}",
            },
            {
                "title": "Automated research per capita, M/N (log scale)",
                "field": "log_automated_research_per_capita",
                "transform": log10_level_transform,
                "format": lambda value: f"{10.0 ** value:.2g}",
            },
            {
                "title": "Human researchers as share of population, H/N",
                "field": "human_research_share",
                "transform": percent_transform,
                "format": lambda value: f"{value:.0f}%",
                "ylim": (0.0, 22.0),
            },
            {
                "title": "Growth of the human-research population share",
                "field": "human_research_population_share_growth",
                "transform": percent_transform,
                "format": lambda value: f"{value:.0f}%",
            },
        ],
        research_labor_rows,
        {
            "research_sigma_cd": "sigma = 1",
            "research_sigma_substitutes": "sigma = 2",
            "research_sigma_high": "sigma = 2.25",
        },
        research_labor_palette,
        markers=research_labor_markers,
    )

    initial_palette = {
        "a_low": COLORS["blue"],
        "a_base": COLORS["gold"],
        "a_high": COLORS["orange"],
    }
    draw_multiplot(
        FIGURE_DIR / "numerical_initial_conditions.png",
        "Transitions from different initial capability stocks",
        "Stable Cobb--Douglas parameterization; common initial capital and population",
        [
            {
                "title": "Capability growth",
                "field": "capability_growth",
                "transform": percent_transform,
                "format": lambda value: f"{value:.1f}%",
            },
            {
                "title": "Output per capita (log10, common index)",
                "field": "log10_output_per_capita_common_index",
                "format": lambda value: f"{value:.2f}",
            },
            {
                "title": "Automated share of effective research",
                "field": "automated_research_share",
                "transform": percent_transform,
                "format": lambda value: f"{value:.0f}%",
                "ylim": (0.0, 100.0),
            },
            {
                "title": "Human researchers as share of population",
                "field": "human_research_share",
                "transform": percent_transform,
                "format": lambda value: f"{value:.1f}%",
            },
        ],
        initial_rows,
        {
            "a_low": "A0 = 0.5",
            "a_base": "A0 = 1",
            "a_high": "A0 = 2",
        },
        initial_palette,
    )

    long_baseline_rows = simulate(
        "stable_feedback", baseline, initial_state, horizon=600.0, step=2.0
    )
    feedback_comparison = {
        "stable_feedback": long_baseline_rows,
        "high_feedback": feedback_rows,
    }
    draw_multiplot(
        FIGURE_DIR / "numerical_growth_feedback.png",
        "Numerical transitions across research-feedback regimes",
        "Cobb--Douglas production; common initial capability growth and allocation shares",
        [
            {
                "title": "Capability growth",
                "field": "capability_growth",
                "transform": percent_transform,
                "format": lambda value: f"{value:.0f}%",
                "ylim": (0.0, 20.0),
            },
            {
                "title": "Output per capita (log10 index)",
                "field": "log_output_per_capita",
                "transform": log10_index_transform,
                "format": lambda value: f"{value:.1f}",
            },
            {
                "title": "Automated share of effective research",
                "field": "automated_research_share",
                "transform": percent_transform,
                "format": lambda value: f"{value:.0f}%",
                "ylim": (0.0, 100.0),
            },
            {
                "title": "Human researchers as share of population",
                "field": "human_research_share",
                "transform": percent_transform,
                "format": lambda value: f"{value:.1f}%",
            },
        ],
        feedback_comparison,
        {
            "stable_feedback": "stable feedback",
            "high_feedback": "high feedback",
        },
        {
            "stable_feedback": COLORS["blue"],
            "high_feedback": COLORS["orange"],
        },
    )

    print("Analytical baseline:")
    for key, value in analytical.items():
        print(f"  {key}: {value:.8f}")
    print(f"  chi: {baseline.chi:.8f}")
    print(f"  initial_capital: {initial_capital:.8f}")
    print("\nScenario summary:")
    for row in summary_rows:
        print(
            f"  {row['scenario']}: T={row['last_year']:.1f}, "
            f"gA={row['final_capability_growth_pct']:.2f}%, "
            f"gy={row['final_output_per_capita_growth_pct']:.2f}%, "
            f"sX={row['final_ai_production_share_pct']:.1f}%, "
            f"sM={row['final_automated_research_share_pct']:.1f}%"
        )


if __name__ == "__main__":
    main()
