"""Free-boundary equilibrium transitions for gross production substitution.

For ``sigma_xl > 1`` and unbounded capability, the model has no finite-rate
terminal balanced-growth path.  This module therefore does not impose one.  It
uses the AI-dominated equilibrium asymptotics to terminate the path at a large
output--capital ratio and treats calendar time to that boundary as an endogenous
free parameter.  Raising the terminal ratio supplies a convergence test toward
the finite-time singularity of the limiting equilibrium.

The paper proves that the asymptotic boundary is internally consistent for
``1 < sigma_xl < 1 / alpha`` and ``sigma_hm > 1``.  The same limiting ratios can
be imposed above ``1 / alpha`` to search for a conditional AI-dominated branch,
but that exercise is an extrapolation rather than a proved terminal condition.
Such paths must therefore be checked ex post for AI dominance, automation,
boundary convergence, interiority, and all equilibrium residuals.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np

import simulate_equilibrium as equilibrium
from scipy.interpolate import CubicSpline


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "numerical"
FIGURE_DIR = ROOT / "figures"


def high_sigma_targets(parameters: equilibrium.Parameters) -> dict[str, float]:
    """Return AI-dominated singular-path ratios.

    Write ``z = Y/K`` and ``kappa = (1-alpha)/alpha``.  Along the
    AI-dominated equilibrium singularity, ``g_A / z``, the consumption share,
    the investment share, the automated-research resource share, and ``q A/K``
    converge to constants.
    """

    if parameters.sigma_xl <= 1.0:
        raise ValueError("The high-sigma boundary requires sigma_xl > 1.")
    if parameters.sigma_hm <= 1.0:
        raise ValueError(
            "The current high-sigma boundary requires sigma_hm > 1."
        )

    alpha = parameters.alpha
    kappa = (1.0 - alpha) / alpha
    denominator = 1.0 - parameters.phi + kappa
    inference_share = (1.0 - alpha) ** 2
    capability_growth_to_z = parameters.eta * alpha / denominator
    investment_share = alpha - kappa * capability_growth_to_z
    research_share = parameters.eta * inference_share / denominator
    consumption_share = (
        1.0 - inference_share - investment_share - research_share
    )
    shadow_capability_to_capital = inference_share / (
        parameters.eta * alpha
    )
    singularity_rate = kappa * capability_growth_to_z
    if min(investment_share, research_share, consumption_share) <= 0.0:
        raise ValueError(
            "The singular-path resource shares are not all positive."
        )
    return {
        "kappa": kappa,
        "denominator": denominator,
        "inference_share": inference_share,
        "capability_growth_to_z": capability_growth_to_z,
        "investment_share": investment_share,
        "research_share": research_share,
        "consumption_share": consumption_share,
        "shadow_capability_to_capital": shadow_capability_to_capital,
        "singularity_rate": singularity_rate,
    }


def asymptotic_output_capital_coefficient(
    parameters: equilibrium.Parameters,
) -> float:
    """Coefficient D in Y/K ~ D A**((1-alpha)/alpha)."""

    alpha = parameters.alpha
    exponent = (1.0 - alpha) / alpha
    log_coefficient = exponent * (
        2.0 * math.log1p(-alpha)
        - math.log(parameters.xi)
        + parameters.sigma_xl
        / (parameters.sigma_xl - 1.0)
        * math.log(parameters.omega_x)
    )
    return math.exp(log_coefficient)


def initial_free_boundary_guess(
    parameters: equilibrium.Parameters,
    initial_state: tuple[float, float, float],
    terminal_output_capital_ratio: float,
    duration_guess: float,
    mesh: np.ndarray,
) -> np.ndarray:
    """Construct a singular-scaling guess for the free-boundary solver."""

    targets = high_sigma_targets(parameters)
    log_initial_capital = math.log(initial_state[0])
    log_initial_capability = math.log(initial_state[1])
    coefficient = asymptotic_output_capital_coefficient(parameters)
    log_terminal_capability = max(
        math.log(initial_state[1]) + 0.5,
        (
            math.log(terminal_output_capital_ratio)
            - math.log(coefficient)
        )
        / targets["kappa"],
    )
    capital_capability_elasticity = (
        targets["investment_share"]
        / targets["capability_growth_to_z"]
    )
    log_terminal_capital = (
        log_initial_capital
        + capital_capability_elasticity
        * (log_terminal_capability - log_initial_capability)
    )

    # Concentrate the change near the terminal boundary, as implied by
    # z(t) ~ 1 / [kappa h (T* - t)], while keeping a smooth initial guess.
    curvature = 0.90
    profile = -np.log1p(-curvature * mesh) / -math.log1p(-curvature)
    log_capability = (
        log_initial_capability
        + (log_terminal_capability - log_initial_capability) * profile
    )
    log_capital = (
        log_initial_capital
        + (log_terminal_capital - log_initial_capital) * profile
    )
    log_shadow = (
        math.log(targets["shadow_capability_to_capital"])
        + log_capital
        - log_capability
    )

    log_consumption = np.empty_like(mesh)
    for index, tau in enumerate(mesh):
        time = duration_guess * float(tau)
        block = equilibrium.equilibrium_static_block(
            float(log_capital[index]),
            float(log_capability[index]),
            parameters.n * time,
            float(log_shadow[index]),
            parameters,
        )
        log_consumption[index] = (
            math.log(targets["consumption_share"])
            + block["log_output"]
        )

    return np.vstack(
        [log_capital, log_capability, log_consumption, log_shadow]
    )


def solve_high_sigma_equilibrium(
    parameters: equilibrium.Parameters,
    initial_state: tuple[float, float, float],
    terminal_output_capital_ratio: float,
    duration_guess: float,
    nodes: int = 301,
    tolerance: float = 3e-5,
    previous_solution: object | None = None,
) -> tuple[object, dict[str, float]]:
    """Solve the free-boundary equilibrium at a finite terminal z=Y/K."""

    targets = high_sigma_targets(parameters)
    mesh = np.linspace(0.0, 1.0, nodes)
    if previous_solution is None:
        guess = initial_free_boundary_guess(
            parameters,
            initial_state,
            terminal_output_capital_ratio,
            duration_guess,
            mesh,
        )
    else:
        if getattr(previous_solution, "normalized_domain", False):
            guess = previous_solution.sol(mesh)
            duration_guess = float(math.exp(previous_solution.p[0]))
        else:
            duration_guess = float(previous_solution.duration)
            guess = previous_solution.sol(mesh * duration_guess)
        terminal = guess[:, -1]
        _, terminal_block = equilibrium.equilibrium_rates(
            duration_guess, terminal, parameters
        )
        old_z = math.exp(
            terminal_block["log_output"] - float(terminal[0])
        )
        extension = max(
            0.0,
            math.log(terminal_output_capital_ratio / old_z),
        )
        if extension > 0.0:
            log_capability_increment = extension / targets["kappa"]
            profile = mesh**4
            guess[1] += log_capability_increment * profile
            guess[0] += (
                targets["investment_share"]
                / targets["capability_growth_to_z"]
                * log_capability_increment
                * profile
            )
            guess[2] += (
                targets["investment_share"]
                / targets["capability_growth_to_z"]
                * log_capability_increment
                + extension
            ) * profile
            guess[3] += (
                targets["investment_share"]
                / targets["capability_growth_to_z"]
                - 1.0
            ) * log_capability_increment * profile
            duration_guess += (
                1.0
                / targets["singularity_rate"]
                * (1.0 / old_z - 1.0 / terminal_output_capital_ratio)
            )

    def ode(
        normalized_time: np.ndarray,
        state: np.ndarray,
        log_duration: np.ndarray,
    ) -> np.ndarray:
        duration = equilibrium.bounded_exp(float(log_duration[0]), upper=10.0)
        values = np.empty_like(state)
        for index, tau in enumerate(normalized_time):
            time = duration * float(tau)
            values[:, index] = duration * equilibrium.equilibrium_rates(
                time,
                state[:, index] + reference(float(tau)),
                parameters,
            )[0] - reference_derivative(float(tau))
        return values

    def boundary(
        left: np.ndarray,
        right: np.ndarray,
        log_duration: np.ndarray,
    ) -> np.ndarray:
        duration = equilibrium.bounded_exp(float(log_duration[0]), upper=10.0)
        raw_left = left + reference(0.0)
        raw_right = right + reference(1.0)
        _, terminal_block = equilibrium.equilibrium_rates(
            duration, raw_right, parameters
        )
        log_terminal_consumption_share = (
            raw_right[2] - terminal_block["log_output"]
        )
        log_terminal_shadow_ratio = (
            raw_right[3] + raw_right[1] - raw_right[0]
        )
        log_terminal_z = terminal_block["log_output"] - raw_right[0]
        return np.asarray(
            [
                raw_left[0] - math.log(initial_state[0]),
                raw_left[1] - math.log(initial_state[1]),
                log_terminal_consumption_share
                - math.log(targets["consumption_share"]),
                log_terminal_shadow_ratio
                - math.log(targets["shadow_capability_to_capital"]),
                log_terminal_z - math.log(terminal_output_capital_ratio),
            ]
        )

    reference_splines = [
        CubicSpline(mesh, guess[index], bc_type="natural")
        for index in range(guess.shape[0])
    ]

    def reference(normalized_time: np.ndarray | float) -> np.ndarray:
        values = np.asarray(
            [spline(normalized_time) for spline in reference_splines]
        )
        return values

    def reference_derivative(
        normalized_time: np.ndarray | float,
    ) -> np.ndarray:
        values = np.asarray(
            [spline(normalized_time, 1) for spline in reference_splines]
        )
        return values

    scaled_guess = np.zeros_like(guess)
    solution = equilibrium.solve_bvp(
        ode,
        boundary,
        mesh,
        scaled_guess,
        p=np.asarray([math.log(duration_guess)]),
        tol=tolerance,
        max_nodes=12000,
        verbose=1,
    )
    scaled_solution = solution.sol

    def raw_solution(normalized_time: np.ndarray | float) -> np.ndarray:
        return scaled_solution(normalized_time) + reference(normalized_time)

    def raw_derivative(normalized_time: np.ndarray | float) -> np.ndarray:
        return (
            scaled_solution(normalized_time, 1)
            + reference_derivative(normalized_time)
        )

    solution.scaled_sol = scaled_solution
    solution.sol = raw_solution
    solution.y = solution.y + reference(solution.x)
    solution.duration = math.exp(float(solution.p[0]))
    solution.terminal_output_capital_ratio = terminal_output_capital_ratio
    solution.normalized_domain = True
    solution.calendar_sol = lambda times: solution.sol(
        np.asarray(times) / solution.duration
    )
    solution.calendar_derivative = lambda times: raw_derivative(
        np.asarray(times) / solution.duration
    ) / solution.duration
    return solution, targets


def solve_high_sigma_shooting(
    parameters: equilibrium.Parameters,
    initial_state: tuple[float, float, float],
    terminal_output_capital_ratio: float,
    duration_guess: float,
    jump_guess: tuple[float, float] = (-0.60, -0.80),
    tolerance: float = 2e-7,
) -> tuple[object, dict[str, float]]:
    """Shoot on initial consumption, capability value, and terminal time."""

    targets = high_sigma_targets(parameters)
    fixed_initial = np.asarray(
        [math.log(initial_state[0]), math.log(initial_state[1])]
    )

    def integrate(unknowns: np.ndarray, dense_output: bool = False) -> object:
        duration = equilibrium.bounded_exp(float(unknowns[2]), upper=9.0)
        initial = np.asarray(
            [fixed_initial[0], fixed_initial[1], unknowns[0], unknowns[1]]
        )
        return equilibrium.solve_ivp(
            lambda time, state: equilibrium.equilibrium_rates(
                float(time), state, parameters
            )[0],
            (0.0, duration),
            initial,
            method="DOP853",
            rtol=2e-7,
            atol=2e-9,
            max_step=max(min(duration / 250.0, 2.0), 0.05),
            dense_output=dense_output,
        )

    evaluations = {"count": 0}

    def residual(unknowns: np.ndarray) -> np.ndarray:
        evaluations["count"] += 1
        try:
            solution = integrate(unknowns)
            if not solution.success or solution.t[-1] <= 0.99 * math.exp(
                float(unknowns[2])
            ):
                return np.asarray([20.0, 20.0, 20.0])
            terminal = solution.y[:, -1]
            duration = float(solution.t[-1])
            _, block = equilibrium.equilibrium_rates(
                duration, terminal, parameters
            )
            return np.asarray(
                [
                    terminal[2]
                    - block["log_output"]
                    - math.log(targets["consumption_share"]),
                    terminal[3]
                    + terminal[1]
                    - terminal[0]
                    - math.log(targets["shadow_capability_to_capital"]),
                    block["log_output"]
                    - terminal[0]
                    - math.log(terminal_output_capital_ratio),
                ]
            )
        except (ArithmeticError, ValueError, RuntimeError):
            return np.asarray([20.0, 20.0, 20.0])

    initial_unknowns = np.asarray(
        [jump_guess[0], jump_guess[1], math.log(duration_guess)]
    )
    root = equilibrium.least_squares(
        residual,
        initial_unknowns,
        bounds=(
            np.asarray([-8.0, -12.0, math.log(1.0)]),
            np.asarray([2.0, 8.0, math.log(6000.0)]),
        ),
        xtol=tolerance,
        ftol=tolerance,
        gtol=tolerance,
        x_scale=np.asarray([1.0, 2.0, 1.0]),
        max_nfev=160,
        verbose=2,
    )
    solution = integrate(root.x, dense_output=True)
    solution.root_result = root
    solution.duration = float(solution.t[-1])
    solution.terminal_output_capital_ratio = terminal_output_capital_ratio
    solution.calendar_sol = solution.sol
    solution.success = bool(
        solution.success
        and root.success
        and np.max(np.abs(residual(root.x))) < 2e-4
    )
    return solution, targets


def solve_high_sigma_fixed_horizon(
    parameters: equilibrium.Parameters,
    initial_state: tuple[float, float, float],
    horizon: float,
    terminal_z_guess: float = 1.0,
    nodes: int = 201,
    tolerance: float = 3e-5,
    previous_solution: object | None = None,
) -> tuple[object, dict[str, float]]:
    """Solve a fixed-horizon BVP using the singular limiting ratios."""

    targets = high_sigma_targets(parameters)
    mesh = np.linspace(0.0, horizon, nodes)
    if previous_solution is None:
        guess = initial_free_boundary_guess(
            parameters,
            initial_state,
            terminal_z_guess,
            horizon,
            mesh / horizon,
        )
    else:
        old_horizon = float(previous_solution.duration)
        old_times = np.minimum(mesh, old_horizon)
        guess = previous_solution.sol(old_times)
        if horizon > old_horizon:
            tail = mesh > old_horizon
            if np.any(tail):
                terminal_state = previous_solution.sol(old_horizon)
                terminal_rates, _ = equilibrium.equilibrium_rates(
                    old_horizon, terminal_state, parameters
                )
                guess[:, tail] = (
                    terminal_state[:, None]
                    + terminal_rates[:, None]
                    * (mesh[tail] - old_horizon)[None, :]
                )

    def ode(times: np.ndarray, states: np.ndarray) -> np.ndarray:
        values = np.empty_like(states)
        for index, time in enumerate(times):
            values[:, index] = equilibrium.equilibrium_rates(
                float(time), states[:, index], parameters
            )[0]
        return values

    def boundary(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        _, block = equilibrium.equilibrium_rates(horizon, right, parameters)
        return np.asarray(
            [
                left[0] - math.log(initial_state[0]),
                left[1] - math.log(initial_state[1]),
                right[2]
                - block["log_output"]
                - math.log(targets["consumption_share"]),
                right[3]
                + right[1]
                - right[0]
                - math.log(targets["shadow_capability_to_capital"]),
            ]
        )

    solution = equilibrium.solve_bvp(
        ode,
        boundary,
        mesh,
        guess,
        tol=tolerance,
        max_nodes=12000,
        verbose=1,
    )
    solution.duration = horizon
    solution.terminal_output_capital_ratio = math.nan
    solution.normalized_domain = False
    solution.calendar_sol = solution.sol
    return solution, targets


def evaluate_free_boundary_solution(
    name: str,
    solution: object,
    parameters: equilibrium.Parameters,
    step: float = 1.0,
) -> list[dict[str, float | str]]:
    duration = float(solution.duration)
    targets = high_sigma_targets(parameters)
    times = np.arange(0.0, duration, step)
    if times.size == 0 or duration - float(times[-1]) > 1e-8:
        times = np.append(times, duration)
    states = solution.calendar_sol(times)
    path_derivatives = (
        solution.calendar_derivative(times)
        if hasattr(solution, "calendar_derivative")
        else None
    )
    rows: list[dict[str, float | str]] = []
    for index, time in enumerate(times):
        derivatives, block = equilibrium.equilibrium_rates(
            float(time), states[:, index], parameters
        )
        log_capital, log_capability, log_consumption, log_shadow = map(
            float, states[:, index]
        )
        log_population = parameters.n * float(time)
        output_capital_ratio = math.exp(block["log_output"] - log_capital)
        investment_share = (
            (derivatives[0] + parameters.delta) / output_capital_ratio
        )
        consumption_share = math.exp(log_consumption - block["log_output"])
        shadow_capability_to_capital = math.exp(
            log_shadow + log_capability - log_capital
        )
        direction = np.asarray(
            [derivatives[0], derivatives[1], parameters.n, derivatives[3]]
        )
        directional_step = 1e-4 / max(1.0, float(np.max(np.abs(direction))))

        def shifted_static(sign: float) -> dict[str, float]:
            shift = sign * directional_step * direction
            return equilibrium.equilibrium_static_block(
                log_capital + float(shift[0]),
                log_capability + float(shift[1]),
                log_population + float(shift[2]),
                log_shadow + float(shift[3]),
                parameters,
            )

        plus_block = shifted_static(1.0)
        minus_block = shifted_static(-1.0)
        output_growth = (
            plus_block["log_output"] - minus_block["log_output"]
        ) / (2.0 * directional_step)
        wage_growth = (
            plus_block["log_wage"] - minus_block["log_wage"]
        ) / (2.0 * directional_step)
        inverse_elasticity = (
            (1.0 - block["ai_share"]) / parameters.sigma_xl
            + parameters.alpha * block["ai_share"]
        )
        inverse_elasticity_derivative = (
            (parameters.alpha - 1.0 / parameters.sigma_xl)
            * (1.0 - 1.0 / parameters.sigma_xl)
            * block["ai_share"]
            * (1.0 - block["ai_share"])
        )
        monopoly_soc_margin = (
            inverse_elasticity * (1.0 - inverse_elasticity)
            + inverse_elasticity_derivative
        )
        log_price = (
            math.log1p(-parameters.alpha)
            + math.log(block["ai_share"])
            + block["log_output"]
            - block["log_ai_services"]
        )
        log_service_composite = (
            block["log_output"] - parameters.alpha * log_capital
        ) / (1.0 - parameters.alpha)
        log_ai_marginal_cost = math.log(parameters.xi) - log_capability
        ai_markup = math.exp(log_price - log_ai_marginal_cost)
        ai_profit_share = (
            (1.0 - parameters.alpha) * block["ai_share"]
            - block["inference_share"]
        )
        monopoly_foc_log_error = (
            log_price
            + math.log(1.0 - inverse_elasticity)
            - math.log(parameters.xi)
            + log_capability
        )
        log_f_m = (
            math.log(parameters.chi)
            + math.log(parameters.eta)
            + parameters.phi * log_capability
            + (parameters.eta - 1.0) * block["log_effective_research"]
            + math.log(parameters.omega_m)
            + (
                block["log_effective_research"]
                - block["log_automated_research"]
            )
            / parameters.sigma_hm
        )
        log_f_h = (
            math.log(parameters.chi)
            + math.log(parameters.eta)
            + parameters.phi * log_capability
            + (parameters.eta - 1.0) * block["log_effective_research"]
            + math.log1p(-parameters.omega_m)
            + (
                block["log_effective_research"]
                - block["log_human_research"]
            )
            / parameters.sigma_hm
        )
        row: dict[str, float | str] = {
            "scenario": name,
            "time": float(time),
            "time_to_terminal": duration - float(time),
            "log_capital": log_capital,
            "log_capability": log_capability,
            "log_population": log_population,
            "log_consumption": log_consumption,
            "log_shadow_value": log_shadow,
            "log_output": block["log_output"],
            "log_service_composite": log_service_composite,
            "log_wage": block["log_wage"],
            "log_ai_price": log_price,
            "log_ai_marginal_cost": log_ai_marginal_cost,
            "log_ai_services": block["log_ai_services"],
            "log_inference_compute": block["log_inference_compute"],
            "log_human_research": block["log_human_research"],
            "log_production_labor": block["log_production_labor"],
            "log_automated_research": block["log_automated_research"],
            "log_effective_research": block["log_effective_research"],
            "log_output_per_capita": block["log_output"] - log_population,
            "log_consumption_per_capita": log_consumption - log_population,
            "log_capital_per_capita": log_capital - log_population,
            "capital_growth": float(derivatives[0]),
            "capability_growth": float(derivatives[1]),
            "consumption_growth": float(derivatives[2]),
            "consumption_per_capita_growth": float(derivatives[2])
            - parameters.n,
            "shadow_growth": float(derivatives[3]),
            "output_growth": float(output_growth),
            "output_per_capita_growth": float(output_growth - parameters.n),
            "wage_growth": float(wage_growth),
            "output_capital_ratio": output_capital_ratio,
            "capability_growth_to_output_capital": float(derivatives[1])
            / output_capital_ratio,
            "gross_capital_return": block["gross_capital_return"],
            "net_capital_return": block["gross_capital_return"]
            - parameters.delta,
            "human_research_share": block["human_share"],
            "production_labor_population_share": 1.0 - block["human_share"],
            "ai_share": block["ai_share"],
            "inverse_demand_elasticity": inverse_elasticity,
            "monopoly_soc_margin": monopoly_soc_margin,
            "automated_research_share": block["automated_research_share"],
            "inference_share": block["inference_share"],
            "research_resource_share": block["research_resource_share"],
            "investment_share": investment_share,
            "consumption_share": consumption_share,
            "resource_share_sum": (
                consumption_share
                + investment_share
                + block["inference_share"]
                + block["research_resource_share"]
            ),
            "production_labor_share": (
                (1.0 - parameters.alpha) * (1.0 - block["ai_share"])
            ),
            "aggregate_labor_share": math.exp(
                block["log_wage"] + log_population - block["log_output"]
            ),
            "ai_markup": ai_markup,
            "ai_profit_share": ai_profit_share,
            "shadow_capability_to_output": math.exp(
                log_shadow + log_capability - block["log_output"]
            ),
            "shadow_capability_to_capital": shadow_capability_to_capital,
            "human_to_automated_research_ratio": math.exp(
                block["log_human_research"]
                - block["log_automated_research"]
            ),
            "singularity_time_estimate": float(time)
            + 1.0
            / targets["singularity_rate"]
            / output_capital_ratio,
            "monopoly_foc_log_error": monopoly_foc_log_error,
            "research_compute_foc_log_error": (
                log_shadow + log_f_m - math.log(parameters.xi)
            ),
            "research_human_foc_log_error": (
                log_shadow + log_f_h - block["log_wage"]
            ),
            "labor_market_error": (
                math.exp(block["log_production_labor"] - log_population)
                + math.exp(block["log_human_research"] - log_population)
                - 1.0
            ),
            "euler_residual": float(derivatives[2])
            - (
                parameters.n
                + block["gross_capital_return"]
                - parameters.delta
                - parameters.discount
            ),
        }
        if path_derivatives is not None:
            dynamic_residuals = path_derivatives[:, index] - derivatives
            row.update(
                {
                    "capital_law_residual": float(dynamic_residuals[0]),
                    "capability_law_residual": float(dynamic_residuals[1]),
                    "consumption_euler_path_residual": float(dynamic_residuals[2]),
                    "shadow_costate_residual": float(dynamic_residuals[3]),
                }
            )
        rows.append(row)
    return rows


def write_rows(path: Path, rows: Iterable[dict[str, float | str]]) -> None:
    materialized = list(rows)
    fieldnames = list(
        dict.fromkeys(key for row in materialized for key in row.keys())
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialized)


def load_calendar_solution(path: Path) -> object:
    """Load a saved path as a spline-valued continuation guess."""

    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 4:
        raise ValueError(f"Not enough observations in {path}.")
    times = np.asarray([float(row["time"]) for row in rows])
    fields = (
        "log_capital",
        "log_capability",
        "log_consumption",
        "log_shadow_value",
    )
    splines = [
        CubicSpline(
            times,
            np.asarray([float(row[field]) for row in rows]),
            bc_type="natural",
        )
        for field in fields
    ]

    def solution(times_to_evaluate: np.ndarray | float) -> np.ndarray:
        return np.asarray(
            [spline(times_to_evaluate) for spline in splines]
        )

    return SimpleNamespace(
        sol=solution,
        calendar_sol=solution,
        duration=float(times[-1]),
        normalized_domain=False,
    )


def draw_published_figures(
    scenario_rows: dict[str, list[dict[str, float | str]]]
) -> None:
    labels = {
        "equilibrium_sigma_1_35": "sigma_XL = 1.35",
        "equilibrium_sigma_1_50": "sigma_XL = 1.50",
        "equilibrium_sigma_2_00": "sigma_XL = 2.00",
    }
    palette = {
        "equilibrium_sigma_1_35": equilibrium.mechanism.COLORS["blue"],
        "equilibrium_sigma_1_50": equilibrium.mechanism.COLORS["orange"],
        "equilibrium_sigma_2_00": equilibrium.mechanism.COLORS["olive"],
    }
    markers = {
        "equilibrium_sigma_1_35": "circle",
        "equilibrium_sigma_1_50": "square",
        "equilibrium_sigma_2_00": "triangle",
    }
    percent = lambda rows, values: 100.0 * values
    log_change = lambda rows, values: values - values[0]
    per_capita_log_change = lambda rows, values: (
        values - np.asarray([float(row["log_population"]) for row in rows])
        - (
            values[0]
            - float(rows[0]["log_population"])
        )
    )
    log_level = lambda rows, values: np.log(values)

    levels = {
        key: [
            row for row in rows if float(row["output_capital_ratio"]) <= 20.0
        ]
        for key, rows in scenario_rows.items()
    }
    growth = {
        key: [
            row for row in rows if float(row["output_capital_ratio"]) <= 5.0
        ]
        for key, rows in scenario_rows.items()
    }
    equilibrium.mechanism.draw_multiplot(
        FIGURE_DIR / "high_sigma_equilibrium_levels.png",
        "Gross substitution: equilibrium quantities",
        "Natural-log change from date zero; paths end before the asymptotic boundary",
        [
            {"title": "AI capability", "field": "log_capability", "transform": log_change},
            {"title": "Output per capita", "field": "log_output_per_capita", "transform": log_change},
            {"title": "Consumption per capita", "field": "log_consumption_per_capita", "transform": log_change},
            {"title": "Real wage", "field": "log_wage", "transform": log_change},
        ],
        levels,
        labels,
        palette,
        markers,
    )
    equilibrium.mechanism.draw_multiplot(
        FIGURE_DIR / "high_sigma_equilibrium_growth.png",
        "Gross substitution: equilibrium growth and returns",
        "Annual percent; displayed through Y/K = 5 to keep the transition visible",
        [
            {"title": "Capability growth", "field": "capability_growth", "transform": percent, "format": lambda value: f"{value:.0f}%"},
            {"title": "Output growth per capita", "field": "output_per_capita_growth", "transform": percent, "format": lambda value: f"{value:.0f}%"},
            {"title": "Wage growth", "field": "wage_growth", "transform": percent, "format": lambda value: f"{value:.0f}%"},
            {"title": "Net return to capital", "field": "net_capital_return", "transform": percent, "format": lambda value: f"{value:.0f}%"},
        ],
        growth,
        labels,
        palette,
        markers,
    )
    equilibrium.mechanism.draw_multiplot(
        FIGURE_DIR / "high_sigma_equilibrium_production_chain.png",
        "Gross substitution: the production chain",
        "Natural-log change per capita, except for the AI-service price",
        [
            {"title": "Inference compute per capita", "field": "log_inference_compute", "transform": per_capita_log_change},
            {"title": "AI services per capita", "field": "log_ai_services", "transform": per_capita_log_change},
            {"title": "Service composite per capita", "field": "log_service_composite", "transform": per_capita_log_change},
            {"title": "Real AI-service price", "field": "log_ai_price", "transform": log_change},
        ],
        levels,
        labels,
        palette,
        markers,
    )
    equilibrium.mechanism.draw_multiplot(
        FIGURE_DIR / "high_sigma_equilibrium_research_chain.png",
        "Gross substitution: AI research",
        "Natural-log change per capita; H/M is shown in natural logs",
        [
            {"title": "Human research per capita", "field": "log_human_research", "transform": per_capita_log_change},
            {"title": "Automated research per capita", "field": "log_automated_research", "transform": per_capita_log_change},
            {"title": "Effective research per capita", "field": "log_effective_research", "transform": per_capita_log_change},
            {"title": "ln human-machine research ratio", "field": "human_to_automated_research_ratio", "transform": log_level},
        ],
        levels,
        labels,
        palette,
        markers,
    )
    equilibrium.mechanism.draw_multiplot(
        FIGURE_DIR / "high_sigma_equilibrium_resource_allocation.png",
        "Gross substitution: uses of output",
        "Shares of final output in percent; the four uses sum to one",
        [
            {"title": "Consumption / output", "field": "consumption_share", "transform": percent, "format": lambda value: f"{value:.0f}%"},
            {"title": "Investment / output", "field": "investment_share", "transform": percent, "format": lambda value: f"{value:.0f}%"},
            {"title": "Inference resources / output", "field": "inference_share", "transform": percent, "format": lambda value: f"{value:.0f}%"},
            {"title": "Automated research / output", "field": "research_resource_share", "transform": percent, "format": lambda value: f"{value:.1f}%"},
        ],
        levels,
        labels,
        palette,
        markers,
    )
    equilibrium.mechanism.draw_multiplot(
        FIGURE_DIR / "high_sigma_equilibrium_monopoly_block.png",
        "Gross substitution: the integrated AI developer",
        "Natural-log price changes, markup ratio, and operating profits as a share of output",
        [
            {"title": "AI-service price", "field": "log_ai_price", "transform": log_change},
            {"title": "AI-service marginal cost", "field": "log_ai_marginal_cost", "transform": log_change},
            {"title": "Price / marginal cost", "field": "ai_markup"},
            {"title": "Operating profits / output", "field": "ai_profit_share", "transform": percent, "format": lambda value: f"{value:.1f}%"},
        ],
        levels,
        labels,
        palette,
        markers,
    )
    equilibrium.mechanism.draw_multiplot(
        FIGURE_DIR / "high_sigma_equilibrium_shares.png",
        "Gross substitution: equilibrium factor and research shares",
        "Shares in percent; aggregate labor includes production and research wages",
        [
            {"title": "AI contribution to production", "field": "ai_share", "transform": percent, "format": lambda value: f"{value:.0f}%"},
            {"title": "Automated contribution to research", "field": "automated_research_share", "transform": percent, "format": lambda value: f"{value:.0f}%"},
            {"title": "Aggregate labor share", "field": "aggregate_labor_share", "transform": percent, "format": lambda value: f"{value:.0f}%"},
            {"title": "Human researchers / population", "field": "human_research_share", "transform": percent, "format": lambda value: f"{value:.0f}%"},
        ],
        levels,
        labels,
        palette,
        markers,
    )
    asymptotic_rows = {
        key: [
            row for row in rows if float(row["output_capital_ratio"]) >= 1.0
        ]
        for key, rows in scenario_rows.items()
    }
    equilibrium.mechanism.draw_multiplot(
        FIGURE_DIR / "high_sigma_equilibrium_asymptotics.png",
        "Free-boundary convergence toward the singular equilibrium",
        "The common theoretical limit is g_A/(Y/K)=0.0624; T* is scenario-specific",
        [
            {"title": "ln output-capital ratio", "field": "output_capital_ratio", "transform": log_level},
            {"title": "Capability growth / (Y/K)", "field": "capability_growth_to_output_capital", "transform": percent, "format": lambda value: f"{value:.1f}%"},
            {"title": "Estimated singularity year", "field": "singularity_time_estimate"},
            {"title": "Consumption share", "field": "consumption_share", "transform": percent, "format": lambda value: f"{value:.0f}%"},
        ],
        asymptotic_rows,
        labels,
        palette,
        markers,
    )


def assemble_published_results(
    baseline: equilibrium.Parameters,
    initial_state: tuple[float, float, float],
) -> None:
    """Re-solve and publish the validated free-boundary equilibria.

    Saved annual paths are used only as continuation guesses.  The published
    rows and residuals come from a fresh collocation solution, not from a spline
    interpolation of the saved observations.
    """

    sources = [
        (
            "equilibrium_sigma_1_35",
            1.35,
            RESULT_DIR / "high_sigma_equilibrium_1_35_extreme_paths.csv",
            RESULT_DIR / "high_sigma_equilibrium_1_35_extreme_summary.csv",
        ),
        (
            "equilibrium_sigma_1_50",
            1.50,
            RESULT_DIR / "high_sigma_equilibrium_1_5_paths.csv",
            RESULT_DIR / "high_sigma_equilibrium_1_5_summary.csv",
        ),
        (
            "equilibrium_sigma_2_00",
            2.00,
            RESULT_DIR / "high_sigma_equilibrium_2_extreme_paths.csv",
            RESULT_DIR / "high_sigma_equilibrium_2_extreme_summary.csv",
        ),
    ]
    scenario_rows: dict[str, list[dict[str, float | str]]] = {}
    combined_rows: list[dict[str, float | str]] = []
    summaries: list[dict[str, float | str]] = []
    for name, sigma_xl, path, summary_path in sources:
        parameters = replace(baseline, sigma_xl=sigma_xl)
        loaded = load_calendar_solution(path)
        with summary_path.open("r", newline="", encoding="utf-8") as handle:
            saved_summary = next(csv.DictReader(handle))
        solution, targets = solve_high_sigma_equilibrium(
            parameters,
            initial_state,
            terminal_output_capital_ratio=float(
                saved_summary["terminal_output_capital_ratio"]
            ),
            duration_guess=float(saved_summary["duration"]),
            nodes=301,
            tolerance=3e-5,
            previous_solution=loaded,
        )
        if not solution.success:
            raise RuntimeError(f"{name}: {solution.message}")
        rows = evaluate_free_boundary_solution(
            name,
            solution,
            parameters,
            step=1.0,
        )
        scenario_rows[name] = rows
        combined_rows.extend(rows)
        final = rows[-1]
        first_growth_above_five = next(
            (
                float(row["time"])
                for row in rows
                if float(row["output_per_capita_growth"]) >= 0.05
            ),
            math.nan,
        )
        summaries.append(
            {
                "scenario": name,
                "sigma_xl": sigma_xl,
                "terminal_output_capital_ratio": final["output_capital_ratio"],
                "terminal_time": final["time"],
                "estimated_singularity_time": final["singularity_time_estimate"],
                "year_output_per_capita_growth_exceeds_5_percent": first_growth_above_five,
                "initial_consumption_share": rows[0]["consumption_share"],
                "initial_capability_growth": rows[0]["capability_growth"],
                "minimum_wage_growth": min(float(row["wage_growth"]) for row in rows),
                "terminal_ai_share": final["ai_share"],
                "terminal_automated_research_share": final["automated_research_share"],
                "terminal_capability_growth_to_z": final["capability_growth_to_output_capital"],
                "target_capability_growth_to_z": targets["capability_growth_to_z"],
                "terminal_consumption_share": final["consumption_share"],
                "target_consumption_share": targets["consumption_share"],
                "max_rms_residual": float(np.max(solution.rms_residuals)),
                "max_abs_resource_residual": max(
                    abs(float(row["resource_share_sum"]) - 1.0) for row in rows
                ),
                "max_abs_monopoly_foc_log_error": max(
                    abs(float(row["monopoly_foc_log_error"])) for row in rows
                ),
                "max_abs_research_compute_foc_log_error": max(
                    abs(float(row["research_compute_foc_log_error"])) for row in rows
                ),
                "max_abs_research_human_foc_log_error": max(
                    abs(float(row["research_human_foc_log_error"])) for row in rows
                ),
                "max_abs_labor_market_error": max(
                    abs(float(row["labor_market_error"])) for row in rows
                ),
                "max_abs_capital_law_residual": max(
                    abs(float(row["capital_law_residual"])) for row in rows
                ),
                "max_abs_capability_law_residual": max(
                    abs(float(row["capability_law_residual"])) for row in rows
                ),
                "max_abs_consumption_path_residual": max(
                    abs(float(row["consumption_euler_path_residual"]))
                    for row in rows
                ),
                "max_abs_shadow_costate_residual": max(
                    abs(float(row["shadow_costate_residual"])) for row in rows
                ),
                "minimum_monopoly_soc_margin": min(
                    float(row["monopoly_soc_margin"]) for row in rows
                ),
            }
        )
    write_rows(RESULT_DIR / "high_sigma_equilibrium_paths.csv", combined_rows)
    write_rows(RESULT_DIR / "high_sigma_equilibrium_summary.csv", summaries)
    draw_published_figures(scenario_rows)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument(
        "--population-growth",
        type=float,
        default=None,
        help=(
            "Population growth used in the equilibrium path. The initial "
            "capital stock and research productivity remain at their benchmark "
            "calibration so this option isolates the change in n."
        ),
    )
    parser.add_argument(
        "--allow-extrapolative-boundary",
        action="store_true",
        help=(
            "Allow sigma_xl >= 1 / alpha. The resulting free-boundary path "
            "is a conditional branch outside the paper's proved boundary region."
        ),
    )
    parser.add_argument("--terminal-z", type=float, default=10.0)
    parser.add_argument("--duration-guess", type=float, default=300.0)
    parser.add_argument("--initial-log-consumption", type=float, default=-0.60)
    parser.add_argument("--initial-log-shadow", type=float, default=-0.80)
    parser.add_argument("--nodes", type=int, default=301)
    parser.add_argument(
        "--horizon-sequence",
        default="",
        help="Comma-separated continuation horizons for fixed-bvp.",
    )
    parser.add_argument(
        "--free-terminal-z",
        type=float,
        default=0.0,
        help="After fixed-horizon continuation, solve a free boundary at this Y/K.",
    )
    parser.add_argument(
        "--free-z-sequence",
        default="",
        help="Comma-separated free-boundary Y/K targets after fixed continuation.",
    )
    parser.add_argument(
        "--method",
        choices=("shooting", "bvp", "fixed-bvp"),
        default=None,
    )
    parser.add_argument("--output-prefix", default="high_sigma_probe")
    parser.add_argument(
        "--previous-path",
        default="",
        help="Saved calendar-time equilibrium path used as a continuation guess.",
    )
    parser.add_argument("--assemble-published", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
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
    if arguments.assemble_published:
        assemble_published_results(baseline, initial_state)
        return
    if arguments.method is None:
        raise SystemExit(
            "Specify --assemble-published or an explicit --method."
        )
    parameters = replace(
        baseline,
        sigma_xl=arguments.sigma,
        n=(
            baseline.n
            if arguments.population_growth is None
            else arguments.population_growth
        ),
    )
    if (
        parameters.sigma_xl >= 1.0 / parameters.alpha
        and not arguments.allow_extrapolative_boundary
    ):
        raise SystemExit(
            "sigma_xl >= 1 / alpha uses an extrapolative asymptotic boundary; "
            "rerun with --allow-extrapolative-boundary and audit the resulting "
            "conditional branch."
        )
    if arguments.method == "shooting":
        solution, targets = solve_high_sigma_shooting(
            parameters,
            initial_state,
            terminal_output_capital_ratio=arguments.terminal_z,
            duration_guess=arguments.duration_guess,
            jump_guess=(
                arguments.initial_log_consumption,
                arguments.initial_log_shadow,
            ),
        )
    elif arguments.method == "bvp":
        solution, targets = solve_high_sigma_equilibrium(
            parameters,
            initial_state,
            terminal_output_capital_ratio=arguments.terminal_z,
            duration_guess=arguments.duration_guess,
            nodes=arguments.nodes,
        )
    else:
        previous_solution = (
            load_calendar_solution(ROOT / arguments.previous_path)
            if arguments.previous_path
            else None
        )
        horizons = (
            [float(value) for value in arguments.horizon_sequence.split(",")]
            if arguments.horizon_sequence
            else ([] if previous_solution is not None else [arguments.duration_guess])
        )
        continuation_rows: list[dict[str, float | str]] = []
        for horizon in horizons:
            solution, targets = solve_high_sigma_fixed_horizon(
                parameters,
                initial_state,
                horizon=horizon,
                terminal_z_guess=arguments.terminal_z,
                nodes=arguments.nodes,
                previous_solution=previous_solution,
            )
            if not solution.success:
                raise RuntimeError(
                    f"Continuation failed at T={horizon}: {solution.message}"
                )
            terminal_state = solution.sol(horizon)
            _, terminal_block = equilibrium.equilibrium_rates(
                horizon, terminal_state, parameters
            )
            terminal_z = math.exp(
                terminal_block["log_output"] - terminal_state[0]
            )
            continuation_rows.append(
                {
                    "sigma_xl": arguments.sigma,
                    "horizon": horizon,
                    "terminal_output_capital_ratio": terminal_z,
                    "initial_log_consumption": float(solution.sol(0.0)[2]),
                    "initial_log_shadow_value": float(solution.sol(0.0)[3]),
                    "mesh_nodes": solution.x.size,
                    "max_rms_residual": float(np.max(solution.rms_residuals)),
                }
            )
            print(
                f"continuation T={horizon:g}: z_T={terminal_z:.6g}, "
                f"nodes={solution.x.size}, "
                f"residual={np.max(solution.rms_residuals):.3e}",
                flush=True,
            )
            previous_solution = solution
        RESULT_DIR.mkdir(exist_ok=True)
        if continuation_rows:
            write_rows(
                RESULT_DIR / f"{arguments.output_prefix}_continuation.csv",
                continuation_rows,
            )
        free_targets = (
            [float(value) for value in arguments.free_z_sequence.split(",")]
            if arguments.free_z_sequence
            else (
                [arguments.free_terminal_z]
                if arguments.free_terminal_z > 0.0
                else []
            )
        )
        free_rows: list[dict[str, float | str]] = []
        for free_target in free_targets:
            solution, targets = solve_high_sigma_equilibrium(
                parameters,
                initial_state,
                terminal_output_capital_ratio=free_target,
                duration_guess=float(previous_solution.duration),
                nodes=arguments.nodes,
                previous_solution=previous_solution,
            )
            if not solution.success:
                raise RuntimeError(
                    f"Free-boundary continuation failed at z={free_target}: "
                    + solution.message
                )
            free_rows.append(
                {
                    "sigma_xl": arguments.sigma,
                    "terminal_output_capital_ratio": free_target,
                    "duration": solution.duration,
                    "initial_log_consumption": float(solution.sol(0.0)[2]),
                    "initial_log_shadow_value": float(solution.sol(0.0)[3]),
                    "mesh_nodes": solution.x.size,
                    "max_rms_residual": float(np.max(solution.rms_residuals)),
                }
            )
            print(
                f"free boundary z_T={free_target:g}: "
                f"T={solution.duration:.6f}, nodes={solution.x.size}, "
                f"residual={np.max(solution.rms_residuals):.3e}",
                flush=True,
            )
            previous_solution = solution
        if free_rows:
            write_rows(
                RESULT_DIR / f"{arguments.output_prefix}_free_continuation.csv",
                free_rows,
            )
    if not solution.success:
        diagnostic = ""
        if hasattr(solution, "root_result"):
            diagnostic = (
                f"; unknowns={solution.root_result.x}; "
                f"terminal residuals={solution.root_result.fun}"
            )
        raise RuntimeError(solution.message + diagnostic)
    rows = evaluate_free_boundary_solution(
        f"equilibrium_sigma_{arguments.sigma:g}",
        solution,
        parameters,
    )
    RESULT_DIR.mkdir(exist_ok=True)
    write_rows(
        RESULT_DIR / f"{arguments.output_prefix}_paths.csv",
        rows,
    )
    summary = {
        "sigma_xl": arguments.sigma,
        "population_growth": parameters.n,
        "boundary_is_analytically_sufficient": (
            parameters.sigma_xl < 1.0 / parameters.alpha
        ),
        "terminal_output_capital_ratio": arguments.terminal_z,
        "duration": solution.duration,
        "mesh_nodes": (
            solution.x.size if hasattr(solution, "x") else solution.t.size
        ),
        "max_rms_residual": (
            float(np.max(solution.rms_residuals))
            if hasattr(solution, "rms_residuals")
            else math.nan
        ),
        **{f"target_{key}": value for key, value in targets.items()},
        **{
            f"terminal_{key}": value
            for key, value in rows[-1].items()
            if isinstance(value, float)
        },
    }
    write_rows(
        RESULT_DIR / f"{arguments.output_prefix}_summary.csv",
        [summary],
    )
    print(
        f"sigma={arguments.sigma:g}, "
        f"z_T={float(rows[-1]['output_capital_ratio']):g}, "
        f"T={solution.duration:.6f}, "
        f"terminal residual="
        f"{np.max(np.abs(solution.root_result.fun)) if hasattr(solution, 'root_result') else np.max(solution.rms_residuals):.3e}"
    )


if __name__ == "__main__":
    main()
