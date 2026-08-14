"""Free-boundary A*M equilibrium transitions for gross substitution.

For ``sigma_xl > 1`` and unbounded capability, the model has no finite-rate
terminal balanced-growth path.  This module therefore does not impose one.  It
uses the AI-dominated necessary-condition asymptotics to terminate the path at a large
output--capital ratio and treats calendar time to that boundary as an endogenous
free parameter.  Raising the terminal ratio supplies a convergence test toward
the finite-time singularity of the limiting branch.

The asymptotic ratios characterize a conditional AI-dominated branch for
``sigma_xl > 1`` and ``sigma_hm > 1``; they do not prove that the branch is
reached from arbitrary initial conditions.  Every candidate path must be checked
ex post for AI dominance, automation, boundary convergence, interiority,
monopoly optimality, and all equilibrium residuals.
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

import simulate_axm_equilibrium as equilibrium
from scipy.interpolate import CubicSpline


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "numerical_axm"
FIGURE_DIR = ROOT / "figures_axm"


def high_sigma_targets(parameters: equilibrium.Parameters) -> dict[str, float]:
    """Return AI-dominated singular-path ratios.

    Write ``z = Y/K`` and ``kappa = (1-alpha)/alpha``.  Along the
    AI-dominated necessary-condition singularity, ``g_A / z``, the consumption share,
    the investment share, the automated-research resource share, and ``q A/K``
    converge to constants.
    """

    if parameters.sigma_xl <= 1.0:
        raise ValueError("The high-sigma boundary requires sigma_xl > 1.")
    if parameters.sigma_hm <= 1.0:
        raise ValueError(
            "The current high-sigma boundary requires sigma_hm > 1."
        )
    if not 0.0 < parameters.eta < 1.0:
        raise ValueError("The current singular ratios require 0 < eta < 1.")

    alpha = parameters.alpha
    kappa = (1.0 - alpha) / alpha
    # In the automated limit, the generalized CES implies
    # dot A proportional to (A*M)^eta, so A enters research directly through
    # A*M. The singular denominator is therefore
    # 1+kappa-eta, not the denominator of a model in which raw M enters
    # the research CES without being augmented by A.
    denominator = 1.0 + kappa - parameters.eta
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
        + parameters.sigma_xl
        / (parameters.sigma_xl - 1.0)
        * math.log(parameters.omega_x)
    )
    return math.exp(log_coefficient)


def positive_duration(raw_duration: float) -> float:
    """Map an unconstrained scalar smoothly into a positive duration.

    A hard cap on ``exp(raw_duration)`` makes the duration derivative exactly
    zero during Newton trials and can create a singular BVP Jacobian.  The
    softplus map remains smooth and has unit slope for economically relevant
    long durations.
    """

    if raw_duration > 30.0:
        return raw_duration
    if raw_duration < -30.0:
        return math.exp(raw_duration)
    return math.log1p(math.exp(raw_duration))


def inverse_positive_duration(duration: float) -> float:
    """Inverse of :func:`positive_duration` for a strictly positive value."""

    if duration <= 0.0:
        raise ValueError("The duration guess must be strictly positive.")
    if duration > 30.0:
        return duration
    return math.log(math.expm1(duration))


def asymptotic_terminal_state(
    parameters: equilibrium.Parameters,
    initial_state: tuple[float, float, float],
    terminal_output_capital_ratio: float,
    duration_guess: float,
    targets: dict[str, float],
) -> np.ndarray:
    """Construct an exact-static terminal state near the singular boundary.

    The analytical scaling supplies starting levels for ``A`` and ``K``.  A
    two-dimensional least-squares correction then imposes, using the full
    intratemporal block at the guessed terminal date, both ``Y/K=z_T`` and
    ``g_A/z=h``.  Consumption and the capability shadow value follow from the
    two terminal ratios that the free-boundary problem imposes.
    """

    if terminal_output_capital_ratio <= 0.0:
        raise ValueError("The terminal output-capital ratio must be positive.")
    coefficient = asymptotic_output_capital_coefficient(parameters)
    log_terminal_z = math.log(terminal_output_capital_ratio)
    log_terminal_capability = max(
        math.log(initial_state[1]) + 0.25,
        (log_terminal_z - math.log(coefficient)) / targets["kappa"],
    )
    automated_service_weight = parameters.omega_m ** (
        parameters.sigma_hm / (parameters.sigma_hm - 1.0)
    )
    log_terminal_capital = (
        math.log(targets["capability_growth_to_z"] / parameters.chi)
        / parameters.eta
        - math.log(automated_service_weight)
        - math.log(targets["research_share"])
        + (1.0 - parameters.eta)
        / parameters.eta
        * (log_terminal_z + log_terminal_capability)
    )

    def terminal_residual(levels: np.ndarray) -> np.ndarray:
        log_capital, log_capability = map(float, levels)
        log_shadow = (
            math.log(targets["shadow_capability_to_capital"])
            + log_capital
            - log_capability
        )
        block = equilibrium.equilibrium_static_block(
            log_capital,
            log_capability,
            math.log(initial_state[2]) + parameters.n * duration_guess,
            log_shadow,
            parameters,
        )
        log_z = block["log_output"] - log_capital
        log_g_a = (
            math.log(parameters.chi)
            + parameters.eta * block["log_effective_research"]
            - log_capability
        )
        return np.asarray(
            [
                log_z - log_terminal_z,
                log_g_a
                - log_z
                - math.log(targets["capability_growth_to_z"]),
            ]
        )

    correction = equilibrium.least_squares(
        terminal_residual,
        np.asarray([log_terminal_capital, log_terminal_capability]),
        xtol=1e-11,
        ftol=1e-11,
        gtol=1e-11,
        max_nfev=100,
    )
    if (not correction.success) or np.max(np.abs(correction.fun)) > 1e-7:
        raise RuntimeError(
            "Could not construct a terminal singular-scaling state: "
            f"residual={correction.fun}."
        )
    log_terminal_capital, log_terminal_capability = map(float, correction.x)
    log_shadow = (
        math.log(targets["shadow_capability_to_capital"])
        + log_terminal_capital
        - log_terminal_capability
    )
    block = equilibrium.equilibrium_static_block(
        log_terminal_capital,
        log_terminal_capability,
        math.log(initial_state[2]) + parameters.n * duration_guess,
        log_shadow,
        parameters,
    )
    log_consumption = (
        math.log(targets["consumption_share"]) + block["log_output"]
    )
    return np.asarray(
        [
            log_terminal_capital,
            log_terminal_capability,
            log_consumption,
            log_shadow,
        ]
    )


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
    terminal_state = asymptotic_terminal_state(
        parameters,
        initial_state,
        terminal_output_capital_ratio,
        duration_guess,
        targets,
    )
    (
        log_terminal_capital,
        log_terminal_capability,
        log_terminal_consumption,
        log_terminal_shadow,
    ) = map(float, terminal_state)

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
    log_consumption = (
        math.log(targets["consumption_share"])
        + log_capital
        + math.log(terminal_output_capital_ratio)
        + targets["kappa"]
        * (log_capability - log_terminal_capability)
    )
    # Match all four terminal values exactly while leaving the initial levels
    # close to the current-state normalization.
    terminal_profile = mesh**4
    log_consumption += (
        log_terminal_consumption - float(log_consumption[-1])
    ) * terminal_profile
    log_shadow += (
        log_terminal_shadow - float(log_shadow[-1])
    ) * terminal_profile

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
            duration_guess = float(previous_solution.duration)
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
            duration_guess += (
                1.0
                / targets["singularity_rate"]
                * (1.0 / old_z - 1.0 / terminal_output_capital_ratio)
            )
        target_terminal = asymptotic_terminal_state(
            parameters,
            initial_state,
            terminal_output_capital_ratio,
            duration_guess,
            targets,
        )
        profile = mesh**4
        guess += (target_terminal - guess[:, -1])[:, None] * profile[None, :]

    def ode(
        normalized_time: np.ndarray,
        state: np.ndarray,
        raw_duration: np.ndarray,
    ) -> np.ndarray:
        duration = positive_duration(float(raw_duration[0]))
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
        raw_duration: np.ndarray,
    ) -> np.ndarray:
        duration = positive_duration(float(raw_duration[0]))
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
        p=np.asarray([inverse_positive_duration(duration_guess)]),
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
    solution.duration = positive_duration(float(solution.p[0]))
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
    times_to_evaluate: np.ndarray | None = None,
) -> list[dict[str, float | str]]:
    duration = float(solution.duration)
    targets = high_sigma_targets(parameters)
    if times_to_evaluate is None:
        regular_times = np.arange(0.0, duration, step)
        adaptive_times = (
            np.asarray(solution.x) * duration
            if getattr(solution, "normalized_domain", False)
            else np.asarray(getattr(solution, "x", []), dtype=float)
        )
        maximum_tail_gap = min(50.0, duration)
        minimum_tail_gap = max(1e-8, duration * 1e-10)
        tail_gaps = np.geomspace(
            minimum_tail_gap,
            max(maximum_tail_gap, minimum_tail_gap),
            240,
        )
        tail_times = duration - tail_gaps
        times = np.unique(
            np.clip(
                np.concatenate(
                    [regular_times, adaptive_times, tail_times, [duration]]
                ),
                0.0,
                duration,
            )
        )
    else:
        times = np.unique(
            np.clip(np.asarray(times_to_evaluate, dtype=float), 0.0, duration)
        )
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
        log_ai_marginal_cost = -log_capability
        ai_markup = math.exp(log_price - log_ai_marginal_cost)
        ai_profit_share = (
            (1.0 - parameters.alpha) * block["ai_share"]
            - block["inference_share"]
        )
        monopoly_foc_log_error = (
            log_price
            + math.log(1.0 - inverse_elasticity)
            + log_capability
        )
        technology_errors = equilibrium.technology_log_errors(
            block,
            log_capital,
            log_capability,
            parameters,
        )
        log_f_m = (
            math.log(parameters.chi)
            + math.log(parameters.eta)
            + (parameters.eta - 1.0) * block["log_effective_research"]
            + math.log(parameters.omega_m)
            + log_capability
            + (
                block["log_effective_research"]
                - block["log_automated_research_services"]
            )
            / parameters.sigma_hm
        )
        log_f_h = (
            math.log(parameters.chi)
            + math.log(parameters.eta)
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
            "log_automated_research_services": block[
                "log_automated_research_services"
            ],
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
            "human_to_automated_service_ratio": math.exp(
                block["log_human_research"]
                - block["log_automated_research_services"]
            ),
            "singularity_time_estimate": float(time)
            + 1.0
            / targets["singularity_rate"]
            / output_capital_ratio,
            "monopoly_foc_log_error": monopoly_foc_log_error,
            **technology_errors,
            "research_compute_foc_log_error": (
                log_shadow + log_f_m
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
            {"title": "Effective-research index per capita", "field": "log_effective_research", "transform": per_capita_log_change},
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
        "Free-boundary convergence toward the singular limit",
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


PUBLISHED_SIGMA_SEQUENCE = (1.02, 1.05, 1.10, 1.20, 1.30, 1.40, 1.50)
PUBLISHED_HORIZON_SEQUENCE = (
    150.0,
    200.0,
    300.0,
    400.0,
    600.0,
    700.0,
    750.0,
    800.0,
    825.0,
    840.0,
)
PUBLISHED_COARSE_FREE_BOUNDARY_SEQUENCE = (
    0.5,
    0.6,
    0.8,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    8.0,
    12.0,
    16.0,
    24.0,
    32.0,
    48.0,
    64.0,
)
PUBLISHED_REFINEMENT_BOUNDARIES = (16.0, 32.0, 64.0, 128.0)
PUBLISHED_PRELIMINARY_TOLERANCE = 3e-5
PUBLISHED_REPORTED_TOLERANCE = 1e-6
PUBLISHED_PREFIX = "high_sigma_sigma150_validated"
PUBLISHED_EXTRA_PREFIX = "high_sigma_sigma150_z128_validated"


def published_reproduction_plan() -> dict[str, object]:
    """Return the immutable continuation plan used for the paper."""

    return {
        "sigma_xl": 1.5,
        "sigma_hm": 2.0,
        "unit_elasticity_seed_horizon": 1600.0,
        "initial_high_sigma_horizon": 100.0,
        "sigma_sequence": PUBLISHED_SIGMA_SEQUENCE,
        "horizon_sequence": PUBLISHED_HORIZON_SEQUENCE,
        "coarse_free_boundary_sequence": (
            PUBLISHED_COARSE_FREE_BOUNDARY_SEQUENCE
        ),
        "refinement_boundary_sequence": PUBLISHED_REFINEMENT_BOUNDARIES,
        "saved_boundaries": PUBLISHED_REFINEMENT_BOUNDARIES,
        "preliminary_tolerance": PUBLISHED_PRELIMINARY_TOLERANCE,
        "reported_tolerance": PUBLISHED_REPORTED_TOLERANCE,
        "canonical_outputs": (
            f"numerical_axm/{PUBLISHED_PREFIX}_free_continuation.csv",
            f"numerical_axm/{PUBLISHED_PREFIX}_boundary_paths.csv",
            f"numerical_axm/{PUBLISHED_EXTRA_PREFIX}_free_continuation.csv",
            f"numerical_axm/{PUBLISHED_EXTRA_PREFIX}_boundary_paths.csv",
        ),
    }


def describe_published_reproduction() -> None:
    """Print the exact paper-continuation plan without solving it."""

    plan = published_reproduction_plan()
    print("Validated sigma_XL=1.5 reproduction plan", flush=True)
    for key, value in plan.items():
        print(f"  {key}: {value}", flush=True)


def _check_continuation_solution(
    solution: object,
    stage: str,
) -> None:
    """Fail immediately when a continuation stage is not a valid BVP solve."""

    if not solution.success:
        raise RuntimeError(f"{stage}: {solution.message}")
    maximum_residual = float(np.max(solution.rms_residuals))
    if not math.isfinite(maximum_residual):
        raise RuntimeError(f"{stage}: non-finite collocation residual.")
    print(
        f"{stage}: nodes={solution.x.size}, "
        f"max RMS residual={maximum_residual:.3e}",
        flush=True,
    )


def _published_continuation_row(
    solution: object,
    sigma_xl: float,
    boundary: float,
    targets: dict[str, float],
) -> dict[str, float]:
    """Summarize one canonical free-boundary solution."""

    initial = solution.sol(0.0)
    return {
        "sigma_xl": sigma_xl,
        "terminal_output_capital_ratio": boundary,
        "duration": float(solution.duration),
        "initial_log_consumption": float(initial[2]),
        "initial_log_shadow_value": float(initial[3]),
        "mesh_nodes": float(solution.x.size),
        "max_rms_residual": float(np.max(solution.rms_residuals)),
        "estimated_singularity_time": (
            float(solution.duration)
            + 1.0 / targets["singularity_rate"] / boundary
        ),
    }


def assemble_published_results(
    baseline: equilibrium.Parameters,
    initial_state: tuple[float, float, float],
    nodes: int = 301,
) -> None:
    """Reproduce the validated ``sigma_XL=1.5`` continuation from scratch.

    The route is deliberately explicit.  It first solves the reported
    unit-elasticity, ``sigma_HM=2`` path, moves ``sigma_XL`` away from one at a
    short fixed horizon, lengthens that horizon, and only then releases the
    terminal date while increasing ``Y/K``.  Intermediate solutions are
    continuation guesses only.  The function writes paths and metadata for the
    four audited boundaries and no per-stage diagnostic copies.  Preliminary
    sigma, horizon, and boundary continuation uses the looser tolerance that
    generated the successful guesses; only the second-pass boundary solutions
    at tolerance ``1e-6`` are reported.
    """

    RESULT_DIR.mkdir(exist_ok=True)
    unit_parameters = replace(baseline, sigma_xl=1.0, sigma_hm=2.0)
    print(
        "Solving the unit-elasticity sigma_HM=2 seed at T=1600...",
        flush=True,
    )
    previous_solution, _ = equilibrium.solve_equilibrium(
        unit_parameters,
        initial_state,
        horizon=1600.0,
        nodes=nodes,
        tolerance=PUBLISHED_PRELIMINARY_TOLERANCE,
    )
    _check_continuation_solution(previous_solution, "unit-elasticity seed")
    previous_solution.duration = 1600.0
    previous_solution.normalized_domain = False
    previous_solution.calendar_sol = previous_solution.sol

    parameters = unit_parameters
    for sigma_xl in PUBLISHED_SIGMA_SEQUENCE:
        parameters = replace(unit_parameters, sigma_xl=sigma_xl)
        previous_solution, _ = solve_high_sigma_fixed_horizon(
            parameters,
            initial_state,
            horizon=100.0,
            terminal_z_guess=1.0,
            nodes=nodes,
            tolerance=PUBLISHED_PRELIMINARY_TOLERANCE,
            previous_solution=previous_solution,
        )
        _check_continuation_solution(
            previous_solution,
            f"sigma continuation sigma_XL={sigma_xl:g}, T=100",
        )

    for horizon in PUBLISHED_HORIZON_SEQUENCE:
        previous_solution, _ = solve_high_sigma_fixed_horizon(
            parameters,
            initial_state,
            horizon=horizon,
            terminal_z_guess=1.0,
            nodes=nodes,
            tolerance=PUBLISHED_PRELIMINARY_TOLERANCE,
            previous_solution=previous_solution,
        )
        _check_continuation_solution(
            previous_solution,
            f"horizon continuation sigma_XL=1.5, T={horizon:g}",
        )

    for boundary in PUBLISHED_COARSE_FREE_BOUNDARY_SEQUENCE:
        previous_solution, targets = solve_high_sigma_equilibrium(
            parameters,
            initial_state,
            terminal_output_capital_ratio=boundary,
            duration_guess=float(previous_solution.duration),
            nodes=nodes,
            tolerance=PUBLISHED_PRELIMINARY_TOLERANCE,
            previous_solution=previous_solution,
        )
        _check_continuation_solution(
            previous_solution,
            f"coarse free-boundary continuation z={boundary:g}",
        )

    saved_rows: dict[float, list[dict[str, float | str]]] = {}
    saved_summaries: dict[float, dict[str, float]] = {}
    for boundary in PUBLISHED_REFINEMENT_BOUNDARIES:
        previous_solution, targets = solve_high_sigma_equilibrium(
            parameters,
            initial_state,
            terminal_output_capital_ratio=boundary,
            duration_guess=float(previous_solution.duration),
            nodes=nodes,
            tolerance=PUBLISHED_REPORTED_TOLERANCE,
            previous_solution=previous_solution,
        )
        _check_continuation_solution(
            previous_solution,
            f"refined free-boundary continuation z={boundary:g}",
        )
        rows = evaluate_free_boundary_solution(
            f"equilibrium_sigma_1.5_z_{boundary:g}",
            previous_solution,
            parameters,
        )
        for row in rows:
            row["terminal_boundary_z"] = boundary
        saved_rows[boundary] = rows
        saved_summaries[boundary] = _published_continuation_row(
            previous_solution,
            sigma_xl=1.5,
            boundary=boundary,
            targets=targets,
        )

    missing = set(PUBLISHED_REFINEMENT_BOUNDARIES) - set(saved_rows)
    if missing:
        raise RuntimeError(f"Canonical boundaries were not solved: {missing}.")

    primary_boundaries = (16.0, 32.0, 64.0)
    write_rows(
        RESULT_DIR / f"{PUBLISHED_PREFIX}_free_continuation.csv",
        [saved_summaries[value] for value in primary_boundaries],
    )
    write_rows(
        RESULT_DIR / f"{PUBLISHED_PREFIX}_boundary_paths.csv",
        [row for value in primary_boundaries for row in saved_rows[value]],
    )
    write_rows(
        RESULT_DIR / f"{PUBLISHED_EXTRA_PREFIX}_free_continuation.csv",
        [saved_summaries[128.0]],
    )
    write_rows(
        RESULT_DIR / f"{PUBLISHED_EXTRA_PREFIX}_boundary_paths.csv",
        saved_rows[128.0],
    )
    print("Canonical high-sigma outputs written:", flush=True)
    for path in published_reproduction_plan()["canonical_outputs"]:
        print(f"  {path}", flush=True)


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
        "--research-productivity",
        type=float,
        default=None,
        help="Override chi for an explicitly labeled acceleration experiment.",
    )
    parser.add_argument("--terminal-z", type=float, default=10.0)
    parser.add_argument("--duration-guess", type=float, default=300.0)
    parser.add_argument("--initial-log-consumption", type=float, default=-0.60)
    parser.add_argument("--initial-log-shadow", type=float, default=-0.80)
    parser.add_argument("--nodes", type=int, default=301)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=3e-5,
        help="Relative tolerance passed to the boundary-value solver.",
    )
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
    parser.add_argument(
        "--assemble-published",
        action="store_true",
        help=(
            "Run the exact staged sigma_XL=1.5 continuation used by the "
            "paper and write its four canonical solver outputs."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the published continuation plan without solving or writing.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    baseline = replace(
        equilibrium.Parameters(),
        chi=0.01,
        sigma_xl=1.0,
        sigma_hm=2.0,
    )
    seed_guess = equilibrium.fixed_share_guess(
        baseline,
        (1.0, 1.0, 1.0),
        horizon=1.0,
        mesh=np.asarray([0.0]),
    )
    initial_capital = math.exp(float(seed_guess[0, 0]))
    initial_state = (initial_capital, 1.0, 1.0)
    if arguments.assemble_published:
        if arguments.dry_run:
            describe_published_reproduction()
            return
        assemble_published_results(
            baseline,
            initial_state,
            nodes=arguments.nodes,
        )
        return
    if arguments.dry_run:
        raise SystemExit("--dry-run requires --assemble-published.")
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
        chi=(
            baseline.chi
            if arguments.research_productivity is None
            else arguments.research_productivity
        ),
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
            tolerance=arguments.tolerance,
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
                tolerance=arguments.tolerance,
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
        all_boundary_paths: list[dict[str, float | str]] = []
        for free_target in free_targets:
            solution, targets = solve_high_sigma_equilibrium(
                parameters,
                initial_state,
                terminal_output_capital_ratio=free_target,
                duration_guess=float(previous_solution.duration),
                nodes=arguments.nodes,
                tolerance=arguments.tolerance,
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
                    "estimated_singularity_time": (
                        solution.duration
                        + 1.0
                        / targets["singularity_rate"]
                        / free_target
                    ),
                }
            )
            boundary_rows = evaluate_free_boundary_solution(
                f"equilibrium_sigma_{arguments.sigma:g}_z_{free_target:g}",
                solution,
                parameters,
            )
            for row in boundary_rows:
                row["terminal_boundary_z"] = free_target
            all_boundary_paths.extend(boundary_rows)
            safe_target = str(free_target).replace(".", "p")
            write_rows(
                RESULT_DIR
                / f"{arguments.output_prefix}_z_{safe_target}_paths.csv",
                boundary_rows,
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
            write_rows(
                RESULT_DIR / f"{arguments.output_prefix}_boundary_paths.csv",
                all_boundary_paths,
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
        "research_productivity": parameters.chi,
        "boundary_is_conditional": True,
        "terminal_output_capital_ratio": float(
            rows[-1]["output_capital_ratio"]
        ),
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
