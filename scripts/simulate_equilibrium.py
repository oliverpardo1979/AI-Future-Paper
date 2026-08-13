"""Perfect-foresight equilibrium transitions for the AI automation model.

This module replaces the proportional-investment and proportional-research
closure in ``simulate_model.py`` with the household Euler equation and the
integrated developer's research first-order and costate conditions.  Initial
capital and capability are predetermined.  Initial consumption and the private
shadow value of capability jump so that the path approaches the relevant
constant-growth limit.

The boundary-value problem currently covers regimes with a finite asymptotic
constant-growth path.  It deliberately does not label a finite-horizon
high-substitution path as an infinite-horizon equilibrium when the model has no
finite-rate terminal steady state.
"""

from __future__ import annotations

import csv
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TMP_DEPS = ROOT / "tmp" / "pydeps"
if TMP_DEPS.exists():
    sys.path.insert(0, str(TMP_DEPS))
LOCAL_DEPS = ROOT / ".python-packages"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))
sys.path.insert(0, str(ROOT / "scripts"))

from scipy.integrate import solve_bvp, solve_ivp  # noqa: E402
from scipy.optimize import brentq, least_squares  # noqa: E402

import simulate_model as mechanism  # noqa: E402


Parameters = mechanism.Parameters
RESULT_DIR = ROOT / "numerical"


def logsumexp_pair(left: float, right: float) -> float:
    maximum = max(left, right)
    return maximum + math.log(
        math.exp(left - maximum) + math.exp(right - maximum)
    )


def logistic(value: float) -> float:
    if value >= 0.0:
        exponential = math.exp(-value)
        return 1.0 / (1.0 + exponential)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def bounded_exp(value: float, upper: float = 60.0) -> float:
    """Exponentiate safely during damped Newton trials of the BVP solver."""

    return math.exp(min(max(value, -700.0), upper))


def research_unit_cost(log_wage: float, parameters: Parameters) -> float:
    sigma = parameters.sigma_hm
    if abs(sigma - 1.0) <= 1e-10:
        omega_m = parameters.omega_m
        omega_h = 1.0 - omega_m
        return (
            omega_h * (log_wage - math.log(omega_h))
            + omega_m * (math.log(parameters.xi) - math.log(omega_m))
        )
    log_human_term = (
        sigma * math.log1p(-parameters.omega_m)
        + (1.0 - sigma) * log_wage
    )
    log_machine_term = (
        sigma * math.log(parameters.omega_m)
        + (1.0 - sigma) * math.log(parameters.xi)
    )
    return logsumexp_pair(log_human_term, log_machine_term) / (1.0 - sigma)


def monopoly_service_block(
    log_capital: float,
    log_labor: float,
    log_capability: float,
    parameters: Parameters,
) -> dict[str, float]:
    """Solve monopoly service supply using the CES share representation."""

    if abs(parameters.sigma_xl - 1.0) <= 1e-10:
        beta = (1.0 - parameters.alpha) * parameters.omega_x
        log_ai_services = (
            2.0 * math.log(beta)
            + log_capability
            - math.log(parameters.xi)
            + parameters.alpha * log_capital
            + (1.0 - parameters.alpha)
            * (1.0 - parameters.omega_x)
            * log_labor
        ) / (1.0 - beta)
        log_output = (
            parameters.alpha * log_capital
            + (1.0 - parameters.alpha)
            * (1.0 - parameters.omega_x)
            * log_labor
            + beta * log_ai_services
        )
        return {
            "log_ai_ratio": log_ai_services - log_labor,
            "log_output": log_output,
            "ai_share": parameters.omega_x,
        }

    rho = (parameters.sigma_xl - 1.0) / parameters.sigma_xl

    def quantities(log_ratio: float) -> tuple[float, float]:
        log_human_term = math.log1p(-parameters.omega_x)
        log_ai_term = math.log(parameters.omega_x) + rho * log_ratio
        log_denominator = logsumexp_pair(log_human_term, log_ai_term)
        log_z_per_worker = log_denominator / rho
        ai_share = math.exp(log_ai_term - log_denominator)
        log_output = (
            parameters.alpha * log_capital
            + (1.0 - parameters.alpha)
            * (log_labor + log_z_per_worker)
        )
        return log_output, min(max(ai_share, 1e-14), 1.0 - 1e-14)

    def residual(log_ratio: float) -> float:
        log_output, ai_share = quantities(log_ratio)
        inverse_elasticity = (
            (1.0 - ai_share) / parameters.sigma_xl
            + parameters.alpha * ai_share
        )
        markup_term = 1.0 - inverse_elasticity
        if markup_term <= 0.0:
            return -1e6
        log_price = (
            math.log1p(-parameters.alpha)
            + math.log(ai_share)
            + log_output
            - log_labor
            - log_ratio
        )
        return (
            log_price
            + math.log(markup_term)
            - math.log(parameters.xi)
            + log_capability
        )

    lower = -500.0
    upper = 500.0
    lower_value = residual(lower)
    upper_value = residual(upper)
    if lower_value * upper_value <= 0.0:
        log_ai_ratio = brentq(
            residual,
            lower,
            upper,
            xtol=1e-10,
            rtol=1e-10,
            maxiter=80,
        )
        log_output, ai_share = quantities(log_ai_ratio)
        return {
            "log_ai_ratio": log_ai_ratio,
            "log_output": log_output,
            "ai_share": ai_share,
        }

    grid = np.linspace(lower, upper, 101)
    previous_x = float(grid[0])
    previous_value = residual(previous_x)
    candidates: list[tuple[float, float]] = [(abs(previous_value), previous_x)]
    for value in grid[1:]:
        current_x = float(value)
        current_value = residual(current_x)
        candidates.append((abs(current_value), current_x))
        if previous_value * current_value <= 0.0:
            log_ai_ratio = brentq(
                residual,
                previous_x,
                current_x,
                xtol=1e-11,
                rtol=1e-11,
                maxiter=120,
            )
            log_output, ai_share = quantities(log_ai_ratio)
            return {
                "log_ai_ratio": log_ai_ratio,
                "log_output": log_output,
                "ai_share": ai_share,
            }
        previous_x = current_x
        previous_value = current_value
    log_ai_ratio = min(candidates)[1]
    log_output, ai_share = quantities(log_ai_ratio)
    return {
        "log_ai_ratio": log_ai_ratio,
        "log_output": log_output,
        "ai_share": ai_share,
    }


def equilibrium_static_block(
    log_capital: float,
    log_capability: float,
    log_population: float,
    log_shadow_value: float,
    parameters: Parameters,
) -> dict[str, float]:
    """Solve all intratemporal equilibrium conditions at one date."""

    def allocation(logit_human_share: float) -> tuple[float, dict[str, float]]:
        human_share = min(
            max(logistic(logit_human_share), 1e-14), 1.0 - 1e-14
        )
        log_human_research = log_population + math.log(human_share)
        log_production_labor = log_population + math.log1p(-human_share)

        production = monopoly_service_block(
            log_capital,
            log_production_labor,
            log_capability,
            parameters,
        )
        log_ai_ratio = production["log_ai_ratio"]
        log_output = production["log_output"]
        ai_share = production["ai_share"]
        log_wage = (
            math.log1p(-parameters.alpha)
            + math.log1p(-ai_share)
            + log_output
            - log_production_labor
        )
        log_research_price = research_unit_cost(log_wage, parameters)
        log_effective_research = (
            log_shadow_value
            + math.log(parameters.chi)
            + math.log(parameters.eta)
            + parameters.phi * log_capability
            - log_research_price
        ) / (1.0 - parameters.eta)
        log_human_demand = (
            parameters.sigma_hm * math.log1p(-parameters.omega_m)
            + parameters.sigma_hm * (log_research_price - log_wage)
            + log_effective_research
        )
        residual = log_human_research - log_human_demand
        return residual, {
            "human_share": human_share,
            "log_human_research": log_human_research,
            "log_production_labor": log_production_labor,
            "log_ai_ratio": log_ai_ratio,
            "log_output": log_output,
            "ai_share": ai_share,
            "log_wage": log_wage,
            "log_research_price": log_research_price,
            "log_effective_research": log_effective_research,
        }

    lower = -35.0
    upper = 35.0
    lower_residual = allocation(lower)[0]
    upper_residual = allocation(upper)[0]
    if lower_residual * upper_residual > 0.0:
        grid = np.linspace(-60.0, 60.0, 121)
        previous = allocation(float(grid[0]))[0]
        bracket: tuple[float, float] | None = None
        for left, right in zip(grid[:-1], grid[1:]):
            current = allocation(float(right))[0]
            if previous * current <= 0.0:
                bracket = (float(left), float(right))
                break
            previous = current
        if bracket is None:
            # A Newton trial in the outer boundary-value solver can temporarily
            # imply a corner labor allocation even when the converged path is
            # interior.  Returning the closest endpoint keeps the static map
            # defined during those trials.  The reported equilibrium is rejected
            # unless its labor-market and research FOCs are interior and small.
            endpoint_residuals = [
                (abs(allocation(float(value))[0]), float(value))
                for value in grid
            ]
            logit_human_share = min(endpoint_residuals)[1]
        else:
            lower, upper = bracket
            logit_human_share = brentq(
                lambda value: allocation(value)[0],
                lower,
                upper,
                xtol=1e-11,
                rtol=1e-11,
                maxiter=120,
            )
    else:
        logit_human_share = brentq(
            lambda value: allocation(value)[0],
            lower,
            upper,
            xtol=1e-11,
            rtol=1e-11,
            maxiter=120,
        )
    _, block = allocation(logit_human_share)

    sigma = parameters.sigma_hm
    log_automated_research = (
        sigma * math.log(parameters.omega_m)
        + sigma * (block["log_research_price"] - math.log(parameters.xi))
        + block["log_effective_research"]
    )
    log_ai_services = block["log_production_labor"] + block["log_ai_ratio"]
    log_inference_compute = log_ai_services - log_capability
    log_capability_flow = (
        math.log(parameters.chi)
        + parameters.phi * log_capability
        + parameters.eta * block["log_effective_research"]
    )
    capability_growth = bounded_exp(log_capability_flow - log_capability)
    gross_capital_return = parameters.alpha * bounded_exp(
        block["log_output"] - log_capital
    )
    automated_share = logistic(
        math.log(parameters.xi)
        + log_automated_research
        - block["log_wage"]
        - block["log_human_research"]
    )
    inference_share = bounded_exp(
        math.log(parameters.xi)
        + log_inference_compute
        - block["log_output"]
    )
    research_resource_share = bounded_exp(
        math.log(parameters.xi)
        + log_automated_research
        - block["log_output"]
    )
    capability_profit_derivative = parameters.xi * bounded_exp(
        log_ai_services - 2.0 * log_capability
    )

    block.update(
        {
            "log_automated_research": log_automated_research,
            "log_ai_services": log_ai_services,
            "log_inference_compute": log_inference_compute,
            "capability_growth": capability_growth,
            "gross_capital_return": gross_capital_return,
            "automated_research_share": automated_share,
            "inference_share": inference_share,
            "research_resource_share": research_resource_share,
            "capability_profit_derivative": capability_profit_derivative,
        }
    )
    return block


def equilibrium_rates(
    time: float,
    state: Iterable[float],
    parameters: Parameters,
) -> tuple[np.ndarray, dict[str, float]]:
    log_capital, log_capability, log_consumption, log_shadow_value = map(
        float, state
    )
    log_population = parameters.n * time
    block = equilibrium_static_block(
        log_capital,
        log_capability,
        log_population,
        log_shadow_value,
        parameters,
    )

    output_capital_ratio = bounded_exp(block["log_output"] - log_capital)
    consumption_capital_ratio = bounded_exp(log_consumption - log_capital)
    inference_capital_ratio = bounded_exp(
        math.log(parameters.xi)
        + block["log_inference_compute"]
        - log_capital
    )
    research_capital_ratio = bounded_exp(
        math.log(parameters.xi)
        + block["log_automated_research"]
        - log_capital
    )
    capital_growth = (
        output_capital_ratio
        - consumption_capital_ratio
        - inference_capital_ratio
        - research_capital_ratio
        - parameters.delta
    )
    consumption_growth = (
        parameters.n
        + block["gross_capital_return"]
        - parameters.delta
        - parameters.discount
    )
    profit_shadow_ratio = block["capability_profit_derivative"] * bounded_exp(
        -log_shadow_value
    )
    shadow_growth = (
        block["gross_capital_return"]
        - parameters.delta
        - parameters.phi * block["capability_growth"]
        - profit_shadow_ratio
    )
    derivatives = np.asarray(
        [
            capital_growth,
            block["capability_growth"],
            consumption_growth,
            shadow_growth,
        ],
        dtype=float,
    )
    block.update(
        {
            "capital_growth": capital_growth,
            "consumption_growth": consumption_growth,
            "shadow_growth": shadow_growth,
            "log_population": log_population,
            "log_consumption": log_consumption,
            "log_shadow_value": log_shadow_value,
        }
    )
    return derivatives, block


def asymptotic_targets(parameters: Parameters) -> dict[str, float | str]:
    if parameters.sigma_xl < 1.0 - 1e-9:
        capability_growth = (
            parameters.eta
            * parameters.n
            / (1.0 - parameters.phi + parameters.eta)
        )
        research_input_growth = parameters.n - capability_growth
        investment_share = (
            parameters.alpha
            * (parameters.n + parameters.delta)
            / (parameters.discount + parameters.delta)
        )
        return {
            "aggregate_growth": parameters.n,
            "capability_growth": capability_growth,
            "shadow_growth": parameters.n - 2.0 * capability_growth,
            "consumption_share": 1.0 - investment_share,
            "terminal_shadow_object": "profit_shadow_ratio",
            "terminal_shadow_target": parameters.discount
            - (1.0 - parameters.eta) * research_input_growth,
        }
    if abs(parameters.sigma_xl - 1.0) <= 1e-9:
        beta = (1.0 - parameters.alpha) * parameters.omega_x
        upsilon = beta / (1.0 - parameters.alpha - beta)
        human_essential = abs(parameters.sigma_hm - 1.0) <= 1e-9
        research_feedback_weight = (
            parameters.omega_m if human_essential else 1.0
        )
        denominator = (
            1.0
            - parameters.phi
            - parameters.eta * research_feedback_weight * upsilon
        )
        if denominator <= 0.0:
            raise ValueError("The Cobb--Douglas asymptotic denominator is not positive.")
        capability_growth = parameters.eta * parameters.n / denominator
        per_capita_growth = upsilon * capability_growth
        research_share = (
            beta**2
            * parameters.eta
            * research_feedback_weight
            * capability_growth
            / (
                parameters.discount
                - parameters.n
                + (1.0 - parameters.phi) * capability_growth
            )
        )
        investment_share = (
            parameters.alpha
            * (
                parameters.n
                + per_capita_growth
                + parameters.delta
            )
            / (
                parameters.discount
                + parameters.delta
                + per_capita_growth
            )
        )
        return {
            "aggregate_growth": parameters.n + per_capita_growth,
            "capability_growth": capability_growth,
            "shadow_growth": (
                parameters.n + per_capita_growth - capability_growth
            ),
            "consumption_share": (
                1.0 - investment_share - beta**2 - research_share
            ),
            "terminal_shadow_object": "shadow_capability_output_ratio",
            "terminal_shadow_target": research_share
            / (
                parameters.eta
                * research_feedback_weight
                * capability_growth
            ),
        }
    raise ValueError(
        "No finite-rate terminal steady state exists for sigma_xl > 1 with "
        "unbounded capability; an additional asymptotic boundary condition is required."
    )


def fixed_share_guess(
    parameters: Parameters,
    initial_state: tuple[float, float, float],
    horizon: float,
    mesh: np.ndarray,
) -> np.ndarray:
    step = max(horizon / 250.0, 0.25)
    rows = mechanism.simulate(
        "equilibrium_guess",
        parameters,
        initial_state,
        horizon=horizon,
        step=step,
        acceleration_cutoff=5.0,
        catch_numerical_errors=True,
    )
    times = np.asarray([float(row["time"]) for row in rows])
    if times[-1] < 0.99 * horizon:
        raise ValueError("The mechanism path ended before the BVP horizon.")

    log_capital = np.interp(
        mesh, times, [float(row["log_capital"]) for row in rows]
    )
    log_capability = np.interp(
        mesh, times, [float(row["log_capability"]) for row in rows]
    )
    log_consumption = np.interp(
        mesh,
        times,
        [
            float(row["log_output"]) + math.log(float(row["consumption_share"]))
            for row in rows
        ],
    )

    log_shadow_values = []
    for row in rows:
        log_f_m = (
            math.log(parameters.chi)
            + math.log(parameters.eta)
            + parameters.phi * float(row["log_capability"])
            + (parameters.eta - 1.0) * float(row["log_effective_research"])
            + math.log(parameters.omega_m)
            + (
                float(row["log_effective_research"])
                - float(row["log_automated_research"])
            )
            / parameters.sigma_hm
        )
        log_shadow_values.append(math.log(parameters.xi) - log_f_m)
    log_shadow = np.interp(mesh, times, log_shadow_values)
    return np.vstack(
        [log_capital, log_capability, log_consumption, log_shadow]
    )


def solve_equilibrium(
    parameters: Parameters,
    initial_state: tuple[float, float, float],
    horizon: float,
    nodes: int = 201,
    tolerance: float = 2e-5,
) -> tuple[object, dict[str, float]]:
    targets = asymptotic_targets(parameters)
    mesh = np.linspace(0.0, horizon, nodes)
    growth_scales = np.asarray(
        [
            targets["aggregate_growth"],
            targets["capability_growth"],
            targets["aggregate_growth"],
            targets["shadow_growth"],
        ]
    )
    raw_guess = fixed_share_guess(parameters, initial_state, horizon, mesh)
    guess = raw_guess - growth_scales[:, None] * mesh[None, :]

    def ode(times: np.ndarray, states: np.ndarray) -> np.ndarray:
        values = np.empty_like(states)
        for index, time in enumerate(times):
            raw_state = states[:, index] + growth_scales * float(time)
            values[:, index] = equilibrium_rates(
                float(time), raw_state, parameters
            )[0] - growth_scales
        return values

    def boundary(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        terminal_raw = right + growth_scales * horizon
        _, terminal_block = equilibrium_rates(
            horizon, terminal_raw, parameters
        )
        terminal_consumption_share = math.exp(
            terminal_raw[2] - terminal_block["log_output"]
        )
        if targets["terminal_shadow_object"] == "profit_shadow_ratio":
            terminal_shadow_object = (
                terminal_block["capability_profit_derivative"]
                * bounded_exp(-terminal_raw[3])
            )
        else:
            terminal_shadow_object = math.exp(
                terminal_raw[3]
                + terminal_raw[1]
                - terminal_block["log_output"]
            )
        return np.asarray(
            [
                left[0] - math.log(initial_state[0]),
                left[1] - math.log(initial_state[1]),
                math.log(terminal_consumption_share)
                - math.log(targets["consumption_share"]),
                math.log(terminal_shadow_object)
                - math.log(targets["terminal_shadow_target"]),
            ]
        )

    solution = solve_bvp(
        ode,
        boundary,
        mesh,
        guess,
        tol=tolerance,
        max_nodes=8000,
        verbose=1,
    )
    scaled_solution = solution.sol

    def raw_solution(times: np.ndarray | float) -> np.ndarray:
        time_array = np.asarray(times)
        values = scaled_solution(times)
        if time_array.ndim == 0:
            return values + growth_scales * float(time_array)
        return values + growth_scales[:, None] * time_array[None, :]

    def raw_derivative(times: np.ndarray | float) -> np.ndarray:
        time_array = np.asarray(times)
        values = scaled_solution(times, 1)
        if time_array.ndim == 0:
            return values + growth_scales
        return values + growth_scales[:, None]

    solution.scaled_sol = scaled_solution
    solution.sol = raw_solution
    solution.calendar_derivative = raw_derivative
    solution.y = solution.y + growth_scales[:, None] * solution.x[None, :]
    return solution, targets


def solve_equilibrium_shooting(
    parameters: Parameters,
    initial_state: tuple[float, float, float],
    horizon: float,
    tolerance: float = 1e-8,
) -> tuple[object, dict[str, float]]:
    """Two-dimensional shooting in initial consumption and capability value."""

    targets = asymptotic_targets(parameters)
    jump_guess = fixed_share_guess(
        parameters, initial_state, horizon, np.asarray([0.0])
    )[[2, 3], 0]

    def integrate(jumps: np.ndarray, dense_output: bool = False) -> object:
        initial = np.asarray(
            [
                math.log(initial_state[0]),
                math.log(initial_state[1]),
                float(jumps[0]),
                float(jumps[1]),
            ]
        )
        return solve_ivp(
            lambda time, state: equilibrium_rates(
                float(time), state, parameters
            )[0],
            (0.0, horizon),
            initial,
            method="DOP853",
            rtol=3e-7,
            atol=3e-9,
            max_step=max(horizon / 300.0, 0.10),
            dense_output=dense_output,
        )

    def residual(jumps: np.ndarray) -> np.ndarray:
        solution = integrate(jumps)
        terminal = solution.y[:, -1]
        rates, _ = equilibrium_rates(
            float(solution.t[-1]), terminal, parameters
        )
        missing_horizon = max(0.0, horizon - float(solution.t[-1])) / horizon
        penalty = 10.0 * missing_horizon
        return np.asarray(
            [
                (rates[0] - targets["aggregate_growth"]) / 0.01 + penalty,
                (rates[2] - targets["aggregate_growth"]) / 0.01 + penalty,
            ]
        )

    initial_block = equilibrium_static_block(
        math.log(initial_state[0]),
        math.log(initial_state[1]),
        math.log(initial_state[2]),
        float(jump_guess[1]),
        parameters,
    )
    log_initial_output = initial_block["log_output"]
    lower_bounds = np.asarray(
        [log_initial_output + math.log(0.01), jump_guess[1] - 12.0]
    )
    upper_bounds = np.asarray(
        [log_initial_output + math.log(0.98), jump_guess[1] + 8.0]
    )
    root = least_squares(
        residual,
        jump_guess,
        bounds=(lower_bounds, upper_bounds),
        xtol=tolerance,
        ftol=tolerance,
        gtol=tolerance,
        max_nfev=150,
        verbose=1,
    )
    solution = integrate(root.x, dense_output=True)
    solution.root_result = root
    solution.success = bool(
        solution.success and np.max(np.abs(residual(root.x))) < 2e-4
    )
    return solution, targets


def evaluate_solution(
    name: str,
    solution: object,
    parameters: Parameters,
    horizon: float,
    step: float = 1.0,
) -> list[dict[str, float | str]]:
    times = np.arange(0.0, horizon + 0.5 * step, step)
    states = solution.sol(times)
    path_derivatives = (
        solution.calendar_derivative(times)
        if hasattr(solution, "calendar_derivative")
        else None
    )
    rows: list[dict[str, float | str]] = []
    for index, time in enumerate(times):
        derivatives, block = equilibrium_rates(
            float(time), states[:, index], parameters
        )
        log_capital, log_capability, log_consumption, log_shadow = map(
            float, states[:, index]
        )
        log_population = parameters.n * float(time)
        investment_share = (
            (derivatives[0] + parameters.delta)
            * math.exp(log_capital - block["log_output"])
        )
        consumption_share = math.exp(
            log_consumption - block["log_output"]
        )
        resource_share = (
            consumption_share
            + investment_share
            + block["inference_share"]
            + block["research_resource_share"]
        )
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
        row = {
            "scenario": name,
            "time": float(time),
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
            "capital_growth": derivatives[0],
            "capability_growth": derivatives[1],
            "consumption_growth": derivatives[2],
            "consumption_per_capita_growth": derivatives[2] - parameters.n,
            "shadow_growth": derivatives[3],
            "output_growth": math.nan,
            "output_per_capita_growth": math.nan,
            "wage_growth": math.nan,
            "gross_capital_return": block["gross_capital_return"],
            "net_capital_return": block["gross_capital_return"] - parameters.delta,
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
            "resource_share_sum": resource_share,
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
            "shadow_capability_to_capital": math.exp(
                log_shadow + log_capability - log_capital
            ),
            "human_to_automated_research_ratio": math.exp(
                block["log_human_research"]
                - block["log_automated_research"]
            ),
            "log_output_per_capita": block["log_output"] - log_population,
            "log_consumption_per_capita": log_consumption - log_population,
            "log_capital_per_capita": log_capital - log_population,
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
            "euler_residual": derivatives[2]
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

    # Replace the one-sided display derivative for output with centered numerical
    # derivatives.  The equilibrium equations themselves do not use this object.
    log_output = np.asarray([float(row["log_output"]) for row in rows])
    log_wage = np.asarray([float(row["log_wage"]) for row in rows])
    output_growth = np.gradient(log_output, times, edge_order=2)
    wage_growth = np.gradient(log_wage, times, edge_order=2)
    for row, growth, wage_rate in zip(rows, output_growth, wage_growth):
        row["output_growth"] = float(growth)
        row["output_per_capita_growth"] = float(growth - parameters.n)
        row["wage_growth"] = float(wage_rate)
    return rows


def write_rows(path: Path, rows: list[dict[str, float | str]]) -> None:
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def draw_equilibrium_figures(
    scenario_rows: dict[str, list[dict[str, float | str]]]
) -> None:
    display_rows = {
        key: [row for row in rows if float(row["time"]) <= 600.0]
        for key, rows in scenario_rows.items()
        if key in {"equilibrium_sigma_0_75", "equilibrium_sigma_1_00"}
    }
    labels = {
        "equilibrium_sigma_0_75": "sigma_XL = 0.75",
        "equilibrium_sigma_1_00": "sigma_XL = 1.00",
        "equilibrium_sigma_1_00_hm_1_00": "sigma_HM = 1",
    }
    palette = {
        "equilibrium_sigma_0_75": mechanism.COLORS["blue"],
        "equilibrium_sigma_1_00": mechanism.COLORS["orange"],
        "equilibrium_sigma_1_00_hm_1_00": mechanism.COLORS["olive"],
    }
    markers = {
        "equilibrium_sigma_0_75": "circle",
        "equilibrium_sigma_1_00": "square",
        "equilibrium_sigma_1_00_hm_1_00": "triangle",
    }
    log_change = lambda rows, values: values - values[0]
    per_capita_log_change = lambda rows, values: (
        values - np.asarray([float(row["log_population"]) for row in rows])
        - (
            values[0]
            - float(rows[0]["log_population"])
        )
    )
    log_level = lambda rows, values: np.log(values)
    percent = lambda rows, values: 100.0 * values

    mechanism.draw_multiplot(
        ROOT / "figures" / "equilibrium_levels.png",
        "Perfect-foresight equilibrium: quantities",
        "Natural-log change from each path's date-zero level; common initial K, A, and N",
        [
            {"title": "AI capability", "field": "log_capability", "transform": log_change},
            {"title": "Output per capita", "field": "log_output_per_capita", "transform": log_change},
            {"title": "Consumption per capita", "field": "log_consumption_per_capita", "transform": log_change},
            {"title": "Capital per capita", "field": "log_capital_per_capita", "transform": log_change},
        ],
        display_rows,
        labels,
        palette,
        markers,
    )
    mechanism.draw_multiplot(
        ROOT / "figures" / "equilibrium_growth_rates.png",
        "Equilibrium growth and returns",
        "Annual rates in percent; paths satisfy household and developer optimality",
        [
            {"title": "Capability growth", "field": "capability_growth", "transform": percent, "format": lambda value: f"{value:.1f}%"},
            {"title": "Output growth per capita", "field": "output_per_capita_growth", "transform": percent, "format": lambda value: f"{value:.1f}%", "reference_y": 0.0},
            {"title": "Consumption growth per capita", "field": "consumption_per_capita_growth", "transform": percent, "format": lambda value: f"{value:.1f}%", "reference_y": 0.0},
            {"title": "Net return to capital", "field": "net_capital_return", "transform": percent, "format": lambda value: f"{value:.1f}%"},
        ],
        display_rows,
        labels,
        palette,
        markers,
    )
    mechanism.draw_multiplot(
        ROOT / "figures" / "equilibrium_production_chain.png",
        "Perfect-foresight equilibrium: the production chain",
        "Natural-log change per capita, except for the AI-service price",
        [
            {"title": "Inference compute per capita", "field": "log_inference_compute", "transform": per_capita_log_change},
            {"title": "AI services per capita", "field": "log_ai_services", "transform": per_capita_log_change},
            {"title": "Service composite per capita", "field": "log_service_composite", "transform": per_capita_log_change},
            {"title": "Real AI-service price", "field": "log_ai_price", "transform": log_change},
        ],
        display_rows,
        labels,
        palette,
        markers,
    )
    mechanism.draw_multiplot(
        ROOT / "figures" / "equilibrium_research_chain.png",
        "Perfect-foresight equilibrium: AI research",
        "Natural-log change per capita; H/M is shown in natural logs",
        [
            {"title": "Human research per capita", "field": "log_human_research", "transform": per_capita_log_change},
            {"title": "Automated research per capita", "field": "log_automated_research", "transform": per_capita_log_change},
            {"title": "Effective research per capita", "field": "log_effective_research", "transform": per_capita_log_change},
            {"title": "ln human-machine research ratio", "field": "human_to_automated_research_ratio", "transform": log_level},
        ],
        display_rows,
        labels,
        palette,
        markers,
    )
    mechanism.draw_multiplot(
        ROOT / "figures" / "equilibrium_resource_allocation.png",
        "Perfect-foresight equilibrium: uses of output",
        "Shares of final output in percent; the four uses sum to one",
        [
            {"title": "Consumption / output", "field": "consumption_share", "transform": percent, "format": lambda value: f"{value:.0f}%"},
            {"title": "Investment / output", "field": "investment_share", "transform": percent, "format": lambda value: f"{value:.0f}%"},
            {"title": "Inference resources / output", "field": "inference_share", "transform": percent, "format": lambda value: f"{value:.1f}%"},
            {"title": "Automated research / output", "field": "research_resource_share", "transform": percent, "format": lambda value: f"{value:.2f}%"},
        ],
        display_rows,
        labels,
        palette,
        markers,
    )
    mechanism.draw_multiplot(
        ROOT / "figures" / "equilibrium_monopoly_block.png",
        "Perfect-foresight equilibrium: the integrated AI developer",
        "Natural-log price changes, markup ratio, and operating profits as a share of output",
        [
            {"title": "AI-service price", "field": "log_ai_price", "transform": log_change},
            {"title": "AI-service marginal cost", "field": "log_ai_marginal_cost", "transform": log_change},
            {"title": "Price / marginal cost", "field": "ai_markup"},
            {"title": "Operating profits / output", "field": "ai_profit_share", "transform": percent, "format": lambda value: f"{value:.1f}%"},
        ],
        display_rows,
        labels,
        palette,
        markers,
    )

    cobb_douglas_rows = {
        "equilibrium_sigma_1_00": scenario_rows["equilibrium_sigma_1_00"]
    }
    mechanism.draw_multiplot(
        ROOT / "figures" / "equilibrium_cobb_douglas_long_run.png",
        "Cobb-Douglas equilibrium: long-run transition",
        "Annual rates and shares in percent; the full 2,000-year solution is shown",
        [
            {"title": "Capability growth", "field": "capability_growth", "transform": percent, "format": lambda value: f"{value:.1f}%"},
            {"title": "Output growth per capita", "field": "output_per_capita_growth", "transform": percent, "format": lambda value: f"{value:.1f}%", "reference_y": 0.0},
            {"title": "Automated share of research", "field": "automated_research_share", "transform": percent, "format": lambda value: f"{value:.0f}%", "ylim": (0.0, 100.0)},
            {"title": "Human researchers / population", "field": "human_research_share", "transform": percent, "format": lambda value: f"{value:.2f}%"},
        ],
        cobb_douglas_rows,
        labels,
        palette,
        markers,
    )
    mechanism.draw_multiplot(
        ROOT / "figures" / "equilibrium_factor_shares.png",
        "Perfect-foresight equilibrium: labor and research allocation",
        "Shares in percent; production and aggregate labor shares use output as denominator",
        [
            {"title": "Production labor share", "field": "production_labor_share", "transform": percent, "format": lambda value: f"{value:.0f}%"},
            {"title": "Aggregate labor share", "field": "aggregate_labor_share", "transform": percent, "format": lambda value: f"{value:.0f}%"},
            {"title": "Human researchers / population", "field": "human_research_share", "transform": percent, "format": lambda value: f"{value:.1f}%"},
            {"title": "Automated share of research", "field": "automated_research_share", "transform": percent, "format": lambda value: f"{value:.0f}%"},
        ],
        display_rows,
        labels,
        palette,
        markers,
    )

    research_technology_rows = {
        "equilibrium_sigma_1_00_hm_1_00": scenario_rows[
            "equilibrium_sigma_1_00_hm_1_00"
        ],
        "equilibrium_sigma_1_00": scenario_rows["equilibrium_sigma_1_00"],
    }
    research_labels = {
        "equilibrium_sigma_1_00_hm_1_00": "Human-essential, sigma_HM = 1",
        "equilibrium_sigma_1_00": "Gross substitutes, sigma_HM = 2",
    }
    mechanism.draw_multiplot(
        ROOT / "figures" / "equilibrium_research_technology_comparison.png",
        "Research technology at Cobb-Douglas production",
        "Annual rates and shares in percent; both paths solve the canonical equilibrium system",
        [
            {"title": "Capability growth", "field": "capability_growth", "transform": percent, "format": lambda value: f"{value:.1f}%"},
            {"title": "Output growth per capita", "field": "output_per_capita_growth", "transform": percent, "format": lambda value: f"{value:.1f}%", "reference_y": 0.0},
            {"title": "Automated share of research", "field": "automated_research_share", "transform": percent, "format": lambda value: f"{value:.0f}%", "ylim": (0.0, 100.0)},
            {"title": "Human researchers / population", "field": "human_research_share", "transform": percent, "format": lambda value: f"{value:.2f}%"},
        ],
        research_technology_rows,
        research_labels,
        palette,
        markers,
    )


def main() -> None:
    RESULT_DIR.mkdir(exist_ok=True)
    baseline, analytical = mechanism.analytical_calibration(Parameters())
    initial_capital = mechanism.calibrate_initial_capital(
        baseline, analytical["capital_output_ratio"]
    )
    initial_state = (initial_capital, 1.0, 1.0)
    baseline = mechanism.calibrate_research_productivity(
        baseline, initial_state, analytical["capability_growth"]
    )

    summaries: list[dict[str, float | str]] = []
    all_rows: list[dict[str, float | str]] = []
    scenario_rows: dict[str, list[dict[str, float | str]]] = {}
    for name, sigma_xl, sigma_hm, horizon in [
        ("equilibrium_sigma_0_75", 0.75, 2.00, 600.0),
        ("equilibrium_sigma_1_00", 1.00, 2.00, 2000.0),
        ("equilibrium_sigma_1_00_hm_1_00", 1.00, 1.00, 2000.0),
    ]:
        parameters = replace(
            baseline,
            sigma_xl=sigma_xl,
            sigma_hm=sigma_hm,
        )
        solution, targets = solve_equilibrium(
            parameters,
            initial_state,
            horizon=horizon,
        )
        if not solution.success:
            raise RuntimeError(f"{name}: {solution.message}")
        rows = evaluate_solution(
            name,
            solution,
            parameters,
            horizon,
            step=1.0 if sigma_xl < 1.0 else 2.0,
        )
        scenario_rows[name] = rows
        all_rows.extend(rows)
        initial = rows[0]
        final = rows[-1]
        summaries.append(
            {
                "scenario": name,
                "sigma_xl": sigma_xl,
                "sigma_hm": sigma_hm,
                "horizon": horizon,
                "solver_status": solution.status,
                "solver_message": solution.message,
                "mesh_nodes": solution.x.size,
                "max_rms_residual": float(np.max(solution.rms_residuals)),
                "target_aggregate_growth": targets["aggregate_growth"],
                "target_capability_growth": targets["capability_growth"],
                "initial_consumption_share": initial["consumption_share"],
                "initial_capability_growth": initial["capability_growth"],
                "terminal_capital_growth": final["capital_growth"],
                "terminal_output_per_capita_growth": final[
                    "output_per_capita_growth"
                ],
                "terminal_consumption_per_capita_growth": final[
                    "consumption_per_capita_growth"
                ],
                "terminal_capability_growth": final["capability_growth"],
                "terminal_resource_share_sum": final["resource_share_sum"],
                "terminal_ai_share": final["ai_share"],
                "terminal_automated_research_share": final[
                    "automated_research_share"
                ],
                "minimum_consumption_share": min(
                    float(row["consumption_share"]) for row in rows
                ),
                "max_abs_euler_residual": max(
                    abs(float(row["euler_residual"])) for row in rows
                ),
                "max_abs_resource_residual": max(
                    abs(float(row["resource_share_sum"]) - 1.0)
                    for row in rows
                ),
                "max_abs_research_compute_foc_log_error": max(
                    abs(float(row["research_compute_foc_log_error"]))
                    for row in rows
                ),
                "max_abs_research_human_foc_log_error": max(
                    abs(float(row["research_human_foc_log_error"]))
                    for row in rows
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

    write_rows(RESULT_DIR / "equilibrium_transition_paths.csv", all_rows)
    write_rows(RESULT_DIR / "equilibrium_transition_summary.csv", summaries)
    draw_equilibrium_figures(scenario_rows)


if __name__ == "__main__":
    main()
