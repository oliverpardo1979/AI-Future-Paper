"""Browser-facing equilibrium solver for the AI Future Paper.

The module is loaded by Pyodide inside a Web Worker. It mirrors the canonical
system used by the paper's replication scripts. The result is an interior
canonical equilibrium branch conditional on convergence and the selected
terminal restrictions; it is not a certificate of global dynamic optimality.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from scipy.integrate import solve_bvp
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq


@dataclass(frozen=True)
class Parameters:
    alpha: float = 0.33
    omega_x: float = 0.20
    sigma_xl: float = 1.00
    n: float = 0.012
    delta: float = 0.05
    discount: float = 0.04
    xi: float = 1.00
    omega_m: float = 0.35
    sigma_hm: float = 2.00
    phi: float = 0.65
    eta: float = 0.45
    chi: float = 0.0983886742632607
    capital_initial: float = 2.5093528013468305
    capability_initial: float = 1.0
    population_initial: float = 1.0
    horizon: float = 2000.0
    terminal_z: float = 0.2899267966577116
    points: int = 181
    tolerance: float = 5e-5
    high_method: str = "fixed_horizon"

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "Parameters":
        allowed = set(cls.__dataclass_fields__)
        clean = {key: values[key] for key in values if key in allowed}
        clean["points"] = int(clean.get("points", cls.points))
        return cls(**clean)


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
    return math.exp(min(max(value, -700.0), upper))


def validate(parameters: Parameters) -> list[str]:
    errors: list[str] = []
    for name in (
        "alpha",
        "omega_x",
        "omega_m",
        "eta",
    ):
        value = getattr(parameters, name)
        if not 0.0 < value < 1.0:
            errors.append(f"{name} must lie strictly between zero and one.")
    if parameters.sigma_xl <= 0.0:
        errors.append("sigma_xl must be positive.")
    if parameters.sigma_hm < 1.0:
        errors.append("This version requires sigma_hm >= 1.")
    if parameters.phi >= 1.0:
        errors.append("This version requires phi < 1.")
    if parameters.discount <= parameters.n:
        errors.append("The household problem requires discount > n.")
    for name in (
        "delta",
        "discount",
        "xi",
        "chi",
        "capital_initial",
        "capability_initial",
        "population_initial",
        "horizon",
        "terminal_z",
        "tolerance",
    ):
        if getattr(parameters, name) <= 0.0:
            errors.append(f"{name} must be positive.")
    if not 61 <= parameters.points <= 401:
        errors.append("points must be between 61 and 401.")
    d_cd = (
        1.0
        - parameters.phi
        - parameters.eta
        * parameters.omega_m
        * parameters.omega_x
        / (1.0 - parameters.omega_x)
    )
    if d_cd <= 0.0:
        errors.append("The maintained paper restriction D_CD > 0 is violated.")
    if parameters.sigma_xl > 1.0 + 1e-8:
        if parameters.sigma_xl >= 1.0 / parameters.alpha:
            errors.append(
                "The current singular boundary requires 1 < sigma_xl < 1/alpha."
            )
        if parameters.high_method not in {"fixed_horizon", "free_boundary"}:
            errors.append("high_method must be fixed_horizon or free_boundary.")
    return errors


def research_unit_cost(log_wage: float, p: Parameters) -> float:
    sigma = p.sigma_hm
    if abs(sigma - 1.0) <= 1e-10:
        omega_m = p.omega_m
        omega_h = 1.0 - omega_m
        return (
            omega_h * (log_wage - math.log(omega_h))
            + omega_m * (math.log(p.xi) - math.log(omega_m))
        )
    log_human_term = (
        sigma * math.log1p(-p.omega_m) + (1.0 - sigma) * log_wage
    )
    log_machine_term = (
        sigma * math.log(p.omega_m)
        + (1.0 - sigma) * math.log(p.xi)
    )
    return logsumexp_pair(log_human_term, log_machine_term) / (1.0 - sigma)


def monopoly_service_block(
    log_capital: float,
    log_labor: float,
    log_capability: float,
    p: Parameters,
) -> dict[str, float]:
    if abs(p.sigma_xl - 1.0) <= 1e-10:
        beta = (1.0 - p.alpha) * p.omega_x
        log_ai_services = (
            2.0 * math.log(beta)
            + log_capability
            - math.log(p.xi)
            + p.alpha * log_capital
            + (1.0 - p.alpha) * (1.0 - p.omega_x) * log_labor
        ) / (1.0 - beta)
        log_output = (
            p.alpha * log_capital
            + (1.0 - p.alpha) * (1.0 - p.omega_x) * log_labor
            + beta * log_ai_services
        )
        return {
            "log_ai_ratio": log_ai_services - log_labor,
            "log_output": log_output,
            "ai_share": p.omega_x,
        }

    rho = (p.sigma_xl - 1.0) / p.sigma_xl

    def quantities(log_ratio: float) -> tuple[float, float]:
        log_human_term = math.log1p(-p.omega_x)
        log_ai_term = math.log(p.omega_x) + rho * log_ratio
        log_denominator = logsumexp_pair(log_human_term, log_ai_term)
        log_z_per_worker = log_denominator / rho
        ai_share = math.exp(log_ai_term - log_denominator)
        log_output = p.alpha * log_capital + (1.0 - p.alpha) * (
            log_labor + log_z_per_worker
        )
        return log_output, min(max(ai_share, 1e-14), 1.0 - 1e-14)

    def residual(log_ratio: float) -> float:
        log_output, ai_share = quantities(log_ratio)
        inverse_elasticity = (
            (1.0 - ai_share) / p.sigma_xl + p.alpha * ai_share
        )
        markup_term = 1.0 - inverse_elasticity
        if markup_term <= 0.0:
            return -1e6
        log_price = (
            math.log1p(-p.alpha)
            + math.log(ai_share)
            + log_output
            - log_labor
            - log_ratio
        )
        return log_price + math.log(markup_term) - math.log(p.xi) + log_capability

    grid = np.linspace(-500.0, 500.0, 121)
    previous_x = float(grid[0])
    previous_value = residual(previous_x)
    best = (abs(previous_value), previous_x)
    for value in grid[1:]:
        current_x = float(value)
        current_value = residual(current_x)
        if abs(current_value) < best[0]:
            best = (abs(current_value), current_x)
        if previous_value * current_value <= 0.0:
            log_ai_ratio = brentq(
                residual,
                previous_x,
                current_x,
                xtol=1e-10,
                rtol=1e-10,
                maxiter=100,
            )
            log_output, ai_share = quantities(log_ai_ratio)
            return {
                "log_ai_ratio": log_ai_ratio,
                "log_output": log_output,
                "ai_share": ai_share,
            }
        previous_x, previous_value = current_x, current_value
    log_output, ai_share = quantities(best[1])
    return {
        "log_ai_ratio": best[1],
        "log_output": log_output,
        "ai_share": ai_share,
    }


def static_block(
    log_capital: float,
    log_capability: float,
    log_population: float,
    log_shadow_value: float,
    p: Parameters,
) -> dict[str, float]:
    def allocation(logit_human_share: float) -> tuple[float, dict[str, float]]:
        human_share = min(
            max(logistic(logit_human_share), 1e-14), 1.0 - 1e-14
        )
        log_human_research = log_population + math.log(human_share)
        log_production_labor = log_population + math.log1p(-human_share)
        production = monopoly_service_block(
            log_capital, log_production_labor, log_capability, p
        )
        log_output = production["log_output"]
        ai_share = production["ai_share"]
        log_wage = (
            math.log1p(-p.alpha)
            + math.log1p(-ai_share)
            + log_output
            - log_production_labor
        )
        log_research_price = research_unit_cost(log_wage, p)
        log_effective_research = (
            log_shadow_value
            + math.log(p.chi)
            + math.log(p.eta)
            + p.phi * log_capability
            - log_research_price
        ) / (1.0 - p.eta)
        log_human_demand = (
            p.sigma_hm * math.log1p(-p.omega_m)
            + p.sigma_hm * (log_research_price - log_wage)
            + log_effective_research
        )
        return log_human_research - log_human_demand, {
            "human_share": human_share,
            "log_human_research": log_human_research,
            "log_production_labor": log_production_labor,
            "log_ai_ratio": production["log_ai_ratio"],
            "log_output": log_output,
            "ai_share": ai_share,
            "log_wage": log_wage,
            "log_research_price": log_research_price,
            "log_effective_research": log_effective_research,
        }

    grid = np.linspace(-60.0, 60.0, 121)
    previous_x = float(grid[0])
    previous_value = allocation(previous_x)[0]
    bracket = None
    best = (abs(previous_value), previous_x)
    for value in grid[1:]:
        current_x = float(value)
        current_value = allocation(current_x)[0]
        if abs(current_value) < best[0]:
            best = (abs(current_value), current_x)
        if previous_value * current_value <= 0.0:
            bracket = (previous_x, current_x)
            break
        previous_x, previous_value = current_x, current_value
    if bracket is None:
        logit_human_share = best[1]
    else:
        logit_human_share = brentq(
            lambda value: allocation(value)[0],
            bracket[0],
            bracket[1],
            xtol=1e-10,
            rtol=1e-10,
            maxiter=100,
        )
    research_foc_residual, block = allocation(logit_human_share)
    sigma = p.sigma_hm
    log_automated_research = (
        sigma * math.log(p.omega_m)
        + sigma * (block["log_research_price"] - math.log(p.xi))
        + block["log_effective_research"]
    )
    log_ai_services = block["log_production_labor"] + block["log_ai_ratio"]
    log_inference_compute = log_ai_services - log_capability
    log_capability_flow = (
        math.log(p.chi)
        + p.phi * log_capability
        + p.eta * block["log_effective_research"]
    )
    block.update(
        {
            "log_automated_research": log_automated_research,
            "log_ai_services": log_ai_services,
            "log_inference_compute": log_inference_compute,
            "capability_growth": bounded_exp(log_capability_flow - log_capability),
            "gross_capital_return": p.alpha
            * bounded_exp(block["log_output"] - log_capital),
            "automated_research_share": logistic(
                math.log(p.xi)
                + log_automated_research
                - block["log_wage"]
                - block["log_human_research"]
            ),
            "inference_share": bounded_exp(
                math.log(p.xi)
                + log_inference_compute
                - block["log_output"]
            ),
            "research_resource_share": bounded_exp(
                math.log(p.xi)
                + log_automated_research
                - block["log_output"]
            ),
            "capability_profit_derivative": p.xi
            * bounded_exp(log_ai_services - 2.0 * log_capability),
            "research_foc_residual": research_foc_residual,
        }
    )
    return block


def rates(
    time: float, state: Iterable[float], p: Parameters
) -> tuple[np.ndarray, dict[str, float]]:
    log_capital, log_capability, log_consumption, log_shadow = map(float, state)
    log_population = math.log(p.population_initial) + p.n * time
    block = static_block(
        log_capital, log_capability, log_population, log_shadow, p
    )
    capital_growth = (
        bounded_exp(block["log_output"] - log_capital)
        - bounded_exp(log_consumption - log_capital)
        - bounded_exp(math.log(p.xi) + block["log_inference_compute"] - log_capital)
        - bounded_exp(math.log(p.xi) + block["log_automated_research"] - log_capital)
        - p.delta
    )
    consumption_growth = (
        p.n + block["gross_capital_return"] - p.delta - p.discount
    )
    profit_shadow_ratio = block["capability_profit_derivative"] * bounded_exp(
        -log_shadow
    )
    shadow_growth = (
        block["gross_capital_return"]
        - p.delta
        - p.phi * block["capability_growth"]
        - profit_shadow_ratio
    )
    derivatives = np.asarray(
        [capital_growth, block["capability_growth"], consumption_growth, shadow_growth]
    )
    block.update(
        {
            "capital_growth": capital_growth,
            "consumption_growth": consumption_growth,
            "shadow_growth": shadow_growth,
            "log_population": log_population,
            "log_consumption": log_consumption,
            "log_shadow_value": log_shadow,
        }
    )
    return derivatives, block


def finite_targets(p: Parameters) -> dict[str, float | str]:
    if p.sigma_xl < 1.0 - 1e-8:
        capability_growth = p.eta * p.n / (1.0 - p.phi + p.eta)
        research_growth = p.n - capability_growth
        investment_share = p.alpha * (p.n + p.delta) / (p.discount + p.delta)
        return {
            "aggregate_growth": p.n,
            "capability_growth": capability_growth,
            "shadow_growth": p.n - 2.0 * capability_growth,
            "consumption_share": 1.0 - investment_share,
            "terminal_shadow_object": "profit_shadow_ratio",
            "terminal_shadow_target": p.discount - (1.0 - p.eta) * research_growth,
        }
    beta = (1.0 - p.alpha) * p.omega_x
    upsilon = beta / (1.0 - p.alpha - beta)
    human_essential = abs(p.sigma_hm - 1.0) <= 1e-9
    research_feedback_weight = p.omega_m if human_essential else 1.0
    denominator = (
        1.0 - p.phi - p.eta * research_feedback_weight * upsilon
    )
    if denominator <= 0.0:
        raise ValueError("The Cobb-Douglas asymptotic denominator is not positive.")
    capability_growth = p.eta * p.n / denominator
    per_capita_growth = upsilon * capability_growth
    research_share = (
        beta**2
        * p.eta
        * research_feedback_weight
        * capability_growth
        / (p.discount - p.n + (1.0 - p.phi) * capability_growth)
    )
    investment_share = p.alpha * (
        p.n + per_capita_growth + p.delta
    ) / (p.discount + p.delta + per_capita_growth)
    consumption_share = 1.0 - investment_share - beta**2 - research_share
    if consumption_share <= 0.0:
        raise ValueError("The Cobb-Douglas terminal consumption share is not positive.")
    return {
        "aggregate_growth": p.n + per_capita_growth,
        "capability_growth": capability_growth,
        "shadow_growth": p.n + per_capita_growth - capability_growth,
        "consumption_share": consumption_share,
        "terminal_shadow_object": "shadow_capability_output_ratio",
        "terminal_shadow_target": research_share
        / (p.eta * research_feedback_weight * capability_growth),
    }


def initial_shadow_guess(p: Parameters, human_share: float = 0.01) -> float:
    log_k = math.log(p.capital_initial)
    log_a = math.log(p.capability_initial)
    log_n = math.log(p.population_initial)
    log_h = log_n + math.log(human_share)
    log_l = log_n + math.log1p(-human_share)
    production = monopoly_service_block(log_k, log_l, log_a, p)
    log_wage = (
        math.log1p(-p.alpha)
        + math.log1p(-production["ai_share"])
        + production["log_output"]
        - log_l
    )


def research_aggregate(
    log_human: float, log_machine: float, p: Parameters
) -> tuple[float, float]:
    if abs(p.sigma_hm - 1.0) <= 1e-10:
        return (
            (1.0 - p.omega_m) * log_human + p.omega_m * log_machine,
            p.omega_m,
        )
    rho = (p.sigma_hm - 1.0) / p.sigma_hm
    log_human_term = math.log1p(-p.omega_m) + rho * log_human
    log_machine_term = math.log(p.omega_m) + rho * log_machine
    log_sum = logsumexp_pair(log_human_term, log_machine_term)
    return log_sum / rho, math.exp(log_machine_term - log_sum)


def mechanism_guess(
    p: Parameters, targets: dict[str, float | str], mesh: np.ndarray
) -> np.ndarray:
    """Generate a nearby feasible path used only as a BVP initial guess."""

    investment_share = min(
        max(p.alpha * (p.n + p.delta) / (p.discount + p.delta), 0.05),
        0.70,
    )
    research_share = min(0.008, 0.15 * (1.0 - investment_share))

    def block(state: np.ndarray) -> dict[str, float]:
        log_k, log_a, log_n = map(float, state)

        def labor_residual(logit_h: float) -> tuple[float, dict[str, float]]:
            human_share = min(max(logistic(logit_h), 1e-12), 1.0 - 1e-12)
            log_h = log_n + math.log(human_share)
            log_l = log_n + math.log1p(-human_share)
            production = monopoly_service_block(log_k, log_l, log_a, p)
            log_y = production["log_output"]
            log_w = (
                math.log1p(-p.alpha)
                + math.log1p(-production["ai_share"])
                + log_y
                - log_l
            )
            log_m = math.log(research_share) + log_y - math.log(p.xi)
            target_ratio = p.sigma_hm * (
                math.log1p(-p.omega_m)
                - math.log(p.omega_m)
                + math.log(p.xi)
                - log_w
            )
            return log_h - log_m - target_ratio, {
                "log_h": log_h,
                "log_l": log_l,
                "log_m": log_m,
                "log_y": log_y,
                "log_w": log_w,
                "log_x": log_l + production["log_ai_ratio"],
            }

        grid = np.linspace(-40.0, 40.0, 81)
        previous_x = float(grid[0])
        previous_value = labor_residual(previous_x)[0]
        bracket = None
        best = (abs(previous_value), previous_x)
        for value in grid[1:]:
            current_x = float(value)
            current_value = labor_residual(current_x)[0]
            if abs(current_value) < best[0]:
                best = (abs(current_value), current_x)
            if previous_value * current_value <= 0.0:
                bracket = (previous_x, current_x)
                break
            previous_x, previous_value = current_x, current_value
        if bracket is None:
            logit_h = best[1]
        else:
            logit_h = brentq(
                lambda value: labor_residual(value)[0],
                bracket[0],
                bracket[1],
                xtol=1e-8,
                rtol=1e-8,
            )
        _, values = labor_residual(logit_h)
        log_e, _ = research_aggregate(values["log_h"], values["log_m"], p)
        log_u = values["log_x"] - log_a
        consumption_share = (
            1.0
            - investment_share
            - research_share
            - math.exp(math.log(p.xi) + log_u - values["log_y"])
        )
        values.update(
            {
                "log_e": log_e,
                "log_u": log_u,
                "consumption_share": max(consumption_share, 0.02),
            }
        )
        return values

    def state_rates(state: np.ndarray) -> np.ndarray:
        values = block(state)
        return np.asarray(
            [
                investment_share * math.exp(values["log_y"] - state[0]) - p.delta,
                p.chi
                * math.exp((p.phi - 1.0) * state[1] + p.eta * values["log_e"]),
                p.n,
            ]
        )

    step = max(p.horizon / 300.0, 0.25)
    state = np.log(
        np.asarray([p.capital_initial, p.capability_initial, p.population_initial])
    )
    times = [0.0]
    states = [state.copy()]
    current = 0.0
    while current < p.horizon - 1e-12:
        dt = min(step, p.horizon - current)
        k1 = state_rates(state)
        k2 = state_rates(state + 0.5 * dt * k1)
        k3 = state_rates(state + 0.5 * dt * k2)
        k4 = state_rates(state + dt * k3)
        state = state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        current += dt
        times.append(current)
        states.append(state.copy())

    states_array = np.asarray(states)
    log_k = np.interp(mesh, times, states_array[:, 0])
    log_a = np.interp(mesh, times, states_array[:, 1])
    log_c_values: list[float] = []
    log_q_values: list[float] = []
    for state_value in states:
        values = block(state_value)
        log_c_values.append(
            values["log_y"] + math.log(values["consumption_share"])
        )
        log_e = values["log_e"]
        log_f_m = (
            math.log(p.chi)
            + math.log(p.eta)
            + p.phi * state_value[1]
            + (p.eta - 1.0) * log_e
            + math.log(p.omega_m)
            + (log_e - values["log_m"]) / p.sigma_hm
        )
        log_q_values.append(math.log(p.xi) - log_f_m)
    log_c = np.interp(mesh, times, log_c_values)
    log_q = np.interp(mesh, times, log_q_values)
    return np.vstack([log_k, log_a, log_c, log_q])
    log_price = research_unit_cost(log_wage, p)
    log_effective = (
        log_h
        - p.sigma_hm * math.log1p(-p.omega_m)
        - p.sigma_hm * (log_price - log_wage)
    )
    return (
        (1.0 - p.eta) * log_effective
        - math.log(p.chi)
        - math.log(p.eta)
        - p.phi * log_a
        + log_price
    )


def solve_finite(p: Parameters) -> tuple[Any, dict[str, float | str], float]:
    targets = finite_targets(p)
    mesh = np.linspace(0.0, p.horizon, max(61, min(p.points, 161)))
    growth = np.asarray(
        [
            targets["aggregate_growth"],
            targets["capability_growth"],
            targets["aggregate_growth"],
            targets["shadow_growth"],
        ],
        dtype=float,
    )
    guess = mechanism_guess(p, targets, mesh)
    detrended_guess = guess - growth[:, None] * mesh[None, :]

    def ode(times: np.ndarray, states: np.ndarray) -> np.ndarray:
        values = np.empty_like(states)
        for index, time in enumerate(times):
            raw = states[:, index] + growth * float(time)
            values[:, index] = rates(float(time), raw, p)[0] - growth
        return values

    def boundary(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        terminal = right + growth * p.horizon
        _, block = rates(p.horizon, terminal, p)
        consumption_share = math.exp(terminal[2] - block["log_output"])
        if targets["terminal_shadow_object"] == "profit_shadow_ratio":
            shadow_object = block["capability_profit_derivative"] * bounded_exp(
                -terminal[3]
            )
        else:
            shadow_object = math.exp(
                terminal[3] + terminal[1] - block["log_output"]
            )
        return np.asarray(
            [
                left[0] - math.log(p.capital_initial),
                left[1] - math.log(p.capability_initial),
                math.log(consumption_share) - math.log(targets["consumption_share"]),
                math.log(shadow_object) - math.log(targets["terminal_shadow_target"]),
            ]
        )

    solution = solve_bvp(
        ode,
        boundary,
        mesh,
        detrended_guess,
        tol=p.tolerance,
        max_nodes=6000,
        verbose=0,
    )
    scaled_solution = solution.sol

    def raw_solution(times: np.ndarray | float) -> np.ndarray:
        time_array = np.asarray(times)
        values = scaled_solution(times)
        if time_array.ndim == 0:
            return values + growth * float(time_array)
        return values + growth[:, None] * time_array[None, :]

    def raw_derivative(times: np.ndarray | float) -> np.ndarray:
        time_array = np.asarray(times)
        values = scaled_solution(times, 1)
        if time_array.ndim == 0:
            return values + growth
        return values + growth[:, None]

    solution.calendar_sol = raw_solution
    solution.calendar_derivative = raw_derivative
    return solution, targets, p.horizon


def high_targets(p: Parameters) -> dict[str, float]:
    kappa = (1.0 - p.alpha) / p.alpha
    denominator = 1.0 - p.phi + kappa
    inference_share = (1.0 - p.alpha) ** 2
    capability_growth_to_z = p.eta * p.alpha / denominator
    investment_share = p.alpha - kappa * capability_growth_to_z
    research_share = p.eta * inference_share / denominator
    consumption_share = 1.0 - inference_share - investment_share - research_share
    shadow_ratio = inference_share / (p.eta * p.alpha)
    if min(investment_share, research_share, consumption_share) <= 0.0:
        raise ValueError("The singular-boundary resource shares are not positive.")
    return {
        "kappa": kappa,
        "inference_share": inference_share,
        "capability_growth_to_z": capability_growth_to_z,
        "investment_share": investment_share,
        "research_share": research_share,
        "consumption_share": consumption_share,
        "shadow_capability_to_capital": shadow_ratio,
        "singularity_rate": kappa * capability_growth_to_z,
    }


def high_initial_guess(
    p: Parameters, targets: dict[str, float], terminal_z: float, duration: float, mesh: np.ndarray
) -> np.ndarray:
    exponent = targets["kappa"]
    log_coefficient = exponent * (
        2.0 * math.log1p(-p.alpha)
        - math.log(p.xi)
        + p.sigma_xl / (p.sigma_xl - 1.0) * math.log(p.omega_x)
    )
    log_terminal_a = max(
        math.log(p.capability_initial) + 0.5,
        (math.log(terminal_z) - log_coefficient) / exponent,
    )
    capital_elasticity = targets["investment_share"] / targets[
        "capability_growth_to_z"
    ]
    log_terminal_k = math.log(p.capital_initial) + capital_elasticity * (
        log_terminal_a - math.log(p.capability_initial)
    )
    curvature = 0.90
    profile = -np.log1p(-curvature * mesh) / -math.log1p(-curvature)
    log_a = math.log(p.capability_initial) + (
        log_terminal_a - math.log(p.capability_initial)
    ) * profile
    log_k = math.log(p.capital_initial) + (
        log_terminal_k - math.log(p.capital_initial)
    ) * profile
    log_q = math.log(targets["shadow_capability_to_capital"]) + log_k - log_a
    log_c = np.empty_like(mesh)
    for index, tau in enumerate(mesh):
        block = static_block(
            float(log_k[index]),
            float(log_a[index]),
            math.log(p.population_initial) + p.n * duration * float(tau),
            float(log_q[index]),
            p,
        )
        log_c[index] = block["log_output"] + math.log(targets["consumption_share"])
    return np.vstack([log_k, log_a, log_c, log_q])


def solve_high_once(
    p: Parameters,
    terminal_z: float,
    duration_guess: float,
    previous: Any | None = None,
) -> tuple[Any, dict[str, float], float]:
    targets = high_targets(p)
    mesh = np.linspace(0.0, 1.0, max(71, min(p.points, 141)))
    if previous is None:
        guess = high_initial_guess(p, targets, terminal_z, duration_guess, mesh)
    else:
        guess = previous.calendar_normalized_sol(mesh)
        duration_guess = previous.duration
        _, old_block = rates(duration_guess, guess[:, -1], p)
        old_z = math.exp(old_block["log_output"] - guess[0, -1])
        extension = max(0.0, math.log(terminal_z / old_z))
        if extension:
            increment_a = extension / targets["kappa"]
            profile = mesh**4
            capital_elasticity = targets["investment_share"] / targets[
                "capability_growth_to_z"
            ]
            guess[1] += increment_a * profile
            guess[0] += capital_elasticity * increment_a * profile
            guess[2] += (capital_elasticity * increment_a + extension) * profile
            guess[3] += (capital_elasticity - 1.0) * increment_a * profile
            duration_guess += (
                (1.0 / old_z - 1.0 / terminal_z) / targets["singularity_rate"]
            )

    splines = [CubicSpline(mesh, guess[index], bc_type="natural") for index in range(4)]

    def reference(tau: np.ndarray | float) -> np.ndarray:
        return np.asarray([spline(tau) for spline in splines])

    def reference_derivative(tau: np.ndarray | float) -> np.ndarray:
        return np.asarray([spline(tau, 1) for spline in splines])

    def ode(tau: np.ndarray, state: np.ndarray, log_duration: np.ndarray) -> np.ndarray:
        duration = bounded_exp(float(log_duration[0]), upper=10.0)
        values = np.empty_like(state)
        for index, normalized_time in enumerate(tau):
            raw = state[:, index] + reference(float(normalized_time))
            values[:, index] = duration * rates(
                duration * float(normalized_time), raw, p
            )[0] - reference_derivative(float(normalized_time))
        return values

    def boundary(left: np.ndarray, right: np.ndarray, log_duration: np.ndarray) -> np.ndarray:
        duration = bounded_exp(float(log_duration[0]), upper=10.0)
        raw_left = left + reference(0.0)
        raw_right = right + reference(1.0)
        _, block = rates(duration, raw_right, p)
        return np.asarray(
            [
                raw_left[0] - math.log(p.capital_initial),
                raw_left[1] - math.log(p.capability_initial),
                raw_right[2]
                - block["log_output"]
                - math.log(targets["consumption_share"]),
                raw_right[3]
                + raw_right[1]
                - raw_right[0]
                - math.log(targets["shadow_capability_to_capital"]),
                block["log_output"] - raw_right[0] - math.log(terminal_z),
            ]
        )

    solution = solve_bvp(
        ode,
        boundary,
        mesh,
        np.zeros_like(guess),
        p=np.asarray([math.log(duration_guess)]),
        tol=p.tolerance,
        max_nodes=8000,
        verbose=0,
    )
    scaled = solution.sol

    def normalized_solution(tau: np.ndarray | float) -> np.ndarray:
        return scaled(tau) + reference(tau)

    def normalized_derivative(tau: np.ndarray | float) -> np.ndarray:
        return scaled(tau, 1) + reference_derivative(tau)

    solution.duration = math.exp(float(solution.p[0]))
    solution.calendar_normalized_sol = normalized_solution
    solution.calendar_sol = lambda times: normalized_solution(
        np.asarray(times) / solution.duration
    )
    solution.calendar_derivative = lambda times: normalized_derivative(
        np.asarray(times) / solution.duration
    ) / solution.duration
    solution.terminal_z = terminal_z
    return solution, targets, solution.duration


def solve_high(p: Parameters) -> tuple[Any, dict[str, float], float]:
    targets = high_targets(p)
    duration_guess = min(max(p.horizon, 40.0), 1500.0)
    target_sequence = sorted(
        set(value for value in (2.0, 5.0, 10.0, p.terminal_z) if value <= p.terminal_z)
    )
    previous = None
    solution = None
    for target in target_sequence:
        solution, _, _ = solve_high_once(
            p, target, duration_guess, previous=previous
        )
        if not solution.success:
            if target != p.terminal_z:
                continue
            break
        previous = solution
        duration_guess = solution.duration
    if solution is None:
        raise RuntimeError("The high-substitution continuation did not start.")
    solution.estimated_singularity_time = solution.duration + 1.0 / (
        targets["singularity_rate"] * p.terminal_z
    )
    return solution, targets, solution.duration


def warm_high_guess(
    p: Parameters,
    warm_start: list[dict[str, Any]],
    mesh: np.ndarray,
) -> np.ndarray | None:
    """Map a published canonical path to a custom fixed horizon."""

    required = (
        "time",
        "log_capital",
        "log_capability",
        "log_consumption",
        "log_shadow_value",
    )
    clean = [
        row for row in warm_start
        if all(key in row and math.isfinite(float(row[key])) for key in required)
    ]
    if len(clean) < 4:
        return None
    source_time = np.asarray([float(row["time"]) for row in clean])
    if source_time[-1] <= source_time[0]:
        return None
    source_tau = (source_time - source_time[0]) / (
        source_time[-1] - source_time[0]
    )
    target_tau = mesh / p.horizon
    fields = (
        "log_capital",
        "log_capability",
        "log_consumption",
        "log_shadow_value",
    )
    guess = np.vstack(
        [
            np.interp(
                target_tau,
                source_tau,
                np.asarray([float(row[field]) for row in clean]),
            )
            for field in fields
        ]
    )
    capital_shift = math.log(p.capital_initial) - guess[0, 0]
    capability_shift = math.log(p.capability_initial) - guess[1, 0]
    population_shift = math.log(p.population_initial)
    guess[0] += capital_shift
    guess[1] += capability_shift
    guess[2] += population_shift
    guess[3] += capital_shift - capability_shift
    return guess


def solve_high_fixed(
    p: Parameters,
    warm_start: list[dict[str, Any]] | None = None,
) -> tuple[Any, dict[str, Any], float]:
    """Solve a faster truncated high-substitution branch at a fixed horizon."""

    targets: dict[str, Any] = high_targets(p)
    targets["boundary_mode"] = "fixed_horizon"
    mesh = np.linspace(0.0, p.horizon, max(71, min(p.points, 161)))
    guess = warm_high_guess(p, warm_start or [], mesh)
    if guess is None:
        guess = high_initial_guess(
            p, targets, p.terminal_z, p.horizon, mesh / p.horizon
        )

    def ode(times: np.ndarray, states: np.ndarray) -> np.ndarray:
        values = np.empty_like(states)
        for index, time in enumerate(times):
            values[:, index] = rates(float(time), states[:, index], p)[0]
        return values

    def boundary(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        _, block = rates(p.horizon, right, p)
        return np.asarray(
            [
                left[0] - math.log(p.capital_initial),
                left[1] - math.log(p.capability_initial),
                right[2]
                - block["log_output"]
                - math.log(targets["consumption_share"]),
                right[3]
                + right[1]
                - right[0]
                - math.log(targets["shadow_capability_to_capital"]),
            ]
        )

    solution = solve_bvp(
        ode,
        boundary,
        mesh,
        guess,
        tol=p.tolerance,
        max_nodes=7000,
        verbose=0,
    )
    solution.duration = p.horizon
    solution.calendar_sol = solution.sol
    solution.calendar_derivative = lambda times: solution.sol(times, 1)
    terminal = solution.sol(p.horizon)
    _, terminal_block = rates(p.horizon, terminal, p)
    terminal_z = math.exp(terminal_block["log_output"] - terminal[0])
    solution.terminal_z = terminal_z
    solution.estimated_singularity_time = p.horizon + 1.0 / (
        targets["singularity_rate"] * terminal_z
    )
    return solution, targets, p.horizon


def endpoint_residual(
    solution: Any,
    p: Parameters,
    targets: dict[str, Any],
    duration: float,
) -> float:
    left = solution.calendar_sol(0.0)
    right = solution.calendar_sol(duration)
    _, block = rates(duration, right, p)
    residuals = [
        left[0] - math.log(p.capital_initial),
        left[1] - math.log(p.capability_initial),
    ]
    if p.sigma_xl <= 1.0 + 1e-8:
        residuals.append(
            right[2] - block["log_output"] - math.log(targets["consumption_share"])
        )
        if targets["terminal_shadow_object"] == "profit_shadow_ratio":
            value = block["capability_profit_derivative"] * bounded_exp(-right[3])
        else:
            value = math.exp(right[3] + right[1] - block["log_output"])
        residuals.append(math.log(value) - math.log(targets["terminal_shadow_target"]))
    else:
        residuals.extend(
            [
                right[2]
                - block["log_output"]
                - math.log(targets["consumption_share"]),
                right[3]
                + right[1]
                - right[0]
                - math.log(targets["shadow_capability_to_capital"]),
            ]
        )
        if targets.get("boundary_mode") != "fixed_horizon":
            residuals.append(
                block["log_output"] - right[0] - math.log(p.terminal_z)
            )
    return max(abs(float(value)) for value in residuals)


def evaluate(
    solution: Any,
    p: Parameters,
    targets: dict[str, Any],
    duration: float,
) -> dict[str, Any]:
    sample_count = max(81, min(p.points, 301))
    times = np.linspace(0.0, duration, sample_count)
    states = solution.calendar_sol(times)
    derivatives = solution.calendar_derivative(times)
    rows: list[dict[str, float]] = []
    max_dynamic = 0.0
    max_static = 0.0
    min_soc = float("inf")
    for index, time in enumerate(times):
        state = states[:, index]
        rhs, block = rates(float(time), state, p)
        max_dynamic = max(
            max_dynamic,
            float(np.max(np.abs(derivatives[:, index] - rhs))),
        )
        human_share = block["human_share"]
        labor_error = abs(
            math.exp(block["log_production_labor"] - block["log_population"])
            + human_share
            - 1.0
        )
        max_static = max(max_static, labor_error, abs(block["research_foc_residual"]))
        inverse_elasticity = (
            (1.0 - block["ai_share"]) / p.sigma_xl
            + p.alpha * block["ai_share"]
        )
        soc_margin = (
            inverse_elasticity * (1.0 - inverse_elasticity)
            + (p.alpha - 1.0 / p.sigma_xl)
            * (1.0 - 1.0 / p.sigma_xl)
            * block["ai_share"]
            * (1.0 - block["ai_share"])
        )
        min_soc = min(min_soc, soc_margin)
        output_capital = math.exp(block["log_output"] - state[0])
        consumption_share = math.exp(state[2] - block["log_output"])
        investment_share = (
            (block["capital_growth"] + p.delta) / output_capital
        )
        production_labor_share = (1.0 - p.alpha) * (1.0 - block["ai_share"])
        aggregate_labor_share = math.exp(
            block["log_wage"] + block["log_population"] - block["log_output"]
        )
        log_service_composite = (
            block["log_output"] - p.alpha * state[0]
        ) / (1.0 - p.alpha)
        log_ai_price = (
            math.log1p(-p.alpha)
            + math.log(block["ai_share"])
            + block["log_output"]
            - block["log_ai_services"]
        )
        log_ai_marginal_cost = math.log(p.xi) - state[1]
        row = {
            "time": float(time),
            "log_capability": float(state[1]),
            "log_output_per_capita": block["log_output"] - block["log_population"],
            "log_consumption_per_capita": state[2] - block["log_population"],
            "log_capital_per_capita": state[0] - block["log_population"],
            "log_service_composite_per_capita": (
                log_service_composite - block["log_population"]
            ),
            "log_ai_services_per_capita": (
                block["log_ai_services"] - block["log_population"]
            ),
            "log_inference_compute_per_capita": (
                block["log_inference_compute"] - block["log_population"]
            ),
            "log_human_research_per_capita": (
                block["log_human_research"] - block["log_population"]
            ),
            "log_automated_research_per_capita": (
                block["log_automated_research"] - block["log_population"]
            ),
            "log_effective_research_per_capita": (
                block["log_effective_research"] - block["log_population"]
            ),
            "log_wage": block["log_wage"],
            "log_ai_price": log_ai_price,
            "log_ai_marginal_cost": log_ai_marginal_cost,
            "capability_growth": block["capability_growth"],
            "consumption_per_capita_growth": block["consumption_growth"] - p.n,
            "net_interest": block["gross_capital_return"] - p.delta,
            "human_research_share": human_share,
            "production_labor_population_share": 1.0 - human_share,
            "ai_share": block["ai_share"],
            "automated_research_share": block["automated_research_share"],
            "human_research_aggregate_share": (
                1.0 - block["automated_research_share"]
            ),
            "production_labor_share": production_labor_share,
            "aggregate_labor_share": aggregate_labor_share,
            "ai_markup": math.exp(log_ai_price - log_ai_marginal_cost),
            "ai_profit_share": (
                (1.0 - p.alpha) * block["ai_share"]
                - block["inference_share"]
            ),
            "shadow_capability_to_output": math.exp(
                state[3] + state[1] - block["log_output"]
            ),
            "shadow_capability_to_capital": math.exp(
                state[3] + state[1] - state[0]
            ),
            "consumption_share": consumption_share,
            "investment_share": investment_share,
            "inference_share": block["inference_share"],
            "research_resource_share": block["research_resource_share"],
            "output_capital_ratio": output_capital,
            "human_machine_ratio": math.exp(
                block["log_human_research"] - block["log_automated_research"]
            ),
        }
        rows.append(row)

    log_output_pc = np.asarray([row["log_output_per_capita"] for row in rows])
    log_wage = np.asarray([row["log_wage"] for row in rows])
    output_growth_pc = np.gradient(log_output_pc, times, edge_order=2)
    wage_growth = np.gradient(log_wage, times, edge_order=2)
    for row, growth, wage_rate in zip(rows, output_growth_pc, wage_growth):
        row["output_per_capita_growth"] = float(growth)
        row["wage_growth"] = float(wage_rate)

    for field in (
        "log_capability",
        "log_output_per_capita",
        "log_consumption_per_capita",
        "log_capital_per_capita",
        "log_service_composite_per_capita",
        "log_ai_services_per_capita",
        "log_inference_compute_per_capita",
        "log_human_research_per_capita",
        "log_automated_research_per_capita",
        "log_effective_research_per_capita",
        "log_wage",
        "log_ai_price",
        "log_ai_marginal_cost",
    ):
        initial = rows[0][field]
        for row in rows:
            row[field + "_change"] = row[field] - initial

    boundary_error = endpoint_residual(solution, p, targets, duration)
    collocation = float(
        np.max(solution.rms_residuals) if len(solution.rms_residuals) else math.nan
    )
    interior = all(
        row["consumption_share"] > 0.0
        and 0.0 < row["human_research_share"] < 1.0
        and 0.0 < row["ai_share"] < 1.0
        for row in rows
    )
    passed = bool(
        solution.success
        and collocation <= max(8.0 * p.tolerance, 1e-4)
        and max_dynamic <= max(8.0 * p.tolerance, 2e-4)
        and max_static <= 1e-6
        and boundary_error <= 1e-5
        and min_soc > 0.0
        and interior
    )
    d_cd = (
        1.0
        - p.phi
        - p.eta * p.omega_m * p.omega_x / (1.0 - p.omega_x)
    )
    d_ai = 1.0 - p.phi - p.eta * p.omega_x / (1.0 - p.omega_x)
    regime = (
        "complements" if p.sigma_xl < 1.0 - 1e-8 else
        "cobb_douglas" if abs(p.sigma_xl - 1.0) <= 1e-8 else
        "gross_substitutes"
    )
    diagnostics = {
        "passed": passed,
        "solver_success": bool(solution.success),
        "solver_message": str(solution.message),
        "collocation_residual": collocation,
        "dynamic_residual": max_dynamic,
        "static_residual": max_static,
        "endpoint_residual": boundary_error,
        "minimum_monopoly_margin": min_soc,
        "interior": interior,
        "duration": duration,
        "estimated_singularity_time": (
            float(solution.estimated_singularity_time)
            if hasattr(solution, "estimated_singularity_time")
            else None
        ),
        "d_cd": d_cd,
        "d_ai": d_ai,
        "regime": regime,
        "interpretation": (
            "Canonical equilibrium branch passed the reported numerical checks. "
            "Global dynamic optimality, existence, and uniqueness are not certified."
            if passed
            else
            "The numerical checks did not all pass. Do not interpret this path as an equilibrium."
        ),
    }
    public_targets = {
        key: value
        for key, value in targets.items()
        if isinstance(value, (int, float, str))
    }
    return {"diagnostics": diagnostics, "targets": public_targets, "series": rows}


def simulate(values: dict[str, Any]) -> dict[str, Any]:
    p = Parameters.from_mapping(values)
    errors = validate(p)
    if errors:
        return {"errors": errors, "diagnostics": {"passed": False}}
    try:
        if p.sigma_xl <= 1.0 + 1e-8:
            solution, targets, duration = solve_finite(p)
        else:
            if p.high_method == "free_boundary":
                solution, targets, duration = solve_high(p)
            else:
                warm_start = values.get("warm_start")
                if not isinstance(warm_start, list):
                    warm_start = None
                solution, targets, duration = solve_high_fixed(p, warm_start)
        result = evaluate(solution, p, targets, duration)
        result["parameters"] = {
            key: getattr(p, key) for key in p.__dataclass_fields__
        }
        return result
    except Exception as error:  # Browser API must return a structured failure.
        return {
            "errors": [f"{type(error).__name__}: {error}"],
            "diagnostics": {"passed": False},
        }


def simulate_json(values_json: str) -> str:
    return json.dumps(simulate(json.loads(values_json)), allow_nan=False)
