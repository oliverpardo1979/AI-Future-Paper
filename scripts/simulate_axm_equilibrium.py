"""Finite-horizon perfect-foresight path approximations with A*M research services.

This module replaces the proportional-investment and proportional-research
closure in ``simulate_model.py`` with the household Euler equation and the
integrated developer's research first-order and costate conditions.  Initial
capital and capability are predetermined.  Initial consumption and the private
shadow value of capability jump so that the path approaches the relevant
constant-growth limit.

The research technology is the generalized CES
F(A,H,M)=chi*((1-omega_m)*H**rho_hm + omega_m*(A*M)**rho_hm)**(eta/rho_hm),
where rho_hm=(sigma_hm-1)/sigma_hm, or equivalently
chi*E(H,A*M)**eta, where E is the effective-research CES index. Thus U and M are raw compute, A*U is
inference output, and A*M is automated-research services. This file is
intentionally separate from ``simulate_equilibrium``:
the extra A inside the research CES changes both the research price and the
developer's costate equation.
"""

from __future__ import annotations

import csv
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TMP_DEPS = ROOT / "tmp" / "pydeps"
LOCAL_DEPS = ROOT / ".python-packages"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))
elif TMP_DEPS.exists():
    sys.path.insert(0, str(TMP_DEPS))
sys.path.insert(0, str(ROOT / "scripts"))

from scipy.integrate import solve_bvp, solve_ivp  # noqa: E402
from scipy.optimize import brentq, least_squares  # noqa: E402

import simulate_model as mechanism  # noqa: E402



@dataclass(frozen=True)
class Parameters:
    """Parameters of the A*M specification; no direct A**phi term exists."""

    alpha: float = 0.33
    omega_x: float = 0.20
    sigma_xl: float = 1.00
    n: float = 0.012
    delta: float = 0.05
    discount: float = 0.04
    omega_m: float = 0.35
    sigma_hm: float = 2.00
    eta: float = 0.45
    chi: float = 0.01


RESULT_DIR = ROOT / "numerical_axm"
FIGURE_DIR = ROOT / "figures_axm"


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


def research_unit_cost(
    log_wage: float,
    log_capability: float,
    parameters: Parameters,
) -> float:
    """Log unit expenditure for the effective-research index E."""

    sigma = parameters.sigma_hm
    log_machine_service_price = -log_capability
    if abs(sigma - 1.0) <= 1e-10:
        omega_m = parameters.omega_m
        omega_h = 1.0 - omega_m
        return (
            omega_h * (log_wage - math.log(omega_h))
            + omega_m * (log_machine_service_price - math.log(omega_m))
        )
    log_human_term = (
        sigma * math.log1p(-parameters.omega_m)
        + (1.0 - sigma) * log_wage
    )
    log_machine_term = (
        sigma * math.log(parameters.omega_m)
        + (1.0 - sigma) * log_machine_service_price
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
            "monopoly_root_fallback": False,
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
            xtol=1e-13,
            rtol=1e-13,
            maxiter=120,
        )
        log_output, ai_share = quantities(log_ai_ratio)
        return {
            "log_ai_ratio": log_ai_ratio,
            "log_output": log_output,
            "ai_share": ai_share,
            "monopoly_root_fallback": False,
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
                xtol=1e-13,
                rtol=1e-13,
                maxiter=160,
            )
            log_output, ai_share = quantities(log_ai_ratio)
            return {
                "log_ai_ratio": log_ai_ratio,
                "log_output": log_output,
                "ai_share": ai_share,
                "monopoly_root_fallback": False,
            }
        previous_x = current_x
        previous_value = current_value
    log_ai_ratio = min(candidates)[1]
    log_output, ai_share = quantities(log_ai_ratio)
    return {
        "log_ai_ratio": log_ai_ratio,
        "log_output": log_output,
        "ai_share": ai_share,
        "monopoly_root_fallback": True,
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
        log_research_price = research_unit_cost(
            log_wage, log_capability, parameters
        )
        log_effective_research = (
            log_shadow_value
            + math.log(parameters.chi)
            + math.log(parameters.eta)
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
            "monopoly_root_fallback": production["monopoly_root_fallback"],
            "log_wage": log_wage,
            "log_research_price": log_research_price,
            "log_effective_research": log_effective_research,
        }

    labor_root_fallback = False
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
            labor_root_fallback = True
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
    # Conditional demand is for automated-research services R=A*M.  The
    # resource constraint prices normalized raw compute M=R/A at one.
    log_automated_research_services = (
        sigma * math.log(parameters.omega_m)
        + sigma
        * (
            block["log_research_price"]
            + log_capability
        )
        + block["log_effective_research"]
    )
    log_automated_research = (
        log_automated_research_services - log_capability
    )
    log_ai_services = block["log_production_labor"] + block["log_ai_ratio"]
    log_inference_compute = log_ai_services - log_capability
    log_capability_flow = (
        math.log(parameters.chi)
        + parameters.eta * block["log_effective_research"]
    )
    capability_growth = bounded_exp(log_capability_flow - log_capability)
    gross_capital_return = parameters.alpha * bounded_exp(
        block["log_output"] - log_capital
    )
    automated_share = logistic(
        log_automated_research
        - block["log_wage"]
        - block["log_human_research"]
    )
    inference_share = bounded_exp(
        log_inference_compute
        - block["log_output"]
    )
    research_resource_share = bounded_exp(
        log_automated_research
        - block["log_output"]
    )
    capability_profit_derivative = bounded_exp(
        log_ai_services - 2.0 * log_capability
    )

    block.update(
        {
            "log_automated_research": log_automated_research,
            "log_automated_research_services": (
                log_automated_research_services
            ),
            "log_ai_services": log_ai_services,
            "log_inference_compute": log_inference_compute,
            "capability_growth": capability_growth,
            "gross_capital_return": gross_capital_return,
            "automated_research_share": automated_share,
            "inference_share": inference_share,
            "research_resource_share": research_resource_share,
            "capability_profit_derivative": capability_profit_derivative,
            "labor_root_fallback": labor_root_fallback,
        }
    )
    return block


def equilibrium_rates(
    time: float,
    state: Iterable[float],
    parameters: Parameters,
    log_initial_population: float = 0.0,
) -> tuple[np.ndarray, dict[str, float]]:
    log_capital, log_capability, log_consumption, log_shadow_value = map(
        float, state
    )
    log_population = log_initial_population + parameters.n * time
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
        block["log_inference_compute"]
        - log_capital
    )
    research_capital_ratio = bounded_exp(
        block["log_automated_research"]
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
        - parameters.eta * block["automated_research_share"]
        * block["capability_growth"]
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


def technology_log_errors(
    block: dict[str, float],
    log_capital: float,
    log_capability: float,
    parameters: Parameters,
) -> dict[str, float]:
    """Reconstruct the three static technologies independently in logs."""

    log_labor = block["log_production_labor"]
    log_ai_services = block["log_ai_services"]
    if abs(parameters.sigma_xl - 1.0) <= 1e-10:
        log_composite = (
            (1.0 - parameters.omega_x) * log_labor
            + parameters.omega_x * log_ai_services
        )
    else:
        ces_power = (parameters.sigma_xl - 1.0) / parameters.sigma_xl
        log_composite = logsumexp_pair(
            math.log1p(-parameters.omega_x) + ces_power * log_labor,
            math.log(parameters.omega_x)
            + ces_power * log_ai_services,
        ) / ces_power
    reconstructed_output = (
        parameters.alpha * log_capital
        + (1.0 - parameters.alpha) * log_composite
    )

    log_human_research = block["log_human_research"]
    log_automated_services = block["log_automated_research_services"]
    if abs(parameters.sigma_hm - 1.0) <= 1e-10:
        reconstructed_research = (
            (1.0 - parameters.omega_m) * log_human_research
            + parameters.omega_m * log_automated_services
        )
    else:
        ces_power = (parameters.sigma_hm - 1.0) / parameters.sigma_hm
        reconstructed_research = logsumexp_pair(
            math.log1p(-parameters.omega_m)
            + ces_power * log_human_research,
            math.log(parameters.omega_m)
            + ces_power * log_automated_services,
        ) / ces_power

    return {
        "final_production_log_error": (
            block["log_output"] - reconstructed_output
        ),
        "inference_identity_log_error": (
            log_ai_services
            - log_capability
            - block["log_inference_compute"]
        ),
        "research_ces_log_error": (
            block["log_effective_research"] - reconstructed_research
        ),
    }


def asymptotic_targets(parameters: Parameters) -> dict[str, float | str]:
    if parameters.sigma_hm < 1.0 - 1e-9:
        raise ValueError(
            "The finite balanced-growth terminal map is implemented only for "
            "sigma_HM=1 or sigma_HM>1. The complements branch needs its own "
            "terminal conditions."
        )
    human_essential = abs(parameters.sigma_hm - 1.0) <= 1e-9
    research_feedback_weight = (
        parameters.omega_m if human_essential else 1.0
    )

    if parameters.sigma_xl < 1.0 - 1e-9:
        # When final-production inputs are gross complements, X/L converges
        # to a finite constant while the resource costs U=X/A and M vanish
        # relative to output.  The limiting Euler equation therefore gives
        # r=rho and K/Y=alpha/(rho+delta).  A constant U/(qA) closes the
        # developer costate equation and yields the growth rates below.
        capability_growth = (
            parameters.eta
            * parameters.n
            / (
                1.0
                + parameters.eta
                * (1.0 - research_feedback_weight)
            )
        )
        aggregate_growth = parameters.n
        shadow_growth = parameters.n - 2.0 * capability_growth
        capital_output_ratio = parameters.alpha / (
            parameters.discount + parameters.delta
        )
        investment_share = (
            parameters.n + parameters.delta
        ) * capital_output_ratio
        consumption_share = 1.0 - investment_share
        profit_shadow_ratio = (
            parameters.discount
            - parameters.n
            + (
                2.0
                - parameters.eta * research_feedback_weight
            )
            * capability_growth
        )
        if consumption_share <= 0.0:
            raise ValueError(
                "The complementarity terminal consumption share is not positive."
            )
        if profit_shadow_ratio <= 0.0:
            raise ValueError(
                "The complementarity terminal profit-shadow ratio is not positive."
            )
        limiting_ai_share = (
            1.0 - parameters.sigma_xl
        ) / (1.0 - parameters.alpha * parameters.sigma_xl)
        limiting_ai_labor_ratio = (
            limiting_ai_share
            / (1.0 - limiting_ai_share)
            * (1.0 - parameters.omega_x)
            / parameters.omega_x
        ) ** (
            parameters.sigma_xl / (parameters.sigma_xl - 1.0)
        )
        return {
            "aggregate_growth": aggregate_growth,
            "capability_growth": capability_growth,
            "shadow_growth": shadow_growth,
            "consumption_share": consumption_share,
            "investment_share": investment_share,
            "research_share": 0.0,
            "capital_output_ratio": capital_output_ratio,
            "terminal_shadow_object": "profit_shadow_ratio",
            "terminal_shadow_target": profit_shadow_ratio,
            "limiting_research_weight": research_feedback_weight,
            "limiting_ai_share": limiting_ai_share,
            "limiting_ai_labor_ratio": limiting_ai_labor_ratio,
            "limiting_net_interest_rate": parameters.discount,
        }

    if abs(parameters.sigma_xl - 1.0) <= 1e-9:
        beta = (1.0 - parameters.alpha) * parameters.omega_x
        upsilon = beta / (1.0 - parameters.alpha - beta)
        denominator = (
            1.0
            - parameters.eta
            * research_feedback_weight
            * (1.0 + upsilon)
        )
        if denominator <= 0.0:
            raise ValueError(
                "The A*M Cobb--Douglas asymptotic denominator is not positive."
            )
        capability_growth = parameters.eta * parameters.n / denominator
        per_capita_growth = upsilon * capability_growth
        research_denominator = (
            parameters.discount
            - parameters.n
            + (
                1.0
                - parameters.eta * research_feedback_weight
            )
            * capability_growth
        )
        research_share = (
            beta**2
            * parameters.eta
            * research_feedback_weight
            * capability_growth
            / research_denominator
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
        gross_return = (
            parameters.discount
            + parameters.delta
            + per_capita_growth
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
            "investment_share": investment_share,
            "research_share": research_share,
            "capital_output_ratio": parameters.alpha / gross_return,
            "terminal_shadow_object": "shadow_capability_output_ratio",
            "terminal_shadow_target": research_share
            / (
                parameters.eta
                * research_feedback_weight
                * capability_growth
            ),
            "limiting_research_weight": research_feedback_weight,
        }
    raise ValueError(
        "This solver only imposes the finite Cobb--Douglas terminal path. "
        "Use the separate free-boundary solver when sigma_xl > 1."
    )


def fixed_share_guess(
    parameters: Parameters,
    initial_state: tuple[float, float, float],
    horizon: float,
    mesh: np.ndarray,
) -> np.ndarray:
    targets = asymptotic_targets(parameters)
    log_initial_capability = math.log(initial_state[1])
    log_initial_population = math.log(initial_state[2])
    terminal_shadow_object = str(targets["terminal_shadow_object"])

    def initial_ratios(values: np.ndarray) -> np.ndarray:
        log_capital, log_shadow = map(float, values)
        block = equilibrium_static_block(
            log_capital,
            log_initial_capability,
            log_initial_population,
            log_shadow,
            parameters,
        )
        if terminal_shadow_object == "profit_shadow_ratio":
            shadow_residual = (
                math.log(block["capability_profit_derivative"])
                - log_shadow
                - math.log(float(targets["terminal_shadow_target"]))
            )
        else:
            shadow_residual = (
                log_shadow
                + log_initial_capability
                - block["log_output"]
                - math.log(float(targets["terminal_shadow_target"]))
            )
        return np.asarray(
            [
                log_capital
                - block["log_output"]
                - math.log(float(targets["capital_output_ratio"])),
                shadow_residual,
            ]
        )

    ratio_root = least_squares(
        initial_ratios,
        np.asarray([math.log(initial_state[0]), -1.0]),
        xtol=1e-11,
        ftol=1e-11,
        gtol=1e-11,
        max_nfev=200,
    )
    if (
        not ratio_root.success
        or float(np.max(np.abs(initial_ratios(ratio_root.x)))) >= 1e-8
    ):
        raise RuntimeError(
            "The asymptotic fixed-share initial guess could not be constructed."
        )
    log_capital_zero, log_shadow_zero = map(float, ratio_root.x)
    aggregate_growth = float(targets["aggregate_growth"])
    capability_growth = float(targets["capability_growth"])
    shadow_growth = float(targets["shadow_growth"])
    log_capital = log_capital_zero + aggregate_growth * mesh
    log_capability = log_initial_capability + capability_growth * mesh
    log_shadow = log_shadow_zero + shadow_growth * mesh
    log_consumption = np.empty_like(mesh)
    for index, time in enumerate(mesh):
        block = equilibrium_static_block(
            float(log_capital[index]),
            float(log_capability[index]),
            log_initial_population + parameters.n * float(time),
            float(log_shadow[index]),
            parameters,
        )
        log_consumption[index] = (
            block["log_output"]
            + math.log(float(targets["consumption_share"]))
        )
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
    log_initial_population = math.log(initial_state[2])
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
                float(time), raw_state, parameters, log_initial_population
            )[0] - growth_scales
        return values

    def boundary(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        terminal_raw = right + growth_scales * horizon
        _, terminal_block = equilibrium_rates(
            horizon, terminal_raw, parameters, log_initial_population
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
    log_initial_population = math.log(initial_state[2])
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
                float(time), state, parameters, log_initial_population
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
            float(solution.t[-1]), terminal, parameters, log_initial_population
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
    initial_population: float = 1.0,
) -> list[dict[str, float | str]]:
    times = np.arange(0.0, horizon + 0.5 * step, step)
    states = solution.sol(times)
    path_derivatives = (
        solution.calendar_derivative(times)
        if hasattr(solution, "calendar_derivative")
        else None
    )
    rows: list[dict[str, float | str]] = []
    log_initial_population = math.log(initial_population)
    for index, time in enumerate(times):
        derivatives, block = equilibrium_rates(
            float(time), states[:, index], parameters, log_initial_population
        )
        log_capital, log_capability, log_consumption, log_shadow = map(
            float, states[:, index]
        )
        log_population = log_initial_population + parameters.n * float(time)
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
        technology_errors = technology_log_errors(
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
        if abs(parameters.sigma_xl - 1.0) <= 1e-10:
            reconstructed_ai_share = parameters.omega_x
        else:
            final_ces_power = (
                parameters.sigma_xl - 1.0
            ) / parameters.sigma_xl
            reconstructed_ai_share = logistic(
                math.log(parameters.omega_x)
                - math.log1p(-parameters.omega_x)
                + final_ces_power
                * (
                    block["log_ai_services"]
                    - block["log_production_labor"]
                )
            )
        reconstructed_research_price = research_unit_cost(
            block["log_wage"], log_capability, parameters
        )
        reconstructed_automated_share = logistic(
            block["log_automated_research"]
            - block["log_wage"]
            - block["log_human_research"]
        )
        log_human_conditional_demand = (
            parameters.sigma_hm * math.log1p(-parameters.omega_m)
            + parameters.sigma_hm
            * (block["log_research_price"] - block["log_wage"])
            + block["log_effective_research"]
        )
        log_automated_service_conditional_demand = (
            parameters.sigma_hm * math.log(parameters.omega_m)
            + parameters.sigma_hm
            * (block["log_research_price"] + log_capability)
            + block["log_effective_research"]
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
            "log_research_price": block["log_research_price"],
            "log_ai_services": block["log_ai_services"],
            "log_inference_compute": block["log_inference_compute"],
            "log_human_research": block["log_human_research"],
            "log_production_labor": block["log_production_labor"],
            "log_automated_research": block["log_automated_research"],
            "log_automated_research_services": block[
                "log_automated_research_services"
            ],
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
            "profit_shadow_ratio": (
                block["capability_profit_derivative"]
                * math.exp(-log_shadow)
            ),
            "ai_labor_ratio": math.exp(block["log_ai_ratio"]),
            "monopoly_root_fallback": float(
                bool(block["monopoly_root_fallback"])
            ),
            "labor_root_fallback": float(bool(block["labor_root_fallback"])),
            "human_to_automated_research_ratio": math.exp(
                block["log_human_research"]
                - block["log_automated_research"]
            ),
            "human_to_automated_service_ratio": math.exp(
                block["log_human_research"]
                - block["log_automated_research_services"]
            ),
            "log_output_per_capita": block["log_output"] - log_population,
            "log_consumption_per_capita": log_consumption - log_population,
            "log_capital_per_capita": log_capital - log_population,
            "monopoly_foc_log_error": monopoly_foc_log_error,
            "capital_price_error": (
                block["gross_capital_return"]
                - parameters.alpha
                * math.exp(block["log_output"] - log_capital)
            ),
            "wage_foc_log_error": (
                block["log_wage"]
                - math.log1p(-parameters.alpha)
                - math.log1p(-block["ai_share"])
                - block["log_output"]
                + block["log_production_labor"]
            ),
            "ai_price_foc_log_error": (
                log_price
                - math.log1p(-parameters.alpha)
                - math.log(block["ai_share"])
                - block["log_output"]
                + block["log_ai_services"]
            ),
            "ai_share_definition_error": (
                block["ai_share"] - reconstructed_ai_share
            ),
            "research_price_log_error": (
                block["log_research_price"] - reconstructed_research_price
            ),
            "human_conditional_demand_log_error": (
                block["log_human_research"]
                - log_human_conditional_demand
            ),
            "automated_service_demand_log_error": (
                block["log_automated_research_services"]
                - log_automated_service_conditional_demand
            ),
            "automated_share_definition_error": (
                block["automated_research_share"]
                - reconstructed_automated_share
            ),
            "research_scale_foc_log_error": (
                log_shadow
                + math.log(parameters.chi)
                + math.log(parameters.eta)
                + (parameters.eta - 1.0)
                * block["log_effective_research"]
                - block["log_research_price"]
            ),
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


def path_diagnostics(
    rows: list[dict[str, float | str]],
    parameters: Parameters,
) -> dict[str, float]:
    """Audit every dated equation and finite-horizon terminal proxy on a path."""

    ordered = sorted(rows, key=lambda row: float(row["time"]))
    if not ordered:
        raise ValueError("Cannot audit an empty equilibrium path.")

    discounted_interest = 0.0
    for left, right in zip(ordered[:-1], ordered[1:]):
        time_step = float(right["time"]) - float(left["time"])
        discounted_interest += 0.5 * time_step * (
            float(left["net_capital_return"])
            + float(right["net_capital_return"])
        )
    terminal = ordered[-1]
    terminal_time = float(terminal["time"])
    household_tvc_log = (
        -parameters.discount * terminal_time
        + float(terminal["log_population"])
        + float(terminal["log_capital"])
        - float(terminal["log_consumption"])
    )
    developer_tvc_log = (
        -discounted_interest
        + float(terminal["log_shadow_value"])
        + float(terminal["log_capability"])
    )

    return {
        "minimum_consumption_share": min(
            float(row["consumption_share"]) for row in ordered
        ),
        "minimum_investment_share": min(
            float(row["investment_share"]) for row in ordered
        ),
        "minimum_inference_share": min(
            float(row["inference_share"]) for row in ordered
        ),
        "minimum_research_resource_share": min(
            float(row["research_resource_share"]) for row in ordered
        ),
        "minimum_human_research_share": min(
            float(row["human_research_share"]) for row in ordered
        ),
        "minimum_production_labor_share": min(
            float(row["production_labor_population_share"]) for row in ordered
        ),
        "max_abs_euler_residual": max(
            abs(float(row["euler_residual"])) for row in ordered
        ),
        "max_abs_resource_residual": max(
            abs(float(row["resource_share_sum"]) - 1.0) for row in ordered
        ),
        "max_abs_monopoly_foc_log_error": max(
            abs(float(row["monopoly_foc_log_error"])) for row in ordered
        ),
        "max_abs_factor_price_error": max(
            abs(float(row[field]))
            for row in ordered
            for field in (
                "capital_price_error",
                "wage_foc_log_error",
                "ai_price_foc_log_error",
            )
        ),
        "max_abs_share_definition_error": max(
            abs(float(row[field]))
            for row in ordered
            for field in (
                "ai_share_definition_error",
                "automated_share_definition_error",
            )
        ),
        "max_abs_research_dual_error": max(
            abs(float(row[field]))
            for row in ordered
            for field in (
                "research_price_log_error",
                "human_conditional_demand_log_error",
                "automated_service_demand_log_error",
                "research_scale_foc_log_error",
            )
        ),
        "max_abs_technology_log_error": max(
            abs(float(row[field]))
            for row in ordered
            for field in (
                "final_production_log_error",
                "inference_identity_log_error",
                "research_ces_log_error",
            )
        ),
        "max_abs_research_compute_foc_log_error": max(
            abs(float(row["research_compute_foc_log_error"]))
            for row in ordered
        ),
        "max_abs_research_human_foc_log_error": max(
            abs(float(row["research_human_foc_log_error"]))
            for row in ordered
        ),
        "max_abs_labor_market_error": max(
            abs(float(row["labor_market_error"])) for row in ordered
        ),
        "max_abs_capital_law_residual": max(
            abs(float(row["capital_law_residual"])) for row in ordered
        ),
        "max_abs_capability_law_residual": max(
            abs(float(row["capability_law_residual"])) for row in ordered
        ),
        "max_abs_consumption_path_residual": max(
            abs(float(row["consumption_euler_path_residual"]))
            for row in ordered
        ),
        "max_abs_shadow_costate_residual": max(
            abs(float(row["shadow_costate_residual"])) for row in ordered
        ),
        "minimum_monopoly_soc_margin": min(
            float(row["monopoly_soc_margin"]) for row in ordered
        ),
        "terminal_household_tvc_log_proxy": household_tvc_log,
        "terminal_developer_tvc_log_proxy": developer_tvc_log,
        "terminal_household_tvc_proxy": math.exp(household_tvc_log),
        "terminal_developer_tvc_proxy": math.exp(developer_tvc_log),
    }


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
        if key in {"axm_sigma_xl_1_hm_1", "axm_sigma_xl_1_hm_2"}
    }
    labels = {
        "axm_sigma_xl_1_hm_1": (
            "Cobb-Douglas research (research elasticity = 1)"
        ),
        "axm_sigma_xl_1_hm_2": (
            "Gross-substitutes research (research elasticity = 2)"
        ),
    }
    palette = {
        "axm_sigma_xl_1_hm_1": mechanism.COLORS["blue"],
        "axm_sigma_xl_1_hm_2": mechanism.COLORS["orange"],
    }
    markers = {
        "axm_sigma_xl_1_hm_1": "circle",
        "axm_sigma_xl_1_hm_2": "square",
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
    research_labor_income_percent = lambda rows, values: 100.0 * (
        values
        - np.asarray(
            [float(row["production_labor_share"]) for row in rows]
        )
    )
    automated_to_human_service_log_change = lambda rows, values: -(
        np.log(values) - np.log(values[0])
    )

    mechanism.draw_multiplot(
        FIGURE_DIR / "axm_equilibrium_levels.png",
        "Macroeconomic quantities along unit-elasticity paths",
        "Change in natural logs from each path's date-zero level; quantities other than the wage are per capita",
        [
            {"title": "Output per capita", "field": "log_output_per_capita", "transform": log_change},
            {"title": "Consumption per capita", "field": "log_consumption_per_capita", "transform": log_change},
            {"title": "Real wage", "field": "log_wage", "transform": log_change},
            {"title": "Capital per capita", "field": "log_capital_per_capita", "transform": log_change},
        ],
        display_rows,
        labels,
        palette,
        markers,
    )
    mechanism.draw_multiplot(
        FIGURE_DIR / "axm_equilibrium_growth_rates.png",
        "Macroeconomic growth and returns along unit-elasticity paths",
        "Annual percentage rates over the first 600 years; all panels use linear scales",
        [
            {"title": "Output growth per capita", "field": "output_per_capita_growth", "transform": percent, "format": lambda value: f"{value:.1f}%", "reference_y": 0.0},
            {"title": "Consumption growth per capita", "field": "consumption_per_capita_growth", "transform": percent, "format": lambda value: f"{value:.1f}%", "reference_y": 0.0},
            {"title": "Real-wage growth", "field": "wage_growth", "transform": percent, "format": lambda value: f"{value:.2f}%", "reference_y": 0.0},
            {"title": "Net return to capital", "field": "net_capital_return", "transform": percent, "format": lambda value: f"{value:.1f}%"},
        ],
        display_rows,
        labels,
        palette,
        markers,
    )
    mechanism.draw_multiplot(
        FIGURE_DIR / "axm_equilibrium_production_chain.png",
        "Final-production chain along unit-elasticity paths",
        "Change in natural logs from each path's date-zero per-capita level",
        [
            {"title": "Inference compute per capita", "field": "log_inference_compute", "transform": per_capita_log_change},
            {"title": "AI services per capita", "field": "log_ai_services", "transform": per_capita_log_change},
            {"title": "Service composite per capita", "field": "log_service_composite", "transform": per_capita_log_change},
            {"title": "Final output per capita", "field": "log_output_per_capita", "transform": log_change},
        ],
        display_rows,
        labels,
        palette,
        markers,
    )
    mechanism.draw_multiplot(
        FIGURE_DIR / "axm_equilibrium_research_chain.png",
        "AI-research quantities along unit-elasticity paths",
        "Change in natural logs from each path's date-zero per-capita level",
        [
            {"title": "Human research per capita", "field": "log_human_research", "transform": per_capita_log_change},
            {"title": "Research compute per capita", "field": "log_automated_research", "transform": per_capita_log_change},
            {"title": "AI research services per capita", "field": "log_automated_research_services", "transform": per_capita_log_change},
            {"title": "Effective-research index per capita", "field": "log_effective_research", "transform": per_capita_log_change},
        ],
        display_rows,
        labels,
        palette,
        markers,
    )
    mechanism.draw_multiplot(
        FIGURE_DIR / "axm_equilibrium_resource_allocation.png",
        "Allocation of final output along unit-elasticity paths",
        "Shares of final output in percent; all panels use linear scales and the four uses sum to one",
        [
            {"title": "Consumption / output", "field": "consumption_share", "transform": percent, "format": lambda value: f"{value:.0f}%"},
            {"title": "Investment / output", "field": "investment_share", "transform": percent, "format": lambda value: f"{value:.0f}%"},
            {"title": "Inference resources / output", "field": "inference_share", "transform": percent, "format": lambda value: f"{value:.3f}%", "ylim": (1.79, 1.80)},
            {"title": "Research compute / output", "field": "research_resource_share", "transform": percent, "format": lambda value: f"{value:.2f}%"},
        ],
        display_rows,
        labels,
        palette,
        markers,
    )
    mechanism.draw_multiplot(
        FIGURE_DIR / "axm_equilibrium_monopoly_block.png",
        "Perfect-foresight equilibrium-path approximation: the integrated AI developer",
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
        "axm_sigma_xl_1_hm_2": scenario_rows["axm_sigma_xl_1_hm_2"]
    }
    mechanism.draw_multiplot(
        FIGURE_DIR / "axm_cobb_douglas_long_run.png",
        "Cobb-Douglas final production: long-run equilibrium-path approximation",
        "Annual rates and shares in percent; the full 1,600-year solution is shown",
        [
            {"title": "Capability growth", "field": "capability_growth", "transform": percent, "format": lambda value: f"{value:.1f}%"},
            {"title": "Output growth per capita", "field": "output_per_capita_growth", "transform": percent, "format": lambda value: f"{value:.1f}%", "reference_y": 0.0},
            {"title": "Automated research expenditure share", "field": "automated_research_share", "transform": percent, "format": lambda value: f"{value:.0f}%", "ylim": (0.0, 100.0)},
            {"title": "Human researchers / population", "field": "human_research_share", "transform": percent, "format": lambda value: f"{value:.2f}%"},
        ],
        cobb_douglas_rows,
        labels,
        palette,
        markers,
    )
    mechanism.draw_multiplot(
        FIGURE_DIR / "axm_equilibrium_factor_shares.png",
        "Labor income and labor allocation along unit-elasticity paths",
        "wL/Y, wH/Y, and wN/Y are income shares; L/N is an allocation share; all panels are percentages",
        [
            {"title": "Production labor income / output", "field": "production_labor_share", "transform": percent, "format": lambda value: f"{value:.1f}%", "ylim": (53.0, 54.0)},
            {"title": "Research labor income / output", "field": "aggregate_labor_share", "transform": research_labor_income_percent, "format": lambda value: f"{value:.3f}%"},
            {"title": "Aggregate labor income / output", "field": "aggregate_labor_share", "transform": percent, "format": lambda value: f"{value:.2f}%"},
            {"title": "Production labor / population", "field": "production_labor_population_share", "transform": percent, "format": lambda value: f"{value:.2f}%"},
        ],
        display_rows,
        labels,
        palette,
        markers,
    )
    mechanism.draw_multiplot(
        FIGURE_DIR / "axm_wages_and_factor_shares.png",
        "Wages and labor income along an equilibrium-path approximation",
        "Blue circles: sigma_HM = 1; orange squares: sigma_HM = 2; wage changes in logs, other panels in percent",
        [
            {"title": "Real wage", "field": "log_wage", "transform": log_change},
            {"title": "Real-wage growth", "field": "wage_growth", "transform": percent, "format": lambda value: f"{value:.2f}%", "reference_y": 0.0},
            {"title": "Production labor income / output", "field": "production_labor_share", "transform": percent, "format": lambda value: f"{value:.1f}%", "ylim": (53.0, 54.0)},
            {"title": "Aggregate labor income / output", "field": "aggregate_labor_share", "transform": percent, "format": lambda value: f"{value:.2f}%"},
        ],
        display_rows,
        labels,
        palette,
        markers,
    )

    research_technology_rows = {
        "axm_sigma_xl_1_hm_1": scenario_rows[
            "axm_sigma_xl_1_hm_1"
        ],
        "axm_sigma_xl_1_hm_2": scenario_rows["axm_sigma_xl_1_hm_2"],
    }
    research_labels = {
        "axm_sigma_xl_1_hm_1": (
            "Cobb-Douglas research (research elasticity = 1)"
        ),
        "axm_sigma_xl_1_hm_2": (
            "Gross-substitutes research (research elasticity = 2)"
        ),
    }
    mechanism.draw_multiplot(
        FIGURE_DIR / "axm_research_technology_comparison.png",
        "Research adjustment under alternative substitution elasticities",
        "Capability growth is an annual percent rate; both shares are percentages; AM/H is a natural-log change",
        [
            {"title": "Capability growth", "field": "capability_growth", "transform": percent, "format": lambda value: f"{value:.1f}%"},
            {"title": "Automated research expenditure share", "field": "automated_research_share", "transform": percent, "format": lambda value: f"{value:.0f}%", "ylim": (0.0, 100.0)},
            {"title": "Human researchers / population", "field": "human_research_share", "transform": percent, "format": lambda value: f"{value:.2f}%"},
            {"title": "Automated research services / human research", "field": "human_to_automated_service_ratio", "transform": automated_to_human_service_log_change},
        ],
        research_technology_rows,
        research_labels,
        palette,
        markers,
    )


def main() -> None:
    RESULT_DIR.mkdir(exist_ok=True)
    FIGURE_DIR.mkdir(exist_ok=True)
    baseline = Parameters()
    reference_parameters = replace(
        baseline, sigma_xl=1.0, sigma_hm=2.0
    )
    seed_guess = fixed_share_guess(
        reference_parameters,
        (1.0, 1.0, 1.0),
        horizon=1.0,
        mesh=np.asarray([0.0]),
    )
    initial_state = (math.exp(float(seed_guess[0, 0])), 1.0, 1.0)

    summaries: list[dict[str, float | str]] = []
    all_rows: list[dict[str, float | str]] = []
    scenario_rows: dict[str, list[dict[str, float | str]]] = {}
    primary_initial_jumps: dict[float, dict[str, float]] = {}
    for name, sigma_xl, sigma_hm, horizon in [
        ("axm_sigma_xl_1_hm_1", 1.00, 1.00, 1200.0),
        ("axm_sigma_xl_1_hm_2", 1.00, 2.00, 1600.0),
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
            step=2.0,
            initial_population=initial_state[2],
        )
        scenario_rows[name] = rows
        all_rows.extend(rows)
        initial = rows[0]
        final = rows[-1]
        primary_initial_jumps[sigma_hm] = {
            "horizon": horizon,
            "initial_log_consumption": float(initial["log_consumption"]),
            "initial_log_shadow_value": float(initial["log_shadow_value"]),
            "max_rms_residual": float(np.max(solution.rms_residuals)),
        }
        diagnostics = path_diagnostics(rows, parameters)
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
                "initial_capital_stock": initial_state[0],
                "initial_capability_stock": initial_state[1],
                "initial_population": initial_state[2],
                "initial_log_consumption": initial["log_consumption"],
                "initial_log_shadow_value": initial["log_shadow_value"],
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
                **diagnostics,
            }
        )

    write_rows(RESULT_DIR / "equilibrium_transition_paths.csv", all_rows)
    write_rows(RESULT_DIR / "equilibrium_transition_summary.csv", summaries)

    horizon_rows: list[dict[str, float | str]] = []
    all_horizon_path_rows: list[dict[str, float | str]] = []
    for sigma_hm, horizons in {
        1.0: (1000.0, 1200.0, 1400.0),
        2.0: (1400.0, 1600.0, 1800.0),
    }.items():
        parameters = replace(
            baseline,
            sigma_xl=1.0,
            sigma_hm=sigma_hm,
        )
        primary = primary_initial_jumps[sigma_hm]
        targets = asymptotic_targets(parameters)
        for horizon in horizons:
            robustness_name = (
                f"axm_sigma_xl_1_hm_{sigma_hm:g}_T_{horizon:g}"
            )
            if math.isclose(horizon, primary["horizon"]):
                values = dict(primary)
                base_name = f"axm_sigma_xl_1_hm_{sigma_hm:g}"
                robustness_rows = [
                    {**row, "scenario": robustness_name}
                    for row in scenario_rows[base_name]
                ]
            else:
                robustness_solution, _ = solve_equilibrium(
                    parameters,
                    initial_state,
                    horizon=horizon,
                )
                if not robustness_solution.success:
                    raise RuntimeError(
                        "Horizon robustness failed for "
                        f"sigma_HM={sigma_hm:g}, T={horizon:g}."
                    )
                initial_values = robustness_solution.sol(0.0)
                values = {
                    "initial_log_consumption": float(initial_values[2]),
                    "initial_log_shadow_value": float(initial_values[3]),
                    "max_rms_residual": float(
                        np.max(robustness_solution.rms_residuals)
                    ),
                }
                robustness_rows = evaluate_solution(
                    robustness_name,
                    robustness_solution,
                    parameters,
                    horizon,
                    step=2.0,
                    initial_population=initial_state[2],
                )
            diagnostics = path_diagnostics(robustness_rows, parameters)
            terminal = robustness_rows[-1]
            terminal_shadow_ratio = math.exp(
                float(terminal["log_shadow_value"])
                + float(terminal["log_capability"])
                - float(terminal["log_output"])
            )
            values.update(
                {
                    **diagnostics,
                    "initial_capital_log_error": abs(
                        float(robustness_rows[0]["log_capital"])
                        - math.log(initial_state[0])
                    ),
                    "initial_capability_log_error": abs(
                        float(robustness_rows[0]["log_capability"])
                        - math.log(initial_state[1])
                    ),
                    "initial_population_log_error": abs(
                        float(robustness_rows[0]["log_population"])
                        - math.log(initial_state[2])
                    ),
                    "terminal_consumption_target_error": abs(
                        float(terminal["consumption_share"])
                        - float(targets["consumption_share"])
                    ),
                    "terminal_shadow_target_error": abs(
                        terminal_shadow_ratio
                        - float(targets["terminal_shadow_target"])
                    ),
                }
            )
            all_horizon_path_rows.extend(robustness_rows)
            horizon_rows.append(
                {
                    "sigma_hm": sigma_hm,
                    "horizon": horizon,
                    **values,
                }
            )
    write_rows(
        RESULT_DIR / "equilibrium_horizon_robustness.csv",
        horizon_rows,
    )
    write_rows(
        RESULT_DIR / "equilibrium_horizon_paths.csv",
        all_horizon_path_rows,
    )
    draw_equilibrium_figures(scenario_rows)


if __name__ == "__main__":
    main()
