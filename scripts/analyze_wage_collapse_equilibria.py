"""Summarize equilibrium searches for a real-wage collapse.

The source paths satisfy the canonical household--firm--developer equilibrium
equations on their computed domains.  Because all cases use
``sigma_xl > 1 / alpha``, their AI-dominated terminal boundary is extrapolated
beyond the paper's analytically sufficient region.  They are conditional branch
searches, not existence proofs.  This script does not use the legacy
fixed-saving-share experiments.  It combines representative paths, computes wage
drawdowns and equation residuals, and draws a transition-focused figure.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable

import numpy as np

import simulate_model as mechanism


ROOT = Path(__file__).resolve().parents[1]
NUMERICAL = ROOT / "numerical"
FIGURES = ROOT / "figures"

SCENARIOS = (
    {
        "key": "sigma4_n012",
        "label": "sigma_XL = 4.00, n = 1.2%",
        "sigma_xl": 4.0,
        "population_growth": 0.012,
        "terminal_z": 3.0,
        "path": "wage_collapse_probe_sigma4_n012_z3_paths.csv",
        "solver_summary": "wage_collapse_probe_sigma4_n012_z3_summary.csv",
    },
    {
        "key": "sigma5_n030",
        "label": "sigma_XL = 5.00, n = 3.0%",
        "sigma_xl": 5.0,
        "population_growth": 0.030,
        "terminal_z": 3.0,
        "path": "wage_collapse_probe_sigma5_n030_z3_paths.csv",
        "solver_summary": "wage_collapse_probe_sigma5_n030_z3_summary.csv",
    },
    {
        "key": "sigma6_2_n030",
        "label": "sigma_XL = 6.20, n = 3.0%",
        "sigma_xl": 6.2,
        "population_growth": 0.030,
        "terminal_z": 10.0,
        "path": "wage_collapse_probe_sigma6_2_n030_z10_paths.csv",
        "solver_summary": "wage_collapse_probe_sigma6_2_n030_z10_summary.csv",
    },
    {
        "key": "sigma6_28_n039",
        "label": "sigma_XL = 6.28, n = 3.9%",
        "sigma_xl": 6.28,
        "population_growth": 0.039,
        "terminal_z": 50.0,
        "path": "wage_collapse_probe_sigma6_28_n039_z50_paths.csv",
        "solver_summary": "wage_collapse_probe_sigma6_28_n039_z50_summary.csv",
    },
)

RESIDUAL_FIELDS = (
    "monopoly_foc_log_error",
    "research_compute_foc_log_error",
    "research_human_foc_log_error",
    "labor_market_error",
    "euler_residual",
    "capital_law_residual",
    "capability_law_residual",
    "consumption_euler_path_residual",
    "shadow_costate_residual",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, float | str]]) -> None:
    materialized = list(rows)
    fieldnames = list(
        dict.fromkeys(key for row in materialized for key in row.keys())
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialized)


def maximum_drawdown(log_wages: list[float]) -> tuple[float, int, int]:
    running_peak = log_wages[0]
    running_peak_index = 0
    worst = (0.0, 0, 0)
    for index, log_wage in enumerate(log_wages):
        if log_wage > running_peak:
            running_peak = log_wage
            running_peak_index = index
        drawdown = math.exp(log_wage - running_peak) - 1.0
        if drawdown < worst[0]:
            worst = (drawdown, running_peak_index, index)
    return worst


def summarize(
    scenario: dict[str, float | str],
    rows: list[dict[str, str]],
    solver_summary: dict[str, str],
) -> dict[str, float | str]:
    log_wages = [float(row["log_wage"]) for row in rows]
    wage_growth = [float(row["wage_growth"]) for row in rows]
    drawdown, peak_index, trough_index = maximum_drawdown(log_wages)
    minimum_growth_index = min(
        range(len(rows)), key=lambda index: wage_growth[index]
    )
    minimum_level_index = min(
        range(len(rows)), key=lambda index: log_wages[index]
    )
    first_negative = next(
        (
            float(row["time"])
            for row in rows
            if float(row["wage_growth"]) < 0.0
        ),
        math.nan,
    )
    maximum_equation_residual = max(
        abs(float(row[field]))
        for row in rows
        for field in RESIDUAL_FIELDS
        if row.get(field, "") not in ("", "nan")
    )
    maximum_resource_residual = max(
        abs(float(row["resource_share_sum"]) - 1.0) for row in rows
    )
    minimum_soc_margin = min(
        float(row["monopoly_soc_margin"]) for row in rows
    )
    partial_equilibrium_wage_threshold = 1.0 / 0.33
    return {
        "scenario": scenario["key"],
        "sigma_xl": scenario["sigma_xl"],
        "population_growth": scenario["population_growth"],
        "terminal_output_capital_ratio": scenario["terminal_z"],
        "partial_equilibrium_wage_threshold": (
            partial_equilibrium_wage_threshold
        ),
        "boundary_is_in_proved_region": (
            float(scenario["sigma_xl"])
            < partial_equilibrium_wage_threshold
        ),
        "terminal_time": float(rows[-1]["time"]),
        "initial_wage": math.exp(log_wages[0]),
        "minimum_wage": math.exp(log_wages[minimum_level_index]),
        "minimum_wage_year": float(rows[minimum_level_index]["time"]),
        "minimum_wage_relative_to_initial_pct": 100.0
        * (math.exp(log_wages[minimum_level_index] - log_wages[0]) - 1.0),
        "maximum_peak_to_trough_decline_pct": max(0.0, -100.0 * drawdown),
        "peak_year": float(rows[peak_index]["time"]),
        "trough_year": float(rows[trough_index]["time"]),
        "trough_wage_relative_to_initial_pct": 100.0
        * (math.exp(log_wages[trough_index] - log_wages[0]) - 1.0),
        "minimum_wage_growth_pct": 100.0 * wage_growth[minimum_growth_index],
        "minimum_wage_growth_year": float(
            rows[minimum_growth_index]["time"]
        ),
        "first_negative_wage_growth_year": first_negative,
        "terminal_wage": math.exp(log_wages[-1]),
        "terminal_wage_growth_pct": 100.0 * wage_growth[-1],
        "terminal_automated_research_share": float(
            rows[-1]["automated_research_share"]
        ),
        "terminal_human_research_population_share": float(
            rows[-1]["human_research_share"]
        ),
        "max_solver_rms_residual": float(solver_summary["max_rms_residual"]),
        "max_equilibrium_equation_residual": maximum_equation_residual,
        "max_resource_constraint_residual": maximum_resource_residual,
        "minimum_monopoly_soc_margin": minimum_soc_margin,
    }


def draw_figure(
    transition_rows: dict[str, list[dict[str, float | str]]]
) -> None:
    labels = {str(item["key"]): str(item["label"]) for item in SCENARIOS}
    palette = {
        "sigma4_n012": mechanism.COLORS["blue"],
        "sigma5_n030": mechanism.COLORS["gold"],
        "sigma6_2_n030": mechanism.COLORS["orange"],
        "sigma6_28_n039": mechanism.COLORS["pink"],
    }
    markers = {
        "sigma4_n012": "circle",
        "sigma5_n030": "square",
        "sigma6_2_n030": "triangle",
        "sigma6_28_n039": "diamond",
    }

    def wage_index(
        rows: list[dict[str, float | str]], values: np.ndarray
    ) -> np.ndarray:
        return 100.0 * np.exp(values - values[0])

    percent = lambda rows, values: 100.0 * values
    mechanism.draw_multiplot(
        FIGURES / "equilibrium_wage_decline_search.png",
        "Conditional equilibrium-branch wage-decline search",
        "Full equilibrium equations; extrapolated AI-dominated boundary; displayed through Y/K = 1",
        [
            {
                "title": "Real wage index",
                "field": "log_wage",
                "transform": wage_index,
                "reference_y": 100.0,
                "format": lambda value: f"{value:.0f}",
            },
            {
                "title": "Real-wage growth",
                "field": "wage_growth",
                "transform": percent,
                "reference_y": 0.0,
                "format": lambda value: f"{value:.0f}%",
            },
            {
                "title": "Aggregate labor share",
                "field": "aggregate_labor_share",
                "transform": percent,
                "format": lambda value: f"{value:.0f}%",
            },
            {
                "title": "Automated contribution to research",
                "field": "automated_research_share",
                "transform": percent,
                "format": lambda value: f"{value:.0f}%",
            },
        ],
        transition_rows,
        labels,
        palette,
        markers,
    )


def compare_boundaries(
    scenario: str,
    short_path: str,
    long_path: str,
    comparison_window_end: float,
    short_terminal_z: float,
    long_terminal_z: float,
) -> dict[str, float | str]:
    short = read_csv(NUMERICAL / short_path)
    long = read_csv(NUMERICAL / long_path)
    short_by_time = {float(row["time"]): row for row in short}
    long_by_time = {float(row["time"]): row for row in long}
    times = sorted(
        time
        for time in short_by_time
        if time in long_by_time and time <= comparison_window_end
    )
    fields = (
        "log_wage",
        "wage_growth",
        "log_capability",
        "log_capital",
        "log_consumption",
        "log_shadow_value",
        "ai_share",
        "automated_research_share",
    )
    result: dict[str, float | str] = {
        "scenario": scenario,
        "comparison_window_end": comparison_window_end,
        "short_terminal_z": short_terminal_z,
        "long_terminal_z": long_terminal_z,
    }
    for field in fields:
        result[f"max_abs_difference_{field}"] = max(
            abs(
                float(short_by_time[time][field])
                - float(long_by_time[time][field])
            )
            for time in times
        )
    return result


def main() -> None:
    NUMERICAL.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    combined_rows: list[dict[str, float | str]] = []
    summaries: list[dict[str, float | str]] = []
    transition_rows: dict[str, list[dict[str, float | str]]] = {}

    for scenario in SCENARIOS:
        rows = read_csv(NUMERICAL / str(scenario["path"]))
        solver_rows = read_csv(NUMERICAL / str(scenario["solver_summary"]))
        if len(solver_rows) != 1:
            raise ValueError(f"Expected one solver summary for {scenario['key']}.")
        summaries.append(summarize(scenario, rows, solver_rows[0]))
        enriched: list[dict[str, float | str]] = []
        for row in rows:
            item: dict[str, float | str] = dict(row)
            item.update(
                {
                    "scenario": str(scenario["key"]),
                    "sigma_xl": float(scenario["sigma_xl"]),
                    "population_growth": float(
                        scenario["population_growth"]
                    ),
                    "terminal_output_capital_ratio": float(
                        scenario["terminal_z"]
                    ),
                }
            )
            enriched.append(item)
            combined_rows.append(item)
        transition_rows[str(scenario["key"])] = [
            row
            for row in enriched
            if float(row["output_capital_ratio"]) <= 1.0
        ]

    write_csv(NUMERICAL / "wage_collapse_equilibrium_paths.csv", combined_rows)
    write_csv(NUMERICAL / "wage_collapse_equilibrium_summary.csv", summaries)
    write_csv(
        NUMERICAL / "wage_collapse_boundary_convergence.csv",
        [
            compare_boundaries(
                "sigma6_2_n030",
                "wage_collapse_probe_sigma6_2_n030_z3_paths.csv",
                "wage_collapse_probe_sigma6_2_n030_z10_paths.csv",
                60.0,
                3.0,
                10.0,
            ),
            compare_boundaries(
                "sigma6_28_n039",
                "wage_collapse_probe_sigma6_28_n039_z10_paths.csv",
                "wage_collapse_probe_sigma6_28_n039_z50_paths.csv",
                55.0,
                10.0,
                50.0,
            ),
        ],
    )
    draw_figure(transition_rows)

    for row in summaries:
        print(
            f"{row['scenario']}: min g_w={row['minimum_wage_growth_pct']:.3f}%, "
            f"drawdown={row['maximum_peak_to_trough_decline_pct']:.2f}%, "
            f"trough vs initial={row['trough_wage_relative_to_initial_pct']:.2f}%, "
            f"max residual={row['max_equilibrium_equation_residual']:.2e}"
        )


if __name__ == "__main__":
    main()
