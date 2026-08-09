#!/usr/bin/env python3
"""Research Cobb--Douglas and production-CES transition experiment."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import simulate_model as model


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "numerical"
FIGURE_DIR = ROOT / "figures"


def to_percent(rows, values):
    return 100.0 * values


def main() -> None:
    RESULT_DIR.mkdir(exist_ok=True)
    FIGURE_DIR.mkdir(exist_ok=True)

    baseline, analytical = model.analytical_calibration(model.Parameters())
    initial_capital = model.calibrate_initial_capital(
        baseline, analytical["capital_output_ratio"]
    )
    initial_state = (initial_capital, 1.0, 1.0)

    specifications = {
        "sigma_hm1_sigma_xl0_5": 0.5,
        "sigma_hm1_sigma_xl0_75": 0.75,
        "sigma_hm1_sigma_xl1": 1.0,
        "sigma_hm1_sigma_xl1_1": 1.1,
        "sigma_hm1_sigma_xl2": 2.0,
    }
    parameters = {}
    paths = {}
    cutoff = 0.50
    for key, sigma_xl in specifications.items():
        candidate = replace(baseline, sigma_hm=1.0, sigma_xl=sigma_xl)
        candidate = model.calibrate_research_productivity(
            candidate, initial_state, analytical["capability_growth"]
        )
        parameters[key] = candidate
        paths[key] = model.simulate(
            key,
            candidate,
            initial_state,
            horizon=800.0,
            step=0.5,
            acceleration_cutoff=cutoff,
        )

    all_rows = [row for rows in paths.values() for row in rows]
    model.write_rows(RESULT_DIR / "sigma_hm_one_sigma_xl_paths.csv", all_rows)

    summary_rows = []
    for key, rows in paths.items():
        final = rows[-1]
        reached_cutoff = (
            float(final["capital_growth"]) >= cutoff
            or float(final["capability_growth"]) >= cutoff
        )
        summary_rows.append(
            {
                "scenario": key,
                "sigma_hm": 1.0,
                "sigma_xl": parameters[key].sigma_xl,
                "last_year": float(final["time"]),
                "acceleration_cutoff_reached": reached_cutoff,
                "initial_capability_growth_pct": 100.0
                * float(rows[0]["capability_growth"]),
                "final_capability_growth_pct": 100.0
                * float(final["capability_growth"]),
                "final_capital_growth_pct": 100.0
                * float(final["capital_growth"]),
                "final_output_per_capita_growth_pct": 100.0
                * float(final["output_per_capita_growth"]),
                "final_ai_production_share_pct": 100.0
                * float(final["ai_share"]),
                "final_automated_research_share_pct": 100.0
                * float(final["automated_research_share"]),
                "final_human_research_population_pct": 100.0
                * float(final["human_research_share"]),
                "initial_production_labor_share_pct": 100.0
                * float(rows[0]["production_labor_income_share"]),
                "final_production_labor_share_pct": 100.0
                * float(final["production_labor_income_share"]),
                "production_labor_share_change_pp": 100.0
                * (
                    float(final["production_labor_income_share"])
                    - float(rows[0]["production_labor_income_share"])
                ),
                "initial_aggregate_labor_share_pct": 100.0
                * float(rows[0]["aggregate_labor_income_share"]),
                "final_aggregate_labor_share_pct": 100.0
                * float(final["aggregate_labor_income_share"]),
                "aggregate_labor_share_change_pp": 100.0
                * (
                    float(final["aggregate_labor_income_share"])
                    - float(rows[0]["aggregate_labor_income_share"])
                ),
                "minimum_consumption_share_pct": 100.0
                * min(float(row["consumption_share"]) for row in rows),
            }
        )
    model.write_rows(RESULT_DIR / "sigma_hm_one_sigma_xl_summary.csv", summary_rows)

    growth_keys = (
        "sigma_hm1_sigma_xl1",
        "sigma_hm1_sigma_xl1_1",
        "sigma_hm1_sigma_xl2",
    )
    growth_paths = {key: paths[key] for key in growth_keys}
    model.draw_multiplot(
        FIGURE_DIR / "numerical_sigma_hm_one_sigma_xl.png",
        "Human-essential research does not ensure balanced growth",
        "Research is Cobb-Douglas (σ_HM = 1); production substitution varies",
        [
            {
                "title": "Capability growth",
                "field": "capability_growth",
                "transform": to_percent,
                "format": lambda value: f"{value:.0f}%",
                "ylim": (0.0, 15.0),
                "xlim": (0.0, 800.0),
            },
            {
                "title": "Capital growth",
                "field": "capital_growth",
                "transform": to_percent,
                "format": lambda value: f"{value:.0f}%",
                "ylim": (0.0, 55.0),
                "xlim": (0.0, 800.0),
            },
            {
                "title": "Output growth per capita",
                "field": "output_per_capita_growth",
                "transform": to_percent,
                "format": lambda value: f"{value:.0f}%",
                "ylim": (0.0, 75.0),
                "xlim": (0.0, 800.0),
            },
            {
                "title": "Human research share of population",
                "field": "human_research_share",
                "transform": to_percent,
                "format": lambda value: f"{value:.0f}%",
                "ylim": (0.0, 60.0),
                "xlim": (0.0, 800.0),
            },
        ],
        growth_paths,
        {
            "sigma_hm1_sigma_xl1": "σ_XL = 1",
            "sigma_hm1_sigma_xl1_1": "σ_XL = 1.1",
            "sigma_hm1_sigma_xl2": "σ_XL = 2",
        },
        {
            "sigma_hm1_sigma_xl1": model.COLORS["blue"],
            "sigma_hm1_sigma_xl1_1": model.COLORS["olive"],
            "sigma_hm1_sigma_xl2": model.COLORS["gold"],
        },
        {
            "sigma_hm1_sigma_xl1": "circle",
            "sigma_hm1_sigma_xl1_1": "square",
            "sigma_hm1_sigma_xl2": "triangle",
        },
    )

    low_sigma_xl_keys = (
        "sigma_hm1_sigma_xl0_5",
        "sigma_hm1_sigma_xl0_75",
        "sigma_hm1_sigma_xl1",
    )
    low_sigma_xl_paths = {key: paths[key] for key in low_sigma_xl_keys}
    model.draw_multiplot(
        FIGURE_DIR / "numerical_sigma_hm_one_low_sigma_xl.png",
        "Labor-income shares with human-essential research",
        "Research is Cobb-Douglas (σ_HM = 1); income shares are percentages of output",
        [
            {
                "title": "Production labor income share",
                "field": "production_labor_income_share",
                "transform": to_percent,
                "format": lambda value: f"{value:.0f}%",
                "ylim": (0.0, 60.0),
                "xlim": (0.0, 800.0),
            },
            {
                "title": "Aggregate labor income share",
                "field": "aggregate_labor_income_share",
                "transform": to_percent,
                "format": lambda value: f"{value:.0f}%",
                "ylim": (0.0, 60.0),
                "xlim": (0.0, 800.0),
            },
            {
                "title": "AI expenditure share in service composite",
                "field": "ai_share",
                "transform": to_percent,
                "format": lambda value: f"{value:.0f}%",
                "ylim": (0.0, 80.0),
                "xlim": (0.0, 800.0),
            },
            {
                "title": "Human research share of population",
                "field": "human_research_share",
                "transform": to_percent,
                "format": lambda value: f"{value:.0f}%",
                "ylim": (0.0, 6.0),
                "xlim": (0.0, 800.0),
            },
        ],
        low_sigma_xl_paths,
        {
            "sigma_hm1_sigma_xl0_5": "σ_XL = 0.5",
            "sigma_hm1_sigma_xl0_75": "σ_XL = 0.75",
            "sigma_hm1_sigma_xl1": "σ_XL = 1",
        },
        {
            "sigma_hm1_sigma_xl0_5": model.COLORS["blue"],
            "sigma_hm1_sigma_xl0_75": model.COLORS["gold"],
            "sigma_hm1_sigma_xl1": model.COLORS["olive"],
        },
        {
            "sigma_hm1_sigma_xl0_5": "circle",
            "sigma_hm1_sigma_xl0_75": "square",
            "sigma_hm1_sigma_xl1": "triangle",
        },
    )

    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()
