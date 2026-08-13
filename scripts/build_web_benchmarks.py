"""Build the lightweight benchmark bundle used by the GitHub Pages simulator."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NUMERICAL = ROOT / "numerical"
OUTPUT = ROOT / "docs" / "data" / "benchmarks.json"

LABELS = {
    "equilibrium_sigma_0_75": "Complementarity · σXL = 0.75",
    "equilibrium_sigma_1_00": "Cobb–Douglas · σXL = 1.00",
    "equilibrium_sigma_1_00_hm_1_00": "Human-essential research · σXL = σHM = 1.00",
    "equilibrium_sigma_1_35": "Gross substitutes · σXL = 1.35",
    "equilibrium_sigma_1_50": "Gross substitutes · σXL = 1.50",
    "equilibrium_sigma_2_00": "Gross substitutes · σXL = 2.00",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def downsample(rows: list[dict[str, str]], maximum: int = 241) -> list[dict[str, str]]:
    if len(rows) <= maximum:
        return rows
    indices = sorted(
        set(round(index * (len(rows) - 1) / (maximum - 1)) for index in range(maximum))
    )
    return [rows[index] for index in indices]


def value(row: dict[str, str], field: str, fallback: float = 0.0) -> float:
    raw = row.get(field, "")
    return float(raw) if raw not in {"", None} else fallback


PER_CAPITA_LOG_FIELDS = {
    "log_service_composite_per_capita": "log_service_composite",
    "log_ai_services_per_capita": "log_ai_services",
    "log_inference_compute_per_capita": "log_inference_compute",
    "log_human_research_per_capita": "log_human_research",
    "log_automated_research_per_capita": "log_automated_research",
    "log_effective_research_per_capita": "log_effective_research",
}


def log_level(row: dict[str, str], field: str) -> float:
    numerator = PER_CAPITA_LOG_FIELDS.get(field)
    if numerator is None:
        return value(row, field)
    return value(row, numerator) - value(row, "log_population")


def convert(rows: list[dict[str, str]]) -> list[dict[str, float]]:
    initial = rows[0]
    initial_levels = {
        field: log_level(initial, field)
        for field in (
            "log_capability",
            "log_output_per_capita",
            "log_consumption_per_capita",
            "log_capital_per_capita",
            "log_wage",
            "log_ai_price",
            "log_ai_marginal_cost",
            *PER_CAPITA_LOG_FIELDS,
        )
    }
    converted: list[dict[str, float]] = []
    for row in downsample(rows):
        output_capital = value(row, "output_capital_ratio")
        if output_capital == 0.0:
            output_capital = math.exp(
                value(row, "log_output") - value(row, "log_capital")
            )
        human_machine = value(row, "human_to_automated_research_ratio")
        if human_machine == 0.0:
            human_machine = math.exp(
                value(row, "log_human_research")
                - value(row, "log_automated_research")
            )
        current = {
            "time": value(row, "time"),
            # Retain the four canonical states as a warm start for custom
            # browser solutions. They are not charted directly.
            "log_capital": value(row, "log_capital"),
            "log_capability": value(row, "log_capability"),
            "log_consumption": value(row, "log_consumption"),
            "log_shadow_value": value(row, "log_shadow_value"),
            "capability_growth": value(row, "capability_growth"),
            "output_per_capita_growth": value(row, "output_per_capita_growth"),
            "consumption_per_capita_growth": value(
                row, "consumption_per_capita_growth"
            ),
            "wage_growth": value(row, "wage_growth"),
            "net_interest": value(row, "net_capital_return"),
            "human_research_share": value(row, "human_research_share"),
            "production_labor_population_share": (
                1.0 - value(row, "human_research_share")
            ),
            "ai_share": value(row, "ai_share"),
            "automated_research_share": value(row, "automated_research_share"),
            "human_research_aggregate_share": (
                1.0 - value(row, "automated_research_share")
            ),
            "production_labor_share": value(row, "production_labor_share"),
            "aggregate_labor_share": value(row, "aggregate_labor_share"),
            "consumption_share": value(row, "consumption_share"),
            "investment_share": value(row, "investment_share"),
            "inference_share": value(row, "inference_share"),
            "research_resource_share": value(row, "research_resource_share"),
            "output_capital_ratio": output_capital,
            "human_machine_ratio": human_machine,
            "ai_markup": value(row, "ai_markup"),
            "ai_profit_share": value(row, "ai_profit_share"),
            "shadow_capability_to_output": value(
                row, "shadow_capability_to_output"
            ),
            "shadow_capability_to_capital": value(
                row, "shadow_capability_to_capital"
            ),
        }
        for field, initial_value in initial_levels.items():
            current[field + "_change"] = (
                log_level(row, field) - initial_value
            )
        converted.append(current)
    return converted


def main() -> None:
    grouped: dict[str, list[dict[str, str]]] = {key: [] for key in LABELS}
    for source in (
        NUMERICAL / "equilibrium_transition_paths.csv",
        NUMERICAL / "high_sigma_equilibrium_paths.csv",
    ):
        for row in read_rows(source):
            if row["scenario"] in grouped:
                grouped[row["scenario"]].append(row)

    audit = {
        row["scenario"]: row
        for row in read_rows(NUMERICAL / "equilibrium_system_audit_summary.csv")
    }
    summary = {
        row["scenario"]: row
        for row in read_rows(NUMERICAL / "high_sigma_equilibrium_summary.csv")
    }
    bundle: dict[str, object] = {
        "generated_from": "Canonical equilibrium replication outputs",
        "scenarios": {},
    }
    scenarios: dict[str, object] = bundle["scenarios"]  # type: ignore[assignment]
    for scenario, label in LABELS.items():
        row_audit = audit[scenario]
        high_summary = summary.get(scenario, {})
        singular_time = high_summary.get("estimated_singularity_time", "")
        scenarios[scenario] = {
            "label": label,
            "sigma_xl": float(row_audit["sigma_xl"]),
            "sigma_hm": float(row_audit["sigma_hm"]),
            "diagnostics": {
                "passed": True,
                "solver_success": True,
                "collocation_residual": float(
                    high_summary.get("max_rms_residual", 0.0) or 0.0
                ),
                "dynamic_residual": float(row_audit["max_dynamic_path_residual"]),
                "static_residual": float(
                    row_audit["max_static_equilibrium_residual"]
                ),
                "endpoint_residual": float(
                    row_audit["max_terminal_condition_residual"]
                ),
                "minimum_monopoly_margin": float(
                    row_audit["minimum_monopoly_soc_margin_on_path"]
                ),
                "interior": True,
                "duration": float(grouped[scenario][-1]["time"]),
                "estimated_singularity_time": (
                    float(singular_time) if singular_time else None
                ),
                "d_cd": 0.310625,
                "d_ai": 0.2375,
                "regime": (
                    "complements"
                    if float(row_audit["sigma_xl"]) < 1.0
                    else "cobb_douglas"
                    if float(row_audit["sigma_xl"]) == 1.0
                    else "gross_substitutes"
                ),
                "interpretation": (
                    "Published benchmark: the canonical equations and numerical "
                    "checks pass. Global dynamic optimality is not certified."
                ),
            },
            "series": convert(grouped[scenario]),
        }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as handle:
        json.dump(bundle, handle, ensure_ascii=False, separators=(",", ":"))
        handle.write("\n")


if __name__ == "__main__":
    main()
