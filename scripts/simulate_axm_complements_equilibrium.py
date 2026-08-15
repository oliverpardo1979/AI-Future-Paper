"""Equilibrium paths for gross complements in final production.

This runner deliberately imports the canonical A*M equilibrium solver rather
than duplicating its economic equations.  It solves the sigma_XL=0.75 branch
from a cold start at three horizons for both Cobb--Douglas and gross-substitute
AI research.  The longest horizon is the primary path used for figures.

The four CSV outputs are intentionally separate from the unit-elasticity and
high-substitution experiments.  Every saved path includes the full dated
equilibrium audit produced by ``simulate_axm_equilibrium.evaluate_solution``.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw

import simulate_axm_equilibrium as core


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "numerical_axm"
FIGURE_DIR = ROOT / "figures_axm"

SIGMA_XL = 0.75
SIGMA_HM_VALUES = (1.0, 2.0)
HORIZONS = (1600.0, 1800.0, 2000.0)
PRIMARY_HORIZON = 2000.0
ACCEPTANCE_TOLERANCE = 1e-5
SOLVER_TOLERANCE = 1e-6
MONOPOLY_FOC_TOLERANCE = 1e-9
PATH_STEP = 2.0

SCENARIO_KEYS = {
    1.0: "axm_complements_sigma_xl_075_hm_1",
    2.0: "axm_complements_sigma_xl_075_hm_2",
}
LABELS = {
    1.0: "Cobb-Douglas research (research elasticity = 1)",
    2.0: "Substitutable research (research elasticity = 2)",
}
PALETTE = {
    1.0: core.mechanism.COLORS["blue"],
    2.0: core.mechanism.COLORS["orange"],
}
LINE_STYLES = {1.0: "solid", 2.0: "dashed"}
MARKERS = {1.0: "circle", 2.0: "square"}


def initial_state() -> tuple[float, float, float]:
    """Use the same predetermined stocks as the canonical A*M experiments."""

    reference = replace(core.Parameters(), sigma_xl=1.0, sigma_hm=2.0)
    seed = core.fixed_share_guess(
        reference,
        (1.0, 1.0, 1.0),
        horizon=1.0,
        mesh=np.asarray([0.0]),
    )
    return (math.exp(float(seed[0, 0])), 1.0, 1.0)


def add_provenance(
    rows: list[dict[str, float | str]],
    sigma_hm: float,
    horizon: float,
) -> list[dict[str, float | str]]:
    """Attach parameters that identify the independently solved path."""

    return [
        {
            "scenario": row["scenario"],
            "sigma_xl": SIGMA_XL,
            "sigma_hm": sigma_hm,
            "horizon": horizon,
            "acceptance_tolerance": ACCEPTANCE_TOLERANCE,
            "solver_tolerance": SOLVER_TOLERANCE,
            **{key: value for key, value in row.items() if key != "scenario"},
        }
        for row in rows
    ]


def terminal_shadow_value(
    terminal: dict[str, float | str],
    targets: dict[str, float | str],
) -> float:
    object_name = str(targets["terminal_shadow_object"])
    if object_name == "profit_shadow_ratio":
        return float(terminal["profit_shadow_ratio"])
    if object_name == "shadow_capability_output_ratio":
        return float(terminal["shadow_capability_to_output"])
    raise ValueError(f"Unknown terminal shadow object: {object_name}")


def accepted_path_exp_diagnostics(
    rows: list[dict[str, float | str]],
    parameters: core.Parameters,
) -> dict[str, float]:
    """Audit bounded exponentials evaluated on the accepted reported path.

    This does not instrument rejected Newton trial states inside ``solve_bvp``;
    it certifies that evaluation of the accepted path itself uses no clipping.
    """

    arguments: list[float] = []
    for row in rows:
        log_output_capital = float(row["log_output"]) - float(
            row["log_capital"]
        )
        arguments.extend(
            [
                math.log(parameters.chi)
                + parameters.eta * float(row["log_effective_research"])
                - float(row["log_capability"]),
                log_output_capital,
                float(row["log_inference_compute"])
                - float(row["log_output"]),
                float(row["log_automated_research"])
                - float(row["log_output"]),
                float(row["log_ai_services"])
                - 2.0 * float(row["log_capability"]),
                log_output_capital,
                float(row["log_consumption"])
                - float(row["log_capital"]),
                float(row["log_inference_compute"])
                - float(row["log_capital"]),
                float(row["log_automated_research"])
                - float(row["log_capital"]),
                -float(row["log_shadow_value"]),
            ]
        )
    clipping_count = sum(
        value < -700.0 or value > 60.0 for value in arguments
    )
    return {
        "accepted_path_bounded_exp_argument_count": float(len(arguments)),
        "accepted_path_bounded_exp_clipping_count": float(clipping_count),
        "accepted_path_min_bounded_exp_argument": min(arguments),
        "accepted_path_max_bounded_exp_argument": max(arguments),
        "accepted_path_max_monopoly_root_fallback": max(
            float(row["monopoly_root_fallback"]) for row in rows
        ),
        "accepted_path_max_labor_root_fallback": max(
            float(row["labor_root_fallback"]) for row in rows
        ),
    }


def flatten_targets(
    targets: dict[str, float | str],
) -> dict[str, float | str]:
    return {f"target_{key}": value for key, value in targets.items()}


def validate_path(
    name: str,
    solution: object,
    rows: list[dict[str, float | str]],
    diagnostics: dict[str, float],
) -> None:
    """Reject numerical paths that do not satisfy the dated equilibrium."""

    if not bool(solution.success):
        raise RuntimeError(f"{name}: solver failed: {solution.message}")
    max_rms_residual = float(np.max(solution.rms_residuals))
    if max_rms_residual > SOLVER_TOLERANCE * (1.0 + 1e-8):
        raise RuntimeError(
            f"{name}: RMS residual {max_rms_residual:.3e} exceeds "
            f"tol={SOLVER_TOLERANCE:.1e}."
        )

    residual_fields = [
        field
        for field in diagnostics
        if field.startswith("max_abs_")
    ]
    worst_equilibrium_residual = max(
        diagnostics[field] for field in residual_fields
    )
    if worst_equilibrium_residual > ACCEPTANCE_TOLERANCE:
        raise RuntimeError(
            f"{name}: dated-equilibrium residual "
            f"{worst_equilibrium_residual:.3e} exceeds tolerance."
        )
    monopoly_foc_error = diagnostics["max_abs_monopoly_foc_log_error"]
    if monopoly_foc_error > MONOPOLY_FOC_TOLERANCE:
        raise RuntimeError(
            f"{name}: monopoly FOC error {monopoly_foc_error:.3e} exceeds "
            f"the independent gate {MONOPOLY_FOC_TOLERANCE:.1e}."
        )
    if diagnostics["minimum_monopoly_soc_margin"] <= 0.0:
        raise RuntimeError(f"{name}: monopoly second-order condition failed.")

    minimum_fields = (
        "minimum_consumption_share",
        "minimum_investment_share",
        "minimum_inference_share",
        "minimum_research_resource_share",
        "minimum_human_research_share",
        "minimum_production_labor_share",
    )
    if min(diagnostics[field] for field in minimum_fields) <= 0.0:
        raise RuntimeError(f"{name}: an interior allocation became nonpositive.")
    if max(float(row["monopoly_root_fallback"]) for row in rows) > 0.0:
        raise RuntimeError(f"{name}: monopoly root fallback used on saved path.")
    if max(float(row["labor_root_fallback"]) for row in rows) > 0.0:
        raise RuntimeError(f"{name}: labor root fallback used on saved path.")


def solve_cold_path(
    sigma_hm: float,
    horizon: float,
    stocks: tuple[float, float, float],
) -> tuple[
    object,
    dict[str, float | str],
    list[dict[str, float | str]],
    dict[str, float],
]:
    """Solve one horizon without importing a previous numerical solution."""

    parameters = replace(
        core.Parameters(),
        sigma_xl=SIGMA_XL,
        sigma_hm=sigma_hm,
    )
    solution, targets = core.solve_equilibrium(
        parameters,
        stocks,
        horizon=horizon,
        tolerance=SOLVER_TOLERANCE,
    )
    scenario = f"{SCENARIO_KEYS[sigma_hm]}_T_{horizon:g}"
    raw_rows = core.evaluate_solution(
        scenario,
        solution,
        parameters,
        horizon,
        step=PATH_STEP,
        initial_population=stocks[2],
    )
    rows = add_provenance(raw_rows, sigma_hm, horizon)
    diagnostics = core.path_diagnostics(rows, parameters)
    validate_path(scenario, solution, rows, diagnostics)
    return solution, targets, rows, diagnostics


def result_record(
    sigma_hm: float,
    horizon: float,
    solution: object,
    targets: dict[str, float | str],
    rows: list[dict[str, float | str]],
    diagnostics: dict[str, float],
    stocks: tuple[float, float, float],
) -> dict[str, float | str]:
    initial = rows[0]
    terminal = rows[-1]
    shadow_value = terminal_shadow_value(terminal, targets)
    parameters = replace(
        core.Parameters(), sigma_xl=SIGMA_XL, sigma_hm=sigma_hm
    )
    exp_diagnostics = accepted_path_exp_diagnostics(rows, parameters)
    record: dict[str, float | str] = {
        "scenario": str(initial["scenario"]),
        "sigma_xl": SIGMA_XL,
        "sigma_hm": sigma_hm,
        "horizon": horizon,
        "cold_start": 1.0,
        "acceptance_tolerance": ACCEPTANCE_TOLERANCE,
        "solver_tolerance": SOLVER_TOLERANCE,
        "monopoly_foc_tolerance": MONOPOLY_FOC_TOLERANCE,
        "solver_success": float(bool(solution.success)),
        "solver_status": float(solution.status),
        "solver_message": str(solution.message),
        "mesh_nodes": float(solution.x.size),
        "max_rms_residual": float(np.max(solution.rms_residuals)),
        "initial_capital_stock": stocks[0],
        "initial_capability_stock": stocks[1],
        "initial_population": stocks[2],
        "initial_log_consumption": float(initial["log_consumption"]),
        "initial_log_shadow_value": float(initial["log_shadow_value"]),
        "initial_consumption_share": float(initial["consumption_share"]),
        "initial_capability_growth": float(initial["capability_growth"]),
        "terminal_capital_growth": float(terminal["capital_growth"]),
        "terminal_output_growth": float(terminal["output_growth"]),
        "terminal_output_per_capita_growth": float(
            terminal["output_per_capita_growth"]
        ),
        "terminal_consumption_per_capita_growth": float(
            terminal["consumption_per_capita_growth"]
        ),
        "terminal_capability_growth": float(terminal["capability_growth"]),
        "terminal_net_interest_rate": float(terminal["net_capital_return"]),
        "terminal_consumption_share": float(terminal["consumption_share"]),
        "terminal_resource_share_sum": float(terminal["resource_share_sum"]),
        "terminal_ai_share": float(terminal["ai_share"]),
        "terminal_ai_labor_ratio": float(terminal["ai_labor_ratio"]),
        "terminal_automated_research_share": float(
            terminal["automated_research_share"]
        ),
        "terminal_human_research_share": float(
            terminal["human_research_share"]
        ),
        "terminal_shadow_object_value": shadow_value,
        "terminal_consumption_target_error": abs(
            float(terminal["consumption_share"])
            - float(targets["consumption_share"])
        ),
        "terminal_shadow_target_error": abs(
            shadow_value - float(targets["terminal_shadow_target"])
        ),
        "terminal_aggregate_growth_target_error": abs(
            float(terminal["output_growth"])
            - float(targets["aggregate_growth"])
        ),
        "terminal_capability_growth_target_error": abs(
            float(terminal["capability_growth"])
            - float(targets["capability_growth"])
        ),
        "terminal_ai_share_target_error": abs(
            float(terminal["ai_share"])
            - float(targets["limiting_ai_share"])
        ),
        "terminal_ai_labor_ratio_target_error": abs(
            float(terminal["ai_labor_ratio"])
            - float(targets["limiting_ai_labor_ratio"])
        ),
        "terminal_net_interest_target_error": abs(
            float(terminal["net_capital_return"])
            - float(targets["limiting_net_interest_rate"])
        ),
        **flatten_targets(targets),
        **exp_diagnostics,
        **diagnostics,
    }
    return record


def nice_ticks(lower: float, upper: float, count: int = 5) -> list[float]:
    return core.mechanism.nice_ticks(lower, upper, count)


def draw_styled_line(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[float, float]],
    color: str,
    style: str,
    width: int = 6,
) -> None:
    """Draw a continuous or globally phased dashed polyline."""

    if len(points) < 2:
        return
    if style == "solid":
        draw.line(points, fill=color, width=width, joint="curve")
        return

    dash_pattern = (22.0, 13.0)
    pattern_index = 0
    pattern_left = dash_pattern[0]
    drawing = True
    for start, end in zip(points[:-1], points[1:]):
        x0, y0 = start
        x1, y1 = end
        segment_length = math.hypot(x1 - x0, y1 - y0)
        if segment_length <= 1e-12:
            continue
        traversed = 0.0
        while traversed < segment_length - 1e-12:
            travel = min(pattern_left, segment_length - traversed)
            left_weight = traversed / segment_length
            right_weight = (traversed + travel) / segment_length
            segment_start = (
                x0 + (x1 - x0) * left_weight,
                y0 + (y1 - y0) * left_weight,
            )
            segment_end = (
                x0 + (x1 - x0) * right_weight,
                y0 + (y1 - y0) * right_weight,
            )
            if drawing:
                draw.line(
                    (segment_start, segment_end),
                    fill=color,
                    width=width,
                )
            traversed += travel
            pattern_left -= travel
            if pattern_left <= 1e-12:
                pattern_index = (pattern_index + 1) % len(dash_pattern)
                pattern_left = dash_pattern[pattern_index]
                drawing = pattern_index % 2 == 0


Transform = Callable[
    [list[dict[str, float | str]], np.ndarray], np.ndarray
]


def draw_figure(
    output_path: Path,
    title: str,
    subtitle: str,
    panels: list[dict[str, object]],
    paths: dict[float, list[dict[str, float | str]]],
) -> None:
    """Draw a two-column figure with explicit color and line encodings."""

    columns = 2
    panel_rows = math.ceil(len(panels) / columns)
    panel_height = 605
    row_gap = 80
    header_height = 245
    bottom_margin = 65
    width = 2400
    height = (
        header_height
        + panel_rows * panel_height
        + max(panel_rows - 1, 0) * row_gap
        + bottom_margin
    )
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = core.mechanism.load_font(48, bold=True)
    subtitle_font = core.mechanism.load_font(28)
    panel_font = core.mechanism.load_font(30, bold=True)
    axis_font = core.mechanism.load_font(23)
    legend_font = core.mechanism.load_font(24)
    ink = core.mechanism.COLORS["ink"]
    muted = core.mechanism.COLORS["muted"]
    grid = core.mechanism.COLORS["grid"]

    draw.text((120, 65), title, fill=ink, font=title_font)
    draw.text((120, 130), subtitle, fill=muted, font=subtitle_font)
    column_boxes = ((120, 1150), (1270, 2300))
    boxes = []
    for panel_index in range(len(panels)):
        row = panel_index // columns
        column = panel_index % columns
        top = header_height + row * (panel_height + row_gap)
        left, right = column_boxes[column]
        boxes.append((left, top, right, top + panel_height))

    for panel, box in zip(panels, boxes):
        left, top, right, bottom = box
        plot_left, plot_top = left + 120, top + 75
        plot_right, plot_bottom = right - 35, bottom - 85
        draw.text((left, top), str(panel["title"]), fill=ink, font=panel_font)

        transformed: dict[float, tuple[np.ndarray, np.ndarray]] = {}
        all_x: list[float] = []
        all_y: list[float] = []
        field = str(panel["field"])
        transform = panel.get("transform")
        for sigma_hm, rows in paths.items():
            x_values = np.asarray([float(row["time"]) for row in rows])
            y_values = np.asarray([float(row[field]) for row in rows])
            if callable(transform):
                y_values = transform(rows, y_values)
            valid = np.isfinite(x_values) & np.isfinite(y_values)
            x_values, y_values = x_values[valid], y_values[valid]
            transformed[sigma_hm] = (x_values, y_values)
            all_x.extend(x_values.tolist())
            all_y.extend(y_values.tolist())

        references = [float(value) for value in panel.get("references", [])]
        all_y.extend(references)
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        padding = 0.08 * max(y_max - y_min, 1e-8)
        y_min -= padding
        y_max += padding
        if y_max - y_min <= 1e-12:
            y_min -= 1.0
            y_max += 1.0

        for tick in nice_ticks(y_min, y_max, 5):
            y_pixel = plot_bottom - (tick - y_min) / (y_max - y_min) * (
                plot_bottom - plot_top
            )
            draw.line(
                (plot_left, y_pixel, plot_right, y_pixel),
                fill=grid,
                width=2,
            )
            formatter = panel.get("format", lambda value: f"{value:.2f}")
            label = formatter(tick) if callable(formatter) else f"{tick:.2f}"
            label_box = draw.textbbox((0, 0), label, font=axis_font)
            draw.text(
                (plot_left - 15 - label_box[2], y_pixel - 12),
                label,
                fill=muted,
                font=axis_font,
            )

        for tick in nice_ticks(x_min, x_max, 5):
            x_pixel = plot_left + (tick - x_min) / (x_max - x_min) * (
                plot_right - plot_left
            )
            draw.line(
                (x_pixel, plot_bottom, x_pixel, plot_bottom + 8),
                fill=ink,
                width=2,
            )
            label = f"{tick:.0f}"
            label_box = draw.textbbox((0, 0), label, font=axis_font)
            draw.text(
                (x_pixel - (label_box[2] - label_box[0]) / 2, plot_bottom + 14),
                label,
                fill=muted,
                font=axis_font,
            )

        draw.line(
            (plot_left, plot_top, plot_left, plot_bottom), fill=ink, width=3
        )
        draw.line(
            (plot_left, plot_bottom, plot_right, plot_bottom), fill=ink, width=3
        )
        draw.text(
            ((plot_left + plot_right) / 2 - 35, plot_bottom + 50),
            "Years",
            fill=muted,
            font=axis_font,
        )

        for reference in references:
            reference_y = plot_bottom - (reference - y_min) / (y_max - y_min) * (
                plot_bottom - plot_top
            )
            draw_styled_line(
                draw,
                [(plot_left, reference_y), (plot_right, reference_y)],
                ink,
                "dashed",
                width=3,
            )

        for sigma_hm, (x_values, y_values) in transformed.items():
            points = [
                (
                    plot_left
                    + (x_value - x_min) / (x_max - x_min)
                    * (plot_right - plot_left),
                    plot_bottom
                    - (y_value - y_min) / (y_max - y_min)
                    * (plot_bottom - plot_top),
                )
                for x_value, y_value in zip(x_values, y_values)
            ]
            draw_styled_line(
                draw,
                points,
                PALETTE[sigma_hm],
                LINE_STYLES[sigma_hm],
            )
            marker_step = max(1, len(points) // 9)
            marker_points = points[::marker_step]
            if marker_points and marker_points[-1] != points[-1]:
                marker_points.append(points[-1])
            for x_value, y_value in marker_points:
                core.mechanism.draw_marker(
                    draw,
                    x_value,
                    y_value,
                    PALETTE[sigma_hm],
                    MARKERS[sigma_hm],
                    radius=7,
                )

    legend_x, legend_y = 130, 190
    for sigma_hm in SIGMA_HM_VALUES:
        draw_styled_line(
            draw,
            [(legend_x, legend_y + 13), (legend_x + 52, legend_y + 13)],
            PALETTE[sigma_hm],
            LINE_STYLES[sigma_hm],
        )
        core.mechanism.draw_marker(
            draw,
            legend_x + 26,
            legend_y + 13,
            PALETTE[sigma_hm],
            MARKERS[sigma_hm],
            radius=7,
        )
        draw.text(
            (legend_x + 66, legend_y),
            LABELS[sigma_hm],
            fill=ink,
            font=legend_font,
        )
        legend_x += int(
            66 + draw.textlength(LABELS[sigma_hm], font=legend_font) + 65
        )

    image.save(output_path, dpi=(220, 220))


def draw_figures(
    primary_paths: dict[float, list[dict[str, float | str]]],
    targets_by_sigma: dict[float, dict[str, float | str]],
) -> None:
    percent = lambda rows, values: 100.0 * values
    log_change = lambda rows, values: values - values[0]
    limiting_interest = float(
        targets_by_sigma[1.0]["limiting_net_interest_rate"]
    )
    limiting_ai_share = float(targets_by_sigma[1.0]["limiting_ai_share"])
    limiting_production_labor_share = (
        1.0 - core.Parameters().alpha
    ) * (1.0 - limiting_ai_share)

    draw_figure(
        FIGURE_DIR / "axm_complements_macro_prices.png",
        "Gross complements: macroeconomic outcomes",
        "Natural-log changes from date zero; the annual net interest rate is a linear percent",
        [
            {
                "title": "Output per capita: change in ln(Y/N)",
                "field": "log_output_per_capita",
                "transform": log_change,
            },
            {
                "title": "Consumption per capita: change in ln(C/N)",
                "field": "log_consumption_per_capita",
                "transform": log_change,
            },
            {
                "title": "Real wage: change in ln(w)",
                "field": "log_wage",
                "transform": log_change,
            },
            {
                "title": "Net interest rate (percent)",
                "field": "net_capital_return",
                "transform": percent,
                "format": lambda value: f"{value:.1f}%",
                "references": [100.0 * limiting_interest],
            },
        ],
        primary_paths,
    )
    draw_figure(
        FIGURE_DIR / "axm_complements_factor_shares_automation.png",
        "Gross complements: production and labor allocation",
        "Ratios and shares use linear scales; dashed lines mark conditional analytical limits",
        [
            {
                "title": "Production labor / population, L/N (percent)",
                "field": "production_labor_population_share",
                "transform": percent,
                "format": lambda value: f"{value:.2f}%",
                "references": [100.0],
            },
            {
                "title": "AI services / production labor, X/L",
                "field": "ai_labor_ratio",
                "format": lambda value: f"{value:.2f}",
                "references": [
                    float(targets_by_sigma[1.0]["limiting_ai_labor_ratio"])
                ],
            },
            {
                "title": "AI share in the L--X composite, s_X (percent)",
                "field": "ai_share",
                "transform": percent,
                "format": lambda value: f"{value:.0f}%",
                "references": [100.0 * limiting_ai_share],
            },
            {
                "title": "Inference resources / output, U/Y (percent)",
                "field": "inference_share",
                "transform": percent,
                "format": lambda value: f"{value:.1f}%",
                "references": [0.0],
            },
            {
                "title": "Production labor income / output, wL/Y",
                "field": "production_labor_share",
                "transform": percent,
                "format": lambda value: f"{value:.0f}%",
                "references": [100.0 * limiting_production_labor_share],
            },
            {
                "title": "Total labor income / output, wN/Y",
                "field": "aggregate_labor_share",
                "transform": percent,
                "format": lambda value: f"{value:.0f}%",
                "references": [100.0 * limiting_production_labor_share],
            },
        ],
        primary_paths,
    )
    draw_figure(
        FIGURE_DIR / "axm_complements_research_resources.png",
        "Gross complements: AI research",
        "Capability is a natural-log change; shares and resource ratios are linear percentages",
        [
            {
                "title": "AI capability: change in ln(A)",
                "field": "log_capability",
                "transform": log_change,
            },
            {
                "title": "Human researchers / population, H/N (percent)",
                "field": "human_research_share",
                "transform": percent,
                "format": lambda value: f"{value:.1f}%",
                "references": [0.0],
            },
            {
                "title": "Automated research expenditure share, s_M",
                "field": "automated_research_share",
                "transform": percent,
                "format": lambda value: f"{value:.0f}%",
            },
            {
                "title": "Research compute / output, M/Y (percent)",
                "field": "research_resource_share",
                "transform": percent,
                "format": lambda value: f"{value:.2f}%",
                "references": [0.0],
            },
        ],
        primary_paths,
    )


def main() -> None:
    RESULT_DIR.mkdir(exist_ok=True)
    FIGURE_DIR.mkdir(exist_ok=True)
    stocks = initial_state()

    solved: dict[
        tuple[float, float],
        tuple[
            object,
            dict[str, float | str],
            list[dict[str, float | str]],
            dict[str, float],
        ],
    ] = {}
    for sigma_hm in SIGMA_HM_VALUES:
        for horizon in HORIZONS:
            solved[(sigma_hm, horizon)] = solve_cold_path(
                sigma_hm, horizon, stocks
            )

    horizon_paths: list[dict[str, float | str]] = []
    robustness: list[dict[str, float | str]] = []
    for sigma_hm in SIGMA_HM_VALUES:
        primary_record: dict[str, float | str] | None = None
        horizon_records: list[dict[str, float | str]] = []
        for horizon in HORIZONS:
            solution, targets, rows, diagnostics = solved[(sigma_hm, horizon)]
            horizon_paths.extend(rows)
            record = result_record(
                sigma_hm,
                horizon,
                solution,
                targets,
                rows,
                diagnostics,
                stocks,
            )
            horizon_records.append(record)
            if math.isclose(horizon, PRIMARY_HORIZON):
                primary_record = record
        if primary_record is None:
            raise RuntimeError(f"Missing T={PRIMARY_HORIZON:g} primary path.")
        for record in horizon_records:
            record["initial_log_consumption_distance_to_T2000"] = abs(
                float(record["initial_log_consumption"])
                - float(primary_record["initial_log_consumption"])
            )
            record["initial_log_shadow_distance_to_T2000"] = abs(
                float(record["initial_log_shadow_value"])
                - float(primary_record["initial_log_shadow_value"])
            )
            robustness.append(record)

    primary_paths: dict[float, list[dict[str, float | str]]] = {}
    primary_rows: list[dict[str, float | str]] = []
    summaries: list[dict[str, float | str]] = []
    targets_by_sigma: dict[float, dict[str, float | str]] = {}
    for sigma_hm in SIGMA_HM_VALUES:
        solution, targets, rows, diagnostics = solved[
            (sigma_hm, PRIMARY_HORIZON)
        ]
        primary_name = SCENARIO_KEYS[sigma_hm]
        rows = [{**row, "scenario": primary_name} for row in rows]
        primary_paths[sigma_hm] = rows
        primary_rows.extend(rows)
        targets_by_sigma[sigma_hm] = targets
        summary = result_record(
            sigma_hm,
            PRIMARY_HORIZON,
            solution,
            targets,
            rows,
            diagnostics,
            stocks,
        )
        summary["scenario"] = primary_name
        summaries.append(summary)

    core.write_rows(
        RESULT_DIR / "complements_transition_paths.csv", primary_rows
    )
    core.write_rows(
        RESULT_DIR / "complements_transition_summary.csv", summaries
    )
    core.write_rows(
        RESULT_DIR / "complements_horizon_paths.csv", horizon_paths
    )
    core.write_rows(
        RESULT_DIR / "complements_horizon_robustness.csv", robustness
    )
    draw_figures(primary_paths, targets_by_sigma)

    for summary in summaries:
        print(
            f"{summary['scenario']}: success=1, "
            f"max_rms={float(summary['max_rms_residual']):.3e}, "
            "max_eq_residual="
            f"{max(float(summary[key]) for key in summary if key.startswith('max_abs_')):.3e}, "
            "terminal g_(Y/N)="
            f"{float(summary['terminal_output_per_capita_growth']):.3e}"
        )


if __name__ == "__main__":
    main()
