"""Plot a plug-in unit-elasticity balanced-growth feedback diagnostic.

The script reads the two canonical, previously solved ``sigma_XL = 1``
candidate paths satisfying the dated equilibrium conditions. It does not call an
equilibrium solver or modify the saved paths. Following the scalar feedback
accounting used in the manuscript, it
plots

    G_T(t) = eta * s_M(t),
    G_E(t) = eta * s_M(t) * nu,
    G(t)   = eta * s_M(t) * (1 + nu),

where ``nu = omega_X / omega_L``. These formulas are balanced-growth
accounting evaluated at dated ``s_M`` values, not the transition Jacobian. The
horizontal line at one is a diagnostic reference for the limiting denominator
``D = 1 - G``; it is not, by itself, an equilibrium explosion condition.

The technological/economic terminology follows Davidson, Halperin, Houlden,
and Korinek (2026), NBER Working Paper 35155.
"""

from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "numerical_axm"
FIGURE_DIR = ROOT / "figures_axm"
PATH_FILE = RESULT_DIR / "equilibrium_transition_paths.csv"
SUMMARY_FILE = RESULT_DIR / "equilibrium_transition_summary.csv"
AUDIT_FILE = RESULT_DIR / "audit_report.csv"
OUTPUT_FILE = FIGURE_DIR / "axm_unit_elasticity_feedback_diagnostic.png"

SCENARIOS = {
    "axm_sigma_xl_1_hm_1": {
        "sigma_hm": 1.0,
        "title": "Cobb–Douglas AI research (σₕₘ = 1)",
    },
    "axm_sigma_xl_1_hm_2": {
        "sigma_hm": 2.0,
        "title": "Gross substitutes in AI research (σₕₘ = 2)",
    },
}

COLORS = {
    "technology": "#245B9E",
    "economy": "#D2601A",
    "total": "#22272E",
    "muted": "#66717E",
    "grid": "#D9DEE5",
    "light": "#F5F7FA",
    "reference": "#7C8794",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No observations found in {path}.")
    return rows


def load_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(filename, size=size)
    except OSError as exc:
        raise RuntimeError(
            "This figure requires the cross-platform DejaVu Sans fonts; "
            f"Pillow could not resolve {filename}."
        ) from exc


def numeric(row: dict[str, str], field: str) -> float:
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {field} in scenario {row.get('scenario')}.")
    return value


def validate_and_derive(
    path_rows: list[dict[str, str]],
    summary_rows: list[dict[str, str]],
    audit_rows: list[dict[str, str]],
) -> tuple[dict[str, list[dict[str, float]]], float, float]:
    """Validate canonical inputs and recover eta and nu from dated identities."""

    required = {
        "scenario",
        "time",
        "ai_share",
        "automated_research_share",
        "capability_growth",
        "research_resource_share",
        "shadow_capability_to_output",
    }
    missing = required.difference(path_rows[0])
    if missing:
        raise ValueError(f"Path file is missing columns: {sorted(missing)}")

    grouped: dict[str, list[dict[str, str]]] = {key: [] for key in SCENARIOS}
    for row in path_rows:
        scenario = row["scenario"]
        if scenario not in grouped:
            raise ValueError(f"Unexpected scenario in unit-elasticity file: {scenario}")
        grouped[scenario].append(row)

    summary_by_scenario = {row["scenario"]: row for row in summary_rows}
    if set(summary_by_scenario) != set(SCENARIOS):
        raise ValueError("The transition summary does not contain exactly the two expected scenarios.")

    eta_observations: list[float] = []
    ai_shares: list[float] = []
    derived: dict[str, list[dict[str, float]]] = {}
    for scenario, metadata in SCENARIOS.items():
        summary = summary_by_scenario[scenario]
        if not math.isclose(numeric(summary, "sigma_xl"), 1.0, abs_tol=1e-12):
            raise ValueError(f"{scenario} is not a sigma_XL=1 path.")
        if not math.isclose(
            numeric(summary, "sigma_hm"), metadata["sigma_hm"], abs_tol=1e-12
        ):
            raise ValueError(f"Unexpected sigma_HM for {scenario}.")
        if int(float(summary["solver_status"])) != 0:
            raise ValueError(f"Canonical solver status is not successful for {scenario}.")
        if numeric(summary, "max_rms_residual") > 2.0e-5:
            raise ValueError(f"Canonical collocation residual is too large for {scenario}.")

        rows = sorted(grouped[scenario], key=lambda row: numeric(row, "time"))
        if len(rows) < 8:
            raise ValueError(f"Too few dated observations for {scenario}.")
        times = [numeric(row, "time") for row in rows]
        if any(right <= left for left, right in zip(times, times[1:])):
            raise ValueError(f"Times are not strictly increasing for {scenario}.")

        for row in rows:
            s_m = numeric(row, "automated_research_share")
            s_x = numeric(row, "ai_share")
            if not (0.0 < s_m < 1.0):
                raise ValueError(f"s_M leaves the open unit interval in {scenario}.")
            if not (0.0 < s_x < 1.0):
                raise ValueError(f"s_X leaves the open unit interval in {scenario}.")
            ai_shares.append(s_x)

            # The research-compute FOC stored in the equilibrium system is
            # M/Y = eta*s_M*(qA/Y)*g_A.  Recovering eta row by row prevents the
            # plotter from silently using a parameter different from the path.
            denominator = (
                s_m
                * numeric(row, "shadow_capability_to_output")
                * numeric(row, "capability_growth")
            )
            if denominator <= 0.0:
                raise ValueError(f"Cannot recover eta from {scenario}.")
            eta_observations.append(
                numeric(row, "research_resource_share") / denominator
            )

        derived[scenario] = [{"time": time} for time in times]

    eta = statistics.median(eta_observations)
    if max(abs(value - eta) for value in eta_observations) > 2.0e-9:
        raise ValueError("The dated research FOC does not imply a common eta.")
    if not (0.0 < eta < 1.0):
        raise ValueError(f"Recovered eta is outside (0,1): {eta}")

    omega_x = statistics.median(ai_shares)
    if max(abs(value - omega_x) for value in ai_shares) > 2.0e-12:
        raise ValueError("At sigma_XL=1, the saved AI CES share is not constant.")
    omega_l = 1.0 - omega_x
    nu = omega_x / omega_l

    for scenario in SCENARIOS:
        rows = sorted(grouped[scenario], key=lambda row: numeric(row, "time"))
        for target, row in zip(derived[scenario], rows):
            s_m = numeric(row, "automated_research_share")
            target["technology"] = eta * s_m
            target["economy"] = eta * s_m * nu
            target["total"] = eta * s_m * (1.0 + nu)
            if not math.isclose(
                target["total"], target["technology"] + target["economy"],
                rel_tol=0.0, abs_tol=1.0e-14,
            ):
                raise ValueError("Feedback components do not add to total G.")

    audit = {row["object"]: row for row in audit_rows}
    for scenario in SCENARIOS:
        relevant = [
            row for key, row in audit.items()
            if key.startswith(f"{scenario}:")
        ]
        if not relevant or any(row.get("status") != "pass" for row in relevant):
            raise ValueError(
                f"The canonical audit does not pass every dated check for {scenario}."
            )
    all_saved = [
        row for key, row in audit.items() if key.startswith("all_saved_paths:")
    ]
    if not all_saved or any(row.get("status") != "pass" for row in all_saved):
        raise ValueError("The canonical audit does not pass every all-saved-path check.")
    # The manuscript's two limiting denominators provide an independent check
    # on the recovered eta and nu.  With sigma_HM=1, sbar is the constant saved
    # s_M; with sigma_HM>1, the analytical limit is sbar=1.
    sbar_cd = derived["axm_sigma_xl_1_hm_1"][-1]["total"] / (eta * (1.0 + nu))
    implied_d_cd = 1.0 - eta * sbar_cd * (1.0 + nu)
    implied_d_ai = 1.0 - eta * (1.0 + nu)
    for label, implied in (("D_CD", implied_d_cd), ("D_AI", implied_d_ai)):
        if label not in audit or audit[label].get("status") != "pass":
            raise ValueError(f"Missing passing {label} entry in audit_report.csv.")
        recorded = numeric(audit[label], "value")
        if not math.isclose(recorded, implied, rel_tol=0.0, abs_tol=2.0e-10):
            raise ValueError(
                f"Recovered parameters imply {label}={implied}, not {recorded}."
            )

    return derived, eta, nu


def patterned_line(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[float, float]],
    *,
    fill: str,
    width: int,
    pattern: tuple[float, ...] | None = None,
) -> None:
    if len(points) < 2:
        return
    if pattern is None:
        draw.line(points, fill=fill, width=width, joint="curve")
        return

    pattern_index = 0
    remaining = pattern[0]
    drawing = True
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        dx, dy = x1 - x0, y1 - y0
        segment = math.hypot(dx, dy)
        if segment <= 0.0:
            continue
        used = 0.0
        while used < segment - 1.0e-9:
            step = min(remaining, segment - used)
            start_fraction = used / segment
            end_fraction = (used + step) / segment
            start = (x0 + dx * start_fraction, y0 + dy * start_fraction)
            end = (x0 + dx * end_fraction, y0 + dy * end_fraction)
            if drawing:
                draw.line((start, end), fill=fill, width=width)
            used += step
            remaining -= step
            if remaining <= 1.0e-9:
                pattern_index = (pattern_index + 1) % len(pattern)
                remaining = pattern[pattern_index]
                drawing = pattern_index % 2 == 0


def text_right(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    value: str,
    *,
    fill: str,
    font: ImageFont.ImageFont,
) -> None:
    box = draw.textbbox((0, 0), value, font=font)
    draw.text((xy[0] - (box[2] - box[0]), xy[1]), value, fill=fill, font=font)


def draw_figure(
    data: dict[str, list[dict[str, float]]], eta: float, nu: float
) -> None:
    width, height = 2400, 1240
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(52, bold=True)
    subtitle_font = load_font(34)
    panel_font = load_font(39, bold=True)
    axis_font = load_font(38)
    legend_font = load_font(36)
    value_font = load_font(34, bold=True)

    draw.text(
        (110, 52),
        "Plug-in balanced-growth feedback at unit final-production elasticity",
        fill=COLORS["total"],
        font=title_font,
    )
    draw.text(
        (110, 116),
        "Scalar adaptation of Davidson et al. (2026); evaluated at dated sₘ, not a transition Jacobian",
        fill=COLORS["muted"],
        font=subtitle_font,
    )

    legend_items = (
        ("Technological: η × sₘ", COLORS["technology"], (20.0, 10.0), 5),
        ("Economic: η × sₘ × ν", COLORS["economy"], (4.0, 8.0), 5),
        ("Total: η × sₘ × (1 + ν)", COLORS["total"], None, 7),
        ("BGP reference: G = 1", COLORS["reference"], (16.0, 10.0), 4),
    )
    legend_origins = ((115.0, 177.0), (1210.0, 177.0), (115.0, 226.0), (1210.0, 226.0))
    for (label, color, pattern, line_width), (legend_x, legend_y) in zip(legend_items, legend_origins):
        patterned_line(
            draw,
            [(legend_x, legend_y + 13), (legend_x + 58, legend_y + 13)],
            fill=color,
            width=line_width,
            pattern=pattern,
        )
        draw.text((legend_x + 70, legend_y), label, fill=COLORS["total"], font=legend_font)

    panel_boxes = (
        (110, 305, 1160, 1110),
        (1280, 305, 2330, 1110),
    )
    y_min, y_max = 0.0, 1.05
    for (scenario, metadata), box in zip(SCENARIOS.items(), panel_boxes):
        rows = data[scenario]
        left, top, right, bottom = box
        plot_left, plot_top = left + 115, top + 78
        plot_right, plot_bottom = right - 42, bottom - 88
        draw.text((left, top), metadata["title"], fill=COLORS["total"], font=panel_font)

        for tick in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
            y_pixel = plot_bottom - (tick - y_min) / (y_max - y_min) * (plot_bottom - plot_top)
            draw.line((plot_left, y_pixel, plot_right, y_pixel), fill=COLORS["grid"], width=2)
            text_right(
                draw,
                (plot_left - 14, y_pixel - 12),
                f"{tick:.1f}",
                fill=COLORS["muted"],
                font=axis_font,
            )

        x_min, x_max = rows[0]["time"], rows[-1]["time"]
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            value = x_min + fraction * (x_max - x_min)
            x_pixel = plot_left + fraction * (plot_right - plot_left)
            draw.line((x_pixel, plot_bottom, x_pixel, plot_bottom + 8), fill=COLORS["total"], width=2)
            label = f"{value:.0f}"
            box_text = draw.textbbox((0, 0), label, font=axis_font)
            draw.text(
                (x_pixel - (box_text[2] - box_text[0]) / 2, plot_bottom + 14),
                label,
                fill=COLORS["muted"],
                font=axis_font,
            )

        draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=COLORS["total"], width=3)
        draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=COLORS["total"], width=3)
        draw.text(
            ((plot_left + plot_right) / 2 - 30, plot_bottom + 51),
            "Years",
            fill=COLORS["muted"],
            font=axis_font,
        )

        reference_y = plot_bottom - (1.0 - y_min) / (y_max - y_min) * (plot_bottom - plot_top)
        patterned_line(
            draw,
            [(plot_left, reference_y), (plot_right, reference_y)],
            fill=COLORS["reference"],
            width=4,
            pattern=(16.0, 10.0),
        )

        styles = (
            ("technology", COLORS["technology"], (20.0, 10.0), 5),
            ("economy", COLORS["economy"], (4.0, 8.0), 5),
            ("total", COLORS["total"], None, 7),
        )
        for field, color, pattern, line_width in styles:
            points: list[tuple[float, float]] = []
            for row in rows:
                x_pixel = plot_left + (row["time"] - x_min) / (x_max - x_min) * (plot_right - plot_left)
                y_pixel = plot_bottom - (row[field] - y_min) / (y_max - y_min) * (plot_bottom - plot_top)
                points.append((x_pixel, y_pixel))
            patterned_line(draw, points, fill=color, width=line_width, pattern=pattern)
            final_x, final_y = points[-1]
            draw.ellipse(
                (final_x - 6, final_y - 6, final_x + 6, final_y + 6),
                fill=color,
                outline="white",
                width=2,
            )
            text_right(
                draw,
                (plot_right - 10, final_y - 27),
                f"{rows[-1][field]:.3f}",
                fill=color,
                font=value_font,
            )

    FIGURE_DIR.mkdir(exist_ok=True)
    image.save(OUTPUT_FILE, dpi=(220, 220))


def main() -> None:
    paths = read_rows(PATH_FILE)
    summaries = read_rows(SUMMARY_FILE)
    audit = read_rows(AUDIT_FILE)
    data, eta, nu = validate_and_derive(paths, summaries, audit)
    draw_figure(data, eta, nu)

    for scenario in SCENARIOS:
        rows = data[scenario]
        print(
            f"{scenario}: G(0)={rows[0]['total']:.9f}, "
            f"G(T)={rows[-1]['total']:.9f}, max G={max(row['total'] for row in rows):.9f}"
        )
    print(f"eta={eta:.12f}; nu={nu:.12f}")
    print(f"Wrote {OUTPUT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
