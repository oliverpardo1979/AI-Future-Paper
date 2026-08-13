"""Build the empirical-motivation figures from source-specific CSV snapshots.

The underlying measurements are descriptive proxies for the scale, quality, and
use of AI. They do not identify either substitution elasticity in the model.
"""

from __future__ import annotations

import csv
import math
from datetime import date
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "empirical"
FIGURE_DIR = ROOT / "figures"

COLORS = {
    "blue": "#205493",
    "gold": "#C69214",
    "orange": "#D2601A",
    "olive": "#667A2C",
    "ink": "#22272E",
    "muted": "#66717E",
    "grid": "#D9DEE5",
    "light": "#F5F7FA",
    "light_orange": "#FFF3E8",
    "white": "#FFFFFF",
}


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def read_csv(name: str) -> list[dict[str, str]]:
    with (DATA_DIR / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> float:
    box = draw.textbbox((0, 0), text, font=font)
    return float(box[2] - box[0])


def dashed_line(
    draw: ImageDraw.ImageDraw,
    coordinates: tuple[float, float, float, float],
    fill: str,
    width: int = 3,
    dash: int = 18,
    gap: int = 12,
) -> None:
    x1, y1, x2, y2 = coordinates
    length = math.hypot(x2 - x1, y2 - y1)
    if length == 0:
        return
    dx, dy = (x2 - x1) / length, (y2 - y1) / length
    position = 0.0
    while position < length:
        end = min(position + dash, length)
        draw.line(
            (
                x1 + dx * position,
                y1 + dy * position,
                x1 + dx * end,
                y1 + dy * end,
            ),
            fill=fill,
            width=width,
        )
        position += dash + gap


def draw_header(
    draw: ImageDraw.ImageDraw,
    title: str,
    subtitle: str,
    width: int,
) -> None:
    title_font = load_font(44, bold=True)
    subtitle_font = load_font(25)
    draw.text((80, 48), title, fill=COLORS["ink"], font=title_font)
    draw.text((80, 112), subtitle, fill=COLORS["muted"], font=subtitle_font)
    draw.line((80, 158, width - 80, 158), fill=COLORS["grid"], width=2)


def draw_panel_title(
    draw: ImageDraw.ImageDraw,
    left: int,
    top: int,
    title: str,
    subtitle: str,
) -> None:
    draw.text((left, top), title, fill=COLORS["ink"], font=load_font(31, bold=True))
    draw.text((left, top + 46), subtitle, fill=COLORS["muted"], font=load_font(22))


def draw_scale_figure() -> None:
    production_rows = read_csv("ai_production_2023_2025.csv")
    energy_rows = read_csv("data_center_energy.csv")

    by_year = {int(row["year"]): row for row in production_rows}
    assert set(by_year) == {2023, 2024, 2025}
    assert math.isclose(float(by_year[2025]["qa_inference_index_2023"]), 38.98 * 38.54)
    assert math.isclose(float(by_year[2025]["qa_training_index_2023"]), 8.82 * 8.88)

    width, height = 2400, 1120
    image = Image.new("RGB", (width, height), COLORS["white"])
    draw = ImageDraw.Draw(image)
    draw_header(
        draw,
        "AI-sector scale and physical inputs",
        "Discrete source estimates; quality-adjusted quantities set 2023 = 1",
        width,
    )

    left_box = (80, 205, 1170, 1050)
    right_box = (1260, 205, 2320, 1050)

    draw_panel_title(
        draw,
        left_box[0],
        left_box[1],
        "A. U.S. AI production measures",
        "Horizontal position is the natural log of the index relative to 2023",
    )
    plot_left, plot_right = left_box[0] + 380, left_box[2] - 90
    plot_top, plot_bottom = left_box[1] + 145, left_box[3] - 130
    x_min, x_max = 0.0, 8.0
    axis_font = load_font(22)
    label_font = load_font(23)
    small_font = load_font(20)

    def x_pixel(value: float) -> float:
        return plot_left + (value - x_min) / (x_max - x_min) * (plot_right - plot_left)

    for tick in (0, 2, 4, 6, 8):
        x = x_pixel(float(tick))
        draw.line((x, plot_top, x, plot_bottom), fill=COLORS["grid"], width=2)
        tick_text = str(tick)
        draw.text(
            (x - text_width(draw, tick_text, axis_font) / 2, plot_bottom + 18),
            tick_text,
            fill=COLORS["muted"],
            font=axis_font,
        )
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=COLORS["ink"], width=3)
    axis_title = "ln(index, 2023 = 1)"
    draw.text(
        ((plot_left + plot_right) / 2 - text_width(draw, axis_title, axis_font) / 2, plot_bottom + 64),
        axis_title,
        fill=COLORS["muted"],
        font=axis_font,
    )

    metrics = [
        (
            "Nominal compute spending",
            float(by_year[2024]["compute_spending_usd_billion"]) / float(by_year[2023]["compute_spending_usd_billion"]),
            float(by_year[2025]["compute_spending_usd_billion"]) / float(by_year[2023]["compute_spending_usd_billion"]),
            COLORS["blue"],
        ),
        (
            "Raw compute capacity",
            float(by_year[2024]["raw_compute_h100e_million"]) / float(by_year[2023]["raw_compute_h100e_million"]),
            float(by_year[2025]["raw_compute_h100e_million"]) / float(by_year[2023]["raw_compute_h100e_million"]),
            COLORS["gold"],
        ),
        (
            "QA training output",
            float(by_year[2024]["qa_training_index_2023"]),
            float(by_year[2025]["qa_training_index_2023"]),
            COLORS["orange"],
        ),
        (
            "QA inference output",
            float(by_year[2024]["qa_inference_index_2023"]),
            float(by_year[2025]["qa_inference_index_2023"]),
            COLORS["olive"],
        ),
    ]
    row_gap = (plot_bottom - plot_top) / len(metrics)
    for index, (label, value_2024, value_2025, color) in enumerate(metrics):
        y = plot_top + row_gap * (index + 0.5)
        draw.text(
            (plot_left - 24 - text_width(draw, label, label_font), y - 14),
            label,
            fill=COLORS["ink"],
            font=label_font,
        )
        x_2024 = x_pixel(math.log(value_2024))
        x_2025 = x_pixel(math.log(value_2025))
        draw.line((x_2024, y, x_2025, y), fill=color, width=5)
        draw.ellipse((x_2024 - 10, y - 10, x_2024 + 10, y + 10), fill=COLORS["white"], outline=color, width=4)
        draw.ellipse((x_2025 - 11, y - 11, x_2025 + 11, y + 11), fill=color, outline=color)
        label_2024 = f"{value_2024:.1f}x"
        label_2025 = f"{value_2025:,.0f}x" if value_2025 >= 100 else f"{value_2025:.1f}x"
        draw.text((x_2024 - text_width(draw, label_2024, small_font) / 2, y - 43), label_2024, fill=COLORS["ink"], font=small_font)
        right_anchor = min(x_2025 + 14, plot_right - text_width(draw, label_2025, small_font))
        draw.text((right_anchor, y + 16), label_2025, fill=COLORS["ink"], font=small_font)

    legend_y = left_box[3] - 48
    draw.ellipse((left_box[0] + 320, legend_y - 8, left_box[0] + 336, legend_y + 8), fill=COLORS["white"], outline=COLORS["ink"], width=3)
    draw.text((left_box[0] + 348, legend_y - 13), "2024", fill=COLORS["ink"], font=small_font)
    draw.ellipse((left_box[0] + 455, legend_y - 8, left_box[0] + 471, legend_y + 8), fill=COLORS["ink"], outline=COLORS["ink"])
    draw.text((left_box[0] + 483, legend_y - 13), "2025", fill=COLORS["ink"], font=small_font)

    draw_panel_title(
        draw,
        right_box[0],
        right_box[1],
        "B. Global data-center electricity",
        "TWh; 2030 is the IEA central projection",
    )
    e_plot_left, e_plot_right = right_box[0] + 120, right_box[2] - 65
    e_plot_top, e_plot_bottom = right_box[1] + 145, right_box[3] - 130
    y_max = 1000.0

    def energy_y(value: float) -> float:
        return e_plot_bottom - value / y_max * (e_plot_bottom - e_plot_top)

    for tick in (0, 200, 400, 600, 800, 1000):
        y = energy_y(float(tick))
        draw.line((e_plot_left, y, e_plot_right, y), fill=COLORS["grid"], width=2)
        label = f"{tick:,}"
        draw.text((e_plot_left - 18 - text_width(draw, label, axis_font), y - 12), label, fill=COLORS["muted"], font=axis_font)
    draw.line((e_plot_left, e_plot_top, e_plot_left, e_plot_bottom), fill=COLORS["ink"], width=3)
    draw.line((e_plot_left, e_plot_bottom, e_plot_right, e_plot_bottom), fill=COLORS["ink"], width=3)
    draw.text((right_box[0] + 9, (e_plot_top + e_plot_bottom) / 2 - 12), "TWh", fill=COLORS["muted"], font=axis_font)

    bar_centers = [e_plot_left + 160, (e_plot_left + e_plot_right) / 2, e_plot_right - 160]
    bar_width = 155
    for row, center in zip(energy_rows, bar_centers):
        value = float(row["total_electricity_twh"])
        top = energy_y(value)
        left, right = center - bar_width / 2, center + bar_width / 2
        if row["status"] == "observed":
            draw.rectangle((left, top, right, e_plot_bottom), fill=COLORS["blue"], outline=COLORS["blue"])
        else:
            draw.rectangle((left, top, right, e_plot_bottom), fill=COLORS["white"], outline=COLORS["blue"], width=5)
        value_label = f"{value:,.0f}"
        draw.text((center - text_width(draw, value_label, label_font) / 2, top - 38), value_label, fill=COLORS["ink"], font=label_font)
        year_label = row["year"]
        draw.text((center - text_width(draw, year_label, axis_font) / 2, e_plot_bottom + 18), year_label, fill=COLORS["muted"], font=axis_font)
        if row["status"] != "observed":
            projection = "projection"
            draw.text((center - text_width(draw, projection, small_font) / 2, e_plot_bottom + 52), projection, fill=COLORS["muted"], font=small_font)

    callout = "AI-focused centers: +50% in 2025; 3x projected from 2025 to 2030"
    draw.text((right_box[0] + 70, right_box[3] - 48), callout, fill=COLORS["ink"], font=small_font)

    FIGURE_DIR.mkdir(exist_ok=True)
    image.save(FIGURE_DIR / "empirical_ai_scale.png", dpi=(220, 220))


def draw_automation_figure() -> None:
    horizon_rows = read_csv("metr_time_horizon_1_1.csv")
    delegation_rows = read_csv("anthropic_directive_share.csv")
    assert len(horizon_rows) >= 20
    assert [float(row["directive_share_percent"]) for row in delegation_rows] == [27.0, 39.0, 32.0]

    width, height = 2400, 1120
    image = Image.new("RGB", (width, height), COLORS["white"])
    draw = ImageDraw.Draw(image)
    draw_header(
        draw,
        "Capability and realized delegation",
        "Capability evaluations and platform use are informative proxies, not elasticity estimates",
        width,
    )

    left_box = (80, 205, 1510, 1050)
    right_box = (1600, 205, 2320, 1050)
    draw_panel_title(
        draw,
        left_box[0],
        left_box[1],
        "A. METR 50% task horizon",
        "Human-expert task duration; natural-log minutes and 95% intervals",
    )

    plot_left, plot_right = left_box[0] + 140, left_box[2] - 55
    plot_top, plot_bottom = left_box[1] + 145, left_box[3] - 125
    date_min, date_max = date(2019, 1, 1).toordinal(), date(2026, 6, 30).toordinal()
    y_min, y_max = math.log(0.005), math.log(4000.0)
    axis_font = load_font(21)
    small_font = load_font(19)
    label_font = load_font(21)

    def time_x(value: date) -> float:
        ordinal = value.toordinal()
        return plot_left + (ordinal - date_min) / (date_max - date_min) * (plot_right - plot_left)

    def log_y(minutes: float) -> float:
        return plot_bottom - (math.log(minutes) - y_min) / (y_max - y_min) * (plot_bottom - plot_top)

    reliability_y = log_y(960.0)
    draw.rectangle(
        (plot_left, plot_top, plot_right, reliability_y),
        fill=COLORS["light_orange"],
    )

    tick_values = [0.01, 0.1, 1, 10, 60, 600, 960, 3600]
    tick_labels = ["0.01m", "0.1m", "1m", "10m", "1h", "10h", "16h", "60h"]
    for value, label in zip(tick_values, tick_labels):
        y = log_y(value)
        draw.line((plot_left, y, plot_right, y), fill=COLORS["grid"], width=2)
        draw.text((plot_left - 16 - text_width(draw, label, axis_font), y - 12), label, fill=COLORS["muted"], font=axis_font)
    for year in range(2019, 2027):
        x = time_x(date(year, 1, 1))
        draw.line((x, plot_bottom, x, plot_bottom + 8), fill=COLORS["ink"], width=2)
        label = str(year)
        draw.text((x - text_width(draw, label, axis_font) / 2, plot_bottom + 16), label, fill=COLORS["muted"], font=axis_font)
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=COLORS["ink"], width=3)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=COLORS["ink"], width=3)
    dashed_line(draw, (plot_left, reliability_y, plot_right, reliability_y), COLORS["orange"], width=3)
    reliability_label = "METR: estimates above 16h are unreliable"
    draw.text((plot_left + 18, plot_top + 10), reliability_label, fill=COLORS["ink"], font=small_font)

    label_offsets = {
        "GPT-4": (-52, -38),
        "Claude Opus 4.6": (-174, 22),
    }
    for row in horizon_rows:
        release = date.fromisoformat(row["release_date"])
        estimate = float(row["p50_minutes"])
        lower = float(row["ci_low_minutes"])
        upper = float(row["ci_high_minutes"])
        is_sota = row["is_sota"].lower() == "true"
        x = time_x(release)
        y = log_y(estimate)
        draw.line((x, log_y(lower), x, log_y(upper)), fill=COLORS["blue"] if is_sota else COLORS["muted"], width=3)
        radius = 9 if is_sota else 8
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=COLORS["blue"] if is_sota else COLORS["white"],
            outline=COLORS["blue"] if is_sota else COLORS["muted"],
            width=3,
        )
        if row["agent"] in label_offsets:
            dx, dy = label_offsets[row["agent"]]
            draw.text((x + dx, y + dy), row["agent"], fill=COLORS["ink"], font=small_font)

    legend_y = left_box[3] - 46
    draw.ellipse((left_box[0] + 395, legend_y - 8, left_box[0] + 411, legend_y + 8), fill=COLORS["blue"], outline=COLORS["blue"])
    draw.text((left_box[0] + 424, legend_y - 13), "frontier at release", fill=COLORS["ink"], font=small_font)
    draw.ellipse((left_box[0] + 640, legend_y - 8, left_box[0] + 656, legend_y + 8), fill=COLORS["white"], outline=COLORS["muted"], width=3)
    draw.text((left_box[0] + 669, legend_y - 13), "other evaluated model", fill=COLORS["ink"], font=small_font)

    draw_panel_title(
        draw,
        right_box[0],
        right_box[1],
        "B. Directive use on Claude.ai",
        "Share of sampled conversations; directive is a subset of automation",
    )
    d_plot_left, d_plot_right = right_box[0] + 105, right_box[2] - 45
    d_plot_top, d_plot_bottom = right_box[1] + 145, right_box[3] - 130
    d_y_max = 45.0

    def delegation_y(value: float) -> float:
        return d_plot_bottom - value / d_y_max * (d_plot_bottom - d_plot_top)

    for tick in (0, 10, 20, 30, 40):
        y = delegation_y(float(tick))
        draw.line((d_plot_left, y, d_plot_right, y), fill=COLORS["grid"], width=2)
        label = f"{tick}%"
        draw.text((d_plot_left - 14 - text_width(draw, label, axis_font), y - 12), label, fill=COLORS["muted"], font=axis_font)
    draw.line((d_plot_left, d_plot_top, d_plot_left, d_plot_bottom), fill=COLORS["ink"], width=3)
    draw.line((d_plot_left, d_plot_bottom, d_plot_right, d_plot_bottom), fill=COLORS["ink"], width=3)

    centers = [d_plot_left + 105, (d_plot_left + d_plot_right) / 2, d_plot_right - 105]
    period_labels = {"2025-01": "Jan. 2025", "2025-08": "Aug. 2025", "2025-11": "Nov. 2025"}
    bar_width = 105
    for row, center in zip(delegation_rows, centers):
        value = float(row["directive_share_percent"])
        top = delegation_y(value)
        draw.rectangle((center - bar_width / 2, top, center + bar_width / 2, d_plot_bottom), fill=COLORS["olive"], outline=COLORS["olive"])
        value_label = f"{value:.0f}%"
        draw.text((center - text_width(draw, value_label, label_font) / 2, top - 38), value_label, fill=COLORS["ink"], font=label_font)
        period_label = period_labels[row["period"]]
        draw.text((center - text_width(draw, period_label, small_font) / 2, d_plot_bottom + 18), period_label, fill=COLORS["muted"], font=small_font)

    FIGURE_DIR.mkdir(exist_ok=True)
    image.save(FIGURE_DIR / "empirical_automation_proxies.png", dpi=(220, 220))


def main() -> None:
    draw_scale_figure()
    draw_automation_figure()
    print("Wrote figures/empirical_ai_scale.png")
    print("Wrote figures/empirical_automation_proxies.png")


if __name__ == "__main__":
    main()
