"""Plot the validated sigma_XL > 1 finite-boundary approximation.

The economic panels use the refined z_T = 128 collocation path, where z_T is
terminal Y/K.  The free-boundary diagnostic combines the saved z_T = 16, 32,
64, and 128 continuation summaries.  This script does not solve or alter the
model.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
TMP_DEPS = ROOT / "tmp" / "pydeps"
if TMP_DEPS.exists():
    sys.path.insert(0, str(TMP_DEPS))
LOCAL_DEPS = ROOT / ".python-packages"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

import simulate_model as mechanism  # noqa: E402


RESULT_DIR = ROOT / "numerical_axm"
FIGURE_DIR = ROOT / "figures_axm"
PATH_FILE = (
    RESULT_DIR
    / "high_sigma_sigma150_z128_validated_boundary_paths.csv"
)
CONTINUATION_FILES = (
    RESULT_DIR / "high_sigma_sigma150_validated_free_continuation.csv",
    RESULT_DIR
    / "high_sigma_sigma150_z128_validated_free_continuation.csv",
)
PRIMARY_PATH_FILE = (
    RESULT_DIR / "high_sigma_sigma150_validated_boundary_paths.csv"
)
AUDIT_MANIFEST_FILE = (
    RESULT_DIR / "high_sigma_sigma150_validated_audit_manifest.json"
)
AUDIT_SCRIPT_FILE = ROOT / "scripts" / "audit_axm_high_sigma.py"
CANONICAL_INPUT_FILES = (
    PRIMARY_PATH_FILE,
    CONTINUATION_FILES[0],
    PATH_FILE,
    CONTINUATION_FILES[1],
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No observations found in {path}.")
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_current_pass_manifest() -> None:
    """Refuse to label figures validated unless the exact inputs passed audit."""

    if not AUDIT_MANIFEST_FILE.exists():
        raise FileNotFoundError(
            "The high-sigma PASS manifest is missing. Run "
            "python scripts/audit_axm_high_sigma.py before plotting."
        )
    with AUDIT_MANIFEST_FILE.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("overall_accepted") is not True:
        raise ValueError(
            "The high-sigma audit manifest does not record a PASS."
        )

    declared_inputs: dict[Path, dict[str, object]] = {}
    for entry in manifest.get("inputs", []):
        declared = (ROOT / Path(str(entry["path"]))).resolve()
        if declared in declared_inputs:
            raise ValueError(f"Duplicate input in audit manifest: {declared}")
        declared_inputs[declared] = entry
    expected_inputs = {path.resolve() for path in CANONICAL_INPUT_FILES}
    if set(declared_inputs) != expected_inputs:
        raise ValueError(
            "The PASS manifest does not cover exactly the four canonical "
            "high-sigma solver inputs. Re-run the current audit."
        )
    for path in CANONICAL_INPUT_FILES:
        resolved = path.resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        entry = declared_inputs[resolved]
        if int(entry.get("bytes", -1)) != path.stat().st_size:
            raise ValueError(f"Audited input size has changed: {path}")
        if str(entry.get("sha256", "")) != sha256_file(path):
            raise ValueError(f"Audited input hash has changed: {path}")

    audit_entry = manifest.get("audit_script", {})
    declared_audit_path = (
        ROOT / Path(str(audit_entry.get("path", "")))
    ).resolve()
    if declared_audit_path != AUDIT_SCRIPT_FILE.resolve():
        raise ValueError("The PASS manifest names a different audit script.")
    if str(audit_entry.get("sha256", "")) != sha256_file(AUDIT_SCRIPT_FILE):
        raise ValueError(
            "The audit script changed after the PASS manifest was written. "
            "Re-run python scripts/audit_axm_high_sigma.py."
        )


def add_vertical_padding(
    path: Path, *, top: int = 140, bottom: int = 140
) -> None:
    """Add print-safe whitespace without changing a chart's plotted geometry."""

    with Image.open(path) as source:
        source_rgb = source.convert("RGB")
        padded = Image.new(
            "RGB",
            (source_rgb.width, source_rgb.height + top + bottom),
            "white",
        )
        padded.paste(source_rgb, (0, top))
    padded.save(path, dpi=(220, 220))


def validate_path(rows: list[dict[str, str]]) -> None:
    required = {
        "time",
        "output_capital_ratio",
        "capability_growth",
        "output_per_capita_growth",
        "wage_growth",
        "net_capital_return",
        "ai_share",
        "automated_research_share",
        "aggregate_labor_share",
        "human_research_share",
        "capability_growth_to_output_capital",
        "inference_share",
        "research_resource_share",
        "investment_share",
        "terminal_boundary_z",
    }
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Path file is missing columns: {sorted(missing)}")

    times = np.asarray([float(row["time"]) for row in rows])
    if not np.all(np.isfinite(times)) or not np.all(np.diff(times) > 0.0):
        raise ValueError("Path times must be finite and strictly increasing.")

    numeric_fields = required.difference({"time"})
    for field in numeric_fields:
        values = np.asarray([float(row[field]) for row in rows])
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Non-finite values found in {field}.")

    terminal_boundaries = {
        round(float(row["terminal_boundary_z"]), 8) for row in rows
    }
    if terminal_boundaries != {128.0}:
        raise ValueError(
            f"Expected only z_T=128; found {sorted(terminal_boundaries)}."
        )
    terminal_ratio = float(rows[-1]["output_capital_ratio"])
    if not math.isclose(terminal_ratio, 128.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(f"Terminal Y/K is {terminal_ratio}, not 128.")

    share_fields = (
        "ai_share",
        "automated_research_share",
        "aggregate_labor_share",
        "human_research_share",
        "inference_share",
        "research_resource_share",
        "investment_share",
    )
    for field in share_fields:
        values = [float(row[field]) for row in rows]
        if min(values) < -1e-10 or max(values) > 1.0 + 1e-10:
            raise ValueError(f"{field} leaves the unit interval.")


def load_continuation_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in CONTINUATION_FILES:
        rows.extend(read_rows(path))
    rows.sort(key=lambda row: float(row["terminal_output_capital_ratio"]))
    boundaries = [float(row["terminal_output_capital_ratio"]) for row in rows]
    if boundaries != [16.0, 32.0, 64.0, 128.0]:
        raise ValueError(
            "Expected continuation boundaries 16, 32, 64, 128; "
            f"found {boundaries}."
        )
    for row in rows:
        values = (
            float(row["duration"]),
            float(row["estimated_singularity_time"]),
            float(row["max_rms_residual"]),
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Continuation summary contains a non-finite value.")
        if float(row["max_rms_residual"]) > 1.1e-6:
            raise ValueError("A continuation residual exceeds the validated gate.")
    return rows


def derive_metadata_and_targets(
    path_rows: list[dict[str, str]],
    continuation_rows: list[dict[str, str]],
) -> dict[str, float]:
    """Recover plotted metadata and conditional limits from canonical files.

    The dated identities imply ``alpha = (gross return)/(Y/K)`` and the
    research-compute FOC implies

        eta = (M/Y) / [s_M (qA/Y) g_A].

    For the research CES, the slope of ``logit(s_M)`` with respect to
    ``log(Aw)`` is ``sigma_HM - 1``.  These identities let the plotter recover
    the parameters it needs from the audited path instead of depending on a
    duplicate solver summary.  The continuation file supplies ``sigma_XL`` and
    the terminal-boundary sequence.
    """

    sigma_xl_values = {
        round(float(row["sigma_xl"]), 12) for row in continuation_rows
    }
    if len(sigma_xl_values) != 1:
        raise ValueError(
            f"Continuation files disagree on sigma_XL: {sigma_xl_values}."
        )
    sigma_xl = sigma_xl_values.pop()
    if sigma_xl <= 1.0:
        raise ValueError("The high-sigma plot requires sigma_XL > 1.")

    continuation_boundary = float(
        continuation_rows[-1]["terminal_output_capital_ratio"]
    )
    path_boundary = float(path_rows[-1]["terminal_boundary_z"])
    if not math.isclose(
        continuation_boundary, path_boundary, rel_tol=0.0, abs_tol=1e-10
    ):
        raise ValueError(
            "Canonical path and continuation file have different terminal "
            f"boundaries: {path_boundary} and {continuation_boundary}."
        )

    alpha_values = np.asarray(
        [
            float(row["gross_capital_return"])
            / float(row["output_capital_ratio"])
            for row in path_rows
        ]
    )
    alpha = float(np.median(alpha_values))
    if np.max(np.abs(alpha_values - alpha)) > 1e-10:
        raise ValueError("The canonical path does not imply a constant alpha.")

    eta_values = np.asarray(
        [
            float(row["research_resource_share"])
            / (
                float(row["automated_research_share"])
                * float(row["shadow_capability_to_output"])
                * float(row["capability_growth"])
            )
            for row in path_rows
        ]
    )
    eta = float(np.median(eta_values))
    if np.max(np.abs(eta_values - eta)) > 1e-8:
        raise ValueError("The canonical path does not imply a constant eta.")

    research_share_pairs = []
    for row in path_rows:
        automated_share = float(row["automated_research_share"])
        if 1e-8 < automated_share < 1.0 - 1e-6:
            log_aw = float(row["log_capability"]) + float(row["log_wage"])
            log_odds = math.log(automated_share / (1.0 - automated_share))
            research_share_pairs.append((log_aw, log_odds))
    if len(research_share_pairs) < 12:
        raise ValueError("Too few interior research shares to recover sigma_HM.")
    log_aw = np.asarray([pair[0] for pair in research_share_pairs])
    log_odds = np.asarray([pair[1] for pair in research_share_pairs])
    slope, intercept = np.polyfit(log_aw, log_odds, 1)
    share_fit_error = np.max(
        np.abs(log_odds - (slope * log_aw + intercept))
    )
    if share_fit_error > 1e-8:
        raise ValueError("Research-share odds do not identify one sigma_HM.")
    sigma_hm = 1.0 + float(slope)

    if not 0.0 < alpha < 1.0 or not 0.0 < eta < 1.0:
        raise ValueError(f"Invalid recovered alpha={alpha} or eta={eta}.")
    if sigma_hm <= 1.0:
        raise ValueError("The plotted high-sigma branch requires sigma_HM > 1.")

    kappa = (1.0 - alpha) / alpha
    denominator = 1.0 + kappa - eta
    inference_share = (1.0 - alpha) ** 2
    capability_growth_to_z = eta * alpha / denominator
    investment_share = alpha - kappa * capability_growth_to_z
    research_share = eta * inference_share / denominator
    if min(investment_share, research_share) <= 0.0:
        raise ValueError("Recovered conditional resource shares are not positive.")

    return {
        "sigma_xl": sigma_xl,
        "sigma_hm": sigma_hm,
        "alpha": alpha,
        "eta": eta,
        "terminal_boundary_z": path_boundary,
        "inference_share": inference_share,
        "capability_growth_to_z": capability_growth_to_z,
        "investment_share": investment_share,
        "research_share": research_share,
        "singularity_rate": kappa * capability_growth_to_z,
    }


def verify_singularity_estimates(
    continuation_rows: list[dict[str, str]], metadata: dict[str, float]
) -> None:
    """Verify, then replace, saved T* values with their dated reconstruction."""

    rate = metadata["singularity_rate"]
    for row in continuation_rows:
        boundary = float(row["terminal_output_capital_ratio"])
        recomputed = float(row["duration"]) + 1.0 / (rate * boundary)
        saved = float(row["estimated_singularity_time"])
        if not math.isclose(saved, recomputed, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                f"Saved T* at z_T={boundary:g} is {saved}, but the dated "
                f"reconstruction is {recomputed}. Re-run the audit."
            )
        row["estimated_singularity_time"] = f"{recomputed:.17g}"


def _panel_y_values(
    rows: list[dict[str, str]], panel: dict
) -> np.ndarray:
    values = np.asarray([float(row[panel["field"]]) for row in rows])
    transform = panel.get("transform")
    if transform is not None:
        values = transform(rows, values)
    return values


def draw_multiplot_by_x(
    output_path: Path,
    title: str,
    subtitle: str,
    panels: list[dict],
    rows: list[dict[str, str]],
    *,
    x_value: Callable[[dict[str, str]], float],
    x_label: str,
    x_tick_values: list[float],
    x_tick_labels: list[str],
    series_label: str,
    canvas_height: int = 1600,
    panel_y_offset: int = 0,
) -> None:
    """Draw a four-panel figure against a non-calendar horizontal axis."""

    width, height = 2400, canvas_height
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = mechanism.load_font(48, bold=True)
    subtitle_font = mechanism.load_font(28)
    panel_title_font = mechanism.load_font(30, bold=True)
    axis_font = mechanism.load_font(23)
    legend_font = mechanism.load_font(24)
    reference_font = mechanism.load_font(20)

    draw.text((120, 65), title, fill=mechanism.COLORS["ink"], font=title_font)
    draw.text(
        (120, 130),
        subtitle,
        fill=mechanism.COLORS["muted"],
        font=subtitle_font,
    )
    legend_x, legend_y = 130, 190
    draw.line(
        (legend_x, legend_y + 13, legend_x + 48, legend_y + 13),
        fill=mechanism.COLORS["blue"],
        width=6,
    )
    mechanism.draw_marker(
        draw,
        legend_x + 24,
        legend_y + 13,
        mechanism.COLORS["blue"],
        "circle",
        radius=7,
    )
    draw.text(
        (legend_x + 62, legend_y),
        series_label,
        fill=mechanism.COLORS["ink"],
        font=legend_font,
    )

    x_values = np.asarray([x_value(row) for row in rows])
    if not np.all(np.isfinite(x_values)) or np.any(np.diff(x_values) <= 0.0):
        raise ValueError("The appendix x-axis must be finite and increasing.")
    x_min = min(float(x_values[0]), min(x_tick_values))
    x_max = max(float(x_values[-1]), max(x_tick_values))

    panel_boxes = [
        (120, 245 + panel_y_offset, 1150, 850 + panel_y_offset),
        (1270, 245 + panel_y_offset, 2300, 850 + panel_y_offset),
        (120, 930 + panel_y_offset, 1150, 1535 + panel_y_offset),
        (1270, 930 + panel_y_offset, 2300, 1535 + panel_y_offset),
    ]
    for panel, box in zip(panels, panel_boxes):
        left, top, right, bottom = box
        plot_left, plot_top = left + 120, top + 75
        plot_right, plot_bottom = right - 35, bottom - 85
        draw.text(
            (left, top),
            panel["title"],
            fill=mechanism.COLORS["ink"],
            font=panel_title_font,
        )

        y_values = _panel_y_values(rows, panel)
        if not np.all(np.isfinite(y_values)):
            raise ValueError(f"Non-finite plotted values in {panel['field']}.")
        if panel.get("ylim") is not None:
            y_min, y_max = map(float, panel["ylim"])
        else:
            y_min, y_max = float(np.min(y_values)), float(np.max(y_values))
            reference = panel.get("reference_y")
            if reference is not None:
                y_min = min(y_min, float(reference))
                y_max = max(y_max, float(reference))
            padding = 0.08 * max(y_max - y_min, 1e-8)
            y_min, y_max = y_min - padding, y_max + padding

        for tick in mechanism.nice_ticks(y_min, y_max, 5):
            y_pixel = plot_bottom - (tick - y_min) / (y_max - y_min) * (
                plot_bottom - plot_top
            )
            draw.line(
                (plot_left, y_pixel, plot_right, y_pixel),
                fill=mechanism.COLORS["grid"],
                width=2,
            )
            label = panel.get("format", lambda value: f"{value:.2f}")(tick)
            bbox = draw.textbbox((0, 0), label, font=axis_font)
            draw.text(
                (plot_left - 15 - (bbox[2] - bbox[0]), y_pixel - 12),
                label,
                fill=mechanism.COLORS["muted"],
                font=axis_font,
            )

        for tick, label in zip(x_tick_values, x_tick_labels):
            if tick < x_min - 1e-12 or tick > x_max + 1e-12:
                continue
            x_pixel = plot_left + (tick - x_min) / (x_max - x_min) * (
                plot_right - plot_left
            )
            draw.line(
                (x_pixel, plot_bottom, x_pixel, plot_bottom + 8),
                fill=mechanism.COLORS["ink"],
                width=2,
            )
            bbox = draw.textbbox((0, 0), label, font=axis_font)
            draw.text(
                (x_pixel - (bbox[2] - bbox[0]) / 2, plot_bottom + 14),
                label,
                fill=mechanism.COLORS["muted"],
                font=axis_font,
            )

        draw.line(
            (plot_left, plot_top, plot_left, plot_bottom),
            fill=mechanism.COLORS["ink"],
            width=3,
        )
        draw.line(
            (plot_left, plot_bottom, plot_right, plot_bottom),
            fill=mechanism.COLORS["ink"],
            width=3,
        )

        reference = panel.get("reference_y")
        if reference is not None and y_min <= reference <= y_max:
            reference_pixel = plot_bottom - (
                (float(reference) - y_min) / (y_max - y_min)
            ) * (plot_bottom - plot_top)
            start = plot_left
            while start < plot_right:
                draw.line(
                    (
                        start,
                        reference_pixel,
                        min(start + 18, plot_right),
                        reference_pixel,
                    ),
                    fill=mechanism.COLORS["ink"],
                    width=3,
                )
                start += 30
            reference_label = panel.get("reference_label")
            if reference_label:
                bbox = draw.textbbox((0, 0), reference_label, font=reference_font)
                draw.rectangle(
                    (
                        plot_right - (bbox[2] - bbox[0]) - 12,
                        reference_pixel - 27,
                        plot_right + 2,
                        reference_pixel - 2,
                    ),
                    fill="white",
                )
                draw.text(
                    (
                        plot_right - (bbox[2] - bbox[0]) - 6,
                        reference_pixel - 27,
                    ),
                    reference_label,
                    fill=mechanism.COLORS["ink"],
                    font=reference_font,
                )

        points: list[tuple[float, float]] = []
        for x_coordinate, y_value in zip(x_values, y_values):
            x_pixel = plot_left + (x_coordinate - x_min) / (x_max - x_min) * (
                plot_right - plot_left
            )
            y_pixel = plot_bottom - (y_value - y_min) / (y_max - y_min) * (
                plot_bottom - plot_top
            )
            points.append((x_pixel, y_pixel))
        draw.line(
            points,
            fill=mechanism.COLORS["blue"],
            width=6,
            joint="curve",
        )
        marker_step = max(1, len(points) // 9)
        for x_pixel, y_pixel in points[::marker_step]:
            mechanism.draw_marker(
                draw,
                x_pixel,
                y_pixel,
                mechanism.COLORS["blue"],
                "circle",
                radius=7,
            )

        bbox = draw.textbbox((0, 0), x_label, font=axis_font)
        draw.text(
            (
                (plot_left + plot_right - (bbox[2] - bbox[0])) / 2,
                plot_bottom + 50,
            ),
            x_label,
            fill=mechanism.COLORS["muted"],
            font=axis_font,
        )

    image.save(output_path, dpi=(220, 220))


def draw_boundary_convergence(
    output_path: Path, rows: list[dict[str, str]]
) -> None:
    """Draw a focused T*(z_T) plot alongside its exact source values."""

    width, height = 2400, 1150
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = mechanism.load_font(48, bold=True)
    subtitle_font = mechanism.load_font(28)
    panel_title_font = mechanism.load_font(30, bold=True)
    axis_font = mechanism.load_font(23)
    table_font = mechanism.load_font(22)
    table_header_font = mechanism.load_font(22, bold=True)
    note_font = mechanism.load_font(21)

    draw.text(
        (120, 65),
        "Estimated T*(z_T) across terminal Y/K boundaries",
        fill=mechanism.COLORS["ink"],
        font=title_font,
    )
    draw.text(
        (120, 130),
        (
            "Finite-boundary approximations to a conditional pre-singular branch "
            "satisfying dated necessary conditions; z_T is terminal Y/K"
        ),
        fill=mechanism.COLORS["muted"],
        font=subtitle_font,
    )

    plot_left, plot_top, plot_right, plot_bottom = 235, 280, 1320, 900
    draw.text(
        (120, 215),
        "Estimated singularity date",
        fill=mechanism.COLORS["ink"],
        font=panel_title_font,
    )
    boundaries = np.asarray(
        [float(row["terminal_output_capital_ratio"]) for row in rows]
    )
    x_values = np.log(boundaries)
    estimates = np.asarray(
        [float(row["estimated_singularity_time"]) for row in rows]
    )
    x_min, x_max = float(x_values[0]), float(x_values[-1])
    y_padding = 0.12 * max(float(np.ptp(estimates)), 1e-5)
    y_min = float(np.min(estimates)) - y_padding
    y_max = float(np.max(estimates)) + y_padding

    for tick in mechanism.nice_ticks(y_min, y_max, 5):
        y_pixel = plot_bottom - (tick - y_min) / (y_max - y_min) * (
            plot_bottom - plot_top
        )
        draw.line(
            (plot_left, y_pixel, plot_right, y_pixel),
            fill=mechanism.COLORS["grid"],
            width=2,
        )
        label = f"{tick:.3f}"
        bbox = draw.textbbox((0, 0), label, font=axis_font)
        draw.text(
            (plot_left - 15 - (bbox[2] - bbox[0]), y_pixel - 12),
            label,
            fill=mechanism.COLORS["muted"],
            font=axis_font,
        )
    draw.line(
        (plot_left, plot_top, plot_left, plot_bottom),
        fill=mechanism.COLORS["ink"],
        width=3,
    )
    draw.line(
        (plot_left, plot_bottom, plot_right, plot_bottom),
        fill=mechanism.COLORS["ink"],
        width=3,
    )

    points: list[tuple[float, float]] = []
    for boundary, x_value, estimate in zip(boundaries, x_values, estimates):
        x_pixel = plot_left + (x_value - x_min) / (x_max - x_min) * (
            plot_right - plot_left
        )
        y_pixel = plot_bottom - (estimate - y_min) / (y_max - y_min) * (
            plot_bottom - plot_top
        )
        points.append((x_pixel, y_pixel))
        label = f"{boundary:.0f}"
        bbox = draw.textbbox((0, 0), label, font=axis_font)
        draw.text(
            (x_pixel - (bbox[2] - bbox[0]) / 2, plot_bottom + 14),
            label,
            fill=mechanism.COLORS["muted"],
            font=axis_font,
        )
    draw.line(points, fill=mechanism.COLORS["blue"], width=6, joint="curve")
    for (x_pixel, y_pixel), estimate in zip(points, estimates):
        draw.ellipse(
            (x_pixel - 9, y_pixel - 9, x_pixel + 9, y_pixel + 9),
            fill="white",
            outline=mechanism.COLORS["blue"],
            width=5,
        )
        label = f"{estimate:.4f}"
        bbox = draw.textbbox((0, 0), label, font=axis_font)
        draw.text(
            (x_pixel - (bbox[2] - bbox[0]) / 2, y_pixel - 42),
            label,
            fill=mechanism.COLORS["ink"],
            font=axis_font,
        )
    x_label = "z_T: terminal Y/K (natural-log spacing)"
    bbox = draw.textbbox((0, 0), x_label, font=axis_font)
    draw.text(
        (
            (plot_left + plot_right - (bbox[2] - bbox[0])) / 2,
            plot_bottom + 58,
        ),
        x_label,
        fill=mechanism.COLORS["muted"],
        font=axis_font,
    )

    table_left, table_top, table_right = 1450, 245, 2290
    row_height = 88
    columns = [
        ("z_T", 1450, 1570),
        ("T(z_T)", 1580, 1805),
        ("T*(z_T)", 1815, 2040),
        ("Max RMS", 2050, 2290),
    ]
    for label, left, right in columns:
        bbox = draw.textbbox((0, 0), label, font=table_header_font)
        draw.text(
            ((left + right - (bbox[2] - bbox[0])) / 2, table_top),
            label,
            fill=mechanism.COLORS["ink"],
            font=table_header_font,
        )
    draw.line(
        (table_left, table_top + 46, table_right, table_top + 46),
        fill=mechanism.COLORS["ink"],
        width=3,
    )
    for row_index, row in enumerate(rows):
        y = table_top + 72 + row_index * row_height
        values = (
            f"{float(row['terminal_output_capital_ratio']):.0f}",
            f"{float(row['duration']):.4f}",
            f"{float(row['estimated_singularity_time']):.4f}",
            f"{float(row['max_rms_residual']):.2e}",
        )
        if row_index % 2 == 0:
            draw.rectangle(
                (table_left, y - 13, table_right, y + 48),
                fill=mechanism.COLORS["light"],
            )
        for value, (_, left, right) in zip(values, columns):
            bbox = draw.textbbox((0, 0), value, font=table_font)
            draw.text(
                ((left + right - (bbox[2] - bbox[0])) / 2, y),
                value,
                fill=mechanism.COLORS["ink"],
                font=table_font,
            )
    note = (
        "z_T denotes terminal Y/K; T(z_T) is the date that boundary is reached.\n"
        "T*(z_T): singularity date estimated from the terminal asymptotic tail."
    )
    draw.multiline_text(
        (1450, 690),
        note,
        fill=mechanism.COLORS["muted"],
        font=note_font,
        spacing=8,
    )
    image.save(output_path, dpi=(220, 220))


def draw_transition_allocation_figures(
    rows: list[dict[str, str]], metadata: dict[str, float]
) -> None:
    """Separate production allocation from the AI-research allocation."""

    percent = lambda _rows, values: 100.0 * values
    normalized_time_ticks = [0.0, 200.0, 400.0, 600.0, 800.0]
    normalized_time_labels = [f"{value:.0f}" for value in normalized_time_ticks]
    subtitle = (
        "Percent on linear scales; final-production elasticity "
        f"{metadata['sigma_xl']:.2f}, research elasticity "
        f"{metadata['sigma_hm']:.2f}; displayed through Y/K ≤ 5"
    )
    series_label = "Audited finite-boundary path"

    draw_multiplot_by_x(
        FIGURE_DIR / "high_sigma_validated_production_allocation.png",
        "Gross substitution: production and income allocation",
        subtitle,
        [
            {
                "title": "AI share of the labor and AI composite",
                "field": "ai_share",
                "transform": percent,
                "format": lambda value: f"{value:.0f}%",
                "ylim": (0.0, 100.0),
            },
            {
                "title": "Inference resources as a share of output",
                "field": "inference_share",
                "transform": percent,
                "format": lambda value: f"{value:.0f}%",
                "ylim": (0.0, 50.0),
            },
            {
                "title": "Gross investment as a share of output",
                "field": "investment_share",
                "transform": percent,
                "format": lambda value: f"{value:.0f}%",
                "ylim": (0.0, 35.0),
            },
            {
                "title": "Labor income as a share of output",
                "field": "aggregate_labor_share",
                "transform": percent,
                "format": lambda value: f"{value:.0f}%",
                "ylim": (0.0, 70.0),
            },
        ],
        rows,
        x_value=lambda row: float(row["time"]),
        x_label="Model time (years)",
        x_tick_values=normalized_time_ticks,
        x_tick_labels=normalized_time_labels,
        series_label=series_label,
    )
    add_vertical_padding(
        FIGURE_DIR / "high_sigma_validated_production_allocation.png"
    )

    draw_multiplot_by_x(
        FIGURE_DIR / "high_sigma_validated_research_allocation.png",
        "Gross substitution: AI research allocation",
        subtitle,
        [
            {
                "title": "Automated share of research expenditure",
                "field": "automated_research_share",
                "transform": percent,
                "format": lambda value: f"{value:.0f}%",
                "ylim": (0.0, 100.0),
            },
            {
                "title": "Human researchers as a share of population",
                "field": "human_research_share",
                "transform": percent,
                "format": lambda value: f"{value:.2f}%",
                "ylim": (0.0, 0.75),
            },
            {
                "title": "Research compute as a share of output",
                "field": "research_resource_share",
                "transform": percent,
                "format": lambda value: f"{value:.0f}%",
                "ylim": (0.0, 10.0),
            },
            {
                "title": "Capability growth relative to output per unit of capital",
                "field": "capability_growth_to_output_capital",
                "transform": percent,
                "format": lambda value: f"{value:.0f}%",
                "ylim": (0.0, 9.0),
            },
        ],
        rows,
        x_value=lambda row: float(row["time"]),
        x_label="Model time (years)",
        x_tick_values=normalized_time_ticks,
        x_tick_labels=normalized_time_labels,
        series_label=series_label,
        canvas_height=1700,
        panel_y_offset=100,
    )
    add_vertical_padding(
        FIGURE_DIR / "high_sigma_validated_research_allocation.png",
        top=200,
    )


def main() -> None:
    FIGURE_DIR.mkdir(exist_ok=True)
    require_current_pass_manifest()
    rows = read_rows(PATH_FILE)
    validate_path(rows)
    continuation_rows = load_continuation_rows()
    metadata = derive_metadata_and_targets(rows, continuation_rows)
    verify_singularity_estimates(continuation_rows, metadata)

    transition_rows = [
        row for row in rows if float(row["output_capital_ratio"]) <= 5.0
    ]
    if len(transition_rows) < 12:
        raise ValueError("Too few observations through Y/K <= 5.")

    series = {"validated_sigma_1_50_z_128": transition_rows}
    path_label = (
        "Finite-boundary approximation satisfying dated necessary conditions "
        "within reported numerical tolerances"
    )
    parameter_label = (
        f"final-production elasticity = {metadata['sigma_xl']:.2f}, "
        f"research elasticity = {metadata['sigma_hm']:.2f}; "
        f"terminal Y/K = {metadata['terminal_boundary_z']:.0f}"
    )
    labels = {"validated_sigma_1_50_z_128": path_label}
    palette = {
        "validated_sigma_1_50_z_128": mechanism.COLORS["blue"]
    }
    markers = {"validated_sigma_1_50_z_128": "circle"}
    percent = lambda _rows, values: 100.0 * values

    mechanism.draw_multiplot(
        FIGURE_DIR / "high_sigma_validated_transition_rates.png",
        "Gross substitution: growth and returns",
        (
            f"Annual percent; {parameter_label}; displayed through Y/K <= 5"
        ),
        [
            {
                "title": "Capability growth",
                "field": "capability_growth",
                "transform": percent,
                "format": lambda value: f"{value:.0f}%",
                "reference_y": 0.0,
            },
            {
                "title": "Output-per-capita growth",
                "field": "output_per_capita_growth",
                "transform": percent,
                "format": lambda value: f"{value:.0f}%",
                "reference_y": 0.0,
            },
            {
                "title": "Real-wage growth",
                "field": "wage_growth",
                "transform": percent,
                "format": lambda value: f"{value:.0f}%",
                "reference_y": 0.0,
            },
            {
                "title": "Net return to capital",
                "field": "net_capital_return",
                "transform": percent,
                "format": lambda value: f"{value:.0f}%",
                "reference_y": 0.0,
            },
        ],
        series,
        labels,
        palette,
        markers,
    )

    mechanism.draw_multiplot(
        FIGURE_DIR / "high_sigma_validated_transition_shares.png",
        "Gross substitution: production and research shares",
        (
            f"Percent; {parameter_label}; s_X is the CES technological share within Z, "
            "not an output share; Y/K <= 5"
        ),
        [
            {
                "title": "AI-service CES share within Z, s_X",
                "field": "ai_share",
                "transform": percent,
                "format": lambda value: f"{value:.0f}%",
                "ylim": (0.0, 100.0),
            },
            {
                "title": "Automated-research expenditure share, s_M",
                "field": "automated_research_share",
                "transform": percent,
                "format": lambda value: f"{value:.0f}%",
                "ylim": (0.0, 100.0),
            },
            {
                "title": "Aggregate wage bill / output, wN/Y",
                "field": "aggregate_labor_share",
                "transform": percent,
                "format": lambda value: f"{value:.0f}%",
                "ylim": (0.0, 70.0),
            },
            {
                "title": "Human researchers / population, H/N",
                "field": "human_research_share",
                "transform": percent,
                "format": lambda value: f"{value:.2f}%",
                "ylim": (0.0, 0.75),
            },
        ],
        series,
        labels,
        palette,
        markers,
    )
    add_vertical_padding(
        FIGURE_DIR / "high_sigma_validated_transition_shares.png"
    )
    draw_transition_allocation_figures(transition_rows, metadata)

    rising_start = min(
        range(len(rows)), key=lambda index: float(rows[index]["output_capital_ratio"])
    )
    convergence_rows = [
        row
        for row in rows[rising_start:]
        if float(row["output_capital_ratio"]) >= 1.0
    ]
    if len(convergence_rows) < 12:
        raise ValueError("Too few observations on the rising portion of Y/K.")
    z_values = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]
    appendix_panels = [
        {
            "title": "Capability growth / (Y/K)",
            "field": "capability_growth_to_output_capital",
            "transform": percent,
            "format": lambda value: f"{value:.1f}%",
            "reference_y": 100.0 * metadata["capability_growth_to_z"],
            "reference_label": "conditional analytical limit",
        },
        {
            "title": "Inference resources / output, U/Y",
            "field": "inference_share",
            "transform": percent,
            "format": lambda value: f"{value:.1f}%",
            "reference_y": 100.0 * metadata["inference_share"],
            "reference_label": "conditional analytical limit",
        },
        {
            "title": "Research compute / output, M/Y",
            "field": "research_resource_share",
            "transform": percent,
            "format": lambda value: f"{value:.1f}%",
            "reference_y": 100.0 * metadata["research_share"],
            "reference_label": "conditional analytical limit",
        },
        {
            "title": "Gross investment / output, (Kdot + delta K)/Y",
            "field": "investment_share",
            "transform": percent,
            "format": lambda value: f"{value:.1f}%",
            "reference_y": 100.0 * metadata["investment_share"],
            "reference_label": "conditional analytical limit",
        },
    ]
    draw_multiplot_by_x(
        FIGURE_DIR / "high_sigma_validated_asymptotic_ratios.png",
        "Gross substitution: asymptotic ratios",
        (
            f"Rising portion of the approximation, Y/K >= 1; horizontal axis "
            f"ln(Y/K); dashed = "
            f"conditional analytical limits; z_T is terminal Y/K = "
            f"{metadata['terminal_boundary_z']:.0f}"
        ),
        appendix_panels,
        convergence_rows,
        x_value=lambda row: math.log(float(row["output_capital_ratio"])),
        x_label="ln(Y/K)",
        x_tick_values=[math.log(value) for value in z_values],
        x_tick_labels=[f"{math.log(value):.1f}" for value in z_values],
        series_label=path_label,
    )

    draw_boundary_convergence(
        FIGURE_DIR / "high_sigma_validated_boundary_convergence.png",
        continuation_rows,
    )

    print(f"Read {len(rows):,} path observations from {PATH_FILE.name}.")
    print(
        f"Transition display: {len(transition_rows):,} observations through "
        f"Y/K={float(transition_rows[-1]['output_capital_ratio']):.3f}."
    )
    print("Validated continuation boundaries: 16, 32, 64, 128.")
    print(
        "Recovered from canonical outputs: "
        f"alpha={metadata['alpha']:.6f}, eta={metadata['eta']:.6f}, "
        f"sigma_XL={metadata['sigma_xl']:.6f}, "
        f"sigma_HM={metadata['sigma_hm']:.6f}."
    )
    for name in (
        "high_sigma_validated_transition_rates.png",
        "high_sigma_validated_transition_shares.png",
        "high_sigma_validated_production_allocation.png",
        "high_sigma_validated_research_allocation.png",
        "high_sigma_validated_asymptotic_ratios.png",
        "high_sigma_validated_boundary_convergence.png",
    ):
        print(FIGURE_DIR / name)


if __name__ == "__main__":
    main()
