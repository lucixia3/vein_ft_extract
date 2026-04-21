"""Generate static PNG/JPG and interactive HTML polar venous charts from a CSV.

For each case, the script creates two static polar charts (mean density in absolute
HU and volume fraction relative to ICV) and one interactive HTML chart with a
metric selector. Outputs are saved in the corresponding case folder under the
chosen output directory.

Missing structures do not stop the batch: their sectors are drawn in gray.
"""

from __future__ import annotations

import argparse
import math
import platform
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.patches import Wedge
from matplotlib.ticker import FormatStrFormatter

# -----------------------------------------------------------------------------
# Fixed geometry / labels
# -----------------------------------------------------------------------------
# NOTE: `structure` is the structure name expected in the CSV.
# `display` is the text rendered in the charts.
LAYOUT = [
    # Center
    {"structure": "ICV", "display": "ICV", "r0": 0.0, "r1": 0.8, "theta1": 0, "theta2": 360},
    # Ring 1
    {"structure": "VOG", "display": "VOG", "r0": 0.8, "r1": 1.6, "theta1": 45, "theta2": 135},
    {"structure": "RBVR", "display": "RBVR", "r0": 0.8, "r1": 1.6, "theta1": 180, "theta2": 360},
    # Spacer ring: 1.6 -> 2.4 (empty)
    # Ring 3
    {"structure": "Cortical-L", "display": "Cort-L", "r0": 2.4, "r1": 3.4, "theta1": 90, "theta2": 270},
    {"structure": "Cortical-R", "display": "Cort-R", "r0": 2.4, "r1": 3.4, "theta1": 270, "theta2": 90},
    # Ring 4
    {"structure": "SSS", "display": "SSS", "r0": 3.4, "r1": 4.5, "theta1": 0, "theta2": 180},
    {"structure": "TransvSig-L", "display": "TransvSig-L", "r0": 3.4, "r1": 4.5, "theta1": 180, "theta2": 255},
    {"structure": "STS", "display": "STS", "r0": 3.4, "r1": 4.5, "theta1": 255, "theta2": 285},
    {"structure": "TransvSig-R", "display": "TransvSig-R", "r0": 3.4, "r1": 4.5, "theta1": 285, "theta2": 360},
]

STRUCTURE_ORDER = [item["structure"] for item in LAYOUT]
LAYOUT_BY_STRUCTURE = {item["structure"]: item for item in LAYOUT}
MAX_RADIUS = max(float(item["r1"]) for item in LAYOUT)
MISSING_FACE = "#bdbdbd"
EDGE_COLOR = "#1f4ea8"
DEFAULT_BG_COLOR = "white"
CMAP_NAME = "RdYlBu" 

# Robust scaling percentiles kept from the previous logic.
PERCENTILES_BY_METRIC = {
    "density": (5.0, 95.0),
    "volume": (0.0, 95.0),
}


# -----------------------------------------------------------------------------
# Metric configuration
# -----------------------------------------------------------------------------
def metric_configs() -> Dict[str, Dict[str, Optional[str]]]:
    return {
        "density": {
            "value_col": "mean_hu_abs",
            "std_col": "std_hu_abs",
            "title": "Mean density",
            "colorbar_label": "Mean density (HU)",
            "file_stub": "density_hu_abs",
            "value_fmt": ".1f",
            "unit": " HU",
        },
        "volume": {
            "value_col": "volume_fraction_icv_percent",
            "std_col": None,
            "title": "Volume density",
            "colorbar_label": "Volume density relative to ICV (%)",
            "file_stub": "volume_fraction_icv_percent",
            "value_fmt": ".3f",
            "unit": " %",
        },
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate two static polar charts and one interactive HTML chart per case "
            "from a vein-features CSV. The two metrics are mean density in absolute "
            "HU and volume fraction relative to ICV. Outputs are saved inside the "
            "corresponding case folders under the chosen output directory."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python generate_polar_charts.py --input results/vein_features_all_cases.csv --output results\n"
            "  python generate_polar_charts.py --input results/vein_features_all_cases.csv --output results --cases sub-stroke_0002 sub-stroke_0004\n"
            "  python generate_polar_charts.py --input results/vein_features_all_cases.csv --output results --scale-scope case\n"
            "  python generate_polar_charts.py --input results/vein_features_all_cases.csv --output results --list-cases\n"
        ),
    )
    parser.add_argument("--input", type=Path, required=True, help="Path to vein_features_all_cases.csv")
    parser.add_argument("--output", type=Path, required=True, help="Base output directory where case folders live")
    parser.add_argument(
        "--cases",
        nargs="+",
        help="Specific case IDs to process. If omitted, all cases in the CSV are processed.",
    )
    parser.add_argument("--list-cases", action="store_true", help="List available case IDs and exit.")
    parser.add_argument(
        "--scale-scope",
        choices=["global", "selected", "case"],
        default="global",
        help=(
            "How to set the color scale for each metric:\n"
            "  global   = use all cases in the CSV\n"
            "  selected = use only the selected cases\n"
            "  case     = each case gets its own robust scale"
        ),
    )
    parser.add_argument(
        "--image-format",
        choices=["png", "jpg", "jpeg"],
        default="png",
        help="Static image format for saved charts (default: png).",
    )
    parser.add_argument("--dpi", type=int, default=300, help="DPI for saved static charts (default: 300).")
    parser.add_argument(
        "--bg-color",
        default=DEFAULT_BG_COLOR,
        help=f"Background color for static and interactive charts (default: {DEFAULT_BG_COLOR}).",
    )
    return parser


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------
def unwrap_angles(theta1: float, theta2: float) -> Tuple[float, float]:
    t1 = float(theta1)
    t2 = float(theta2)
    if t2 <= t1:
        t2 += 360.0
    return t1, t2

def compute_text_position(item: Dict[str, float]) -> Tuple[float, float]:
    t1, t2 = unwrap_angles(item["theta1"], item["theta2"])
    if math.isclose(t2 - t1, 360.0, rel_tol=0.0, abs_tol=1e-9):
        return 0.0, 0.0
    mid_deg = (t1 + t2) / 2.0
    mid_rad = np.deg2rad(mid_deg)
    r_text = (float(item["r0"]) + float(item["r1"])) / 2.0
    return r_text * np.cos(mid_rad), r_text * np.sin(mid_rad)

def sector_polygon(item: Dict[str, float], n_points: int = 120) -> Tuple[np.ndarray, np.ndarray]:
    t1, t2 = unwrap_angles(item["theta1"], item["theta2"])
    r0 = float(item["r0"])
    r1 = float(item["r1"])
    theta = np.deg2rad(np.linspace(t1, t2, n_points))
    x_outer = r1 * np.cos(theta)
    y_outer = r1 * np.sin(theta)
    x_inner = r0 * np.cos(theta[::-1])
    y_inner = r0 * np.sin(theta[::-1])
    x = np.concatenate([x_outer, x_inner, [x_outer[0]]])
    y = np.concatenate([y_outer, y_inner, [y_outer[0]]])
    return x, y


# -----------------------------------------------------------------------------
# Data helpers
# -----------------------------------------------------------------------------
def resolve_path(path: Path) -> Path:
    return path.expanduser().resolve()

def warn_if_windows_rootlike_output(path: Path) -> None:
    if platform.system().lower().startswith("win"):
        p = str(path).replace("\\", "/")
        if p.startswith("/results"):
            print("Warning: on Windows, '/results' points to a drive-root location, not your project folder.")
            print("Use something like './results' or 'results' instead.")

def load_dataframe(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {
        "case",
        "structure",
        "mean_hu_abs",
        "std_hu_abs",
        "volume_fraction_icv_percent",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")
    return df

def determine_selected_cases(df: pd.DataFrame, requested_cases: Optional[Iterable[str]], list_only: bool = False) -> List[str]:
    # Simplified case extraction
    available_cases = df["case"].astype(str).unique().tolist()

    if list_only:
        for case in available_cases:
            print(case)
        raise SystemExit(0)

    if requested_cases is None:
        return available_cases

    requested = [str(c) for c in requested_cases]
    missing = sorted(set(requested) - set(available_cases))
    if missing:
        raise ValueError(f"Requested case(s) not found in CSV: {missing}")
    return requested

def validate_case_frame(case_df: pd.DataFrame, case: str) -> None:
    duplicated = case_df[case_df.duplicated(subset=["structure"], keep=False)]
    if not duplicated.empty:
        dups = duplicated["structure"].tolist()
        raise ValueError(f"Case {case} has duplicated structure rows: {dups}")

def robust_limits(values: np.ndarray, metric_name: str) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (0.0, 1.0)

    lo_pct, hi_pct = PERCENTILES_BY_METRIC[metric_name]
    vmin = float(np.nanpercentile(values, lo_pct))
    vmax = float(np.nanpercentile(values, hi_pct))
    if math.isclose(vmin, vmax, rel_tol=0.0, abs_tol=1e-12):
        vmax = vmin + 1e-9
    return vmin, vmax

def build_scale_limits(
    df: pd.DataFrame,
    selected_cases: List[str],
    metric_cfgs: Dict[str, Dict[str, Optional[str]]],
    scale_scope: str,
) -> Dict[str, Tuple[float, float]]:
    if scale_scope == "global":
        src = df
    elif scale_scope == "selected":
        src = df[df["case"].isin(selected_cases)]
    elif scale_scope == "case":
        return {}
    else:
        raise ValueError(f"Unsupported scale scope: {scale_scope}")

    limits: Dict[str, Tuple[float, float]] = {}
    for metric_name, cfg in metric_cfgs.items():
        vals = pd.to_numeric(src[cfg["value_col"]], errors="coerce").dropna().to_numpy(dtype=float)
        if vals.size == 0:
            raise ValueError(f"No valid values found for metric '{metric_name}'")
        limits[metric_name] = robust_limits(vals, metric_name)
    return limits

def case_metric_lookup(
    case_df: pd.DataFrame,
    metric_cfgs: Dict[str, Dict[str, Optional[str]]],
) -> Dict[str, Dict[str, Dict[str, Optional[float]]]]:
    by_structure = case_df.set_index("structure")
    out: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}

    for metric_name, cfg in metric_cfgs.items():
        metric_map: Dict[str, Dict[str, Optional[float]]] = {}
        for structure in STRUCTURE_ORDER:
            if structure in by_structure.index:
                row = by_structure.loc[structure]
                value = pd.to_numeric(row[cfg["value_col"]], errors="coerce")
                std_col = cfg.get("std_col")
                std = pd.to_numeric(row[std_col], errors="coerce") if std_col else np.nan
                metric_map[structure] = {
                    "value": None if pd.isna(value) else float(value),
                    "std": None if pd.isna(std) else float(std),
                }
            else:
                metric_map[structure] = {"value": None, "std": None}
        out[metric_name] = metric_map
    return out


# -----------------------------------------------------------------------------
# Matplotlib static charts
# -----------------------------------------------------------------------------
def add_sector_matplotlib(ax, item: Dict[str, float], facecolor, edgecolor: str = EDGE_COLOR, lw: float = 2.0) -> None:
    t1, t2 = unwrap_angles(item["theta1"], item["theta2"])
    wedge = Wedge(
        center=(0, 0),
        r=float(item["r1"]),
        theta1=t1,
        theta2=t2,
        width=float(item["r1"] - item["r0"]),
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=lw,
    )
    ax.add_patch(wedge)

    text = item["display"]
    if len(text) > 10:
        fontsize = 8.5
    elif len(text) > 7:
        fontsize = 9.5
    else:
        fontsize = 10.5

    x, y = compute_text_position(item)
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        color="black",
    )

def build_static_chart(
    case: str,
    metric_name: str,
    metric_cfg: Dict[str, Optional[str]],
    metric_values: Dict[str, Dict[str, Optional[float]]],
    limits: Tuple[float, float],
    bg_color: str,
) -> mpl.figure.Figure:
    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    ax.set_aspect("equal")
    ax.axis("off")

    cmap = plt.get_cmap(CMAP_NAME)
    norm = mpl.colors.Normalize(vmin=limits[0], vmax=limits[1])

    for structure in STRUCTURE_ORDER:
        item = LAYOUT_BY_STRUCTURE[structure]
        value = metric_values[structure]["value"]
        if value is None:
            facecolor = MISSING_FACE
        else:
            clipped = float(np.clip(value, limits[0], limits[1]))
            facecolor = cmap(norm(clipped))
        add_sector_matplotlib(ax, item, facecolor)

    pad = 0.5
    ax.set_xlim(-MAX_RADIUS - pad, MAX_RADIUS + pad)
    ax.set_ylim(-MAX_RADIUS - pad, MAX_RADIUS + pad)
    ax.set_title(f"{case} - {metric_cfg['title']}", fontsize=16, fontweight="bold", pad=18)

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.05, pad=0.04)
    cbar.set_label(metric_cfg["colorbar_label"], rotation=270, labelpad=18)
    if metric_name == "volume":
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    else:
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))

    plt.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# Plotly interactive chart
# -----------------------------------------------------------------------------
def rgba_to_plotly(rgba: Tuple[float, float, float, float]) -> str:
    r, g, b, a = rgba
    return f"rgba({int(round(r * 255))},{int(round(g * 255))},{int(round(b * 255))},{a:.6f})"

def metric_hover_text(item: Dict[str, float], metric_cfg: Dict[str, Optional[str]], entry: Dict[str, Optional[float]]) -> str:
    shown = item["display"]
    value = entry["value"]
    std = entry["std"]
    if value is None:
        return "<extra></extra>"

    fmt = metric_cfg["value_fmt"]
    unit = metric_cfg["unit"] or ""
    text = f"<b>{shown}</b><br>{metric_cfg['title']}: {value:{fmt}}{unit}"
    if std is not None:
        text += f"<br>SD: {std:{fmt}}{unit}"
    return text + "<extra></extra>"

def mpl_cmap_to_plotly(cmap_name: str, n: int = 256) -> List[List[object]]:
    cmap = plt.get_cmap(cmap_name)
    colorscale: List[List[object]] = []
    for i in range(n):
        x = i / max(n - 1, 1)
        colorscale.append([x, rgba_to_plotly(cmap(x))])
    return colorscale

def build_plotly_figure(
    case: str,
    metric_cfgs: Dict[str, Dict[str, Optional[str]]],
    case_metrics: Dict[str, Dict[str, Dict[str, Optional[float]]]],
    scale_limits: Dict[str, Tuple[float, float]],
    bg_color: str,
) -> go.Figure:
    fig = go.Figure()
    metric_names = list(metric_cfgs.keys())
    n_structures = len(STRUCTURE_ORDER)
    colorscale = mpl_cmap_to_plotly(CMAP_NAME)

    for metric_index, metric_name in enumerate(metric_names):
        cfg = metric_cfgs[metric_name]
        limits = scale_limits[metric_name]
        cmap = plt.get_cmap(CMAP_NAME)
        norm = mpl.colors.Normalize(vmin=limits[0], vmax=limits[1])

        for structure in STRUCTURE_ORDER:
            item = LAYOUT_BY_STRUCTURE[structure]
            x, y = sector_polygon(item)
            entry = case_metrics[metric_name][structure]
            value = entry["value"]
            if value is None:
                fillcolor = MISSING_FACE
            else:
                clipped = float(np.clip(value, limits[0], limits[1]))
                fillcolor = rgba_to_plotly(cmap(norm(clipped)))

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    fill="toself",
                    fillcolor=fillcolor,
                    line=dict(color=EDGE_COLOR, width=2),
                    hoverinfo="skip",
                    showlegend=False,
                    visible=(metric_index == 0),
                )
            )

            cx, cy = compute_text_position(item)
            hovertemplate = metric_hover_text(item, cfg, entry)
            fig.add_trace(
                go.Scatter(
                    x=[cx],
                    y=[cy],
                    mode="markers",
                    marker=dict(size=28, color="rgba(0,0,0,0.001)"),
                    hovertemplate=hovertemplate,
                    hoverinfo="text",
                    hoverlabel=dict(bgcolor="white", font=dict(color="black")),
                    showlegend=False,
                    visible=(metric_index == 0),
                )
            )

        cmin, cmax = limits
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(
                    colorscale=colorscale,
                    cmin=cmin,
                    cmax=cmax,
                    color=[cmin, cmax],
                    showscale=True,
                    colorbar=dict(
                        title=cfg["colorbar_label"],
                        x=1.02,
                        y=0.5,
                        len=0.82,
                        thickness=18,
                        tickformat=".0f" if metric_name == "density" else None
                    ),
                    size=0.1,
                ),
                hoverinfo="skip",
                showlegend=False,
                visible=(metric_index == 0),
            )
        )

    traces_per_metric = n_structures * 2 + 1
    buttons = []
    for metric_index, metric_name in enumerate(metric_names):
        visible = [False] * len(fig.data)
        start = metric_index * traces_per_metric
        end = start + traces_per_metric
        for i in range(start, end):
            visible[i] = True
        buttons.append(
            dict(
                label=metric_cfgs[metric_name]["title"],
                method="update",
                args=[
                    {"visible": visible},
                    {"title": {"text": f"{case} - {metric_cfgs[metric_name]['title']}", "x": 0.5, "y": 0.96, "xanchor": "center"}},
                ],
            )
        )

    annotations = []
    for structure in STRUCTURE_ORDER:
        item = LAYOUT_BY_STRUCTURE[structure]
        x, y = compute_text_position(item)
        disp = item["display"]
        if len(disp) > 10:
            font_size = 11
        elif len(disp) > 7:
            font_size = 12
        else:
            font_size = 13
        annotations.append(
            dict(
                x=x,
                y=y,
                text=f"<b>{disp}</b>",
                showarrow=False,
                font=dict(size=font_size, color="black"),
                xanchor="center",
                yanchor="middle",
            )
        )

    pad = 0.5
    fig.update_layout(
        title=dict(text=f"{case} - {metric_cfgs[metric_names[0]]['title']}", x=0.5, y=0.96),
        width=900,
        height=860,
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        margin=dict(l=30, r=90, t=120, b=30),
        xaxis=dict(visible=False, scaleanchor="y", scaleratio=1, range=[-MAX_RADIUS - pad, MAX_RADIUS + pad]),
        yaxis=dict(visible=False, range=[-MAX_RADIUS - pad, MAX_RADIUS + pad]),
        updatemenus=[
            dict(
                type="dropdown",
                buttons=buttons,
                x=0.5,
                y=1.10,
                xanchor="center",
                yanchor="top",
                showactive=True,
            )
        ],
        annotations=annotations,
    )
    return fig


# -----------------------------------------------------------------------------
# Case processing
# -----------------------------------------------------------------------------
def process_case(
    case: str,
    case_df: pd.DataFrame,
    out_dir: Path,
    metric_cfgs: Dict[str, Dict[str, Optional[str]]],
    global_limits: Dict[str, Tuple[float, float]],
    scale_scope: str,
    image_format: str,
    dpi: int,
    bg_color: str,
) -> None:
    validate_case_frame(case_df, case)
    case_dir = out_dir / case
    case_dir.mkdir(parents=True, exist_ok=True)

    # Simplified lookup
    present_structures = set(case_df["structure"])
    missing = [s for s in STRUCTURE_ORDER if s not in present_structures]
    
    if missing:
        print(f"[WARN] {case}: missing structures will be shown in gray: {missing}")

    case_metrics = case_metric_lookup(case_df, metric_cfgs)

    if scale_scope == "case":
        scale_limits = {}
        for metric_name in metric_cfgs:
            vals = [entry["value"] for entry in case_metrics[metric_name].values() if entry["value"] is not None]
            scale_limits[metric_name] = robust_limits(np.asarray(vals, dtype=float), metric_name)
    else:
        scale_limits = global_limits

    for metric_name, cfg in metric_cfgs.items():
        fig = build_static_chart(
            case=case,
            metric_name=metric_name,
            metric_cfg=cfg,
            metric_values=case_metrics[metric_name],
            limits=scale_limits[metric_name],
            bg_color=bg_color,
        )
        out_path = case_dir / f"{case}_polar_{cfg['file_stub']}.{image_format}"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor=bg_color)
        plt.close(fig)

    interactive = build_plotly_figure(
        case=case,
        metric_cfgs=metric_cfgs,
        case_metrics=case_metrics,
        scale_limits=scale_limits,
        bg_color=bg_color,
    )
    interactive.write_html(case_dir / f"{case}_polar_metrics_interactive.html", include_plotlyjs="cdn")
    print(f"Saved charts for {case} -> {case_dir}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)
    warn_if_windows_rootlike_output(args.output)

    print(f"Input CSV: {input_path}")
    print(f"Output dir: {output_path}")

    if not input_path.exists():
        raise FileNotFoundError(f"CSV file not found: {input_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    df = load_dataframe(input_path)
    selected_cases = determine_selected_cases(df, args.cases, list_only=args.list_cases)
    metric_cfgs = metric_configs()
    global_limits = build_scale_limits(df, selected_cases, metric_cfgs, args.scale_scope)

    for case in selected_cases:
        case_df = df[df["case"] == case].copy()
        process_case(
            case=case,
            case_df=case_df,
            out_dir=output_path,
            metric_cfgs=metric_cfgs,
            global_limits=global_limits,
            scale_scope=args.scale_scope,
            image_format=args.image_format,
            dpi=args.dpi,
            bg_color=args.bg_color,
        )


if __name__ == "__main__":
    main()