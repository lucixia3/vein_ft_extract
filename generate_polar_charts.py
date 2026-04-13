"""
Generate polar venous charts from a vein-features CSV.

For each case, the script creates three static PNG charts and one interactive
HTML chart in the case output folder. The charts summarize maximum diameter,
density (relative to SSS by default, or absolute HU), and volume fraction
relative to ICV on a fixed schematic polar layout.

Missing structures do not stop the batch: their sectors are drawn in gray.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.patches import Wedge
from matplotlib.ticker import FormatStrFormatter

# -----------------------------
# Fixed geometry / labels
# -----------------------------
LAYOUT = [
    # Center
    {"label": "ICV", "r0": 0.0, "r1": 0.8, "theta1": 0, "theta2": 360},
    # Ring 1
    {"label": "VOG", "r0": 0.8, "r1": 1.6, "theta1": 45, "theta2": 135},
    {"label": "STS", "r0": 0.8, "r1": 1.6, "theta1": 180, "theta2": 360},
    # Spacer ring: 1.6 -> 2.4
    # Ring 3
    {"label": "RBVR", "r0": 2.4, "r1": 3.1, "theta1": 45, "theta2": 135},
    {"label": "Cortical-L", "r0": 2.4, "r1": 3.1, "theta1": 135, "theta2": 260},
    {"label": "Cortical-R", "r0": 2.4, "r1": 3.1, "theta1": 280, "theta2": 45},
    # Ring 4
    {"label": "VCM-L", "r0": 3.1, "r1": 4.1, "theta1": 90, "theta2": 270},
    {"label": "VCM-R", "r0": 3.1, "r1": 4.1, "theta1": 270, "theta2": 90},
    # Ring 5
    {"label": "SSS", "r0": 4.1, "r1": 5.2, "theta1": 0, "theta2": 180},
    {"label": "TransvSig-L", "r0": 4.1, "r1": 5.2, "theta1": 180, "theta2": 255},
    {"label": "TransvSig-R", "r0": 4.1, "r1": 5.2, "theta1": 285, "theta2": 360},
]

LABEL_ORDER = [item["label"] for item in LAYOUT]
LAYOUT_BY_LABEL = {item["label"]: item for item in LAYOUT}
MAX_RADIUS = max(item["r1"] for item in LAYOUT)
DISPLAY_LABELS = {
    "Cortical-L": "Cort-L",
    "Cortical-R": "Cort-R",
}
MISSING_FACE = "#bdbdbd"
EDGE_COLOR = "#1f4ea8"
BG_COLOR = "#e0e0e0"
PCT_LOW = 5.0
PCT_HIGH = 95.0


def metric_configs(density_mode: str) -> Dict[str, Dict[str, Optional[str]]]:
    density_mode = density_mode.lower()
    if density_mode not in {"ref_sss", "abs"}:
        raise ValueError(f"Unsupported density mode: {density_mode}")

    density_cfg = {
        "ref_sss": {
            "value_col": "mean_hu_ref_sss",
            "std_col": "std_hu_ref_sss",
            "title": "Mean density relative to SSS",
            "colorbar_label": "Mean density relative to SSS",
            "file_stub": "density_ref_sss",
            "value_fmt": ".3f",
            "unit": "",
            "center": 1.0,
        },
        "abs": {
            "value_col": "mean_hu_abs",
            "std_col": "std_hu_abs",
            "title": "Mean density (HU)",
            "colorbar_label": "Mean density (HU)",
            "file_stub": "density_hu_abs",
            "value_fmt": ".1f",
            "unit": " HU",
            "center": None,
        },
    }[density_mode]

    return {
        "diameter": {
            "value_col": "max_diameter_mm",
            "std_col": "std_diameter_mm",
            "title": "Maximum diameter",
            "colorbar_label": "Maximum diameter (mm)",
            "file_stub": "max_diameter_mm",
            "value_fmt": ".3f",
            "unit": " mm",
            "center": None,
        },
        "density": density_cfg,
        "volume": {
            "value_col": "volume_fraction_icv_percent",
            "std_col": None,
            "title": "Volume fraction relative to ICV (%)",
            "colorbar_label": "Volume fraction relative to ICV (%)",
            "file_stub": "volume_fraction_icv_percent",
            "value_fmt": ".3f",
            "unit": " %",
            "center": None,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate static PNG and interactive HTML polar venous charts from "
            "a vein-features CSV. Outputs are saved inside the corresponding "
            "case folders under the chosen output directory."
        )
    )
    parser.add_argument("--input", required=True, type=Path, help="Path to vein_features_all_cases.csv")
    parser.add_argument("--output", required=True, type=Path, help="Base output directory (e.g. results)")
    parser.add_argument("--cases", nargs="+", help="Optional list of case IDs to process")
    parser.add_argument(
        "--density-mode",
        default="ref_sss",
        choices=["ref_sss", "abs"],
        help="Use density relative to SSS (default) or absolute HU",
    )
    parser.add_argument(
        "--scale-scope",
        default="global",
        choices=["global", "selected", "case"],
        help=(
            "How color limits are computed: across all cases, only selected cases, "
            "or independently per case"
        ),
    )
    parser.add_argument("--list-cases", action="store_true", help="List available cases and exit")
    parser.add_argument(
        "--dpi", type=int, default=300, help="DPI for saved PNG charts (default: 300)"
    )
    return parser.parse_args()


# -----------------------------
# Geometry helpers
# -----------------------------
def display_label(label: str) -> str:
    return DISPLAY_LABELS.get(label, label)



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


# -----------------------------
# Data prep / scaling
# -----------------------------
def load_dataframe(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    expected = {
        "case",
        "structure",
        "mean_hu_abs",
        "std_hu_abs",
        "mean_hu_ref_sss",
        "std_hu_ref_sss",
        "volume_fraction_icv_percent",
        "max_diameter_mm",
        "std_diameter_mm",
    }
    missing = sorted(expected - set(df.columns))
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")
    return df



def get_selected_cases(df: pd.DataFrame, requested_cases: Optional[Iterable[str]]) -> List[str]:
    available = list(dict.fromkeys(df["case"].astype(str).tolist()))
    if requested_cases is None:
        return available
    requested = [str(c) for c in requested_cases]
    unknown = [c for c in requested if c not in set(available)]
    if unknown:
        raise ValueError(f"Requested cases not found in CSV: {unknown}")
    return requested



def validate_case_frame(case_df: pd.DataFrame, case: str) -> None:
    duplicated = case_df[case_df.duplicated(subset=["structure"], keep=False)]
    if not duplicated.empty:
        dups = duplicated["structure"].tolist()
        raise ValueError(f"Case {case} has duplicated structure rows: {dups}")



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
        limits[metric_name] = robust_limits(vals, center=cfg.get("center"))
    return limits



def robust_limits(values: np.ndarray, center: Optional[float] = None) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (0.0, 1.0)

    lo = float(np.percentile(values, PCT_LOW))
    hi = float(np.percentile(values, PCT_HIGH))

    if center is not None:
        dist = max(abs(lo - center), abs(hi - center), 1e-6)
        vmin = center - dist
        vmax = center + dist
    else:
        vmin, vmax = lo, hi
        if math.isclose(vmin, vmax, rel_tol=0.0, abs_tol=1e-12):
            vmax = vmin + 1e-9

    return float(vmin), float(vmax)



def get_norm(metric_cfg: Dict[str, Optional[str]], limits: Tuple[float, float]):
    vmin, vmax = limits
    if metric_cfg.get("center") is not None:
        return mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    return mpl.colors.Normalize(vmin=vmin, vmax=vmax)



def clip_for_color(value: float, limits: Tuple[float, float]) -> float:
    vmin, vmax = limits
    return float(np.clip(value, vmin, vmax))



def case_metric_lookup(case_df: pd.DataFrame, metric_cfgs: Dict[str, Dict[str, Optional[str]]]) -> Dict[str, Dict[str, Dict[str, Optional[float]]]]:
    by_structure = case_df.set_index("structure")
    out: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
    for metric_name, cfg in metric_cfgs.items():
        metric_map: Dict[str, Dict[str, Optional[float]]] = {}
        for label in LABEL_ORDER:
            if label in by_structure.index:
                row = by_structure.loc[label]
                value = pd.to_numeric(row[cfg["value_col"]], errors="coerce")
                std_col = cfg.get("std_col")
                std = pd.to_numeric(row[std_col], errors="coerce") if std_col else np.nan
                metric_map[label] = {
                    "value": None if pd.isna(value) else float(value),
                    "std": None if pd.isna(std) else float(std),
                }
            else:
                metric_map[label] = {"value": None, "std": None}
        out[metric_name] = metric_map
    return out


# -----------------------------
# Matplotlib static charts
# -----------------------------
def add_sector_matplotlib(
    ax,
    item: Dict[str, float],
    label: str,
    facecolor,
    edgecolor: str = EDGE_COLOR,
    lw: float = 2.0,
) -> None:
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

    x, y = compute_text_position(item)
    text = display_label(label)
    if len(text) > 10:
        fontsize = 8.5
    elif len(text) > 7:
        fontsize = 9.5
    else:
        fontsize = 10.5

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
) -> mpl.figure.Figure:
    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_aspect("equal")
    ax.axis("off")

    cmap = plt.get_cmap("YlOrRd")
    norm = get_norm(metric_cfg, limits)

    for label in LABEL_ORDER:
        item = LAYOUT_BY_LABEL[label]
        value = metric_values[label]["value"]
        if value is None:
            facecolor = MISSING_FACE
        else:
            facecolor = cmap(norm(clip_for_color(value, limits)))
        add_sector_matplotlib(ax, item, label, facecolor)

    pad = 0.5
    ax.set_xlim(-MAX_RADIUS - pad, MAX_RADIUS + pad)
    ax.set_ylim(-MAX_RADIUS - pad, MAX_RADIUS + pad)
    ax.set_title(f"{metric_cfg['title']}\n{case}", fontsize=16, fontweight="bold", pad=18)

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.05, pad=0.04)
    cbar.set_label(metric_cfg["colorbar_label"], rotation=270, labelpad=18)
    if metric_name in {"volume", "density"}:
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    else:
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    plt.tight_layout()
    return fig


# -----------------------------
# Plotly interactive chart
# -----------------------------
def mpl_cmap_to_plotly(cmap_name: str, n: int = 256) -> List[List[object]]:
    cmap = plt.get_cmap(cmap_name)
    colorscale: List[List[object]] = []
    for i in range(n):
        x = i / max(n - 1, 1)
        r, g, b, a = cmap(x)
        colorscale.append([x, f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{a:.6f})"])
    return colorscale



def metric_hover_text(label: str, metric_cfg: Dict[str, Optional[str]], entry: Dict[str, Optional[float]]) -> str:
    shown = display_label(label)
    value = entry["value"]
    std = entry["std"]
    if value is None:
        return f"<b>{shown}</b><br>No data available<extra></extra>"

    fmt = metric_cfg["value_fmt"]
    unit = metric_cfg["unit"] or ""
    text = f"<b>{shown}</b><br>{metric_cfg['title']}: {value:{fmt}}{unit}"
    if std is not None:
        text += f"<br>SD: {std:{fmt}}{unit}"
    return text + "<extra></extra>"



def build_plotly_figure(
    case: str,
    metric_cfgs: Dict[str, Dict[str, Optional[str]]],
    case_metrics: Dict[str, Dict[str, Dict[str, Optional[float]]]],
    scale_limits: Dict[str, Tuple[float, float]],
) -> go.Figure:
    fig = go.Figure()
    metric_names = list(metric_cfgs.keys())
    n_labels = len(LABEL_ORDER)
    colorscale = mpl_cmap_to_plotly("YlOrRd")

    for metric_index, metric_name in enumerate(metric_names):
        cfg = metric_cfgs[metric_name]
        limits = scale_limits[metric_name]
        norm = get_norm(cfg, limits)

        for label in LABEL_ORDER:
            item = LAYOUT_BY_LABEL[label]
            x, y = sector_polygon(item)
            entry = case_metrics[metric_name][label]
            value = entry["value"]
            if value is None:
                fillcolor = MISSING_FACE
            else:
                rgba = plt.get_cmap("YlOrRd")(norm(clip_for_color(value, limits)))
                fillcolor = f"rgba({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)},{rgba[3]:.6f})"

            hovertemplate = metric_hover_text(label, cfg, entry)
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

            tx, ty = compute_text_position(item)
            fig.add_trace(
                go.Scatter(
                    x=[tx],
                    y=[ty],
                    mode="text",
                    text=[display_label(label)],
                    textfont=dict(size=12, color="black", family="Arial Black"),
                    hoverinfo="skip",
                    showlegend=False,
                    visible=(metric_index == 0),
                )
            )

            # Invisible centroid marker used only for hover, because filled polygon
            # traces often show generic Plotly trace labels instead of useful values.
            fig.add_trace(
                go.Scatter(
                    x=[tx],
                    y=[ty],
                    mode="markers",
                    marker=dict(size=24, color="rgba(0,0,0,0.001)"),
                    hovertemplate=hovertemplate,
                    hoverlabel=dict(bgcolor="white"),
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
                    ),
                    size=0.1,
                ),
                hoverinfo="skip",
                showlegend=False,
                visible=(metric_index == 0),
            )
        )

    total_traces_per_metric = n_labels * 3 + 1
    buttons = []
    for metric_index, metric_name in enumerate(metric_names):
        visible = [False] * (len(metric_names) * total_traces_per_metric)
        start = metric_index * total_traces_per_metric
        end = start + total_traces_per_metric
        for i in range(start, end):
            visible[i] = True
        buttons.append(
            dict(
                label=metric_cfgs[metric_name]["title"],
                method="update",
                args=[
                    {"visible": visible},
                    {"title": {"text": f"{case} — {metric_cfgs[metric_name]['title']}", "y": 0.96}},
                ],
            )
        )

    fig.update_layout(
        title=dict(text=f"{case} — {metric_cfgs[metric_names[0]]['title']}", x=0.5, y=0.96),
        width=900,
        height=860,
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        margin=dict(l=30, r=90, t=120, b=30),
        xaxis=dict(visible=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(visible=False),
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
    )
    pad = 0.5
    fig.update_xaxes(range=[-MAX_RADIUS - pad, MAX_RADIUS + pad])
    fig.update_yaxes(range=[-MAX_RADIUS - pad, MAX_RADIUS + pad])
    return fig


# -----------------------------
# Case processing
# -----------------------------
def process_case(
    case: str,
    case_df: pd.DataFrame,
    out_dir: Path,
    metric_cfgs: Dict[str, Dict[str, Optional[str]]],
    global_limits: Dict[str, Tuple[float, float]],
    scale_scope: str,
    dpi: int,
) -> None:
    validate_case_frame(case_df, case)
    case_dir = out_dir / case
    case_dir.mkdir(parents=True, exist_ok=True)

    missing = [label for label in LABEL_ORDER if label not in set(case_df["structure"].tolist())]
    if missing:
        print(f"[WARN] {case}: missing structures will be shown in gray: {missing}")

    case_metrics = case_metric_lookup(case_df, metric_cfgs)

    if scale_scope == "case":
        scale_limits = {}
        for metric_name, cfg in metric_cfgs.items():
            vals = [entry["value"] for entry in case_metrics[metric_name].values() if entry["value"] is not None]
            vals_arr = np.asarray(vals, dtype=float)
            scale_limits[metric_name] = robust_limits(vals_arr, center=cfg.get("center"))
    else:
        scale_limits = global_limits

    for metric_name, cfg in metric_cfgs.items():
        fig = build_static_chart(case, metric_name, cfg, case_metrics[metric_name], scale_limits[metric_name])
        out_path = case_dir / f"{case}_polar_{cfg['file_stub']}.png"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    interactive = build_plotly_figure(case, metric_cfgs, case_metrics, scale_limits)
    interactive.write_html(case_dir / f"{case}_polar_metrics_interactive.html", include_plotlyjs="cdn")
    print(f"Saved charts for {case} -> {case_dir}")



def main() -> None:
    args = parse_args()

    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    if input_path.suffix.lower() != ".csv":
        raise ValueError(f"Input must be a CSV file: {input_path}")

    if Path("/") == args.output and str(output_path).endswith("\\results"):
        pass

    df = load_dataframe(input_path)

    if args.list_cases:
        for case in dict.fromkeys(df["case"].astype(str).tolist()):
            print(case)
        return

    selected_cases = get_selected_cases(df, args.cases)
    metric_cfgs = metric_configs(args.density_mode)
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
            dpi=args.dpi,
        )


if __name__ == "__main__":
    main()
