from __future__ import annotations
import argparse
import html as html_mod
import json
import os
from pathlib import Path
import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
import pandas as pd
from generar_viewer import build_segmentation_voxel_cloud_diameter, build_segmentation_voxel_cloud_hu
LABEL_PALETTE = {6: ('#6b9bd2', 'L6 (single)'), 9: ('#43D9A2', 'L9 multi (9+11)'), 10: ('#fd9644', 'L10 multi (10+12)')}

def stem_from_seg_path(seg_path: str) -> str:
    p = Path(seg_path)
    n = p.name
    if n.endswith('.nii.gz'):
        return n[:-len('.nii.gz')]
    return p.stem

def _cta_name_score(path: Path, stem: str) -> int:
    n = path.name.lower()
    st = stem.lower().replace(' ', '')
    sc = 0
    if path.stem.lower() == st:
        sc += 55
    elif st in path.stem.lower().replace('_', '').replace('-', ''):
        sc += 28
    for kw in ('_ct.nii', '_cta.nii', 'cta_', 'ct_', '_0000.nii', '_image.nii', '_volume'):
        if kw in n:
            sc += 14
    for kw in ('ct', 'cta', 'image', 'volume', 'scan', 'grayscale', 'raw'):
        if kw in n:
            sc += 5
    for bad in ('seg', 'label', 'mask', 'pred', 'segment', 'artery_label'):
        if bad in n:
            sc -= 30
    return sc

def _try_resolve_cta_in_dir(d: Path, stem: str, seg_res: Path, extra_names: tuple[str, ...]) -> str | None:
    if not d.is_dir():
        return None

    def pick(hit: Path) -> str | None:
        if not hit.is_file() or hit.resolve() == seg_res:
            return None
        return str(hit)
    for name in extra_names:
        out = pick(d / name)
        if out:
            return out
    for hit in sorted(d.glob(f'*{stem}*.nii.gz')):
        out = pick(hit)
        if out:
            return out
    all_nii = [p for p in d.glob('*.nii.gz') if p.resolve() != seg_res]
    if len(all_nii) == 1:
        return str(all_nii[0])
    if len(all_nii) > 1:
        best = max(all_nii, key=lambda p: _cta_name_score(p, stem))
        if _cta_name_score(best, stem) >= 12:
            return str(best)
    return None

def resolve_cta_path(seg_path: str, cta_arg: Path | None) -> str | None:
    seg_res = Path(seg_path).resolve()
    if cta_arg is not None:
        p = Path(cta_arg)
        return str(p) if p.is_file() else None
    stem = stem_from_seg_path(seg_path)
    root = Path(__file__).resolve().parent
    seg_par = seg_res.parent
    grand = seg_par.parent
    extra_names = (f'{stem}.nii.gz', f'{stem}_ct.nii.gz', f'{stem}_CT.nii.gz', f'{stem}_cta.nii.gz', f'{stem}_0000.nii.gz', f'CT_{stem}.nii.gz', f'{stem}_image.nii.gz')
    env = os.environ.get('PHD_CTA_DIR', '').strip()
    env_dir = Path(env) if env else None
    dirs: list[Path] = []
    if env_dir is not None and env_dir.is_dir():
        dirs.append(env_dir)
    dirs.append(root / 'images')
    dirs.extend([root / 'cta', seg_par / 'images', seg_par, grand / 'images', grand])
    seen: set[Path] = set()
    for d in dirs:
        try:
            key = d.resolve()
        except OSError:
            continue
        if not d.is_dir() or key in seen:
            continue
        seen.add(key)
        hit = _try_resolve_cta_in_dir(d, stem, seg_res, extra_names)
        if hit:
            return hit
    return None

def _load_cta_volume_on_seg_grid(seg_path: str, cta_path: str) -> np.ndarray | None:
    snii = nib.load(seg_path)
    cnii = nib.load(cta_path)
    if tuple(snii.shape[:3]) != tuple(cnii.shape[:3]) or not np.allclose(snii.affine, cnii.affine, rtol=1e-05, atol=0.0001):
        try:
            return np.ascontiguousarray(resample_from_to(cnii, snii, order=1, mode='constant', cval=-1024.0).get_fdata().astype(np.float32))
        except Exception:
            return None
    return np.ascontiguousarray(cnii.get_fdata().astype(np.float32))

def _heatmap_z_jsonable(z: np.ndarray) -> list:
    out: list = []
    for row in np.asarray(z, dtype=float):
        out.append([float(x) if np.isfinite(x) else None for x in row])
    return out

def build_mip_2d_payload(seg_path: str, cta_path: str | None, cta_ok: bool, d_clip: tuple[float, float] | None, h_clip: tuple[float, float] | None, mip_axis: int=2) -> dict:
    from scipy import ndimage as ndi
    nii = nib.load(seg_path)
    aff = nii.affine
    data = np.round(nii.get_fdata()).astype(np.int16)
    spacing = tuple((float(x) for x in np.abs(np.diag(aff)[:3])))
    sp = np.abs(np.diag(aff)[:3]).astype(float)
    mask_union = np.zeros(data.shape, dtype=bool)
    vol_d = np.full(data.shape, np.nan, dtype=np.float32)
    for lab in LABEL_PALETTE:
        m = data == lab
        mask_union |= m
        if not m.any():
            continue
        edt = ndi.distance_transform_edt(m.astype(bool), sampling=spacing)
        vol_d[m] = 2.0 * edt[m]
    out: dict = {'hasDiam2d': False, 'hasHu2d': False, 'traceDiam': None, 'traceHu': None, 'layoutDiam2d': None, 'layoutHu2d': None, 'mip_axis': mip_axis}
    if not mask_union.any():
        return out
    v_d = np.where(np.isfinite(vol_d), vol_d, -np.inf)
    mip_d = np.max(v_d, axis=mip_axis)
    mip_d = np.where(np.isneginf(mip_d), np.nan, mip_d)
    flat_d = mip_d[np.isfinite(mip_d)]
    if flat_d.size > 0:
        if d_clip is not None:
            cmin_d, cmax_d = (float(d_clip[0]), float(d_clip[1]))
        elif flat_d.size > 50:
            cmin_d = float(np.percentile(flat_d, 1.0))
            cmax_d = float(np.percentile(flat_d, 99.0))
        else:
            cmin_d, cmax_d = (float(np.min(flat_d)), float(np.max(flat_d)))
        if cmax_d <= cmin_d:
            cmax_d = cmin_d + 1e-06
        ni, nj = mip_d.shape
        scaleratio = float(ni * float(sp[0]) / max(nj * float(sp[1]), 1e-09))
        zplot = np.where(np.isfinite(mip_d), np.round(mip_d, 3), np.nan)
        out['traceDiam'] = {'type': 'heatmap', 'z': _heatmap_z_jsonable(zplot), 'x': list(range(nj)), 'y': list(range(ni)), 'colorscale': 'Turbo', 'zmin': cmin_d, 'zmax': cmax_d, 'colorbar': {'title': {'text': 'Ø MIP (mm)', 'font': {'size': 14}}, 'tickfont': {'size': 12}, 'len': 0.92, 'thickness': 20, 'bgcolor': 'rgba(8,11,18,0.92)'}, 'hoverongaps': False}
        out['layoutDiam2d'] = {'paper_bgcolor': '#06080d', 'plot_bgcolor': '#080b10', 'font': {'color': '#c8d0dc'}, 'title': {'text': f'MIP eix {mip_axis} (màx. Ø per raig)', 'font': {'size': 13, 'color': '#93a7ff'}}, 'xaxis': {'title': {'text': 'Índex j (eix 1)', 'font': {'size': 11}}, 'gridcolor': '#1e2436', 'color': '#8b92a8'}, 'yaxis': {'title': {'text': 'Índex i (eix 0)', 'font': {'size': 11}}, 'gridcolor': '#1e2436', 'color': '#8b92a8', 'scaleanchor': 'x', 'scaleratio': scaleratio, 'autorange': 'reversed'}, 'margin': {'l': 70, 'r': 30, 't': 45, 'b': 55}}
        out['hasDiam2d'] = True
    if cta_ok and cta_path:
        cta_vol = _load_cta_volume_on_seg_grid(seg_path, cta_path)
        if cta_vol is not None:
            vol_h = np.full(data.shape, np.nan, dtype=np.float32)
            vol_h[mask_union] = cta_vol[mask_union]
            v_h = np.where(np.isfinite(vol_h), vol_h, -np.inf)
            mip_h = np.max(v_h, axis=mip_axis)
            mip_h = np.where(np.isneginf(mip_h), np.nan, mip_h)
            flat_h = mip_h[np.isfinite(mip_h)]
            if flat_h.size > 0:
                if h_clip is not None:
                    cmin_h, cmax_h = (float(h_clip[0]), float(h_clip[1]))
                elif flat_h.size > 50:
                    cmin_h = float(np.percentile(flat_h, 0.5))
                    cmax_h = float(np.percentile(flat_h, 99.5))
                else:
                    cmin_h, cmax_h = (float(np.min(flat_h)), float(np.max(flat_h)))
                if cmax_h <= cmin_h:
                    cmax_h = cmin_h + 0.001
                ni, nj = mip_h.shape
                scaleratio_h = float(ni * float(sp[0]) / max(nj * float(sp[1]), 1e-09))
                zplot_h = np.where(np.isfinite(mip_h), np.round(mip_h, 2), np.nan)
                out['traceHu'] = {'type': 'heatmap', 'z': _heatmap_z_jsonable(zplot_h), 'x': list(range(nj)), 'y': list(range(ni)), 'colorscale': 'RdYlBu_r', 'zmin': cmin_h, 'zmax': cmax_h, 'colorbar': {'title': {'text': 'HU MIP', 'font': {'size': 14}}, 'tickfont': {'size': 12}, 'len': 0.92, 'thickness': 20, 'bgcolor': 'rgba(8,11,18,0.92)'}, 'hoverongaps': False}
                out['layoutHu2d'] = {'paper_bgcolor': '#06080d', 'plot_bgcolor': '#080b10', 'font': {'color': '#c8d0dc'}, 'title': {'text': f'MIP eix {mip_axis} (màx. HU per raig)', 'font': {'size': 13, 'color': '#93a7ff'}}, 'xaxis': {'title': {'text': 'Índex j (eix 1)', 'font': {'size': 11}}, 'gridcolor': '#1e2436', 'color': '#8b92a8'}, 'yaxis': {'title': {'text': 'Índex i (eix 0)', 'font': {'size': 11}}, 'gridcolor': '#1e2436', 'color': '#8b92a8', 'scaleanchor': 'x', 'scaleratio': scaleratio_h, 'autorange': 'reversed'}, 'margin': {'l': 70, 'r': 30, 't': 45, 'b': 55}}
                out['hasHu2d'] = True
    return out

def cta_usable_for_seg(seg_path: str, cta_path: str) -> tuple[bool, str]:
    try:
        sn, cn = (nib.load(seg_path), nib.load(cta_path))
    except OSError:
        return (False, 'no_leer_nifti')
    if tuple(sn.shape[:3]) != tuple(cn.shape[:3]) or not np.allclose(sn.affine, cn.affine, rtol=1e-05, atol=0.0001):
        try:
            resample_from_to(cn, sn, order=1, mode='constant', cval=-1024.0)
            return (True, 'resampled')
        except Exception:
            return (False, 'resample_fallo')
    return (True, 'native')

def collect_diameter_bounds_mm(resultados: Path) -> tuple[float | None, float | None]:
    chunks: list[np.ndarray] = []
    for label in LABEL_PALETTE:
        ld = resultados / f'label_{label:03d}'
        if not ld.exists():
            continue
        for csv_p in sorted(ld.glob('comp_*_centerlines.csv')):
            df = pd.read_csv(csv_p)
            if 'diameter_mm' not in df.columns:
                continue
            v = df['diameter_mm'].dropna().to_numpy(dtype=float)
            if v.size:
                chunks.append(v)
    if not chunks:
        return (None, None)
    allv = np.concatenate(chunks)
    return (float(np.nanmin(allv)), float(np.nanmax(allv)))

def collect_hu_bounds(resultados: Path) -> tuple[float | None, float | None]:
    chunks: list[np.ndarray] = []
    for label in LABEL_PALETTE:
        ld = resultados / f'label_{label:03d}'
        if not ld.exists():
            continue
        for csv_p in sorted(ld.glob('comp_*_centerlines.csv')):
            df = pd.read_csv(csv_p)
            if 'hu' not in df.columns:
                continue
            v = df['hu'].dropna().to_numpy(dtype=float)
            if v.size:
                chunks.append(v)
    if not chunks:
        return (None, None)
    allv = np.concatenate(chunks)
    return (float(np.nanmin(allv)), float(np.nanmax(allv)))

def _cl_solid(df: pd.DataFrame, label: int, comp_id: int, color: str) -> dict:
    lw = 7 if label == 6 else 4
    xs, ys, zs = ([], [], [])
    for bid in sorted(df['branch_id'].unique()):
        grp = df[df['branch_id'] == bid].sort_values('point_id')
        xs.extend(grp['x'].astype(float).tolist() + [None])
        ys.extend(grp['y'].astype(float).tolist() + [None])
        zs.extend(grp['z'].astype(float).tolist() + [None])
    return {'type': 'scatter3d', 'x': xs, 'y': ys, 'z': zs, 'mode': 'lines', 'name': f'[{LABEL_PALETTE[label][1]}] comp{comp_id} CL', 'line': {'color': color, 'width': lw}, 'legendgroup': f'cl_{label}_{comp_id}', 'showlegend': True, 'hoverinfo': 'skip'}

def _cl_diam_colored(df: pd.DataFrame, label: int, comp_id: int, d_clip: tuple[float, float], lw: int, show_cb: bool, bi: int, bid: int) -> dict:
    grp = df[df['branch_id'] == bid].sort_values('point_id')
    dmin, dmax = d_clip
    if dmax <= dmin:
        dmax = dmin + 1e-06
    return {'type': 'scatter3d', 'x': grp['x'].astype(float).tolist(), 'y': grp['y'].astype(float).tolist(), 'z': grp['z'].astype(float).tolist(), 'mode': 'lines', 'name': f'[{LABEL_PALETTE[label][1]}] c{comp_id} br{bid}', 'line': {'width': lw, 'color': grp['diameter_mm'].astype(float).tolist(), 'colorscale': 'Plasma', 'cmin': dmin, 'cmax': dmax, 'showscale': show_cb and bi == 0, 'colorbar': {'title': {'text': 'Ø (mm)', 'side': 'right', 'font': {'size': 13}}, 'len': 0.75, 'thickness': 22, 'tickfont': {'size': 11}, 'bgcolor': 'rgba(8,11,18,0.92)', 'bordercolor': 'rgba(147,167,255,0.35)', 'borderwidth': 1}}, 'legendgroup': 'cld', 'showlegend': bi == 0, 'hoverinfo': 'skip'}

def _cl_hu_colored(df: pd.DataFrame, label: int, comp_id: int, h_clip: tuple[float, float], lw: int, show_cb: bool, bi: int, bid: int) -> dict:
    grp = df[df['branch_id'] == bid].sort_values('point_id')
    hmin, hmax = h_clip
    if hmax <= hmin:
        hmax = hmin + 0.001
    return {'type': 'scatter3d', 'x': grp['x'].astype(float).tolist(), 'y': grp['y'].astype(float).tolist(), 'z': grp['z'].astype(float).tolist(), 'mode': 'lines', 'name': f'[{LABEL_PALETTE[label][1]}] c{comp_id} br{bid}', 'line': {'width': max(lw - 1, 2), 'color': grp['hu'].astype(float).tolist(), 'colorscale': 'RdYlBu', 'cmin': hmin, 'cmax': hmax, 'showscale': show_cb and bi == 0, 'colorbar': {'title': {'text': 'HU', 'side': 'right', 'font': {'size': 13}}, 'len': 0.75, 'thickness': 22, 'tickfont': {'size': 11}, 'bgcolor': 'rgba(8,11,18,0.92)', 'bordercolor': 'rgba(147,167,255,0.35)', 'borderwidth': 1}}, 'legendgroup': 'clh', 'showlegend': bi == 0, 'hoverinfo': 'skip'}

def build_colormap_scenes(resultados: Path, seg_path: str, cta_file: str | None, max_voxels: int=55000, with_centerlines: bool=False, marker_size: float | None=None) -> tuple[list[dict], list[dict], dict]:
    nii = nib.load(seg_path)
    seg = np.round(nii.get_fdata()).astype(np.int16)
    ulab = np.unique(seg)
    d_lo, d_hi = collect_diameter_bounds_mm(resultados)
    d_clip = (d_lo, d_hi) if d_lo is not None and d_hi is not None and np.isfinite(d_lo) and np.isfinite(d_hi) else None
    h_lo, h_hi = collect_hu_bounds(resultados)
    h_clip = (h_lo, h_hi) if h_lo is not None and h_hi is not None and np.isfinite(h_lo) and np.isfinite(h_hi) else None
    cta_ok = False
    cta_mode: str | None = None
    if cta_file is not None and Path(cta_file).is_file():
        cta_ok, cta_mode = cta_usable_for_seg(seg_path, cta_file)
    hu_skip_reason: str | None = None
    if not cta_file or not Path(cta_file).is_file():
        hu_skip_reason = 'no_cta_file'
    elif not cta_ok:
        hu_skip_reason = cta_mode or 'cta_invalid'
    traces_d: list[dict] = []
    traces_h: list[dict] = []
    first_voxel_d_cb = True
    first_voxel_h_cb = True
    show_d_cl_cb = d_clip is not None
    show_h_cl_cb = h_clip is not None and cta_ok
    for label, (color, _desc) in LABEL_PALETTE.items():
        if label not in ulab:
            continue
        ld = resultados / f'label_{label:03d}'
        vd = build_segmentation_voxel_cloud_diameter(seg_path, label, max_voxels=max_voxels, d_clip=d_clip, showscale=first_voxel_d_cb, marker_size=marker_size)
        if vd is not None:
            vd['name'] = f'[{LABEL_PALETTE[label][1]}] voxeles Ø'
            first_voxel_d_cb = False
            traces_d.append(vd)
        if cta_ok:
            vh = build_segmentation_voxel_cloud_hu(seg_path, label, str(cta_file), max_voxels=max_voxels, hu_clip=h_clip, showscale=first_voxel_h_cb, marker_size=marker_size)
            if vh is not None:
                vh['name'] = f'[{LABEL_PALETTE[label][1]}] voxeles HU'
                first_voxel_h_cb = False
                traces_h.append(vh)
        if not with_centerlines or not ld.is_dir():
            continue
        lw = 7 if label == 6 else 4
        for csv_p in sorted(ld.glob('comp_*_centerlines.csv')):
            stem = csv_p.name.replace('_centerlines.csv', '')
            comp_id = int(stem.split('_')[1])
            df = pd.read_csv(csv_p)
            if df.empty:
                continue
            has_d = d_clip is not None and 'diameter_mm' in df.columns and df['diameter_mm'].notna().any()
            if has_d:
                bids = sorted(df['branch_id'].unique())
                for bi, bid in enumerate(bids):
                    use_cb = show_d_cl_cb
                    traces_d.append(_cl_diam_colored(df, label, comp_id, d_clip, lw, use_cb, bi, bid))
                if show_d_cl_cb:
                    show_d_cl_cb = False
            else:
                traces_d.append(_cl_solid(df, label, comp_id, color))
            if cta_ok:
                has_h = h_clip is not None and 'hu' in df.columns and df['hu'].notna().any()
                if has_h:
                    bids = sorted(df['branch_id'].unique())
                    for bi, bid in enumerate(bids):
                        use_cb = show_h_cl_cb
                        traces_h.append(_cl_hu_colored(df, label, comp_id, h_clip, lw, use_cb, bi, bid))
                    if show_h_cl_cb:
                        show_h_cl_cb = False
                else:
                    traces_h.append(_cl_solid(df, label, comp_id, color))
    if cta_ok and len(traces_h) > 0:
        hu_skip_reason = None
    elif cta_ok and len(traces_h) == 0:
        hu_skip_reason = 'no_labels_hu'
    meta = {'has_hu_scene': cta_ok and len(traces_h) > 0, 'diameter_csv': d_clip is not None, 'hu_csv': h_clip is not None, 'cta_path': cta_file if cta_ok else None, 'cta_mode': cta_mode, 'cta_file_arg': str(cta_file) if cta_file and Path(cta_file).is_file() else None, 'hu_skip_reason': hu_skip_reason, 'voxel_mode': True, 'with_centerlines': with_centerlines}
    cta_str = str(cta_file) if cta_file and Path(cta_file).is_file() else None
    mip_2d = build_mip_2d_payload(seg_path, cta_str, cta_ok, d_clip, h_clip)
    return (traces_d, traces_h, meta, mip_2d)

def _hu_empty_message(meta: dict) -> str:
    r = meta.get('hu_skip_reason') or ''
    if r == 'no_cta_file':
        return "No s'ha trobat cap CT/CTA. La primera cerca és la carpeta <code>images</code> dins del projecte (p. ex. <code>...\\\\PhD\\\\images</code>): mateix nom que la segmentació, o <code>_ct</code> / <code>_0000</code>, etc. Si només hi ha un .nii.gz allà (que no sigui el seg), s'usa com a CT. També: variable <code>PHD_CTA_DIR</code> o <code>--cta</code>."
    if r == 'resample_fallo':
        return 'El volum CTA no comparteix graella amb la segmentació i el remostreig automàtic ha fallat. Comprova que sigui el CT del mateix cas.'
    if r == 'no_leer_nifti':
        return "No s'ha pogut llegir el CTA (ruta o format)."
    if r == 'no_labels_hu':
        return 'La segmentació no conté les etiquetes 6, 9 o 10.'
    if r == 'cta_invalid':
        return 'El CTA no és usable amb aquesta segmentació.'
    return 'Panell HU buit.'

def _scene_layout() -> dict:
    return {'bgcolor': '#06080d', 'xaxis': {'backgroundcolor': '#080b10', 'gridcolor': '#1e2436', 'showbackground': True, 'color': '#8b92a8', 'title': {'text': 'X (mm)', 'font': {'size': 11}}}, 'yaxis': {'backgroundcolor': '#080b10', 'gridcolor': '#1e2436', 'showbackground': True, 'color': '#8b92a8', 'title': {'text': 'Y (mm)', 'font': {'size': 11}}}, 'zaxis': {'backgroundcolor': '#080b10', 'gridcolor': '#1e2436', 'showbackground': True, 'color': '#8b92a8', 'title': {'text': 'Z (mm)', 'font': {'size': 11}}}, 'aspectmode': 'data', 'camera': {'eye': {'x': 1.02, 'y': 0.95, 'z': 0.82}}}

def write_colormap_html(traces_diam: list[dict], traces_hu: list[dict], meta: dict, out_path: Path, case_title: str, mip_2d: dict | None=None) -> None:
    mip_2d = mip_2d or {}
    pack = {'tracesDiam': traces_diam, 'tracesHu': traces_hu, 'hasHu': bool(meta.get('has_hu_scene')), 'huEmptyMsg': _hu_empty_message(meta), 'mip2d': mip_2d}
    data_json = json.dumps(pack)
    layout_d = {'paper_bgcolor': 'rgba(0,0,0,0)', 'plot_bgcolor': 'rgba(0,0,0,0)', 'margin': {'l': 0, 'r': 0, 't': 8, 'b': 0}, 'scene': _scene_layout(), 'showlegend': True, 'legend': {'bgcolor': 'rgba(8,11,18,0.88)', 'bordercolor': 'rgba(147,167,255,0.25)', 'borderwidth': 1, 'font': {'color': '#c8d0dc', 'size': 11}, 'x': 0.02, 'y': 0.98, 'xanchor': 'left', 'yanchor': 'top'}}
    layout_h = {**layout_d}
    html_page = f"""<!DOCTYPE html>\n<html lang="ca">\n<head>\n<meta charset="UTF-8">\n<meta name="viewport" content="width=device-width, initial-scale=1">\n<title>Mapes de color Ø / HU — {html_mod.escape(case_title)}</title>\n<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>\n<style>\n* {{ box-sizing: border-box; }}\nbody {{\n  margin: 0;\n  min-height: 100vh;\n  font-family: "Segoe UI", system-ui, sans-serif;\n  background: linear-gradient(165deg, #070a10 0%, #0f1422 45%, #0a0d14 100%);\n  color: #e4e9f2;\n}}\n.hdr {{\n  padding: 20px 28px 16px;\n  border-bottom: 1px solid rgba(120, 140, 255, 0.22);\n  background: rgba(12, 16, 28, 0.6);\n  backdrop-filter: blur(8px);\n}}\n.hdr h1 {{\n  margin: 0 0 8px 0;\n  font-size: 1.35rem;\n  font-weight: 600;\n  color: #a8b8ff;\n  letter-spacing: 0.02em;\n}}\n.hdr p {{\n  margin: 0;\n  font-size: 0.82rem;\n  color: #7d8aad;\n  line-height: 1.45;\n  max-width: 900px;\n}}\n.grid {{\n  display: grid;\n  grid-template-columns: 1fr 1fr;\n  gap: 18px;\n  padding: 18px 22px 28px;\n  min-height: calc(100vh - 120px);\n  align-items: stretch;\n}}\n.grid.single {{ grid-template-columns: 1fr; max-width: 1400px; margin: 0 auto; }}\n.panel {{\n  background: linear-gradient(180deg, rgba(20,26,42,0.95) 0%, rgba(14,18,30,0.98) 100%);\n  border-radius: 16px;\n  border: 1px solid rgba(100, 120, 220, 0.18);\n  box-shadow: 0 12px 48px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.04);\n  display: flex;\n  flex-direction: column;\n  overflow: hidden;\n  min-height: 480px;\n}}\n.panel h2 {{\n  margin: 0;\n  padding: 16px 20px 12px;\n  font-size: 1.05rem;\n  font-weight: 600;\n  color: #93a7ff;\n  border-bottom: 1px solid rgba(255,255,255,0.06);\n}}\n.panel .plot {{\n  flex: 1;\n  min-height: 440px;\n  width: 100%;\n}}\n.panel.empty .plot {{\n  display: flex;\n  align-items: center;\n  justify-content: center;\n  color: #5c6a8a;\n  font-size: 0.95rem;\n  padding: 32px;\n  text-align: center;\n  line-height: 1.5;\n}}\n.section-label {{\n  margin: 14px 22px 6px;\n  font-size: 0.8rem;\n  color: #6a7390;\n  letter-spacing: 0.04em;\n  text-transform: uppercase;\n}}\n@media (max-width: 1100px) {{\n  .grid {{ grid-template-columns: 1fr; }}\n}}\n</style>\n</head>\n<body>\n<header class="hdr">\n  <h1>Mapes de color — {html_mod.escape(case_title)}</h1>\n</header>\n<p class="section-label">3D — volum segmentat</p>\n<div class="grid" id="mainGrid">\n  <section class="panel">\n    <h2>Diàmetre local — volum segmentat</h2>\n    <div id="plotDiam" class="plot"></div>\n  </section>\n  <section class="panel" id="panelHu">\n    <h2>HU (CTA) — volum segmentat</h2>\n    <div id="plotHu" class="plot"></div>\n  </section>\n</div>\n<p class="section-label">2D — MIP</p>\n<div class="grid" id="grid2d">\n  <section class="panel">\n    <h2>Ø — projecció màxima (MIP)</h2>\n    <div id="plot2dDiam" class="plot"></div>\n  </section>\n  <section class="panel" id="panel2dHu">\n    <h2>HU — projecció màxima (MIP)</h2>\n    <div id="plot2dHu" class="plot"></div>\n  </section>\n</div>\n<script>\nconst PACK = {data_json};\nconst layoutDiam = {json.dumps(layout_d)};\nconst layoutHu = {json.dumps(layout_h)};\nfunction go() {{\n  const grid = document.getElementById('mainGrid');\n  const huPanel = document.getElementById('panelHu');\n  if (PACK.tracesDiam && PACK.tracesDiam.length) {{\n    Plotly.newPlot('plotDiam', PACK.tracesDiam, layoutDiam, {{responsive:true, displaylogo:false}});\n  }} else {{\n    document.getElementById('plotDiam').innerHTML = 'Sense traces de diàmetre.';\n  }}\n  if (PACK.hasHu && PACK.tracesHu && PACK.tracesHu.length) {{\n    Plotly.newPlot('plotHu', PACK.tracesHu, layoutHu, {{responsive:true, displaylogo:false}});\n  }} else {{\n    huPanel.classList.add('empty');\n    document.getElementById('plotHu').innerHTML = PACK.huEmptyMsg || 'Panell HU buit.';\n    grid.classList.add('single');\n  }}\n  const M = PACK.mip2d || {{}};\n  if (M.traceDiam && M.layoutDiam2d) {{\n    Plotly.newPlot('plot2dDiam', [M.traceDiam], M.layoutDiam2d, {{responsive:true, displaylogo:false}});\n  }} else {{\n    document.getElementById('plot2dDiam').innerHTML = 'Sense MIP de diàmetre.';\n  }}\n  if (M.traceHu && M.layoutHu2d) {{\n    Plotly.newPlot('plot2dHu', [M.traceHu], M.layoutHu2d, {{responsive:true, displaylogo:false}});\n  }} else {{\n    document.getElementById('panel2dHu').classList.add('empty');\n    document.getElementById('plot2dHu').innerHTML = PACK.huEmptyMsg || 'Sense MIP de HU (falta CTA o dades).';\n  }}\n}}\nwindow.addEventListener('resize', () => {{\n  try {{ Plotly.Plots.resize('plotDiam'); }} catch(e) {{}}\n  try {{ Plotly.Plots.resize('plotHu'); }} catch(e) {{}}\n  try {{ Plotly.Plots.resize('plot2dDiam'); }} catch(e) {{}}\n  try {{ Plotly.Plots.resize('plot2dHu'); }} catch(e) {{}}\n}});\ngo();\n</script>\n</body>\n</html>"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_page, encoding='utf-8')

def is_case_result_dir(d: Path) -> bool:
    if not d.is_dir():
        return False
    return bool(list(d.glob('label_*/comp_*_centerlines.csv')))

def parse_args():
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description='Visor HTML dedicado: colormaps Ø y HU')
    p.add_argument('--resultados-root', type=Path, default=root / 'centerlines_unified_v1', help='Carpeta con un subdirectorio por caso (batch).')
    p.add_argument('--seg-dir', type=Path, default=root / 'segments_labelled_v1', help='Carpeta de segmentaciones por caso (por defecto la original v1; usa *_unified si hace falta).')
    p.add_argument('--cases', nargs='*', default=None, help='Stems de casos (ej. CTA_final_hybrid_002). Por defecto todos los que tengan centerlines.')
    p.add_argument('--resultados', type=Path, default=None, help='Si se indica, solo ese caso (junto con --seg).')
    p.add_argument('--seg', type=str, default=None, help='NIfTI unificado (modo un caso).')
    p.add_argument('--output', type=Path, default=None, help='Salida HTML (solo modo un caso).')
    p.add_argument('--cta', type=Path, default=None, help='CTA explícito (si no, se busca en images/, cta/, etc.).')
    p.add_argument('--max-voxels', type=int, default=55000, help='Máximo de vóxeles por label (menor = HTML más ligero; más = más detalle).')
    p.add_argument('--centerlines', action='store_true', help='Superponer centerlines (por defecto solo volumen sólido).')
    p.add_argument('--marker-size', type=float, default=None, metavar='PX', help='Tamaño fijo de marcador 3D en píxeles (Plotly). Sin indicar, valor moderado según espaciado.')
    return p.parse_args()

def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent
    if args.resultados is not None:
        if not args.seg:
            print('Modo un caso: necesitas --seg')
            return 1
        seg = Path(args.seg)
        if not seg.is_file():
            print(f'No existe segmentación: {seg}')
            return 1
        cta = resolve_cta_path(str(seg), args.cta)
        if cta:
            print(f'CTA (HU): {cta}')
        out = args.output or args.resultados / 'viewer_colormaps_L6_L9_L10.html'
        td, th, meta, mip_2d = build_colormap_scenes(args.resultados, str(seg), cta, max_voxels=args.max_voxels, with_centerlines=args.centerlines, marker_size=args.marker_size)
        write_colormap_html(td, th, meta, out, args.resultados.name, mip_2d)
        print(f'Listo: {out}')
        if not meta.get('has_hu_scene'):
            r = meta.get('hu_skip_reason')
            if r == 'no_cta_file':
                print('Aviso: panel HU vacío — no se encontró CTA. Coloca el CT en ...\\PhD\\images\\ (o usa --cta / PHD_CTA_DIR).')
            elif r:
                print(f'Aviso: panel HU vacío ({r}).')
        return 0
    res_root = args.resultados_root
    if not res_root.is_dir():
        print(f'No existe --resultados-root: {res_root}')
        return 1
    seg_dir = args.seg_dir
    if not seg_dir.is_dir():
        print(f'No existe --seg-dir: {seg_dir}')
        return 1
    if args.cases:
        stems = args.cases
    else:
        stems = sorted((d.name for d in res_root.iterdir() if is_case_result_dir(d)))
    if not stems:
        print('No hay casos con label_*/comp_*_centerlines.csv')
        return 1
    ok, skip = (0, 0)
    for stem in stems:
        case_dir = res_root / stem
        seg = seg_dir / f'{stem}.nii.gz'
        if not case_dir.is_dir():
            print(f'[skip] no carpeta resultados: {case_dir}')
            skip += 1
            continue
        if not seg.is_file():
            print(f'[skip] no hay segmentación: {seg}')
            skip += 1
            continue
        cta = resolve_cta_path(str(seg), args.cta)
        if cta:
            print(f'  [{stem}] CTA: {cta}')
        out = case_dir / 'viewer_colormaps_L6_L9_L10.html'
        try:
            td, th, meta, mip_2d = build_colormap_scenes(case_dir, str(seg), cta, max_voxels=args.max_voxels, with_centerlines=args.centerlines, marker_size=args.marker_size)
            write_colormap_html(td, th, meta, out, stem, mip_2d)
            print(f'OK {stem} -> {out.name}')
            if not meta.get('has_hu_scene') and meta.get('hu_skip_reason') == 'no_cta_file':
                print(f'  [aviso {stem}] sin CTA para HU — usa --cta o images|cta/')
            ok += 1
        except Exception as ex:
            print(f'[error] {stem}: {ex}')
            skip += 1
    print(f'\nHecho: {ok} HTML generados, {skip} omitidos o error.')
    return 0 if ok else 1
if __name__ == '__main__':
    raise SystemExit(main())
