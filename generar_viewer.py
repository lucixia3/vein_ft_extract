from __future__ import annotations
import argparse
import json
import numpy as np
import pandas as pd
import nibabel as nib
from nibabel.processing import resample_from_to
from pathlib import Path
from graph_utils import filter_nodes_incident_to_edges
COLORS = ['#4F8EF7', '#F76C6C', '#43D9A2', '#F7B731', '#A55EEA', '#FC5C65', '#26de81', '#fd9644', '#45aaf2', '#2bcbba', '#fed330', '#eb3b5a', '#20bf6b', '#3867d6', '#fd79a8']

def build_mesh_trace(seg_path: str, label: int, comp_mask: np.ndarray=None, affine: np.ndarray=None, max_faces: int=8000):
    from skimage.measure import marching_cubes
    if comp_mask is not None and affine is not None:
        mask = comp_mask.astype(np.float32)
        aff = affine
    else:
        nii = nib.load(seg_path)
        data = np.round(nii.get_fdata()).astype(np.int16)
        mask = (data == label).astype(np.float32)
        aff = nii.affine
    if mask.sum() == 0:
        return None
    try:
        verts, faces, _, _ = marching_cubes(mask, level=0.5)
    except Exception as e:
        print(f'  Marching cubes fallo: {e}')
        return None
    ones = np.ones((len(verts), 1))
    verts_kji = np.hstack([verts[:, 2:3], verts[:, 1:2], verts[:, 0:1], ones])
    verts_world = (aff @ verts_kji.T).T[:, :3]
    if len(faces) > max_faces:
        idx = np.random.choice(len(faces), max_faces, replace=False)
        faces = faces[idx]
    x = verts_world[:, 0].tolist()
    y = verts_world[:, 1].tolist()
    z = verts_world[:, 2].tolist()
    i = faces[:, 0].tolist()
    j = faces[:, 1].tolist()
    k = faces[:, 2].tolist()
    return {'type': 'mesh3d', 'x': x, 'y': y, 'z': z, 'i': i, 'j': j, 'k': k, 'color': '#7ec8a0', 'opacity': 0.18, 'name': 'Segmento', 'hoverinfo': 'skip', 'showlegend': True, 'flatshading': False, 'lighting': {'ambient': 0.7, 'diffuse': 0.8, 'specular': 0.2, 'roughness': 0.5}, 'lightposition': {'x': 1000, 'y': 1000, 'z': 1000}}

def _mesh3d_flat_face_intensity(verts_world: np.ndarray, faces: np.ndarray, vert_values: np.ndarray) -> tuple[list[float], list[float], list[float], list[int], list[int], list[int], list[float]]:
    v = np.asarray(vert_values, dtype=float)
    xs, ys, zs, ints = ([], [], [], [])
    ii, jj, kk = ([], [], [])
    nv = 0
    for f in faces:
        i0, i1, i2 = (int(f[0]), int(f[1]), int(f[2]))
        val = float((v[i0] + v[i1] + v[i2]) / 3.0)
        for idx in (i0, i1, i2):
            xs.append(float(verts_world[idx, 0]))
            ys.append(float(verts_world[idx, 1]))
            zs.append(float(verts_world[idx, 2]))
            ints.append(val)
        ii.append(nv)
        jj.append(nv + 1)
        kk.append(nv + 2)
        nv += 3
    return (xs, ys, zs, ii, jj, kk, ints)

def _subsample_mask_ijk_strided(mask: np.ndarray, max_voxels: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ii, jj, kk = np.where(mask)
    n = int(ii.size)
    if n == 0 or n <= max_voxels:
        return (ii, jj, kk)
    imin, imax = (int(ii.min()), int(ii.max()))
    jmin, jmax = (int(jj.min()), int(jj.max()))
    kmin, kmax = (int(kk.min()), int(kk.max()))

    def count_step(step: int) -> int:
        s = max(1, step)
        slab = mask[imin:imax + 1:s, jmin:jmax + 1:s, kmin:kmax + 1:s]
        return int(slab.sum())
    step = 1
    while count_step(step) > max_voxels:
        step += 1
        if step > 4000:
            break
    s = max(1, step)
    slab = mask[imin:imax + 1:s, jmin:jmax + 1:s, kmin:kmax + 1:s]
    ri, rj, rk = np.where(slab)
    ri = imin + ri * s
    rj = jmin + rj * s
    rk = kmin + rk * s
    if int(ri.size) > max_voxels:
        rng = np.random.default_rng(0)
        pick = rng.choice(int(ri.size), size=max_voxels, replace=False)
        ri, rj, rk = (ri[pick], rj[pick], rk[pick])
    return (ri, rj, rk)

def _marker_px_for_spacing_mm(affine: np.ndarray) -> float:
    sp = float(np.mean(np.abs(np.diag(affine)[:3])))
    return float(np.clip(2.0 + 2.0 * sp, 2.0, 4.0))

def _ijk_numpy_to_world_mm(ijk: np.ndarray, affine: np.ndarray) -> np.ndarray:
    ijk = np.asarray(ijk, dtype=float)
    if len(ijk) == 0:
        return np.empty((0, 3))
    ones = np.ones((len(ijk), 1))
    hom = np.hstack([ijk[:, [2, 1, 0]], ones])
    return (affine @ hom.T).T[:, :3]

def _scatter3d_compact_lists(xyz: np.ndarray, vals: np.ndarray, *, xyz_decimals: int=2, val_decimals: int=3) -> tuple[list, list, list, list]:
    xyz = np.asarray(xyz, dtype=np.float64)
    vals = np.asarray(vals, dtype=np.float64)
    xyz = np.round(xyz, xyz_decimals)
    vals = np.round(vals, val_decimals)
    return (xyz[:, 0].tolist(), xyz[:, 1].tolist(), xyz[:, 2].tolist(), vals.tolist())

def build_segmentation_voxel_cloud_diameter(seg_path: str, label: int, *, max_voxels: int=55000, d_clip: tuple[float, float] | None=None, showscale: bool=True, marker_size: float | None=None, contrast_percentiles: tuple[float, float] | None=(1.0, 99.0)):
    from scipy import ndimage as ndi
    nii = nib.load(seg_path)
    aff = nii.affine
    data = np.round(nii.get_fdata()).astype(np.int16)
    mask = data == label
    if not np.any(mask):
        return None
    spacing = tuple((float(x) for x in np.abs(np.diag(aff)[:3])))
    edt = ndi.distance_transform_edt(mask.astype(bool), sampling=spacing)
    ii, jj, kk = _subsample_mask_ijk_strided(mask, max_voxels)
    vals = (2.0 * edt[ii, jj, kk]).astype(float)
    ijk = np.column_stack([ii, jj, kk])
    xyz = _ijk_numpy_to_world_mm(ijk, aff)
    ms = marker_size if marker_size is not None else _marker_px_for_spacing_mm(aff)
    if d_clip is not None:
        cmin, cmax = (float(d_clip[0]), float(d_clip[1]))
    elif contrast_percentiles is not None and vals.size > 50:
        lo, hi = contrast_percentiles
        cmin = float(np.percentile(vals, lo))
        cmax = float(np.percentile(vals, hi))
    else:
        cmin = float(np.nanmin(vals))
        cmax = float(np.nanmax(vals))
    if cmax <= cmin:
        cmax = cmin + 1e-06
    cb = {'title': {'text': 'Ø en voxeles (mm)', 'font': {'size': 15}}, 'len': 0.78, 'thickness': 28, 'tickfont': {'size': 13}, 'bgcolor': 'rgba(8,11,18,0.92)', 'bordercolor': 'rgba(147,167,255,0.35)', 'borderwidth': 1}
    xs, ys, zs, cs = _scatter3d_compact_lists(xyz, vals, val_decimals=3)
    return {'type': 'scatter3d', 'x': xs, 'y': ys, 'z': zs, 'mode': 'markers', 'marker': {'size': ms, 'color': cs, 'colorscale': 'Turbo', 'cmin': cmin, 'cmax': cmax, 'opacity': 0.82, 'line': {'width': 0}, 'showscale': showscale, 'colorbar': cb if showscale else None}, 'name': f'label {label} seg (Ø)', 'hoverinfo': 'skip'}

def build_segmentation_voxel_cloud_hu(seg_path: str, label: int, cta_path: str, *, max_voxels: int=55000, hu_clip: tuple[float, float] | None=None, showscale: bool=True, marker_size: float | None=None, contrast_percentiles: tuple[float, float] | None=(0.5, 99.5)):
    snii = nib.load(seg_path)
    cnii = nib.load(cta_path)
    aff = snii.affine
    data = np.round(snii.get_fdata()).astype(np.int16)
    mask = data == label
    if not np.any(mask):
        return None
    same = tuple(snii.shape[:3]) == tuple(cnii.shape[:3]) and np.allclose(snii.affine, cnii.affine, rtol=1e-05, atol=0.0001)
    if same:
        cta = np.ascontiguousarray(cnii.get_fdata().astype(np.float32))
    else:
        try:
            cta = np.ascontiguousarray(resample_from_to(cnii, snii, order=1, mode='constant', cval=-1024.0).get_fdata().astype(np.float32))
        except Exception:
            return None
    ii, jj, kk = _subsample_mask_ijk_strided(mask, max_voxels)
    vals = cta[ii, jj, kk].astype(float)
    ijk = np.column_stack([ii, jj, kk])
    xyz = _ijk_numpy_to_world_mm(ijk, aff)
    if hu_clip is not None:
        hmin, hmax = (float(hu_clip[0]), float(hu_clip[1]))
    elif contrast_percentiles is not None and vals.size > 50:
        lo, hi = contrast_percentiles
        hmin = float(np.percentile(vals, lo))
        hmax = float(np.percentile(vals, hi))
    else:
        hmin = float(np.nanmin(vals))
        hmax = float(np.nanmax(vals))
    if not np.isfinite(hmin) or not np.isfinite(hmax) or hmax <= hmin:
        hmax = hmin + 0.001
    ms = marker_size if marker_size is not None else _marker_px_for_spacing_mm(aff)
    cb = {'title': {'text': 'HU (voxeles)', 'font': {'size': 15}}, 'len': 0.78, 'thickness': 28, 'tickfont': {'size': 13}, 'bgcolor': 'rgba(8,11,18,0.92)', 'bordercolor': 'rgba(147,167,255,0.35)', 'borderwidth': 1}
    xs, ys, zs, cs = _scatter3d_compact_lists(xyz, vals, val_decimals=2)
    return {'type': 'scatter3d', 'x': xs, 'y': ys, 'z': zs, 'mode': 'markers', 'marker': {'size': ms, 'color': cs, 'colorscale': 'RdYlBu_r', 'cmin': hmin, 'cmax': hmax, 'opacity': 0.82, 'line': {'width': 0}, 'showscale': showscale, 'colorbar': cb if showscale else None}, 'name': f'label {label} seg (HU)', 'hoverinfo': 'skip'}

def build_mesh_trace_local_diameter_colormap(seg_path: str, label: int, comp_mask: np.ndarray=None, affine: np.ndarray=None, max_faces: int=8000):
    from scipy.ndimage import map_coordinates
    from skimage.measure import marching_cubes
    if comp_mask is not None and affine is not None:
        mask = comp_mask.astype(np.float32)
        aff = affine
    else:
        nii = nib.load(seg_path)
        data = np.round(nii.get_fdata()).astype(np.int16)
        mask = (data == label).astype(np.float32)
        aff = nii.affine
    if mask.sum() == 0:
        return None
    spacing = tuple((float(x) for x in np.abs(np.diag(aff)[:3])))
    from scipy import ndimage as ndi
    edt = ndi.distance_transform_edt(mask.astype(bool), sampling=spacing)
    try:
        verts, faces, _, _ = marching_cubes(mask, level=0.5)
    except Exception:
        return None
    vc = np.stack([verts[:, 0], verts[:, 1], verts[:, 2]], axis=0)
    rad = map_coordinates(edt, vc, order=1, mode='nearest')
    local_diam = (2.0 * rad).astype(float)
    ones = np.ones((len(verts), 1))
    verts_kji = np.hstack([verts[:, 2:3], verts[:, 1:2], verts[:, 0:1], ones])
    verts_world = (aff @ verts_kji.T).T[:, :3]
    if len(faces) > max_faces:
        idx = np.random.choice(len(faces), max_faces, replace=False)
        faces = faces[idx]
    xs, ys, zs, ii, jj, kk, ints = _mesh3d_flat_face_intensity(verts_world, faces, local_diam)
    dmin = float(np.nanmin(ints))
    dmax = float(np.nanmax(ints))
    if not np.isfinite(dmin) or not np.isfinite(dmax) or dmax <= dmin:
        dmax = dmin + 1e-06
    return {'type': 'mesh3d', 'x': xs, 'y': ys, 'z': zs, 'i': ii, 'j': jj, 'k': kk, 'intensity': ints, 'colorscale': 'Turbo', 'cmin': dmin, 'cmax': dmax, 'opacity': 0.98, 'name': 'Malla Ø local', 'hoverinfo': 'skip', 'showlegend': True, 'colorbar': {'title': {'text': 'Ø mm', 'side': 'right'}, 'tickfont': {'size': 10}, 'len': 0.5}, 'showscale': True, 'flatshading': True, 'lighting': {'ambient': 0.92, 'diffuse': 0.35, 'specular': 0.04, 'roughness': 0.85, 'fresnel': 0.08}, 'lightposition': {'x': 200, 'y': 400, 'z': 500}}

def build_mesh_trace_local_hu_colormap(seg_path: str, label: int, cta_path: str, comp_mask: np.ndarray=None, affine: np.ndarray=None, max_faces: int=8000, hu_clip: tuple[float, float] | None=None):
    from scipy.ndimage import map_coordinates
    from skimage.measure import marching_cubes
    snii = nib.load(seg_path)
    cnii = nib.load(cta_path)
    if tuple(snii.shape[:3]) != tuple(cnii.shape[:3]) or not np.allclose(snii.affine, cnii.affine, rtol=1e-05, atol=0.0001):
        return None
    cta_data = cnii.get_fdata().astype(np.float32)
    if comp_mask is not None and affine is not None:
        mask = comp_mask.astype(np.float32)
        aff = affine
    else:
        data = np.round(snii.get_fdata()).astype(np.int16)
        mask = (data == label).astype(np.float32)
        aff = snii.affine
    if mask.sum() == 0:
        return None
    try:
        verts, faces, _, _ = marching_cubes(mask, level=0.5)
    except Exception:
        return None
    vc = np.stack([verts[:, 0], verts[:, 1], verts[:, 2]], axis=0)
    hu_vals = map_coordinates(cta_data, vc, order=1, mode='nearest').astype(float)
    ones = np.ones((len(verts), 1))
    verts_kji = np.hstack([verts[:, 2:3], verts[:, 1:2], verts[:, 0:1], ones])
    verts_world = (aff @ verts_kji.T).T[:, :3]
    if len(faces) > max_faces:
        idx = np.random.choice(len(faces), max_faces, replace=False)
        faces = faces[idx]
    if hu_clip is not None:
        hmin, hmax = (float(hu_clip[0]), float(hu_clip[1]))
    else:
        hmin = float(np.nanmin(hu_vals))
        hmax = float(np.nanmax(hu_vals))
    if not np.isfinite(hmin) or not np.isfinite(hmax) or hmax <= hmin:
        hmax = hmin + 0.001
    xs, ys, zs, ii, jj, kk, ints = _mesh3d_flat_face_intensity(verts_world, faces, hu_vals)
    return {'type': 'mesh3d', 'x': xs, 'y': ys, 'z': zs, 'i': ii, 'j': jj, 'k': kk, 'intensity': ints, 'colorscale': 'RdYlBu_r', 'cmin': hmin, 'cmax': hmax, 'opacity': 0.98, 'name': 'Malla HU', 'hoverinfo': 'skip', 'showlegend': True, 'colorbar': {'title': {'text': 'HU', 'side': 'right'}, 'tickfont': {'size': 10}, 'len': 0.5}, 'showscale': True, 'flatshading': True, 'lighting': {'ambient': 0.92, 'diffuse': 0.35, 'specular': 0.04, 'roughness': 0.85, 'fresnel': 0.08}, 'lightposition': {'x': 200, 'y': 400, 'z': 500}}

def load_component(base_dir: Path, comp_id: int, seg_path: str, label: int, seg_data: np.ndarray, affine: np.ndarray):
    prefix = base_dir / f'comp_{comp_id:03d}'
    csv_path = Path(str(prefix) + '_centerlines.csv')
    stat_path = Path(str(prefix) + '_branch_stats.csv')
    skel_path = Path(str(prefix) + '_skeleton.nii.gz')
    nodes_path = Path(str(prefix) + '_graph_nodes.csv')
    edges_path = Path(str(prefix) + '_graph_edges.csv')
    if not csv_path.exists():
        return None
    df_pts = pd.read_csv(csv_path)
    df_stats = pd.read_csv(stat_path) if stat_path.exists() else pd.DataFrame()
    df_nodes = pd.read_csv(nodes_path) if nodes_path.exists() else pd.DataFrame()
    df_edges = pd.read_csv(edges_path) if edges_path.exists() else pd.DataFrame()
    if not df_nodes.empty and (not df_edges.empty):
        nn = df_nodes.drop(columns=[c for c in ('label', 'component') if c in df_nodes.columns], errors='ignore')
        ee = df_edges.drop(columns=[c for c in ('label', 'component') if c in df_edges.columns], errors='ignore')
        nn, ee = filter_nodes_incident_to_edges(nn, ee)
        nn = nn.copy()
        ee = ee.copy()
        nn.insert(0, 'component', comp_id)
        nn.insert(0, 'label', label)
        ee.insert(0, 'component', comp_id)
        ee.insert(0, 'label', label)
        df_nodes, df_edges = (nn, ee)
    skel_pts = []
    if skel_path.exists():
        nii = nib.load(str(skel_path))
        skel = nii.get_fdata().astype(bool)
        aff = nii.affine
        voxels = np.argwhere(skel)
        if len(voxels) > 3000:
            idx = np.random.choice(len(voxels), 3000, replace=False)
            voxels = voxels[idx]
        ijk1 = np.hstack([voxels[:, [2, 1, 0]], np.ones((len(voxels), 1))])
        skel_pts = (aff @ ijk1.T).T[:, :3].tolist()
    from scipy import ndimage
    mask_label = (seg_data == label).astype(np.uint8)
    labeled, _ = ndimage.label(mask_label)
    mesh_trace = None
    if comp_id <= labeled.max() and (labeled == comp_id).any():
        comp_mask = (labeled == comp_id).astype(np.uint8)
        mesh_trace = build_mesh_trace(seg_path, label, comp_mask=comp_mask, affine=affine)
    else:
        print(f'  Componente {comp_id} no encontrado en labeled (max={labeled.max()})')
    return {'df_pts': df_pts, 'df_stats': df_stats, 'df_nodes': df_nodes, 'df_edges': df_edges, 'skel_pts': skel_pts, 'mesh_trace': mesh_trace}

def trace_graph_edges(df_nodes: pd.DataFrame, df_edges: pd.DataFrame, comp_id: int):
    if df_nodes.empty or df_edges.empty or 'node_id' not in df_nodes.columns:
        return None
    need = {'x', 'y', 'z'}
    if not need.issubset(df_nodes.columns):
        return None
    pos = df_nodes.set_index('node_id')[['x', 'y', 'z']]
    xs, ys, zs = ([], [], [])
    for _, row in df_edges.iterrows():
        try:
            a, b = (int(row['node_start']), int(row['node_end']))
        except (KeyError, ValueError):
            continue
        if a not in pos.index or b not in pos.index:
            continue
        p0, p1 = (pos.loc[a], pos.loc[b])
        xs += [float(p0['x']), float(p1['x']), None]
        ys += [float(p0['y']), float(p1['y']), None]
        zs += [float(p0['z']), float(p1['z']), None]
    if not xs:
        return None
    return {'type': 'scatter3d', 'x': xs, 'y': ys, 'z': zs, 'mode': 'lines', 'name': 'Aristas grafo', 'line': {'color': 'rgba(255,200,120,0.75)', 'width': 3}, 'hoverinfo': 'skip', 'showlegend': True}

def trace_bridges_for_comp(bridges_df: pd.DataFrame, comp_id: int, label: int):
    if bridges_df is None or bridges_df.empty:
        return None
    if 'comp_a' not in bridges_df.columns:
        return None
    sub = bridges_df[(bridges_df['comp_a'] == comp_id) | (bridges_df['comp_b'] == comp_id)]
    if sub.empty:
        return None
    xs, ys, zs = ([], [], [])
    for _, r in sub.iterrows():
        xs += [float(r['xa']), float(r['xb']), None]
        ys += [float(r['ya']), float(r['yb']), None]
        zs += [float(r['za']), float(r['zb']), None]
    return {'type': 'scatter3d', 'x': xs, 'y': ys, 'z': zs, 'mode': 'lines', 'name': 'Puentes inter-comp', 'line': {'color': 'rgba(200,140,255,0.95)', 'width': 5, 'dash': 'dash'}, 'hoverinfo': 'skip', 'showlegend': True}

def trace_graph_nodes(df_nodes: pd.DataFrame, comp_id: int):
    if df_nodes.empty or 'node_id' not in df_nodes.columns:
        return None
    if not {'x', 'y', 'z'}.issubset(df_nodes.columns):
        return None
    colors, sizes, texts = ([], [], [])
    for _, r in df_nodes.iterrows():
        role = str(r.get('role', '')).lower()
        nid = int(r['node_id'])
        deg = int(r['degree']) if 'degree' in r and pd.notna(r['degree']) else 0
        if role == 'junction' or deg >= 3:
            colors.append('#ffd93d')
            sizes.append(11)
            lab = 'Bifurcación'
        else:
            colors.append('#ff6b6b')
            sizes.append(9)
            lab = 'Extremo'
        texts.append(f'Nodo {nid} · {lab}<br>grado {deg}<br>Comp {comp_id}')
    return {'type': 'scatter3d', 'x': df_nodes['x'].astype(float).tolist(), 'y': df_nodes['y'].astype(float).tolist(), 'z': df_nodes['z'].astype(float).tolist(), 'mode': 'markers', 'name': 'Nodos', 'marker': {'size': sizes, 'color': colors, 'line': {'width': 1.5, 'color': 'rgba(255,255,255,0.85)'}, 'opacity': 0.95}, 'text': texts, 'hovertemplate': '%{text}<extra></extra>', 'showlegend': True}

def build_component_data(data: dict, comp_id: int, bridges_df: pd.DataFrame | None=None, label: int | None=None):
    df_pts = data['df_pts']
    df_stats = data['df_stats']
    df_nodes = data.get('df_nodes', pd.DataFrame())
    df_edges = data.get('df_edges', pd.DataFrame())
    skel_pts = data['skel_pts']
    mesh_trace = data['mesh_trace']
    lb = int(label) if label is not None else int(df_pts['label'].iloc[0])
    traces_3d = []
    traces_curv = []
    if mesh_trace is not None:
        traces_3d.append(mesh_trace)
    if skel_pts:
        traces_3d.append({'type': 'scatter3d', 'x': [p[0] for p in skel_pts], 'y': [p[1] for p in skel_pts], 'z': [p[2] for p in skel_pts], 'mode': 'markers', 'name': 'Skeleton', 'marker': {'size': 1.5, 'color': 'rgba(200,200,100,0.4)'}, 'hoverinfo': 'skip'})
    et = trace_graph_edges(df_nodes, df_edges, comp_id)
    if et is not None:
        traces_3d.append(et)
    bt = trace_bridges_for_comp(bridges_df, comp_id, lb)
    if bt is not None:
        traces_3d.append(bt)
    branch_ids = sorted(df_pts['branch_id'].unique())
    for i, bid in enumerate(branch_ids):
        grp = df_pts[df_pts['branch_id'] == bid].sort_values('point_id')
        color = COLORS[i % len(COLORS)]
        tooltip = f'Comp {comp_id} · Rama {bid}'
        if not df_stats.empty:
            row = df_stats[df_stats['branch_id'] == bid]
            if not row.empty:
                r = row.iloc[0]
                tooltip = f"Comp {comp_id} · Rama {bid}<br>Long: {r.get('length_mm', 0):.1f} mm<br>Tortuosidad: {r.get('tortuosity', 0):.3f}<br>Curv media: {r.get('curvature_mean', 0):.4f}"
        traces_3d.append({'type': 'scatter3d', 'x': grp['x'].tolist(), 'y': grp['y'].tolist(), 'z': grp['z'].tolist(), 'mode': 'lines+markers', 'name': f'Rama {bid}', 'line': {'color': color, 'width': 6}, 'marker': {'size': 3, 'color': color}, 'text': tooltip, 'hovertemplate': '%{text}<extra></extra>'})
        if 'curvature' in grp.columns:
            traces_curv.append({'type': 'scatter', 'x': list(range(len(grp))), 'y': grp['curvature'].fillna(0).tolist(), 'mode': 'lines', 'name': f'Rama {bid}', 'line': {'color': color, 'width': 2}})
    nt = trace_graph_nodes(df_nodes, comp_id)
    if nt is not None:
        traces_3d.append(nt)
    if not df_stats.empty:
        cols = [c for c in ['branch_id', 'length_mm', 'euclidean_mm', 'tortuosity', 'curvature_mean', 'curvature_max', 'torsion_mean'] if c in df_stats.columns]
        table_html = df_stats[cols].to_html(index=False, classes='statstable', border=0, float_format=lambda x: f'{x:.4f}')
    else:
        table_html = "<p style='color:#6b7280;padding:12px'>Sin stats.</p>"
    nodes_html = ''
    if not df_nodes.empty:
        ncols = [c for c in ['node_id', 'role', 'degree', 'x', 'y', 'z'] if c in df_nodes.columns]
        if ncols:
            nodes_html = "<h4 style='margin:14px 0 8px;font-size:12px;color:#9ca3af'>Nodos del grafo</h4>" + df_nodes[ncols].to_html(index=False, classes='statstable', border=0, float_format=lambda x: f'{x:.4f}')
    if nodes_html:
        table_html = table_html + nodes_html
    total_length = df_stats['length_mm'].sum() if not df_stats.empty and 'length_mm' in df_stats.columns else 0
    n_nodes = len(df_nodes) if not df_nodes.empty else 0
    return {'traces_3d': traces_3d, 'traces_curv': traces_curv, 'table_html': table_html, 'n_branches': len(branch_ids), 'n_pts': len(df_pts), 'n_nodes': n_nodes, 'total_length': round(float(total_length), 1)}

def generate_html(label: int, components: dict) -> str:
    comp_ids = sorted(components.keys())
    js_data = {}
    for cid in comp_ids:
        d = components[cid]
        js_data[cid] = {'traces3d': d['traces_3d'], 'tracesCurv': d['traces_curv'], 'tableHtml': d['table_html'], 'nBranches': d['n_branches'], 'nPts': d['n_pts'], 'nNodes': d.get('n_nodes', 0), 'totalLength': d['total_length']}
    js_data_json = json.dumps(js_data)
    first_comp = comp_ids[0]
    sidebar_buttons = ''
    for cid in comp_ids:
        d = components[cid]
        nn = d.get('n_nodes', 0)
        nodestr = f"{nn} nod{('os' if nn != 1 else 'o')}"
        sidebar_buttons += f'''\n        <button class="comp-btn" data-id="{cid}" onclick="selectComp({cid})">\n          <span class="comp-num">#{cid}</span>\n          <span class="comp-meta">{d['n_branches']} rama{('s' if d['n_branches'] != 1 else '')} &middot; {nodestr} &middot; {d['total_length']} mm</span>\n        </button>'''
    return f"""<!DOCTYPE html>\n<html lang="es">\n<head>\n<meta charset="UTF-8">\n<title>Viewer — Label {label}</title>\n<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>\n<style>\n* {{ box-sizing:border-box; margin:0; padding:0; }}\nbody {{ font-family:system-ui,sans-serif; background:#0f1117; color:#e0e0e0;\n        display:flex; flex-direction:column; height:100vh; overflow:hidden; }}\nheader {{ padding:12px 20px; background:#1a1d27; border-bottom:1px solid #2a2d3e;\n          display:flex; align-items:center; gap:14px; flex-shrink:0; }}\nheader h1 {{ font-size:15px; font-weight:500; color:#a0c4ff; }}\n#comp-info {{ font-size:12px; color:#6b7280; margin-left:auto; }}\n.main {{ display:flex; flex:1; overflow:hidden; }}\n\n.sidebar {{ width:185px; background:#1a1d27; border-right:1px solid #2a2d3e;\n            overflow-y:auto; flex-shrink:0; padding:8px; }}\n.sidebar-title {{ font-size:10px; color:#4b5563; text-transform:uppercase;\n                  letter-spacing:.07em; padding:4px 6px 8px; }}\n.comp-btn {{ width:100%; background:none; border:1px solid transparent; border-radius:7px;\n             padding:8px 10px; cursor:pointer; text-align:left; color:#9ca3af;\n             transition:all .15s; display:flex; flex-direction:column; gap:2px; }}\n.comp-btn:hover {{ background:#22253a; border-color:#2a2d3e; color:#d1d5db; }}\n.comp-btn.active {{ background:#1e2d4a; border-color:#3b6cb7; color:#93c5fd; }}\n.comp-num  {{ font-size:13px; font-weight:500; }}\n.comp-meta {{ font-size:11px; opacity:.7; }}\n\n.content {{ flex:1; display:grid; grid-template-rows:1fr 170px; overflow:hidden; }}\n.top-row {{ display:grid; grid-template-columns:1.7fr 1fr; overflow:hidden; gap:0; }}\n.card {{ background:#1a1d27; border:1px solid #2a2d3e; overflow:hidden;\n         display:flex; flex-direction:column; }}\n.card-title {{ padding:7px 14px; font-size:10px; font-weight:500; color:#6b7280;\n               border-bottom:1px solid #2a2d3e; flex-shrink:0;\n               text-transform:uppercase; letter-spacing:.05em; display:flex; align-items:center; gap:10px; }}\n.toggle-wrap {{ display:flex; gap:8px; margin-left:auto; }}\n.toggle-wrap label {{ font-size:11px; color:#9ca3af; display:flex; align-items:center; gap:4px; cursor:pointer; }}\n.toggle-wrap input {{ cursor:pointer; accent-color:#4F8EF7; }}\n#plot3d   {{ flex:1; min-height:0; }}\n#plotcurv {{ flex:1; min-height:0; }}\n.bottom-row {{ border-top:1px solid #2a2d3e; overflow:auto; padding:8px 12px; }}\n.statstable {{ width:100%; border-collapse:collapse; font-size:12px; }}\n.statstable th {{ background:#22253a; color:#9ca3af; padding:5px 10px; text-align:left;\n                  position:sticky; top:0; font-weight:500; }}\n.statstable td {{ padding:4px 10px; border-bottom:1px solid #1e2130; color:#d1d5db; }}\n.statstable tr:hover td {{ background:#22253a; }}\n</style>\n</head>\n<body>\n<header>\n  <h1>Viewer — Label {label}</h1>\n  <span id="comp-info">\n    Comp <b id="active-comp">—</b> &nbsp;|&nbsp; <span id="active-stats">—</span>\n  </span>\n</header>\n\n<div class="main">\n  <div class="sidebar">\n    <div class="sidebar-title">Componentes ({len(comp_ids)})</div>\n    {sidebar_buttons}\n  </div>\n\n  <div class="content">\n    <div class="top-row">\n      <div class="card">\n        <div class="card-title">\n          Vista 3D\n          <div class="toggle-wrap">\n            <label><input type="checkbox" id="tog-mesh"  checked onchange="toggleTrace('mesh')">  Segmento</label>\n            <label><input type="checkbox" id="tog-skel"  checked onchange="toggleTrace('skel')">  Skeleton</label>\n            <label><input type="checkbox" id="tog-cl"    checked onchange="toggleTrace('cl')">    Centerline</label>\n            <label><input type="checkbox" id="tog-edges" checked onchange="toggleTrace('edges')">  Aristas</label>\n            <label><input type="checkbox" id="tog-bridges" checked onchange="toggleTrace('bridges')">  Puentes</label>\n            <label><input type="checkbox" id="tog-nodes" checked onchange="toggleTrace('nodes')">  Nodos</label>\n          </div>\n        </div>\n        <div id="plot3d"></div>\n      </div>\n      <div class="card">\n        <div class="card-title">Curvatura por rama</div>\n        <div id="plotcurv"></div>\n      </div>\n    </div>\n    <div class="bottom-row" id="stats-table"></div>\n  </div>\n</div>\n\n<script>\nconst DATA = {js_data_json};\n\nconst layout3d = {{\n  paper_bgcolor:"#1a1d27", plot_bgcolor:"#1a1d27",\n  scene:{{\n    bgcolor:"#0f1117",\n    xaxis:{{gridcolor:"#1e2130",color:"#374151",title:"X"}},\n    yaxis:{{gridcolor:"#1e2130",color:"#374151",title:"Y"}},\n    zaxis:{{gridcolor:"#1e2130",color:"#374151",title:"Z"}},\n    aspectmode:"data",\n  }},\n  legend:{{bgcolor:"#1a1d27",font:{{color:"#9ca3af",size:11}},\n           bordercolor:"#2a2d3e",borderwidth:1,x:0,y:1}},\n  margin:{{l:0,r:0,t:0,b:0}},\n}};\n\nconst layoutCurv = {{\n  paper_bgcolor:"#1a1d27", plot_bgcolor:"#1a1d27",\n  xaxis:{{gridcolor:"#2a2d3e",color:"#4b5563",title:"Punto"}},\n  yaxis:{{gridcolor:"#2a2d3e",color:"#4b5563",title:"Curvatura"}},\n  legend:{{bgcolor:"#22253a",font:{{color:"#9ca3af",size:10}}}},\n  margin:{{l:50,r:14,t:10,b:40}},\n}};\n\nlet initialized = false;\nlet currentTraces = [];\n\nfunction selectComp(id) {{\n  document.querySelectorAll('.comp-btn').forEach(b => b.classList.toggle('active', +b.dataset.id===id));\n  const d = DATA[id];\n  document.getElementById('active-comp').textContent = '#'+id;\n  document.getElementById('active-stats').textContent =\n    d.nBranches+' rama'+(d.nBranches!==1?'s':'')+' · '+(d.nNodes||0)+' nodos · '+d.nPts+' pts · '+d.totalLength+' mm';\n  document.getElementById('stats-table').innerHTML = d.tableHtml;\n\n  currentTraces = d.traces3d;\n  applyVisibility();\n\n  if (!initialized) {{\n    Plotly.newPlot('plot3d',   currentTraces, layout3d,   {{responsive:true}});\n    Plotly.newPlot('plotcurv', d.tracesCurv,  layoutCurv, {{responsive:true}});\n    initialized = true;\n  }} else {{\n    Plotly.react('plot3d',   currentTraces, layout3d);\n    Plotly.react('plotcurv', d.tracesCurv,  layoutCurv);\n  }}\n}}\n\nfunction applyVisibility() {{\n  const showMesh = document.getElementById('tog-mesh').checked;\n  const showSkel = document.getElementById('tog-skel').checked;\n  const showCl   = document.getElementById('tog-cl').checked;\n  const showEdges = document.getElementById('tog-edges').checked;\n  const showBridges = document.getElementById('tog-bridges').checked;\n  const showNodes = document.getElementById('tog-nodes').checked;\n  currentTraces.forEach(t => {{\n    if (t.name === 'Segmento') t.visible = showMesh ? true : 'legendonly';\n    else if (t.name === 'Skeleton') t.visible = showSkel ? true : 'legendonly';\n    else if (t.name === 'Aristas grafo') t.visible = showEdges ? true : 'legendonly';\n    else if (t.name === 'Puentes inter-comp') t.visible = showBridges ? true : 'legendonly';\n    else if (t.name === 'Nodos') t.visible = showNodes ? true : 'legendonly';\n    else if (typeof t.name === 'string' && t.name.startsWith('Rama ')) t.visible = showCl ? true : 'legendonly';\n    else t.visible = showCl ? true : 'legendonly';\n  }});\n}}\n\nfunction toggleTrace(which) {{\n  applyVisibility();\n  Plotly.react('plot3d', currentTraces, layout3d);\n}}\n\nselectComp({first_comp});\n</script>\n</body>\n</html>"""

def build_viewer_for_label(resultados: Path, seg_path: str, label: int, out_path: Path):
    base = resultados / f'label_{label:03d}'
    if not base.exists():
        print(f'ERROR: No existe {base}')
        return False
    print('Cargando segmentacion...')
    nii = nib.load(seg_path)
    seg_data = np.round(nii.get_fdata()).astype(np.int16)
    affine = nii.affine
    comp_csvs = sorted(base.glob('comp_*_centerlines.csv'))
    comp_ids = sorted(set((int(f.name.split('_')[1]) for f in comp_csvs)))
    if not comp_ids:
        print(f'No hay CSVs en {base}')
        return False
    bridges_path = base / 'graph_bridges.csv'
    bridges_df = pd.read_csv(bridges_path) if bridges_path.exists() else pd.DataFrame()
    print(f'Label {label} — componentes: {comp_ids}')
    if not bridges_df.empty:
        print(f'  Puentes inter-comp: {len(bridges_df)} segmentos')
    components = {}
    for cid in comp_ids:
        print(f'  Comp {cid}...')
        data = load_component(base, cid, seg_path, label, seg_data, affine)
        if data is not None:
            components[cid] = build_component_data(data, cid, bridges_df=bridges_df, label=label)
            c = components[cid]
            print(f"    {c['n_branches']} ramas, {c['n_pts']} pts, {c.get('n_nodes', 0)} nodos")
    if not components:
        print('Sin componentes validos')
        return False
    html = generate_html(label, components)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'  -> {out_path}')
    return True

def main():
    p = argparse.ArgumentParser(description='Viewer HTML: malla + skeleton + centerlines + nodos y aristas del grafo')
    p.add_argument('--resultados', required=True, help='Directorio de resultados (carpeta del caso o raiz)')
    p.add_argument('--label', type=int, default=None, help='Una etiqueta (ej. 6). Usar con --output o nombre por defecto.')
    p.add_argument('--labels', nargs='+', type=int, default=None, help='Varias etiquetas: genera un HTML por etiqueta (viewer_labelN.html)')
    p.add_argument('--seg', required=True, help='Segmentacion .nii.gz alineada con resultados')
    p.add_argument('--output', default=None, help='Salida (solo si una etiqueta con --label)')
    p.add_argument('--output-dir', default=None, help='Carpeta de salida si usas --labels (por defecto junto a --resultados)')
    args = p.parse_args()
    res = Path(args.resultados)
    labels = list(args.labels) if args.labels else [] if args.label is None else [args.label]
    if not labels:
        p.error('Indica --label N o --labels 6 9 10')
    if len(labels) == 1:
        out = Path(args.output) if args.output else Path(f'viewer_label{labels[0]}.html')
        build_viewer_for_label(res, args.seg, labels[0], out)
        return
    out_dir = Path(args.output_dir) if args.output_dir else res
    out_dir.mkdir(parents=True, exist_ok=True)
    for lb in labels:
        outp = out_dir / f'viewer_label{lb}.html'
        build_viewer_for_label(res, args.seg, lb, outp)
if __name__ == '__main__':
    main()
