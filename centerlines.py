
import sys
import argparse
import logging
import warnings
from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd
import nibabel as nib

from graph_utils import filter_nodes_incident_to_edges, save_bridges_for_label_dir

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("CenterlinePipeline")


def load_segmentation(nii_path):
    log.info(f"Cargando segmentacion: {nii_path}")
    nii = nib.load(nii_path)
    data = np.round(nii.get_fdata()).astype(np.int16)
    affine = nii.affine
    spacing = np.abs(np.diag(affine)[:3])
    labels = sorted([int(l) for l in np.unique(data) if l > 0])
    log.info(f"  Shape   : {data.shape}")
    log.info(f"  Spacing : {spacing} mm")
    log.info(f"  Labels  : {labels}")
    return data, affine, spacing, labels


def get_components(mask):
    from scipy import ndimage
    labeled, n = ndimage.label(mask)
    components = []
    for cid in range(1, n + 1):
        cm = (labeled == cid).astype(np.uint8)
        components.append((cid, cm, int(cm.sum())))
    components.sort(key=lambda x: x[2], reverse=True)
    return components


def compute_skeleton(mask):
    from skimage.morphology import skeletonize
    log.info("    Calculando esqueleto 3D...")
    skel = skeletonize(mask.astype(bool))
    log.info(f"    Voxels esqueleto (bruto): {skel.sum()}")
    return skel


def prune_skeleton(skel, min_branch_voxels=20):
    from scipy.ndimage import convolve
    kernel = np.ones((3, 3, 3), dtype=np.uint8)
    kernel[1, 1, 1] = 0
    offsets = [(di,dj,dk) for di in [-1,0,1] for dj in [-1,0,1] for dk in [-1,0,1]
               if not (di==0 and dj==0 and dk==0)]
    skel = skel.copy()
    shape = skel.shape

    for _ in range(100):
        nc = convolve(skel.astype(np.uint8), kernel, mode='constant', cval=0)
        endpoints = list(map(tuple, np.argwhere(skel & (nc == 1))))
        if not endpoints:
            break
        removed_any = False
        for ep in endpoints:
            branch, current, previous = [], ep, None
            while True:
                branch.append(current)
                ci, cj, ck = current
                nbs = [(ci+di,cj+dj,ck+dk) for di,dj,dk in offsets
                       if 0<=ci+di<shape[0] and 0<=cj+dj<shape[1] and 0<=ck+dk<shape[2]
                       and skel[(ci+di,cj+dj,ck+dk)] and (ci+di,cj+dj,ck+dk)!=previous]
                if not nbs or len(nbs)>=2 or len(branch)>=min_branch_voxels:
                    break
                previous, current = current, nbs[0]
            if len(branch) < min_branch_voxels:
                li,lj,lk = branch[-1]
                deg = sum(1 for di,dj,dk in offsets
                          if 0<=li+di<shape[0] and 0<=lj+dj<shape[1] and 0<=lk+dk<shape[2]
                          and skel[(li+di,lj+dj,lk+dk)])
                if deg < 3:
                    for v in branch:
                        skel[v] = False
                    removed_any = True
        if not removed_any:
            break

    log.info(f"    Voxels esqueleto (podado): {skel.sum()}")
    return skel


def build_graph(skel):
    voxels = np.argwhere(skel)
    vset = set(map(tuple, voxels))
    v2id = {tuple(v): i for i, v in enumerate(voxels)}
    offsets = [(di,dj,dk) for di in [-1,0,1] for dj in [-1,0,1] for dk in [-1,0,1]
               if not (di==0 and dj==0 and dk==0)]
    adj = {i: [] for i in range(len(voxels))}
    for i, v in enumerate(voxels):
        vt = tuple(v)
        for off in offsets:
            nb = (vt[0]+off[0], vt[1]+off[1], vt[2]+off[2])
            if nb in vset:
                adj[i].append(v2id[nb])
    return voxels, adj


def bfs_farthest(start, adj):
    visited = {start: None}
    queue = deque([start])
    last = start
    while queue:
        node = queue.popleft()
        last = node
        for nb in adj[node]:
            if nb not in visited:
                visited[nb] = node
                queue.append(nb)
    path = []
    curr = last
    while curr is not None:
        path.append(curr)
        curr = visited[curr]
    path.reverse()
    return last, path


def find_longest_path(voxels, adj, spacing):
    degrees = {i: len(nb) for i, nb in adj.items()}
    endpoints = [i for i, d in degrees.items() if d == 1]
    if not endpoints:
        centroid = voxels.mean(axis=0)
        dists = np.linalg.norm((voxels - centroid) * spacing, axis=1)
        endpoints = [int(np.argmax(dists))]
    far1, _ = bfs_farthest(endpoints[0], adj)
    far2, path = bfs_farthest(far1, adj)
    log.info(f"    Camino principal: {len(path)} voxels")
    return path


def voxel_to_mm(voxels_ijk, affine):
    if len(voxels_ijk) == 0:
        return np.empty((0, 3))
    ijk1 = np.hstack([voxels_ijk[:, [2, 1, 0]], np.ones((len(voxels_ijk), 1))])
    return (affine @ ijk1.T).T[:, :3]


def mm_to_voxel_array_indices(pts_mm, affine):
    """Inverso coherente con voxel_to_mm: pts mm -> índices (dim0,dim1,dim2) del array."""
    pts_mm = np.asarray(pts_mm, dtype=float).reshape(-1, 3)
    if len(pts_mm) == 0:
        return np.empty((0, 3), dtype=int)
    hom = np.concatenate([pts_mm, np.ones((len(pts_mm), 1))], axis=1)
    vox_mix = (np.linalg.inv(affine) @ hom.T).T[:, :3]
    ijk = np.stack([vox_mix[:, 2], vox_mix[:, 1], vox_mix[:, 0]], axis=1)
    return np.round(ijk).astype(np.int32)


def diameter_sampled_along_branches_mm(branches, comp_mask, affine, spacing):
    """Diámetro local (2*EDT) en mm en cada punto de centerline, misma máscara que el comp."""
    from scipy import ndimage as ndi

    sp = tuple(float(s) for s in spacing)
    edt = ndi.distance_transform_edt(comp_mask.astype(bool), sampling=sp)
    shape = comp_mask.shape
    out = []
    for pts in branches:
        ijk = mm_to_voxel_array_indices(pts, affine)
        d = np.full(len(pts), np.nan, dtype=float)
        for i in range(len(pts)):
            ii, jj, kk = int(ijk[i, 0]), int(ijk[i, 1]), int(ijk[i, 2])
            if 0 <= ii < shape[0] and 0 <= jj < shape[1] and 0 <= kk < shape[2]:
                d[i] = float(2.0 * edt[ii, jj, kk])
        out.append(d)
    return out


def cta_matches_segmentation(cta_nii, seg_shape: tuple, seg_affine: np.ndarray) -> bool:
    sh = tuple(int(x) for x in cta_nii.shape[:3])
    if sh != tuple(int(x) for x in seg_shape[:3]):
        return False
    return bool(np.allclose(cta_nii.affine, seg_affine, rtol=1e-5, atol=1e-4))


def hu_sampled_along_branches(cta_data: np.ndarray, affine: np.ndarray, branches):
    """HU del CTA (interpolación orden 1) en cada punto de centerline, mm -> vox."""
    from scipy.ndimage import map_coordinates

    inv = np.linalg.inv(affine)
    out: list[np.ndarray] = []
    for pts in branches:
        pts = np.asarray(pts, dtype=float).reshape(-1, 3)
        if len(pts) == 0:
            out.append(np.array([], dtype=float))
            continue
        hom = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
        vox_mix = (inv @ hom.T).T[:, :3]
        ijk = np.stack([vox_mix[:, 2], vox_mix[:, 1], vox_mix[:, 0]], axis=0)
        vals = map_coordinates(
            np.asarray(cta_data, dtype=float), ijk, order=1, mode="nearest"
        )
        out.append(vals.astype(float))
    return out


def smooth_centerline(pts, sigma=2.0):
    from scipy.ndimage import gaussian_filter1d
    if len(pts) < 5:
        return pts
    return np.column_stack([
        gaussian_filter1d(pts[:, 0], sigma=sigma),
        gaussian_filter1d(pts[:, 1], sigma=sigma),
        gaussian_filter1d(pts[:, 2], sigma=sigma),
    ])


def _walk_to_next_special(adj, special, start_nb, from_node):
    """Camina desde start_nb hasta el siguiente nodo especial (grafo de esqueleto)."""
    p = [from_node, start_nb]
    curr, previous = start_nb, from_node
    while True:
        if curr in special and curr != from_node:
            break
        nbs = [n for n in adj[curr] if n != previous]
        if not nbs:
            break
        if len(nbs) > 1:
            break
        previous, curr = curr, nbs[0]
        p.append(curr)
    return p


def skeleton_to_branches_multi_with_graph(
    skel, affine, spacing, min_branch_mm=5.0, smooth_sigma=2.0
):
    """
    Ramas entre nodos especiales (grado 1 o >=3) + tablas de nodos y aristas.
    Los nodos son bifurcaciones y extremos del esqueleto (donde se juntan ramas).
    """
    voxels, adj = build_graph(skel)
    if len(voxels) == 0:
        return [], pd.DataFrame(), pd.DataFrame()
    degrees = {i: len(nb) for i, nb in adj.items()}
    special = set(i for i, d in degrees.items() if d == 1 or d >= 3)
    if not special:
        centroid = voxels.mean(axis=0)
        dists = np.linalg.norm(voxels - centroid, axis=1)
        special = {int(np.argmax(dists))}

    special_sorted = sorted(special)
    gid_to_nid = {gid: nid for nid, gid in enumerate(special_sorted)}
    nodes_rows = []
    for gid in special_sorted:
        xyz = voxel_to_mm(voxels[[gid]], affine)[0]
        deg = int(degrees[gid])
        role = "junction" if deg >= 3 else "endpoint"
        nodes_rows.append(
            {
                "node_id": gid_to_nid[gid],
                "graph_vertex_id": int(gid),
                "degree": deg,
                "role": role,
                "x": float(xyz[0]),
                "y": float(xyz[1]),
                "z": float(xyz[2]),
            }
        )
    nodes_df = pd.DataFrame(nodes_rows)

    branches = []
    branch_rows = []
    visited_pairs = set()
    bid = 0

    for node in special:
        for nb in adj[node]:
            p = _walk_to_next_special(adj, special, nb, node)
            end_node = p[-1]
            pair = (min(node, end_node), max(node, end_node), len(p))
            if pair in visited_pairs:
                continue
            visited_pairs.add(pair)
            mm = voxel_to_mm(voxels[p], affine)
            length = float(np.linalg.norm(np.diff(mm, axis=0), axis=1).sum())
            if length >= min_branch_mm:
                pts = smooth_centerline(mm, smooth_sigma)
                branches.append(pts)
                branch_rows.append(
                    {
                        "branch_id": bid,
                        "node_start": gid_to_nid[node],
                        "node_end": gid_to_nid[end_node],
                        "length_mm": round(length, 4),
                    }
                )
                bid += 1

    log.info(f"    Ramas (multi): {len(branches)}  |  nodos: {len(nodes_df)}")
    return branches, nodes_df, pd.DataFrame(branch_rows)


def skeleton_to_branches_multi(skel, affine, spacing, min_branch_mm=5.0):
    branches, _, _ = skeleton_to_branches_multi_with_graph(
        skel, affine, spacing, min_branch_mm=min_branch_mm
    )
    return branches


def single_branch_with_graph(voxels, adj, spacing, affine, smooth_sigma=2.0):
    """Una sola rama (camino mas largo) y dos nodos en los extremos."""
    path_ids = find_longest_path(voxels, adj, spacing)
    if not path_ids:
        return [], pd.DataFrame(), pd.DataFrame()
    pts_mm = smooth_centerline(
        voxel_to_mm(voxels[path_ids], affine), smooth_sigma
    )
    branches = [pts_mm]
    n0, n1 = int(path_ids[0]), int(path_ids[-1])
    p0 = voxel_to_mm(voxels[[n0]], affine)[0]
    p1 = voxel_to_mm(voxels[[n1]], affine)[0]
    d0, d1 = len(adj[n0]), len(adj[n1])
    nodes_df = pd.DataFrame(
        [
            {
                "node_id": 0,
                "graph_vertex_id": n0,
                "degree": d0,
                "role": "endpoint",
                "x": float(p0[0]),
                "y": float(p0[1]),
                "z": float(p0[2]),
            },
            {
                "node_id": 1,
                "graph_vertex_id": n1,
                "degree": d1,
                "role": "endpoint",
                "x": float(p1[0]),
                "y": float(p1[1]),
                "z": float(p1[2]),
            },
        ]
    )
    length_mm = float(np.linalg.norm(np.diff(pts_mm, axis=0), axis=1).sum())
    edges_df = pd.DataFrame(
        [
            {
                "branch_id": 0,
                "node_start": 0,
                "node_end": 1,
                "length_mm": round(length_mm, 4),
            }
        ]
    )
    log.info(f"    Nodos (single): 2 extremos  |  longitud rama: {length_mm:.2f} mm")
    return branches, nodes_df, edges_df


def compute_curvature(pts):
    if len(pts) < 3:
        return np.full(len(pts), np.nan)
    out = np.full(len(pts), np.nan)
    for i in range(1, len(pts)-1):
        v1, v2 = pts[i]-pts[i-1], pts[i+1]-pts[i]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            continue
        ds = (n1+n2)/2
        out[i] = np.linalg.norm(v2/n2 - v1/n1) / ds if ds > 0 else np.nan
    return out


def compute_torsion(pts):
    if len(pts) < 4:
        return np.full(len(pts), np.nan)
    out = np.full(len(pts), np.nan)
    for i in range(1, len(pts)-2):
        t1,t2,t3 = pts[i]-pts[i-1], pts[i+1]-pts[i], pts[i+2]-pts[i+1]
        b1,b2 = np.cross(t1,t2), np.cross(t2,t3)
        nb1,nb2 = np.linalg.norm(b1), np.linalg.norm(b2)
        if nb1 < 1e-9 or nb2 < 1e-9:
            continue
        ds = np.linalg.norm(t2)
        out[i] = np.arccos(np.clip(np.dot(b1/nb1,b2/nb2),-1,1)) / ds if ds > 0 else np.nan
    return out


def branches_to_dataframe(
    branches,
    label,
    comp_id,
    diameter_per_branch=None,
    hu_per_branch=None,
):
    rows = []
    for bid, pts in enumerate(branches):
        curv, tors = compute_curvature(pts), compute_torsion(pts)
        diam = diameter_per_branch[bid] if diameter_per_branch is not None else None
        hu_b = hu_per_branch[bid] if hu_per_branch is not None else None
        for i, (x, y, z) in enumerate(pts):
            row = {
                "label": label,
                "component": comp_id,
                "branch_id": bid,
                "point_id": i,
                "x": x,
                "y": y,
                "z": z,
                "curvature": float(curv[i]) if not np.isnan(curv[i]) else np.nan,
                "torsion": float(tors[i]) if not np.isnan(tors[i]) else np.nan,
            }
            if diam is not None:
                row["diameter_mm"] = (
                    float(diam[i]) if not np.isnan(diam[i]) else np.nan
                )
            if hu_b is not None and i < len(hu_b):
                row["hu"] = float(hu_b[i]) if not np.isnan(hu_b[i]) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def compute_branch_stats(df):
    records = []
    for (label, comp, bid), grp in df.groupby(["label", "component", "branch_id"]):
        pts = grp[["x", "y", "z"]].values
        length_mm = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
        euclidean = float(np.linalg.norm(pts[-1] - pts[0])) if len(pts) > 1 else 0.0
        tortuosity = (length_mm / euclidean) if euclidean > 0 else np.nan
        rec = {
            "label": label,
            "component": comp,
            "branch_id": bid,
            "n_points": len(grp),
            "length_mm": round(length_mm, 3),
            "euclidean_mm": round(euclidean, 3),
            "tortuosity": round(tortuosity, 4) if not np.isnan(tortuosity) else np.nan,
            "curvature_mean": round(grp["curvature"].mean(), 5),
            "curvature_max": round(grp["curvature"].max(), 5),
            "torsion_mean": round(grp["torsion"].mean(), 5),
        }
        if "diameter_mm" in grp.columns and grp["diameter_mm"].notna().any():
            g = grp["diameter_mm"].dropna()
            rec["diameter_min_mm"] = round(float(g.min()), 4)
            rec["diameter_max_mm"] = round(float(g.max()), 4)
            rec["diameter_mean_mm"] = round(float(g.mean()), 4)
            rec["diameter_std_mm"] = (
                round(float(g.std()), 4) if len(g) > 1 else 0.0
            )
        if "hu" in grp.columns and grp["hu"].notna().any():
            h = grp["hu"].dropna()
            rec["hu_min"] = round(float(h.min()), 2)
            rec["hu_max"] = round(float(h.max()), 2)
            rec["hu_mean"] = round(float(h.mean()), 2)
            rec["hu_std"] = round(float(h.std()), 2) if len(h) > 1 else 0.0
        records.append(rec)
    return pd.DataFrame(records)


def save_csv(df, path):
    df.to_csv(path, index=False, float_format="%.4f")
    log.info(f"  CSV: {path}")


def save_skeleton_nii(skel, affine, path):
    nib.save(nib.Nifti1Image(skel.astype(np.uint8), affine), path)
    log.info(f"  Skeleton NIfTI: {path}")


def save_vtp(branches, path):
    try:
        import vtk
        poly = vtk.vtkPolyData()
        pts  = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        pid = 0
        for branch in branches:
            if len(branch) < 2:
                continue
            line = vtk.vtkPolyLine()
            line.GetPointIds().SetNumberOfIds(len(branch))
            for i, (x,y,z) in enumerate(branch):
                pts.InsertNextPoint(float(x),float(y),float(z))
                line.GetPointIds().SetId(i, pid)
                pid += 1
            lines.InsertNextCell(line)
        poly.SetPoints(pts)
        poly.SetLines(lines)
        w = vtk.vtkXMLPolyDataWriter()
        w.SetInputData(poly)
        w.SetFileName(path)
        w.Write()
        log.info(f"  VTP: {path}")
    except Exception as e:
        log.warning(f"  VTP no guardado: {e}")


def run_pipeline(
    input_path,
    output_dir,
    target_labels=None,
    mode="single",
    label_modes=None,
    prune_voxels=20,
    smooth_sigma=2.0,
    min_branch_mm=5.0,
    min_voxels=50,
    save_skeleton=False,
    bridge_max_mm=45.0,
    bridge_labels=(9, 10),
    cta_path=None,
):
    """
    Si label_modes es un dict {label: 'single'|'multi'}, se usa ese modo por etiqueta
    y se ignoran target_labels/mode por defecto salvo que label_modes sea None.

    bridge_labels: en modo multi, tras procesar la etiqueta se calculan puentes heuristicos
    entre componentes conectadas-disjuntas (mm) y se guarda graph_bridges.csv.
    """

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(out / "pipeline.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    log.addHandler(fh)

    data, affine, spacing, all_labels = load_segmentation(input_path)

    cta_arr = None
    if cta_path:
        cn = nib.load(str(cta_path))
        if cta_matches_segmentation(cn, data.shape, affine):
            cta_arr = cn.get_fdata().astype(np.float32)
            log.info(f"CTA alineado con segmentación (HU en centerlines): {cta_path}")
        else:
            log.warning(
                f"CTA no coincide en shape/affine con la segmentación; se omite HU. "
                f"cta={cn.shape[:3]} seg={data.shape}"
            )

    if label_modes is not None:
        labels_to_process = sorted(label_modes.keys())
    else:
        labels_to_process = target_labels if target_labels else all_labels
    all_pts, all_stats = [], []
    all_nodes, all_edges = [], []

    for label in labels_to_process:
        eff_mode = label_modes[label] if label_modes else mode
        if label not in all_labels:
            log.warning(f"Label {label} no se encuentra, saltando")
            continue
        log.info(f"\n{'='*60}\nLabel {label}  [modo={eff_mode}]\n{'='*60}")
        mask = (data == label).astype(np.uint8)
        components = get_components(mask)
        log.info(f"  COmponents: {len(components)}")
        label_dir = out / f"label_{label:03d}"
        label_dir.mkdir(exist_ok=True)

        for comp_id, comp_mask, n_voxels in components:
            log.info(f"\n  Componente {comp_id} -- {n_voxels} voxels")
            if n_voxels < min_voxels:
                log.warning("   molt petit, saltando.")
                continue
            prefix = str(label_dir / f"comp_{comp_id:03d}")
            skel_raw = compute_skeleton(comp_mask)
            # Prune adaptativo: no mas del 25% del esqueleto para componentes pequenos
            skel_size = int(skel_raw.sum())
            adaptive_prune = min(prune_voxels, max(3, skel_size // 6))
            log.info(f"    Prune adaptativo: {adaptive_prune} voxels (esqueleto={skel_size})")
            skel     = prune_skeleton(skel_raw, min_branch_voxels=adaptive_prune)
            if save_skeleton:
                save_skeleton_nii(skel_raw, affine, f"{prefix}_skeleton.nii.gz")
            if skel.sum() == 0:
                log.warning(" skeleton buid")
                continue
            if eff_mode == "single":
                voxels, adj = build_graph(skel)
                branches, nodes_df, edges_df = single_branch_with_graph(
                    voxels, adj, spacing, affine, smooth_sigma
                )
            else:
                branches, nodes_df, edges_df = skeleton_to_branches_multi_with_graph(
                    skel, affine, spacing, min_branch_mm, smooth_sigma
                )
            if not branches:
                log.warning("    Sin ramas validas.")
                continue
            if not nodes_df.empty and not edges_df.empty:
                nodes_df, edges_df = filter_nodes_incident_to_edges(
                    nodes_df, edges_df
                )
            save_vtp(branches, f"{prefix}_centerlines.vtp")
            if not nodes_df.empty:
                nd = nodes_df.copy()
                nd.insert(0, "component", comp_id)
                nd.insert(0, "label", label)
                save_csv(nd, f"{prefix}_graph_nodes.csv")
                all_nodes.append(nd)
            if not edges_df.empty:
                ed = edges_df.copy()
                ed.insert(0, "component", comp_id)
                ed.insert(0, "label", label)
                save_csv(ed, f"{prefix}_graph_edges.csv")
                all_edges.append(ed)
            diam_list = diameter_sampled_along_branches_mm(
                branches, comp_mask.astype(bool), affine, spacing
            )
            hu_list = (
                hu_sampled_along_branches(cta_arr, affine, branches)
                if cta_arr is not None
                else None
            )
            df_pts = branches_to_dataframe(
                branches,
                label,
                comp_id,
                diameter_per_branch=diam_list,
                hu_per_branch=hu_list,
            )
            save_csv(df_pts, f"{prefix}_centerlines.csv")
            all_pts.append(df_pts)
            df_stats = compute_branch_stats(df_pts)
            save_csv(df_stats, f"{prefix}_branch_stats.csv")
            all_stats.append(df_stats)
            log.info(f"    Ramas: {len(branches)}  |  Puntos: {len(df_pts)}")
            log.info(df_stats.to_string(index=False))

        if label in bridge_labels and eff_mode == "multi":
            save_bridges_for_label_dir(label_dir, label, max_bridge_mm=bridge_max_mm)
            bp = label_dir / "graph_bridges.csv"
            if bp.exists():
                log.info(f"  Puentes inter-componente guardados: {bp}")

    if all_pts:
        save_csv(pd.concat(all_pts, ignore_index=True), str(out / "all_centerlines.csv"))
    if all_stats:
        save_csv(pd.concat(all_stats, ignore_index=True), str(out / "all_branch_stats.csv"))
    if all_nodes:
        save_csv(
            pd.concat(all_nodes, ignore_index=True), str(out / "all_graph_nodes.csv")
        )
    if all_edges:
        save_csv(
            pd.concat(all_edges, ignore_index=True), str(out / "all_graph_edges.csv")
        )
    log.info(f"\nPipeline completo. Outputs en: {output_dir}")
    log.removeHandler(fh)
    fh.close()


def parse_args():
    p = argparse.ArgumentParser(description="Centerline pipeline sin VMTK v3")
    p.add_argument("--input",         required=True)
    p.add_argument("--output",        required=True)
    p.add_argument("--labels",        nargs="+", type=int, default=None)
    p.add_argument("--mode",          choices=["single","multi"], default="single",
                   help="single=camino principal | multi=todas las ramas (default: single)")
    p.add_argument("--prune-voxels",  type=int,   default=20,
                   help="Eliminar ramas del esqueleto con menos de N voxels (default: 20)")
    p.add_argument("--smooth-sigma",  type=float, default=2.0,
                   help="Suavizado gaussiano del centerline (default: 2.0)")
    p.add_argument("--min-branch-mm", type=float, default=5.0,
                   help="Longitud minima de rama en modo multi, mm (default: 5.0)")
    p.add_argument("--min-voxels",    type=int,   default=50)
    p.add_argument("--save-skeleton", action="store_true")
    p.add_argument(
        "--cta",
        type=str,
        default=None,
        help="NIfTI CTA (misma shape/affine que la segmentación) para columna hu en centerlines.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        input_path=args.input,
        output_dir=args.output,
        target_labels=args.labels,
        mode=args.mode,
        prune_voxels=args.prune_voxels,
        smooth_sigma=args.smooth_sigma,
        min_branch_mm=args.min_branch_mm,
        min_voxels=args.min_voxels,
        save_skeleton=args.save_skeleton,
        cta_path=args.cta,
    )