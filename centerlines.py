
import sys
import argparse
import logging
import warnings
from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd
import nibabel as nib

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


def smooth_centerline(pts, sigma=2.0):
    from scipy.ndimage import gaussian_filter1d
    if len(pts) < 5:
        return pts
    return np.column_stack([
        gaussian_filter1d(pts[:, 0], sigma=sigma),
        gaussian_filter1d(pts[:, 1], sigma=sigma),
        gaussian_filter1d(pts[:, 2], sigma=sigma),
    ])


def skeleton_to_branches_multi(skel, affine, spacing, min_branch_mm=5.0):
    voxels, adj = build_graph(skel)
    if len(voxels) == 0:
        return []
    degrees = {i: len(nb) for i, nb in adj.items()}
    special = set(i for i, d in degrees.items() if d == 1 or d >= 3)
    if not special:
        centroid = voxels.mean(axis=0)
        dists = np.linalg.norm(voxels - centroid, axis=1)
        special = {int(np.argmax(dists))}

    branches = []
    visited_pairs = set()

    def walk_to_next_special(start_nb, from_node):
        """Camina desde start_nb hasta el siguiente nodo especial."""
        p = [from_node, start_nb]
        curr, previous = start_nb, from_node
        while True:
            if curr in special and curr != from_node:
                break
            nbs = [n for n in adj[curr] if n != previous]
            if not nbs:
                break
            if len(nbs) > 1:
                # bifurcacion — el propio curr es un nodo especial no detectado
                break
            previous, curr = curr, nbs[0]
            p.append(curr)
        return p

    for node in special:
        for nb in adj[node]:
            p = walk_to_next_special(nb, node)
            end_node = p[-1]
            # Clave canonica para evitar A->B y B->A
            pair = (min(node, end_node), max(node, end_node), len(p))
            if pair in visited_pairs:
                continue
            visited_pairs.add(pair)
            mm = voxel_to_mm(voxels[p], affine)
            length = float(np.linalg.norm(np.diff(mm, axis=0), axis=1).sum())
            if length >= min_branch_mm:
                branches.append(smooth_centerline(mm))

    log.info(f"    Ramas (multi): {len(branches)}")
    return branches


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


def branches_to_dataframe(branches, label, comp_id):
    rows = []
    for bid, pts in enumerate(branches):
        curv, tors = compute_curvature(pts), compute_torsion(pts)
        for i, (x,y,z) in enumerate(pts):
            rows.append({"label":label,"component":comp_id,"branch_id":bid,
                         "point_id":i,"x":x,"y":y,"z":z,
                         "curvature": float(curv[i]) if not np.isnan(curv[i]) else np.nan,
                         "torsion":   float(tors[i]) if not np.isnan(tors[i]) else np.nan})
    return pd.DataFrame(rows)


def compute_branch_stats(df):
    records = []
    for (label,comp,bid), grp in df.groupby(["label","component","branch_id"]):
        pts = grp[["x","y","z"]].values
        length_mm = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
        euclidean = float(np.linalg.norm(pts[-1]-pts[0])) if len(pts)>1 else 0.0
        tortuosity = (length_mm/euclidean) if euclidean > 0 else np.nan
        records.append({"label":label,"component":comp,"branch_id":bid,
                        "n_points":len(grp),
                        "length_mm":round(length_mm,3),
                        "euclidean_mm":round(euclidean,3),
                        "tortuosity":round(tortuosity,4) if not np.isnan(tortuosity) else np.nan,
                        "curvature_mean":round(grp["curvature"].mean(),5),
                        "curvature_max":round(grp["curvature"].max(),5),
                        "torsion_mean":round(grp["torsion"].mean(),5)})
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


def run_pipeline(input_path, output_dir, target_labels=None, mode="single",
                 prune_voxels=20, smooth_sigma=2.0, min_branch_mm=5.0,
                 min_voxels=50, save_skeleton=False):

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(out / "pipeline.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    log.addHandler(fh)

    data, affine, spacing, all_labels = load_segmentation(input_path)
    labels_to_process = target_labels if target_labels else all_labels
    all_pts, all_stats = [], []

    for label in labels_to_process:
        if label not in all_labels:
            log.warning(f"Label {label} no se encuentra, saltando")
            continue
        log.info(f"\n{'='*60}\nLabel {label}  [modo={mode}]\n{'='*60}")
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
            if mode == "single":
                voxels, adj = build_graph(skel)
                path_ids = find_longest_path(voxels, adj, spacing)
                if not path_ids:
                    continue
                pts_mm = smooth_centerline(voxel_to_mm(voxels[path_ids], affine), smooth_sigma)
                branches = [pts_mm]
            else:
                branches = skeleton_to_branches_multi(skel, affine, spacing, min_branch_mm)
            if not branches:
                log.warning("    Sin ramas validas.")
                continue
            save_vtp(branches, f"{prefix}_centerlines.vtp")
            df_pts = branches_to_dataframe(branches, label, comp_id)
            save_csv(df_pts, f"{prefix}_centerlines.csv")
            all_pts.append(df_pts)
            df_stats = compute_branch_stats(df_pts)
            save_csv(df_stats, f"{prefix}_branch_stats.csv")
            all_stats.append(df_stats)
            log.info(f"    Ramas: {len(branches)}  |  Puntos: {len(df_pts)}")
            log.info(df_stats.to_string(index=False))

    if all_pts:
        save_csv(pd.concat(all_pts,   ignore_index=True), str(out/"all_centerlines.csv"))
    if all_stats:
        save_csv(pd.concat(all_stats, ignore_index=True), str(out/"all_branch_stats.csv"))
    log.info(f"\nPipeline completo. Outputs en: {output_dir}")


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
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        input_path    = args.input,
        output_dir    = args.output,
        target_labels = args.labels,
        mode          = args.mode,
        prune_voxels  = args.prune_voxels,
        smooth_sigma  = args.smooth_sigma,
        min_branch_mm = args.min_branch_mm,
        min_voxels    = args.min_voxels,
        save_skeleton = args.save_skeleton,
    )