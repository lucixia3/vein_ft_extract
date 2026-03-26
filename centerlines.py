"""
=============================================================================
  VASCULAR CENTERLINE EXTRACTION PIPELINE
"""

import os
import sys
import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm

os.environ["VTK_SILENCE_GET_VOID_POINTER_WARNINGS"] = "1"
warnings.filterwarnings("ignore")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("CenterlinePipeline")



def _import_vmtk():
    try:
        import vmtk
        from vmtk import vmtkscripts, pypes
        import vtk
        from vtk.util import numpy_support as ns
        return vmtk, vmtkscripts, pypes, vtk, ns
    except ImportError:
        log.error(
            "VMTK not found. Install with:\n"
            "  conda create -n vmtk python=3.8\n"
            "  conda activate vmtk\n"
            "  conda install -c vmtk vmtk\n"
            "  pip install nibabel pandas tqdm"
        )
        sys.exit(1)


def load_segmentation(nii_path: str):
    log.info(f"Loading segmentation: {nii_path}")
    nii = nib.load(nii_path)
    data = np.round(nii.get_fdata()).astype(np.int16)
    affine = nii.affine
    spacing = np.abs(np.diag(affine)[:3])  # voxel size in mm
    labels = sorted([int(l) for l in np.unique(data) if l > 0])
    log.info(f"  Volume shape : {data.shape}")
    log.info(f"  Voxel spacing: {spacing} mm")
    log.info(f"  Labels found : {labels}")
    return data, affine, spacing, labels


def get_components(mask: np.ndarray):
    """
    una mascara per component connectat
    """
    from scipy import ndimage
    labeled, n = ndimage.label(mask)
    components = []
    for comp_id in range(1, n + 1):
        comp_mask = (labeled == comp_id).astype(np.uint8)
        voxels = int(comp_mask.sum())
        components.append((comp_id, comp_mask, voxels))
    components.sort(key=lambda x: x[2], reverse=True)
    return components



def mask_to_surface(mask: np.ndarray, spacing: np.ndarray,
                    smooth_iterations: int = 30, smooth_passband: float = 0.1):

    _, vmtkscripts, _, vtk, ns = _import_vmtk()

    mc = vmtkscripts.vmtkMarchingCubes()
    mc.Image = _numpy_to_vtk_image(mask, spacing)
    mc.Level = 0.5
    mc.Execute()
    surface = mc.Surface

    # Surface cleaning
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(surface)
    cleaner.Update()
    surface = cleaner.GetOutput()

    triangulator = vtk.vtkTriangleFilter()
    triangulator.SetInputData(surface)
    triangulator.Update()
    surface = triangulator.GetOutput()

    smoother = vmtkscripts.vmtkSurfaceSmoothing()
    smoother.Surface = surface
    smoother.Method = "taubin"
    smoother.NumberOfIterations = smooth_iterations
    smoother.PassBand = smooth_passband
    smoother.Execute()
    surface = smoother.Surface

    capper = vmtkscripts.vmtkSurfaceCapper()
    capper.Surface = surface
    capper.Method = "centerpoint"
    capper.Execute()
    surface = capper.Surface

    log.debug(f"  Surface: {surface.GetNumberOfPoints()} pts, "
              f"{surface.GetNumberOfCells()} cells")
    return surface


def _numpy_to_vtk_image(mask: np.ndarray, spacing: np.ndarray):
    _, _, _, vtk, ns = _import_vmtk()
    img = vtk.vtkImageData()
    img.SetDimensions(mask.shape[2], mask.shape[1], mask.shape[0])
    img.SetSpacing(float(spacing[2]), float(spacing[1]), float(spacing[0]))
    flat = mask.flatten(order="C").astype(np.float64)
    arr = ns.numpy_to_vtk(flat, deep=True, array_type=vtk.VTK_DOUBLE)
    arr.SetName("ImageScalars")
    img.GetPointData().SetScalars(arr)
    return img


def extract_centerlines(surface, mode: str = "auto"):
    """
    Extract centerlines from surface using VMTK's Voronoi/Fast-Marching method
    mode = 'auto'        → uses open-profile barycenters as seeds 
    mode = 'interactive' → opens VMTK GUI for manual seed placement

    """
    _, vmtkscripts, _, _, _ = _import_vmtk()

    cl = vmtkscripts.vmtkCenterlines()
    cl.Surface = surface
    cl.AppendEndPoints = 1
    cl.Resampling = 1
    cl.ResamplingStepLength = 1.0  # mm

    if mode == "auto":
        cl.SeedSelectorName = "openprofiles"
    else:
        cl.SeedSelectorName = "pickpoint"

    cl.Execute()
    return cl.Centerlines



def compute_geometry(centerlines):
    _, vmtkscripts, _, _, _ = _import_vmtk()

    geom = vmtkscripts.vmtkCenterlineGeometry()
    geom.Centerlines = centerlines
    geom.LineSmoothing = 1
    geom.NumberOfSmoothingIterations = 100
    geom.Execute()
    return geom.Centerlines


def save_centerlines_vtp(centerlines, path: str):
    _, _, _, vtk, _ = _import_vmtk()
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetInputData(centerlines)
    writer.SetFileName(path)
    writer.Write()
    log.info(f"  Saved VTP: {path}")


def centerlines_to_dataframe(centerlines, label: int, comp_id: int) -> pd.DataFrame:

    _, _, _, vtk, ns = _import_vmtk()

    rows = []
    pd_data = centerlines.GetPointData()

    def _get_array(name):
        arr = pd_data.GetArray(name)
        if arr:
            return ns.vtk_to_numpy(arr)
        return None

    radius_arr    = _get_array("MaximumInscribedSphereRadius")
    curvature_arr = _get_array("Curvature")
    torsion_arr   = _get_array("Torsion")

    points = centerlines.GetPoints()
    lines  = centerlines.GetLines()
    lines.InitTraversal()

    line_id = 0
    id_list = vtk.vtkIdList()

    while lines.GetNextCell(id_list):
        for i in range(id_list.GetNumberOfIds()):
            pid = id_list.GetId(i)
            x, y, z = points.GetPoint(pid)
            row = {
                "label":     label,
                "component": comp_id,
                "line_id":   line_id,
                "point_id":  pid,
                "x": x, "y": y, "z": z,
                "radius_mm":   float(radius_arr[pid])   if radius_arr    is not None else np.nan,
                "curvature":   float(curvature_arr[pid]) if curvature_arr is not None else np.nan,
                "torsion":     float(torsion_arr[pid])   if torsion_arr   is not None else np.nan,
            }
            rows.append(row)
        line_id += 1

    return pd.DataFrame(rows)


def save_centerlines_csv(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False, float_format="%.4f")
    log.info(f"  Saved CSV : {path}")


#pipeline 
def run_pipeline(
    input_path: str,
    output_dir: str,
    target_labels: list = None,
    smooth_iter: int = 30,
    mode: str = "auto",
    min_voxels: int = 50,
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(out / "pipeline.log")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    log.addHandler(fh)

    data, affine, spacing, all_labels = load_segmentation(input_path)
    labels_to_process = target_labels if target_labels else all_labels

    all_dfs = []

    for label in labels_to_process:
        if label not in all_labels:
            log.warning(f"Label {label} not found in segmentation, skipping.")
            continue

        log.info(f"\n{'='*60}")
        log.info(f"Processing label {label}")
        log.info(f"{'='*60}")

        mask = (data == label).astype(np.uint8)
        components = get_components(mask)
        log.info(f"  Connected components: {len(components)}")

        label_dir = out / f"label_{label:03d}"
        label_dir.mkdir(exist_ok=True)

        for comp_id, comp_mask, n_voxels in components:
            log.info(f"\n  Component {comp_id} — {n_voxels} voxels")

            if n_voxels < min_voxels:
                log.warning(f"    Too small ({n_voxels} < {min_voxels}), skipping.")
                continue

            prefix = str(label_dir / f"comp_{comp_id:03d}")

            log.info("    Building surface ...")
            try:
                surface = mask_to_surface(comp_mask, spacing, smooth_iter)
            except Exception as e:
                log.error(f"    Surface extraction failed: {e}")
                continue

            log.info(f"    Extracting centerlines (mode={mode}) ...")
            try:
                centerlines = extract_centerlines(surface, mode=mode)
            except Exception as e:
                log.error(f"    Centerline extraction failed: {e}")
                continue

            log.info("    Computing geometry (curvature, torsion) ...")
            try:
                centerlines = compute_geometry(centerlines)
            except Exception as e:
                log.warning(f"    Geometry computation failed: {e}")

            save_centerlines_vtp(centerlines, f"{prefix}_centerlines.vtp")
            df = centerlines_to_dataframe(centerlines, label, comp_id)
            save_centerlines_csv(df, f"{prefix}_centerlines.csv")
            all_dfs.append(df)

            log.info(f"    Points extracted : {len(df)}")
            if not df["radius_mm"].isna().all():
                log.info(f"    Radius (mm)      : "
                         f"mean={df['radius_mm'].mean():.2f}, "
                         f"min={df['radius_mm'].min():.2f}, "
                         f"max={df['radius_mm'].max():.2f}")

    if all_dfs:
        global_df = pd.concat(all_dfs, ignore_index=True)
        global_csv = out / "all_centerlines.csv"
        save_centerlines_csv(global_df, str(global_csv))
        log.info(f"\nAll centerlines merged → {global_csv}")
        log.info(f"Total centerline points: {len(global_df)}")

    log.info("\nPipeline complete.")
    return out


# --------------------------------------------------------------------------- #
#  CLI                                                                         #
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(
        description="Vascular centerline extraction pipeline (VMTK-based)"
    )
    p.add_argument("--input",   required=True,  help="Input .nii.gz segmentation")
    p.add_argument("--output",  required=True,  help="Output directory")
    p.add_argument("--labels",  nargs="+", type=int, default=None,
                   help="Label IDs to process (default: all)")
    p.add_argument("--smooth",  type=int, default=30,
                   help="Surface smoothing iterations (default: 30)")
    p.add_argument("--mode",    choices=["auto", "interactive"], default="auto",
                   help="Seed selection: 'auto' (no GUI) or 'interactive' (default: auto)")
    p.add_argument("--min-voxels", type=int, default=50,
                   help="Skip components smaller than this (default: 50)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        input_path    = args.input,
        output_dir    = args.output,
        target_labels = args.labels,
        smooth_iter   = args.smooth,
        mode          = args.mode,
        min_voxels    = args.min_voxels,
    )