#!/usr/bin/env python3
"""
Extract per-label venous features from paired brain CTA and vein segmentation files.

Expected default project structure:
/project
├── scripts/
│   └── features.py
├── cases/
│   ├── cta/
│   │   ├── sub-stroke_0002_ct.nii.gz
│   │   └── ...
│   └── segments/
│       ├── sub-stroke_0002_seg.nii.gz
│       └── ...
└── results/
    ├── TotalSegmentator/
│   │   └── <case_id>/
│   ├── <case_id>/
│   │   ├── icv_mask_<case_id>.nii.gz
│   │   └── <case_id>.csv
│   ├── vein_features_all_cases.csv
│   └── failed_cases.csv

Examples
--------
Process all matched cases:
    python features.py

Process selected cases:
    python features.py --case-id sub-stroke_0002 sub-stroke_0004

Process a range of cases:
    python features.py --case-range 20 25

Force re-running TotalSegmentator for selected cases:
    python features.py --case-id sub-stroke_0002 --force-totalseg
"""

from __future__ import annotations

import argparse
import gc
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


def check_conda_environment(required_env: str = "vmtk_env") -> None:
    """Stop early with a clear message if the required conda env is not active."""
    current_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    if current_env != required_env:
        raise EnvironmentError(
            "\n"
            f"[ERROR] This script must be run inside the conda environment '{required_env}'.\n"
            f"Current environment: {current_env or 'none'}.\n\n"
            "Please activate the environment manually and run the script again:\n"
            f"    conda activate {required_env}\n"
            "    python features.py\n"
        )


check_conda_environment()

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

# -----------------------------------------------------------------------------
# Configuration Constants
# -----------------------------------------------------------------------------
LABEL_DICT: Dict[int, str] = {
    1: "VOG",
    2: "STS",
    3: "ICV",
    4: "RBVR",
    5: "SSS",
    8: "TransvSig-L",
    9: "TransvSig-R",
    10: "Cortical-L",
    11: "Cortical-R",
}

ICV_STRUCTURE_FILES: List[str] = [
    "brainstem.nii.gz",
    "venous_sinuses.nii.gz",
    "caudate_nucleus.nii.gz",
    "cerebellum.nii.gz",
    "subarachnoid_space.nii.gz",
    "septum_pellucidum.nii.gz",
    "lentiform_nucleus.nii.gz",
    "insular_cortex.nii.gz",
    "internal_capsule.nii.gz",
    "ventricle.nii.gz",
    "central_sulcus.nii.gz",
    "frontal_lobe.nii.gz",
    "parietal_lobe.nii.gz",
    "occipital_lobe.nii.gz",
    "temporal_lobe.nii.gz",
    "thalamus.nii.gz",
]

TOTALSEG_COMPLETENESS_MARKERS: List[str] = [
    "brainstem.nii.gz",
    "venous_sinuses.nii.gz",
    "subarachnoid_space.nii.gz",
    "ventricle.nii.gz",
    "thalamus.nii.gz",
]

# -----------------------------------------------------------------------------
# Core Functions
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Root directory of the project. Default: parent of the scripts folder.",
    )
    parser.add_argument(
        "--cta-dir",
        type=Path,
        default=None,
        help="CTA directory. Default: <project-root>/cases/cta",
    )
    parser.add_argument(
        "--seg-dir",
        type=Path,
        default=None,
        help="Vein segmentation directory. Default: <project-root>/cases/segments",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Results directory. Default: <project-root>/results",
    )
    parser.add_argument(
        "--case-id",
        type=str,
        nargs="+",
        default=None,
        help="One or more case IDs to process. Omit to process all matched pairs.",
    )
    parser.add_argument(
        "--case-range",
        type=int,
        nargs=2,
        metavar=('START', 'END'),
        help="Process a range of cases by their integer IDs (inclusive). Example: --case-range 20 25",
    )
    parser.add_argument(
        "--case-prefix",
        type=str,
        default="sub-stroke_",
        help="Prefix for generated case IDs when using --case-range. Default: 'sub-stroke_'",
    )
    parser.add_argument(
        "--case-pad",
        type=int,
        default=4,
        help="Zero-padding length for generated case IDs when using --case-range. Default: 4",
    )
    parser.add_argument(
        "--force-totalseg",
        action="store_true",
        help="Re-run TotalSegmentator even if complete output already exists.",
    )
    return parser.parse_args()


def get_case_pairs(
    cta_dir: Path,
    seg_dir: Path,
    selected_case_ids: Optional[Set[str]] = None,
) -> List[Tuple[str, Path, Path]]:
    """Match CTA files with corresponding segmentation files."""
    pairs: List[Tuple[str, Path, Path]] = []
    for cta_path in sorted(cta_dir.glob("*_ct.nii.gz")):
        case_id = cta_path.name.replace("_ct.nii.gz", "")
        
        if selected_case_ids and case_id not in selected_case_ids:
            continue

        seg_path = seg_dir / f"{case_id}_seg.nii.gz"
        if not seg_path.exists():
            print(f"[WARNING] Missing segmentation for {case_id}: {seg_path.name}")
            continue
            
        pairs.append((case_id, cta_path, seg_path))
    return pairs


def run_totalsegmentator(cta_path: Path, out_dir: Path, force: bool = False) -> bool:
    """Run TotalSegmentator if outputs are incomplete or force flag is used."""
    is_complete = out_dir.exists() and all((out_dir / name).exists() for name in TOTALSEG_COMPLETENESS_MARKERS)
    
    if not force and is_complete:
        print(f"  TotalSegmentator outputs already exist and look complete: {out_dir}")
        return False

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["TotalSegmentator", "-i", str(cta_path), "-o", str(out_dir), "--task", "brain_structures"]
    
    print("  Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return True


def build_icv_mask(ts_dir: Path, vein_mask: np.ndarray) -> np.ndarray:
    """Combine specific TotalSegmentator outputs and the vein mask to form the ICV mask."""
    reference_file = ts_dir / ICV_STRUCTURE_FILES[0]
    if not reference_file.exists():
        raise FileNotFoundError(f"Cannot build ICV mask because {reference_file} was not found.")

    icv_mask = np.array(vein_mask, dtype=bool)

    for filename in ICV_STRUCTURE_FILES:
        path = ts_dir / filename
        if path.exists():
            icv_mask |= nib.load(str(path)).get_fdata() > 0
        else:
            print(f"  [WARNING] Missing TotalSegmentator structure: {filename}")

    return icv_mask


def skeleton_diameter_stats_mm(mask: np.ndarray, spacing: Tuple[float, ...]) -> Tuple[float, float, float, float]:
    """Calculate max, mean, median, and std of diameters using Euclidean distance transform, robust to spurs."""
    if not np.any(mask):
        return float("nan"), float("nan"), float("nan"), float("nan")

    distance_map = ndi.distance_transform_edt(mask, sampling=spacing)
    skeleton = skeletonize(mask)
    diameters = 2.0 * distance_map[skeleton]

    if diameters.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")

    # Filter out near-zero values (likely skeleton spurs touching the surface)
    min_valid_diameter = min(spacing)
    valid_diameters = diameters[diameters > min_valid_diameter]

    if valid_diameters.size == 0:
        valid_diameters = diameters # Fallback if the whole vein is tiny

    return (
        float(np.max(valid_diameters)),
        float(np.mean(valid_diameters)),
        float(np.median(valid_diameters)),
        float(np.std(valid_diameters))
    )


def extract_case_features(
    case_id: str,
    cta_path: Path,
    seg_path: Path,
    ts_dir: Path,
    case_results_dir: Path,
) -> pd.DataFrame:
    """Extract features for all present venous labels and calculate volume/density stats."""
    cta_nii = nib.load(str(cta_path))
    seg_nii = nib.load(str(seg_path))

    cta = cta_nii.get_fdata()
    seg = seg_nii.get_fdata().astype(np.int16)

    if cta.shape != seg.shape:
        raise ValueError(f"Shape mismatch for {case_id}: CTA {cta.shape} vs Seg {seg.shape}")

    spacing = tuple(float(x) for x in cta_nii.header.get_zooms()[:3])
    voxel_volume_mm3 = float(np.prod(spacing))

    vein_mask = seg > 0
    total_vein_volume_mm3 = float(np.count_nonzero(vein_mask) * voxel_volume_mm3)

    # Pre-calculate bounding boxes for all present labels to drastically reduce array sizes
    bounding_boxes = ndi.find_objects(seg)
    present_labels = [int(v) for v in np.unique(seg) if v != 0]
    rows: List[Dict[str, Any]] = []

    for label in present_labels:
        mask = seg == label
        values = cta[mask]
        volume_mm3 = float(np.count_nonzero(mask) * voxel_volume_mm3)
        n_components = int(ndi.label(mask)[1])
        
        # Bounding box extraction with a 1-voxel padding safeguard
        # Padding ensures distance_transform_edt calculates correctly at the boundary limits
        slices = bounding_boxes[label - 1]
        if slices is not None:
            padded_slices = tuple(
                slice(max(0, s.start - 1), min(dim, s.stop + 1))
                for s, dim in zip(slices, mask.shape)
            )
            cropped_mask = mask[padded_slices]
        else:
            cropped_mask = mask

        max_diameter_mm, mean_diameter_mm, median_diameter_mm, std_diameter_mm = skeleton_diameter_stats_mm(cropped_mask, spacing)

        rows.append({
            "case": case_id,
            "label": label,
            "structure": LABEL_DICT.get(label, "Unknown"),
            "volume_mm3": volume_mm3,
            "total_vein_volume_mm3": total_vein_volume_mm3,
            "n_connected_components": n_components,
            "mean_hu_abs": float(np.mean(values)),
            "median_hu_abs": float(np.median(values)),
            "std_hu_abs": float(np.std(values)),
            "p05_hu_abs": float(np.percentile(values, 5)),
            "p95_hu_abs": float(np.percentile(values, 95)),
            "max_diameter_mm": max_diameter_mm,
            "mean_diameter_mm": mean_diameter_mm,
            "median_diameter_mm": median_diameter_mm,
            "std_diameter_mm": std_diameter_mm,
        })

    # Clear heavy full-volume arrays out of memory early
    del cta
    del seg
    gc.collect()

    # Generate ICV mask
    icv_mask = build_icv_mask(ts_dir, vein_mask)
    icv_volume_mm3 = float(np.count_nonzero(icv_mask) * voxel_volume_mm3)

    # Save ICV mask
    case_results_dir.mkdir(parents=True, exist_ok=True)
    icv_mask_path = case_results_dir / f"icv_mask_{case_id}.nii.gz"
    nib.save(
        nib.Nifti1Image(icv_mask.astype(np.uint8), cta_nii.affine, cta_nii.header),
        str(icv_mask_path),
    )
    
    del icv_mask
    gc.collect()

    # Retrieve SSS mean HU for reference calculations (label 5)
    sss_mean_hu = next((r["mean_hu_abs"] for r in rows if r["label"] == 5), 0.0)
    if sss_mean_hu == 0.0:
        print("  [WARNING] SSS (label 5) not found or SSS mean HU is zero. Using NaN for SSS-referenced HU features.")

    # Populate final referenced calculations
    for row in rows:
        row["icv_volume_mm3"] = icv_volume_mm3
        
        if sss_mean_hu != 0.0:
            row["mean_hu_ref_sss"] = row["mean_hu_abs"] / sss_mean_hu
            row["std_hu_ref_sss"] = row["std_hu_abs"] / sss_mean_hu
        else:
            row["mean_hu_ref_sss"] = np.nan
            row["std_hu_ref_sss"] = np.nan

        row["volume_fraction_all_veins_percent"] = (
            100.0 * row["volume_mm3"] / total_vein_volume_mm3 if total_vein_volume_mm3 > 0 else np.nan
        )
        row["volume_fraction_icv_percent"] = (
            100.0 * row["volume_mm3"] / icv_volume_mm3 if icv_volume_mm3 > 0 else np.nan
        )

    df = pd.DataFrame(rows).sort_values("label").reset_index(drop=True)
    
    return df[[
        "case", "label", "structure", "volume_mm3", "total_vein_volume_mm3",
        "icv_volume_mm3", "n_connected_components", 
        "mean_hu_abs", "median_hu_abs", "std_hu_abs", "p05_hu_abs", "p95_hu_abs",
        "mean_hu_ref_sss", "std_hu_ref_sss", "volume_fraction_all_veins_percent",
        "volume_fraction_icv_percent", 
        "max_diameter_mm", "mean_diameter_mm", "median_diameter_mm", "std_diameter_mm",
    ]]


def save_failure_log(results_dir: Path, failure_rows: List[Dict[str, str]]) -> None:
    failure_path = results_dir / "failed_cases.csv"
    if failure_rows:
        pd.DataFrame(failure_rows).to_csv(failure_path, index=False)
        print(f"Saved failure log: {failure_path}")
    else:
        if failure_path.exists():
            failure_path.unlink()
        print("No failures. No failure log written.")


def save_combined_csv(results_dir: Path, dataframes: Sequence[pd.DataFrame]) -> None:
    combined_path = results_dir / "vein_features_all_cases.csv"
    if dataframes:
        pd.concat(dataframes, ignore_index=True).to_csv(combined_path, index=False)
        print(f"Saved combined CSV: {combined_path}")
    else:
        print("No successful cases. Combined CSV was not created.")


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
def main() -> int:
    args = parse_args()

    project_root = args.project_root.resolve()
    cta_dir = (args.cta_dir or (project_root / "cases" / "cta")).resolve()
    seg_dir = (args.seg_dir or (project_root / "cases" / "segments")).resolve()
    results_dir = (args.results_dir or (project_root / "results")).resolve()
    totalseg_root = results_dir / "TotalSegmentator"

    results_dir.mkdir(parents=True, exist_ok=True)
    totalseg_root.mkdir(parents=True, exist_ok=True)

    print(f"Project root: {project_root}")
    print(f"CTA dir:       {cta_dir}")
    print(f"SEG dir:       {seg_dir}")
    print(f"Results dir:   {results_dir}\n")

    if not cta_dir.exists():
        raise FileNotFoundError(f"CTA directory not found: {cta_dir}")
    if not seg_dir.exists():
        raise FileNotFoundError(f"Segmentation directory not found: {seg_dir}")

    # Initialize an empty set to gather all targeted case IDs
    selected_case_ids = set(args.case_id) if args.case_id is not None else set()

    # Generate the range if provided and add to the set
    if args.case_range is not None:
        start, end = args.case_range
        for i in range(start, end + 1):
            case_id = f"{args.case_prefix}{i:0{args.case_pad}d}"
            selected_case_ids.add(case_id)

    # Revert to None if the set is empty (meaning "process all cases")
    if not selected_case_ids:
        selected_case_ids = None

    pairs = get_case_pairs(cta_dir, seg_dir, selected_case_ids)
    
    if not pairs:
        print("No valid CTA/segmentation pairs were found.")
        return 1

    print(f"Found {len(pairs)} case(s) to process.\n")

    success_frames: List[pd.DataFrame] = []
    failure_rows: List[Dict[str, str]] = []
    n_totalseg_skipped = 0
    n_success = 0
    n_failed = 0

    for case_id, cta_path, seg_path in pairs:
        print(f"Processing {case_id}")
        print(f"  CTA: {cta_path.name}")
        print(f"  SEG: {seg_path.name}")

        case_results_dir = results_dir / case_id
        ts_dir = totalseg_root / case_id

        try:
            ran_totalseg = run_totalsegmentator(cta_path, ts_dir, force=args.force_totalseg)
            if not ran_totalseg:
                n_totalseg_skipped += 1

            df = extract_case_features(case_id, cta_path, seg_path, ts_dir, case_results_dir)
            success_frames.append(df)
            n_success += 1

            csv_path = case_results_dir / f"{case_id}.csv"
            df.to_csv(csv_path, index=False)
            
            print(f"  Saved per-case CSV: {csv_path}")
            print(f"  Saved ICV mask:     {case_results_dir / f'icv_mask_{case_id}.nii.gz'}\n")

        except Exception as exc:  # noqa: BLE001
            n_failed += 1
            error_text = str(exc)
            
            if isinstance(exc, (MemoryError, subprocess.CalledProcessError)):
                short_reason = "TotalSegmentator failed, likely due to memory exhaustion or an internal TotalSegmentator error."
            else:
                short_reason = error_text.splitlines()[0] if error_text else exc.__class__.__name__

            failure_rows.append({
                "case": case_id,
                "cta_file": str(cta_path),
                "seg_file": str(seg_path),
                "error_type": exc.__class__.__name__,
                "reason": short_reason,
                "details": error_text,
            })
            print(f"  [ERROR] Failed to process {case_id}: {short_reason}\n")
            gc.collect()
            continue

        gc.collect()

    save_combined_csv(results_dir, success_frames)
    save_failure_log(results_dir, failure_rows)

    print("\nSummary")
    print("-------")
    print(f"Requested cases:                 {len(pairs)}")
    print(f"Successfully processed cases:    {n_success}")
    print(f"Failed cases:                    {n_failed}")
    print(f"TotalSegmentator skipped cases:  {n_totalseg_skipped}")

    return 0 if n_success > 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except EnvironmentError as exc:
        print(exc)
        sys.exit(1)
