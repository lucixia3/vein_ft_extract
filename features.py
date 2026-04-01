#!/usr/bin/env python3
"""
Extract per-label venous features from paired brain CTA and vein segmentation files.

Expected default project structure:
/project
├── scripts/
│   └── process_vein_features.py
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
    python process_vein_features.py

Process selected cases:
    python process_vein_features.py --case-id sub-stroke_0002 sub-stroke_0004

Force re-running TotalSegmentator for selected cases:
    python process_vein_features.py --case-id sub-stroke_0002 --force-totalseg
"""

from __future__ import annotations

import os
import sys


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
            "    python process_vein_features.py\n"
        )


check_conda_environment("vmtk_env")

import argparse
import gc
import subprocess
import traceback
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.morphology import skeletonize


LABEL_DICT: Dict[int, str] = {
    1: "VOG",
    2: "STS",
    3: "ICV",
    4: "RBVR",
    6: "SSS",
    7: "TransvSig-R",
    8: "TransvSig-L",
    9: "Cortical-R",
    10: "Cortical-L",
    11: "VCM-R",
    12: "VCM-L",
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
        help=(
            "One or more case IDs to process, for example: "
            "--case-id sub-stroke_0002 sub-stroke_0004. "
            "If omitted, all matched CTA/segmentation pairs are processed."
        ),
    )
    parser.add_argument(
        "--force-totalseg",
        action="store_true",
        help=(
            "Re-run TotalSegmentator even if an apparently complete output folder already exists. "
            "This mainly affects the regenerated TotalSegmentator masks, the saved ICV mask, "
            "and downstream ICV-based quantities such as volume_fraction_icv_percent."
        ),
    )
    return parser.parse_args()


def infer_case_id_from_cta_name(cta_name: str) -> str:
    suffix = "_ct.nii.gz"
    if not cta_name.endswith(suffix):
        raise ValueError(f"Unexpected CTA filename: {cta_name}")
    return cta_name[: -len(suffix)]


def get_case_pairs(
    cta_dir: Path,
    seg_dir: Path,
    selected_case_ids: set[str] | None = None,
) -> List[Tuple[str, Path, Path]]:
    pairs: List[Tuple[str, Path, Path]] = []
    for cta_path in sorted(cta_dir.glob("*_ct.nii.gz")):
        case_id = infer_case_id_from_cta_name(cta_path.name)
        if selected_case_ids is not None and case_id not in selected_case_ids:
            continue

        seg_path = seg_dir / f"{case_id}_seg.nii.gz"
        if not seg_path.exists():
            print(f"[WARNING] Missing segmentation for {cta_path.name}: {seg_path.name}")
            continue
        pairs.append((case_id, cta_path, seg_path))
    return pairs


def totalseg_outputs_look_complete(ts_dir: Path) -> bool:
    return ts_dir.exists() and all((ts_dir / name).exists() for name in TOTALSEG_COMPLETENESS_MARKERS)


def run_totalsegmentator(cta_path: Path, out_dir: Path, force: bool = False) -> bool:
    """Run TotalSegmentator if needed. Returns True if it was executed, False if skipped."""
    if not force and totalseg_outputs_look_complete(out_dir):
        print(f"  TotalSegmentator outputs already exist and look complete: {out_dir}")
        return False

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "TotalSegmentator",
        "-i",
        str(cta_path),
        "-o",
        str(out_dir),
        "--task",
        "brain_structures",
    ]
    print("  Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return True


def build_icv_mask(ts_dir: Path, vein_mask: np.ndarray) -> np.ndarray:
    reference_file = ts_dir / ICV_STRUCTURE_FILES[0]
    if not reference_file.exists():
        raise FileNotFoundError(f"Cannot build ICV mask because {reference_file} was not found.")

    reference_img = nib.load(str(reference_file))
    icv_mask = np.zeros(reference_img.shape, dtype=bool)

    for filename in ICV_STRUCTURE_FILES:
        path = ts_dir / filename
        if path.exists():
            icv_mask |= nib.load(str(path)).get_fdata() > 0
        else:
            print(f"  [WARNING] Missing TotalSegmentator structure: {filename}")

    icv_mask |= vein_mask
    return icv_mask


def skeleton_diameter_stats_mm(mask: np.ndarray, spacing: Tuple[float, float, float]) -> Tuple[float, float]:
    if np.count_nonzero(mask) == 0:
        return float("nan"), float("nan")

    distance_map = ndi.distance_transform_edt(mask, sampling=spacing)
    skeleton = skeletonize(mask)
    diameters = 2.0 * distance_map[skeleton]

    if diameters.size == 0:
        return float("nan"), float("nan")

    return float(np.max(diameters)), float(np.std(diameters))


def extract_case_features(
    case_id: str,
    cta_path: Path,
    seg_path: Path,
    ts_dir: Path,
    case_results_dir: Path,
) -> pd.DataFrame:
    cta_nii = nib.load(str(cta_path))
    seg_nii = nib.load(str(seg_path))

    cta = cta_nii.get_fdata()
    seg = seg_nii.get_fdata().astype(np.int16)

    if cta.shape != seg.shape:
        raise ValueError(f"Shape mismatch for {cta_path.name} and {seg_path.name}: {cta.shape} vs {seg.shape}")

    spacing = tuple(float(x) for x in cta_nii.header.get_zooms()[:3])
    voxel_volume_mm3 = float(np.prod(spacing))

    vein_mask = seg > 0
    total_vein_volume_mm3 = float(np.count_nonzero(vein_mask) * voxel_volume_mm3)

    icv_mask = build_icv_mask(ts_dir, vein_mask)
    icv_volume_mm3 = float(np.count_nonzero(icv_mask) * voxel_volume_mm3)

    case_results_dir.mkdir(parents=True, exist_ok=True)
    icv_mask_path = case_results_dir / f"icv_mask_{case_id}.nii.gz"
    nib.save(
        nib.Nifti1Image(icv_mask.astype(np.uint8), cta_nii.affine, cta_nii.header),
        str(icv_mask_path),
    )

    rows = []
    for label in sorted(int(v) for v in np.unique(seg) if v != 0):
        mask = seg == label
        values = cta[mask]
        n_components = int(ndi.label(mask)[1])
        volume_mm3 = float(np.count_nonzero(mask) * voxel_volume_mm3)
        max_diameter_mm, std_diameter_mm = skeleton_diameter_stats_mm(mask, spacing)

        rows.append(
            {
                "case": case_id,
                "label": label,
                "structure": LABEL_DICT.get(label, "Unknown"),
                "volume_mm3": volume_mm3,
                "total_vein_volume_mm3": total_vein_volume_mm3,
                "icv_volume_mm3": icv_volume_mm3,
                "n_connected_components": n_components,
                "mean_hu_abs": float(np.mean(values)),
                "std_hu_abs": float(np.std(values)),
                "max_diameter_mm": max_diameter_mm,
                "std_diameter_mm": std_diameter_mm,
            }
        )

    df = pd.DataFrame(rows).sort_values("label").reset_index(drop=True)

    sss_rows = df[df["label"] == 6]
    if not sss_rows.empty and float(sss_rows["mean_hu_abs"].iloc[0]) != 0.0:
        sss_mean_hu = float(sss_rows["mean_hu_abs"].iloc[0])
        df["mean_hu_ref_sss"] = df["mean_hu_abs"] / sss_mean_hu
        df["std_hu_ref_sss"] = df["std_hu_abs"] / sss_mean_hu
    else:
        print("  [WARNING] SSS (label 6) not found or SSS mean HU is zero. Using NaN for SSS-referenced HU features.")
        df["mean_hu_ref_sss"] = np.nan
        df["std_hu_ref_sss"] = np.nan

    if total_vein_volume_mm3 > 0:
        df["volume_fraction_all_veins_percent"] = 100.0 * df["volume_mm3"] / total_vein_volume_mm3
    else:
        df["volume_fraction_all_veins_percent"] = np.nan

    if icv_volume_mm3 > 0:
        df["volume_fraction_icv_percent"] = 100.0 * df["volume_mm3"] / icv_volume_mm3
    else:
        df["volume_fraction_icv_percent"] = np.nan

    return df[
        [
            "case",
            "label",
            "structure",
            "volume_mm3",
            "total_vein_volume_mm3",
            "icv_volume_mm3",
            "n_connected_components",
            "mean_hu_abs",
            "std_hu_abs",
            "mean_hu_ref_sss",
            "std_hu_ref_sss",
            "volume_fraction_all_veins_percent",
            "volume_fraction_icv_percent",
            "max_diameter_mm",
            "std_diameter_mm",
        ]
    ]


def save_failure_log(results_dir: Path, failure_rows: List[dict]) -> None:
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
    print(f"Results dir:   {results_dir}")

    if not cta_dir.exists():
        raise FileNotFoundError(f"CTA directory not found: {cta_dir}")
    if not seg_dir.exists():
        raise FileNotFoundError(f"Segmentation directory not found: {seg_dir}")

    selected_case_ids = set(args.case_id) if args.case_id is not None else None
    pairs = get_case_pairs(cta_dir, seg_dir, selected_case_ids)
    if not pairs:
        print("No valid CTA/segmentation pairs were found.")
        return 1

    print(f"Found {len(pairs)} case(s) to process.\n")

    success_frames: List[pd.DataFrame] = []
    failure_rows: List[dict] = []
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
            if "MemoryError" in error_text or "returned non-zero exit status" in error_text:
                short_reason = "TotalSegmentator failed, likely due to memory exhaustion or an internal TotalSegmentator error."
            else:
                short_reason = error_text.splitlines()[0] if error_text else exc.__class__.__name__

            failure_rows.append(
                {
                    "case": case_id,
                    "cta_file": str(cta_path),
                    "seg_file": str(seg_path),
                    "error_type": exc.__class__.__name__,
                    "reason": short_reason,
                    "details": error_text,
                }
            )
            print(f"  [ERROR] Failed to process {case_id}: {short_reason}\n")
            gc.collect()
            continue

        gc.collect()

    save_combined_csv(results_dir, success_frames)
    save_failure_log(results_dir, failure_rows)

    print("\nSummary")
    print("-------")
    print(f"Requested cases:                 {len(pairs)}")
    print(f"Successfully processed cases:   {n_success}")
    print(f"Failed cases:                   {n_failed}")
    print(f"TotalSegmentator skipped cases: {n_totalseg_skipped}")

    return 0 if n_success > 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except EnvironmentError as exc:
        print(exc)
        sys.exit(1)
