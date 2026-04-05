"""

Nomrs es conservan lrs originales 6, 9, 10, 11 y 12. La salida tiene tres etiquetas:
  - 6  : voxels con etiqueta original 6 (centerline en modo single)
  - 9  : unión de originales 9 y 11 (modo multi)
  - 10 : unión de originales 10 y 12 (modo multi)


"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np

MAPPING = {
    "output": {
        "6": {"source_labels": [6], "centerline_mode": "single"},
        "9": {"source_labels": [9, 11], "centerline_mode": "multi"},
        "10": {"source_labels": [10, 12], "centerline_mode": "multi"},
    }
}


def unify_volume(data: np.ndarray) -> np.ndarray:
    d = np.round(data).astype(np.int16)
    out = np.zeros_like(d, dtype=np.int16)
    out[d == 6] = 6
    out[np.isin(d, (9, 11))] = 9
    out[np.isin(d, (10, 12))] = 10
    return out


def main():
    p = argparse.ArgumentParser(description="Unifica etiquetas 6 / 9+11 / 10+12")
    p.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "segments_labelled_v1",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "segments_labelled_v1_unified",
    )
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "label_mapping.json"
    manifest_path.write_text(json.dumps(MAPPING, indent=2), encoding="utf-8")

    files = sorted(args.input_dir.glob("*.nii.gz"))
    if not files:
        raise SystemExit(f"No hay .nii.gz en {args.input_dir}")

    for src in files:
        nii = nib.load(str(src))
        data = nii.get_fdata()
        unified = unify_volume(data)
        out_nii = nib.Nifti1Image(unified, nii.affine, nii.header)
        dst = args.output_dir / src.name
        nib.save(out_nii, str(dst))
        labels = [int(x) for x in np.unique(unified) if x > 0]
        print(f"{src.name} -> {dst.name}  labels: {labels}")

    print(f"\nManifest: {manifest_path}")
    print(f"Procesados: {len(files)} volumenes.")


if __name__ == "__main__":
    main()
