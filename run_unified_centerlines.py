
from __future__ import annotations

import argparse
from pathlib import Path

from centerlines import run_pipeline

DEFAULT_LABEL_MODES = {6: "single", 9: "multi", 10: "multi"}


def case_stem(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    return path.stem


def resolve_cta_for_case(stem: str, cta_dir: Path, cta_file: Path | None) -> Path | None:
    if cta_file is not None:
        return cta_file if cta_file.is_file() else None
    if not cta_dir.is_dir():
        return None
    for name in (f"{stem}.nii.gz", f"{stem}_ct.nii.gz"):
        hit = cta_dir / name
        if hit.is_file():
            return hit
    hits = sorted(cta_dir.glob(f"*{stem}*.nii.gz"))
    return hits[0] if hits else None


def main():
    p = argparse.ArgumentParser(description="Centerlines unificados por caso")
    p.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "segments_labelled_v1_unified",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "centerlines_unified_v1",
    )
    p.add_argument("--prune-voxels", type=int, default=20)
    p.add_argument("--smooth-sigma", type=float, default=2.0)
    p.add_argument("--min-branch-mm", type=float, default=5.0)
    p.add_argument("--min-voxels", type=int, default=50)
    p.add_argument("--save-skeleton", action="store_true")
    p.add_argument(
        "--bridge-max-mm",
        type=float,
        default=45.0,
        help="Distancia maxima (mm) para puentes entre componentes en labels 9 y 10; "
        "se relaja x1.5 y x2.25 si sigue desconectado.",
    )
    p.add_argument(
        "--cta-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "images",
        help="Carpeta con CTA .nii.gz alineados al mismo espacio que la segmentación (por nombre de caso).",
    )
    p.add_argument(
        "--cta",
        type=Path,
        default=None,
        help="Un solo archivo CTA para todos los casos (anula búsqueda por --cta-dir).",
    )
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(args.input_dir.glob("*.nii.gz"))
    if not files:
        raise SystemExit(f"No hay .nii.gz en {args.input_dir}")

    for nii in files:
        stem = case_stem(nii)
        out_case = args.output_dir / stem
        print(f"\n=== {nii.name} -> {out_case} ===")
        cta_path = resolve_cta_for_case(stem, args.cta_dir, args.cta)
        if cta_path:
            print(f"  CTA: {cta_path.name}")
        else:
            print("  CTA: (no encontrado — sin columna hu en centerlines)")
        run_pipeline(
            input_path=str(nii),
            output_dir=str(out_case),
            label_modes=DEFAULT_LABEL_MODES,
            prune_voxels=args.prune_voxels,
            smooth_sigma=args.smooth_sigma,
            min_branch_mm=args.min_branch_mm,
            min_voxels=args.min_voxels,
            save_skeleton=args.save_skeleton,
            bridge_max_mm=args.bridge_max_mm,
            cta_path=str(cta_path) if cta_path else None,
        )

    print(f"\nHecho. Resultados en: {args.output_dir}")


if __name__ == "__main__":
    main()
