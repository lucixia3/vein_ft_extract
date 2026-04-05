

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize


def skeleton_diameter_stats_mm(
    mask: np.ndarray, spacing: Tuple[float, float, float]
) -> Tuple[float, float]:
    if np.count_nonzero(mask) == 0:
        return float("nan"), float("nan")
    distance_map = ndi.distance_transform_edt(mask, sampling=spacing)
    skeleton = skeletonize(mask.astype(bool))
    diameters = 2.0 * distance_map[skeleton]
    if diameters.size == 0:
        return float("nan"), float("nan")
    return float(np.max(diameters)), float(np.std(diameters))


def skeleton_extended_diameter_stats_mm(
    mask: np.ndarray, spacing: Tuple[float, float, float]
) -> Dict[str, Any]:
  
    out: Dict[str, Any] = {
        "max_diameter_mm": float("nan"),
        "min_diameter_mm": float("nan"),
        "mean_diameter_mm": float("nan"),
        "std_diameter_mm": float("nan"),
        "median_diameter_mm": float("nan"),
        "p10_diameter_mm": float("nan"),
        "p90_diameter_mm": float("nan"),
        "n_skeleton_voxels": 0,
    }
    if np.count_nonzero(mask) == 0:
        return out
    distance_map = ndi.distance_transform_edt(mask, sampling=spacing)
    skeleton = skeletonize(mask.astype(bool))
    d = 2.0 * distance_map[skeleton]
    if d.size == 0:
        return out
    out["max_diameter_mm"] = float(np.max(d))
    out["min_diameter_mm"] = float(np.min(d))
    out["mean_diameter_mm"] = float(np.mean(d))
    out["std_diameter_mm"] = float(np.std(d))
    out["median_diameter_mm"] = float(np.median(d))
    out["p10_diameter_mm"] = float(np.percentile(d, 10))
    out["p90_diameter_mm"] = float(np.percentile(d, 90))
    out["n_skeleton_voxels"] = int(d.size)
    return out
