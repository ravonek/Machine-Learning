import numpy as np
from scipy import ndimage as ndi
from typing import Optional, Dict, Any

def compute_regions(mask: np.ndarray):
    # 0 background, 1 edema, 2 non-enhancing core, 3 enhancing tumor
    wt = (mask > 0)
    tc = (mask == 2) | (mask == 3)
    et = (mask == 3)
    return wt, tc, et

def voxel_volume_mm3(spacing_xyz=(1.0, 1.0, 1.0)) -> float:
    sx, sy, sz = spacing_xyz
    return float(sx * sy * sz)

def volume_ml(binary: np.ndarray, spacing_xyz=(1.0, 1.0, 1.0)) -> float:
    v_mm3 = float(binary.sum()) * voxel_volume_mm3(spacing_xyz)
    return float(v_mm3 / 1000.0)  # 1000 mm³ = 1 mL

def connected_components_count(binary: np.ndarray) -> int:
    _, n = ndi.label(binary)
    return int(n)

def laterality(binary: np.ndarray) -> str:
    # Rough laterality estimate in array space (W axis).
    if binary.sum() == 0:
        return "not detected"
    coords = np.argwhere(binary)
    x_mean = float(coords[:, 2].mean())
    w = binary.shape[2]
    return "left" if x_mean < (w / 2.0) else "right"

def localization_hint(binary: np.ndarray) -> str:
    # Very rough localization hint in array space (not anatomical ground truth).
    if binary.sum() == 0:
        return "not detected"
    coords = np.argwhere(binary)
    d, h, w = binary.shape
    y_mean = float(coords[:, 1].mean())
    z_mean = float(coords[:, 0].mean())
    sup_inf = "superior" if y_mean > (h / 2.0) else "inferior"
    ant_post = "posterior" if z_mean > (d / 2.0) else "anterior"
    return f"{sup_inf}-{ant_post} (approx)"

def make_report(mask: np.ndarray, spacing_xyz=(1.0, 1.0, 1.0), qa: Optional[Dict[str, Any]] = None) -> str:
    wt, tc, et = compute_regions(mask)

    wt_ml = volume_ml(wt, spacing_xyz)
    tc_ml = volume_ml(tc, spacing_xyz)
    et_ml = volume_ml(et, spacing_xyz)

    wt_n = connected_components_count(wt)
    side = laterality(wt)
    loc = localization_hint(wt)

    findings = []
    if wt.sum() == 0:
        findings.append("No tumor region detected in the predicted mask.")
    else:
        findings.append(f"Tumor region detected (laterality: {side}; location: {loc}).")
        findings.append(f"Estimated volumes: WT={wt_ml:.2f} mL, TC={tc_ml:.2f} mL, ET={et_ml:.2f} mL.")
        findings.append(f"Number of connected tumor components (WT): {wt_n}.")
        if et.sum() == 0 and tc.sum() > 0:
            findings.append("No enhancing tumor component (ET) detected; tumor core present without enhancement.")
        if et.sum() > 0:
            findings.append("Enhancing tumor component (ET) detected.")

    if qa:
        findings.append(f"Model QA: mean_entropy={qa.get('mean_entropy', None)}, mc_var_mean={qa.get('mc_var_mean', None)}.")
        for w in (qa.get("warnings", []) or []):
            findings.append(f"QA warning: {w}")

    impression = []
    if wt.sum() == 0:
        impression.append("Predicted segmentation shows no abnormal tumor region.")
    else:
        impression.append("Predicted segmentation indicates tumor presence; quantitative summary provided above.")
        impression.append("This output is for research/educational decision support and should be reviewed by a specialist.")

    return "FINDINGS:\n- " + "\n- ".join(findings) + "\n\nIMPRESSION:\n- " + "\n- ".join(impression)
