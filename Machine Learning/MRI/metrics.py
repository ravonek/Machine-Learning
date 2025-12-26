import numpy as np

def dice_coef(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    return float((2.0 * inter + eps) / (denom + eps))

def regions_from_mask(mask: np.ndarray):
    # BraTS/MSD typical labels: 0 bg, 1 edema, 2 non-enhancing core, 3 enhancing
    wt = mask > 0
    tc = (mask == 2) | (mask == 3)
    et = (mask == 3)
    return wt, tc, et
