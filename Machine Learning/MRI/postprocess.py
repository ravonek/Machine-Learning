import numpy as np
from scipy import ndimage as ndi

def keep_largest_component_wt(mask: np.ndarray) -> np.ndarray:
    """
    Keep only the largest connected component of WT (mask>0),
    preserving per-class labels inside that component.
    Helps remove tiny false-positive islands.
    """
    wt = mask > 0
    if wt.sum() == 0:
        return mask
    labeled, n = ndi.label(wt)
    if n <= 1:
        return mask
    sizes = ndi.sum(wt, labeled, index=range(1, n+1))
    largest_idx = int(np.argmax(sizes)) + 1
    keep = labeled == largest_idx
    out = mask.copy()
    out[~keep] = 0
    return out
