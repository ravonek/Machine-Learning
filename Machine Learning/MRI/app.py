import json
import os
import tempfile
import numpy as np
import streamlit as st
import nibabel as nib
import matplotlib.pyplot as plt


# ----------------------------
# File loading (FIX: no BytesIO for nib.load)
# ----------------------------

@st.cache_data(show_spinner=False)
def load_nifti_uploaded(file_bytes: bytes, filename: str):
    """
    nibabel nib.load reliably expects a filesystem path.
    Streamlit gives us bytes, so we write to a temp file and load by path.
    """
    if filename.endswith(".nii.gz"):
        suffix = ".nii.gz"
    elif filename.endswith(".nii"):
        suffix = ".nii"
    else:
        suffix = ".nii.gz"

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        img = nib.load(tmp_path)
        data = img.get_fdata()
        affine = img.affine
        return data, affine
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def _safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def normalize_slice(x2d: np.ndarray):
    x = x2d.astype(np.float32)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=np.float32)
    x = x.copy()
    x[~finite] = 0
    p1, p99 = np.percentile(x, [1, 99])
    if p99 <= p1:
        p1, p99 = float(x.min()), float(x.max() + 1e-6)
    x = np.clip((x - p1) / (p99 - p1 + 1e-6), 0, 1)
    return x


def find_best_z_from_mask(mask3d: np.ndarray):
    wt = mask3d > 0
    if wt.ndim != 3:
        return 0
    counts = wt.sum(axis=(0, 1))  # along z
    if counts.max() == 0:
        return int(mask3d.shape[2] // 2)
    return int(np.argmax(counts))


def compute_volumes(mask3d: np.ndarray, voxel_volume_ml: float = 1.0 / 1000.0):
    """
    Proxy volume: assumes 1mm^3 voxels => 0.001 mL.
    (For demo this is OK; in a medical setting you'd use header spacing.)
    """
    wt = (mask3d > 0)
    tc = np.isin(mask3d, [2, 3])
    et = (mask3d == 3)
    return {
        "WT_ml": float(wt.sum()) * voxel_volume_ml,
        "TC_ml": float(tc.sum()) * voxel_volume_ml,
        "ET_ml": float(et.sum()) * voxel_volume_ml,
        "WT_vox": int(wt.sum()),
        "TC_vox": int(tc.sum()),
        "ET_vox": int(et.sum()),
    }


def centroid_location(mask3d: np.ndarray):
    wt = mask3d > 0
    if wt.sum() == 0:
        return None
    coords = np.argwhere(wt)  # (N,3) in (x,y,z)
    cx, cy, cz = coords.mean(axis=0)
    sx, sy, sz = mask3d.shape
    lat = "right" if cx > sx / 2 else "left"
    ap = "anterior" if cy > sy / 2 else "posterior"
    si = "superior" if cz > sz / 2 else "inferior"
    return {
        "centroid_xyz": (float(cx), float(cy), float(cz)),
        "laterality": lat,
        "location": f"{si}-{ap} (approx.)",
    }


def overlay_rgba(mask2d: np.ndarray, show_wt=True, show_tc=True, show_et=True):
    """
    Overlay:
      WT (mask>0)  -> cyan-ish
      TC (2 or 3)  -> yellow-ish
      ET (3)       -> red-ish
    """
    h, w = mask2d.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)

    if show_wt:
        wt = mask2d > 0
        rgba[wt, 0] = 0.0
        rgba[wt, 1] = 1.0
        rgba[wt, 2] = 1.0
        rgba[wt, 3] = 0.18

    if show_tc:
        tc = np.isin(mask2d, [2, 3])
        rgba[tc, 0] = 1.0
        rgba[tc, 1] = 1.0
        rgba[tc, 2] = 0.0
        rgba[tc, 3] = 0.25

    if show_et:
        et = mask2d == 3
        rgba[et, 0] = 1.0
        rgba[et, 1] = 0.0
        rgba[et, 2] = 0.0
        rgba[et, 3] = 0.35

    return rgba


def render_legend(show_wt, show_tc, show_et):
    items = []
    if show_wt:
        items.append("WT (mask>0)")
    if show_tc:
        items.append("TC (2 or 3)")
    if show_et:
        items.append("ET (3)")
    st.markdown("**Overlay legend:**")
    for it in items:
        st.markdown(f"- **{it}**")


# ----------------------------
# UI
# ----------------------------

st.set_page_config(page_title="Brain MRI Tumor Segmentation Demo", layout="centered")
st.title("Brain MRI Tumor Segmentation Demo (research/educational)")
st.caption("Upload an MRI (3D or 4D) and a predicted mask (+ optional QA json), or use the bundled synthetic sample.")

use_sample = st.toggle("Use bundled sample (no dataset needed)", value=False)

if "z" not in st.session_state:
    st.session_state["z"] = 0

# Uploaders
if not use_sample:
    mri_file = st.file_uploader("Upload MRI (.nii or .nii.gz) — can be 3D or 4D", type=["nii", "gz"])
    mask_file = st.file_uploader("Upload predicted mask (.nii or .nii.gz) — 3D labels {0,1,2,3}", type=["nii", "gz"])
    qa_file = st.file_uploader("Upload QA json (optional)", type=["json"])
else:
    mri_file = None
    mask_file = None
    qa_file = None

mri = None
mask = None
qa = None

# Sample mode
if use_sample:
    rng = np.random.default_rng(0)
    H, W, Z = 160, 160, 160
    mri = (rng.normal(0, 1, size=(H, W, Z)).astype(np.float32) * 0.25 + 0.5)
    mri = np.clip(mri, 0, 1)
    mask = np.zeros((H, W, Z), dtype=np.uint8)

    cx, cy, cz = 95, 70, 80
    rr = 18
    xx, yy, zz = np.meshgrid(np.arange(H), np.arange(W), np.arange(Z), indexing="ij")
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2
    mask[d2 < rr**2] = 1
    mask[d2 < (rr * 0.6) ** 2] = 2
    mask[d2 < (rr * 0.35) ** 2] = 3

    qa = {"mean_entropy": 0.55, "mc_var_mean": 0.05, "warnings": []}

else:
    if mri_file is not None:
        data, _aff = load_nifti_uploaded(mri_file.getvalue(), mri_file.name)
        mri = data

    if mask_file is not None:
        data, _aff = load_nifti_uploaded(mask_file.getvalue(), mask_file.name)
        if data.ndim == 4 and data.shape[-1] == 1:
            data = data[..., 0]
        mask = data.astype(np.uint8)

    if qa_file is not None:
        try:
            qa = json.load(qa_file)
        except Exception:
            qa = None

# 4D MRI handling
modality_idx = 0
mri3d = None

if mri is not None and mri.ndim == 4:
    # detect channel dim at last or first
    if mri.shape[-1] in (3, 4, 5):
        C = mri.shape[-1]
        modality_idx = st.selectbox("MRI is 4D. Select modality/channel", list(range(C)), index=0)
        mri3d = mri[..., modality_idx]
    elif mri.shape[0] in (3, 4, 5):
        C = mri.shape[0]
        modality_idx = st.selectbox("MRI is 4D. Select modality/channel", list(range(C)), index=0)
        mri3d = mri[modality_idx, ...]
    else:
        st.error(f"Unsupported 4D shape for MRI: {mri.shape}. Expected channels in first or last dim.")
        mri3d = None
elif mri is not None and mri.ndim == 3:
    mri3d = mri

# Shape warning
shape_warning = None
if (mri3d is not None) and (mask is not None) and (mri3d.shape != mask.shape):
    shape_warning = (
        f"⚠️ Shape mismatch: MRI {mri3d.shape} vs MASK {mask.shape}. "
        "Overlay may be incorrect. Use MRI preprocessed to mask-space (like *_preproc.nii.gz)."
    )

# Buttons
colA, colB, colC = st.columns([1, 1, 2])
with colA:
    if mask is not None and st.button("Auto-find best slice"):
        st.session_state["z"] = find_best_z_from_mask(mask)
with colB:
    if st.button("Center slice"):
        if mri3d is not None:
            st.session_state["z"] = int(mri3d.shape[2] // 2)
        elif mask is not None:
            st.session_state["z"] = int(mask.shape[2] // 2)

# Overlay toggles
st.subheader("Viewer")
show_overlay = st.toggle("Show overlay", value=True)
show_wt = st.checkbox("WT (mask>0)", value=True, disabled=not show_overlay)
show_tc = st.checkbox("TC (2 or 3)", value=True, disabled=not show_overlay)
show_et = st.checkbox("ET (3)", value=True, disabled=not show_overlay)

# Viewer
if mri3d is None:
    st.info("Upload MRI to view slices.")
else:
    Z = int(mri3d.shape[2])
    st.session_state["z"] = int(np.clip(st.session_state["z"], 0, Z - 1))
    z = st.slider("Slice index (z)", 0, Z - 1, st.session_state["z"], key="z")

    img2d = mri3d[:, :, z]
    img2d_n = normalize_slice(img2d)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img2d_n.T, origin="lower")
    ax.set_title(f"Slice z={z}")
    ax.set_axis_off()

    if mask is not None and show_overlay:
        if mask.ndim == 3 and (z < mask.shape[2]):
            m2d = mask[:, :, z]
            rgba = overlay_rgba(m2d.T, show_wt=show_wt, show_tc=show_tc, show_et=show_et)
            ax.imshow(rgba, origin="lower")
        else:
            st.warning("Mask loaded but cannot overlay on this slice index (shape mismatch or z out of range).")

    st.pyplot(fig)

    if show_overlay:
        render_legend(show_wt, show_tc, show_et)

    if shape_warning:
        st.warning(shape_warning)

# Report
st.subheader("Auto-generated report")

if mask is None:
    st.info("Upload a predicted mask to generate volumes and location summary.")
else:
    vols = compute_volumes(mask)
    loc = centroid_location(mask)

    mean_entropy = None
    mc_var_mean = None
    warnings = []

    if isinstance(qa, dict):
        mean_entropy = _safe_float(qa.get("mean_entropy"))
        mc_var_mean = _safe_float(qa.get("mc_var_mean"))
        if isinstance(qa.get("warnings"), list):
            warnings = [str(x) for x in qa.get("warnings")]

    lines = []
    lines.append("FINDINGS:")
    if loc is None:
        lines.append("- Tumor region: not detected (mask empty).")
    else:
        lines.append(f"- Tumor region detected (laterality: {loc['laterality']}; location: {loc['location']}).")
    lines.append(f"- Estimated volumes (proxy): WT={vols['WT_ml']:.2f} mL, TC={vols['TC_ml']:.2f} mL, ET={vols['ET_ml']:.2f} mL.")
    lines.append(f"- Voxels: WT={vols['WT_vox']}, TC={vols['TC_vox']}, ET={vols['ET_vox']}.")

    if (mean_entropy is not None) or (mc_var_mean is not None):
        lines.append(f"- Model QA: mean_entropy={mean_entropy}, mc_var_mean={mc_var_mean}.")
    for w in warnings:
        lines.append(f"- QA warning: {w}")

    lines.append("")
    lines.append("IMPRESSION:")
    if vols["WT_vox"] == 0:
        lines.append("- Predicted segmentation does not indicate tumor presence in this case.")
    else:
        lines.append("- Predicted segmentation indicates tumor presence; quantitative summary provided.")
    lines.append("- This output is for research/educational decision support and should be reviewed by qualified personnel.")

    st.code("\n".join(lines), language="text")

st.caption("Tip: If MRI is 4D, use the modality selector. For perfect overlay, MRI and mask must have identical 3D shape.")
