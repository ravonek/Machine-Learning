import os
import sys
import json
import numpy as np
import torch
import nibabel as nib

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureTyped,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    DivisiblePadd,   # ✅ FIX
)
from monai.data import Dataset, DataLoader
from monai.networks.nets import SegResNet

from config import CFG
from postprocess import keep_largest_component_wt
from uncertainty import softmax_entropy, mc_dropout_uncertainty


def get_device():
    return torch.device(
        "mps" if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )


def load_model(ckpt_path: str, device: torch.device):
    model = SegResNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=4,
        init_filters=16,
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
        dropout_prob=0.2,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def infer_one(img_4d_path: str, out_mask_path: str, out_qa_path: str, use_mc: bool = True):
    device = get_device()
    print("Device:", device)

    ckpt_path = os.path.join(CFG.ckpt_dir, CFG.ckpt_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Train first.")

    model = load_model(ckpt_path, device)

    tfms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),  # (X,Y,Z,C) -> (C,X,Y,Z)
            EnsureTyped(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
            ScaleIntensityRanged(
                keys=["image"], a_min=0, a_max=3000, b_min=0.0, b_max=1.0, clip=True
            ),
            CropForegroundd(keys=["image"], source_key="image"),
            DivisiblePadd(keys=["image"], k=16),  # ✅ avoid 1-voxel mismatch
        ]
    )

    ds = Dataset(data=[{"image": img_4d_path}], transform=tfms)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    batch = next(iter(dl))
    x = batch["image"].to(device)  # (1,4,D,H,W)

    with torch.no_grad():
        logits = model(x)
        ent = softmax_entropy(logits).squeeze(0).cpu().numpy()  # (D,H,W) or (H,W,D) depending, but used as array
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    mc_var_mean = None
    if use_mc and getattr(CFG, "mc_samples", 0) >= 2:
        _, var_p = mc_dropout_uncertainty(model, x, n_samples=int(CFG.mc_samples))
        mc_var_mean = float(var_p.mean().item())

    # postprocess
    pred = keep_largest_component_wt(pred)

    # QA metrics
    wt = pred > 0
    mean_ent = float(ent[wt].mean()) if wt.any() else float(np.mean(ent))
    wt_volume_ml = float(wt.sum()) / 1000.0  # rough (voxels -> "ml" proxy)

    warnings = []
    if mean_ent >= CFG.entropy_warn_threshold:
        warnings.append(f"High uncertainty (entropy): mean={mean_ent:.3f} >= {CFG.entropy_warn_threshold}.")
    if wt_volume_ml < CFG.min_wt_volume_ml:
        warnings.append(f"Very small predicted WT volume: {wt_volume_ml:.3f} mL < {CFG.min_wt_volume_ml}.")
    if mc_var_mean is not None and mc_var_mean >= CFG.mc_dropout_warn_threshold:
        warnings.append(f"High uncertainty (MC-dropout var): mean={mc_var_mean:.4f} >= {CFG.mc_dropout_warn_threshold}.")

    qa = {
        "case_path": img_4d_path,
        "mean_entropy": mean_ent,
        "wt_volume_ml_approx": wt_volume_ml,
        "mc_var_mean": mc_var_mean,
        "mc_samples": int(CFG.mc_samples) if use_mc else 0,
        "warnings": warnings,
    }

    # Save outputs (note: affine is identity here; for demo it's okay)
    nib.save(nib.Nifti1Image(pred.astype(np.uint8), affine=np.eye(4)), out_mask_path)
    with open(out_qa_path, "w", encoding="utf-8") as f:
        json.dump(qa, f, indent=2)

    print("Saved mask:", out_mask_path)
    print("Saved QA:", out_qa_path)
    if warnings:
        print("QA warnings:")
        for w in warnings:
            print("-", w)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python infer.py <img_4d.nii.gz> <out_mask.nii.gz> <out_qa.json> [--no-mc]")
        raise SystemExit(1)

    img_path = sys.argv[1]
    out_mask = sys.argv[2]
    out_qa = sys.argv[3] if len(sys.argv) >= 4 and not sys.argv[3].startswith("--") else "pred_qa.json"
    use_mc = "--no-mc" not in sys.argv

    infer_one(img_path, out_mask, out_qa, use_mc=use_mc)
