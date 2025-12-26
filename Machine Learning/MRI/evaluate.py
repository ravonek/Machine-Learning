import json
import os
import numpy as np
import torch
from tqdm import tqdm

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureTyped,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    DivisiblePadd,  # ✅ FIX
)
from monai.data import Dataset, DataLoader
from monai.networks.nets import SegResNet

from config import CFG
from data_utils import build_msd_lists
from metrics import dice_coef, regions_from_mask
from postprocess import keep_largest_component_wt


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
        dropout_prob=0.0,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def main():
    device = get_device()
    print("Device:", device)

    ckpt = os.path.join(CFG.ckpt_dir, CFG.ckpt_name)
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}. Train first.")

    _, val_list, _ = build_msd_lists(CFG.data_dir)

    tfms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            EnsureTyped(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            ScaleIntensityRanged(
                keys=["image"], a_min=0, a_max=3000, b_min=0.0, b_max=1.0, clip=True
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),

            # ✅ critical: avoid 1-voxel mismatch in SegResNet decoder
            DivisiblePadd(keys=["image", "label"], k=16),
        ]
    )

    ds = Dataset(data=val_list, transform=tfms)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=CFG.num_workers, pin_memory=False)

    model = load_model(ckpt, device)

    per_case = []
    dices_wt, dices_tc, dices_et = [], [], []

    for batch in tqdm(dl, desc="Evaluating"):
        x = batch["image"].to(device)  # (1,4,D,H,W)
        y = batch["label"].cpu().numpy().squeeze(0)
        if y.ndim == 4:
            y = y.squeeze(0)
        y = y.astype(np.uint8)

        with torch.no_grad():
            logits = model(x)
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        pred = keep_largest_component_wt(pred)

        wt_p, tc_p, et_p = regions_from_mask(pred)
        wt_y, tc_y, et_y = regions_from_mask(y)

        d_wt = dice_coef(wt_p, wt_y)
        d_tc = dice_coef(tc_p, tc_y)
        d_et = dice_coef(et_p, et_y)

        dices_wt.append(d_wt)
        dices_tc.append(d_tc)
        dices_et.append(d_et)

        case_id = batch.get("case_id", ["unknown"])[0] if isinstance(batch.get("case_id"), list) else batch.get("case_id", "unknown")
        per_case.append({"case_id": case_id, "dice_wt": d_wt, "dice_tc": d_tc, "dice_et": d_et})

    summary = {
        "mean_dice_wt": float(np.mean(dices_wt)) if dices_wt else 0.0,
        "mean_dice_tc": float(np.mean(dices_tc)) if dices_tc else 0.0,
        "mean_dice_et": float(np.mean(dices_et)) if dices_et else 0.0,
        "n_val_cases": int(len(per_case)),
        "per_case": per_case,
        "postprocessing": "keep_largest_component_wt",
        "pad_divisible_k": 16,
    }

    os.makedirs("runs", exist_ok=True)
    out_path = os.path.join("runs", "eval_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved:", out_path)
    print(
        "Summary:",
        {k: summary[k] for k in ["mean_dice_wt", "mean_dice_tc", "mean_dice_et", "n_val_cases"]},
    )


if __name__ == "__main__":
    main()
