import os
import random
import numpy as np
import torch
from tqdm import tqdm

from monai.data import DataLoader, Dataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    DivisiblePadd,           # ✅ FIX: pad to divisible by 16
    RandCropByPosNegLabeld,
    RandFlipd,
    RandAffined,
)
from monai.networks.nets import SegResNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric

from config import CFG
from data_utils import build_msd_lists


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_transforms():
    """
    Your dataset: one 4D NIfTI per case in imagesTr/<case>.nii.gz
      image: (X,Y,Z,C) -> EnsureChannelFirstd -> (C,X,Y,Z)
      label: (X,Y,Z)   -> EnsureChannelFirstd -> (1,X,Y,Z)

    ✅ FIX for SegResNet skip-connection mismatch:
    After crop/spacing sizes can become odd, so we pad to k=16 divisibility
    using DivisiblePadd(keys=["image","label"], k=16).
    """

    train_tfms = Compose(
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
                keys=["image"],
                a_min=0,
                a_max=3000,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),

            # ✅ crucial: make spatial dims divisible by 16 to avoid 1-voxel mismatch in decoder
            DivisiblePadd(keys=["image", "label"], k=16),

            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(CFG.roi_x, CFG.roi_y, CFG.roi_z),
                pos=1,
                neg=1,
                num_samples=CFG.samples_per_volume,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandAffined(
                keys=["image", "label"],
                prob=0.3,
                rotate_range=(0.1, 0.1, 0.1),
                shear_range=(0.05, 0.05, 0.05),
                translate_range=(5, 5, 5),
                scale_range=(0.1, 0.1, 0.1),
                mode=("bilinear", "nearest"),
                padding_mode="border",
            ),
        ]
    )

    val_tfms = Compose(
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
                keys=["image"],
                a_min=0,
                a_max=3000,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),

            # ✅ same padding in val to keep shapes safe
            DivisiblePadd(keys=["image", "label"], k=16),
        ]
    )

    return train_tfms, val_tfms


def main():
    set_seed(CFG.seed)

    # ✅ Mac M3: prefer MPS if available
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print("Device:", device)

    train_list, val_list, meta = build_msd_lists(CFG.data_dir)
    print("Train:", len(train_list), "Val:", len(val_list))
    print("Labels mapping:", meta.get("labels"))

    train_tfms, val_tfms = get_transforms()

    train_ds = Dataset(data=train_list, transform=train_tfms)
    val_ds = Dataset(data=val_list, transform=val_tfms)

    train_loader = DataLoader(
        train_ds,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=False,  # on MPS pin_memory isn't helpful
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=False,
    )

    model = SegResNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=4,  # background + {1,2,3}
        init_filters=16,
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
        dropout_prob=0.2,
    ).to(device)

    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=1e-5)

    os.makedirs(CFG.ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(CFG.ckpt_dir, CFG.ckpt_name)
    best_dice = -1.0

    for epoch in range(1, CFG.max_epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{CFG.max_epochs}"):
            x = batch["image"].to(device)  # (B,4,D,H,W)
            y = batch["label"].to(device)  # (B,1,D,H,W)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())

        epoch_loss /= max(1, len(train_loader))

        # validation
        model.eval()
        dice_metric.reset()
        with torch.no_grad():
            for batch in val_loader:
                x = batch["image"].to(device)
                y = batch["label"].to(device)
                logits = model(x)
                dice_metric(y_pred=logits, y=y)

        val_dice = float(dice_metric.aggregate().item())
        print(f"Epoch {epoch}: loss={epoch_loss:.4f}, val_dice={val_dice:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({"model": model.state_dict(), "best_dice": best_dice}, ckpt_path)
            print("Saved best checkpoint:", ckpt_path)

    print("Done. Best val dice:", best_dice)


if __name__ == "__main__":
    main()
