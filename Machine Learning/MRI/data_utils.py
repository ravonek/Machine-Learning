import json
import os
from glob import glob
from typing import List, Dict, Tuple


def read_dataset_json(data_dir: str) -> Dict:
    path = os.path.join(data_dir, "dataset.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_msd_lists(data_dir: str) -> Tuple[List[Dict], List[Dict], Dict]:
    """
    Your dataset layout (single 4D NIfTI per case):
      imagesTr: <case>.nii.gz  (4D: X,Y,Z,C where C=4 modalities)
      labelsTr: <case>.nii.gz  (3D: X,Y,Z)
    """
    meta = read_dataset_json(data_dir)

    images_tr = os.path.join(data_dir, "imagesTr")
    labels_tr = os.path.join(data_dir, "labelsTr")

    label_files = sorted(glob(os.path.join(labels_tr, "*.nii*")))

    items: List[Dict] = []
    skipped = 0

    for lf in label_files:
        base = os.path.basename(lf).replace(".nii.gz", "").replace(".nii", "")

        img_gz = os.path.join(images_tr, f"{base}.nii.gz")
        img_nii = os.path.join(images_tr, f"{base}.nii")

        if os.path.exists(img_gz):
            img_path = img_gz
        elif os.path.exists(img_nii):
            img_path = img_nii
        else:
            skipped += 1
            continue

        items.append({"image": img_path, "label": lf, "case_id": base})

    print(f"Dataset cases: {len(label_files)} | usable: {len(items)} | skipped: {skipped}")

    n = len(items)
    n_train = int(n * 0.9)
    train = items[:n_train]
    val = items[n_train:]
    return train, val, meta
