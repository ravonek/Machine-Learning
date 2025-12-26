from dataclasses import dataclass


@dataclass
class TrainConfig:
    # Dataset root
    data_dir: str = "data/Task01_BrainTumour"

    # Repro
    seed: int = 42
    fold_split: float = 0.9  # train/val split ratio

    # Training (MacBook Air M3 friendly)
    max_epochs: int = 1
    batch_size: int = 1
    lr: float = 1e-4

    # macOS стабильность: DataLoader часто глючит/подвисает с workers>0
    num_workers: int = 0

    # Patch sampling (безопасно для M3)
    roi_x: int = 96
    roi_y: int = 96
    roi_z: int = 96
    samples_per_volume: int = 1

    # Output
    ckpt_dir: str = "checkpoints"
    ckpt_name: str = "segresnet_brats.pt"

    # Inference QA thresholds
    entropy_warn_threshold: float = 1.0   # выше -> более неопределенно
    min_wt_volume_ml: float = 0.10        # слишком маленький объём -> warn

    # MC Dropout (uncertainty) settings
    mc_samples: int = 6                   # 6 быстрее, 8 можно если терпимо
    mc_dropout_warn_threshold: float = 0.15


CFG = TrainConfig()
