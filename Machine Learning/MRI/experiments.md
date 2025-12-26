# Experiments (Capstone)

Goal: run 2–3 meaningful variants and report **Dice** on **WT/TC/ET**.

## Baseline (Exp A) — SegResNet (implemented)
1) Train:
```bash
python train.py
```
2) Evaluate:
```bash
python evaluate.py
```
Outputs: `runs/eval_results.json`

Report in your slides:
- mean_dice_wt, mean_dice_tc, mean_dice_et
- 2–3 qualitative overlays

## Exp B — Patch size / augmentation
Why: ET/TC often benefit from more context and less aggressive transforms.

Suggested changes in `config.py`:
- ROI: 160×160×128 (if GPU allows) OR 96×96×96 (if RAM limited)
- samples_per_volume: 1
And in `train.py`:
- reduce RandAffined(prob=0.2) and rotate_range

Train + evaluate, save results:
- copy `runs/eval_results.json` to `runs/expB_eval_results.json`

## Exp C — Uncertainty + Post-processing
Already included:
- post-processing: keep largest WT component
- uncertainty: entropy + MC-dropout variance proxy

How to demonstrate:
- Run inference with QA JSON:
```bash
python infer.py <...4 modalities...> pred_mask.nii.gz pred_qa.json
```
- Show cases where QA warns (high uncertainty / tiny volume).

## What to include in write-up
- Dataset label convention (0/1/2/3)
- Evaluation on validation split
- Comparison across experiments
- Limitations (not clinical, affine not preserved in demo)
