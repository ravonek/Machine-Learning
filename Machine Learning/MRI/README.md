# Brain MRI Tumor Segmentation + Auto-Report (Capstone-Ready)

This is a **capstone-ready** project skeleton:

- **CV/ML:** 3D brain tumor segmentation (MONAI SegResNet)
- **NLP:** structured report generation (“Findings/Impression”) from predicted mask
- **Originality:** post-processing + uncertainty/auto-QA (entropy + MC-dropout variance proxy)
- **EDA:** notebook `notebooks/EDA.ipynb`
- **Experiments:** `experiments.md` + `evaluate.py` (Dice WT/TC/ET)
- **Deployment:** Streamlit demo + Docker

> Educational/research only. Not for clinical diagnosis.

---

## 1) Quick start (no dataset needed)
The Streamlit UI works instantly with the bundled **synthetic sample**:

```bash
pip install -r requirements.txt
streamlit run app.py
```

In the app, keep **“Use bundled sample”** enabled.

---

## 2) Setup (recommended)
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 3) Dataset (MSD Task01_BrainTumour)
Extract into:

```
data/Task01_BrainTumour/
  dataset.json
  imagesTr/
  labelsTr/
  imagesTs/
```

Expected naming:
- `imagesTr/<case>_0000.nii.gz ... <case>_0003.nii.gz`
- `labelsTr/<case>.nii.gz`

---

## 4) Train
```bash
python train.py
```

Best checkpoint saved to:
`checkpoints/segresnet_brats.pt`

---

## 5) Evaluate (Dice WT/TC/ET)
```bash
python evaluate.py
```

Outputs:
- `runs/eval_results.json`

---

## 6) Inference (one case) + QA JSON
```bash
python infer.py \
  <img_0000.nii.gz> <img_0001.nii.gz> <img_0002.nii.gz> <img_0003.nii.gz> \
  pred_mask.nii.gz pred_qa.json
```

Disable MC-dropout:
```bash
python infer.py <...> pred_mask.nii.gz pred_qa.json --no-mc
```

---

## 7) Run Streamlit Demo
```bash
streamlit run app.py
```

Upload:
- one modality (e.g., `_0000.nii.gz`)
- predicted mask (`pred_mask.nii.gz`)
- optional QA json (`pred_qa.json`)

---

## 8) Docker (one-command)
```bash
docker build -t mri-capstone .
docker run -p 8501:8501 mri-capstone
```

or:
```bash
docker-compose up --build
```

---

## 9) EDA Notebook
Open:
- `notebooks/EDA.ipynb`

---

## 10) Capstone checklist
See:
- `CAPSTONE_CHECKLIST.md`
