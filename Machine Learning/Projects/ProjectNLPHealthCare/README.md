# MedQuAD RAG Assistant (NLP Course Project)

This project implements a medical QA assistant using the MedQuAD dataset with a simple RAG (Retrieval-Augmented Generation) pipeline and a Streamlit demo.

## Dataset
- `data/medquad.csv` — 16412 rows, columns: question, answer, source, focus_area, question_len, answer_len

## Quickstart
```bash
# 1) Create venv
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Build FAISS index (artifacts/)
python src/prepare_data.py --csv data/medquad.csv --out artifacts

# 4) Run Streamlit demo
streamlit run src/app.py
```

## Project Structure
```
project_nlp_medquad/
├─ data/
│  └─ medquad.csv
├─ notebooks/
│  └─ slm-and-rag-process-automation.ipynb   # provided by user
├─ src/
│  ├─ prepare_data.py
│  └─ app.py
├─ artifacts/        # created after building the index
├─ requirements.txt
├─ .env.example
└─ README.md
```

## Rubric Alignment
- **Adaptation / Prompting / RAG (45%)**: RAG pipeline with embedding search (FAISS) and answer formatting.
- **Business & EDA (5%)**: Educational medical assistant; includes basic EDA and focus-area distribution.
- **Deployment (10%)**: Streamlit app for live demo.
- **Originality (20%)**: Medical QA assistant on real dataset (MedQuAD).
- **Progress Report (5%)**: Topic, dataset, plan and metrics suggested below.
- **Presentation (15%)**: Demo-ready app, tables, charts (add more EDA visuals as needed).
- **Bonus (+5%)**: PEP8-compliant code.

## Suggested Metrics
- **Retrieval**: Recall@k / MRR using question→question+answer proximity as proxy (or keyword overlaps).
- **Answering**: ROUGE-L between retrieved answers and ground truth (optional).

## Disclaimers
- The assistant is for educational purposes only and does not substitute professional medical advice.
