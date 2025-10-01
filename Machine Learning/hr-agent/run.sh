#!/usr/bin/env bash
set -e
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python src/etl.py
python src/train_churn.py
python src/train_rating.py
streamlit run app/Home.py
./run.sh
