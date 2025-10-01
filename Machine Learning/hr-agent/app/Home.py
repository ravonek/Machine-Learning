import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="HR Agent — Обзор", layout="wide")
st.title("HR Agent — Обзор")

@st.cache_data
def load_data():
    return pd.read_parquet("data/processed/employees.parquet")

if not Path("data/processed/employees.parquet").exists():
    st.info("Данные ещё не подготовлены. Нажмите кнопку ниже, чтобы выполнить ETL.")
    if st.button("Запустить ETL"):
        import subprocess, sys
        res = subprocess.run([sys.executable, "src/etl.py"], capture_output=True, text=True)
        st.code(res.stdout + "\n" + res.stderr)
else:
    df = load_data()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Сотрудников", len(df))

    if "Status" in df:
        resigned = (df["Status"].astype(str).str.lower() == "resigned").mean()
        c2.metric("% Resigned", f"{resigned*100:.1f}%")

    if "Performance_Rating" in df:
        c3.metric("Средний рейтинг", f"{df['Performance_Rating'].mean():.2f}")

    if "Hire_Date" in df:
        tenure = (pd.Timestamp.today(tz=None) - pd.to_datetime(df["Hire_Date"], errors="coerce")).dt.days / 365
        c4.metric("Средний стаж, лет", f"{tenure.mean():.1f}")

    st.subheader("Превью данных")
    st.dataframe(df.head(50))
    st.download_button(
        "Скачать превью (CSV)",
        df.head(100).to_csv(index=False).encode("utf-8"),
        file_name="preview.csv",
        mime="text/csv",
    )
