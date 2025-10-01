import streamlit as st
import pandas as pd

st.title("Поиск сотрудников")

@st.cache_data
def load_data():
    return pd.read_parquet("data/processed/employees.parquet")

df = load_data()

col1, col2, col3 = st.columns(3)
dept  = col1.multiselect("Department", sorted(df["Department"].dropna().unique().tolist()))
loc   = col2.multiselect("Location",   sorted(df["Location"].dropna().unique().tolist()))
title = col3.multiselect("Job Title",  sorted(df["Job_Title"].dropna().unique().tolist()))

mask = pd.Series(True, index=df.index)
if dept:  mask &= df["Department"].isin(dept)
if loc:   mask &= df["Location"].isin(loc)
if title: mask &= df["Job_Title"].isin(title)

res = df[mask].reset_index(drop=True)
st.dataframe(res)
st.download_button(
    "Скачать результат (CSV)",
    res.to_csv(index=False).encode("utf-8"),
    file_name="search_result.csv",
    mime="text/csv",
)
