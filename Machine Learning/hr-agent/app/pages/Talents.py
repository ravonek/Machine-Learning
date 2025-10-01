import streamlit as st
import pandas as pd
import joblib

st.title("Кадровый резерв (Talents)")

@st.cache_data
def load_data():
    return pd.read_parquet("data/processed/employees.parquet")

df = load_data()

col1, col2 = st.columns(2)
min_rating = col1.slider("Мин. рейтинг", 1.0, 5.0, 4.5, 0.1)
min_tenure = col2.slider("Мин. стаж (лет)", 0.0, 10.0, 3.0, 0.5)

preset = st.checkbox("Показать топ-20 талантов (пресет ТЗ)", value=False)
if preset:
    min_rating, min_tenure = 4.5, 3.0

# загрузка модели рейтинга с возможностью обучить
model = None
try:
    model = joblib.load("models/rating_rf.joblib")
except Exception:
    st.warning("Модель рейтинга ещё не обучена.")
    if st.button("Обучить модель (rating)"):
        import subprocess, sys
        res = subprocess.run([sys.executable, "src/train_rating.py"], capture_output=True, text=True)
        st.code(res.stdout + "\n" + res.stderr)
        st.success("Готово! Страница перезапустится.")
        st.rerun()

if model is None:
    st.stop()

# признаки для предсказаний
tmp = df[["Department","Job_Title","Location","Experience_Years","Performance_Rating","Hire_Date"]].copy()
tmp["tenure_years"] = (pd.Timestamp.today(tz=None) - pd.to_datetime(tmp["Hire_Date"], errors="coerce")).dt.days / 365
tmp = tmp.drop(columns=["Hire_Date"])

pred_rating = model.predict(tmp)

view = df[["Employee_ID","Full_Name","Department","Job_Title","Location","Performance_Rating","Hire_Date"]].copy()
view["tenure_years"] = (pd.Timestamp.today(tz=None) - pd.to_datetime(view["Hire_Date"], errors="coerce")).dt.days / 365
view["pred_rating"] = pred_rating

talents = (
    view[(view["Performance_Rating"] >= min_rating) & (view["tenure_years"] >= min_tenure)]
    .sort_values(["Performance_Rating", "tenure_years"], ascending=[False, False])
)

talents = talents.head(20) if preset else talents.head(200)

st.dataframe(talents)
st.download_button(
    "Скачать talents.csv",
    talents.to_csv(index=False).encode("utf-8"),
    file_name="talents.csv",
    mime="text/csv",
)
