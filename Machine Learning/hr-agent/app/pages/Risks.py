import streamlit as st
import pandas as pd
import joblib
import io
import matplotlib.pyplot as plt

st.title("Риски увольнения (at-risk)")

@st.cache_data
def load_data():
    return pd.read_parquet("data/processed/employees.parquet")

df = load_data()

# --- инициализация session_state ---
if "risk_threshold" not in st.session_state:
    st.session_state["risk_threshold"] = 0.70
if "quick_report" not in st.session_state:
    st.session_state["quick_report"] = False

col_t, col_btn = st.columns([3, 1])

# если флаг quick_report активен – ставим 0.70
default_threshold = 0.70 if st.session_state["quick_report"] else st.session_state["risk_threshold"]

threshold = col_t.slider(
    "Порог вероятности увольнения",
    0.50, 0.95,
    default_threshold, 0.01,
    key="risk_threshold"
)

with col_btn:
    if st.button("Быстрый отчёт: ≥70%"):
        st.session_state["quick_report"] = True
        if hasattr(st, "rerun"):
            st.rerun()
        else:
            st.experimental_rerun()

# --- загрузка модели ---
model = None
try:
    model = joblib.load("models/churn_rf.joblib")
except Exception:
    st.warning("Модель текучести ещё не обучена.")
    if st.button("Обучить модель (churn)"):
        import subprocess, sys
        res = subprocess.run([sys.executable, "src/train_churn.py"], capture_output=True, text=True)
        st.code(res.stdout + "\n" + res.stderr)
        st.success("Готово! Страница перезапустится.")
        if hasattr(st, "rerun"):
            st.rerun()
        else:
            st.experimental_rerun()

if model is None:
    st.stop()

# --- подготовка признаков ---
tmp = df[["Department","Job_Title","Location","Experience_Years","Performance_Rating","Hire_Date"]].copy()
tmp["tenure_years"] = (pd.Timestamp.today(tz=None) - pd.to_datetime(tmp["Hire_Date"], errors="coerce")).dt.days / 365
tmp = tmp.drop(columns=["Hire_Date"])

# --- предсказания ---
proba = model.predict_proba(tmp)[:, 1]
out = df[["Employee_ID","Full_Name","Department","Job_Title","Location","Performance_Rating"]].copy()
out["churn_proba"] = proba

at_risk = out[out["churn_proba"] >= threshold].sort_values("churn_proba", ascending=False)

# --- метрики ---
c1, c2 = st.columns(2)
c1.metric("Сотрудников с риском", len(at_risk))
if len(at_risk):
    c2.metric("Макс. риск", f"{at_risk['churn_proba'].max():.2f}")

# --- таблица ---
st.dataframe(at_risk)
st.download_button(
    "Скачать at_risk.csv",
    at_risk.to_csv(index=False).encode("utf-8"),
    file_name="at_risk.csv",
    mime="text/csv",
)

# --- график по отделам ---
st.subheader("Текучесть по отделам")
dept_churn = (
    out.assign(at_risk=(out["churn_proba"] >= threshold))
      .groupby("Department")["at_risk"]
      .mean()
      .sort_values(ascending=False)
)

fig, ax = plt.subplots()
dept_churn.plot(kind="bar", ax=ax)
ax.set_ylabel("Доля at-risk")
ax.set_xlabel("Department")
ax.set_title("Доля сотрудников с риском ≥ выбранного порога")
fig.tight_layout()

buf = io.BytesIO()
fig.savefig(buf, format="png", bbox_inches="tight")
st.pyplot(fig)
st.download_button(
    "Скачать график (PNG)",
    buf.getvalue(),
    file_name="churn_by_department.png",
    mime="image/png",
)
