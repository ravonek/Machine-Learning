# -*- coding: utf-8 -*-
import os
import streamlit as st
import pandas as pd
import numpy as np

# Надёжный импорт FAISS (Linux/Windows -> faiss, macOS -> faiss.cpu)
try:
    import faiss  # type: ignore
except ModuleNotFoundError:
    import faiss.cpu as faiss  # type: ignore

from sentence_transformers import SentenceTransformer

# Визуализации
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid", context="notebook")

st.set_page_config(page_title="MedQuAD RAG Assistant", page_icon="🩺", layout="wide")
st.title("🩺 MedQuAD RAG Assistant")
st.caption("Retrieval-Augmented QA on medical Q&A (educational use only).")

@st.cache_resource
def load_index_and_data(artifacts_dir: str):
    index = faiss.read_index(os.path.join(artifacts_dir, "medquad.index"))
    df = pd.read_parquet(os.path.join(artifacts_dir, "medquad.parquet"))
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return index, df, model

def search(query: str, index, model, df, k: int = 5):
    q_emb = model.encode([query], normalize_embeddings=True)
    D, I = index.search(np.asarray(q_emb, dtype=np.float32), k)
    hits = df.iloc[I[0]].copy()
    hits["score"] = D[0]
    return hits

def format_answer(query: str, hits: pd.DataFrame) -> str:
    best = hits.iloc[0]
    answer = str(best["answer"])
    source = str(best.get("source", ""))
    focus = str(best.get("focus_area", ""))
    disclaimer = "\n\n**Disclaimer:** This information is for educational purposes and does not replace professional medical advice."
    header = f"**Topic:** {focus}  \n**Source:** {source}"
    return header + "\n\n" + answer + disclaimer

# Sidebar help
st.sidebar.header("How to use")
st.sidebar.markdown("""
1. Prepare artifacts with **`prepare_data.py`** to build FAISS index.  
2. Click **Load index & data**.  
3. Type a question and click **Search**.  
4. Review retrieved answers and RAG output.
""")

# Controls
artifacts_dir = st.text_input("Artifacts directory", value="artifacts")

# Tabs: Assistant (RAG) | Analytics (EDA)
tab_assistant, tab_analytics = st.tabs(["Assistant", "Analytics"])

with tab_assistant:
    if st.button("Load index & data", use_container_width=False):
        with st.spinner("Loading..."):
            try:
                index, df, model = load_index_and_data(artifacts_dir)
                st.success(f"Loaded index with {index.ntotal} items.")
                st.session_state["index"] = index
                st.session_state["df"] = df
                st.session_state["model"] = model
            except Exception as e:
                st.error(str(e))

    if all(k in st.session_state for k in ["index", "df", "model"]):
        query = st.text_input("Ask a medical question")
        top_k = st.slider("Top-K", min_value=1, max_value=10, value=5, step=1)
        if st.button("Search"):
            hits = search(query, st.session_state["index"], st.session_state["model"], st.session_state["df"], k=top_k)
            st.subheader("Top results")
            st.dataframe(hits[["score", "focus_area", "source", "question"]], use_container_width=True)
            st.subheader("Answer (RAG)")
            st.markdown(format_answer(query, hits))

with tab_analytics:
    st.write("Basic exploratory data analysis (EDA) of the MedQuAD dataset used for retrieval.")
    if "df" not in st.session_state:
        st.info("Load index & data on the Assistant tab to view analytics.")
    else:
        df = st.session_state["df"].copy()

        # Summary cards
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Total rows", len(df))
        with c2: st.metric("Missing answers", int(df["answer"].isna().sum()) if "answer" in df else 0)
        with c3: st.metric("Unique focus areas", df["focus_area"].nunique() if "focus_area" in df else 0)
        with c4: st.metric("Sources", df["source"].nunique() if "source" in df else 0)

        # Top focus areas
        st.subheader("Top focus areas")
        if "focus_area" in df.columns:
            top_n = st.slider("Show top-N focus areas", 5, 30, 15)
            fa_counts = df["focus_area"].astype(str).value_counts().head(top_n)

            st.dataframe(fa_counts.rename("count"), use_container_width=True)

            fig1, ax1 = plt.subplots(figsize=(9, 6))
            sns.barplot(y=fa_counts.index, x=fa_counts.values, ax=ax1)
            ax1.set_xlabel("Count")
            ax1.set_ylabel("Focus area")
            ax1.set_title("Top focus areas")
            st.pyplot(fig1)
        else:
            st.warning("Column `focus_area` not found.")

        # Text lengths
        st.subheader("Text length distributions")
        df["question_len"] = df["question"].astype(str).str.len() if "question" in df else 0
        df["answer_len"] = df["answer"].astype(str).str.len() if "answer" in df else 0

        c5, c6 = st.columns(2)
        with c5:
            fig2, ax2 = plt.subplots(figsize=(7, 4))
            sns.histplot(df["question_len"], bins=40, kde=True, ax=ax2)
            ax2.set_title("Question length distribution")
            ax2.set_xlabel("Characters")
            ax2.set_ylabel("Frequency")
            st.pyplot(fig2)

        with c6:
            fig3, ax3 = plt.subplots(figsize=(7, 4))
            sns.histplot(df["answer_len"], bins=40, kde=True, ax=ax3)
            ax3.set_title("Answer length distribution")
            ax3.set_xlabel("Characters")
            ax3.set_ylabel("Frequency")
            st.pyplot(fig3)

        # Raw sample browser
        st.subheader("Sample browser")
        n = st.slider("Row index", 0, max(0, len(df)-1), 0)
        st.write("**Question:**", df.iloc[n]["question"] if "question" in df else "")
        st.write("**Focus area:**", df.iloc[n].get("focus_area", ""))
        st.write("**Source:**", df.iloc[n].get("source", ""))
        st.write("**Answer (truncated):**")
        text_full = str(df.iloc[n].get("answer", ""))
        st.text((text_full[:1200] + ("..." if len(text_full) > 1200 else "")))
