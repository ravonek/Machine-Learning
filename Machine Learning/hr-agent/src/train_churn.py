import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
import joblib
from utils import load_config, ensure_parents
from features import add_derived_features

def main():
    cfg = load_config()
    df = pd.read_parquet(cfg["data"]["parquet"])
    df = add_derived_features(df)

    y = (df["Status"].astype(str).str.lower().eq("resigned")).astype(int)
    X = df[cfg["features"]["categorical"] + cfg["features"]["numerical"]]

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cfg["features"]["categorical"]),
        ("num", "passthrough", cfg["features"]["numerical"]),
    ])

    clf = RandomForestClassifier(
        n_estimators=300, n_jobs=-1, class_weight="balanced_subsample", random_state=42
    )

    pipe = Pipeline([("pre", pre), ("clf", clf)])

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    proba = pipe.predict_proba(X_te)[:,1]

    acc = accuracy_score(y_te, pred)
    f1 = f1_score(y_te, pred)
    auc = roc_auc_score(y_te, proba)

    print("=== Churn model metrics ===")
    print(f"Accuracy: {acc:.3f}")
    print(f"F1: {f1:.3f}")
    print(f"ROC-AUC: {auc:.3f}")
    print("\nClassification report:\n", classification_report(y_te, pred, digits=3))

    out = cfg["models"]["churn_path"]
    ensure_parents(out)
    joblib.dump(pipe, out)
    print(f"Saved model -> {out}")

if __name__ == "__main__":
    main()
