import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import joblib
from utils import load_config, ensure_parents
from features import add_derived_features

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def main():
    cfg = load_config()
    df = pd.read_parquet(cfg["data"]["parquet"])
    df = add_derived_features(df)

    y = df["Performance_Rating"].astype(float)
    X = df[cfg["features"]["categorical"] + cfg["features"]["numerical"]]

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cfg["features"]["categorical"]),
        ("num", "passthrough", cfg["features"]["numerical"]),
    ])

    reg = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=42)
    pipe = Pipeline([("pre", pre), ("reg", reg)])

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)

    mae = mean_absolute_error(y_te, pred)
    r2 = r2_score(y_te, pred)
    print("=== Rating model metrics ===")
    print(f"MAE: {mae:.3f}")
    print(f"R2: {r2:.3f}")

    out = cfg["models"]["rating_path"]
    ensure_parents(out)
    joblib.dump(pipe, out)
    print(f"Saved model -> {out}")

if __name__ == "__main__":
    main()
