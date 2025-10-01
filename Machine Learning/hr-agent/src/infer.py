import pandas as pd
import joblib
from utils import load_config
from features import add_derived_features

_cfg = load_config()
_churn = None
_rating = None

def _load_models():
    global _churn, _rating
    if _churn is None:
        _churn = joblib.load(_cfg["models"]["churn_path"])
    if _rating is None:
        _rating = joblib.load(_cfg["models"]["rating_path"])

def predict_batch(df: pd.DataFrame):
    _load_models()
    df = add_derived_features(df)
    feat_cat = _cfg["features"]["categorical"]
    feat_num = _cfg["features"]["numerical"]
    X = df[feat_cat + feat_num]
    churn_proba = _churn.predict_proba(X)[:,1]
    rating_pred = _rating.predict(X)
    return churn_proba, rating_pred
