import pandas as pd

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Hire_Date" in df.columns:
        df["tenure_years"] = (
            pd.Timestamp.today(tz=None) - pd.to_datetime(df["Hire_Date"], errors="coerce")
        ).dt.days / 365
    else:
        df["tenure_years"] = None
    return df
