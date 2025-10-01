import pandas as pd
from utils import load_config, ensure_parents

def load_and_clean(csv_path: str, out_parquet: str):
    chunks = pd.read_csv(csv_path, chunksize=200_000)
    dfs = []
    for c in chunks:
        # Basic cleanup
        if "Hire_Date" in c.columns:
            c["Hire_Date"] = pd.to_datetime(c["Hire_Date"], errors="coerce")
        if "Employee_ID" in c.columns:
            c = c.drop_duplicates(subset=["Employee_ID"])
        dfs.append(c)
    df = pd.concat(dfs, ignore_index=True)
    ensure_parents(out_parquet)
    df.to_parquet(out_parquet, index=False)
    print(f"Saved parquet -> {out_parquet} (rows={len(df)})")

if __name__ == "__main__":
    cfg = load_config()
    load_and_clean(cfg["data"]["raw_csv"], cfg["data"]["parquet"])
