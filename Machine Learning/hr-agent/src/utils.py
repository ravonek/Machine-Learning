from pathlib import Path
import yaml

def load_config(path: str = "src/config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_parents(path: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
