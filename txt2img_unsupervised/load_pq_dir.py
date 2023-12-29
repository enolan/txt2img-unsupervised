from datasets import Dataset
from pathlib import Path


def load_pq_dir(dir_path):
    pq_paths = [str(p) for p in sorted(dir_path.glob("**/*.parquet"))]
    return Dataset.from_parquet(pq_paths).with_format("numpy")
