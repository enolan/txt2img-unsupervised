# Script that reads all image files and outputs metadata to CSV for size analysis

import argparse
import concurrent.futures
import jax.numpy as jnp
import numpy as np
import pandas as pd
import PIL.Image
import random
from pathlib import Path
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("input", help="Input directory", type=Path)

args = parser.parse_args()

paths = list(args.input.glob("*/deduped/*")) + list(args.input.glob("*/video_stills/*"))
random.shuffle(paths)
print(f"Found {len(paths)} files")
# paths = paths[:2000]

rows = []

with tqdm(total=len(paths), desc="processed") as pbar:
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:

        def process_img_path(p):
            img = PIL.Image.open(p)
            return {
                "path": str(p),
                "format": img.format,
                "mode": img.mode,
                "size": p.stat().st_size,
                "width": img.size[0],
                "height": img.size[1],
            }

        futures = []
        for path in tqdm(paths, desc="queued"):
            futures.append(executor.submit(process_img_path, path))
        for future in concurrent.futures.as_completed(futures):
            try:
                img_data = future.result()
                rows.append(img_data)
            except Exception as exc:
                tqdm.write(f"Exception while processing {future}: {exc}")
            pbar.update(1)

df = pd.DataFrame(rows)
df.to_csv("img_size_analysis.csv", index=False)
