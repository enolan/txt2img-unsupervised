import json
import numpy as np
import random
import subprocess
import time

from pathlib import Path
from tqdm import tqdm
from tqdm.contrib import tenumerate

import spherical_space_partitioning


def gen_params(used_params):
    attempts = 0
    while True:
        k = random.choice([8, 16, 32, 64, 128])
        iters = random.choice([8, 16, 32, 64, 128])
        batch_size = random.choice([128, 256, 512, 1024, 2048, 4096, 8192])
        while True:
            max_leaf_size = random.choice(
                [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
            )
            if max_leaf_size >= k:
                break
        res = (k, iters, batch_size, max_leaf_size)
        if res not in used_params:
            used_params.add(res)
            return {
                "k": k,
                "iters": iters,
                "batch_size": batch_size,
                "max_leaf_size": max_leaf_size,
            }
        attempts += 1
        if attempts > 1000:
            raise Exception("Too many attempts to generate params")


def build_tree(params):
    # Build the tree
    save_dir = (
        Path("sweep-captrees")
        / f"{params['k']}-{params['iters']}-{params['batch_size']}-{params['max_leaf_size']}"
    )
    cmdline = [
        "python",
        "spherical_space_partitioning.py",
        "--pq-dir",
        "/home/enolan/datasets/preprocessed/128x128-randomcrops",
        "--subset",
        "1_000_000",
        "--read-dup-blacklist",
        "dup-blacklist.json",
        "--k",
        str(params["k"]),
        "--batch-size",
        str(params["batch_size"]),
        "--k-means-iters",
        str(params["iters"]),
        "--max-leaf-size",
        str(params["max_leaf_size"]),
        "--save-dir",
        str(save_dir),
        "--summary-file",
        f"{save_dir}-summary.json",
    ]

    print(f"Running {' '.join(cmdline)}")
    subprocess.check_call(cmdline)
    print("Tree built")

    # Load the tree
    captree = spherical_space_partitioning.CapTree.load_from_disk(
        save_dir, save_cache=False
    )
    print("Tree loaded.")
    return captree


def run_benchmarks(captree, vecs):
    # Run benchmarks
    print("Running benchmarks...")
    results_all = {}
    for i, v in tqdm(enumerate(vecs), unit="vectors", total=len(vecs)):
        for max_cos_distance in tqdm([0.2, 0.5], unit="max_cos_distances", leave=False):
            results = []
            for _ in tqdm(range(10), unit="trials", leave=False):
                start = time.monotonic()
                captree.sample_in_cap(v, max_cos_distance)
                end = time.monotonic()
                elapsed = end - start
                results.append(elapsed)
            results_all.setdefault(i, {})[max_cos_distance] = results
    print("Benchmarks done.")
    return results_all


def main():
    used_params = set()
    with open("sweep-captrees/results.json", "r") as f:
        for line in f:
            params = json.loads(line)["params"]
            used_params.add(
                (
                    params["k"],
                    params["iters"],
                    params["batch_size"],
                    params["max_leaf_size"],
                )
            )
    print(f"Used params: {used_params}")
    vecs = np.random.default_rng(69_420).standard_normal((8, 768), dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=-1, keepdims=True)

    while True:
        params = gen_params(used_params)
        print(f"Params: {params}")
        captree = build_tree(params)
        results_all = run_benchmarks(captree, vecs)
        print(f"Results: {results_all}")
        with open("sweep-captrees/results.json", "a") as f:
            json.dump({"params": params, "results": results_all}, f)
            f.write("\n")


if __name__ == "__main__":
    main()
