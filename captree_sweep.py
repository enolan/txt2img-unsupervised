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
        k = random.choice([64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384])
        outlier_removal_level = random.uniform(0.0, 1.0)
        iters = random.choice([8, 16, 32, 64, 128, 256])
        batch_size = random.choice([2048, 4096, 8192, 16384, 32768])
        while True:
            max_leaf_size = random.choice(
                [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
            )
            if max_leaf_size >= k:
                break
        res = (k, outlier_removal_level, iters, batch_size, max_leaf_size)
        if res not in used_params:
            used_params.add(res)
            return {
                "k": k,
                "outlier_removal_level": outlier_removal_level,
                "iters": iters,
                "batch_size": batch_size,
                "max_leaf_size": max_leaf_size,
            }
        attempts += 1
        if attempts > 1000:
            raise Exception("Too many attempts to generate params")


def params_to_path(params):
    return (Path("sweep-captrees") / f"{params['k']}-{params['outlier_removal_level']:.4f}-{params['iters']}-{params['batch_size']}-{params['max_leaf_size']}")


def build_tree(params):
    # Build the tree
    save_dir = params_to_path(params)
    cmdline = [
        "python",
        "spherical_space_partitioning.py",
        "--pq-dir",
        "/home/enolan/datasets/preprocessed/128x128-randomcrops",
        "--subset",
        "1_000_000",
        "--read-dup-blacklist",
        "dup-blacklist.json",
        "--thin",
        "--k",
        str(params["k"]),
        "--outlier-removal-level",
        str(params["outlier_removal_level"]),
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
    start = time.monotonic()
    subprocess.check_call(cmdline)
    end = time.monotonic()
    print("Tree built")

    # Load the tree
    captree = spherical_space_partitioning.CapTree.load_from_disk(
        save_dir, save_cache=False
    )
    print("Tree loaded.")
    return captree, end - start


def run_benchmarks(captree, vecs):
    # Run benchmarks
    print("Running benchmarks...")
    results_all = {}
    for i, v in tqdm(enumerate(vecs), unit="vectors", total=len(vecs), leave=None):
        for max_cos_distance in tqdm(np.linspace(0, 2.0, 21), unit="max_cos_distances", leave=False):
            results = []
            for _ in tqdm(range(5), unit="trials", leave=False):
                start = time.monotonic()
                captree.sample_in_cap_approx(v, max_cos_distance)
                end = time.monotonic()
                elapsed = end - start
                results.append(elapsed)
            results_all.setdefault(i, {})[max_cos_distance] = results
    print("Benchmarks done.")
    return results_all


def mk_test_vecs():
    vecs = np.random.default_rng(69_420).standard_normal((8, 768), dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=-1, keepdims=True)
    return vecs


def main():
    # used_params = set()
    # try:
    #     with open("sweep-captrees/results-2024-01-30.json", "r") as f:
    #         for line in f:
    #             params = json.loads(line)["params"]
    #             used_params.add(
    #                 (
    #                     params["k"],
    #                     params["outlier_removal_level"],
    #                     params["iters"],
    #                     params["batch_size"],
    #                     params["max_leaf_size"],
    #                 )
    #             )
    # except FileNotFoundError:
    #     pass
    # print(f"Used params: {used_params}")
    vecs = mk_test_vecs()[:2]
    tree = spherical_space_partitioning.CapTree.load_from_disk(
        Path("sweep-captrees/64-0.3144-8-2048-4096"), save_cache=True
    )
    results = run_benchmarks(tree, vecs)
    print(results)

    # while True:
    #     params = gen_params(used_params)
    #     print(f"Params: {params}")
    #     captree, build_time = build_tree(params)
    #     results_all = run_benchmarks(captree, vecs)
    #     print(f"Results: {results_all}")
    #     with open("sweep-captrees/results-2024-01-30.json", "a") as f:
    #         json.dump({"params": params, "results": results_all, "build_time": build_time}, f)
    #         f.write("\n")
    #         f.flush()


if __name__ == "__main__":
    main()
