import json
import random
import shutil
import subprocess
import time

from pathlib import Path


def gen_params(used_params):
    attempts = 0
    while True:
        k = random.choice([64, 128, 256, 512])
        outlier_removal_level = random.uniform(0.0, 1.0)
        iters = random.choice([16])
        batch_size = random.choice([4096])
        while True:
            max_leaf_size = random.choice(
                [
                    512,
                    1024,
                    2048,
                    4096,
                    8192,
                    16384,
                    32768,
                    65536,
                    131072,
                    262144,
                ]
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
    return Path("/mnt/sweeps") / "-".join(
        f"{k}={v}" for k, v in sorted(params.items())
    )


def build_tree(params):
    # Build the tree
    save_dir = params_to_path(params)
    cmdline = [
        "python",
        "gen_captree.py",
        "--pq-dir",
        "/mnt/datasets/128x128-randomcrops",
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
    print(f"Tree built in {end - start} seconds")

    return save_dir, end - start


def gen_examples(tree_path, k):
    # Generate the cap-image pairs
    out_path = tree_path.parent / f"{tree_path.name}-examples.parquet"
    cmdline = [
        "python",
        "-m",
        "txt2img_unsupervised.gen_training_caps",
        "--batch-size",
        "16384",
        "--seed",
        "69",
        "--stop-after",
        "100_000",
        "--no-save-cache",
        "--density-estimate-samples",
        str(64 * 512 // k),  # Hold density estimate samples equal to k=64 samples=512
        "--tree-path",
        str(tree_path),
        "--out",
        str(out_path),
    ]
    print(f"Running {' '.join(cmdline)}")
    start = time.monotonic()
    subprocess.check_call(cmdline)
    end = time.monotonic()
    print(f"Examples generated in {end - start:0.2f} seconds")

    return out_path, end - start


def main():
    used_params = set()
    try:
        with open("/mnt/sweeps/results-2024-02-10.json", "r") as f:
            for line in f:
                params = json.loads(line)["params"]
                used_params.add(
                    (
                        params["k"],
                        params["outlier_removal_level"],
                        params["iters"],
                        params["batch_size"],
                        params["max_leaf_size"],
                    )
                )
    except FileNotFoundError:
        pass
    print(f"Used params: {used_params}")
    while True:
        params = gen_params(used_params)
        print(f"Params: {params}")
        tree_dir, build_time = build_tree(params)
        ex_path, gen_time = gen_examples(tree_dir, params["k"])
        with open("/mnt/sweeps/results-2024-02-10.json", "a") as f:
            json.dump(
                {
                    "params": params,
                    "examples_gen_time": gen_time,
                    "tree_build_time": build_time,
                },
                f,
            )
            f.write("\n")
            f.flush()
        print(f"Deleting {tree_dir} and {ex_path}")
        shutil.rmtree(tree_dir)
        ex_path.unlink()

if __name__ == "__main__":
    main()
