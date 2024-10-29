"""
A simple and fast way to evaluate the diversity of a dataset by computing the mean cosine distance
between CLIP embeddings. The expected distance is 1.0 for a uniformly distributed set of embeddings.
The upper bound is actually higher than 1.0 but it decreases rapidly with n.
"""
import argparse
import numpy as np
from einops import rearrange
from pathlib import Path

from txt2img_unsupervised.load_pq_dir import load_pq_dir


def main():
    parser = argparse.ArgumentParser(
        description="Calculate the average cosine distance between CLIP embeddings in a dataset."
    )
    parser.add_argument("dataset_path", type=Path)
    parser.add_argument("-n", "--n_samples", type=int, default=8192)
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    args = parser.parse_args()

    dset = load_pq_dir(args.dataset_path)

    # Sample a bunch of clip embeddings
    idxs = np.random.randint(len(dset), size=args.n_samples)
    clips = dset.select_columns("clip_embedding")[idxs]["clip_embedding"]
    assert clips.shape == (args.n_samples, 768)

    # Pair them up and calculate cosine distances
    # I don't understand why you can't just write the number 2 in the rearrange expression...
    pairs = rearrange(clips, "(n two) d -> n two d", two=2)
    sims = np.sum(pairs[:, 0, :] * pairs[:, 1, :], axis=1)
    dists = 1 - sims

    # Calculate mean and a bootstrap CI
    mean_dist = np.mean(dists)
    bootstrap_means = []
    for _ in range(args.n_bootstrap):
        bootstrap_sample = np.random.choice(dists, size=len(dists), replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))

    ci_lower, ci_upper = np.percentile(bootstrap_means, [5, 95])

    print(f"Average distance: {mean_dist:.4f}")
    print(f"90% confidence interval: [{ci_lower:.4f}, {ci_upper:.4f}]")


if __name__ == "__main__":
    main()
