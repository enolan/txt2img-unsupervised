"""
Munge a checkpoint directory. Two operations are supported:

* inference: transform a checkpoint directory into a single-checkpoint, params-only version for
  inference. With a schedule-free optimizer this also improves sample quality, since the params
  used for evaluation differ from the ones used for training; otherwise it just saves space.
* migrate-transformer-mup: bring a transformer checkpoint saved before ImageModel supported muP
  up to date so it can be resumed or finetuned from, without changing how the model behaves.
"""

import argparse
from pathlib import Path

import jax
import orbax.checkpoint as ocp

from txt2img_unsupervised.checkpoint import (
    checkpoint_metadata,
    migrate_transformer_checkpoint_to_mup,
    mk_checkpoint_manager,
    train_state_class_for_config,
)
from txt2img_unsupervised.config import BaseModelConfig


def make_inference_checkpoint(src_dir: Path, dst_dir: Path, step: int | None) -> None:
    """Save the eval params of one step of a checkpoint directory as a params-only checkpoint."""
    print("Creating checkpoint manager for source directory...")
    src_checkpoint_manager = mk_checkpoint_manager(src_dir)
    metadata = checkpoint_metadata(src_checkpoint_manager)
    print(f"Metadata: {metadata}")
    if step is None:
        step = src_checkpoint_manager.latest_step()

    print(f"Loading checkpoint at step {step}...")
    model_cfg = BaseModelConfig.from_json_dict(metadata["model_cfg"])
    ts, _ = train_state_class_for_config(model_cfg).load_from_checkpoint(
        src_checkpoint_manager, step
    )
    print("Computing eval params...")
    eval_params = ts.get_eval_params()
    jax.tree.map(lambda a: a.block_until_ready(), eval_params)

    print("Creating checkpoint manager for destination directory...")
    dst_checkpoint_manager = ocp.CheckpointManager(
        dst_dir.absolute(),
        options=ocp.CheckpointManagerOptions(enable_async_checkpointing=False),
        item_names=("params",),
        metadata=metadata,
    )
    print("Saving checkpoint...")
    save_args = ocp.args.Composite(params=ocp.args.StandardSave(eval_params))
    dst_checkpoint_manager.save(step, args=save_args)
    dst_checkpoint_manager.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inference = subparsers.add_parser(
        "inference", help="Make a params-only checkpoint for inference"
    )
    inference.add_argument("checkpoint_dir", type=Path)
    inference.add_argument("out_dir", type=Path)
    inference.add_argument("--step", type=int, required=False)

    migrate = subparsers.add_parser(
        "migrate-transformer-mup",
        help="Migrate a transformer checkpoint from before muP support to the current format",
    )
    migrate.add_argument("checkpoint_dir", type=Path)
    migrate.add_argument("out_dir", type=Path)
    migrate.add_argument("--step", type=int, required=False)
    migrate.add_argument(
        "--batches-total",
        type=int,
        required=False,
        help="Total batches in the training run. Only needed if the learning rate schedule "
        "depends on it (e.g. warmup plus cosine)",
    )

    args = parser.parse_args()
    if args.command == "inference":
        make_inference_checkpoint(args.checkpoint_dir, args.out_dir, args.step)
    elif args.command == "migrate-transformer-mup":
        migrate_transformer_checkpoint_to_mup(
            args.checkpoint_dir, args.out_dir, args.step, args.batches_total
        )
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
