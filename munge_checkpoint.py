import argparse
import jax
import orbax.checkpoint as ocp
from pathlib import Path

from txt2img_unsupervised.checkpoint import TrainState, mk_checkpoint_manager


def main():
    parser = argparse.ArgumentParser(
        description="Munge a checkpoint directory. Currently only transforms checkpoint directories "
        "into single checkpoint inference-only versions. Note that when using a schedule-free "
        "optimizer this processing step should improve sample quality. Otherwise it only reduces "
        "file size."
    )
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--step", type=int, required=False)
    args = parser.parse_args()

    print("Creating checkpoint manager for source directory...")
    src_checkpoint_manager = mk_checkpoint_manager(args.checkpoint_dir)
    print(f"Metadata: {src_checkpoint_manager.metadata()}")
    if args.step is None:
        step = src_checkpoint_manager.latest_step()
    else:
        step = args.step

    print(f"Loading checkpoint at step {step}...")
    ts, _ = TrainState.load_from_checkpoint(src_checkpoint_manager, step)
    print("Computing eval params...")
    eval_params = ts.get_eval_params()
    jax.tree.map(lambda a: a.block_until_ready(), eval_params)


    print("Creating checkpoint manager for destination directory...")
    dst_checkpoint_manager = ocp.CheckpointManager(
        args.out_dir.absolute(),
        options=ocp.CheckpointManagerOptions(enable_async_checkpointing=False),
        item_names=("params",),
        metadata=src_checkpoint_manager.metadata(),
    )
    print("Saving checkpoint...")
    save_args = ocp.args.Composite(params=ocp.args.StandardSave(eval_params))
    dst_checkpoint_manager.save(step, args=save_args)

if __name__ == "__main__":
    main()
