"Coordinate check for flow matching models to check whether muP is working."

from datasets import Dataset
from functools import partial
from pathlib import Path
from tqdm import tqdm, trange
from tqdm.contrib import tenumerate
import argparse
import gc
import jax
import jax.nn as nn
import jax.numpy as jnp
import math
import matplotlib.pyplot as plt
import numpy as np
import optax
import optax.transforms
from contextlib import nullcontext
from mpl_toolkits.mplot3d import Axes3D

from .cap_sampling import LogitsTable
from .flow_matching import CapConditionedVectorField
from . import flow_matching
from .transformer_model import ImageModel, loss_batch
from . import transformer_model
from .muon import muon


def process_intermediates(intermediates):
    """
    Convert the raw intermediates dictionary into a more readable format.

    This transforms the nested dictionary structure from captured intermediates
    into a flattened dictionary with clear names, separating the MLP blocks.

    Args:
        intermediates: The raw intermediates dictionary from model execution

    Returns:
        A dictionary with human-readable keys and extracted values
    """
    result = {}
    result["model_output"] = intermediates["__call__"][0]
    result["final_norm_output"] = intermediates["final_norm"]["__call__"][0]
    result["output_projection"] = intermediates["out_proj"]["__call__"][0]

    if "pre_mlp_proj" in intermediates:
        result["pre_mlp_projection"] = intermediates["pre_mlp_proj"]["__call__"][0]

    # Process each MLP block separately
    n_layers = intermediates["mlp_blocks"]["__call__"][0][0].shape[0]
    for layer_idx in range(n_layers):
        result[f"mlp_block_{layer_idx}_output"] = intermediates["mlp_blocks"][
            "__call__"
        ][0][0][layer_idx]
        result[f"mlp_block_{layer_idx}_gate"] = intermediates["mlp_blocks"][
            "gate_proj"
        ]["__call__"][0][layer_idx]
        result[f"mlp_block_{layer_idx}_norm"] = intermediates["mlp_blocks"]["norm"][
            "__call__"
        ][0][layer_idx]
        result[f"mlp_block_{layer_idx}_out_proj"] = intermediates["mlp_blocks"][
            "out_proj"
        ]["__call__"][0][layer_idx]
        result[f"mlp_block_{layer_idx}_value"] = intermediates["mlp_blocks"][
            "value_proj"
        ]["__call__"][0][layer_idx]

    return result


def process_transformer_intermediates(intermediates, model):
    """
    Convert the raw intermediates dictionary from transformer model into a readable format.

    This extracts activation values from the transformer layers for muP analysis.

    Args:
        intermediates: The raw intermediates dictionary from transformer model execution
        model: The ImageModel instance to determine which components are present

    Returns:
        A dictionary with human-readable keys and extracted activation values
    """
    result = {}

    result["input_embedding"] = intermediates["in_embed"]["__call__"][0]
    result["clip_projection"] = intermediates["clip_proj"]["__call__"][0]

    transformer_intermediates = intermediates["transformer_layers"]
    n_layers = transformer_intermediates["__call__"][0][0].shape[0]
    
    for layer_idx in range(n_layers):
        mha = transformer_intermediates["mha"]
        result[f"transformer_layer_{layer_idx}_query"] = mha["query"]["__call__"][0][layer_idx]
        result[f"transformer_layer_{layer_idx}_key"] = mha["key"]["__call__"][0][layer_idx]
        result[f"transformer_layer_{layer_idx}_value"] = mha["value"]["__call__"][0][layer_idx]
        result[f"transformer_layer_{layer_idx}_attention_out"] = mha["out"]["__call__"][0][layer_idx]
        
        result[f"transformer_layer_{layer_idx}_linear_1"] = transformer_intermediates["linear_1"]["__call__"][0][layer_idx]
        result[f"transformer_layer_{layer_idx}_linear_2"] = transformer_intermediates["linear_2"]["__call__"][0][layer_idx]
        
        result[f"transformer_layer_{layer_idx}_layer_norm_1"] = transformer_intermediates["layer_norm_1"]["__call__"][0][layer_idx]
        result[f"transformer_layer_{layer_idx}_layer_norm_2"] = transformer_intermediates["layer_norm_2"]["__call__"][0][layer_idx]

    result["final_layer_norm"] = intermediates["final_layer_norm"]["__call__"][0]
    result["output_projection"] = intermediates["logits_decoder"]["__call__"][0]
    result["model_output"] = intermediates["__call__"][0]

    return result


@partial(
    jax.jit,
    static_argnames=["mdl", "model_type"],
    donate_argnames=["rng"],
)
def compute_loss_no_grad(logits_tbl, mdl, params, rng, data, model_type="flow"):
    """Compute loss without gradients for test evaluation."""
    rng, next_rng = jax.random.split(rng)
    
    if model_type == "flow":
        loss = flow_matching.compute_batch_loss(
            mdl,
            params,
            {"point_vec": data},
            rng,
            logits_tbl,
            capture_intermediates=False,
        )
    else:  # transformer
        images, clips = data
        max_cos_distances = jnp.empty((images.shape[0], 0))  # empty for clip_caps=False
        loss = loss_batch(
            mdl,
            params,
            rng,
            images,
            clips,
            max_cos_distances,
        )
    
    return loss, next_rng


def compute_test_loss(
    logits_tbl, mdl, params, rng, test_data, batch_size, model_type="flow"
):
    """Compute average loss over the test dataset."""
    if model_type == "flow":
        n_batches = len(test_data) // batch_size
        total_loss = 0.0

        for i in trange(n_batches, desc="Evaluating test batches"):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch = test_data[start_idx:end_idx]
            loss, rng = compute_loss_no_grad(logits_tbl, mdl, params, rng, batch, model_type)
            total_loss += loss

        return total_loss / n_batches
    else:  # transformer
        images, clips = test_data
        n_batches = len(images) // batch_size
        total_loss = 0.0

        for i in trange(n_batches, desc="Evaluating test batches"):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            images_batch = images[start_idx:end_idx]
            clips_batch = clips[start_idx:end_idx]
            
            batch = (images_batch, clips_batch)
            loss, rng = compute_loss_no_grad(logits_tbl, mdl, params, rng, batch, model_type)
            total_loss += loss

        return total_loss / n_batches


@partial(
    jax.jit,
    static_argnames=["mdl", "model_type"],
    donate_argnames=["rng"],
)
def compute_gradients(logits_tbl, mdl, params, rng, data, model_type="flow"):
    """
    Compute gradients. This is split from apply_updates so we can do this on GPU and
    apply_updates on CPU.
    """
    rng, next_rng = jax.random.split(rng)

    if model_type == "flow":
        loss_fn = lambda params: flow_matching.compute_batch_loss(
            mdl,
            params,
            {"point_vec": data},
            rng,
            logits_tbl,
            capture_intermediates=True,
        )
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, intermediates), grad = grad_fn(params)
        processed_intermediates = jax.tree.map(
            lambda x: jnp.mean(jnp.abs(x)),
            process_intermediates(intermediates["intermediates"]),
        )
    else:  # transformer
        # data contains (images, clips)
        images, clips = data
        max_cos_distances = jnp.empty((images.shape[0], 0))  # empty for clip_caps=False

        def transformer_loss_fn(params):
            # Capture intermediates during forward pass
            logits, intermediates = mdl.apply(
                params,
                rngs={"dropout": rng},
                images=images,
                clip_embeddings=clips,
                max_cos_distances=max_cos_distances,
                capture_intermediates=True,
            )
            per_token_loss = optax.softmax_cross_entropy(
                logits, jax.nn.one_hot(images, 8192)
            )
            loss = jnp.mean(per_token_loss)
            return loss, intermediates

        grad_fn = jax.value_and_grad(transformer_loss_fn, has_aux=True)
        (loss, intermediates), grad = grad_fn(params)
        processed_intermediates = jax.tree.map(
            lambda x: jnp.mean(jnp.abs(x)),
            process_transformer_intermediates(intermediates["intermediates"], mdl),
        )

    return loss, processed_intermediates, grad, next_rng


@partial(
    jax.jit,
    static_argnames=["opt"],
    donate_argnames=["opt_state", "params"],
)
def grad_update(opt, grad, opt_state, params):
    """Do a gradient descent step given gradients."""
    updates, new_opt_state = opt.update(grad, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state


str_devices = lambda x: jax.tree.map(lambda y: y.device, x)


def train_step(
    logits_tbl,
    mdl,
    opt,
    params,
    opt_state,
    rng,
    data,
    use_cpu_offload=False,
    model_type="flow",
):
    """Complete training step, optionally with CPU-GPU split."""
    gpu_params = (
        jax.device_put(params, device=jax.devices("gpu")[0])
        if use_cpu_offload
        else params
    )
    loss, processed_intermediates, grad, next_rng = compute_gradients(
        logits_tbl, mdl, gpu_params, rng, data, model_type
    )

    if use_cpu_offload:
        processed_intermediates = jax.device_put(
            processed_intermediates, jax.devices("cpu")[0]
        )
        grad = jax.device_put(grad, jax.devices("cpu")[0])
        loss = jax.device_put(loss, jax.devices("cpu")[0])

    new_params, new_opt_state = grad_update(opt, grad, opt_state, params)
    return loss, processed_intermediates, new_params, new_opt_state, next_rng


@partial(jax.jit, static_argnames=["model"])
def init_flow_model_params(model, init_key):
    """JIT-compiled flow model initialization function."""
    dummy_inputs = model.dummy_inputs()
    return model.init(init_key, *dummy_inputs)


@partial(jax.jit, static_argnames=["model"])
def init_transformer_model_params(model, init_key):
    """JIT-compiled transformer model initialization function."""
    dummy_inputs = model.dummy_inputs()
    return model.init(init_key, *dummy_inputs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["flow", "transformer"],
        required=True,
        help="Type of model to test: 'flow' or 'transformer'",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="Path to dataset (3D for flow, image for transformer)",
    )
    parser.add_argument(
        "--lr-base",
        type=float,
        required=False,
        help="Base learning rate (will be ignored if lr-low and lr-high are provided)",
    )
    parser.add_argument(
        "--lr-low", type=float, required=False, help="Lowest learning rate to test"
    )
    parser.add_argument(
        "--lr-high", type=float, required=False, help="Highest learning rate to test"
    )
    parser.add_argument(
        "--n-lr-points",
        type=int,
        required=False,
        default=5,
        help="Number of learning rate points to test between lr-low and lr-high",
    )
    parser.add_argument("--d-model-low", type=int, required=True)
    parser.add_argument("--d-model-high", type=int, required=True)

    # Flow model specific arguments
    parser.add_argument(
        "--reference-directions", type=int, required=False, default=None
    )
    parser.add_argument("--time-dim", type=int, required=False)
    parser.add_argument("--use-pre-mlp-projection", type=bool, required=False)
    parser.add_argument("--mlp-expansion-factor", type=int, required=False, default=4)

    # Transformer model specific arguments
    parser.add_argument("--num-heads", type=int, required=False, default=8)
    parser.add_argument("--ff-dim", type=int, required=False)
    parser.add_argument("--image-tokens", type=int, required=False, default=256)
    parser.add_argument("--dropout", type=float, required=False, default=0.1)

    # Common arguments
    parser.add_argument("--n-layers", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--n-seeds", type=int, required=False, default=5)
    parser.add_argument(
        "--cpu-offload-threshold",
        type=int,
        required=False,
        default=2048,
        help="d_model threshold above which optimizer state and weight updates are offloaded to CPU",
    )
    parser.add_argument(
        "--n-train-steps",
        type=int,
        required=False,
        default=10,
        help="Number of training steps to perform for each seed",
    )
    parser.add_argument(
        "--n-test-batches",
        type=int,
        required=False,
        default=10,
        help="Number of test batches to evaluate",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=False,
        default=".",
        help="Directory to save charts in",
    )
    # Muon optimizer arguments
    parser.add_argument(
        "--use-muon",
        action="store_true",
        help="Enable Muon optimizer for MLP projection kernels (Adam for other parameters)",
    )
    parser.add_argument(
        "--muon-beta",
        type=float,
        required=False,
        default=0.95,
        help="Momentum parameter for Muon optimizer (default: 0.95)",
    )
    parser.add_argument(
        "--muon-lr-low",
        type=float,
        required=False,
        help="Lowest Muon learning rate to test (only used with --use-muon)",
    )
    parser.add_argument(
        "--muon-lr-high",
        type=float,
        required=False,
        help="Highest Muon learning rate to test (only used with --use-muon)",
    )
    parser.add_argument(
        "--n-muon-lr-points",
        type=int,
        required=False,
        default=5,
        help="Number of Muon learning rate points to test between muon-lr-low and muon-lr-high",
    )

    args = parser.parse_args()

    # Validate model-specific arguments
    if args.model_type == "flow":
        if args.time_dim is None:
            raise ValueError("--time-dim is required for flow models")
        if args.use_pre_mlp_projection is None:
            raise ValueError("--use-pre-mlp-projection is required for flow models")
    elif args.model_type == "transformer":
        if args.ff_dim is None:
            args.ff_dim = 4 * args.d_model_low  # Default ff_dim as 4x d_model
        # Set defaults for transformer-specific args if not provided
        if args.num_heads is None:
            args.num_heads = 8

    if args.n_seeds <= 0:
        raise ValueError("n_seeds must be at least 1")
    if args.n_train_steps <= 0:
        raise ValueError("n_train_steps must be at least 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Charts will be saved to: {args.output_dir}")

    dsets = (
        Dataset.from_parquet(str(args.dataset_path))
        .with_format("numpy")
        .train_test_split(test_size=args.batch_size * args.n_test_batches)
    )
    dset_train = dsets["train"]
    dset_test = dsets["test"]
    print(
        f"Dataset loaded with {len(dset_train)} training examples and {len(dset_test)} test examples. First example: {dset_train[0]}"
    )
    dset_train = dset_train.select(range(args.batch_size * args.n_train_steps))

    # Create logits table only for flow models
    logits_table = LogitsTable(d=3 - 1, n=8192) if args.model_type == "flow" else None

    doing_lr_sweep = args.lr_low is not None and args.lr_high is not None
    if doing_lr_sweep and args.lr_base is not None:
        print("Warning: lr-base is ignored when lr-low and lr-high are provided")
    elif not doing_lr_sweep and args.lr_base is None:
        raise ValueError("Either lr-base or both lr-low and lr-high must be provided")

    # Validate Muon arguments
    if args.use_muon:
        if args.muon_lr_low is None or args.muon_lr_high is None:
            raise ValueError(
                "When using Muon, both --muon-lr-low and --muon-lr-high must be provided"
            )
        if args.muon_beta <= 0 or args.muon_beta >= 1:
            raise ValueError(f"muon-beta must be between 0 and 1, got {args.muon_beta}")
        print(f"Using Muon optimizer with beta={args.muon_beta}")
    else:
        if args.muon_lr_low is not None or args.muon_lr_high is not None:
            print(
                "Warning: Muon learning rate arguments are ignored when --use-muon is not specified"
            )

    # Set up learning rate ranges
    if doing_lr_sweep:
        adam_lr_values = list(
            np.logspace(np.log10(args.lr_low), np.log10(args.lr_high), args.n_lr_points)
        )
        print(f"Testing Adam learning rates: {adam_lr_values}")
    else:
        adam_lr_values = [args.lr_base]

    if args.use_muon:
        muon_lr_values = list(
            np.logspace(
                np.log10(args.muon_lr_low),
                np.log10(args.muon_lr_high),
                args.n_muon_lr_points,
            )
        )
        print(f"Testing Muon learning rates: {muon_lr_values}")
        # Create cartesian product of Adam and Muon learning rates
        lr_combinations = [
            (adam_lr, muon_lr)
            for adam_lr in adam_lr_values
            for muon_lr in muon_lr_values
        ]
        print(f"Total learning rate combinations: {len(lr_combinations)}")
    else:
        muon_lr_values = []
        lr_combinations = [(adam_lr, None) for adam_lr in adam_lr_values]

    print(f"Number of learning rate combinations to test: {len(lr_combinations)}")
    if len(lr_combinations) == 0:
        raise ValueError("No learning rate combinations to test!")

    # Generate exponentially spaced d_model values with base 2
    low_exp = math.log2(args.d_model_low)
    high_exp = math.log2(args.d_model_high)

    if not (
        2 ** int(low_exp) == args.d_model_low
        and 2 ** int(high_exp) == args.d_model_high
    ):
        raise ValueError("d-model-low and d-model-high must be powers of 2")

    d_model_values = [2**i for i in range(int(low_exp), int(high_exp) + 1)]

    print(f"Testing d_model values: {d_model_values}")

    activations = []
    losses = np.zeros((len(d_model_values), len(lr_combinations), args.n_train_steps))
    test_losses = np.zeros((len(d_model_values), len(lr_combinations)))

    n_combinations = len(d_model_values) * len(lr_combinations)
    all_keys = jax.random.split(jax.random.PRNGKey(20250319), n_combinations)
    key_idx = 0

    for d_model_idx, d_model in tenumerate(d_model_values, desc="d_model values"):
        for lr_idx, (adam_lr, muon_lr) in tenumerate(
            lr_combinations, desc=f"LR for d_model={d_model}"
        ):
            tqdm.write(
                f"\nTraining with d_model = {d_model}, Adam LR = {adam_lr}, Muon LR = {muon_lr}"
            )

            use_cpu_offload = d_model > args.cpu_offload_threshold
            if use_cpu_offload:
                tqdm.write(f"Using CPU offloading for d_model={d_model}")

            master_key = all_keys[key_idx]
            key_idx += 1
            seed_keys = jax.random.split(master_key, args.n_seeds)

            if args.model_type == "flow":
                model = CapConditionedVectorField(
                    domain_dim=3,
                    reference_directions=args.reference_directions,
                    conditioning_dim=None,
                    time_dim=args.time_dim,
                    use_pre_mlp_projection=args.use_pre_mlp_projection,
                    n_layers=args.n_layers,
                    d_model=d_model,
                    mlp_expansion_factor=args.mlp_expansion_factor,
                    mlp_dropout_rate=None,
                    input_dropout_rate=None,
                )
            else:  # transformer
                model = ImageModel(
                    d_model=d_model,
                    num_heads=args.num_heads,
                    ff_dim=args.ff_dim,
                    dropout=args.dropout,
                    image_dropout=None,
                    clip_dropout=None,
                    n_layers=args.n_layers,
                    image_tokens=args.image_tokens,
                    clip_conditioning=True,
                    clip_caps=False,
                    clip_cap_count=3,
                    corrected_cap_projections=True,
                    do_clip_feedforward=False,
                    norm_clip_embeddings=True,
                    use_biases=True,
                    activations_dtype=jnp.float32,
                    activation_function=nn.gelu,
                    weights_dtype=jnp.float32,
                    pre_norm=True,
                )
            tqdm.write(f"Model: {model}")
            tqdm.write(f"m_d = {model.d_model_scale_factor}")

            if args.use_muon:
                # Mixed Muon/Adam optimization
                adam_fixed_opt = optax.adam(adam_lr)
                adam_scaled_opt = optax.adam(model.scale_lr(adam_lr))
                muon_fixed_opt = muon(
                    learning_rate=muon_lr,
                    beta=args.muon_beta,
                    weight_decay=0.0,
                )
                muon_scaled_opt = muon(
                    learning_rate=model.scale_lr(muon_lr),
                    beta=args.muon_beta,
                    weight_decay=0.0,
                )

                opt = optax.transforms.partition(
                    {
                        "adam_fixed": adam_fixed_opt,
                        "adam_scaled": adam_scaled_opt,
                        "muon_fixed": muon_fixed_opt,
                        "muon_scaled": muon_scaled_opt,
                    },
                    model.mk_partition_map(use_muon=True),
                )
            else:
                # Pure Adam optimization
                opt_fixed_lr = optax.adam(adam_lr)
                opt_scaled_lr = optax.adam(model.scale_lr(adam_lr))

                # Use the mk_partition_map method from the model for proper muP scaling
                opt = optax.transforms.partition(
                    {"fixed_lr": opt_fixed_lr, "scaled_lr": opt_scaled_lr},
                    model.mk_partition_map(use_muon=False),
                )
            init_opt_state = jax.jit(opt.init)

            # List to hold activations for all seeds for this d_model and learning rate
            all_seed_activations = []
            # Array to hold losses for all seeds for this d_model and learning rate
            seed_losses = np.zeros((args.n_seeds, args.n_train_steps))
            # Array to hold test losses for all seeds
            seed_test_losses = np.zeros(args.n_seeds)

            tqdm.write(f"Starting training with {args.n_seeds} seeds")

            for seed_idx in trange(args.n_seeds, desc="Seeds", leave=False):
                tqdm.write(f"Training with seed {seed_idx+1}/{args.n_seeds}")

                init_key, train_key = jax.random.split(seed_keys[seed_idx])

                device_ctx = (
                    jax.default_device(jax.devices("cpu")[0])
                    if use_cpu_offload
                    else nullcontext()
                )
                with device_ctx:
                    tqdm.write("Initializing parameters")
                    if args.model_type == "flow":
                        params = init_flow_model_params(model, init_key)
                    else:  # transformer
                        params = init_transformer_model_params(model, init_key)

                    tqdm.write(f"Initializing optimizer state")
                    opt_state = init_opt_state(params)

                tqdm.write("Training")
                rng = train_key
                activations_this_seed = []
                for i, batch in tenumerate(
                    dset_train.iter(args.batch_size, drop_last_batch=True),
                    desc=f"Seed {seed_idx+1} steps",
                    total=args.n_train_steps,
                ):
                    if args.model_type == "flow":
                        batch_data = batch["vec"]
                    else:  # transformer
                        batch_data = (
                            batch["encoded_img"],
                            batch["clip_embedding"],  # single CLIP embedding per sample (batch, 768)
                        )

                    loss, processed_intermediates, params, opt_state, rng = train_step(
                        logits_table,
                        model,
                        opt,
                        params,
                        opt_state,
                        rng,
                        batch_data,
                        use_cpu_offload,
                        args.model_type,
                    )
                    activations_this_seed.append(processed_intermediates)
                    # Convert loss to numpy and store
                    seed_losses[seed_idx, i] = np.array(loss)
                    # Only log losses occasionally to avoid flooding the output
                    log_interval = max(1, args.n_train_steps // 10)
                    if i % log_interval == 0 or i == args.n_train_steps - 1:
                        tqdm.write(f"Loss: {loss}, Step {i}/{args.n_train_steps}")

                # After training, evaluate on test set
                tqdm.write("Evaluating on test set")
                if args.model_type == "flow":
                    test_data = dset_test["vec"]
                else:  # transformer
                    test_data = (
                        dset_test["encoded_img"],
                        dset_test["clip_embedding"],  # single CLIP embedding per sample (batch, 768)
                    )
                test_loss = compute_test_loss(
                    logits_table,
                    model,
                    params,
                    rng,
                    test_data,
                    args.batch_size,
                    args.model_type,
                )
                seed_test_losses[seed_idx] = np.array(test_loss)
                tqdm.write(f"Test loss: {test_loss}")

                tqdm.write(f"Seed {seed_idx+1} training complete")
                all_seed_activations.append(activations_this_seed)
                del processed_intermediates, params, opt_state, rng
                gc.collect()

            # Average the losses across seeds
            avg_losses = np.mean(seed_losses, axis=0)
            avg_test_loss = np.mean(seed_test_losses)
            losses[d_model_idx, lr_idx] = avg_losses
            test_losses[d_model_idx, lr_idx] = avg_test_loss
            tqdm.write(f"Average test loss: {avg_test_loss}")

            # Only keep activations for the lowest learning rate if doing a sweep
            if lr_idx == 0:
                # Convert to numpy array for easier averaging
                # First, convert each dictionary element to a list of arrays
                num_steps = len(all_seed_activations[0])
                activation_keys = all_seed_activations[0][0].keys()

                # For each step, average across all seeds
                averaged_activations = []
                for step in range(num_steps):
                    step_dict = {}
                    for key in activation_keys:
                        # Average the same key across all seeds for this step
                        values = [
                            seed_activations[step][key]
                            for seed_activations in all_seed_activations
                        ]
                        step_dict[key] = np.mean(values)
                    averaged_activations.append(step_dict)

                tqdm.write(f"Appending activations for d_model={d_model}")
                d_model_activations = averaged_activations

        tqdm.write(f"Storing activations for d_model={d_model}")
        activations.append(d_model_activations)

    # Generate activation charts
    generate_activation_charts(d_model_values, activations, args.n_layers, args)

    # Generate loss charts if doing a learning rate sweep
    if doing_lr_sweep:
        generate_loss_charts(d_model_values, lr_combinations, losses, test_losses, args)


def generate_activation_charts(d_model_values, activations, n_layers, args):
    """
    Generate charts showing activation values across different model dimensions.

    Args:
        d_model_values: List of d_model values used in training
        activations: List of activation values for each model dimension
        n_layers: Number of MLP layers in the model
        args: Command-line arguments to include in the legend
    """
    # Create a colormap for train steps (only show first 10)
    num_train_steps = min(len(activations[0]), 10)
    colors = plt.cm.viridis(np.linspace(0, 1, num_train_steps))

    params = {
        "model_type": args.model_type,
        "dataset": args.dataset_path,
        "lr_base": args.lr_base if args.lr_base is not None else "N/A",
        "lr_range": f"{args.lr_low}-{args.lr_high}"
        if args.lr_low is not None
        else "N/A",
        "n_lr_points": args.n_lr_points if args.lr_low is not None else "N/A",
        "d_model_range": f"{args.d_model_low}-{args.d_model_high}",
        "reference_directions": args.reference_directions,
        "time_dim": args.time_dim,
        "use_pre_mlp_projection": args.use_pre_mlp_projection,
        "n_layers": args.n_layers,
        "mlp_expansion_factor": args.mlp_expansion_factor,
        "batch_size": args.batch_size,
        "n_seeds": args.n_seeds,
        "n_train_steps": args.n_train_steps,
        "n_test_batches": args.n_test_batches,
    }

    def create_chart(key, title, filename):
        """Helper function to create and save a chart for a specific activation type"""
        plt.figure(figsize=(18, 8))

        # Create a layout with two subplots - one for the chart, one for the legend
        gs = plt.GridSpec(1, 2, width_ratios=[3, 1])  # 3:1 ratio of chart to legend
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])

        # Plot activation values on the main axis (only first 10 steps)
        for step in range(num_train_steps):
            values = [model_activations[step][key] for model_activations in activations]
            ax1.plot(
                d_model_values,
                values,
                marker="o",
                color=colors[step],
                label=f"Step {step+1}",
            )

        # Configure the main chart
        ax1.set_xscale("log", base=2)
        ax1.set_ylim(bottom=0)
        ax1.set_xlabel("d_model (log scale)")
        ax1.set_ylabel("Mean Absolute Activation")
        ax1.set_title(title)
        ax1.grid(True, which="both", linestyle="--", alpha=0.6)

        # Create the steps legend on the first axis
        ax1.legend(loc="upper right")

        # Create a separate legend for parameters on the second axis
        ax2.axis("off")  # Turn off axis
        param_labels = [
            f"{param_name}: {param_value}" for param_name, param_value in params.items()
        ]
        ax2.text(0, 0.5, "\n".join(param_labels), va="center", fontsize=10)

        plt.tight_layout()
        plt.savefig(args.output_dir / filename, bbox_inches="tight", dpi=300)

    if args.model_type == "transformer":
        create_chart(
            "input_embedding",
            "Input Embedding Activations",
            "input_embedding_activation_chart.png",
        )

        create_chart(
            "clip_projection",
            "CLIP Projection Activations",
            "clip_projection_activation_chart.png",
        )

        for layer_idx in range(n_layers):
            create_chart(
                f"transformer_layer_{layer_idx}_attention_out",
                f"Transformer Layer {layer_idx} Attention Output Activations",
                f"transformer_layer_{layer_idx}_attention_out_chart.png",
            )

            create_chart(
                f"transformer_layer_{layer_idx}_linear_2",
                f"Transformer Layer {layer_idx} FFW Output Activations",
                f"transformer_layer_{layer_idx}_linear_2_chart.png",
            )

        create_chart(
            "final_layer_norm",
            "Final Layer Norm Activations",
            "final_layer_norm_activation_chart.png",
        )
    else:
        create_chart(
            "pre_mlp_projection",
            "Pre-MLP Projection Activations",
            "pre_mlp_activation_chart.png",
        )

        for layer_idx in range(n_layers):
            create_chart(
                f"mlp_block_{layer_idx}_out_proj",
                f"MLP Block {layer_idx} Output Projection Activations",
                f"mlp_block_{layer_idx}_out_proj_activation_chart.png",
            )

            create_chart(
                f"mlp_block_{layer_idx}_gate",
                f"MLP Block {layer_idx} Gate Activations",
                f"mlp_block_{layer_idx}_gate_activation_chart.png",
            )

            create_chart(
                f"mlp_block_{layer_idx}_value",
                f"MLP Block {layer_idx} Value Activations",
                f"mlp_block_{layer_idx}_value_activation_chart.png",
            )

    create_chart(
        "output_projection",
        "Model Output Projection Activations",
        "model_output_projection_activation_chart.png",
    )

    print("Charts generated successfully!")


def generate_loss_charts(d_model_values, lr_combinations, losses, test_losses, args):
    """
    Generate charts showing loss values across different learning rates for each d_model.

    Args:
        d_model_values: List of d_model values used in training
        lr_combinations: List of learning rate combinations used in training (tuples of (adam_lr, muon_lr))
        losses: 3D numpy array of shape (n_d_models, n_lr_combinations, n_steps) containing loss values
        test_losses: 2D numpy array of shape (n_d_models, n_lr_combinations) containing test loss values
        args: Command-line arguments to include in the legend
    """
    # Select 10 evenly spaced steps (or all steps if fewer than 10)
    num_plots = min(10, args.n_train_steps)
    if num_plots < args.n_train_steps:
        step_indices = np.linspace(0, args.n_train_steps - 1, num_plots, dtype=int)
    else:
        step_indices = np.array(range(args.n_train_steps))

    # Extract learning rate values for plotting
    if args.use_muon:
        # For mixed optimization, we'll plot against Adam LR with different lines for each Muon LR
        adam_lrs = [lr_combo[0] for lr_combo in lr_combinations]
        muon_lrs = [lr_combo[1] for lr_combo in lr_combinations]
        unique_adam_lrs = sorted(list(set(adam_lrs)))
        unique_muon_lrs = sorted(list(set(muon_lrs)))

        # Create a mapping from lr_combinations to indices
        lr_combo_to_idx = {combo: idx for idx, combo in enumerate(lr_combinations)}
    else:
        # For Adam-only, just extract the Adam learning rates
        adam_lrs = [lr_combo[0] for lr_combo in lr_combinations]

    # Generate charts for the selected steps
    for plot_idx, step_idx in enumerate(step_indices):
        plt.figure(figsize=(12, 8))

        for d_idx, d_model in enumerate(d_model_values):
            if args.use_muon:
                # For mixed optimization, plot separate lines for each Muon LR
                for muon_lr in unique_muon_lrs:
                    step_losses = []
                    x_values = []
                    for adam_lr in unique_adam_lrs:
                        if (adam_lr, muon_lr) in lr_combo_to_idx:
                            combo_idx = lr_combo_to_idx[(adam_lr, muon_lr)]
                            step_losses.append(losses[d_idx, combo_idx, step_idx])
                            x_values.append(adam_lr)

                    if step_losses:  # Only plot if we have data
                        plt.plot(
                            x_values,
                            step_losses,
                            marker="o",
                            label=f"d_model={d_model}, Muon_LR={muon_lr:.2e}",
                        )
            else:
                # For Adam-only optimization
                step_losses = losses[d_idx, :, step_idx]
                plt.plot(adam_lrs, step_losses, marker="o", label=f"d_model={d_model}")

        plt.xscale("log")
        plt.ylim(bottom=0)
        plt.xlabel("Adam Learning Rate (log scale)")
        plt.ylabel("Loss")
        title = f"Loss vs Learning Rate at Step {step_idx+1}/{args.n_train_steps}"
        if args.use_muon:
            title += " (Mixed Adam/Muon)"
        plt.title(title)
        plt.grid(True, which="both", linestyle="--", alpha=0.6)
        plt.legend()

        # Add parameter information as text
        param_info = (
            f"Dataset: {args.dataset_path}\n"
            f"d_model range: {args.d_model_low}-{args.d_model_high}\n"
        )
        if args.use_muon:
            param_info += (
                f"Adam LR range: {args.lr_low}-{args.lr_high}, n_points={args.n_lr_points}\n"
                f"Muon LR range: {args.muon_lr_low}-{args.muon_lr_high}, n_points={args.n_muon_lr_points}\n"
                f"Muon beta: {args.muon_beta}\n"
            )
        else:
            param_info += (
                f"LR range: {args.lr_low}-{args.lr_high}, n_points={args.n_lr_points}\n"
            )

        param_info += (
            f"reference_directions: {args.reference_directions}\n"
            f"time_dim: {args.time_dim}\n"
            f"pre_mlp_projection: {args.use_pre_mlp_projection}\n"
            f"n_layers: {args.n_layers}\n"
            f"mlp_expansion_factor: {args.mlp_expansion_factor}\n"
            f"batch_size: {args.batch_size}\n"
            f"n_seeds: {args.n_seeds}\n"
            f"n_test_batches: {args.n_test_batches}\n"
        )
        plt.figtext(0.01, 0.01, param_info, fontsize=8, va="bottom")

        plt.tight_layout()
        plt.savefig(
            args.output_dir / f"loss_vs_lr_step_{step_idx:06d}.png",
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()

    # Generate a chart for the test losses
    plt.figure(figsize=(15, 10))

    # Define colors for each d_model
    model_colors = plt.cm.viridis(np.linspace(0, 1, len(d_model_values)))

    # Track minimum loss points and their values
    min_points = []

    # Plot test losses
    for d_idx, d_model in enumerate(d_model_values):
        model_test_losses = test_losses[d_idx, :]

        # Find the learning rate combination with the lowest test loss
        min_loss_idx = np.argmin(model_test_losses)
        min_loss = model_test_losses[min_loss_idx]
        min_lr_combo = lr_combinations[min_loss_idx]

        # Store the minimum point information
        min_points.append((d_model, min_lr_combo, min_loss, model_colors[d_idx]))

        if args.use_muon:
            # For mixed optimization, plot separate lines for each Muon LR
            for muon_lr in unique_muon_lrs:
                test_losses_for_muon_lr = []
                x_values = []
                for adam_lr in unique_adam_lrs:
                    if (adam_lr, muon_lr) in lr_combo_to_idx:
                        combo_idx = lr_combo_to_idx[(adam_lr, muon_lr)]
                        test_losses_for_muon_lr.append(model_test_losses[combo_idx])
                        x_values.append(adam_lr)

                if test_losses_for_muon_lr:  # Only plot if we have data
                    line = plt.plot(
                        x_values,
                        test_losses_for_muon_lr,
                        marker="o",
                        label=f"d_model={d_model}, Muon_LR={muon_lr:.2e}",
                        color=model_colors[d_idx],
                        alpha=0.7,
                    )

                    # Highlight the minimum point if it's in this line
                    for i, (adam_lr, loss) in enumerate(
                        zip(x_values, test_losses_for_muon_lr)
                    ):
                        if (adam_lr, muon_lr) == min_lr_combo:
                            plt.plot(
                                adam_lr,
                                loss,
                                "o",
                                markersize=10,
                                color=model_colors[d_idx],
                                markeredgecolor="black",
                                markeredgewidth=2,
                            )
        else:
            # For Adam-only optimization
            line = plt.plot(
                adam_lrs,
                model_test_losses,
                marker="o",
                label=f"d_model={d_model}",
                color=model_colors[d_idx],
            )

            # Highlight the minimum point
            plt.plot(
                min_lr_combo[0],
                min_loss,
                "o",
                markersize=8,
                color=model_colors[d_idx],
                markeredgecolor="black",
                markeredgewidth=1.5,
            )

    # Create a table showing the best results
    table_data = []
    if args.use_muon:
        for d_model, min_lr_combo, min_loss, color in min_points:
            table_data.append(
                [
                    d_model,
                    f"{min_lr_combo[0]:.2e}",
                    f"{min_lr_combo[1]:.2e}",
                    f"{min_loss:.4f}",
                ]
            )
        col_labels = ["d_model", "Best Adam LR", "Best Muon LR", "Min Loss"]
        col_widths = [0.08, 0.12, 0.12, 0.08]
        bbox = [0.60, 0.60, 0.38, 0.30]
    else:
        for d_model, min_lr_combo, min_loss, color in min_points:
            table_data.append([d_model, f"{min_lr_combo[0]:.2e}", f"{min_loss:.4f}"])
        col_labels = ["d_model", "Best LR", "Min Loss"]
        col_widths = [0.1, 0.15, 0.1]
        bbox = [0.65, 0.65, 0.33, 0.25]

    # Get the current axes for the plot
    ax = plt.gca()

    # Create a table
    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        colWidths=col_widths,
        cellLoc="center",
        loc="upper right",
        bbox=bbox,
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)

    # Style the header row and data
    for (i, j), cell in table.get_celld().items():
        if i == 0:  # Header row
            cell.set_text_props(weight="bold")
            cell.set_facecolor("lightgray")
        else:  # Data rows
            if j == 0:  # d_model column
                cell.set_text_props(color=model_colors[i - 1], weight="bold")
            elif (not args.use_muon and j == 2) or (
                args.use_muon and j == 3
            ):  # Min loss column
                cell.set_text_props(weight="bold")

    plt.xscale("log")
    plt.ylim(bottom=0)
    plt.xlabel("Adam Learning Rate (log scale)")
    plt.ylabel("Test Loss")
    title = "Test Loss vs Learning Rate (Averaged Over Test Set)"
    if args.use_muon:
        title += " - Mixed Adam/Muon Optimization"
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend()

    # Add parameter information as text
    param_info = (
        f"Dataset: {args.dataset_path}\n"
        f"d_model range: {args.d_model_low}-{args.d_model_high}\n"
    )
    if args.use_muon:
        param_info += (
            f"Adam LR range: {args.lr_low}-{args.lr_high}, n_points={args.n_lr_points}\n"
            f"Muon LR range: {args.muon_lr_low}-{args.muon_lr_high}, n_points={args.n_muon_lr_points}\n"
            f"Muon beta: {args.muon_beta}\n"
        )
    else:
        param_info += (
            f"LR range: {args.lr_low}-{args.lr_high}, n_points={args.n_lr_points}\n"
        )

    param_info += (
        f"reference_directions: {args.reference_directions}\n"
        f"time_dim: {args.time_dim}\n"
        f"pre_mlp_projection: {args.use_pre_mlp_projection}\n"
        f"n_layers: {args.n_layers}\n"
        f"mlp_expansion_factor: {args.mlp_expansion_factor}\n"
        f"batch_size: {args.batch_size}\n"
        f"n_seeds: {args.n_seeds}\n"
        f"n_test_batches: {args.n_test_batches}\n"
    )
    plt.figtext(0.01, 0.01, param_info, fontsize=8, va="bottom")

    plt.tight_layout()
    plt.savefig(args.output_dir / "final_loss_vs_lr.png", bbox_inches="tight", dpi=300)

    # Generate 3D plots for mixed optimization
    if args.use_muon:
        generate_3d_loss_plots(d_model_values, lr_combinations, test_losses, args)

    print("Loss charts generated successfully!")


def generate_3d_loss_plots(d_model_values, lr_combinations, test_losses, args):
    """
    Generate 3D plots showing Adam LR vs Muon LR vs Test Loss.

    Args:
        d_model_values: List of d_model values used in training
        lr_combinations: List of (adam_lr, muon_lr) tuples
        test_losses: 2D numpy array of shape (n_d_models, n_lr_combinations)
        args: Command-line arguments
    """
    # Extract unique learning rates
    adam_lrs = [lr_combo[0] for lr_combo in lr_combinations]
    muon_lrs = [lr_combo[1] for lr_combo in lr_combinations]
    unique_adam_lrs = sorted(list(set(adam_lrs)))
    unique_muon_lrs = sorted(list(set(muon_lrs)))

    # Create grids for 3D plotting
    adam_grid, muon_grid = np.meshgrid(unique_adam_lrs, unique_muon_lrs)

    # Create a mapping from lr_combinations to indices for quick lookup
    lr_combo_to_idx = {combo: idx for idx, combo in enumerate(lr_combinations)}

    # Generate a 3D plot for each d_model
    for d_idx, d_model in enumerate(d_model_values):
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")

        # Create loss grid for this d_model
        loss_grid = np.full_like(adam_grid, np.nan)

        for i, adam_lr in enumerate(unique_adam_lrs):
            for j, muon_lr in enumerate(unique_muon_lrs):
                if (adam_lr, muon_lr) in lr_combo_to_idx:
                    combo_idx = lr_combo_to_idx[(adam_lr, muon_lr)]
                    loss_grid[j, i] = test_losses[d_idx, combo_idx]

        # Create the surface plot
        surf = ax.plot_surface(
            np.log10(adam_grid),
            np.log10(muon_grid),
            loss_grid,
            cmap="viridis",
            alpha=0.8,
            edgecolor="none",
        )

        # Add scatter points for actual data points
        adam_log = [np.log10(combo[0]) for combo in lr_combinations]
        muon_log = [np.log10(combo[1]) for combo in lr_combinations]
        losses_flat = test_losses[d_idx, :]

        scatter = ax.scatter(
            adam_log,
            muon_log,
            losses_flat,
            c=losses_flat,
            cmap="viridis",
            s=50,
            alpha=0.9,
            edgecolor="black",
            linewidth=0.5,
        )

        # Find and highlight the minimum loss point
        min_loss_idx = np.argmin(losses_flat)
        min_adam_lr, min_muon_lr = lr_combinations[min_loss_idx]
        min_loss = losses_flat[min_loss_idx]

        ax.scatter(
            [np.log10(min_adam_lr)],
            [np.log10(min_muon_lr)],
            [min_loss],
            c="red",
            s=200,
            marker="*",
            edgecolor="black",
            linewidth=2,
            label=f"Best: Adam={min_adam_lr:.2e}, Muon={min_muon_lr:.2e}, Loss={min_loss:.4f}",
        )

        # Set labels and title
        ax.set_xlabel("log₁₀(Adam Learning Rate)")
        ax.set_ylabel("log₁₀(Muon Learning Rate)")
        ax.set_zlabel("Test Loss")
        ax.set_title(
            f"Test Loss vs Learning Rates\nd_model={d_model}, muon_beta={args.muon_beta}"
        )

        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20, label="Test Loss")

        # Add legend
        ax.legend(loc="upper left")

        # Set custom tick formatters to show actual learning rates
        def log_tick_formatter(x, pos):
            return f"{10**x:.1e}"

        # Create custom ticks at reasonable intervals
        adam_log_min, adam_log_max = np.log10(min(unique_adam_lrs)), np.log10(
            max(unique_adam_lrs)
        )
        muon_log_min, muon_log_max = np.log10(min(unique_muon_lrs)), np.log10(
            max(unique_muon_lrs)
        )

        adam_ticks = np.linspace(
            adam_log_min, adam_log_max, min(5, len(unique_adam_lrs))
        )
        muon_ticks = np.linspace(
            muon_log_min, muon_log_max, min(5, len(unique_muon_lrs))
        )

        ax.set_xticks(adam_ticks)
        ax.set_yticks(muon_ticks)
        ax.set_xticklabels([f"{10**x:.1e}" for x in adam_ticks])
        ax.set_yticklabels([f"{10**x:.1e}" for x in muon_ticks])

        # Rotate tick labels for better readability
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=45)

        plt.tight_layout()
        plt.savefig(
            args.output_dir / f"3d_loss_surface_d_model_{d_model}.png",
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()

    # Generate a combined 3D plot with all d_model values
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection="3d")

    # Define colors for each d_model
    model_colors = plt.cm.tab10(np.linspace(0, 1, len(d_model_values)))

    # Plot scatter points for each d_model
    for d_idx, d_model in enumerate(d_model_values):
        adam_log = [np.log10(combo[0]) for combo in lr_combinations]
        muon_log = [np.log10(combo[1]) for combo in lr_combinations]
        losses_flat = test_losses[d_idx, :]

        scatter = ax.scatter(
            adam_log,
            muon_log,
            losses_flat,
            c=[model_colors[d_idx]] * len(losses_flat),
            s=60,
            alpha=0.7,
            label=f"d_model={d_model}",
            edgecolor="black",
            linewidth=0.5,
        )

        # Highlight the minimum loss point for this d_model
        min_loss_idx = np.argmin(losses_flat)
        min_adam_lr, min_muon_lr = lr_combinations[min_loss_idx]
        min_loss = losses_flat[min_loss_idx]

        ax.scatter(
            [np.log10(min_adam_lr)],
            [np.log10(min_muon_lr)],
            [min_loss],
            c=[model_colors[d_idx]],
            s=200,
            marker="*",
            edgecolor="black",
            linewidth=2,
            alpha=1.0,
        )

    # Set labels and title
    ax.set_xlabel("log₁₀(Adam Learning Rate)")
    ax.set_ylabel("log₁₀(Muon Learning Rate)")
    ax.set_zlabel("Test Loss")
    ax.set_title(
        f"Test Loss vs Learning Rates (All d_model values)\nmuon_beta={args.muon_beta}"
    )

    # Add legend
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98))

    # Set custom ticks
    adam_log_min, adam_log_max = np.log10(min(unique_adam_lrs)), np.log10(
        max(unique_adam_lrs)
    )
    muon_log_min, muon_log_max = np.log10(min(unique_muon_lrs)), np.log10(
        max(unique_muon_lrs)
    )

    adam_ticks = np.linspace(adam_log_min, adam_log_max, min(5, len(unique_adam_lrs)))
    muon_ticks = np.linspace(muon_log_min, muon_log_max, min(5, len(unique_muon_lrs)))

    ax.set_xticks(adam_ticks)
    ax.set_yticks(muon_ticks)
    ax.set_xticklabels([f"{10**x:.1e}" for x in adam_ticks])
    ax.set_yticklabels([f"{10**x:.1e}" for x in muon_ticks])

    # Rotate tick labels for better readability
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=45)

    # Add parameter information as text
    param_text = (
        f"Adam LR: {args.lr_low:.1e} - {args.lr_high:.1e} ({args.n_lr_points} points)\n"
        f"Muon LR: {args.muon_lr_low:.1e} - {args.muon_lr_high:.1e} ({args.n_muon_lr_points} points)\n"
        f"d_model range: {args.d_model_low} - {args.d_model_high}\n"
        f"Stars (*) indicate best LR combination for each d_model"
    )

    # Position text in the plot
    ax.text2D(
        0.02,
        0.02,
        param_text,
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(
        args.output_dir / "3d_loss_surface_combined.png", bbox_inches="tight", dpi=300
    )
    plt.close()

    print("3D loss surface plots generated successfully!")


if __name__ == "__main__":
    main()
