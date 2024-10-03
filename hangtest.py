# debugging hang with cpp flash attention on multi-gpu
from txt2img_unsupervised.checkpoint import TrainState
from txt2img_unsupervised.config import LearningRateSchedule, ModelConfig, TrainingConfig
from txt2img_unsupervised.transformer_model import (
    AttnMethod,
    ImageModel,
    TransformerLayer,
    gpt_1_config,
    loss_batch,
)

import jax
import jax.numpy as jnp

from copy import deepcopy
from einops import repeat
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jax.experimental import mesh_utils

devices = mesh_utils.create_device_mesh((jax.device_count(),))
mesh = Mesh(devices, axis_names=("dev",))


def mk_tl(attn=AttnMethod.STANDARD):
    return TransformerLayer(
        d_model=32,
        num_heads=2,
        ff_dim=128,
        dropout=None,
        use_biases=True,
        activations_dtype=jnp.bfloat16,
        activation_function=jax.nn.gelu,
        pre_norm=True,
        kernel_init=jax.nn.initializers.normal(stddev=0.02),
        out_proj_kernel_init=jax.nn.initializers.normal(stddev=0.02),
        decode=False,
        attn_method=AttnMethod.STANDARD,
        record_attention_weights=False,
    )


tl_s = mk_tl(AttnMethod.STANDARD)
tl_fj = mk_tl(AttnMethod.FLASH_JAX)
tl_fc = mk_tl(AttnMethod.FLASH_CPP)

tl_params = mk_tl().init(jax.random.PRNGKey(0), jnp.zeros((1, 2, 32)), None)
tl_params_replicated = jax.device_put(
    tl_params, NamedSharding(mesh, PartitionSpec(None))
)

tl_inputs = jax.random.normal(jax.random.PRNGKey(1), (2, 2, 32))
tl_inputs_sharded = jax.device_put(tl_inputs, NamedSharding(mesh, PartitionSpec("dev")))

tl_loss_fn = lambda mdl, params, inputs: (mdl.apply(params, inputs, None)[0].sum()) ** 2

tl_grad_fn = jax.grad(tl_loss_fn, argnums=1)

mdl_s = ImageModel(**gpt_1_config, attn_method=AttnMethod.STANDARD)
mdl_fj = ImageModel(**gpt_1_config, attn_method=AttnMethod.FLASH_JAX)
mdl_fc = ImageModel(**gpt_1_config, attn_method=AttnMethod.FLASH_CPP)

dummy_images, dummy_clip_embeddings, dummy_max_cos_distance = mdl_s.dummy_inputs()
batch_images = repeat(dummy_images, "1 t -> b t", b=4)
batch_clip_embeddings = repeat(dummy_clip_embeddings, "1 c -> b c", b=4)
batch_max_cos_distance = repeat(dummy_max_cos_distance, "1 d -> b d", b=4)

ts = TrainState.new(
    jax.random.PRNGKey(0),
    mdl_s,
    TrainingConfig(
        learning_rate=1e-3,
        batch_size=4,
        epochs=1,
        learning_rate_schedule=LearningRateSchedule.WARMUP_PLUS_COSINE,
        gradient_accumulation_steps=1,
        gradient_clipping=None,
        warmup_steps=400,
        schedule_free_beta1=0.98,
        training_images=0,
        loss_decay_constant=1.0,
    ),
    1000,
)

ts_replicated = ts.replicate_for_multi_gpu()

mdl_grad_fn = jax.grad(loss_batch, argnums=1)