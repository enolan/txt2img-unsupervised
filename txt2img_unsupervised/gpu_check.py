import jax


def gpu_is_ampere_or_newer() -> bool:
    """Check if the GPU is an Ampere or newer architecture. Warning: this will fail to detect GPUs
    released after ca September 2024."""
    gpu_devices = jax.devices("gpu")
    if len(gpu_devices) == 0:
        return False
    device_kind = gpu_devices[0].device_kind
    # See https://en.wikipedia.org/wiki/List_of_Nvidia_graphics_processing_units
    rtx_3k = [f"RTX 30{i:02d}" for i in range(1, 91, 10)]
    rtx_4k = [f"RTX 40{i:02d}" for i in range(1, 91, 10)]
    datacenter_ampere_gpus = [f"A{i}" for i in [2, 10, 16, 30, 40, 100]]
    datacenter_hopper_and_lovelace_gpus = ["H100", "L40", "L4"]

    acceptable_gpus = (
        rtx_3k + rtx_4k + datacenter_ampere_gpus + datacenter_hopper_and_lovelace_gpus
    )
    return device_kind in acceptable_gpus
