from matplotlib import cm, animation
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import argparse
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import os

from .flow_matching import VectorField, create_train_state, sample_sphere


def create_rotating_animation(
    save_path=None,
    n_samples=300,
    fps=30,
    duration=5,  # Duration in seconds
    rotation_degrees=180,  # Total rotation in degrees
    seed=42,
    model_config=None,
    arrow_scale=0.15,
    time=0.5,
    animate_time=False,
    time_start=0.0,
    time_end=1.0,
):
    """
    Create an animation of a vector field on a sphere, rotating the view.

    Args:
        save_path: Path to save the animation (as mp4 or gif). If None, the animation is displayed.
        n_samples: Number of sample points for the vector field
        fps: Frames per second in the output animation
        duration: Duration of the animation in seconds
        rotation_degrees: Total rotation in degrees
        seed: Random seed for reproducibility
        model_config: Optional dictionary with model configuration parameters
        arrow_scale: Scaling factor for the arrows representing the vector field
        time: Time value at which to visualize the vector field (used when animate_time=False)
        animate_time: Whether to animate the time parameter
        time_start: Starting time value for animation (when animate_time=True)
        time_end: Ending time value for animation (when animate_time=True)

    Returns:
        anim: The animation object
    """
    n_frames = int(duration * fps)

    rng = jax.random.PRNGKey(seed)
    params_rng, points_rng = jax.random.split(rng)

    if model_config is None:
        model_config = {
            "domain_dim": 3,  # Only support 3D for sphere rotation
            "conditioning_dim": 0,
            "n_layers": 2,
            "d_model": 32,
            "mlp_expansion_factor": 4,
        }

    print("Initializing model")
    model = VectorField(
        activations_dtype=jnp.float32, weights_dtype=jnp.float32, **model_config
    )

    state = create_train_state(params_rng, model, learning_rate_or_schedule=1e-3)

    print("Sampling points")
    points = sample_sphere(points_rng, n_samples, model.domain_dim)
    points_np = np.array(points)

    cond_vecs = jnp.zeros((n_samples, model.conditioning_dim))

    # If we're animating time, we'll compute vector fields for each frame later
    if not animate_time:
        print("Evaluating vector field")
        times = jnp.full((n_samples,), time)
        vector_field_values = model.apply(state.params, points, times, cond_vecs)
        vectors_np = np.array(vector_field_values)
    else:
        print("Time animation enabled - vector field will be computed for each frame")
        # We'll compute this during animation

    print("Setting up animation")
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    sphere_x = np.outer(np.cos(u), np.sin(v))
    sphere_y = np.outer(np.sin(u), np.sin(v))
    sphere_z = np.outer(np.ones(np.size(u)), np.cos(v))

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    title_time = time if not animate_time else time_start
    ax.set_title(
        f"Vector Field on Sphere (S²)\n"
        f"{model.n_layers} layers, {model.d_model} width, t={title_time:.2f}",
        pad=20,
    )

    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_zlim([-1.1, 1.1])

    angle_per_frame = rotation_degrees / n_frames

    if animate_time:
        time_values = np.linspace(time_start, time_end, n_frames)

    # Create initial surface and quiver plots
    surface = ax.plot_surface(
        sphere_x,
        sphere_y,
        sphere_z,
        color="lightgray",
        alpha=0.7,
        edgecolor="gray",
        linewidth=0.5,
        shade=True,
    )

    if animate_time:
        # Compute initial vector field
        times = jnp.full((n_samples,), time_values[0])
        vector_field_values = model.apply(state.params, points, times, cond_vecs)
        vectors_np = np.array(vector_field_values)

    # Store quiver plot in a list to allow updating it from the update function
    quiver_container = [
        ax.quiver(
            points_np[:, 0],
            points_np[:, 1],
            points_np[:, 2],
            vectors_np[:, 0],
            vectors_np[:, 1],
            vectors_np[:, 2],
            color="blue",
            length=arrow_scale,
            normalize=True,
            alpha=0.7,
        )
    ]

    def init():
        # For 3D animations, it's better to return an empty list
        # The objects are already added to the axes
        return []

    def update(frame):
        # Update rotation
        elev = 30
        azim = frame * angle_per_frame
        ax.view_init(elev=elev, azim=azim)

        # Update time and vector field if animating time
        if animate_time:
            current_time = time_values[frame]

            # Update title with current time
            ax.set_title(
                f"Vector Field on Sphere (S²)\n"
                f"{model.n_layers} layers, {model.d_model} width, t={current_time:.2f}",
                pad=20,
            )

            # Compute new vector field for current time
            times = jnp.full((n_samples,), current_time)
            vector_field_values = model.apply(state.params, points, times, cond_vecs)
            vectors_np = np.array(vector_field_values)

            # For 3D quiver plots, we need to remove the old quiver and create a new one
            # since set_UVC only works for 2D quiver plots
            quiver_container[0].remove()
            quiver_container[0] = ax.quiver(
                points_np[:, 0],
                points_np[:, 1],
                points_np[:, 2],
                vectors_np[:, 0],
                vectors_np[:, 1],
                vectors_np[:, 2],
                color="blue",
                length=arrow_scale,
                normalize=True,
                alpha=0.7,
            )

        # In 3D animations, it's better to return an empty list
        # The objects are already added to the axes
        return []

    # Create info text
    if animate_time:
        time_info = f"- Time: {time_start:.2f} to {time_end:.2f} (animated)"
    else:
        time_info = f"- Time: {time:.2f} (fixed)"

    info_text = (
        f"Model configuration:\n"
        f"- Domain dimension: {model.domain_dim} (Sphere)\n"
        f"- Layers: {model.n_layers}\n"
        f"- Width: {model.d_model}\n"
        f"- Expansion factor: {model.mlp_expansion_factor}\n"
        f"{time_info}\n"
        f"- Samples: {n_samples}\n"
        f"- Rotation: {rotation_degrees}° over {duration}s\n"
        f"- Seed: {seed}"
    )

    plt.figtext(0.02, 0.02, info_text, fontsize=10)

    print("Creating animation")
    anim = animation.FuncAnimation(
        fig=fig,
        func=update,
        init_func=init,
        frames=n_frames,
        interval=1000 / fps,  # milliseconds between frames
        blit=False,  # Redraw the whole figure (needed for 3D)
    )

    pbar = tqdm(total=n_frames)

    if save_path:
        # Create a callback to update the progress bar
        def progress_callback(current_frame, total_frames):
            pbar.n = current_frame + 1
            pbar.refresh()

        print(f"Saving animation to {save_path}...")
        if save_path.endswith(".mp4"):
            temp_save_path = save_path.replace(".mp4", ".temp.mp4")
            writer = animation.FFMpegWriter(
                fps=fps, metadata=dict(artist="VectorFieldViz")
            )
            anim.save(
                temp_save_path, writer=writer, progress_callback=progress_callback
            )
        elif save_path.endswith(".gif"):
            temp_save_path = save_path.replace(".gif", ".temp.gif")
            anim.save(
                temp_save_path,
                writer="pillow",
                fps=fps,
                progress_callback=progress_callback,
            )
        else:
            raise ValueError("Save path must end with .mp4 or .gif")
        os.rename(temp_save_path, save_path)
        print(f"Animation saved to {save_path}")
    else:
        plt.show()

    pbar.close()

    return anim


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create a rotating animation of a vector field on a sphere."
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the animation (must end with .mp4 or .gif)",
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="Frames per second for the animation"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Duration of the animation in seconds",
    )
    parser.add_argument(
        "--rotation",
        type=float,
        default=180.0,
        help="Degrees to rotate during the animation",
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=300,
        help="Number of sample points for visualization",
    )
    parser.add_argument(
        "--arrow-scale", type=float, default=0.15, help="Scaling factor for the arrows"
    )

    # Time parameters
    time_group = parser.add_argument_group("Time parameters")
    time_group.add_argument(
        "--time",
        type=float,
        default=0.5,
        help="Time value at which to visualize the vector field (when not animating time)",
    )
    time_group.add_argument(
        "--animate-time",
        action="store_true",
        help="Animate the time parameter along with rotation",
    )
    time_group.add_argument(
        "--time-start",
        type=float,
        default=0.0,
        help="Starting time value for animation (when animating time)",
    )
    time_group.add_argument(
        "--time-end",
        type=float,
        default=1.0,
        help="Ending time value for animation (when animating time)",
    )

    parser.add_argument(
        "--n-layers", type=int, default=2, help="Number of layers in the model"
    )
    parser.add_argument("--d-model", type=int, default=32, help="Width of the model")
    parser.add_argument(
        "--expansion-factor", type=int, default=4, help="Expansion factor for the MLP"
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    model_config = {
        "domain_dim": 3,  # Always 3D for sphere
        "conditioning_dim": 0,
        "n_layers": args.n_layers,
        "d_model": args.d_model,
        "mlp_expansion_factor": args.expansion_factor,
    }

    anim = create_rotating_animation(
        save_path=args.output,
        n_samples=args.samples,
        fps=args.fps,
        duration=args.duration,
        rotation_degrees=args.rotation,
        seed=args.seed,
        model_config=model_config,
        arrow_scale=args.arrow_scale,
        time=args.time,
        animate_time=args.animate_time,
        time_start=args.time_start,
        time_end=args.time_end,
    )
