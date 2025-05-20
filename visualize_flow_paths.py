import argparse
from pathlib import Path
import sys # For sys.exit

import jax
import jax.numpy as jnp
import matplotlib
# matplotlib.use('Agg') # Uncomment if running in a headless environment and issues arise
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from txt2img_unsupervised.checkpoint import load_params
from txt2img_unsupervised.flow_matching import generate_samples, VectorField
try:
    from txt2img_unsupervised.flow_matching import CapConditionedVectorField
except ImportError:
    CapConditionedVectorField = None 
from txt2img_unsupervised.training_infra import setup_jax_for_training

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Visualize flow paths from a trained model.")
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        required=True,
        help="Directory containing model checkpoints and configuration.",
    )
    parser.add_argument(
        "--output_video_path",
        type=Path,
        default=Path("flow_animation.mp4"),
        help="Path to save the output animation video.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Specific checkpoint step to load. If None, loads the latest.",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=10, 
        help="Number of flow paths to visualize.",
    )
    parser.add_argument(
        "--batch_size", 
        type=int,
        default=10,
        help="Batch size for generating samples. Should be >= n_samples for current script version.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for JAX PRNG.",
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=100,
        help="Number of integration steps for the ODE solver.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Spherical Flow Paths Animation",
        help="Title for the animation.",
    )
    parser.add_argument(
        "--figure_size",
        type=str,
        default="10,8",
        help="Comma-separated width and height for the Matplotlib figure (e.g., \"10,8\").",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=100,
        help="DPI for the output video.",
    )
    parser.add_argument(
        "--frames_per_second",
        type=int,
        default=30,
        help="FPS for the output video.",
    )
    parser.add_argument(
        "--initial_point_size",
        type=float,
        default=20,
        help="Size of the scatter plot markers for the initial x0 points.",
    )
    parser.add_argument(
        "--path_point_size",
        type=float,
        default=5,
        help="Size of the scatter plot markers for points along the paths during animation.",
    )
    parser.add_argument(
        "--path_line_width",
        type=float,
        default=0.5,
        help="Line width for the path trails.",
    )
    parser.add_argument(
        "--initial_point_color",
        type=str,
        default="blue",
        help="Color for the initial x0 points.",
    )
    parser.add_argument(
        "--path_color",
        type=str,
        default="red",
        help="Color for the animated paths.",
    )
    parser.add_argument(
        "--camera_elevation",
        type=float,
        default=30,
        help="Initial camera elevation angle (degrees) for the 3D plot.",
    )
    parser.add_argument(
        "--camera_azimuth",
        type=float,
        default=-60,
        help="Initial camera azimuth angle (degrees) for the 3D plot.",
    )
    parser.add_argument(
        "--skip_animation_frames",
        type=int,
        default=1,
        help="Render only every Nth frame of path trajectories (N > 0).",
    )
    parser.add_argument(
        "--animation_interval",
        type=int,
        default=20,
        help="Delay between frames in milliseconds for FuncAnimation.",
    )
    
    args = parser.parse_args()

    if args.skip_animation_frames <= 0:
        raise ValueError("--skip_animation_frames must be > 0")

    if args.n_samples > args.batch_size:
        print(f"Warning: n_samples ({args.n_samples}) > batch_size ({args.batch_size}). "
              f"Clipping n_samples to batch_size for this script version.")
        args.n_samples = args.batch_size
    return args

def main():
    """Main function to generate and visualize flow paths."""
    args = parse_arguments()

    setup_jax_for_training()

    print(f"Loading model from checkpoint directory: {args.checkpoint_dir}, step: {args.step}")
    mdl, params = load_params(args.checkpoint_dir, args.step)
    print(f"Model loaded. Type: {type(mdl)}")

    if mdl.domain_dim != 3:
        print(f"Error: Visualization is only supported for 3D spherical flows (mdl.domain_dim == 3). Model has domain_dim = {mdl.domain_dim}")
        sys.exit(1)

    rng = jax.random.PRNGKey(args.seed)
    sample_rng, cap_rng = jax.random.split(rng)

    cond_vecs = None
    cap_centers = None
    cap_d_maxes = None

    if CapConditionedVectorField and isinstance(mdl, CapConditionedVectorField):
        print("Model is CapConditionedVectorField. Setting up for unconditioned (full sphere) sampling.")
        cap_centers = jax.random.normal(cap_rng, (args.n_samples, mdl.domain_dim))
        cap_centers = cap_centers / jnp.linalg.norm(cap_centers, axis=1, keepdims=True)
        cap_d_maxes = jnp.full((args.n_samples,), 2.0)
    elif isinstance(mdl, VectorField):
        print("Model is VectorField. Setting cond_vecs for unconditioned sampling if needed.")
        if mdl.conditioning_dim > 0:
            cond_vecs = jnp.zeros((args.n_samples, mdl.conditioning_dim))
    else:
        print(f"Warning: Unknown model type {type(mdl)}. Assuming no special conditioning needed.")

    print(f"Generating {args.n_samples} flow paths with {args.n_steps} ODE steps...")
    final_samples, paths = generate_samples(
        model=mdl,
        params=params,
        rng=sample_rng,
        batch_size=args.n_samples,
        cond_vecs=cond_vecs,
        cap_centers=cap_centers,
        cap_d_maxes=cap_d_maxes,
        n_steps=args.n_steps,
        return_paths=True,
        method="rk4"
    )

    print(f"Generated final_samples shape: {final_samples.shape}")
    print(f"Generated paths shape: {paths.shape}")

    # 1. Prepare Data for Animation
    paths_for_animation = paths[:, ::args.skip_animation_frames, :]
    num_animation_frames = paths_for_animation.shape[1]
    initial_points = paths_for_animation[:, 0, :]
    print(f"Number of frames for animation after skipping: {num_animation_frames}")

    # 2. Set up the Matplotlib Figure and 3D Axes
    fig_width, fig_height = map(int, args.figure_size.split(','))
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=args.dpi)
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=args.camera_elevation, azim=args.camera_azimuth)

    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    if args.title:
        ax.set_title(args.title)

    # 3. Draw Static Elements (Transparent Sphere)
    u_s = np.linspace(0, 2 * np.pi, 100)
    v_s = np.linspace(0, np.pi, 100)
    x_s = np.outer(np.cos(u_s), np.sin(v_s))
    y_s = np.outer(np.sin(u_s), np.sin(v_s))
    z_s = np.outer(np.ones(np.size(u_s)), np.cos(v_s))
    ax.plot_surface(x_s, y_s, z_s, color='gray', alpha=0.1, rstride=5, cstride=5, linewidth=0, antialiased=False)

    # 4. Initialize Animated Elements
    ax.scatter(initial_points[:, 0], initial_points[:, 1], initial_points[:, 2],
               s=args.initial_point_size, color=args.initial_point_color, label="Initial Positions (x0)")

    path_heads = ax.scatter(np.array([]), np.array([]), np.array([]),
                            s=args.path_point_size, color=args.path_color, depthshade=True, label="Current Positions")

    path_lines = [ax.plot([], [], [], lw=args.path_line_width, color=args.path_color, alpha=0.7)[0]
                  for _ in range(args.n_samples)]
    
    ax.legend()

    # 5. Create update function for FuncAnimation
    def update(frame_num, paths_data, heads_scatter, lines_list):
        current_points = paths_data[:, frame_num, :]
        heads_scatter._offsets3d = (current_points[:, 0], current_points[:, 1], current_points[:, 2])

        for i in range(args.n_samples):
            trail = paths_data[i, :frame_num + 1, :]
            lines_list[i].set_data(trail[:, 0], trail[:, 1])
            lines_list[i].set_3d_properties(trail[:, 2])
        
        return [heads_scatter] + lines_list

    # 6. Create and Save Animation
    print(f"Creating animation with {num_animation_frames} frames...")
    # Using blit=False as blit=True often causes issues with 3D artists in Matplotlib.
    ani = animation.FuncAnimation(
        fig, update, frames=num_animation_frames,
        fargs=(paths_for_animation, path_heads, path_lines),
        interval=args.animation_interval, blit=False 
    )

    try:
        FFMpegWriter = animation.writers['ffmpeg']
        writer = FFMpegWriter(fps=args.frames_per_second)
        print(f"Saving animation to {args.output_video_path} (this may take a while)...")
        ani.save(str(args.output_video_path), writer=writer)
        print(f"Animation saved successfully to {args.output_video_path}")
    except FileNotFoundError:
        print("Error: ffmpeg not found. Please install ffmpeg to save the animation as MP4.")
        print("You might need to install it using: sudo apt-get install ffmpeg")
        print("Alternatively, consider saving as a GIF if ffmpeg is not available (requires Pillow writer):")
        # gif_path = args.output_video_path.with_suffix('.gif')
        # print(f"Attempting to save as GIF to: {gif_path}")
        # try:
        #     ani.save(str(gif_path), writer='pillow', fps=args.frames_per_second)
        #     print(f"Animation saved successfully as GIF to {gif_path}")
        # except Exception as e_gif:
        #     print(f"Failed to save as GIF: {e_gif}")
    except Exception as e:
        print(f"An error occurred during animation saving: {e}")
    finally:
        plt.close(fig) 

    print("Done!")

if __name__ == "__main__":
    main()
