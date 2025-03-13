import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import jax
import jax.numpy as jnp
from mpl_toolkits.mplot3d import Axes3D
import argparse
import pandas as pd
import os

# Import the necessary components from flow_matching.py
from flow_matching import (
    VectorField,
    create_train_state,
    sample_sphere,
)


def save_vector_field_to_csv(points, times, vector_field_values, output_csv):
    """
    Save vector field data to a CSV file.

    Args:
        points: Points on the sphere/circle [n_samples, domain_dim]
        times: Time values [n_samples]
        vector_field_values: Vector field values at points [n_samples, domain_dim]
        output_csv: Path to save the CSV file
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)

    domain_dim = points.shape[1]

    if domain_dim == 2:
        df = pd.DataFrame(
            {
                "x": points[:, 0],
                "y": points[:, 1],
                "time": times,
                "field_x": vector_field_values[:, 0],
                "field_y": vector_field_values[:, 1],
                "field_magnitude": np.linalg.norm(vector_field_values, axis=1),
            }
        )
    else:
        df = pd.DataFrame(
            {
                "x": points[:, 0],
                "y": points[:, 1],
                "z": points[:, 2],
                "time": times,
                "field_x": vector_field_values[:, 0],
                "field_y": vector_field_values[:, 1],
                "field_z": vector_field_values[:, 2],
                "field_magnitude": np.linalg.norm(vector_field_values, axis=1),
            }
        )

    df.to_csv(output_csv, index=False)
    print(f"Vector field data saved to {output_csv}")


def save_trajectory_data_to_csv(trajectory_points, time_values, output_csv):
    """
    Save trajectory data to a CSV file.

    Args:
        trajectory_points: List of arrays of trajectory points
        time_values: List of arrays of time values for each trajectory
        output_csv: Path to save the CSV file
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)

    traj_data = []

    for traj_idx, (points, times) in enumerate(zip(trajectory_points, time_values)):
        domain_dim = points.shape[1]

        for step_idx, (point, t) in enumerate(zip(points, times)):
            point_data = {
                "trajectory_id": traj_idx + 1,
                "step": step_idx,
                "time": t,
            }

            if domain_dim == 2:
                point_data.update(
                    {
                        "x": point[0],
                        "y": point[1],
                    }
                )
            else:
                point_data.update(
                    {
                        "x": point[0],
                        "y": point[1],
                        "z": point[2],
                    }
                )

            traj_data.append(point_data)

    df = pd.DataFrame(traj_data)
    df.to_csv(output_csv, index=False)
    print(f"Trajectory data saved to {output_csv}")


def visualize_vector_field(
    model_config=None,
    seed=42,
    n_samples=100,
    arrow_scale=0.15,
    time=0.5,
    output=None,
    output_csv=None,
):
    """
    Visualize a vector field on a sphere or circle.

    Args:
        model_config: Dictionary with model configuration parameters
        seed: Random seed for reproducibility
        n_samples: Number of sample points for visualization
        arrow_scale: Scaling factor for the arrows
        time: Time value at which to visualize the vector field
        output: Path to save the visualization. If None, the plot is displayed.
        output_csv: Path to save the vector field data as CSV. If None, no CSV is generated.
    """
    params_rng, points_rng = jax.random.split(jax.random.PRNGKey(seed))

    if model_config is None:
        model_config = {
            "domain_dim": 3,
            "conditioning_dim": 0,
            "n_layers": 2,
            "d_model": 32,
            "mlp_expansion_factor": 4,
        }

    model = VectorField(
        activations_dtype=jnp.float32, weights_dtype=jnp.float32, **model_config
    )

    is_2d = model.domain_dim == 2

    state = create_train_state(params_rng, model, learning_rate_or_schedule=1e-3)

    points = sample_sphere(points_rng, n_samples, model.domain_dim)

    times = jnp.full((n_samples,), time)

    cond_vecs = jnp.zeros((n_samples, model.conditioning_dim))

    vector_field_values = model.apply(state.params, points, times, cond_vecs)

    points_np = np.array(points)
    vectors_np = np.array(vector_field_values)

    if output_csv:
        save_vector_field_to_csv(points_np, np.array(times), vectors_np, output_csv)

    fig = plt.figure(figsize=(10, 10))

    if is_2d:
        ax = fig.add_subplot(111)

        theta = np.linspace(0, 2 * np.pi, 100)
        circle_x = np.cos(theta)
        circle_y = np.sin(theta)
        ax.plot(circle_x, circle_y, color="gray", alpha=0.2)

        for i in range(n_samples):
            ax.arrow(
                points_np[i, 0],
                points_np[i, 1],
                vectors_np[i, 0] * arrow_scale,
                vectors_np[i, 1] * arrow_scale,
                head_width=0.05,
                head_length=0.08,
                fc="blue",
                ec="blue",
                alpha=0.7,
            )

        ax.set_aspect("equal")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        ax.set_xlim([-1.2, 1.2])
        ax.set_ylim([-1.2, 1.2])

    else:
        ax = fig.add_subplot(111, projection="3d")

        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        sphere_x = np.outer(np.cos(u), np.sin(v))
        sphere_y = np.outer(np.sin(u), np.sin(v))
        sphere_z = np.outer(np.ones(np.size(u)), np.cos(v))

        ax.plot_surface(
            sphere_x,
            sphere_y,
            sphere_z,
            color="lightgray",
            alpha=0.7,
            edgecolor="gray",
            linewidth=0.5,
            shade=True,
        )

        for i in range(n_samples):
            ax.quiver(
                points_np[i, 0],
                points_np[i, 1],
                points_np[i, 2],
                vectors_np[i, 0],
                vectors_np[i, 1],
                vectors_np[i, 2],
                color="blue",
                length=arrow_scale,
                normalize=True,
                alpha=0.7,
            )

        ax.set_box_aspect([1, 1, 1])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-1.1, 1.1])
        ax.set_zlim([-1.1, 1.1])

    shape_name = "Circle (S¹)" if is_2d else "Sphere (S²)"
    ax.set_title(
        f"Vector Field on {shape_name}\n"
        f"{model.n_layers} layers, {model.d_model} width, t={time}",
        pad=20,
    )

    info_text = (
        f"Model configuration:\n"
        f"- Domain dimension: {model.domain_dim} ({'Circle' if is_2d else 'Sphere'})\n"
        f"- Layers: {model.n_layers}\n"
        f"- Width: {model.d_model}\n"
        f"- Expansion factor: {model.mlp_expansion_factor}\n"
        f"- Time: {time}\n"
        f"- Samples: {n_samples}"
    )

    if output_csv:
        info_text += f"\n- Data saved to: {os.path.basename(output_csv)}"

    plt.figtext(0.02, 0.02, info_text, fontsize=10)

    if output:
        plt.savefig(output, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {output}")
    else:
        plt.show()

    plt.close(fig)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize vector fields on sphere or circle."
    )

    parser.add_argument(
        "--output", type=str, default=None, help="Path to save the visualization"
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Path to save the vector field data as CSV. If not provided, no CSV is generated.",
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of sample points for visualization",
    )
    parser.add_argument(
        "--arrow-scale", type=float, default=0.15, help="Scaling factor for the arrows"
    )
    parser.add_argument(
        "--time",
        type=float,
        default=0.5,
        help="Time value at which to visualize the vector field (without trajectories)",
    )

    parser.add_argument(
        "--domain-dim",
        type=int,
        default=3,
        choices=[2, 3],
        help="Dimension of the domain (2 for circle, 3 for sphere)",
    )
    parser.add_argument(
        "--conditioning-dim",
        type=int,
        default=0,
        help="Dimension of the conditioning vector",
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
        "domain_dim": args.domain_dim,
        "conditioning_dim": args.conditioning_dim,
        "n_layers": args.n_layers,
        "d_model": args.d_model,
        "mlp_expansion_factor": args.expansion_factor,
    }

    visualize_vector_field(
        model_config=model_config,
        seed=args.seed,
        n_samples=args.samples,
        arrow_scale=args.arrow_scale,
        time=args.time,
        output=args.output,
        output_csv=args.output_csv,
    )
