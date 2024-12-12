import argparse
from pathlib import Path
from typing import Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console

from checkpoint import Checkpoint
from environment import Environment
from locomotion import ReflexLocomotionControl
from model import Model

console = Console()


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=Path,
        required=True,
        help="Directory to analyze",
    )
    parser.add_argument("-m", "--model-path", type=Path, help="")
    parser.add_argument("-e", "--exoskeleton", action="store_true", help="")
    parser.add_argument("-v", "--visualize", action="store_true", help="")
    parser.add_argument("-d", "--simulation-duration", type=int, default=60, help="")
    return parser.parse_args()


def analysis(
    checkpoint_path: Path,
    model_path: Union[Path, str, None] = None,
    exoskeleton: bool = False,
    simulation_duration: float = 60.0,
    visualize: bool = False,
) -> Dict[str, List[float]]:
    # get individual

    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"args.checkpoint '{checkpoint_path}' does not exist.")
    filenames = sorted(checkpoint_path.glob("*.pkl"))

    ckpt = Checkpoint.from_path(filenames[-1])

    individual = ckpt.best_individual

    feedback_control = ReflexLocomotionControl(
        individual,
        mode="2D",
        dt=ckpt.simulation_dt,
    )

    # get model

    if model_path is None:
        model_path, exoskeleton = ckpt.model_path, ckpt.exoskeleton

    model = Model(
        str(model_path),
        exoskeleton,
        ckpt.initial_speed,
        visualize=visualize,
        integrator_accuracy=ckpt.simulation_integrator_accuracy,
    )

    # setup environment

    timestep_limit = int(round(simulation_duration / ckpt.simulation_dt))

    env = Environment(
        model=model,
        desired_speed=ckpt.target_speed,
        timestep_limit=timestep_limit,
        difficulty=ckpt.difficulty,
        seed=ckpt.seed,
    )

    env.init()

    # simulate

    observation = env.get_observation()

    done, total_reward, pose, num_steps, logs = False, 0, [0], 0, {}
    with console.status("Processing...", spinner="dots") as status:
        while not done:
            status.update(f"Computing step #{num_steps}...")

            actions = feedback_control.update(observation)
            if env.model.exoskeleton:
                actions = np.concatenate((actions, np.zeros(6)))

            observation, reward, _, pose, done = env.step(actions)
            total_reward += reward

            for k, v in observation["joint_positions"].items():
                if k not in logs:
                    logs[k] = []
                logs[k].append(v)

            num_steps += 1

    console.print(
        f"model: '{model_path}', num_steps: {num_steps}, "
        + f"fitness: {total_reward:.5f}, distance: {pose[0]:.5f}"
    )

    return logs


if __name__ == "__main__":
    # args = get_args()
    # analysis(args.checkpoint_path, visualize=args.visualize)

    simulation_duration = 60

    # init pose

    checkpoint_path = Path(
        "results/cmaes/2024-12-10_22-40-22_gait14dof22musc_pros2_init_pose",
    )

    init_logs = analysis(checkpoint_path, simulation_duration=simulation_duration)

    # with prosthetics

    for model_path in [
        "models/hip-prosthesis/gait14dof22musc_pros2_init_pose_x_n1.osim",
        "models/hip-prosthesis/gait14dof22musc_pros2_init_pose_x_p1.osim",
        "models/hip-prosthesis/gait14dof22musc_pros2_init_pose_y_n1.osim",
        "models/hip-prosthesis/gait14dof22musc_pros2_init_pose_y_p1.osim",
        "models/hip-prosthesis/gait14dof22musc_pros2_init_pose_z_n1.osim",
        "models/hip-prosthesis/gait14dof22musc_pros2_init_pose_z_p1.osim",
    ]:
        logs = analysis(
            checkpoint_path,
            model_path,
            exoskeleton=False,
            simulation_duration=simulation_duration,
        )

        # plot

        plt.rcParams.update({"font.size": 40})

        fig, axes = plt.subplots(len(logs) // 2, 2, figsize=(100, 40))

        title = Path(model_path).stem
        fig.suptitle(title, fontsize=80)

        x = list(range(len(logs["hip_l"])))
        for name, axe in zip(logs.keys(), axes.flatten()):
            axe.plot(
                x,
                init_logs[name],
                label=name,
                linestyle="-",
                marker="o",
            )
            axe.plot(
                x,
                logs[name],
                label=f"{name}_with_pros",
                linestyle="--",
                marker="x",
            )

            axe.set_xlabel("Steps")
            axe.set_ylabel("Position")
            axe.legend()

            axe.grid(True)

        plt.savefig(f"{title}.png")
