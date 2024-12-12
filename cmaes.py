# -*- coding: utf-8 -*

"""coupled_gait_optimization_cmaes.py: Source code of the CMAES optimization for coupled musculoskeletal and exoskeleton systems
This module demonstrates how to use a gym musculoskeletal environment to learn a healthy gait with muscle activities while being attached to an exoskeleton
Example:
    The following command should be typed in the terminal to run CMA-ES simulations with the coupled musculoskeletal and exoskeleton system. ::
        $ mpirun -np 20 python coupled_gait_optimization_cmaes -g 250 --duration=60 --init_speed=1.6 --tgt_speed=1.3 --file control/params_2D.txt
"""

__author__ = "Berat Denizdurduran, Florin Dzeladini, Carla Nannini, Raphael Gaiffe"
__copyright__ = "Copyright 2022, Alpine Intuition SARL"
__license__ = "GPL-3.0 license"
__version__ = "1.0.0"
__email__ = "berat.denizdurduran@alpineintuition.ch"
__status__ = "Stable"
__acknowledgement__ = "The CESPAR Project is supported by European Union’s Horizon 2020 Framework Programme for Research and Innovation under Specific Grant Agreement No: 945539 (Human Brain Project SGA-3)"


# ███╗   ███╗██╗   ██╗███████╗ ██████╗██╗   ██╗██╗      ██████╗ ███████╗██╗  ██╗███████╗██╗     ███████╗████████╗ █████╗ ██╗
# ████╗ ████║██║   ██║██╔════╝██╔════╝██║   ██║██║     ██╔═══██╗██╔════╝██║ ██╔╝██╔════╝██║     ██╔════╝╚══██╔══╝██╔══██╗██║
# ██╔████╔██║██║   ██║███████╗██║     ██║   ██║██║     ██║   ██║███████╗█████╔╝ █████╗  ██║     █████╗     ██║   ███████║██║
# ██║╚██╔╝██║██║   ██║╚════██║██║     ██║   ██║██║     ██║   ██║╚════██║██╔═██╗ ██╔══╝  ██║     ██╔══╝     ██║   ██╔══██║██║
# ██║ ╚═╝ ██║╚██████╔╝███████║╚██████╗╚██████╔╝███████╗╚██████╔╝███████║██║  ██╗███████╗███████╗███████╗   ██║   ██║  ██║███████╗
# ╚═╝     ╚═╝ ╚═════╝ ╚══════╝ ╚═════╝ ╚═════╝ ╚══════╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝╚══════╝╚══════╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚══════╝
#
# ██████╗ ██████╗ ████████╗██╗███╗   ███╗██╗███████╗ █████╗ ████████╗██╗ ██████╗ ███╗   ██╗
# ██╔══██╗██╔══██╗╚══██╔══╝██║████╗ ████║██║╚══███╔╝██╔══██╗╚══██╔══╝██║██╔═══██╗████╗  ██║
# ██║  ██║██████╔╝   ██║   ██║██╔████╔██║██║  ███╔╝ ███████║   ██║   ██║██║   ██║██╔██╗ ██║
# ██   ██║██╔═══╝    ██║   ██║██║╚██╔╝██║██║ ███╔╝  ██╔══██║   ██║   ██║██║   ██║██║╚██╗██║
# ██████╔╝██║        ██║   ██║██║ ╚═╝ ██║██║███████╗██║  ██║   ██║   ██║╚██████╔╝██║ ╚████║
# ╚═════╝ ╚═╝        ╚═╝   ╚═╝╚═╝     ╚═╝╚═╝╚══════╝╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝
#
#  with deap.

import copy
import csv
import logging
import pickle
import random
import time
from pathlib import Path

import numpy as np
from deap import base, cma, creator, tools
from mpi4py import MPI  # type: ignore
from rich.progress import Progress

from locomotion import ReflexLocomotionControl
from checkpoint import Checkpoint
from utils import get_args, setup_logging
from environment import Environment
from model import Model

#
# setup parallel processes details
#

MAIN_PROCESS_RANK = 0
VISUALIZER_PROCESS_RANK = 1

comm = MPI.COMM_WORLD
SIZE = comm.Get_size()
RANK = comm.Get_rank()

#
# parse arguments
#

args = get_args()

#
# setup logs
#

console = setup_logging(args.debug, fixed_width=SIZE > 1)
log = logging.getLogger("cmaes")

#
# setup checkpoint
#

ckpt = Checkpoint.from_args(args)
if RANK == MAIN_PROCESS_RANK:
    ckpt.print()


#
# set seed
#

random.seed(ckpt.seed)
np.random.seed(ckpt.seed)

#
#
#

if SIZE == 1 and not args.test:
    log.warning(
        "Test mode activated with num_processes=1. "
        + "For a full optimization run, set num_processes to 3 or higher."
    )
    args.test = True


def main():
    prefix = "[purple][bold][MAIN][/][/]"

    #
    # init cmaes
    #

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", lambda x: np.round(np.mean(x), 2))
    stats.register("std", lambda x: np.round(np.std(x), 2))
    stats.register("min", lambda x: np.round(np.min(x), 2))
    stats.register("max", lambda x: np.round(np.max(x), 2))

    stats_duration = tools.Statistics(lambda ind: ind.fitness.values[1])
    stats_duration.register("dur", np.max)

    creator.create("FitnessMax", base.Fitness, weights=(1.0, 0.0001, 0.0001))
    creator.create("Individual", list, fitness=creator.FitnessMax)  # type: ignore

    toolbox = base.Toolbox()

    strategy_params = {
        "centroid": ckpt.best_individual,
        "sigma": ckpt.sigma,
        "lambda_": ckpt.num_individual_fb,
    }
    if ckpt.mu is not None:
        strategy_params["mu"] = ckpt.mu
    strategy = cma.Strategy(**strategy_params)

    toolbox.register("generator", strategy.generate, creator.Individual)  # type: ignore
    toolbox.register("update", strategy.update)

    #
    # start compute
    #

    time.sleep(1)  # so all workers are ready

    with Progress(console=console, disable=args.debug) as progress:
        start, end = ckpt.start_gen + 1, args.num_generations + 1
        task = progress.add_task("Computing...", total=end, completed=start)

        generation_without_best = 0
        for generation in range(start, end):
            sub_prefix = f"{prefix} (generation={generation})"

            desc = f"Computing generation {generation}/{args.num_generations }"
            progress.update(task, advance=1, description=desc)

            #
            # generate individuals & dispatch
            #

            individuals = toolbox.generator()  # type: ignore

            working_workers = {}
            for worker_rank, individual in enumerate(individuals, start=1):
                log.debug(
                    f"{sub_prefix} sending individual to WORKER#{worker_rank}:\n"
                    + str(individual)
                )
                comm.send(individual, dest=worker_rank, tag=worker_rank)
                working_workers[worker_rank] = individual

            #
            # wait on results
            #

            results, best_fitness, best_idx, updated_individuals = [], 0, 0, []
            while len(working_workers) > 0:
                rank = random.choice(list(working_workers.keys()))
                if comm.Iprobe(source=rank, tag=SIZE + rank):
                    recv = comm.recv(source=rank, tag=SIZE + rank)
                    log.debug(f"{sub_prefix} got evaluation from [WORKER#{rank}]")
                    fitness, distance, simulation_duration = recv

                    individual = working_workers[rank]
                    individual.fitness.values = (fitness, simulation_duration, 0.0)
                    updated_individuals.append(individual)

                    results.append((fitness, distance, np.array(individual)))

                    if fitness > best_fitness:
                        best_fitness = fitness
                        best_idx = len(results) - 1

                    del working_workers[rank]
                time.sleep(0.001)

            #
            # update and save
            #

            toolbox.update(copy.deepcopy(updated_individuals))  # type: ignore

            fitness = results[best_idx][0]
            distance = results[best_idx][1]
            individual = results[best_idx][2]

            ckpt.log(
                generation,
                individual,
                strategy.sigma,
                fitness,
                distance,
                **stats.compile(updated_individuals),
                **stats_duration.compile(updated_individuals),
            )

            if fitness > ckpt.best_fitness:
                log.info(
                    f"{sub_prefix} New best individual! "
                    + f"fitness={fitness:.4f}, distance={distance:.4f}"
                )

                generation_without_best = 0
                ckpt.best_generation = generation
                ckpt.best_fitness = fitness
                ckpt.best_distance = distance
                ckpt.best_individual = individual
                ckpt.sigma = strategy.sigma

                ckpt.save()

            if generation_without_best > args.early_stop:
                log.info(
                    f"{sub_prefix} early stop! No new best individual for more "
                    + f"than {args.early_stop} genereations"
                )
                break
            generation_without_best += 1

    # Send termination signal

    for worker_rank in range(1, SIZE):
        comm.send(None, dest=worker_rank, tag=worker_rank)
    log.info(f"{prefix} Sent termination signals to all workers.")

    log.info(f"{prefix} results and checkpoints available at '{ckpt.path}'")

    # Save model elements for future ml trainings

    model = Model(ckpt.model_path, ckpt.exoskeleton, ckpt.initial_speed, False)

    elements = model.get_elements()
    elements["best_individual"] = ckpt.best_individual.tolist()

    path = Path("./results/cmaes/training.csv")
    with open(path, mode="a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=elements.keys())
        if not path.is_file():
            writer.writeheader()
        writer.writerow(elements)

    log.info(f"{prefix} training csv ('{path}') updated")


def worker():
    prefix = f"[cadet_blue][bold][WORKER#{RANK}][/][/]"

    #
    # setup simulation environment
    #

    visualize = args.visualize
    if RANK != VISUALIZER_PROCESS_RANK and not args.test:
        visualize = False

    model = Model(
        ckpt.model_path,
        ckpt.exoskeleton,
        ckpt.initial_speed,
        visualize=visualize,
        integrator_accuracy=ckpt.simulation_integrator_accuracy,
    )

    env = Environment(
        model=model,
        desired_speed=ckpt.target_speed,
        timestep_limit=int(round(ckpt.simulation_duration / ckpt.simulation_dt)),
        difficulty=ckpt.difficulty,
        seed=ckpt.seed,
    )

    #
    # compute optimization
    #

    def compute(individual):
        feedback_control = ReflexLocomotionControl(
            individual,
            mode="2D",
            dt=ckpt.simulation_dt,
        )

        env.init()

        done = False
        distance = 0
        num_iterations = 0
        total_reward = 0
        start = time.time()
        speeds = []
        joint_position_log = []
        joint_velocity_log = []
        muscle_activation_log = []

        observation = env.get_observation()

        joint_position_log.append(observation["joint_positions"])
        joint_velocity_log.append(observation["joint_velocities"])

        while not done:
            actions = feedback_control.update(observation)
            if ckpt.exoskeleton:
                # Added exoskeleton: At this state, the exoskeleton is only an added weight of 10kg.
                # In this case, the exoskeleton is partially added (only the hips' actuators).
                exo_actuation = np.zeros(6)
                actions = np.concatenate((actions, exo_actuation))

            observation, reward, speed, pose, done = env.step(actions)

            total_reward += reward
            distance = pose[0]
            speeds.append(speed)

            joint_position_log.append(observation["joint_positions"])
            joint_velocity_log.append(observation["joint_velocities"])
            muscle_activation_log.append(actions)

            num_iterations += 1

        simulation_duration = num_iterations * ckpt.simulation_dt
        real_duration = time.time() - start

        log.debug(
            f"{prefix} done! "
            + f"fitness={total_reward:.4f}, "
            + f"distance={distance:.4f}, "
            + f"num_iterations={num_iterations}, "
            + f"simulation_duration={simulation_duration:.4f} sec, "
            + f"real_duration={real_duration:.4f} sec, "
            # + f"total_obs_sum={total_obs_sum:.4f}, "
            + f"speed={speeds[-1]:.4f} m/s, "
            + f"mean_speed={np.mean(speeds):.4f} m/s"
        )

        if args.test:
            logs_dir = Path(ckpt.path) / "logs"
            logs_dir.mkdir(exist_ok=True, parents=True)

            to_save = [
                (joint_position_log, "joint_positions"),
                (joint_velocity_log, "joint_velocities"),
                (muscle_activation_log, "muscle_activation"),
            ]

            for data, filename in to_save:
                path = logs_dir / f"{ckpt.simulation_duration}_{filename}.pkl"
                with open(path, "wb") as f:
                    pickle.dump(data, f)

        return (total_reward, distance, simulation_duration)

    if args.test:
        for _ in range(args.repeat):
            log.info(
                f"{prefix} The following parameters have been loaded:\n"
                + str(ckpt.best_individual)
            )
            compute(ckpt.best_individual)
        return

    while True:
        if comm.Iprobe(source=MAIN_PROCESS_RANK, tag=RANK):
            individual = comm.recv(source=MAIN_PROCESS_RANK, tag=RANK)
            if individual is None:
                log.debug(f"{prefix} received termination signal, stopping.")
                break
            log.debug(f"{prefix} received individual, doing some work")

            results = compute(individual)

            comm.send(results, dest=MAIN_PROCESS_RANK, tag=SIZE + RANK)
            log.debug(f"{prefix} results sended to main")

        time.sleep(0.001)


if RANK == MAIN_PROCESS_RANK and not args.test:
    main()
else:
    worker()
