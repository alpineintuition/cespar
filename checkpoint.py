import argparse
import csv
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Union

import numpy as np
from deap.tools import HallOfFame, Logbook

log = logging.getLogger("cmaes")


class Checkpoint:
    def __init__(
        self,
        model_path: str,
        exoskeleton: bool,
        difficulty: int,
        num_individual_fb: int,
        initial_speed: float,
        simulation_duration: float,
        simulation_dt: float,
        simulation_integrator_accuracy: float,
        target_speed: float,
        best_fitness: float,
        best_distance: float,
        best_individual,
        sigma: float,
        mu: Union[int, None],
        path: Path,
        seed: int,
        best_generation: int = 0,
        log_frequency: int = 1,
        logbook: Union[Logbook, None] = None,
        logbook_path: Union[str, None] = None,
        hall_of_fame: Union[HallOfFame, None] = None,
        **kwargs,
    ):
        self.model_path = model_path
        self.exoskeleton = exoskeleton

        self.num_individual_fb = num_individual_fb

        self.difficulty = difficulty

        self.simulation_duration = simulation_duration
        self.simulation_dt = simulation_dt

        self.simulation_integrator_accuracy = simulation_integrator_accuracy

        self.initial_speed = initial_speed
        self.target_speed = target_speed

        self.best_generation = best_generation
        self.best_fitness = best_fitness
        self.best_distance = best_distance
        self.best_individual = best_individual

        self.sigma = sigma
        self.mu = mu

        self.path = path

        self.seed = seed

        self.log_frequency = log_frequency

        # self._stats, self._stats_duration = deap_mpi.get_stats()

        self.logbook = logbook
        if self.logbook is None:
            self.logbook = Logbook()
            headers = (
                ["gen", "evals", "sigma", "distance"]
                # + self._stats.fields
                # + self._stats_duration.fields
            )
            self.logbook.header = headers  # type: ignore

        self.logbook_path = logbook_path
        if self.logbook_path is None:
            self.logbook_path = self.path / "logbook.csv"

        self.start_gen = len(self.logbook)
        self.first_run = len(self.logbook) == 0

        self.hall_of_fame = hall_of_fame
        if self.hall_of_fame is None:
            self.hall_of_fame = HallOfFame(1)

    @staticmethod
    def from_path(path: str):
        with open("{}".format(path), "rb") as file:
            ckpt = pickle.load(file)
        return Checkpoint(**ckpt)

    @staticmethod
    def from_args(args: argparse.Namespace):
        if args.checkpoint:
            ckpt = Checkpoint.from_path(args.checkpoint)
            # TODO: restore sigma
            # TODO: clear logbook if restarted from a previous checkpoint
            if args.force_sigma:
                ckpt.sigma = args.sigma
        else:
            if args.output_dir is None:
                now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                path = Path(f"./results/cmaes/{now}_{Path(args.model_path).stem}")
                path.mkdir(parents=True, exist_ok=True)
            else:
                path = Path(args.output_dir)

            ckpt = Checkpoint(
                best_generation=0,
                best_fitness=0.0,
                best_distance=0.0,
                start_gen=0,
                path=path,
                best_individual=np.loadtxt(args.file),
                first_run=True,
                **vars(args),
            )

        return ckpt

    def print(self):
        to_print = []
        for k, v in self.__dict__.items():
            if k in [
                "_stats",
                "_stats_duration",
                "logbook",
                "hall_of_fame",
                "best_individual",
            ]:
                continue
            k = k.upper()
            if "PATH" in k:
                v = f"'{v}'"
            to_print.append(f"{k} : {v}")
        log.info("\n".join(to_print))

    def log(self, generation, offsprings, sigma, fitness, distance, **stats):
        assert self.logbook_path is not None
        assert self.logbook is not None

        self.logbook.record(
            generation=generation,
            evals=len(offsprings),
            sigma=np.round(sigma, 2),
            fitness=np.round(fitness, 2),
            distance=np.round(distance, 2),
            **stats,
        )

        # Saving logbook at each generation

        with open(self.logbook_path, "a") as csvfile:
            logbook = self.logbook[-1]

            writer = csv.DictWriter(csvfile, fieldnames=logbook.keys())
            if self.first_run:
                writer.writeheader()
                self.first_run = False

            writer.writerow(logbook)

    def save(self):
        assert self.logbook is not None
        to_save = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        with open(self.path / f"_{len(self.logbook):05}.pkl", "wb") as file:
            pickle.dump(to_save, file)
