import argparse
import logging
import os
from contextlib import contextmanager

from rich.console import Console
from rich.logging import RichHandler


def get_args() -> argparse.Namespace:
    """ """

    parser = argparse.ArgumentParser(
        prog="Distributed Population Based Optimization (With CMA-ES)"
    )

    parser.add_argument(
        "-mp",
        "--model-path",
        help="",
        type=str,
    )
    parser.add_argument(
        "--exoskeleton",
        help="",
        action="store_true",
    )
    parser.add_argument(
        "-ng",
        "--num-generations",
        help="[OPT] Number of generation",
        default=300,
        type=int,
    )
    parser.add_argument(
        "-s",
        "--seed",
        help="[OPT] Random seed",
        default=64,
        type=int,
    )
    parser.add_argument(
        "-n",
        "--num-individual-fb",
        help="[OPT] The number of individuals for a feedback optimization",
        default=2,
        type=int,
    )
    parser.add_argument(
        "-sig",
        "--sigma",
        help="[OPT] Sigma parameter of the CMA-ES optimization",
        default=0.1,
        type=float,
    )
    parser.add_argument(
        "-fs",
        "--force-sigma",
        help="[OPT] Set the value of sigma even if present in the checkpoint",
        action="store_true",
    )
    parser.add_argument(
        "--difficulty",
        help="[OPT]",
        default=0,
        type=int,
    )
    parser.add_argument(
        "-sd",
        "--simulation-dt",
        help="[OPT]",
        default=0.01,
        type=float,
    )
    parser.add_argument(
        "-sia",
        "--simulation-integrator-accuracy",
        help="[OPT]",
        default=5e-5,
        type=float,
    )
    parser.add_argument(
        "--mu",
        help="[OPT] ",
        default=None,
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        help="Checkpoint to use for initial parameters",
        default="",
    )
    parser.add_argument(
        "-f",
        "--file",
        help="Text file to use for initial parameters, checkpoint takes precendency on this parameter",
        default="./control/params_2D_unscaled.txt",
    )
    parser.add_argument(
        "-v",
        "--visualize",
        help="Whether to visualize the results or not, if used during optimization only the first individual is visualized",
        action="store_true",
    )
    parser.add_argument(
        "-duration",
        "--simulation-duration",
        help="Maximum duration of the simulation",
        default=60.0,
        type=float,
    )
    parser.add_argument(
        "-tgt_speed",
        "--target-speed",
        help="Target/Desired speed",
        default=1.3,
        type=float,
    )
    parser.add_argument(
        "-init_speed",
        "--initial-speed",
        help="Initial speed of the simulation",
        default=1.6,
        type=float,
    )
    parser.add_argument(
        "-repeat",
        "--repeat",
        help="Maximum repeat during testing",
        default=1,
        type=int,
    )
    parser.add_argument(
        "-test",
        "--test",
        help="Testing mode",
        action="store_true",
    )
    parser.add_argument(
        "-debug",
        "--debug",
        help="Enable debug mode",
        action="store_true",
    )
    parser.add_argument(
        "--output-dir",
        help="",
        type=str,
        default=None,
    )

    return parser.parse_args()


def setup_logging(debug: bool = False, fixed_width: bool = False) -> Console:
    console = Console(width=150 if fixed_width else None)

    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(message)s",
        datefmt="[%D %X]",
        handlers=[
            RichHandler(
                console=console,
                markup=True,
                show_path=False,
                omit_repeated_times=False,
                rich_tracebacks=True,
            )
        ],
    )

    for package in ["matplotlib", "PIL"]:
        logging.getLogger(package).setLevel(logging.CRITICAL)

    return console


@contextmanager
def no_stdout():
    """A context manager that redirects stdout to devnull"""
    devnull = os.open(os.devnull, os.O_RDWR)
    old_stdout = os.dup(1)
    try:
        os.dup2(devnull, 1)
        yield
    finally:
        os.dup2(old_stdout, 1)
        os.close(old_stdout)
        os.close(devnull)
