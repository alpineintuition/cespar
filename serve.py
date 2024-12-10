import asyncio
import logging
import pickle
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from control.osim_hbp_cmaes import OsimModel
from utils import decode, setup_logging

NUM_WORKERS = 3
NUM_PROCESSES_PER_OPTIMIZATION = 3
OUTPUT_DIR = Path("./results/serve")

setup_logging(debug=True)

log = logging.getLogger("serve")

app = FastAPI()
queue = asyncio.Queue()


@app.on_event("startup")
async def startup():
    tasks = []
    for i in range(NUM_WORKERS):
        task = asyncio.create_task(worker(i, queue))
        tasks.append(task)


def launch_cmaes_optimization(
    num_processes: int,
    model_path: str,
    num_generations: int,
    output_dir: str,
):
    command = (
        f"mpirun -np {num_processes}  "
        + "python cmaes.py "
        + f"--model-path {model_path} "
        + f"--num-generations {num_generations} "
        + f"--output-dir {output_dir}"
    )

    subprocess.run(
        command.split(),
        cwd=Path.cwd(),
        check=True,
        stdout=subprocess.PIPE,
    )


async def worker(id: int, queue: asyncio.Queue):
    prefix = f"[WORKER#{id}]"
    log.debug(f"{prefix} started, waiting for a job...")

    while True:
        model_path, num_generations, output_dir = await queue.get()
        log.debug(f"{prefix} launching optimization for '{model_path}'")

        await asyncio.get_running_loop().run_in_executor(
            None,
            launch_cmaes_optimization,
            NUM_PROCESSES_PER_OPTIMIZATION,
            model_path,
            num_generations,
            output_dir,
        )

        queue.task_done()


class OptimizationLaunchRequest(BaseModel):
    osim_model_name: str
    osim_model: str
    exoskeleton: bool
    num_generations: int


class OptimizationAnswer(BaseModel):
    id: str


@app.post("/optimizations")
async def post_optimizations(
    request: OptimizationLaunchRequest,
) -> OptimizationAnswer:
    if request.num_generations < 0:
        raise HTTPException(
            status_code=400,
            detail="Number of generations must be a positive integer",
        )

    # setup model

    model_path = f"./models/{request.osim_model_name}.SERVE_TEST.osim"
    with open(model_path, "wb") as file:
        file.write(decode(request.osim_model))

    try:
        OsimModel(model_path, request.exoskeleton, False)
    except RuntimeError:
        raise HTTPException(status_code=400, detail="Invalid model")

    #

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = OUTPUT_DIR / f"{now}_{request.osim_model_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # send to queue

    await queue.put((model_path, request.num_generations, output_dir))

    return OptimizationAnswer(id=str(output_dir))


class OptimizationResults(BaseModel):
    id: str
    best_fitness: float
    best_distance: float
    best_individual: List[float]


def get_best_optimization(dir: Path):
    best_optimization = sorted(dir.glob("*.pkl"))[-1]

    with open(best_optimization, "rb") as f:
        data = pickle.loads(f.read())

    return OptimizationResults(
        id=best_optimization.stem,
        best_fitness=data["best_fitness"],
        best_distance=data["best_distance"],
        best_individual=data["best_individual"].tolist(),
    )


@app.get("/optimizations")
async def get_optimizations() -> List[OptimizationResults]:
    optimizations = []
    for item in OUTPUT_DIR.glob("*"):
        if not item.is_dir():
            continue
        optimizations.append(get_best_optimization(item))
    return optimizations


@app.get("/optimizations/{optimization_id}")
async def get_optimizations_id(optimization_id: str) -> OptimizationResults:
    path = OUTPUT_DIR / optimization_id
    if not path.is_dir():
        raise HTTPException(status_code=404, detail="Optimization does not exist")
    return get_best_optimization(path)
