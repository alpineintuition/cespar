from pathlib import Path

import requests

from serve import OptimizationLaunchRequest
from utils import encode

url = "http://127.0.0.1:8000"

# create a subject

path = Path("models/gait14dof22musc/gait14dof22musc_16y.osim")
with open(path, "rb") as file:
    model = encode(file.read())

request = OptimizationLaunchRequest(
    osim_model_name=path.stem,
    osim_model=model,
    exoskeleton=False,
    num_generations=10,
)

response = requests.post(f"{url}/optimizations", json=request.dict())
if response.status_code == 200:
    optimization_id = response.json()["id"]
    print(f"Optimization id : '{optimization_id}'")
else:
    print(f"Error: {response.status_code}, {response.text}")
