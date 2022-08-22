## Launch the optimization

### Launch optimizations locally

One interesting aspect of the project is to optimize the reflex controller, which is done using a population-based algorithm,
specifically Covariance Matrix Adaptation Evolution Strategy [CMA-ES](https://deap.readthedocs.io/en/master/examples/cmaes.html).

To launch an optimization directly on your local computer:
  1. Activate the virtual environment `opensim-rl`.
  2. Go to the directory that contains the git repository.
  3. Launch the following command:
      ```
      mpirun -np M python coupled_gait_optimization_cmaes.py -n N -g G -t T -duration D --file F
      ```
      where
      - `M`: the number of python to run in parallel. Since one of the script plays the role of the Master (dispatching all the different jobs), the number of python  scripts effectively used for optimization will be `M-1`.
      - `N`: the number of individuals.
      - `G`: the number of generations (typically 150).
      - `T`: the type of optimization used (CMA-ES in this case).
      - `D`: the maximum duration of the simulation, which is generally set to 10sec.
      - `F`: the file with the initial values of the parameters that are being optimized.

**Notes:**
- When running an optimization, the logs are automatically saved in a directory `logs/cmaes_spinal_x`, where `x` is incremented at each new optimization and represents the generation `x`.
- An optimization can be resumed by specifying a checkpoint `.pkl` file, with the `-c` argument.
- To test a solution, simply use `python coupled_gait_optimization_cmaes.py --duration D -c Results_CMAES/cmaes_spinal_x/_xx.pkl ` where `_x`is the checkpoint of generation `x`.

### Launching experiments in CSCS Supercomputer with sbatch

To launch experiments using the .sbatch files, the following command can be used:

  ` sbatch SCRIPT.sbatch -C gpu`

`squeue -u USERNAME` : Show the state of the job in the queue based on your username

 Two files are then automatically created: `out-------.o` and `err-------.e` with the job ID associated to the experiment that you launched.

 To cancel a job you can run `scancel [job_ID]`


### Visualize the optimization results locally

To visualize a simulation, it is necessary to launch it locally by typing the following commands:
  1. Activate the virtual environment `opensim-rl`.
  2. Go to the local directory containing the project.
  3. Launch the following command:
      ```
      python coupled_gait_optimization_cmaes.py --duration D -c Results_CMAES/cmaes_spinal_x.pkl --visualize --test
      ```

**Notes:**
- From the optimizations, `_x.pkl` files are created at each generation `x` and the simulation can be directly launched from one of these files, by specifying a checkpoint with the `-c` argument.

### Tunable & Control Parameters

**Tunable Parameters**: The tunable hyper parameters for each optimization are located in `coupled_gait_optimization_cmaes.py` and are, amongst others:
  - `-g`: Number of generations.
  - `-t`: Type of optimization used (CMA-ES).
  - `-duration`: Maximum duration of the simulation.
  - `init_speed`: Initial speed of the model in the simulation.
  - `tgt_speed`: Desired/Target speed of the model.
  - `-sig`: Initial width of the parameters distribution in the CMA-ES algorithm.

**Control Parameters**: There are 37 control parameters that need to be tuned for the model to walk robustly. They are defined in `control/locoCtrl_balance_reflex_separated.py` and,
in short, represent the contribution of the reflex modules to each muscle's activation, the trunk lean angle as well as the parameters for reactive foot placement.

#### Notes on scaling for optimization

The best way to deal with different scale of parameter variable is to normalize the state space so that every dimension has the same width. This is required at least for CMAES
because it initializes the multi-dimensional gaussian with a diagonal covariance matrix.
This is done by specifying a parameter range set `par_space` with two arrays of the same length representing the lower and upper bound of the parameters.

In the checkpoints, the solutions are saved unscaled, so that they can directly be used for the initialization of the CMAES. This means that when used as a controller, these solutions need to be scaled.

### Create a video

When launching a simulation locally, in the OpenSim visualizer, it is possible to save the images of the simulations.
From this, one can make a video by using the following command: `ffmpeg -framerate 25 -i Frame%04d.png -c:v libx264 output.avi` which should be run from the folder containing the images.
