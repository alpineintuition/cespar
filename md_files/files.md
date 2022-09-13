## Files

- `requirements.txt` : Specifies the required Python packages to run the project.
- `coupled_gait_optimization_cmaes.py` : Contains the CMA-ES optimization algorithm for the control of the reflex model.
- `control/osim.py` : OpenSim Interface and Environment. This file contains the `OsimModel` class responsible for low level communication with the OpenSim simulator, as well as the `OsimEnv` class which is responsible for the management of the user interface for running the reinforcement learning algorithm. Spefically, it includes the values of the different parameters used in the controller (GRFs, joint angles, pelvis state, joint angular velocities, muscles etc.).
- `control/osim_loco_reflex_song2019.py` : Represents the reflex control class in the OpenSim environment.
- `control/locoCtrl_balance_reflex_separated.py` : Contains a spinal controller for the balance and the reflex model, as well as the control parameters for both controllers.
- `control/params_2D.txt` : Contains all the initial values of the control parameters to launch the optimizations for a 2D simulation.
- `control/params_3D_init.txt` : Contains all the initial values of the control parameters to launch the optimizations in 3D (not available yet).
- `lib/` : Includes the `deap` framework for parallelisation mechanism.
- `logs/` : Once the optimizations are launched, their results are stored in a new `cmaes_spinal_x` folder.
- `models/` : Various Osim models (arms, legs, trunk etc.) as well as `.obj` files for the exoskeleton part.
- `Results_CMAES/logbook_analysis.py` : This short script allows to classify the generations' results of an optimization, based on the duration (how long was the model able to walk) and then, on the value of the reward. From this, the best checkpoints can be selected for further analysis. 
