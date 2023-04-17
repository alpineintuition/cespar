# Development Directory

The scripts in this directory were used to develop and debug the scripts integrated into the NeuroRobotics Platform. It includes the following sub-directories:
- `Control`: same control directory as in the main branch; includes the environment and controller classes used in simulation.
- `logs`: used to store muscle activation and joint position and velocity logs that are created during simulation.
- `models`: contains the geometry and .osim xml files for the model used in simulation.
- `Results_CMAES`: contains solutions for the controller parameter optimisation with CMAES.

### Scripts
The `coupled_gait_optimization_cmaes.py` script is the same as the one in the main branch and is used as described in the [documentation](../../main/md_files/howToLaunch.md). The `test_controller.py` script was used to develop the controller class in the NRP integration and isolates the controller functions to reproduce the simulation results.
