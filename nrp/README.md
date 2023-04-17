## NeurorRobin Project

The scripts in this directory were developed to run the coupled gait experiment on the NRP. This directory includes the following sub-directories:
- `engine`: contains the OpenSim interface and simulation manager scripts that provide high level functions for our simulation; those functions are called on in the specific experiment folders.
- `test_cmaes`: this is our experiment folder. It contains the scripts needed to run our experiment such as the control scripts, the model, and the OpenSim manager script in which the specific steps of our simulation are defined.
- `development`: contains scripts used to develop and debug the scripts integrated into the NeuroRobotics Platform.