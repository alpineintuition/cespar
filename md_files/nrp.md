## Neurorobotics Platform Integration

This project has been integrated into the Neurorobotics platform (NRP). The NRP is an initiative started by the Human Brain Project. It aims to provide an open-source framework that supports intelligent and bio-inspired robotics experimentation. To learn more, visit their [page](https://www.humanbrainproject.eu/en/science-development/focus-areas/neurorobotics/). 

### Which experiment was added to the platform?
As of this release, we have integrated the coupled gait optimization experiment into the platform. In the experiment, a reflex controller is used to control the model's muscle activations to walk in a healthy manner with the exoskeleton attached. 

### Run the experiment on the NRP
Navigate to the `nrp` folder, then run the following commands:
```
sudo chmod +x nrp_initializer.sh
sudo ./nrp_initializer.sh                                    
pip install git+https://github.com/stanfordnmbl/osim-rl.git  # install missing dependencies
pip install deap scikit-learn mpi4py                        
./experiment_initializer.sh                                  # run the experiment
```
You can adjust the desired simulation duration in the `test_cmaes/simulation_config.json` file. It is set to 10 seconds by default.