## Installation

### Locally

To run <osim-rl> simulations, Anaconda is needed so as to create a virtual environment containing all the necessary libraries and to avoid conflicts with the already-existing libraries of the OS.

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
2. Based on the guide *Reinforcement learning with musculoskeletal in OpenSim* [OSIM-RL](http://osim-rl.kidzinski.com/docs/quickstart/) [2], a conda environment with the OpenSim package will be created. In a command prompt, type the following commands:

      - **Windows**:
    ```
                     conda create -n opensim-rl -c kidzik opensim python=3.6.12
                     activate opensim-rl
    ```

      - **Linux/OSX**:
    ```
                     conda create -n opensim-rl -c kidzik opensim python=3.6.12
                     source activate opensim-rl
    ```

    From this, the python reinforcement learning environment is installed:
    ```
                    conda install -c conda-forge lapack git
                    pip install git+https://github.com/standfordnmbl/osim-rl.git
    ```
    To test if everything was set up correctly, the command `python -c "import opensim"` should run smoothly. In case of questions, please refer to the [FAQ](http://osim-rl.kidzinski.com/docs/faq) of the Osim-rl website.

    **Note:** The command `source activate opensim-rl` allows to activate the Anaconda virtual environment and should be typed ***everytime*** a new terminal is opened.

3. After creating the virtual environment `opensim-rl` (with the command `source activate opensim-rl`), the required libraries for the project should be installed with pip:
    ```
        pip install -r requirements.txt
    ```

4. Clone the git repository to have access to all the files of the project.

### Install and use in CSCS supercomputer

1. Open a terminal
2. Connect using the following command `ssh ela.cscs.ch -l USERNAME` and change <USERNAME> to yours.
3. Enter the password corresponding to your CSCS account
Normally, you are now connected to the ELA server.

4. Next, type the following commant `ssh daint.cscs.ch`
5. The first time, you should type `yes` once or multiple times and enter the same password as the one of your CSCS account
Normally, you have now added daint as a host and are connected to it.

6. If you type `ls`, you should normally have only one folder `bin`

7. Now, `Anaconda version 4.3.14` should be installed, with the following commands:
```
wget https://repo.anaconda.com/archive/Anaconda3-4.3.1-Linux-x86_64.sh
chmod +x Anaconda3-4.3.1-Linux-x86_64.sh
./Anaconda3-4.3.1-Linux-x86_64.sh
```
8. Next, in the terminal, type:
```
conda config --append channels conda-forge
conda create -n opensim-rl -c kidzik opensim python=3.6.1
source activate opensim-rl
pip install osim-rl
```

9. Then, try `python -c "import opensim"`, if this command runs smoothly, you can continue with the installation.

```
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```

The environment is set and ready, you can now clone the repository, and launch experiments on the CSCS server.

10. Some libraries might still be missing fron your installation, such as MPI, to install it:

```
conda install mpi4py
```

11. Launching experiments with sbatch

To launch experiments using the .sbatch files, the following command can be used:

  ` sbatch SCRIPT.sbatch -C gpu`

`squeue -u USERNAME` : Show the state of the job in the queue based on your username

 12. Two files are then automatically created: `out-------.o` and `err-------.e` with the job ID associated to the experiment that you launched.

 13. To cancel a job you can run `scancel [job_ID]`
