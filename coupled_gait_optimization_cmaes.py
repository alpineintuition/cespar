# -*- coding: utf-8 -*

"""coupled_gait_optimization_cmaes.py: Source code of the CMAES optimization for coupled musculoskeletal and exoskeleton systems
This module demonstrates how to use a gym musculoskeletal environment to learn a healthy gait with muscle activities while being attached to an exoskeleton
Example:
    The following command should be typed in the terminal to run CMA-ES simulations with the coupled musculoskeletal and exoskeleton system. ::
        $ mpirun -np 20 python coupled_gait_optimization_cmaes -g 250 --duration=60 --init_speed=1.6 --tgt_speed=1.3 --file control/params_2D.txt
"""

__author__ = "Berat Denizdurduran, Florin Dzeladini, Carla Nannini, Raphael Gaiffe"
__copyright__ = "Copyright 2022, Alpine Intuition SARL"
__license__ = "GPL-3.0 license"
__version__ = "1.0.0"
__email__ = "berat.denizdurduran@alpineintuition.ch"
__status__ = "Stable"


#███╗   ███╗██╗   ██╗███████╗ ██████╗██╗   ██╗██╗      ██████╗ ███████╗██╗  ██╗███████╗██╗     ███████╗████████╗ █████╗ ██╗
#████╗ ████║██║   ██║██╔════╝██╔════╝██║   ██║██║     ██╔═══██╗██╔════╝██║ ██╔╝██╔════╝██║     ██╔════╝╚══██╔══╝██╔══██╗██║
#██╔████╔██║██║   ██║███████╗██║     ██║   ██║██║     ██║   ██║███████╗█████╔╝ █████╗  ██║     █████╗     ██║   ███████║██║
#██║╚██╔╝██║██║   ██║╚════██║██║     ██║   ██║██║     ██║   ██║╚════██║██╔═██╗ ██╔══╝  ██║     ██╔══╝     ██║   ██╔══██║██║
#██║ ╚═╝ ██║╚██████╔╝███████║╚██████╗╚██████╔╝███████╗╚██████╔╝███████║██║  ██╗███████╗███████╗███████╗   ██║   ██║  ██║███████╗
#╚═╝     ╚═╝ ╚═════╝ ╚══════╝ ╚═════╝ ╚═════╝ ╚══════╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝╚══════╝╚══════╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚══════╝
#
#██████╗ ██████╗ ████████╗██╗███╗   ███╗██╗███████╗ █████╗ ████████╗██╗ ██████╗ ███╗   ██╗
#██╔══██╗██╔══██╗╚══██╔══╝██║████╗ ████║██║╚══███╔╝██╔══██╗╚══██╔══╝██║██╔═══██╗████╗  ██║
#██║  ██║██████╔╝   ██║   ██║██╔████╔██║██║  ███╔╝ ███████║   ██║   ██║██║   ██║██╔██╗ ██║
#██   ██║██╔═══╝    ██║   ██║██║╚██╔╝██║██║ ███╔╝  ██╔══██║   ██║   ██║██║   ██║██║╚██╗██║
#██████╔╝██║        ██║   ██║██║ ╚═╝ ██║██║███████╗██║  ██║   ██║   ██║╚██████╔╝██║ ╚████║
#╚═════╝ ╚═╝        ╚═╝   ╚═╝╚═╝     ╚═╝╚═╝╚══════╝╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝
#
#  with deap.

import sys
from control.osim_HBP_withexo_CMAES import L2M2019Env
from control.osim_loco_reflex_song2019 import OsimReflexCtrl
import numpy as np
import argparse
import random
import pickle
from mpi4py import MPI
import time
import copy
import os
import csv


# MPI Related constant
MASTER              = 0                     # MPI Master node
comm = MPI.COMM_WORLD
size = comm.Get_size() # new: gives number of ranks in comm
rank = comm.Get_rank()
# -----------------------------------------------------------------

# Feedback controller related variable
FBCtrl = None
env    = None

# General environment constant
difficulty = 0
sim_dt = 0.01


# Script parameters
parser = argparse.ArgumentParser(prog='Distributed Population Based Optimization (With CMA-ES)')
parser.add_argument("-g",       "--ngen",            help="[OPT] Number of generation", default=100)
parser.add_argument("-s",       "--seed",            help="[OPT] Random seed", default=64)
parser.add_argument("-n",       "--n-individual-fb", help="[OPT] The number of individuals for a feedback optimization", default=2)
parser.add_argument("-sig",     "--sigma-fb",        help="[OPT] Sigma parameter of the CMA-ES optimization", default=5.0)
parser.add_argument("-fs",      "--force-sigma",     help="[OPT] Set the value of sigma even if present in the checkpoint", action='store_true')
parser.add_argument("-mu",      "--cmaes-mu",        help="[OPT] ", default=None)
parser.add_argument("-c",       "--checkpoint",      help="Checkpoint to use for initial parameters", default="")
parser.add_argument("-f",       "--file",            help="Text file to use for initial parameters, checkpoint takes precendency on this parameter", default=None)
parser.add_argument("-v",       "--visualize",       help="Whether to visualize the results or not, if used during optimization only the first individual is visualized", action='store_true')
parser.add_argument("-duration", "--duration",       help="Maximum duration of the simulation", default=10.0)
parser.add_argument("-tgt_speed","--tgt_speed",      help="Target/Desired speed", default=1.4)
parser.add_argument("-init_speed", "--init_speed",   help="Initial speed of the simulation", default=1.7)
parser.add_argument("-repeat",  "--repeat",          help="Maximum repeat during testing", default=1)
parser.add_argument("-test",    "--test",            help="Testing mode", action='store_true')
parser.add_argument("-debug",   "--debug",           help="Enable debug mode", action='store_true')

args = parser.parse_args()

DEBUG               = args.debug
TEST                = True if size == 1 else args.test
VISUALIZE           = False if(size > 1) else args.visualize
SIGMA               = float(args.sigma_fb)
SIM_T               = float(args.duration)
TIMESTEP_LIMIT      = int(round(SIM_T/sim_dt))
LOG_FREQ            = 1                     # Log every LOG_FREQ generation
SEED                = int(args.seed)+rank   # Random seed
NGEN                = int(args.ngen)        # Number of generation
FB_N                = int(args.n_individual_fb) # Number of individual
FB_DIM              = 46                    # Problem dimension
REPEAT              = int(args.repeat)
DESIRED_SPEED       = float(args.tgt_speed)       # Desired speed (= Target speed)
INITIAL_SPEED       = float(args.init_speed)      # Initial speed of the simulation
'''
Write parameters to the optimization folder, in a params.txt file, during optimization, to know what we are testing and
what set of parameters is used in a specific experiment.
'''
CHECKPOINT_PATH = './Results_CMAES/cmaes_spinal' # Path where the checkpoints are saved
ID = 1
if rank is MASTER and not TEST:
    if os.path.isdir(CHECKPOINT_PATH) == True:
        test_path = "{}_{}".format(CHECKPOINT_PATH,ID)
    while os.path.isdir(test_path):
        ID = ID+1
        test_path = "{}_{}".format(CHECKPOINT_PATH,ID)
    os.mkdir(test_path)
    CHECKPOINT_PATH = test_path
    with open("{}/params.txt".format(CHECKPOINT_PATH), 'w') as f:
        params_summary_txt = '''
TEST {}
VISUALIZE {}
SIGMA {}
SIM_T {}
TIMESTEP_LIMIT {}
LOG_FREQ {}
SEED {}
NGEN {}
FB_N {}
FB_DIM {}
REPEAT {}
DESIRED_SPEED {}
INITIAL_SPEED {}
'''.format(TEST,VISUALIZE,SIGMA,SIM_T,TIMESTEP_LIMIT,LOG_FREQ,SEED,NGEN,FB_N,FB_DIM,REPEAT,DESIRED_SPEED,INITIAL_SPEED)
        f.write(params_summary_txt)

#######################################################################
# MPI Tags
# 0:NP        : sending individual to worker
# NP+1:2NP    : sending fitness    to master
NP = size      # NP >= N : but NP > N => improvement in speed.
               # N  : Number of individual needed to pass to next generation
               # NP : Number of processes
#######################################################################
random.seed(SEED)
np.random.seed(SEED)
sys.path.insert(0,'./lib')
import deap_mpi

'''
    Optimization Parameters Initialization
    ======================================
    Here, we initialize the parameters' space. It is defined as a set of two arrays:
    the first array defines the lower bound and the second array the upper bound.
'''
PAR_SPACE = None

# FEEDBACK PARAMETER SPACE
# The space is defined for the 3D model, and a reduced set is created for the 2D model by dropping the last elements of the array.
FB_PAR_SPACE_3D = (
    [0.0, -1.0, 0.0, 0.0, \
    -2.0, -90/15, -1.0, 0.0, \
    0.0, 0.0, 0.0, 0.0, 0.0, \
    0.0, 0.0, 0.0, 0.0, 0.0, \
    0.0, 0.0, \
    0.0, 0.0, \
    0.0, 0.0, 0.0, \
    0.0, 0.0, 0.0, 0.0, 0.0, \
    0.0, 0.0, 0.0, \
    0.0, \
    0.0, \
    0.0, 0.0, \
    0.0, 0.0, 0.0, \
    0.0, 0.0, 0.0, \
    0.0, 0.0, 0.0],
    [6.0, 3.0, 5.0, 3.0, \
    3.0, 20/15, 15/10, 3.0, \
    3.0, 3.0, 3.0, 3.0, 3.0, \
    3.0, 3.0, 3.0, 3.0, 3.0, \
    3.0, 3.0, \
    3.0, 3.0, \
    3.0, 3.0, 3.0, \
    3.0, 3.0, 3.0, 3.0, 3.0, \
    3.0, 3.0, 3.0, \
    3.0, \
    3.0, \
    3.0, 3.0, \
    2.0, 3.0, 3.0, \
    3.0, 3.0, 3.0, \
    3.0, 3.0, 3.0])

fb_params_2D = np.loadtxt('./control/params_2D.txt')
fb_N_2D = len(fb_params_2D)

'''
    Environment Initialization
    ==========================
    The environment is defined only for the workers (e.g. when rank is not MASTER or when running TEST on an optimized controller)
'''

is_worker = lambda: rank is not MASTER or TEST

if is_worker():
    # Initial position of the model
    INIT_POSE = np.array([
        INITIAL_SPEED,              # forward speed
        .5,                         # rightward speed
        9.023245653983965608e-01,   # pelvis height
        2.012303881285582852e-01,   # trunk lean
        0*np.pi/180,                # [right] hip adduct
        -6.952390849304798115e-01,  # hip flex
        -3.231075259785813891e-01,  # knee extend
        1.709011708233401095e-01,   # ankle flex
        0*np.pi/180,                # [left] hip adduct
        -5.282323914341899296e-02,  # hip flex
        -8.041966456860847323e-01,  # knee extend
        -1.745329251994329478e-01]) # ankle flex

    '''
        Environment Creation
    '''
    def env_init():
        global env
        if rank is 2 or size is 1:
            env = L2M2019Env(visualize=VISUALIZE, seed=SEED, difficulty=difficulty, desired_speed=DESIRED_SPEED)
        else:
            env = L2M2019Env(visualize=False, seed=SEED, difficulty=difficulty, desired_speed=DESIRED_SPEED)
        # Change model parameters. With 2D model, one dimension is fixed, i.e. the model cannot fall to the left or right.
        env.change_model(model='2D', difficulty=difficulty, seed=SEED)
        env.spec.timestep_limit = TIMESTEP_LIMIT

    '''
        Feedback Controller Initialization
        ==================================
        The feedback controller, corresponding to Song's reflex controller from 2015, is initialized.
    '''
    def fb_init():
        global FBCtrl
        FBCtrl = OsimReflexCtrl(mode='2D', dt=sim_dt)

    '''
        Feedback Controller Update
    '''
    def fb_update(obs_dict):
        global FBCtrl
        return FBCtrl.update(obs_dict)

    '''
        Feedback controller initialization
    '''
    def init_ctrl():
        fb_init()

    '''
        Sets the parameters (46) of the feedback controller
    '''
    def set_param(x):
        global FBCtrl
        FBCtrl.set_control_params(np.round(x[0:fb_N_2D],4))
        return x

    '''
        Returns the action of the controller
        ====================================
        Note: 'get_action' does not separate spinal and vestibular contribution.
        If we separate the contribution from spinal and vestibular, we can test the hypothesis that spinal reflexes
        can be replaced by feedforward contribution but not vestibular contribution.
    '''

    def loop(init,set_param):
        global FBCtrl, TEST
        init()
        x=set_param()
        total_reward = 0
        t = 0
        i = 0
        # Restart the environment to the initial state. The function returns obs_dict: an observation dictionary
        # describing the state of muscles, joints and bodies in the biomechanical system.
        obs_dict = env.reset(project=True, seed=SEED, obs_as_dict=True, init_pose=INIT_POSE)
        obs = env.get_observation()
        elapsed_total = 0
        _total_obs_sum = 0 # For reproducibility
        velocity = []
        # Record muscle activities
        muscActLog = []

        # Record joint angles
        JointLog = []
        JointVelLog = []
        joint_pos = env.get_observation_list_joints_pos()
        joint_vel = env.get_observation_list_joints_vel()

        while True:
            _total_obs_sum += sum(obs)
            i += 1
            t += sim_dt
            _t = time.time()
            speed = obs[245]
            # Extract action reflex type
            action_fdb = fb_update(obs_dict)
            action = action_fdb

            # Added exoskeleton: At this state, the exoskeleton is only an added weight of 10kg.
            # In this case, the exoskeleton is partially added (only the hips' actuators).
            exo_actuation = np.zeros(6)
            action = np.concatenate((np.array(action), exo_actuation))

            '''
                Step Function
                =============
                Make a step (one iteration of the simulation) given by the action (a list of length 22 of continuous values in the [0, 1] interval,
                corresponding to the muscle activities).
                The function returns the observation dictionary (obs_dict), the reward gained in the last iteration, 'done' indicates if
                the move was the last step of the environment (total number of iterations reached)
                or if the pelvis height is below 0.6 meters, 'info' for compatibility with OpenAI gym (not used currently).
            '''
            obs_dict, reward, done, info = env.step(action, project = True, obs_as_dict=True)

            if TEST: # Log only when testing
                velocity.append(speed)
                mean_speed = np.mean(velocity)
                muscActLog.append(action)
                JointLog.append(joint_pos)
                JointVelLog.append(joint_vel)
            else:
                mean_speed = 0.0

            obs = env.get_observation()
            joint_pos = env.get_observation_list_joints_pos()
            joint_vel = env.get_observation_list_joints_vel()
            elapsed_total += time.time() - _t
            total_reward += reward

            if(done):
                break

        if DEBUG: print('[WORKER {}] score={} time={}sec, realdt={}s total_obs_sum={} speed={}m/s, i={}, mean_speed={}m/s'.format(rank,total_reward, t, elapsed_total/i,_total_obs_sum, speed, i, mean_speed))

        if DEBUG: sys.stdout.flush()

        # Save the log to the checkpoint folder but only when testing
        if TEST:
            SAVE_PATH = './logs/simulation_CMAES/'
            if(args.checkpoint):
                SAVE_PATH += args.checkpoint.split('/')[-1]


            # Saving the various data into separate .PKL files
            print("Saving logs to {}_[muscle_act|joint_pos|joint_vel]_{}_second.pkl".format(SAVE_PATH.split('.pkl')[0], args.duration))
            musclePath = SAVE_PATH.replace('.pkl', '_' + args.duration + '_second' + '_muscle_act.pkl')
            jointPath = SAVE_PATH.replace('.pkl', '_' + args.duration + '_second_all' + '_joints.pkl')
            jointvelPath = SAVE_PATH.replace('.pkl', '_' + args.duration + '_second_all' + '_joints_vel.pkl')
            pickle.dump(muscActLog, open(musclePath, "wb"))
            pickle.dump(JointLog, open(jointPath, "wb"))
            pickle.dump(JointVelLog, open(jointvelPath, 'wb'))

        #print('    score={} time={}sec, realdt={}s speed={}m/s, i={}'.format(total_reward, t, elapsed_total/i, speed, i))

        return total_reward, t,elapsed_total/i,env.pose[0], speed, i


    def F(x):
        return loop(init_ctrl,lambda: set_param(x))

'''
    Compile statistics on what is going on in the optimization.
'''
def get_stats():
    # Register desired statistic functions inside the 'stats' object
    stats = deap_mpi.tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", lambda x: np.round(np.mean(x),2))
    stats.register("std", lambda x: np.round(np.std(x),2))
    stats.register("min", lambda x: np.round(np.min(x),2))
    stats.register("max", lambda x: np.round(np.max(x),2))

    stats_duration = deap_mpi.tools.Statistics(lambda ind: ind.fitness.values[1])
    stats_duration.register("dur", np.max)

    # Create new classes named resp. FitnessMax & Individual
    deap_mpi.creator.create("FitnessMax", deap_mpi.base.Fitness, weights=(1.0, 0.0001, 0.0001))
    deap_mpi.creator.create("Individual", list, fitness=deap_mpi.creator.FitnessMax)
    return stats, stats_duration


if rank is MASTER or TEST:
    #  ███▄ ▄███▓ ▄▄▄        ██████ ▄▄▄█████▓▓█████  ██▀███
    # ▓██▒▀█▀ ██▒▒████▄    ▒██    ▒ ▓  ██▒ ▓▒▓█   ▀ ▓██ ▒ ██▒
    # ▓██    ▓██░▒██  ▀█▄  ░ ▓██▄   ▒ ▓██░ ▒░▒███   ▓██ ░▄█ ▒
    # ▒██    ▒██ ░██▄▄▄▄██   ▒   ██▒░ ▓██▓ ░ ▒▓█  ▄ ▒██▀▀█▄
    # ▒██▒   ░██▒ ▓█   ▓██▒▒██████▒▒  ▒██▒ ░ ░▒████▒░██▓ ▒██▒
    # ░ ▒░   ░  ░ ▒▒   ▓▒█░▒ ▒▓▒ ▒ ░  ▒ ░░   ░░ ▒░ ░░ ▒▓ ░▒▓░
    # ░  ░      ░  ▒   ▒▒ ░░ ░▒  ░ ░    ░     ░ ░  ░  ░▒ ░ ▒░
    # ░      ░     ░   ▒   ░  ░  ░    ░         ░     ░░   ░
    #        ░         ░  ░      ░              ░  ░   ░
    #
    # Initialization of FB parameters
    init_params = fb_params_2D
    if(args.file):
        init_params = np.loadtxt('{}'.format(args.file))

    stats, stats_duration = get_stats()
    # Initialize checkpoint storage.
    cp = {}
    # Load checkpoint from file
    if(args.checkpoint):
        with open("{}".format(args.checkpoint), "rb") as cp_file:
            cp = pickle.load(cp_file)
        if(args.force_sigma):
            cp["sigma"] = SIGMA
        print("best fitness : {}".format(cp["best_fitness"]))
        print("sigma        : {}".format(cp["sigma"]))
    # Initialize checkpoint from initial parameters
    else:
        cp = {
            'best_fitness'      : 0,
            'best_ind'          : init_params,
            'sigma'             : SIGMA,
            'start_gen'         : 0
        }

    PAR_SPACE = (FB_PAR_SPACE_3D[0][0:fb_N_2D],FB_PAR_SPACE_3D[1][0:fb_N_2D])

    deap_mpi.init(
        debug = DEBUG,
        par_space = PAR_SPACE)

if rank is MASTER and not TEST:
    # The data produced by the statistics is saved for further use in a Logbook. This logbook is intended to be
    # a chronological sequence of entries (as dictionaries).
    cp["logbook"] = deap_mpi.tools.Logbook()
    cp["halloffame"] = deap_mpi.tools.HallOfFame(1)
    cp["logbook"].header    = ["gen", "evals", "sigma", "distance"] + stats.fields + stats_duration.fields

    sys.stdout.flush()

    toolbox = {}

    fb_cmaes_tlbx, fb_cmaes_strategy = deap_mpi.initCMAES(
        best_ind      = deap_mpi.unscale_fb(cp["best_ind"]),
        sigma         = cp["sigma"],
        N             = FB_N,
        mu            = int(args.cmaes_mu) if args.cmaes_mu else None
    )
    toolbox['CMAES_FB'] = fb_cmaes_tlbx
    best_fitness = cp["best_fitness"];
    best_ind = deap_mpi.unscale_fb(cp["best_ind"])


    generators = deap_mpi.getGenerators(toolbox);
    start_gen = cp["start_gen"]
    firstrun = True
    for g in range(start_gen+1, NGEN):
        beta = 0
        #       _ _           _             _
        #    __| (_)___ _ __ | |_ __ _  ___| |__
        #   / _` | / __| '_ \| __/ _` |/ __| '_ \
        #   \__,_|_|___/ .__/ \__\__,_|\___|_| |_|
        #              |_|
        _offsprings, _fitnesses, _sim_durations, _real_durations, _distances = deap_mpi.dispatcher(FB_N, lambda: generators['CMAES_FB']())

        if DEBUG: print("[MASTER] finished dispatch")
        #
        #   _   _ _ __   __| | __ _| |_ ___
        #  | | | | '_ \ / _` |/ _` | __/ _ \
        #   \__,_| .__/ \__,_|\__,_|\__\___| the next generation with the results.
        #        |_|
        toolbox["CMAES_FB"].update(copy.deepcopy(_offsprings))
        if DEBUG: print("[MASTER] finished update")
        #
        #   ___| |_ __ _| |_ ___
        #  / __| __/ _` | __/ __|
        #  \__ \ || (_| | |_\__ \
        #  |___/\__\__,_|\__|___/
        #
        # this is done only for logging. The actual selection of individual is done in the update above.
        if DEBUG: print("[MASTER] logging stats")
        best_idx = np.argmax(_fitnesses)
        if(_fitnesses[best_idx] > best_fitness):
            best_duration = _sim_durations[best_idx]
            best_fitness  = np.max(_fitnesses)
            best_ind      = [i for i in _offsprings[best_idx]]


        _l_best_distance = _distances[best_idx]
        _l_best_fitness  = _fitnesses[best_idx]
        _l_best_fitness  = _fitnesses[best_idx]
        _l_best_duration = _sim_durations[best_idx]

        cp["logbook"].record(
            gen          = f"{str(g).zfill(4)}/{NGEN}",
            evals        = len(_offsprings),
            sigma        = np.round(fb_cmaes_strategy.sigma,2),
            distance     = np.round(_l_best_distance,2),
            **stats.compile(_offsprings),
            **stats_duration.compile(_offsprings),
            )

        print("Results of generation {} are now in {}/logbook.csv".format(str(g), CHECKPOINT_PATH))
        with open(f"{CHECKPOINT_PATH}/logbook.csv", 'a') as csvfile:
            logbook = cp["logbook"][-1]
            writer = csv.DictWriter(csvfile, fieldnames=logbook.keys())
            if firstrun:
                writer.writeheader()
                firstrun = False
            writer.writerow(logbook)
        sys.stdout.flush()

        # Saving checkpoints
        if g % LOG_FREQ == 0:
            # Fill the dictionary using the dict(key=value[, ...]) constructor
            cp = dict(
                best_ind         = deap_mpi.scale_fb(best_ind),
                best_fitness     = best_fitness,
                sigma            = fb_cmaes_strategy.sigma,
                start_gen        = g,
                halloffame       = cp["halloffame"],
                logbook          = cp["logbook"],
                rndstate         = random.getstate()
                )

            with open("{}/{}_{}.pkl".format(CHECKPOINT_PATH, ''.join(args.checkpoint.split('_')[0:-1]),g), "wb") as cp_file:
                pickle.dump(cp, cp_file)

if rank is not MASTER or TEST:
    #   ██████  ██▓    ▄▄▄    ██▒   █▓▓█████   ██████
    # ▒██    ▒ ▓██▒   ▒████▄ ▓██░   █▒▓█   ▀ ▒██    ▒
    # ░ ▓██▄   ▒██░   ▒██  ▀█▄▓██  █▒░▒███   ░ ▓██▄
    #   ▒   ██▒▒██░   ░██▄▄▄▄██▒██ █░░▒▓█  ▄   ▒   ██▒
    # ▒██████▒▒░██████▒▓█   ▓██▒▒▀█░  ░▒████▒▒██████▒▒
    # ▒ ▒▓▒ ▒ ░░ ▒░▓  ░▒▒   ▓▒█░░ ▐░  ░░ ▒░ ░▒ ▒▓▒ ▒ ░
    # ░ ░▒  ░ ░░ ░ ▒  ░ ▒   ▒▒ ░░ ░░   ░ ░  ░░ ░▒  ░ ░
    # ░  ░  ░    ░ ░    ░   ▒     ░░     ░   ░  ░  ░
    #       ░      ░  ░     ░  ░   ░     ░  ░      ░
    #                             ░
    env_init()
    if TEST:
        for i in range(REPEAT):
            print("")
            print("")
            print("")
            print("The following parameters have been loaded:")
            print(cp['best_ind'])
            individual = cp['best_ind']
            ret = F(individual)
            print("rewards: {}".format(ret[0]))
    else:
        while True:
            if(comm.Iprobe(source=0, tag=rank)):
                individual,best,beta = comm.recv(source=0, tag=rank)
                if DEBUG: print("[WORKER {}] received individual, doing some work".format(rank))
                sys.stdout.flush()
                to_send = F(individual)
                comm.send(to_send, dest=0, tag=NP+rank)
                if DEBUG: print("[WORKER {}] done, fit={}".format(rank,to_send[0]))
                if DEBUG: sys.stdout.flush()
            time.sleep(0.001)
