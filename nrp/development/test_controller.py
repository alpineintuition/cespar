"""test_controller.py: Isolates the controller functions to reproduce the simulation results when running the coupled_gait_optimization_cmaes.py script.
    Simply, adjust the checkpoint and duration of the simulation in the main function, then run the script.
"""

__author__ = "Berat Denizdurduran, Florin Dzeladini, Aliaa Diab, Carla Nannini, Raphael Gaiffe"
__copyright__ = "Copyright 2023, Alpine Intuition SARL"
__license__ = "Apache-2.0 license"
__version__ = "1.0.0"
__email__ = "berat.denizdurduran@alpineintuition.ch"
__status__ = "Stable"

import time
import pickle
import numpy as np
from control.osim_HBP_withexo_CMAES import L2M2019Env
from control.osim_loco_reflex_song2019 import OsimReflexCtrl

# simulation parameters
SEED = 64
TEST = True
sim_dt = 0.01
difficulty = 0
VISUALIZE = True
DESIRED_SPEED = 1.4
INITIAL_SPEED = 1.7
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
fb_params_2D = np.loadtxt('./control/params_2D.txt')
fb_N_2D = len(fb_params_2D)
env = L2M2019Env(visualize=VISUALIZE, seed=SEED, difficulty=difficulty, desired_speed=DESIRED_SPEED)

def set_param(x):
    global FBCtrl
    FBCtrl.set_control_params(np.round(x[0:fb_N_2D],4))
    return x

def init_ctrl():
    global FBCtrl
    FBCtrl = OsimReflexCtrl(mode='2D', dt=sim_dt)

def fb_update(obs_dict):
    global FBCtrl
    return FBCtrl.update(obs_dict)

def loop(init, set_param, exp_id, duration):
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

    # Save the log to the checkpoint folder but only when testing
    if TEST:
        SAVE_PATH = './logs/simulation_CMAES/'
        checkpoint = 'Results_CMAES/cmaes_spinal_'+str(exp_id)
        SAVE_PATH += checkpoint.split('/')[-1]

        # Saving the various data into separate .PKL files
        print("Saving logs to {}_[muscle_act|joint_pos|joint_vel]_{}_second.pkl".format(SAVE_PATH.split('.pkl')[0], str(duration)))
        musclePath = SAVE_PATH.replace('.pkl', '_' + str(duration) + '_second' + '_muscle_act.pkl')
        jointPath = SAVE_PATH.replace('.pkl', '_' + str(duration) + '_second_all' + '_joints.pkl')
        jointvelPath = SAVE_PATH.replace('.pkl', '_' + str(duration) + '_second_all' + '_joints_vel.pkl')
        pickle.dump(muscActLog, open(musclePath, "wb"))
        pickle.dump(JointLog, open(jointPath, "wb"))
        pickle.dump(JointVelLog, open(jointvelPath, 'wb'))


    return total_reward, t,elapsed_total/i,env.pose[0], speed, i

def F(x, exp_id: int = 0, duration: float = 5.0):
    return loop(init_ctrl, lambda: set_param(x), exp_id, duration)

def main(checkpoint: str, exp_id: int = 0, duration: float = 5.0):
    cp = {}
    # Load checkpoint from file
    if(checkpoint):
        with open("{}".format(checkpoint), "rb") as cp_file:
            cp = pickle.load(cp_file)
        # if(args.force_sigma):
        #     cp["sigma"] = SIGMA
        print("best fitness : {}".format(cp["best_fitness"]))
        print("sigma        : {}".format(cp["sigma"]))
    
    print("")
    print("")
    print("")
    print("The following parameters have been loaded:")
    print(cp['best_ind'])
    individual = cp['best_ind']
    ret = F(individual, exp_id, duration)
    print("rewards: {}".format(ret[0]))
    print(SEED)

if __name__ == '__main__':
    checkpoint = 'Results_CMAES/cmaes_spinal_1/_40.pkl'
    main(checkpoint, duration = 10)
