"""
control_manager.py: contains the CtrlManager class used to manage the control functions
"""
__author__ = "Aliaa Diab"
__copyright__ = "Copyright 2023, Alpine Intuition SARL"
__license__ = "Apache-2.0 license"
__version__ = "1.0.0"
__email__ = "berat.denizdurduran@alpineintuition.ch"
__status__ = "Stable"


import os
import time
import pickle
import numpy as np
from .reflex_controller import OsimReflexCtrl

class CtrlManager:
    def __init__(self, controller_params: dict, sim_dt: float = 0.01) -> None:
        self.sim_dt = sim_dt
        fb_params_2D = np.loadtxt('./control/params_2D.txt')
        self.fb_N_2D = len(fb_params_2D)
        self.init_ctrl()
        self.set_param(controller_params)
    
    def init_ctrl(self):
        # controller
        self.FBCtrl = OsimReflexCtrl(mode='2D', dt=self.sim_dt)

        # reward and step init
        self.total_reward = 0
        self.t = 0
        self.i = 0
        self.elapsed_total = 0
        self._total_obs_sum = 0 # For reproducibility
        self.velocity = []
        
        # Record muscle activities
        self.muscActLog = []

        # Record joint angles
        self.JointLog = []
        self.JointVelLog = []
        
    def set_param(self, x):
        self.FBCtrl.set_control_params(np.round(x[0:self.fb_N_2D],4))
        return x
    
    def fb_update(self, obs_dict: dict):
        return self.FBCtrl.update(obs_dict)
    
    def run_step(self, obs_dict, obs):
        self._total_obs_sum += sum(obs)
        self.i += 1
        self.t += self.sim_dt
        self._t = time.time()
        speed = obs[3]
        self.velocity.append(speed)
        action_fdb = self.fb_update(obs_dict)
        action = action_fdb
        exo_actuation = np.zeros(6)
        action = np.concatenate((np.array(action), exo_actuation))
        self.muscActLog.append(action)
        return action

    def update_logs(self, joint_pos, joint_vel):
        self.JointLog.append(joint_pos)
        self.JointVelLog.append(joint_vel)

    def update_reward(self, reward):
        self.total_reward += reward

    def update_time(self):
        self.elapsed_total += time.time() - self._t

    def save_logs(self, exp_id, duration):
        SAVE_PATH = './logs/simulation_CMAES/cmaes_spinal_'+str(exp_id)
        if not os.path.exists(SAVE_PATH):
            os.makedirs(SAVE_PATH)

        # Saving the various data into separate .PKL files
        print("Saving logs to {}/_{}_second_[muscle_act|joint_pos|joint_vel].pkl".format(SAVE_PATH.split('.pkl')[0], str(duration)))
        musclePath = SAVE_PATH + '/_' + str(duration) + '_second' + '_muscle_act.pkl'
        jointPath = SAVE_PATH + '/_' + str(duration) + '_second_all' + '_joints.pkl'
        jointvelPath = SAVE_PATH + '/_' + str(duration) + '_second_all' + '_joints_vel.pkl'
        pickle.dump(self.muscActLog, open(musclePath, "wb"))
        pickle.dump(self.JointLog, open(jointPath, "wb"))
        pickle.dump(self.JointVelLog, open(jointvelPath, 'wb'))

        # reset logs
        self.reset_logs()

    def reset_logs(self):
        # reset reward and step init
        self.total_reward = 0
        self.t = 0
        self.i = 0
        self.elapsed_total = 0
        self._total_obs_sum = 0 # For reproducibility
        self.velocity = []

        # Reset muscle activities log
        self.muscActLog = []

        # Reset joint angles log
        self.JointLog = []
        self.JointVelLog = []