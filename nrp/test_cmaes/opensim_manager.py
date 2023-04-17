"""
Base script provided by the HBP team
Modified by: Aliaa Diab and Raphael Gaiffe
Contact: "berat.denizdurduran@alpineintuition.ch"
"""
import pickle
import numpy as np
from control.control_manager import CtrlManager
from nrp_core.engines.opensim import OpenSimEngineScript
from nrp_core.engines.python_json import RegisterEngine
# The API of Opensim is shown in the following link:
# https://simtk.org/api_docs/opensim/api_docs

@RegisterEngine()
class Script(OpenSimEngineScript):
    def initialize(self):
        """Initialize datapack1 with time"""
        print("Server Engine is initializing")
        print("Registering datapack --> for sensors")
        self._registerDataPack("joints")
        self._setDataPack("joints", {"hip_l": 0, "knee_l": 0, "ankle_l": 0,
                                        "hip_r": 0, "knee_r": 0, "ankle_r": 0,
                                        "exo_joint_hip_r": 0, "exo_joint_knee_r": 0,
                                        "exo_joint_ankle_r": 0, "exo_joint_hip_l": 0,
                                        "exo_joint_knee_l": 0, "exo_joint_ankle_l": 0})
        self._registerDataPack("infos")
        self._setDataPack("infos", {"time": 0})

        sim_dt = self.sim_manager.time_step
        self.duration = self.sim_manager.duration

        # load controller parameters
        with open('./_40.pkl', 'rb') as f:
            optimisation = pickle.load(f)

        # controller
        individual = optimisation['best_ind']
        self.controller = CtrlManager(controller_params=individual, sim_dt=sim_dt)

        self.sim_max_step = int(self.duration / sim_dt)

        print("Registering datapack --> for actuators")
        self._registerDataPack("control_cmd")
        self.count = 0
        self.sim_id = 0

    def runLoop(self, timestep):

        if self.count == self.sim_max_step:
            print("rewards: {}".format(self.controller.total_reward))
            print('Saving simulation log ...')
            self.controller.save_logs(self.sim_id, self.duration)
            self.shutdown()

        # All Joints and Muscles can be found in the "*.osim". Obtain the joint data from model
        hip_l_val = self.sim_manager.get_model_property("hip_l", datapack_type="Joint")
        knee_l_val = self.sim_manager.get_model_property("knee_l", datapack_type="Joint")
        ankle_l_val = self.sim_manager.get_model_property("ankle_l", datapack_type="Joint")
        hip_r_val = self.sim_manager.get_model_property("hip_r", datapack_type="Joint")
        knee_r_val = self.sim_manager.get_model_property("knee_r", datapack_type="Joint")
        ankle_r_val = self.sim_manager.get_model_property("ankle_r", datapack_type="Joint")
        exo_hip_r_val = self.sim_manager.get_model_property("exo_joint_hip_r", datapack_type="Joint")
        exo_knee_r_val = self.sim_manager.get_model_property("exo_joint_knee_r", datapack_type="Joint")
        exo_ankle_r_val = self.sim_manager.get_model_property("exo_joint_ankle_r", datapack_type="Joint")
        exo_hip_l_val = self.sim_manager.get_model_property("exo_joint_hip_l", datapack_type="Joint")
        exo_knee_l_val = self.sim_manager.get_model_property("exo_joint_knee_l", datapack_type="Joint")
        exo_ankle_l_val = self.sim_manager.get_model_property("exo_joint_ankle_l", datapack_type="Joint")

        # Send data to TF
        self._setDataPack("joints", {"hip_l": hip_l_val, "knee_l": knee_l_val, "ankle_l": ankle_l_val,
                                    "hip_r": hip_r_val, "knee_r": knee_r_val, "ankle_r": ankle_r_val,
                                    "exo_joint_hip_r": exo_hip_r_val, "exo_joint_knee_r": exo_knee_r_val,
                                    "exo_joint_ankle_r": exo_ankle_r_val, "exo_joint_hip_l": exo_hip_l_val,
                                    "exo_joint_knee_l": exo_knee_l_val, "exo_joint_ankle_l": exo_ankle_l_val})
        self._setDataPack("infos", {"time": self.sim_manager.get_sim_time()})

        # get observations
        obs_dict, obs, joint_pos, joint_vel = self.sim_manager.get_observation()
        self.controller.update_logs(joint_pos, joint_vel)
        action = self.controller.run_step(obs_dict, obs)

        # Set muscles' force to change joints
        reward = self.sim_manager.run_step(action, timestep)
        self.controller.update_time()
        self.controller.update_reward(reward)
        self.count += 1

    def reset(self):
        print("resetting the opensim simulation...")
        # Reset the value of set datapacks
        self._setDataPack("joints", {"hip_l": 0, "knee_l": 0, "ankle_l": 0,
                                     "hip_r": 0, "knee_r": 0, "ankle_r": 0,         
                                        "exo_joint_hip_r": 0, "exo_joint_knee_r": 0,
                                        "exo_joint_ankle_r": 0, "exo_joint_hip_l": 0,
                                        "exo_joint_knee_l": 0, "exo_joint_ankle_l": 0})
        self._setDataPack("infos", {"time": 0})
        # Reset simulation model
        self.sim_manager.reset()
        
    def shutdown(self):
        print("Engine 1 is shutting down")


