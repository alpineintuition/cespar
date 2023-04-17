"""
Base script was provided by the HBP team
Modified by: Aliaa Diab and Raphael Gaiffe
Contact: "berat.denizdurduran@alpineintuition.ch"
"""

import numpy as np
from .envs.osim_envs import L2M2019Env


# Initial position of the model
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

class OpensimInterface(object):

    def __init__(self, model_name, start_visualizer, time_step):
        super(OpensimInterface, self).__init__()

        self.step_size = time_step

        self.env = L2M2019Env(visualize=start_visualizer, seed=0, difficulty=0, desired_speed=1.4, model_path=model_name)
        self.env.change_model(model='2D', difficulty=0, seed=0)

        self.obs_dict = self.env.reset(project=True, seed=0, obs_as_dict=True, init_pose=INIT_POSE)
        self.obs = self.env.get_observation()

        self.n_step = 0
        self.model = self.env.osim_model

        self.jointSet = self.model.jointSet
        self.forceSet = self.model.forceSet

        self.positions = []
        self.velocities = []

        self.state = self.env.osim_model.model_state

    def get_observation(self):
        joint_pos = self.env.get_observation_list_joints_pos()
        joint_vel = self.env.get_observation_list_joints_vel()
        return self.obs_dict, self.obs, joint_pos, joint_vel

    # Run simulation step by step
    def run_one_step(self, action, timestep):
        self.obs_dict, reward, done, info = self.env.step(action, project = True, obs_as_dict=True)
        self.obs = self.env.get_observation()
        # Define the new endtime of the simulation
        self.n_step = self.n_step + 1
        pos = self.env.get_observation_list_joints_pos()
        vel = self.env.get_observation_list_joints_vel()
        self.positions.append(pos)
        self.velocities.append(vel)
        return reward

    def reset(self):
        self.env.reset(project=True, seed=0, obs_as_dict=True, init_pose=INIT_POSE)
        self.n_step = 0

    # Set the value of controller
    def actuate(self, action):
        self.env.osim_model.actuate(action)

    def save_sim_data(self, sim_id: int) -> None:
        positions = np.array(self.positions)
        formatted_postions = np.hstack((positions[:,0,:], positions[:,1,:], positions[:,2,:]))

        velocities = np.array(self.velocities)
        formatted_velocities = np.hstack((velocities[:,0,:], velocities[:,1,:], velocities[:,2,:]))

        np.save('nrp_pos' + str(sim_id) + '.npy', formatted_postions)
        np.save('nrp_vel' + str(sim_id) + '.npy', formatted_velocities)

    # Obtain datapack names, which can also be found in the model file "*.osim"
    def get_model_properties(self, p_type):
        if p_type == "Joint":
            tSet = self.jointSet
        elif p_type == "Force":
            tSet = self.forceSet
        else:
            print("supported types are 'Joint' and 'Force'")
            return []

        return [tSet.get(i).getName() for i in range(tSet.getSize())]

    # Obtain the value of one datapack by the datapack name
    def get_model_property(self, p_name, p_type):
        if p_type == "Joint":
            tSet = self.jointSet
        elif p_type == "Force":
            tSet = self.forceSet
        else:
            print("p_type is error")
            print("In this function, it only supports Joint and Force")
            return []

        if tSet.get(p_name).numCoordinates() == 1:
            prop = tSet.get(p_name).getCoordinate()
        else:
            prop = tSet.get(p_name).get_coordinates(0)
        return prop.getValue(self.state), prop.getSpeedValue(self.state)

    def get_sim_time(self):
        return self.n_step * self.step_size

    def reset_manager(self):
        self.env.osim_model.reset_manager()

    # Obtain a list of the values of the muscle model's joints positions
    def get_observation_list_joints_pos(self):

        state_desc = self.get_state_desc()
        hip_joint_pos = [-state_desc['joint_pos']['hip_{}'.format('r')][0], -state_desc['joint_pos']['hip_{}'.format('l')][0]]
        knee_joint_pos = [state_desc['joint_pos']['knee_{}'.format('r')][0], state_desc['joint_pos']['knee_{}'.format('l')][0]]
        ankle_joint_pos = [-state_desc['joint_pos']['ankle_{}'.format('r')][0], -state_desc['joint_pos']['ankle_{}'.format('l')][0]]
        return [hip_joint_pos, knee_joint_pos, ankle_joint_pos]