import numpy as np

from envs.target import VTgtField
from model import Model

G = 9.80665


def rotate_frame(x, y, theta):
    x_rot = np.cos(theta) * x - np.sin(theta) * y
    y_rot = np.sin(theta) * x + np.cos(theta) * y
    return x_rot, y_rot


class Environment:
    def __init__(
        self,
        model: Model,
        desired_speed: float,
        timestep_limit: float,
        difficulty: int,
        seed: int,
    ):
        self.model = model

        if difficulty not in [0, 1, 2, 3]:
            raise ValueError("difficulty level should be in [0, 1, 2, 3].")

        self.desired_speed = desired_speed
        self.difficulty = difficulty
        self.timestep_limit = timestep_limit
        self.seed = seed

        self.num_steps = 0
        self.leg_length = 1.0  # TODO: remove hardcoded

    def _init_vtgt(self):
        self.vtgt = VTgtField(
            visualize=False,
            version=self.difficulty,
            dt=self.model.step_size,
        )
        self.vtgt.reset(version=self.difficulty, seed=self.seed)

        state = self.model.get_state()
        pose = np.array(
            [
                state["body_positions"]["pelvis"][0],
                -state["body_positions"]["pelvis"][2],
                state["joint_positions"]["ground_pelvis"][2],
            ]
        )
        self.v_tgt_field, self.flag_new_v_tgt_field = self.vtgt.update(pose)

    def _init_footstep(self):
        self.footstep = {
            "n": 0,
            "new": False,
            "r_contact": 1,
            "l_contact": 1,
        }

    def _init_reward(self):
        self.reward = {
            "weight": {
                "footstep": 10,
                "effort": 1,
                "v_tgt": 1,
                "v_tgt_R2": 3,
            },
            "alive": 0.1,
            "effort": 0,
            "footstep": {
                "effort": 0,
                "del_t": 0,
                "del_v": 0,
            },
        }

    def init(self):
        self.model.reset()
        self.num_steps = 0
        self._init_vtgt()
        self._init_footstep()
        self._init_reward()

    def _get_reward(self, observation, dt):
        reward = 0

        # alive reward, should be large enough to search for 'success'
        # solutions (alive to the end) first

        reward += self.reward["alive"]

        # effort ~ muscle fatigue ~ (muscle activation)^2

        muscles = observation["muscles"]

        muscle_activation = 0
        for muscle in muscles.values():
            muscle_activation += np.square(muscle["activation"])

        self.reward["effort"] += muscle_activation * dt
        self.reward["footstep"]["effort"] += muscle_activation * dt
        self.reward["footstep"]["del_t"] += dt

        # reward from velocity (penalize from deviating from v_tgt)

        body_velocities = observation["body_velocities"]
        body_velocity = [body_velocities["pelvis"][0], -body_velocities["pelvis"][2]]
        body_velocity_target = np.array([self.desired_speed, 0.0])
        self.reward["footstep"]["del_v"] += dt * (body_velocity - body_velocity_target)

        # footstep reward (when made a new step)

        if self.footstep["new"]:
            # footstep reward: so that solution does not avoid making footsteps
            # scaled by del_t, so that solution does not get higher rewards by
            # making unnecessary (small) steps

            reward_footstep_0 = (
                self.reward["weight"]["footstep"] * self.reward["footstep"]["del_t"]
            )

            # deviation from target velocity
            # the average velocity a step (instead of instantaneous velocity) is used
            # as velocity fluctuates within a step in normal human walking

            reward_footstep_v = (
                -self.reward["weight"]["v_tgt"]
                * np.linalg.norm(self.reward["footstep"]["del_v"])
                / self.leg_length
            )

            # penalize effort

            reward_footstep_e = (
                -self.reward["weight"]["effort"] * self.reward["footstep"]["effort"]
            )

            self.reward["footstep"]["del_t"] = 0
            self.reward["footstep"]["del_v"] = 0
            self.reward["footstep"]["effort"] = 0

            reward += reward_footstep_0 + reward_footstep_v + reward_footstep_e

        return reward

    def _update_footstep(self, observation):
        forces, mass = observation["forces"], self.model.mass

        r_contact = True if forces["foot_r"][1] < -0.05 * (mass * G) else False
        r_contact = not self.footstep["r_contact"] and r_contact

        l_contact = True if forces["foot_l"][1] < -0.05 * (mass * G) else False
        l_contact = not self.footstep["l_contact"] and l_contact

        self.footstep["new"] = False
        if r_contact or l_contact:
            self.footstep["new"] = True
            self.footstep["n"] += 1
        self.footstep["r_contact"] = r_contact
        self.footstep["l_contact"] = l_contact

    def _get_observations(self, state):
        body_positions = state["body_positions"]
        body_velocities = state["body_velocities"]

        joint_positions = state["joint_positions"]
        joint_velocities = state["joint_velocities"]

        observation = {}

        # velocity target

        observation["v_tgt_field"] = self.v_tgt_field

        # joints positions

        observation["joint_positions"] = {
            "hip_r": -joint_positions["hip_r"][0],
            "hip_l": -joint_positions["hip_l"][0],
            "knee_r": joint_positions["knee_r"][0],
            "knee_l": joint_positions["knee_l"][0],
            "ankle_r": -joint_positions["ankle_r"][0],
            "ankle_l": -joint_positions["ankle_l"][0],
        }

        # joints velocities

        observation["joint_velocities"] = {
            "hip": {
                "r": -joint_velocities["hip_r"][0],
                "l": -joint_velocities["hip_l"][0],
            },
            "knee": {
                "r": joint_velocities["knee_r"][0],
                "l": joint_velocities["knee_l"][0],
            },
            "ankle": {
                "r": -joint_velocities["ankle_r"][0],
                "l": -joint_velocities["ankle_l"][0],
            },
        }

        # pelvis state (in local frame)

        yaw = joint_positions["ground_pelvis"][2]

        dx_local, dy_local = rotate_frame(
            body_velocities["pelvis"][0],
            body_velocities["pelvis"][2],
            yaw,
        )
        dz_local = body_velocities["pelvis"][1]

        observation["pelvis"] = {
            "height": body_positions["pelvis"][1],
            # (+) pitching forward
            "pitch": -joint_positions["ground_pelvis"][0],
            # (+) rolling around the forward axis (to the right)
            "roll": joint_positions["ground_pelvis"][1],
            "vel": [
                # (+) forward
                dx_local,
                # (+) leftward
                -dy_local,
                # (+) upward
                dz_local,
                # (+) pitch angular velocity
                -joint_velocities["ground_pelvis"][0],
                # (+) roll angular velocity
                joint_velocities["ground_pelvis"][1],
                # (+) yaw angular velocity
                joint_velocities["ground_pelvis"][2],
            ],
        }

        # leg state

        for side in ["r", "l"]:
            leg = f"{side}_leg"

            # forces normalized by bodyweight

            forces = state["forces"]
            grf = [f / (self.model.mass * G) for f in forces[f"foot_{side}"][0:3]]
            grfx_local, grfy_local = rotate_frame(-grf[0], -grf[2], yaw)

            # ground reactions forces
            # (+) forward
            # (+) lateral (rightward / leftward)
            # (+) upward

            if side == "r":
                ground_reaction_forces = [grfx_local, grfy_local, -grf[1]]
            else:  # leg == "l"
                ground_reaction_forces = [grfx_local, -grfy_local, -grf[1]]

            # joint angles
            joint = {
                # (+) hip abduction
                "hip_abd": -joint_positions[f"hip_{side}"][1],
                # (+) extension
                "hip": -joint_positions[f"hip_{side}"][0],
                # (+) extension
                "knee": joint_positions[f"knee_{side}"][0],
                # (+) extension
                "ankle": -joint_positions[f"ankle_{side}"][0],
            }

            # joint angular velocities

            d_joint = {
                # (+) hip abduction
                "hip_abd": -joint_velocities[f"hip_{side}"][1],
                # (+) extension
                "hip": -joint_velocities[f"hip_{side}"][0],
                # (+) extension
                "knee": joint_velocities[f"knee_{side}"][0],
                # (+) extension
                "ankle": -joint_velocities[f"ankle_{side}"][0],
            }

            # muscles

            muscles = {}
            for name, abbraviation in self.model.muscles.items():
                muscles[abbraviation] = {}

                fiber_force = state["muscles"][f"{name}_{side}"]["fiber_force"]
                fmax = self.model.Fmax[leg][abbraviation]
                muscles[abbraviation]["f"] = fiber_force / fmax

                fiber_length = state["muscles"][f"{name}_{side}"]["fiber_length"]
                lopt = self.model.lopt[leg][abbraviation]
                muscles[abbraviation]["l"] = fiber_length / lopt

                fiber_velocity = state["muscles"][f"{name}_{side}"]["fiber_velocity"]
                muscles[abbraviation]["v"] = fiber_velocity / lopt

            observation[leg] = {
                "joint": joint,
                "d_joint": d_joint,
                "ground_reaction_forces": ground_reaction_forces,
                **muscles,
            }

        return observation

    def get_observation(self):
        return self._get_observations(self.model.get_state())

    def step(self, action):
        self.num_steps += 1

        self.model.actuate(action)

        state = self.model.get_state()
        observation = self._get_observations(state)
        reward = self._get_reward(state, self.model.step_size)
        speed = 1

        done = False
        if (
            self.num_steps >= self.timestep_limit
            or state["body_positions"]["pelvis"][1] < self.model.height_threshold
        ):
            done = True

        self._update_footstep(state)

        pose = np.array(
            [
                state["body_positions"]["pelvis"][0],
                -state["body_positions"]["pelvis"][2],
                state["joint_positions"]["ground_pelvis"][2],
            ]
        )

        self.v_tgt_field, self.flag_new_v_tgt_field = self.vtgt.update(pose)

        return (observation, reward, speed, pose, done)
