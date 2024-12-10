from typing import Dict, List, Union

import numpy as np
import opensim as osim

from utils import no_stdout

ACTION_TO_MUSCLE = [
    0,
    1,
    4,
    7,
    3,
    2,
    5,
    6,
    8,
    9,
    10,
    11,
    12,
    15,
    18,
    14,
    13,
    16,
    17,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
]


class Model(object):
    def __init__(
        self,
        model_path: str,
        exoskeleton: bool,
        initial_speed: float,
        visualize: bool,
        integrator_accuracy: float = 5e-5,
        step_size: float = 0.01,
    ):
        self.integrator_accuracy = integrator_accuracy
        self.step_size = step_size

        with no_stdout():
            self.model = osim.Model(model_path)
            self.model.initSystem()

        self.model.setUseVisualizer(visualize)

        self.muscle_set = self.model.getMuscles()
        self.force_set = self.model.getForceSet()
        self.body_set = self.model.getBodySet()
        self.joint_set = self.model.getJointSet()
        self.marker_set = self.model.getMarkerSet()
        self.contact_geometry_set = self.model.getContactGeometrySet()

        self.brain = osim.PrescribedController()
        self.maxforces = []
        self.curforces = []
        for j in range(self.muscle_set.getSize()):
            func = osim.Constant(1.0)
            self.brain.addActuator(self.muscle_set.get(j))
            self.brain.prescribeControlForActuator(j, func)
            self.maxforces.append(self.muscle_set.get(j).getMaxIsometricForce())
            self.curforces.append(1.0)

        self.exoskeleton = exoskeleton
        if self.exoskeleton:
            actuators = (
                "exo_ankle_motor_r",
                "exo_ankle_motor_l",
                "exo_knee_motor_r",
                "exo_knee_motor_l",
                "exo_hip_motor_flex_right",
                "exo_hip_motor_flex_left",
            )

            for idx, actuator in enumerate(actuators, start=22):
                func = osim.Constant(1.0)
                self.brain.addActuator(self.model.getActuators().get(actuator))
                self.brain.prescribeControlForActuator(idx, func)

        self.model.addController(self.brain)
        self.model_state = self.model.initSystem()

        state = self.model.updWorkingState()
        self.mass = self.model.getTotalMass(state)

        coords = self.model.updCoordinateSet()
        pelvis_height = coords.get("pelvis_ty").get_default_value()
        self.height_threshold = 0.3 * pelvis_height

        self.initial_pose = np.array(
            [
                initial_speed,  # forward speed
                0.5,  # rightward speed
                pelvis_height,
                2.012303881285582852e-01,  # trunk lean
                0 * np.pi / 180,  # [right] hip adduct
                -6.952390849304798115e-01,  # hip flex
                -3.231075259785813891e-01,  # knee extend
                1.709011708233401095e-01,  # ankle flex
                0 * np.pi / 180,  # [left] hip adduct
                -5.282323914341899296e-02,  # hip flex
                -8.041966456860847323e-01,  # knee extend
                -1.745329251994329478e-01,  # ankle flex
            ]
        )

        self._init_state(self.initial_pose)
        self._init_muscles()
        self._init_manager()

        # self.model.realizeAcceleration(self.state)
        # print("NEW_realize_acceleration")

        self.istep = 0

        self.action_to_muscle = ACTION_TO_MUSCLE
        if not exoskeleton:
            self.action_to_muscle = self.action_to_muscle[:22]

    def _init_muscles(self):
        self.muscles = {
            "abd": "HAB",
            "add": "HAD",
            "iliopsoas": "HFL",
            "glut_max": "GLU",
            "hamstrings": "HAM",
            "rect_fem": "RF",
            "vasti": "VAS",
            "bifemsh": "BFSH",
            "gastroc": "GAS",
            "soleus": "SOL",
            "tib_ant": "TA",
        }

        self.Fmax = {}
        self.lopt = {}

        for side in ["r", "l"]:
            leg = f"{side}_leg"

            self.Fmax[f"{side}_leg"] = {}
            self.lopt[f"{side}_leg"] = {}

            for muscle, abbraviation in self.muscles.items():
                muscle = self.muscle_set.get(f"{muscle}_{side}")
                self.Fmax[leg][abbraviation] = muscle.getMaxIsometricForce()
                self.lopt[leg][abbraviation] = muscle.getOptimalFiberLength()

        with no_stdout():
            self.model.equilibrateMuscles(self.state)

    def _init_state(self, initial_pose):
        state = self.model.initializeState()

        QQ = state.getQ()
        QQDot = state.getQDot()

        QQDot = state.getQDot()
        for i in range(17):
            QQDot[i] = 0
        QQDot[3] = initial_pose[0]  # forward speed
        QQDot[5] = initial_pose[1]  # forward speed

        QQ[3] = 0  # x: (+) forward
        QQ[5] = 0  # z: (+) right
        QQ[1] = 0 * np.pi / 180  # roll
        QQ[2] = 0 * np.pi / 180  # yaw
        QQ[4] = initial_pose[2]  # pelvis height
        QQ[0] = -initial_pose[3]  # trunk lean: (+) backward

        QQ[7] = -initial_pose[4]  # right hip abduct
        QQ[6] = -initial_pose[5]  # right hip flex
        QQ[13] = initial_pose[6]  # right knee extend
        QQ[15] = -initial_pose[7]  # right ankle flex

        QQ[10] = -initial_pose[8]  # left hip adduct
        QQ[9] = -initial_pose[9]  # left hip flex
        QQ[14] = initial_pose[10]  # left knee extend
        QQ[16] = -initial_pose[11]  # left ankle flex

        state.setQ(QQ)
        state.setU(QQDot)

        self.state = state
        self.state.setTime(0)

    def _init_manager(self):
        self.manager = osim.Manager(self.model)
        self.manager.setIntegratorAccuracy(self.integrator_accuracy)
        self.manager.initialize(self.state)

    def reset(self):
        self._init_state(self.initial_pose)
        self._init_muscles()
        self._init_manager()
        self.istep = 0

        self.model.realizeAcceleration(self.state)

    def get_elements(self) -> Dict[str, Union[float, List[float]]]:
        elements = {}

        for i in range(self.body_set.getSize()):
            body_set = self.body_set.get(i)
            name = body_set.getName()

            mass = body_set.getMass()
            elements[f"body_{name}_mass"] = mass

            mass_center = body_set.getMassCenter()
            elements[f"body_{name}_mass_center"] = [
                mass_center.get(0),
                mass_center.get(1),
                mass_center.get(2),
            ]

            frame_geometry = body_set.getComponent("frame_geometry")
            elements[f"{name}_scale_factor"] = [
                frame_geometry.get_scale_factors().get(0),
                frame_geometry.get_scale_factors().get(1),
                frame_geometry.get_scale_factors().get(2),
            ]

        return elements

    def get_state(self):
        #
        # joints
        #

        joint_positions, joint_velocities, joint_accelerations = {}, {}, {}
        for i in range(self.joint_set.getSize()):
            joint = self.joint_set.get(i)
            name = joint.getName()

            joint_positions[name] = [
                joint.get_coordinates(i).getValue(self.state)
                for i in range(joint.numCoordinates())
            ]
            joint_velocities[name] = [
                joint.get_coordinates(i).getSpeedValue(self.state)
                for i in range(joint.numCoordinates())
            ]
            joint_accelerations[name] = [
                joint.get_coordinates(i).getAccelerationValue(self.state)
                for i in range(joint.numCoordinates())
            ]

        #
        # bodies
        #

        body_positions = {}
        body_velocities = {}
        body_accelerations = {}
        body_positions_rotation = {}
        body_velocities_rotation = {}
        body_accelerations_rotation = {}

        for i in range(self.body_set.getSize()):
            body = self.body_set.get(i)
            name = body.getName()

            body_positions[name] = [
                body.getTransformInGround(self.state).p()[i] for i in range(3)
            ]
            body_velocities[name] = [
                body.getVelocityInGround(self.state).get(1).get(i) for i in range(3)
            ]
            body_accelerations[name] = [
                body.getAccelerationInGround(self.state).get(1).get(i) for i in range(3)
            ]

            body_positions_rotation[name] = [
                body.getTransformInGround(self.state)
                .R()
                .convertRotationToBodyFixedXYZ()
                .get(i)
                for i in range(3)
            ]
            body_velocities_rotation[name] = [
                body.getVelocityInGround(self.state).get(0).get(i) for i in range(3)
            ]
            body_accelerations_rotation[name] = [
                body.getAccelerationInGround(self.state).get(0).get(i) for i in range(3)
            ]

        #
        # muscles
        #

        muscles = {}
        for i in range(self.muscle_set.getSize()):
            muscle = self.muscle_set.get(i)
            name = muscle.getName()

            muscles[name] = {
                "activation": muscle.getActivation(self.state),
                "fiber_length": muscle.getFiberLength(self.state),
                "fiber_velocity": muscle.getFiberVelocity(self.state),
                "fiber_force": muscle.getFiberForce(self.state),
            }

        #
        # forces
        #

        forces = {}
        for i in range(self.force_set.getSize()):
            force = self.force_set.get(i)
            name = force.getName()

            values = force.getRecordValues(self.state)
            forces[name] = [values.get(i) for i in range(values.size())]

        #
        # markers
        #

        markers = {}
        for i in range(self.marker_set.getSize()):
            marker = self.marker_set.get(i)
            name = marker.getName()

            markers[name] = {
                "pos": [marker.getLocationInGround(self.state)[i] for i in range(3)],
                "vel": [marker.getVelocityInGround(self.state)[i] for i in range(3)],
                "acc": [
                    marker.getAccelerationInGround(self.state)[i] for i in range(3)
                ],
            }

        #
        # others
        #

        misc = {
            "mass_center_pos": [
                self.model.calcMassCenterPosition(self.state)[i] for i in range(3)
            ],
            "mass_center_vel": [
                self.model.calcMassCenterVelocity(self.state)[i] for i in range(3)
            ],
            "mass_center_acc": [
                self.model.calcMassCenterAcceleration(self.state)[i] for i in range(3)
            ],
        }

        return {
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "joint_accelerations": joint_accelerations,
            "body_positions": body_positions,
            "body_velocities": body_velocities,
            "body_accelerations": body_accelerations,
            "body_positions_rotation": body_positions_rotation,
            "body_velocities_rotation": body_velocities_rotation,
            "body_accelerations_rotation": body_accelerations_rotation,
            "forces": forces,
            "muscles": muscles,
            "markers": markers,
            "misc": misc,
        }

    def actuate(self, action: np.ndarray):
        if np.any(np.isnan(action)):
            raise ValueError(
                "NaN passed in the activation vector. "
                + "Values in [0,1] interval are required."
            )

        action = [action[i] for i in self.action_to_muscle]

        clipped_action = np.clip(action[0:21], 0.0, 1.0)  # muscle actions
        if self.exoskeleton:
            clipped_exo_action = np.clip(action[21:], -10.0, 10.0)
            clipped_action = np.concatenate((clipped_action, clipped_exo_action))

        brain = osim.PrescribedController.safeDownCast(
            self.model.getControllerSet().get(0)
        )
        function_set = brain.get_ControlFunctions()
        for j in range(function_set.getSize()):
            func = osim.Constant.safeDownCast(function_set.get(j))
            func.setValue(float(action[j]))

        self.istep += 1
        self.state = self.manager.integrate(self.step_size * self.istep)

        self.model.realizeAcceleration(self.state)
