# Neuromechanical models of Human Locomotion

There have been decades of studies on locomotion control, specifically focusing on animal locomotion. The first research examined the impact of removing surgically the cerebrum and transecting the spinal cord. This allowed to study locomotion when the descending signals from the brain and from the spinal cord above the transection were removed {cite}`Flourens1824`, {cite}`Sherrington1910`. Sherrington observed that a series of locomotor-like movements of the hind limbs began when releasing one of the hind-limb from a flexed position. This led to an alternating motion from the contracting muscles which suggested that locomotion control is the result of a chain of spinal reflexes.
When examining spinally-transected cats, Sherrington's student, T.G. Brown questioned this idea of locomotion control based solely on reflexes. Indeed, he noticed that the contractions of the extensor and flexor muscles in the hind limbs took place even though the sensory nerves entering the spinal cord were transected {cite}`Brown1911`. Thus, he suggested the presence of an **intrinsic factor** in the spinal cord. This factor should be able to generate alternating locomotor-like motions without the descending signals from the brain, nor sensory inputs.
Since then, Grillner renamed this system: **Central Pattern Generation** {cite}`Grillner1975`, which was, later on, defined as Central Pattern Generators (CPGs). They have been shown to exist in a variety of vertebrates and invertebrates species.

## Feedback Models

Some research groups continue to investigate locomotion whose control doesn't include a CPG layer. The locomotion depends on reflexes which relies on feedback from the various muscles or groups of muscles, depending on the environment and the body. It is thus generated by **reflexive feedback control solely** and this type of control uses various sensors, as in the human body, where these sensors are present in the spinal cord and the supraspinal nervous system {cite}`Geyer2010`, {cite}`Song2015`.

### H. Geyer's Reflex Controller, 2010

In 2010, H. Geyer and his group developed a bipedal walking model that is controlled by muscle reflexes (i.e. interactions between the environment and the body) encoding the principles of legged mechanics, and which can produces a stable and robust walking gait, comparable to human walking {cite}`Geyer2010`.
This model is based on a conceptual bipedal spring-mass model, with several principles of legged mechanics added to it.

![alt text](../../main/md_files/images/Geyer.png "Evolution of the bipedal human model, from a conceptual one, to Geyer's 2010 model")

From the figure above:
A) The point-mass with the 2 mass-less springs representing the trunk and the segmented legs.
B) Added positive force feedback *F+* to the SOL and VAS muscles to generate compliant leg behavior.
C) Added positive force feedback *F+* to the GAS muscle to prevent knee overextension, and inhibition of the VAS muscle. Added TA muscle to avoid ankle overextension by positive length feedback *L+*.
D) The trunk is compelled to a lean angle (with respect to the vertical) by the hip muscles: GLU, HAM, HFL. HAM also counter knee hyperextension.
E) Addition of the swing, by increasing/decreasing the constant stimulation of HFL and GLU respectively. The VAS muscle is also inhibited proportionally to the load the other leg bears.
F) Eased leg swing by HFL using positive length feedback until it is suppressed by the negative length feedback *L-* from HAM. The positive length feedback *L+* from TA allows the flexion of the ankle. The positive force feedback from GLU and HAM enable the leg to retract and to then straighten at the end of the swing phase.

Even though this model is very simple, some characteristics of human walking are recreated, such as the center of mass (COM) dynamics, thanks to compliant leg behavior being added in stance (specifically through the extensor muscles present in the knee and ankle). To allow the control of the balance and cyclic motion, hip muscles and swing leg control were added.

![alt text](../../main/md_files/images/Geyer2.png "Representation of the 7 Hill-type muscles in Geyer's 2010 2D model")

From the figure above, we can therefore see that the model represents a human body with a trunk and two three-segmented legs comprising the hip, knee and ankle joints. Each sagittal leg is actuated by 7 Hill-type muscles, allowing an explicit comparison with the human muscles used during locomotion.

The 2D neuromuscular model developed by Geyer in 2010 is therefore a pure feedback model (i.e. this model doesn't include a feedforward component) which generates a stable, human-like walking gait. From this, a similar 3D neuromuscular model was developed by Song in 2015 {cite}`Song2015`.

### S. Song's Musculoskeletal 3D Model, 2015

This extension of Geyer's 2010 model represents a 3-dimensional musculoskeletal model with additional muscles, a supraspinal layer to adjust the foot placement, as well as 10 functional, spinal reflex modules that constitute the neural control circuitry.
Song's model represents a healthy human with a trunk characterizing the whole upper body, along with two legs (thigh, shank and foot). It consists of 7 segments actuated by 11 muscle tendon units (MTUs) per leg and connected by 8 internal DOFs. 2 DOFs are located at the hips (abduction and flexion, i.e. pitch and roll), 1 DOF at the knees and 1 at the ankles (flexion, i.e. pitch only).
As with Geyer's model, the MTUs actuating each leg are Hill-type muscles. The 9 muscles (BFSH, GAS, GLU, HAM, HFL, RF, SOL, TA, VAS) influence the joints of the hip, knee and ankle in the sagittal plane, and then 2 muscles were added to the hip (HAD, HAB) to actuate it in the lateral plane.
Furthermore, 4 compliant contact points make up each foot allowing continuous GRFs to interact with the model during locomotion, as seen in the figure below.

![alt text](../../main/md_files/images/song.png "3D Neuromuscular human model walking on uneven terrains (snapshots every 600ms")

In the figure above:
- A) The 9 muscles contributing to the 3 joints in the sagittal plane
- B) The 2 hip muscles that actuate the hip in the lateral plane
- C) The 4 contact points of each foot that allows the generation of GRFs
- D) The spinal cord's neural circuitry that is separated into 10 spinal reflex modules.
$t_l$, $t_m$ and $t_s$ correspond respectively to the long, medium and short transmission delays between the spinal cord and the joints.
$t_ss$ represents the transmission delay between the spinal cord and the supraspinal system.

Concerning the control of this musculoskeletal model, it is organized into spinal reflex modules and a supraspinal layer. The first consists of **10 reflex modules** for each leg ($M_1$ to $M_10$) that carry out individual limb functions, essential to legged systems, active during stance or swing phase. The **supraspinal layer**, for its part, allows to adjust the foot placement and the leg's target angles, to select the leg that switches to swing control during double support and to modulate continuously some of the reflex gains.

The different control modules generate the activation of the various modeled muscles that can be active concurrently, with modules $M_1$ to $M_5$ for stance control (compliant behavior and trunk balance), and modules $M_6$ to $M_{10}$ for swing control (ground clearance and leg placement).

Thanks to the 10 reflex modules and the supraspinal controller, Song's 3D model is able to display more locomotion gait and behaviors than Geyer's model, namely > walking and running, acceleration and deceleration, slope and stair negotiation, turning; and deliberate obstacle avoidance.

### Our Adapted Musculoskeletal Model (CESPAR)

For the CESPAR project, we adapt the model from Song {cite}`Song2015` so as to simplify their model and be able to perform different experiments.
One of the characteristics that is modified is that the model used and the control is done in 2-dimension, so the 3D HAB and HAD muscles are removed.

![alt text](../../main/md_files/images/musculo.png "Used musculoskeletal model. The muscles HAB and HAD, corresponding to the hip abductor and hip adductor, are removed from Song's 3D model.")


Thus, now 9 muscles are considered for each leg (GAS, BFSH, GLU, HAM, HFL, RF, SOL, VAS, TA). The model also consists of 7 segments (thigh, shank and foot), as well as the trunk representing the upper body. 3 internal DOFs are considered here: one for the hip corresponding to the flexion, one for the knee and one for the ankle. Therefore, 6 internal DOFs compose the musculoskeletal model used in this project. Furthermore, the 4 contact points that constitute each foot are kept in this model.


The used model can be characterized by a state vector of size **85** composed of:
- the muscle states (length, velocity, force: 2 x 3 x 9 = 54 parameters for both legs)
- the joint states and angles (8 x 2 = 16 parameters)
- the ground contact information (GRF: 3 x 2 = 6 parameters)
- the pelvis state (height, pitch, roll and 6 velocities, resulting in 9 parameters).
The outputs of the controller are described by a vector of size **18** representing the muscle activations of the 9 muscles of each leg, with values in the interval $[0,1]$, generated by the spinal reflex modules.

In short, the 3 segments of this model are connected with joints and the motion is controlled by the excitation of muscles.

### Summary

The muscles involved in each model are presented in the table below:

|       Muscles           |2D Geyer Model | 3D Song Model | Our CESPAR 2D Model |
|:-----------------------:|:-------------:|:-------------:|:-------------------:|
|BFSH                     |               | x             | x                   |
|GAS                      |x              | x             | x                   |
|GLU                      |x              | x             | x                   |
|HAM                      |x              | x             | x                   |
|HFL                      |x              | x             | x                   |
|RF                       |               | x             | x                   |
|SOL                      |x              | x             | x                   |
|TA                       |x              | x             | x                   |
|VAS                      |x              | x             | x                   |
|HAB                      |               | x             |                     |
|HAD                      |               | x             |                     |


