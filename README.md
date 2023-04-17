<!-- =================================================
# Copyright (c) Alpine Intuition SARL.
Authors  :: Berat Denizdurduran (berat.denizdurduran@alpineintuition.ch)

================================================= -->

# Closed-loop exoskeleton simulation for personalized assistive rehabilitation (CESPAR)

![alt text](../main/md_files/autonomyo-and-alpine-intuition.jpg)

CESPAR is a project of coupled musculoskeletal systems and
robotic assistive device environments in order to study simulated neurorohabilitation of patients with lower limp motor control disorders.

## Table of Contents
1. [Installation](../main/md_files/Installation.md)
2. [How to Launch and Test an Optimization](../main/md_files/howToLaunch.md)
3. [Explanation of the Files](../main/md_files/files.md)
4. [Neuromechanical models of Human Locomotion](../main/md_files/neuromechanics.md)
5. [Explanation of Opensim Integration](../main/md_files/opensim.md)
6. [Model Generator](../main/md_files/model_generator.md)
7. [Scaler](../main/md_files/scaler.md)
8. [NRP Integration](../main/md_files/nrp.md)

### General Information

This repository can be used to perform experiments with a controller with population-based optimization using the deap framework. It contains a neuromuscular model of human walking with an Autonomyo exoskeleton attached to the musculoskeletal system. The controller is based on the python reflex controller from Seungmoon Song [1](https://ieeexplore.ieee.org/document/5445011).

In neuroscience, there is a growing need to understand biped locomotion and the neural circuits
involved in motion. Indeed, this field of study could help in better comprehending neuro-motor
pathologies and hence develop rehabilitation strategies more suited to a specific pathology. Also,
a locomotion control system could be applied to orthotics which could allow patients with a variety of neuromuscular conditions to walk in a more natural and simpler way, rather than what
is currently done in passive orthotics (this could be done in a similar manner for prosthetics).

Some interesting applications for sports could be developed as well.
Numerous state-of-the-art bio-inspired controllers have been developed over the years, based
on neural control through virtual muscles driven by reflexes and central pattern generators
(CPGs), which necessitate no dynamics modeling or inverse kinematics but still produce body
dynamics in striking similarity with healthy human walking. This allows for the merging of
modern control theory with biological motor control. However, even though this form of control
is a promising technology, it is yet far from its full potential as a result of the complexity of biped
locomotion (stability constraints, energy efficiency as well as biped dynamics due to multiple
degrees of freedom) and the lack of a clear understanding of how spinal networks are modulated
to adapt to different environmental or internal conditions.

In this pipeline, the various parameters of a reflex model ([1](https://ieeexplore.ieee.org/document/5445011), *A Muscle-Reflex Model That Encodes Principles of Legged Mechanics Produces Human Walking Dynamics and Muscle Activities*) can be optimized alongside an assistive-robotic device to
study neurorehabilitation of patients with lower limb motor control disorders.

![Alt Text](../main/md_files/output.gif)


### License

CESPAR is licensed under the [Apache License v2.0](LICENSE).


### Acknowledgements

The CESPAR Project is supported by European Union’s Horizon 2020 Framework Programme for Research and Innovation under Specific Grant Agreement No: 945539 (Human Brain Project SGA-3)
