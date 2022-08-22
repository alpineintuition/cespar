# OPENSIM

**This section covers the detail of the OPENSIM implementation. I recommend using Opensim 3.3's synthax when you need to read and or change .osim files as I found it to be the easiest way to apprehend and the rest of the guide uses the 3.3's notation.
Notepad++ is a great software for reading .osim files. Whenever there is a reference to <Section_name>, look into the .osim file**
## Adding Exoskeleton model to the musculoskeletal model
1. Open the .osim file of the musculoskeleton on which you want to add your exoskeleton
2. Go right before the end of the <objects> section in the <Bodyset> section

![alt text](../../main/md_files/images/insert_exo.png)

3. Insert the exoskeleton parts like in the following screenshots. Make sure to respect the indentation, the < Body > and </ Body > have to be aligned
and so does every section with the synthax <Section_name> </Section_name>, with <Section_name> being the beginning of the section and </Section_name>
the end of the section

![alt text](../../main/md_files/images/osim_file1.png)

![alt text](../../main/md_files/images/osim_file2.png)

4. Link the exoskeleton parts together

	In the <parent_body> subsection, you need to define the part on which the exoskeleton part is going to be attached to. I recommend having
only one exoskeleton part that is going to be fixed (can't move in any direction) with respect to the musculoskeletal model, so that it serves as an anchor, and then connect the rest of the exo parts together. For example, in the case with a lower limb exoskeleton, the exo torso is first fused to the musculoskeletal pelvis.

	The exo's parts are attached as follow (⟶ means is attached to) exo's feet ⟶ exo's shins ⟶ exo's femur ⟶ exo's hips ⟶ exo's torso. It is better to attach the exo parts together rather than with their musculoskeletal counterparts to avoid misalignment between the exo parts.
	To connect the exo parts between them, you need to create a Joint among the following (taken from [OpenSim website ](https://simtk-confluence.stanford.edu:8443/display/OpenSim33/OpenSim+Models#OpenSimModels-Joints)):

- WeldJoint: introduces no coordinates (degrees of freedom) and fuses bodies together
- PinJoint: one coordinate around the common Z-axis of parent and child joint frames
- SliderJoint: one coordinate along the common X-axis of parent and child joint frames
- BallJoint: three rotational coordinates around X, Y and Z axis of child in parent
- EllipsoidJoint: three rotational coordinates around X, Y and Z axis child in parent with coupled translations such that the child body traces an ellipsoid centered at parent body
- FreeJoint: six coordinates with 3 rotational (like the ball) and 3 translations of child body  in parent body
- CustomJoint: user specifies 1-6 coordinates and user defines spatial transform to locate child body with respect to parent body

	Take the Pinjoint of the EFoot_l from the picture above. In the Pinjoint Section, the parent body, which designates the exo
part to which the EFoot_l has to be attached to, is the Eshin. Then, the position of the joint has to be defined in both the parent body frame and the child body frame (the child body is the current exo part you're working on) in respectively the <location_in_parent> <orientation_in_parent> and the <location> <orientation>.

	Once you've added the body part and created the joint, in the Opensim window, in the navigator tab on the left, you can edit the parent and child frames to align the parent and child body 	in the desired way. You can show the reference frames by right clicking on the frame in the navigator and enable "Show Axis" (see picture below).

![alt text](../../main/md_files/images/joint_placement.PNG)

## Actuators
In the <ForceSet> section, you can add the actuators available in your exoskeleton. Again, you need to be careful with the indentation. The figure below shows the placement of the actuators in the .osim files. The actuators correspond to the exo joints with motors.

![alt text](../../main/md_files/images/actuator.PNG)

## Scaling Tool
To adapt the exoskeleton of your choice to the musculoskeletal model, you can use the Scaling Tool. As its name indicates, it scales the target exoskeleton parts to the desired length along
 a certain axis using scaling factors.

![alt text](../../main/md_files/images/setting_scale.png)

![alt text](../../main/md_files/images/scale_factors.png)

 **One has to be careful with the Scale Tool included in Opensim. The documentation provided by their website can be misleading. First of all, for convenience, the Scale Tool was used on Opensim 3.3 to keep the 3.3 syntax of the .osim file. The main advantage of this syntax is that instead of having one set for the joints and one set for the bodies, you just have the bodyset and within it, you have each joint placed in the section of the body part on which the joint depends, which makes it easier to find information.**

In Opensim 3.3 whenever a scaling factor is applied to a body part, the joint frame locations, the mass center locations, force application points as well as the muscle attachment points are scaled according to the scaling factor. However, when it comes to the mass and the inertia, they are trickier to adapt to the scaling factor.

For the mass, in Opensim 3.3, you have to enter the exact same mass in the ’Mass’ field of post-scaling model then what is written in current model ’Mass’ field. Indeed, the scaling algorithm will multiply each body part mass by the product of the part's scaled factors, then multiply it by the ratio of the new target mass to original mass. So it doesn’t matter that your scaled model is bigger or smaller, i.e. that the mass value should be higher or lower, you should enter the same mass value as in the original one. That way, the scaled parts will have their mass scaled with respect to their scaling factor so the total mass of the new model will be different than the one you entered but it will be the correct one. Though not logical, it is convenient as it avoids having to calculate yourself the new mass of the scaled model.

Now regarding the inertias, the Opensim algorithm doesn’t take into account the scaled factors, only the total scaled mass to total original mass ratio. Therefore, as the mass of the scaled model, the mass value required is the original mass times the scaling factor of a body part. As a result, to get the correct inertia tensor for each scaled body parts, the scaling process has to be repeated as many times as there are different scaled body parts.

This is useful for aligning the rotation axis of the exoskeleton joints with their musculoskeletal counterparts or to adapt the width of the exoskeleton to the width of the musculoskeletal model. As can be seen below, the skeleton's feet are not in the exoskeleton's feet.

![alt text](../../main/md_files/images/intermediate_model.png)

 ## Point on Line Constraints
In the <ConstraintSet> section, you can add the actuators available in your exoskeleton. Again you need to be careful with the indentation. The figure below shows the placement
of the constraints in the .osim files.

![alt text](../../main/md_files/images/constraints.PNG)

First, let's introduce some of the standards:

- ’A’ is for body parts of the musculoskeleton
- ’B’ is for body parts of the exoskeleton

A line axis, let’s call it AxisA is fixed on Body A. The line (in
cyan in the picture below) goes through the anchor point A on Body A and the direction of the line is
defined by the user. Anchor Point B, highlighted by red arrows in pictures, (in our case it is the
same as anchor point A but expressed in the reference frame of Body B) on Body B is constrained
to stay on AxisA. Anchor Point B can move along the line of Body A, so its
position in Body A can change (not always) but not its position in body B. Therefore, whenever
Body A is moving, Anchor Point B has to follow it along the LocalAxisA and since it has to
remain at the exact same location in Body B, Body B will move to follow Anchor Point B. The
cyan lines go through the 2 centers of joints of a body part for the femur and tibia. For the foot,
the line is a bit below the ankle joint center and is parallel to the surface of the exofoot, so that
if the toes move, they slide on the exofoot. In dark blue are the rotation axes of the exoskeleton
and skeleton parts.

![alt text](../../main/md_files/images/femur_constraint.png)


## Contact Points
In the < ContactGeometrySet > section, you can add the contacts shapes you want. Again you need to be careful with the indentation. The figure below shows the placement of the constraints in the .osim files.

![alt text](../../main/md_files/images/contact.PNG)

You may have to change the Contact Points, for example with a lower limb exoskeleton where the musculoskeletal feet are on top of the exoskeleton feet. First, change the position of the pelvis relative to the ground along the z axis so that the entire musculoskeleton moves with it. The shift should be equal to the height of sole of the exo feet. The contact points' location of the feet can be lowered by the thickness of the exoskeleton foot sole and be placed in the exo feet reference frame. To do so, go in the Navigator in Opensim, in Contact geometry and select the contact points you want to lower as in the figure below. In the Physical Frame, you can decide to change the body part linked to the contact sphere to the exo foot. Then you can change its position in Location.

![alt text](../../main/md_files/images/change_contact2.png)
