## Model Generator

When the program is launched, a GUI open where you can enter the scaling factors of the skeleton bones.
When you click on “Create Model” the program scales the bones size and physical properties (mass, mass center, inertias)
and their exoskeleton equivalent parts, as well as muscle attachment points, joints locations, contact surfaces placements and initial height position.

### Execute Program
1. Open a terminal
2. Activate the virtual environment `opensim-rl`.
3. Install the sympy library using the command `conda install sympy`
4. Go to the directory that contains the git repository.
5. Run the command `python model_generator.py`
6. A window should open with different fields named with bones.
7. Enter the scale factors as follow `x_factor y_factor _z_factor` for the bones you want to scale

Notes :
- There is a function to scale PointOnLineConstraints, however it is not useful for the moment in our current
configuration but this could change with other exos.
- The y coordinate (y_factor) corresponds to the length of the bone in the case of the femurs and the tibias
- The x coordinate (x_factor) corresponds to the length of the bones in the case of the feet
- When scaling the feet, only the x coordinate (x_factor) is taken into account for both the skeleton and the exoskeleton
- For the moment, scaling the head does not change anything in the new created model
- The new initial position of the pelvis' height needs to be fine tuned i.e. it is normal if the new model is a bit
 too high above or too low below the ground
