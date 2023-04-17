## Scaler

This script computes the scaling factors to be applied to each of the following five body parts -- torso, pelvis, femurs, tibias, and feet -- of our reference OpenSim model in the x, y, and z directions in order to match them to the subject's anatomy. It can be that the subject markers are not placed in exactly the same anatomical positions as the OpenSim markers used for reference. For that, two options are possible:
1. Altering the positions of the OpenSim markers in `markers/MarkerSet.xml` to adjust their positions to more comparable ones to those of the subject. This must account for the height and lengths of the limbs in the reference (unscaled) OpenSim model.
2. Or, ensuring to place the markers on the subject such that they are as close as possible, anatomically, to their OpenSim counterparts and discarding measurements that aren't.

### How to use the script:

This process requires manual work to match the subject and OpenSim markers and may need to be repeated a few times to get the most accurate results. However, once done, the scales can be directly copied into the GUI of the `model_generator.py` script. The choice of matched OpenSim and subject markers used for scaling the five bodies must be defined in the dictionaries in the script `marker_matching.py`. The `Scaler` class requires as input the path to the OpenSim marker file of the reference (unscaled) OpenSim model (`markers/MarkerSet.xml`). It also requires the subject marker dictionary given in the following form:

    subject_data_dict = {

        'body_part1': [[x0, y0, z0], [x1, y1, z1], [x2, y2, z2], ...]

        'body_part2': ...

          ...
          
    }

In the dictionary above, the indices of the x, y, and z coordinates indicate the frame or sample number of the motion capture device. Note that the value of each key in the dictionary is expected to be a list of lists. 

By default, the scaler uses the first frame (frame 0) to generate the model scales; however, the desired frame number can be passed as a class argument. All marker positions are in meters measured from reference body parts such as torso, pelvis, femur, tibia, etc as demonstrated in the `markers/MarkerSet.xml` file.

Finally, it is important to make sure that the coordinate frame in which the subject data was collected matches that of OpenSim. For reference, in OpenSim, y is the vertical axis, x is perpendicular to the direction of walking, and positive z is the walking direction. To map the coordinates appropriately, pass the experiment coordinates that translate to the x, y, z coordinates of OpenSim in the class argument `subject_marker_coords`. By default, the class assumes that the data entered is in the same coordinate frame as OpenSim.