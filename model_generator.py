# -*- coding: utf-8 -*

"""model_generator.py: Source code of the automatization of the coupled human musculoskeletal system and exoskeleton with
bones size and physical properties (mass, mass center, inertias) and their exoskeleton equivalent parts, as well as
muscle attachment points, joints locations, contact surfaces placements and initial height position.
Example:
    The following command should be typed in the terminal to run model generator with the coupled musculoskeletal and exoskeleton system. ::
        $ python model_generator.py
"""

__author__ = "Raphael Gaiffe"
__copyright__ = "Copyright 2023, Alpine Intuition SARL"
__license__ = "Apache-2.0 license"
__version__ = "1.0.0"
__email__ = "berat.denizdurduran@alpineintuition.ch"
__status__ = "Stable"

import os
import sys
import math
import shutil
import numpy as np
from tkinter import *
#root contains everything in the GUI
window = Tk()
#names windows
window.title('Model Generator')
from sympy.abc import x, y, z
import xml.etree.ElementTree as ET
from sympy import solve_poly_system
from control.osim_HBP_withexo_CMAES import L2M2019Env


#row and columns indexes
n = 0
p = 0

nb_elements = 6
nb_coordinates = 3
mass_index = 5

new_filename = Label(window, text="New Filename :")
new_filename.grid(row=n, column=p)
new_filename_entry = Entry(window, justify='center')
new_filename_entry.insert(0, "gait14dof22musc_withexo_new")
new_filename_entry.grid(row=n, column=p+1, columnspan=2, sticky=EW)

n+=1
var_method = StringVar(None, "Factors")
buttons = []

method_label = Label(window, text="Scaling Method :")
method_label.grid(row=n, column=p)
scaling_choice = ["Size (cm)", "Factors"]

for idx, scale_method in enumerate(scaling_choice):
    buttons.append(Radiobutton(window, text=scale_method, variable=var_method, value=scale_method))
    buttons[-1].grid(row=n, column=p+1+idx)

n += 1
x_label = Label(window, text="X")
x_label.grid(row=n, column=p+1)
y_label = Label(window, text="Y")
y_label.grid(row=n, column=p+2)
z_label = Label(window, text="Z")
z_label.grid(row=n, column=p+3)

# Organizes window's content
#head_label = Label(root, text="Head")
#head_label.grid(row=n, column=p)

torso_label = Label(window, text="Torso")
torso_label.grid(row=n+1, column=p)
pelvis_label = Label(window, text="Pelvis")
pelvis_label.grid(row=n+2, column=p)
femur_label = Label(window, text="Femurs")
femur_label.grid(row=n+3, column=p)
tibia_label = Label(window, text="Tibias")
tibia_label.grid(row=n+4, column=p)
feet_label = Label(window, text="Feet")
feet_label.grid(row=n+5, column=p)
mass_label = Label(window, text="New Mass")
mass_label.grid(row=n+6, column=p)

n+=1
p+=1

#parts = [[parts.append(Entry(root)) for x in range(nb_coordinates)] for y in range(nb_elements)]
parts = []
for i in range(nb_elements):
    parts.append([])
    for j in range(nb_coordinates):
        if i == nb_elements-1 and j > 0:
            continue
        parts[i].append(Entry(window))
        parts[i][j].grid(row=n+i, column=p+j)

#makes a list of model's objects considered as cuboid for inertia calculations
cuboid = ["talus_r", "talus_l", "calcn_r", "calcn_l", "toes_r", "toes_l", "Efemur_l", "Efemur_r", "Efoot_l", "Efoot_r"]


""" Approximate Scale height of pelvis' placement for initial position
    Needs Fine tuning"""
def scale_height_position(root, dict_parts):

    for coordinate in root.iter('Coordinate'):

        if coordinate.get('name') == "pelvis_ty":
            current_height = float(coordinate.findtext("default_value"))
            new_height = 0
            #takes ratio length of femur to pelvis' height of model
            ratio_femur = 0.456
            #takes ratio length of tibia to pelvis' height of model
            ratio_tibia = 0.411

            if "femur_r" in dict_parts.keys():
                scale_factor = dict_parts["femur_r"][1]
                femur_height = ratio_femur*scale_factor*current_height

            else:
                femur_height = ratio_femur*current_height

            if "tibia_r" in dict_parts.keys():
                scale_factor = dict_parts["tibia_r"][1]
                tibia_height = ratio_tibia*scale_factor*current_height

            else:
                tibia_height = ratio_tibia*current_height

    for body_part in root.iter('Body'):
        if body_part.get('name') in dict_parts.keys() and body_part.get('name') == "femur_r":

            offset = 0.011
            new_height = (current_height*(1 - (ratio_femur + ratio_tibia)) ) + femur_height + tibia_height
            coordinate.find("default_value").text = str(new_height + offset)


""" Scale the x, y and z coordinates of a point with respect to the x, y, z scale_factors
    If coor is the output of a function, then multiply it by the single corresponding scale_factor"""
def scale_coordinates(coor, scale_factors):

    coor_y = np.array(coor.text.split())
    if isinstance(scale_factors, float):
        coor_y = [float(a)*scale_factors for a in coor_y]
    else:
        coor_y = [float(a)*b for a,b in zip(coor_y, scale_factors)]
    str_coor = str(coor_y)
    str_coor = str_coor.replace(',', '').replace('[', '').replace(']', '')
    coor.text = str_coor


""" Scale physical properties of a body (bones or exo parts) including inertia matrix, mass center and size of body"""
def scale_physical_properties(root, dict_parts):

    if parts[mass_index][0].get():
        new_body_mass = float(parts[mass_index][0].get())
        sum_scaled_masses = 0
    for body_part in root.iter('Body'):
        if body_part.get('name') in dict_parts.keys():
            scale_factors = dict_parts[body_part.get('name')]
            mass = body_part.find('mass').text
            old_mass = float(mass)
            if var_method.get() == "Factors":
                mass_scaled_factor = np.prod(scale_factors)
                new_mass = old_mass * mass_scaled_factor
                body_part.find('mass').text = str(new_mass)

            inertia_xx = float(body_part.find('inertia_xx').text)
            inertia_yy = float(body_part.find('inertia_yy').text)
            inertia_zz = float(body_part.find('inertia_zz').text)
            inertia_xy = float(body_part.find('inertia_xy').text)
            inertia_xz = float(body_part.find('inertia_xz').text)
            inertia_yz = float(body_part.find('inertia_yz').text)
            inertia_diag = [inertia_xx, inertia_yy, inertia_zz]

            if var_method.get() == "Factors" and all(ele == scale_factors[0] for ele in scale_factors):
                for i in range(3):
                    inertia_diag[i] *= mass_scaled_factor * scale_factors[0]**2

            #Case where the body is considered as a cuboid for inertia related calculations
            elif body_part.get('name') in cuboid:
                a = inertia_xx
                b = inertia_yy
                c = inertia_zz
                m = old_mass

                #the solution of the equations is the sizes' lengths of the cuboid
                solution = solve_poly_system([(1/12)*m*(y**2+z**2) - a, (1/12)*m*(x**2+z**2) - b, (1/12)*m*(x**2+y**2) - c], x, y, z)
                pos_sol = solution[0]

                if var_method.get() == "Size (cm)":
                    new_sizes = scale_factors
                    scale_factors = [new_size/old_size for new_size, old_size in zip(scale_factors, pos_sol)]
                    mass_scaled_factor = np.prod(scale_factors)
                    new_mass = old_mass * mass_scaled_factor
                    body_part.find('mass').text = str(new_mass) 

                else:
                    new_sizes = [a*b for a, b in zip(pos_sol, scale_factors)]

                inertia_diag[0] = 1/12 * mass_scaled_factor * ((new_sizes[1]**2 + new_sizes[2]**2))
                inertia_diag[1] = 1/12 * mass_scaled_factor * ((new_sizes[0]**2 + new_sizes[2]**2))
                inertia_diag[2] = 1/12 * mass_scaled_factor * ((new_sizes[0]**2 + new_sizes[1]**2))

            #Case where we consider the body as a cylinder
            else:
                pos_sol = [None] * 3

                min_inertia = min(inertia_diag)
                len_axis = inertia_diag.index(min(inertia_diag))

                #calculate approximate radius of bone using inertia formulas for solid cylinder
                square_radius = 2 * min_inertia / old_mass
                if square_radius < 0: radius = 0
                else: radius = math.sqrt(square_radius)

                #determine which axis is the one in the length of the cylinder and which axis correspond to the radius
                if len_axis == 0: rad_axis, rad_axis2 = 1, 2
                elif len_axis == 1: rad_axis, rad_axis2 = 0, 2
                else: rad_axis, rad_axis2 = 0, 1

                #calculate approximate length of bone using inertia formulas for solid cylinder
                square_length = 12.0 * (inertia_diag[rad_axis] - 0.25 * old_mass * radius * radius) / old_mass
                if square_length < 0: length = 0
                else: length = math.sqrt(square_length)

                pos_sol[rad_axis] = radius
                pos_sol[rad_axis2] = radius                
                pos_sol[len_axis] = length  

                if var_method.get() == "Size (cm)":
                    new_sizes = scale_factors
                    scale_factors = [new_size/old_size for new_size, old_size in zip(scale_factors, pos_sol)]
                    mass_scaled_factor = np.prod(scale_factors)
                    new_mass = old_mass * mass_scaled_factor
                    body_part.find('mass').text = str(new_mass) 
                
                else:
                    new_sizes = [a*b for a, b in zip(pos_sol, scale_factors)]                

                length *= scale_factors[len_axis]

                #change diagonal terms of inertia matrix
                square_radius = radius * (scale_factors[rad_axis]) * radius * (scale_factors[rad_axis2])
                inertia_diag[len_axis] = 0.5 * new_mass * square_radius
                inertia_diag[rad_axis] = new_mass * ((length * length / 12.0) + 0.25 * square_radius)
                inertia_diag[rad_axis2] = new_mass * ((length * length / 12.0) + 0.25 * square_radius)

            new_inertia_xy = inertia_xy * mass_scaled_factor * scale_factors[0] * scale_factors[1]
            new_inertia_xz = inertia_xz * mass_scaled_factor * scale_factors[0] * scale_factors[2]
            new_inertia_yz = inertia_yz * mass_scaled_factor * scale_factors[1] * scale_factors[2]

            body_part.find('inertia_xx').text = str(inertia_diag[0])
            body_part.find('inertia_yy').text = str(inertia_diag[1])
            body_part.find('inertia_zz').text = str(inertia_diag[2])
            body_part.find('inertia_xy').text = str(new_inertia_xy)
            body_part.find('inertia_xz').text = str(new_inertia_xz)
            body_part.find('inertia_yz').text = str(new_inertia_yz)
            body_part.set('updated', 'yes')

            scale_coordinates(body_part.find('mass_center'), scale_factors)
            scale_coordinates(body_part.find('VisibleObject').find('scale_factors'), scale_factors)

            #Change the joint frame location of the scaled body
            joint = body_part.find('Joint')
            if joint.iter('location'):
                for loc in joint.iter('location'):
                    scale_coordinates(loc, dict_parts[body_part.get('name')])

        if parts[mass_index][0].get() and body_part.get('name')[0] != 'E':
            sum_scaled_masses += float(body_part.findtext('mass'))


""" Scale joints frames positions"""
def scale_joints(root, dict_parts):

    #if the current body is a parent body, then changes the location in parent body of the child body
    for joint in root.iter('Joint'):
        for parent in joint.iter('parent_body'):
            if parent.text in dict_parts.keys() and parent.text != "torso":
                if joint.iter('location_in_parent'):
                    for loc_in_parent in joint.iter('location_in_parent'):
                        scale_coordinates(loc_in_parent, dict_parts[parent.text])
                #special case for femur as the knee joint translational position depends on its rotational position
                if parent.text == "femur_r" or parent.text == "femur_l":
                    for spatial_transform in joint.iter('SpatialTransform'):
                        for transform in spatial_transform.iter("TransformAxis"):
                            if transform.get('name') == "translation1":
                                for coor in transform.iter('y'):
                                    scale_coordinates(coor, dict_parts["femur_r"][0])
                            elif transform.get('name') == "translation2":
                                for coor in transform.iter('y'):
                                    scale_coordinates(coor, dict_parts["femur_r"][1])


"""For a given part, return the location of parent and child joint frames in part coordinates frame """
def get_location(root, part):

    #Change the joint frame location of the scaled body
    joint = part.find('Joint')
    if joint.iter('location'):
        for loc in joint.iter('location'):
            parent_loc = loc.text

    for joint in root.iter('Joint'):
        for parent_body in joint.iter('parent_body'):
            if parent_body.text == part.get('name'):
                for pinjoint in joint.iter('PinJoint'):
                    if pinjoint.get('name') == "exo_joint_hip_r":
                        for child_loc in joint.iter('location_in_parent'):
                            child_loc = child_loc.text

    return parent_loc, child_loc


"""Scale PointOnLineConstraint properties such as Line Direction, Point on Line, and Point on Follower Body
    Not useful in case of Autonmyo"""
def scale_constraints(root, dict_parts, offset):

    if dict_parts["pelvis"][2] != 1 or dict_parts["torso"][2] != 1:
        for constraint in root.iter('PointOnLineConstraint'):
            if constraint.findtext('line_body') in dict_parts.keys():

                line = np.array(constraint.findtext('line_direction_vec').split())
                line = [float(a)*b for a,b in zip(line, dict_parts[constraint.findtext('line_body')])]
                str_line = str(line)
                str_line = str_line.replace(',', '').replace('[', '').replace(']', '')
                constraint.find('line_direction_vec').text = str_line

                point = np.array(constraint.findtext('point_on_line').split())
                point = [float(a)*b for a,b in zip(point, dict_parts[constraint.findtext('line_body')])]
                str_point = str(point)
                str_point = str_point.replace(',', '').replace('[', '').replace(']', '')
                constraint.find('point_on_line').text = str_point

            if constraint.findtext('follower_body') == "Efemur_r" or constraint.findtext('follower_body') == "Efemur_l":
                follower = constraint.findtext('follower_body')
                point = np.array(constraint.findtext('point_on_follower').split())
                point = [float(a)*b if b==1 else float(a)*b + offset  for a,b in zip(point, dict_parts["Ehip_r"])]
                str_point = str(point)
                str_point = str_point.replace(',', '').replace('[', '').replace(']', '')
                constraint.find('point_on_follower').text = str_point

            if constraint.findtext('follower_body') in dict_parts.keys():
                point = np.array(constraint.findtext('point_on_follower').split())
                point = [float(a)*(b+offset) for a,b in zip(point, dict_parts[constraint.findtext('follower_body')])]
                str_point = str(point)
                str_point = str_point.replace(',', '').replace('[', '').replace(']', '')
                constraint.find('point_on_follower').text = str_point

            if constraint.get('name') == "toe_strap_constraint_r_pointonline" or constraint.get('name') == "toe_strap_constraint_l_pointonline":
                scale_coordinates(constraint.find('point_on_follower'), dict_parts["pelvis"])
                scale_coordinates(constraint.find('point_on_line'), dict_parts["pelvis"])
                scale_coordinates(constraint.find('line_direction_vec'), dict_parts["pelvis"])

"""Scale muscle attachments points coordinates according to scale factors
    TO DO/ADD: scale muscle optimal fiber length and tendon slack length"""
def scale_muscles(root, dict_parts):

    muscles = []

    for muscle in root.iter('Thelen2003Muscle'):
        for points in muscle.iter('PathPoint'):
            if points.find('body').text in dict_parts.keys():
                scale_coordinates(points.find('location'), dict_parts[points.find('body').text])
                muscle.set('updated', 'yes')
        for points in muscle.iter('ConditionalPathPoint'):
            if points.find('body').text in dict_parts.keys():
                scale_coordinates(points.find('location'), dict_parts[points.find('body').text])
                muscle.set('updated', 'yes')
        for points in muscle.iter('MovingPathPoint'):
            if points.find('body').text in dict_parts.keys():
                scale_coordinates(points.find('location'), dict_parts[points.find('body').text])
                for i, spline in enumerate(points.iter("SimmSpline")):
                    scale_coordinates(spline.find('y'), dict_parts[points.find('body').text][i])
                    muscle.set('updated', 'yes')


"""Scale contact surfaces, only their positions, not their sizes"""
def scale_contact_surfaces(root, dict_parts):

    for contact_sphere in root.iter('ContactSphere'):
        if contact_sphere.find('body_name').text in dict_parts.keys():
            scale_coordinates(contact_sphere.find('location'), dict_parts[contact_sphere.find('body_name').text])
            contact_sphere.set('updated', 'yes')

    for contact_mesh in root.iter('ContactMesh'):
        if contact_mesh.find('body_name').text in dict_parts.keys():
            scale_coordinates(contact_mesh.find('location'), dict_parts[contact_mesh.find('body_name').text])
            contact_mesh.set('updated', 'yes')


"""Scale constraints markers positions for pelvis, this is what allows the exo hips to be scaled respecting the constraints"""
def scale_constraints_markers(root, dict_parts, offset, factor_to_change):

    for marker in root.iter('Marker'):
        str_words = marker.get("name").split('_')
        if str_words[0] == "Ehip":
            if "pelvis" in dict_parts:
                for body_part in root.iter('Body'):
                    if body_part.get('name') == "femur_r":
                        for joint in body_part.iter('location_in_parent'):
                            femur_loc = joint.text
                            coor = np.array(femur_loc.split()).astype('float')
                            new_coor = np.array([a*b for a,b in zip(coor, dict_parts["pelvis"])]).astype('float')
                            diff_coor = abs(np.subtract(coor, new_coor))

                    if body_part.get('name') == "Ehip_r":
                        part_size = np.array(marker.findtext("location").split()).astype('float')
                        new_part_size = np.subtract(part_size, diff_coor)
                        scale_factors = [(a/b)+offset if c else a/b for a,b,c in zip(new_part_size, part_size, factor_to_change)]
                        dict_parts["Ehip_r"] = scale_factors
                        dict_parts["Ehip_l"] = scale_factors
                        dict_parts["Etrunk_m"] = scale_factors
                        str_coor = str(new_part_size)
                        str_coor = str_coor.replace(',', '').replace('[', '').replace(']', '')
                        marker.find("location").text = str_coor

"""Creates dictionnary with body and exo parts to change. The dictionnary contains the scaling factors for each part to change.
   The scaling factors for a given exoskeleton part are the same as the ones of its musculoskeletal equivalent part"""
def create_scaling_dicts(parts):

    temp_dict_parts = { 0 : 'Torso',
                        1 : 'Pelvis',
                        2 : 'Femur',
                        3 : 'Tibias',
                        4 : 'Feet'}

    dict_parts = {}
    
    for i, part in enumerate(parts):
        scale_elements = [True if scale_element.get() and float(scale_element.get())!=1 else False for scale_element in part]

        if any(scale_elements) and i != mass_index:
            scale_factors = [float(scale_element.get()) if scale_element.get() else 1.0 for scale_element in part]

            if temp_dict_parts[i] == 'Femur':
                femur = ["femur_r", "femur_l", "Efemur_r", "Efemur_l"]
                for part in femur:
                    dict_parts[part] = scale_factors

            elif temp_dict_parts[i] == 'Tibias':
                tibia = ["tibia_r", "tibia_l", "Eshin_r", "Eshin_l"]
                for part in tibia:
                    dict_parts[part] = scale_factors

            elif temp_dict_parts[i] == 'Feet':
                #Only the x value (length) is modified
                scale_factors = [scale_factors[0], 1.0, 1.0]
                exo_scale_factors = [scale_factors[0], 1.0, 1.0]

                feet = ["talus_r", "talus_l", "calcn_r", "calcn_l", "toes_r", "toes_l"]

                for part in feet:
                    dict_parts[part] = scale_factors

                dict_parts["Efoot_r"] = exo_scale_factors
                dict_parts["Efoot_l"] = exo_scale_factors             

            elif temp_dict_parts[i] == 'Pelvis':
                dict_parts["pelvis"] = scale_factors
                dict_parts["Ehip_r"] = scale_factors
                dict_parts["Ehip_l"] = scale_factors
                dict_parts["Etrunk_m"] = scale_factors

            elif temp_dict_parts[i] == "Torso":
                dict_parts["torso"] = scale_factors

            else:
                dict_parts["head"] = scale_factors

    return dict_parts


"""Changes the offset by which to increase or decrease the exo hip size so that it can fit the constraints"""
def change_offset( interval, nb_step, step, orig_min_inter, orig_max_inter,  dict_parts, best_offset, best_assembly_error, change_scale):

    if len(interval) > nb_step:
        
        interval = interval[:nb_step]
        
    elif len(interval) == 0:
        print(f"Best Assembly Error : {best_assembly_error} for Ehip Offset : {best_offset}")
        best_offset_ehip = best_offset
        scale_offset = change_scale

        dict_parts["pelvis"] = [1 if a == 1 else a+scale_offset for a in dict_parts["pelvis"]]
        dict_parts["Ehip_r"] = dict_parts["pelvis"]
        dict_parts["Ehip_l"] = dict_parts["pelvis"]

        if dict_parts["pelvis"][2] != 1:
            scale_factors = [1, 1, dict_parts["pelvis"][2]]
            dict_parts["Etrunk_m"] = scale_factors

        scale_offset = 0.001
        offset_ehip = best_offset
        min_inter = orig_min_inter
        max_inter = orig_max_inter
        #nb_step *= 2
        step = (max_inter-min_inter)/(nb_step)
        interval = np.arange(min_inter, max_inter, step)
        assembly_errors = np.zeros(nb_step)
        print("Changing Interval")

    return interval, step, dict_parts, offset_ehip, 
        

"""Change the interval of scaling factors of the exo hips to find the exo hips dimensions that fit the constraints"""
def change_interval(i, interval, assembly_errors, nb_step, step):

    if i == len(interval):            
        i = 0
        min_index = np.argmin(assembly_errors)
        new_step = 2*step/(nb_step)

        if min_index == 0:
            min_inter = interval[min_index] - step
            max_inter = interval[min_index+1]+new_step

        elif min_index == len(interval)-1:
            min_inter = interval[min_index-1]
            max_inter = interval[min_index] + (step +new_step)

        else:
            min_inter = interval[min_index-1]
            max_inter = interval[min_index+1]+new_step

        step = new_step
        interval = np.arange(min_inter, max_inter, step)

    return i, interval, step


"""Function to execute when the GUI window's "Create Model" button is pressed"""
def click_render():

    assembly_error = 1
    best_assembly_error = 1
    seed = 64
    SEED = int(seed)   # Random seed
    difficulty = 0

    orig_min_inter = -0.1
    orig_max_inter = 0.15
    nb_step = 5
    min_inter = orig_min_inter
    max_inter = orig_max_inter
    step = (max_inter-min_inter)/(nb_step)
    change_scale = -0.005
    interval = np.arange(min_inter, max_inter, step)
    i = 0
    j = 0
    assembly_errors = np.zeros(nb_step)
    offset_ehip = 0 
    ehip = True

    dict_parts = create_scaling_dicts(parts)
    #new_filename_entry.get()
    new_filepath = f'./models/gait14dof22musc_withexo_full_new.osim'
    factor_to_change = None

    original_pelvis_factors = None
    #if the pelvis is scaled, the program will try the different scaling factors that respect the constraints one by one
    if "pelvis" in dict_parts:
        original_pelvis_factors = dict_parts["pelvis"]
        dict_parts["pelvis"] = [1., 1., 1.]
        indices = [i for i, x in enumerate(original_pelvis_factors) if x != 1]
        num_of_index = 0
        dict_parts["pelvis"][indices[num_of_index]] = original_pelvis_factors[indices[num_of_index]]
        factor_to_change = [1 if i == indices[num_of_index] else 0 for i in range(len(dict_parts["pelvis"]))]
        first_time = True
        #num_of_index += 1
                
    while assembly_error > 1e-10 and j < 1000000:
        # Makes copy of .osim model and change it to a .xml file
        shutil.copyfile('./models/gait14dof22musc_withexo_full.osim', new_filepath)
        pre, ext = os.path.splitext(new_filepath)
        temp_xml = pre + '.xml'
        os.rename(new_filepath, temp_xml)

        # :: were causing errors when reading .xml files, so here they are replaced
        with open(temp_xml, 'r') as file:
            data = file.read()
            data = data.replace("::", "xxxx")

        with open(temp_xml, 'w') as file:
            file.write(data)

        # Reads .xml file
        tree = ET.parse(temp_xml)
        root = tree.getroot()

        if j == 0:
            orig_root = tree.getroot()
        
        if ehip:
            offset_ehip = interval[i]            
        #dict_parts = create_scaling_dicts(parts)
        if factor_to_change:
            if first_time:
                offset_ehip = 0
                first_time = False
        else:
            offset_ehip = interval[i]

        scale_constraints_markers(root, dict_parts, offset_ehip, factor_to_change)
        #print(dict_parts, "after scale_constraints_markers")
        scale_physical_properties(root, dict_parts)       
        scale_joints(root, dict_parts)
        scale_muscles(root, dict_parts)
        scale_contact_surfaces(root, dict_parts)
        scale_height_position(root, dict_parts)

        """ if not ehip:
            scale_constraints(root, dict_parts, offset_constraint) """

        tree.write(temp_xml)

        # Changes back to :: and add the original first line of .osim files
        with open(temp_xml, 'r') as file:
            data = file.read()
            data = data.replace("xxxx", "::")

        with open(temp_xml, 'w') as file:
            file.write('<?xml version="1.0" encoding="UTF-8" ?>\n' + data)

        # Changes file extension from .sml to .osim
        pre, ext = os.path.splitext(temp_xml)
        os.rename(temp_xml, pre + '.osim')

        #Takes what's printed in terminal and make a list of strings 
        output_string = ""
        stream = sys.stdout
        original_stdout = sys.stdout.fileno()
        original_stdout2 = os.dup(original_stdout)
        read_pipe, write_pipe = os.pipe()
        os.dup2(write_pipe, original_stdout)
        L2M2019Env(visualize=False, seed=SEED, difficulty=difficulty, desired_speed=1.6)
        stream.write("\b")
        stream.flush()

        while True:
            char = os.read(read_pipe,1).decode('utf8')
            if not char or "\b" in char:
                break
            output_string += char

        output_list = output_string.split(' ')
        os.close(write_pipe)
        os.close(read_pipe)
        os.dup2(original_stdout2, original_stdout)
        os.close(original_stdout2)

        #Check if constraints are respected
        if i == 0:
            list_len = len(output_list)

            if 'achieved:' in output_list:
                achieved_index = output_list.index('achieved:')

            elif not original_pelvis_factors:
                print("Assembly Tolerance Achieved !")
                assembly_error = 1e-10
                best_assembly_error = 1e-10
                break

            #if pelvis is scaled and assembly error achieved
            else:                         
                #if all pelvis scaling factors have been applied
                if num_of_index == len(indices)-1:
                    print("Assembly Tolerance Achieved indices reached!")
                    assembly_error = 1e-10
                    best_assembly_error = 1e-10
                    break

                else:
                    #go on the next pelvis scaling factor to change
                    num_of_index += 1
                    dict_parts["pelvis"][indices[num_of_index]] = original_pelvis_factors[indices[num_of_index]]
                    factor_to_change = [1 if i == indices[num_of_index] else 0 for i in range(len(dict_parts["pelvis"]))]
                    first_time = True

                continue

            assembly_error_index = achieved_index + 1

        #if message of assembly error is different than the first one
        #if len(output_list) != list_len:
        if 'achieved:' not in output_list:
            print("Assembly Tolerance Achieved !")
            assembly_error = 1e-10
            best_assembly_error = 1e-10

        else:
            assembly_error = float(output_list[assembly_error_index])

        if assembly_error < best_assembly_error:
            best_assembly_error = assembly_error
            best_offset = interval[i]
            print(f"Best Assembly Error : {best_assembly_error} for Offset : {best_offset}")

        assembly_errors[i] = assembly_error
        i += 1
        j += 1

        if i == len(interval):
            i, interval, step = change_interval(i, interval, assembly_errors, nb_step, step)

            if len(interval) > nb_step:             
                interval = interval[:nb_step]
                
            elif len(interval) == 0:
                print(f"Best Assembly Error : {best_assembly_error} for Ehip Offset : {best_offset}")
                best_offset_ehip = best_offset
                scale_offset = change_scale

                #add offset to scale factors
                if "pelvis" in dict_parts:
                    dict_parts["pelvis"] = [factor+scale_offset if i == indices[num_of_index] else factor for i, factor in enumerate(dict_parts["pelvis"])]
                    dict_parts["Ehip_r"] = dict_parts["pelvis"]
                    dict_parts["Ehip_l"] = dict_parts["pelvis"]

                scale_offset = 0.001
                offset_ehip = best_offset
                min_inter = orig_min_inter
                max_inter = orig_max_inter
                step = (max_inter-min_inter)/(nb_step)
                interval = np.arange(min_inter, max_inter, step)
                assembly_errors = np.zeros(nb_step)
                print("Changing Interval")

    print(f"Best Assembly Error : {best_assembly_error}")
    os.rename(new_filepath, f"./models/{new_filename_entry.get()}.osim")
    window.destroy()
    
    
myButton = Button(window, text="Create Model", padx=25, command=click_render)
myButton.grid(row=n+11, column=2)

window.mainloop()
