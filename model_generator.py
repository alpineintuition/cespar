# -*- coding: utf-8 -*

"""model_generator.py: Source code of the automatization of the coupled human musculoskeletal system and exoskeleton with
bones size and physical properties (mass, mass center, inertias) and their exoskeleton equivalent parts, as well as
muscle attachment points, joints locations, contact surfaces placements and initial height position.
Example:
    The following command should be typed in the terminal to run model generator with the coupled musculoskeletal and exoskeleton system. ::
        $ python model_generator.py
"""

__author__ = "Raphael Gaiffe"
__copyright__ = "Copyright 2022, Alpine Intuition SARL"
__license__ = "GPL-3.0 license"
__version__ = "1.0.0"
__email__ = "berat.denizdurduran@alpineintuition.ch"
__status__ = "Stable"

from re import M
from tkinter import *
import numpy as np
import math
#root contains everything in the GUI
root = Tk()
#names windows
root.title('Model Generator')

import os
import shutil

import xml.etree.ElementTree as ET

from sympy import solve_poly_system
from sympy.abc import x, y, z, d, h, w, r

n = 0
p = 0

# Organizes window's content
head_label = Label(root, text="Head")
head_label.grid(row=n, column=0)
torso_label = Label(root, text="Torso")
torso_label.grid(row=n+1, column=0)
pelvis_label = Label(root, text="Pelvis")
pelvis_label.grid(row=n+2, column=0)
femur_label = Label(root, text="Femurs")
femur_label.grid(row=n+3, column=0)
tibia_label = Label(root, text="Tibias")
tibia_label.grid(row=n+4, column=0)
feet_label = Label(root, text="Feet")
feet_label.grid(row=n+5, column=0)
mass_label = Label(root, text="New Mass")
mass_label.grid(row=n+6, column=0)

parts = []
for i in range(7):
    parts.append(Entry(root))
    parts[i].grid(row=n+i, column=1)

#makes a list of model's objects considered as cuboid for inertia calculations
cuboid = ["talus_r", "talus_l", "calcn_r", "calcn_l", "toes_r", "toes_l", "Efemur_l", "Efemur_r", "Efoot_l", "Efoot_r"]

""" Approximate Scale height of pelvis' placement for initial position
    Needs Fine tuning"""
def scale_height_position(root, dict_parts):

    for coordinate in root.iter('Coordinate'):
        current_height = float(coordinate.findtext("default_value"))
        new_height = 0
        if coordinate.get('name') == "pelvis_ty":
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

    for body_part in root.iter('Body'):
        if body_part.get('name') in dict_parts.keys():
            scale_factors = dict_parts[body_part.get('name')]
            mass = body_part.find('mass').text
            old_mass = float(mass)
            mass_scaled_factor = np.prod(scale_factors)
            new_mass = old_mass * np.prod(scale_factors)
            body_part.find('mass').text = str(new_mass)
            inertia_xx = float(body_part.find('inertia_xx').text)
            inertia_yy = float(body_part.find('inertia_yy').text)
            inertia_zz = float(body_part.find('inertia_zz').text)
            inertia_xy = float(body_part.find('inertia_xy').text)
            inertia_xz = float(body_part.find('inertia_xz').text)
            inertia_yz = float(body_part.find('inertia_yz').text)

            inertia_diag = [inertia_xx, inertia_yy, inertia_zz]
            if all(ele == scale_factors[0] for ele in scale_factors):
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
                new_sizes = [a*b for a, b in zip(pos_sol, scale_factors)]
                inertia_diag[0] = 1/12 * mass_scaled_factor * ((new_sizes[1]**2 + new_sizes[2]**2))
                inertia_diag[1] = 1/12 * mass_scaled_factor * ((new_sizes[0]**2 + new_sizes[2]**2))
                inertia_diag[2] = 1/12 * mass_scaled_factor * ((new_sizes[0]**2 + new_sizes[1]**2))

            #Case where we consider the body as a cylinder
            else:
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

""" Scale joints frames positions"""
def scale_joints(root, dict_parts):

    #if the current body is a parent body, then changes the location in parent body of the child body
    for joint in root.iter('Joint'):
        for parent in joint.iter('parent_body'):
            if parent.text in dict_parts.keys():
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

"""Scale PointOnLineConstraint properties such as Line Direction, Point on Line, and Point on Follower Body
    Not useful in case of Autonmyo"""
def scale_constraints(root, dict_parts):
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

        if constraint.findtext('follower_body') in dict_parts.keys():
            point = np.array(constraint.findtext('point_on_follower').split())
            point = [float(a)*b for a,b in zip(point, dict_parts[constraint.findtext('follower_body')])]
            str_point = str(point)
            str_point = str_point.replace(',', '').replace('[', '').replace(']', '')
            constraint.find('point_on_follower').text = str_point

"""Scale muscle attachments points coordinates according to scale factors
    TO DO/ADD: scale muscle optimal fiber length and tendon slack length"""
def scale_muscles(root, dict_parts):

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


"""Creates dictionnary with body and exo parts to change. The dictionnary contains the scaling factors for each part to change.
   The scaling factors for a given exoskeleton part are the same as the ones of its musculoskeletal equivalent part"""
def create_scaling_dicts(parts):

    temp_dict_parts = { 0 : 'Head',
                        1 : 'Torso',
                        2 : 'Pelvis',
                        3 : 'Femur',
                        4 : 'Tibias',
                        5 : 'Feet'}

    dict_parts = {}

    for i, part in enumerate(parts):
        if part.get():
            str_scale_factors = part.get()
            arr_part = np.array(str_scale_factors.split())
            scale_factors = [float(factor) for factor in arr_part]
            if temp_dict_parts[i] == 'Femur':

                femur = ["femur_r", "femur_l", "Efemur_r", "Efemur_l"]
                for part in femur:
                    dict_parts[part] = scale_factors

            elif temp_dict_parts[i] == 'Tibias':

                tibia = ["tibia_r", "tibia_l", "Eshin_r", "Eshin_l"]
                for part in tibia:
                    dict_parts[part] = scale_factors

            elif temp_dict_parts[i] == 'Feet':

                #for the feet, only the x value (length) is modified
                scale_factors = [scale_factors[0], 1, 1]
                exo_scale_factors = [scale_factors[0], 1, 1]

                feet = ["talus_r", "talus_l", "calcn_r", "calcn_l", "toes_r", "toes_l"]
                for part in feet:
                    dict_parts[part] = scale_factors

                dict_parts["Efoot_r"] = exo_scale_factors
                dict_parts["Efoot_l"] = exo_scale_factors

            #for the moment, the pelvis and the torso are not scaled
            elif temp_dict_parts[i] == 'Pelvis':

                scale_factors = [1, 1, 1]
                dict_parts["pelvis"] = scale_factors
                dict_parts["Ehip_r"] = scale_factors
                dict_parts["Ehip_l"] = scale_factors
                if scale_factors[2] != 1:
                    dict_parts["Etrunk_m"] = [1, 1, scale_factors[2]]

            elif temp_dict_parts[i] == "Torso":

                scale_factors = [1, 1, 1]
                dict_parts["torso"] = scale_factors

            else:

                dict_parts["head"] = scale_factors

    return dict_parts

"""Function to execute when the GUI window's "Create Model" button is pressed"""
def click_render():

    # Makes copy of .osim model and change it to a .xml file
    new_filepath = f'./models/gait14dof22musc_withexo_new.osim'
    shutil.copyfile('./models/gait14dof22musc_withexo.osim', new_filepath)
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

    dict_parts = create_scaling_dicts(parts)
    scale_physical_properties(root, dict_parts)
    scale_joints(root, dict_parts)
    scale_muscles(root, dict_parts)
    scale_contact_surfaces(root, dict_parts)
    scale_height_position(root, dict_parts)
    #scale_constraints(root, dict_parts)

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

myButton = Button(root, text="Create Model", padx=50, command=click_render)
myButton.grid(row=n+11, column=1)

root.mainloop()
