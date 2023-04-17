# -*- coding: utf-8 -*

"""scaler.py: outputs to a csv file of the scaling factors in the x, y, z directions of the five body parts in our OpenSim model: torso, pelvis, tibias, femus and feet. 
              INPUT_ARGS:
              -----------
              data_dict:                    [dict] dictionary containing the marker measurements of the subject in meters relative to each of the 5 bodies
              osim_marker_xml_filename:     [ str] path to the marker file of the reference (unscaled) OpenSim model to be scaled
              frame_number:                 [ int] index of the frame or sample number of the subject marker position measurements to be used for scaling
              subject_marker_coords:        [list] a list that maps the coordinate frame in which the subject marker data was collected to that of OpenSim
"""

__author__ = "Aliaa Diab"
__copyright__ = "Copyright 2023, Alpine Intuition SARL"
__license__ = "Apache-2.0 license"
__version__ = "1.0.0"
__email__ = "berat.denizdurduran@alpineintuition.ch"
__status__ = "Stable"

import pickle
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from marker_matching import OSIM_MARKERS, SUBJECT_MARKERS

SAVE_MARKER_DATA_CSV = False

'''
Class that compares the positions of key markers on the subject's body to those in the reference OpenSim model found
in models/gait14dof22musc_withexo_full.osim whose marker positions are defined in markers/MarkerSet.xml file. 
To pair subject and osim markers, adjust the respective dictionaries found on marker_matching.py.
'''

class Scaler:
    def __init__(self, data_dict: dict, 
                        osim_marker_xml_filename: str, 
                        frame_number: int = 0,
                        subject_marker_coords: list = ['x', 'y', 'z']) -> None:
        self.subject_marker_coords = subject_marker_coords
        self.osim_data = self.parse_osim_marker_file(osim_marker_xml_filename)
        self.subject_data = self.parse_subject_marker_data(data_dict, frame_number)
        self.print_info()

    def parse_osim_marker_file(self, osim_marker_xml_filename: str) -> pd.DataFrame:
        ''' Parses the OpenSim marker xml file into a dataframe of the x,y,z locations of
            the markers indexed by marker name '''
        osim_marker_file = ET.parse(osim_marker_xml_filename)
        root = osim_marker_file.getroot()
        marker_names = []
        marker_positions = []
        for neighbor in root.iter('Marker'):
            marker_names.append(neighbor.attrib['name'])
            positions_string = neighbor.find('location').text.split(' ')
            marker_positions.append([np.float64(elem) for elem in positions_string if elem != ''])
        marker_positions = np.array(marker_positions)
        osim_data = pd.DataFrame(data = marker_positions, columns = ['x', 'y', 'z'], index = marker_names)
        return osim_data

    def save_osim_data_csv(self, filename: str) -> None:
        ''' Saves the OpenSim marker position dataframe in a csv '''
        self.osim_data.to_csv(filename)

    def parse_subject_marker_data(self, subject_data_dict: dict, frame_number: int = 0) -> pd.DataFrame:
        ''' Parses the subject's marker data dictionary into a dataframe of the x,y,z locations of
            the markers indexed by marker name. The column labels of the dataframe given by the
            class argument subject_marker_coords must match the dimensions of the data collected
            to the OpenSim coordinate frame. '''
        marker_names = list(subject_data_dict.keys())
        marker_positions = []
        for marker in subject_data_dict:
            marker_positions.append(subject_data_dict[marker][frame_number])
        marker_positions = np.array(marker_positions)
        subject_data = pd.DataFrame(data = marker_positions, columns = self.subject_marker_coords, index = marker_names) 
        return subject_data

    def save_subject_data_csv(self, filename: str) -> None:
        ''' Saves the subject marker position dataframe in a csv '''
        self.subject_data.to_csv(filename)

    def compute_scales(self) -> pd.DataFrame:
        ''' Computes the factors by which to scale each of the five bodies in our OpenSim model
            using the OpenSim and subject marker pairs specified in the dictionaries in
            marker_matching.py. The scales are computed as a ratio of the position of the subject
            marker in SUBJECT_MARKERS to that of the matched OpenSim marker in OSIM_MARKERS '''
        bodies = list(OSIM_MARKERS.keys())
        scale_data = []
        for body in OSIM_MARKERS:
            body_scales = []
            for dimension in OSIM_MARKERS[body]:
                osim_marker = OSIM_MARKERS[body][dimension]
                subject_marker = SUBJECT_MARKERS[body][dimension]
                if subject_marker is None:
                    body_scales.append(1)
                else:
                    subject_measurement = self.subject_data[dimension][subject_marker]
                    osim_measurement = self.osim_data[dimension][osim_marker]
                    ratio = subject_measurement / osim_measurement if osim_measurement != 0 else 1 + subject_measurement
                    body_scales.append(ratio)
            scale_data.append(body_scales)
        
        self.scale_df = pd.DataFrame(data = np.array(scale_data), columns = ['x', 'y', 'z'], index = bodies)
        return self.scale_df

    def save_scales_csv(self, filename: str) -> None:
        ''' Saves the scaling factors dataframe in a csv '''
        self.scale_df.to_csv(filename)

    def print_info(self) -> None:
        print('There are {} markers in the osim marker file and {} markers in the subject marker data'.format(self.osim_data.shape[0], self.subject_data.shape[0]))


def main():
    data_file = 'data/sample_data.pkl'
    with open(data_file, 'rb') as f:
        marker_data = pickle.load(f)
    scaler = Scaler(marker_data, 'markers/MarkerSet.xml')

    if SAVE_MARKER_DATA_CSV:
        scaler.save_osim_data_csv('osim.csv')
        scaler.save_subject_data_csv('subject.csv')
    
    scaler.compute_scales()
    scaler.save_scales_csv('scales.csv')


if __name__ == "__main__":
    main()