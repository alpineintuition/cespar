''' 
To be manually set by the user such as to match the marker positions 
in the dataset to the ones in the default OpenSim model MarkerSet
xml file. Use 'None' if no matching marker exists.
'''
OSIM_MARKERS = {
    'torso':  {'x': 'Sternum',       'y': 'Sternum',        'z': 'R.Acromium'},
    'pelvis': {'x': 'R.ASIS',        'y': 'R.ASIS',         'z': 'R.ASIS'},
    'femurs': {'x': 'R.Thigh.Front', 'y': 'R.Thigh.Front',  'z': 'R.Thigh.Front'},
    'tibias': {'x': 'R.Ankle.Lat',   'y': 'R.Ankle.Lat',    'z': 'R.Ankle.Lat'},
    'feet':   {'x': 'R.Toe.Tip',     'y': 'R.Toe.Tip',      'z': 'R.Toe.Tip'}
}

SUBJECT_MARKERS = {
    'torso':  {'x': 'stern',            'y': 'stern',           'z': None},
    'pelvis': {'x': 'right_asis',       'y': 'right_asis',      'z': 'right_asis'},
    'femurs': {'x': 'right_fl1',        'y': 'right_fl1',       'z': 'right_fl1'},
    'tibias': {'x': 'right_lat_mall',   'y': 'right_lat_mall',  'z': 'right_lat_mall'},
    'feet':   {'x': 'right_toes',       'y': 'right_toes',      'z': 'right_toes'}
}