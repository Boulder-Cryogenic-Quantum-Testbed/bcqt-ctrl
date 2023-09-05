# -*- encoding: utf-8 -*-
"""
User file for controlling the Janis and PNA instruments

    Make sure you login to the JetWay session on PuTTY
    with user: 'bco' and password: 'aish8Hu8'

"""
import sys

# Change this path
sys.path.append(r'C:\Users\Lehnert Lab\GitHub\bcqt-ctrl\temperature_control')
from janis_ctrl import measure_multiple_resonators
import numpy as np
import time


# Set the center frequencies, spans, delays, powers
fcs = [4.88144904, 5.3562, 5.843768, 6.32389, 7.25785]
spans = [0.2, 15., 6.5, 0.75, 2.]
delays = [62.86, 62.87, 62.88, 62.82, 62.83] 

# Change the sample name
sample_name = 'SiNb'

powers = np.linspace(-15, -35, 5)
measure_multiple_resonators(fcs, spans, delays, powers,
        ifbw=1., sparam='S12', npts=51,
        adaptive_averaging=False, sample_name=sample_name,
        runtime=1., cal_set = None, start_delay=0.,
        is_segmented=True, offresfraction=0.8, use_homophasal=None,
        Navg_init=None)

powers = np.linspace(-40, -70, 7)
measure_multiple_resonators(fcs, spans, delays, powers,
        ifbw=1., sparam='S12', npts=51,
        adaptive_averaging=True, sample_name=sample_name,
        runtime=0.25, cal_set = None, start_delay=0.,
        is_segmented=True, offresfraction=0.8, use_homophasal=None)

powers = np.linspace(-75, -95, 5)
measure_multiple_resonators(fcs, spans, delays, powers,
        ifbw=1., sparam='S12', npts=51,
        adaptive_averaging=True, sample_name=sample_name,
        runtime=4., cal_set = None, start_delay=0.,
        is_segmented=True, offresfraction=0.8, use_homophasal=None)
