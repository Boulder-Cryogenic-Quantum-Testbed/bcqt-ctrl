# -*- coding : utf-8 -*-
"""
User file to control temperature, take data on Bluefors LD250
with the Lakeshore Model 372 AC resistance bridge

Author: Nick Materise
Date:   230915

"""

import sys
# Need to test relative path, otherwise try the above
sys.path.append(r'C:\Users\68707\Documents\bcqt\bcqt-ctrl\temperature_control')
from bluefors_ctrl import BlueForsCtrl
from bluefors_ctrl import measure_multiple_resonators_tsweep
import numpy as np
import time

# List of temperatures to try
Tlist = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]

# Set the center frequencies, spans, delays, powers
# fcs = [5.2100783, 6.5512265, 7.87484732811]
# spans = [0.5, 1., 0.2]
# delays = [61.32, 59.97, 60.05] 

fcs = [6.5512265, 7.87484732811]
spans = [1., 0.2]
delays = [59.97, 60.05] 
powers = [-35]

# # Change the sample name
# sample_name = 'SPCAl4N_NIST_NOETCH'
# measure_multiple_resonators_tsweep(fcs, spans, delays, powers,
#         ifbw=1., sparam='S21', npts=51,
#         adaptive_averaging=False, sample_name=sample_name,
#         runtime=0.1, cal_set = None, start_delay=0.,
#         is_segmented=True, offresfraction=0.8, use_homophasal=None,
#         Navg_init=None, Tlist=Tlist, thermalization_minutes=10.)

Tlist = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
fcs = [6.5512265, 7.87484732811]
spans = [1., 0.2]
delays = [59., 1., 0.297, 60.05] 

Tlist = [300]
fcs = [5.2100783]
spans = [0.5]
delays = [61.32, 59.97, 60.05] 
sample_name = 'SPCAl4N_NIST_NOETCH'
measure_multiple_resonators_tsweep(fcs, spans, delays, powers,
        ifbw=1., sparam='S21', npts=51,
        adaptive_averaging=False, sample_name=sample_name,
        runtime=0.1, cal_set = None, start_delay=0.,
        is_segmented=True, offresfraction=0.8, use_homophasal=None,
        Navg_init=None, Tlist=Tlist, thermalization_minutes=0.)
