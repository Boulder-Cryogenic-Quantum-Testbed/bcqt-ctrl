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


# Set the center frequencies (GHz), spans (MHz), delays(ns), powers
   

fcs = [4.73167, 5.153348, 5.53986, 5.91299,
        6.326298, 6.7573459, 7.197065957, 7.604238]
spans = [5, 5, 5, 3,
        5, 3, 5, 5]
delays = [59.90, 59.93, 59.93, 59.93,
          59.95, 59.97, 60.0, 60.0]



# Change the sample name
sample_name = 'Nb only Cl QSG'
'''
powers = np.linspace(-15, -35, 5)

measure_multiple_resonators(fcs, spans, delays, powers,
        ifbw=1., sparam='S21', npts=51,
        adaptive_averaging=False, sample_name=sample_name,
        runtime=1., cal_set = None, start_delay=0.,
        is_segmented=True, offresfraction=0.8, use_homophasal=None,
        Navg_init=None)


'''
powers = np.linspace(-40, -70, 7)
measure_multiple_resonators(fcs, spans, delays, powers,
       ifbw=1., sparam='S21', npts=51,
       adaptive_averaging=True, sample_name=sample_name,
       runtime=0.25, cal_set = None, start_delay=0.,
       is_segmented=True, offresfraction=0.8, use_homophasal=None,
       Navg_init=None, bypass_janis=True, Tmxc=32e-3)



powers = np.linspace(-75, -95, 5)
measure_multiple_resonators(fcs, spans, delays, powers,
       ifbw=1., sparam='S21', npts=51,
       adaptive_averaging=True, sample_name=sample_name,
       runtime=4., cal_set = None, start_delay=0.,
       is_segmented=True, offresfraction=0.8, use_homophasal=None,
       Navg_init=None)

