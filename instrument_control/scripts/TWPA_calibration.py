# -*- coding : utf-8 -*-
"""
Anritsu MG3692C Signal Generator control test file

Author: Nick Materise, Kyle Thompson
Date:   220728

05/10/24 - overhauled all filepaths & addresses to be as relative as possible
            rather than hardcoded - Jorge
        
                
                
"""
%load_ext autoreload
%autoreload 3

import sys

sys.path.append(r'./')  # this file is in bcqt-ctrl\instrument_control\scripts
sys.path.append(r'../')  # this file is in bcqt-ctrl\instrument_control\scripts
sys.path.append(r'../pna_control/')  # need this to use VNA
sys.path.append(r'../../resonator_measurements/helper_scripts')  # for temperature reading

from anritsu import AnritsuCtrl
from datetime import datetime
import helper_misc as hm
import numpy as np

output_cmn, output_ls, tstamp = hm.read_temp_JCtrl(print_output=True)


# Color printing
RED   = '\033[31m'
GREEN = '\033[32m'
CRST  = '\033[0m' 

anritsu = AnritsuCtrl()  # always check default addresses
sweep_freqs = np.linspace(6.3, 7.3, 101)
sweep_powers = np.linspace(-10, -25, 10)
vna_dict = {'sample_id' : 'TWPA',
            'centerf' : 6,
            'span' : 4000,
            'temp' : 0,
            'avg' : 3,
            'power' : -25,
            'edelay' : 85,  
            'ifbw' : 100,
            'npts' : 201,
            'sparam' : 'S21',
            'cal_set' : None}

anritsu.power_frequency_sweep_2d(sweep_powers,
                                 sweep_freqs,
                                 sweep_order='frequency_power',
                                 run_vna=True,
                                 vna_dict=vna_dict)
