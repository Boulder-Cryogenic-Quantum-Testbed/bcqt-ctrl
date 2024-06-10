# -*- coding : utf-8 -*-
"""
Anritsu MG3692C Signal Generator control test file

Author: Nick Materise, Kyle Thompson
Date:   220728
#
05/10/24 - overhauled all filepaths & addresses to be as relative as possible
            rather than hardcoded
                - Jorge
                
"""

import sys
# sys.path.append(r'E:\GitHub\bcqt-ctrl\instrument_control')
# sys.path.append(r'E:\GitHub\bcqt-ctrl\pna_control')
sys.path.append(r'../')  # instrument_control
sys.path.append(r'../../pna_control/')  

from anritsu import AnritsuCtrl
import numpy as np
import sys

# Color printing
RED   = '\033[31m'
GREEN = '\033[32m'
CRST  = '\033[0m' 

# hard coded addresses outside of AnritsuControl obj  :)
anritsu_addr='GPIB::7::INSTR'
vna_addr='TCPIP0::192.168.137.178::hislip0::INSTR'

anritsu = AnritsuCtrl(anritsu_addr, vna_addr)
sweep_freqs = np.linspace(6.3, 7.3, 101)
sweep_powers = np.linspace(-10, -25, 10)
vna_dict = {'sample_id' : 'TWPA',
            'centerf' : 6,
            'span' : 4000,
            'temp' : 35,
            'avg' : 3,
            'power' : -25,
            'edelay' : 76,  
            'ifbw' : 100,
            'npts' : 201,
            'sparam' : 'S12s',
            'cal_set' : None}

anritsu.power_frequency_sweep_2d(sweep_powers,
                                 sweep_freqs,
                                 sweep_order='frequency_power',
                                 run_vna=True,
                                 vna_dict=vna_dict)
