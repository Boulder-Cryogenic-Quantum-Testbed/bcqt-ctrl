# -*- coding : utf-8 -*-
"""
Anritsu MG3692C Signal Generator user file
used to calibrate TWPA over pump freq & pump power

Author: Jorge Ramirez
Date:   08/2024
                
    goal:  we want to use the VNA to find the best settings for the TWPA. 
    
        - 1) find the best TWPA pump frequency, should be around 7.9 GHz
        - 2) find the best TWPA pump power, should be around -17 dBm
        
        we first sweep the TWPA pump frequency at a couple different powers, then find
        the best performing frequency and power. Once we get that frequency and power, we
        do a fine tuning and sweep much smaller increments of power and frequency.
        
        what are we optimizing? we want to get the best effective SNR increase, not 
        gain or noise reduction, though they are directly proportional. Effective 
        SNR increase can be characterized by comparing the gain in your "signal" to 
        the gain in your "noise" background.
         
"""
# %%

%load_ext autoreload
%autoreload 3

from pathlib import Path
import sys, os, math, pyvisa

from datetime import datetime
import numpy as np

dstr = str(datetime.now().strftime("%m%d_%H%M%p"))

# system path manipulation 
cwd = Path.cwd()
bcqt_ctrl = cwd.parent.parent.parent
sys.path.append(str(cwd.parent))  # 'Signal Generators'
sys.path.append(str(bcqt_ctrl / 'pna_control'))  
sys.path.append(str(bcqt_ctrl / "resonator_measurements" / 'helper_scripts'))

data_dir = cwd / 'data' / dstr
if not data_dir.exists():
    os.makedirs(data_dir)

# import our code
from anritsu import AnritsuCtrl
import helper_misc as hm

# %%
output_cmn, output_ls, tstamp = hm.read_temp_JCtrl(print_output=True)
temp = math.ceil(float(output_cmn["T"])*1e3)

# Color printing
RED   = '\033[31m'
GREEN = '\033[32m'
CRST  = '\033[0m' 

anritsu = AnritsuCtrl(verbose=True)  # always check default addresses if connection fails

# %% measuremetn parameters

# start with broad sweep over frequency, and only a few powers
sweep_freqs = np.linspace(3, 12, 501)

# sweep_powers = np.linspace(-15, -17, 3)
sweep_powers = [-17]

vna_dict = {
            'sample_id' : 'TWPA',
            'centerf' : 6,
            'span' : 4000,
            'temp' : temp,
            'avg' : 3,
            'power' : -40,
            'edelay' : 86.76,  
            'ifband_khz' : 1,
            'npts' : 501,
            'sparam' : 'S21',
            'cal_set' : None,
            'make_filename' : True,
            'filename_prefix' : '', 
            'filename_suffix' : '',  # will be set by power_frequency_sweep_2d()
            'data_dir' : data_dir,
        }


# %%

attempts_left = 3
while attempts_left > 0:
    display(attempts_left)
    try:
        anritsu.power_frequency_sweep_2d(sweep_powers,
                                        sweep_freqs,
                                        sweep_order='power_frequency',
                                        run_vna=True,
                                        vna_dict=vna_dict)
        break # if successful, then break out of while loop
    except pyvisa.errors.InvalidSession as error:
        print(error)
        attempts_left -= 1

        


# %%
