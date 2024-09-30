#! -*- coding: utf-8 -*-
"""
Basic setup to talk to the VNA

- fixed up to work

    8/30/2024 
        Jorge

"""

%load_ext autoreload
%autoreload 3

# %%

import pyvisa
import pna_control as PNA 

# use NI MAX to get the possible instrument addresses you can use
instr_addr = 'TCPIP0::192.168.0.113::inst0::INSTR'

rm = pyvisa.ResourceManager()
try:
    PNA = rm.open_resource(instr_addr)

    ## Attempt to fix the timeout error in averaging command
    # PNA.timeout = None
    
except Exception as ex:
    print(f'\n----------\nException:\n{ex}\n----------\n')
    raise RuntimeError('Bad instrument address.')


# %%

from pna_control import pna_setup, read_data, get_data
import pandas as pd
import numpy as np
import time

###### measurement setup
points = 20001
centerf_GHz = 7
span_MHz = 10000
ifband_khz = 5
power = -30
edelay = 7.471
averages = 1
sparam = 'S21'
cal_set = None
segments = None
output_filename = None
temp=0
sample_id = "IR_Filter_A"

# pna_setup(PNA, points, centerf_GHz, span_MHz, ifband_khz, power, edelay, averages,
#             sparam, cal_set, segments)


get_data(centerf_GHz,
            span_MHz, 
            temp, 
            averages, 
            power,
            edelay, 
            ifband_khz, 
            points,
            sample_id, 
            instr_addr = "TCPIP0::192.168.0.113::inst0::INSTR",
            cal_set=cal_set, 
            setup_only=False, 
            segments=segments,
            filename_suffix=None,
            data_dir = None,
            verbose = True)

# %%

# %%






# %%
##### make measurement  (aka get_data)

# initiate display and turn on output
PNA.write('INITiate:CONTinuous ON')
PNA.write('OUTPut:STATe ON')
PNA.write('FORMat ASCII')
PNA.write('DISPlay:WINDow1:Y:AUTO')
PNA.write('DISPlay:WINDow2:Y:AUTO')

# check if the VNA has finished every second
check = False
tstart = time.time()
while check is False:
    time.sleep(1)
    t_elapsed = time.time() - tstart
    print(f"      time elapsed: [{t_elapsed:1.0f}s]")
        
    # check_str is a string, "0" or "1"
    check_str = PNA.query('STAT:OPER:AVER1:COND?')[1]
    print(check_str)
    
    # once it is "1", print that we're finished
    if check_str != "0":
        print(f"\nTrace finished. Uploading now.")
        print(f"\n   Total time elapsed: {t_elapsed:1.0f} seconds")
        if t_elapsed >= 600:
            print(f"                     = {t_elapsed/60:1.1f} minutes \n")
    
    # update the variable and let the while finish
    check = bool(check_str)
    
    
PNA.query('*OPC?')
PNA.write('*WAI')
time.sleep(3)
PNA.write('SYSTem:CHANnels:HOLD')

#### read data once finished
read_data(PNA, 
            sample_id='Test',
            points = 20001,
            centerf = 7,
            power = -30,
            temp = 12, 
            segments = None,
            verbose=True)

# PNA.write('SYSTem:CHANnels:SINGLE')
# PNA.write('INITiate:CONTinuous ON')

# # convert data
# phase_rad = np.deg2rad(phase_deg)
# complex = magn * np.exp(1j * phase_deg)
# real, imag = np.real(complex), np.imag(complex)

# ## %% plot data

# import matplotlib.pyplot as plt

# mosaic = "AACC\nBBCC"
# fig, axes = plt.subplot_mosaic(mosaic, figsize=(8,5))
# ax1, ax2, ax3 = axes["A"], axes["B"], axes["C"]

# ax1.plot(freq, magn, "r.")
# ax2.plot(freq, phase_deg, "b.")
# ax3.plot(real, imag, 'g*')

# fig.tight_layout()

