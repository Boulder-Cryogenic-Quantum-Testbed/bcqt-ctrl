#! -*- coding: utf-8 -*-
"""
Basic setup to talk to the VNA

- fixed up to work

    8/30/2024 
        Jorge

"""
# %%
# %load_ext autoreload
# %autoreload 3

# %%
import sys, os
from pathlib import Path
cur_dir = Path(".").absolute()

sys.path.append(str(cur_dir.parent.parent))
sys.path.append(str(cur_dir.parent / "pna_control"))
sys.path.append(str(cur_dir.parent))

import pyvisa
# import pna_control.pna_control as pna_control
import pna_control

from datetime import datetime
from pathlib import Path

# use NI MAX to get the possible instrument addresses you can use
# instr_addr = 'TCPIP0::K-N5231B-57006::inst0::INSTR'
instr_addr = 'TCPIP0::K-N5231B-57006.local::inst0::INSTR'

rm = pyvisa.ResourceManager()
try:
    PNA = rm.open_resource(instr_addr)

    ## Attempt to fix the timeout error in averaging command
    # PNA.timeout = None
    
except Exception as ex:
    print(f'\n----------\nException:\n{ex}\n----------\n')
    raise RuntimeError('Bad instrument address.')


# %%

import pandas as pd
import numpy as np
import time

###### measurement setup
points = 101
centerf = 6.446
span = 1
ifbandwidth = 1
power = -30
edelay = 72.9
averages = 3
sparam = 'S21'
cal_set = None
segments = None
output_filename = None
temp=0,
sample_id = "ExampleVnaCommunication_Test"

dstr = datetime.today().strftime(r'%b_%d_%H%M')
freq_str = f'{centerf:.3f}GHz'.replace('.', 'p')
data_dir = f'data\\{dstr}\\{sample_id}_{freq_str}'

# %%

pna_control.pna_setup(PNA, points, centerf, span, ifbandwidth, power, edelay, averages,
            sparam, cal_set, segments)


pna_control.get_data(   centerf=centerf,
                        span=span, 
                        temp=temp, 
                        averages=averages, 
                        power=power,
                        edelay=edelay, 
                        ifband_khz=ifbandwidth, 
                        points=points,
                        sample_id=sample_id, 
                        instr_addr=instr_addr,
                        cal_set=cal_set, 
                        setup_only=False, 
                        segments=segments,
                        filename_suffix=None,
                        data_dir = data_dir,
                        output_filename=output_filename, 
                        verbose = True
                        )





# %%
##### make measurement  (aka get_data)

# initiate display and turn on output
PNA.write('INITiate:CONTinuous ON')
PNA.write('OUTPut:STATe ON')
PNA.write('FORMat ASCII')
PNA.write('DISPlay:WINDow1:Y:AUTO')
PNA.write('DISPlay:WINDow2:Y:AUTO')

# initiate display and turn on output
# PNA.write('INITiate:CONTinuous ON')  # instead of turning on continuous mode
PNA.write('OUTPut:STATe ON')
PNA.write('ABORT;INITIATE:IMMEDIATE')  # just use INIT:IMM to trigger one sweep
PNA.write('FORMat ASCII')
PNA.write('DISPlay:WINDow1:Y:AUTO')
PNA.write('DISPlay:WINDow2:Y:AUTO')

# check if the VNA has finished every second, in my experience the *OPC? or *WAI command isnt very reliable
check = False
tstart = time.time()
    
    
while check is False:
    time.sleep(1)
    t_elapsed = time.time() - tstart
    print(f"      time elapsed: [{t_elapsed:1.1f}s]")
         
    # check_str is a string, "0" = busy or "1" = complete
    # check_str = PNA.query("*OPC?")[1]
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



# %% ### read data from VNA once finished

#read in frequency
cfreq = float(PNA.query('SENSe1:FREQuency:CENTER?'))

# This obviates the need for points as an input
if segments:
    # Read the list of all segments
    freq = np.array([])
    for s in segments:
        ssplit = s.replace(" ", "").split(',')
        
        # int() doesnt want a string of a float like '12.0', so if it has  
        # a decimal point, turn it into a float first
        nf = int(ssplit[2]) if '.' not in ssplit[2] else int(float(ssplit[2]))  
        f1 = float(ssplit[3])
        f2 = float(ssplit[4])
        f = np.linspace(f1, f2, nf)
        freqs = np.hstack((freq, f))
else:
    gpoints = int(PNA.query(f'SENSe1:SWEep:POINts?'))
    freqs = np.linspace(float(PNA.query('SENSe1:FREQuency:START?')),
            float(PNA.query('SENSe1:FREQuency:STOP?')), gpoints)

# TODO: switch to using CALC:DATA SMEM instead of FDATA to avoid format issues
# read in magn
PNA.write('CALC1:PAR:MNUM 1')  # select ch 1, meas 1
PNA.write('CALC1:FORMat MLOG')
magn = PNA.query_ascii_values('CALC1:DATA? FDATA', container=np.array)

# read in phase
PNA.write('CALC1:PAR:MNUM 2')  # select ch 1, meas 2
PNA.write('CALC1:FORMat PHASe')
PNA.write('DISPlay:WINDow2:Y:AUTO')
phase = PNA.query_ascii_values('CALC1:DATA? FDATA', container=np.array)


# PNA.write('SYSTem:CHANnels:SINGLE')
# PNA.write('INITiate:CONTinuous ON')
pna_control.read_data(PNA,
                      points,sample_id, power,temp, cfreq)

# %% # load data using pathlib

df = pd.DataFrame.from_dict(data={"Frequency":freqs, "S21 [dB]":magn, "Phase [deg]":phase}, orient="columns")


df["Phase [rad]"] = np.deg2rad(np.unwrap(df["Phase [deg]"]))
phase_rad, phase_deg = df["Phase [rad]"], df["Phase [deg]"]
magn_db, freq = df["S21 [dB]"], df["Frequency"]

## convert dataset
phase_rad = np.unwrap(np.deg2rad(phase_deg))
magn_lin = 10**(magn_db/20)
cmpl = magn_lin * np.exp(1j * phase_rad)
real, imag = np.real(cmpl), np.imag(cmpl)


# # %% plot data

import matplotlib.pyplot as plt

mosaic = "AACC\nBBCC"
fig, axes = plt.subplot_mosaic(mosaic, figsize=(10,5))
ax1, ax2, ax3 = axes["A"], axes["B"], axes["C"]

ax1.plot(freq/1e9, magn_lin, "r.")
ax2.plot(freq/1e9, phase_rad, "b.")
ax3.plot(real, imag, 'g*')

ax1.set_title("Freq vs Magn")
ax2.set_title("Freq vs Phase")
ax3.set_title("Real vs Imag")

ax1.set_xlabel("Frequency [GHz]")
ax2.set_xlabel("Frequency [GHz]")
ax3.set_xlabel("Real")

ax1.set_ylabel("S21 [dB]")
ax2.set_ylabel("Phase [Rad]")
ax3.set_ylabel("Imag")

fig.tight_layout()


# %%
