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
import sys, os, pyvisa, time
from pathlib import Path
cur_dir = Path(".").absolute()

sys.path.append(str(cur_dir.parent.parent))
sys.path.append(str(cur_dir.parent / "pna_control"))
sys.path.append(str(cur_dir.parent))

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from datetime import datetime
from pathlib import Path

# use NI MAX to get the possible instrument addresses you can use
# instr_addr = 'TCPIP0::K-N5231B-57006::inst0::INSTR'
instr_addr = 'TCPIP0::K-N5231B-57006.local::inst0::INSTR'

rm = pyvisa.ResourceManager()
# rm.list_resources()

try:
    PNA = rm.open_resource(instr_addr)

    ## Attempt to fix the timeout error in averaging command
    # PNA.timeout = None
    
except Exception as ex:
    print(f'\n----------\nException:\n{ex}\n----------\n')
    raise RuntimeError('Bad instrument address.')


# %%

###### measurement setup
points = 101
centerf = 6.44595e9
span = 0.5e6
if_bandwidth = 1e3
power = -10
edelay = 72.9
averages = 1
sparam = 'S21'
cal_set = None
segments = None
output_filename = None
temp = 0
sample_id = "ExampleVnaCommunication_Test"

dstr = datetime.today().strftime(r'%b_%d_%H%M')
freq_str = f'{centerf:.3f}GHz'.replace('.', 'p')
data_dir = f'data\\{dstr}\\{sample_id}_{freq_str}'

# %% pna_setup
'''
set parameters for the PNA for the sweep (number of points, center
frequency, span of frequencies, IF bandwidth, power, electrical delay and
number of averages)

XXX: Do not change this order:

1.  Define a measurement
2.  Turn on display
3.  Set the number of points
4.  Set the center frequency, span
5.  Turn on sweep time AUTO
6.  Set the electrical delay
7.  Turn on interpolation
8.  Set the calibration
9.  Set the power
10. Turn on averaging
11. Set the IF bandwidth

'''
    
# Send a preset command to the VNA and turn off the RF power
print("Initializing PNA..")

# XXX: Changed to user preset
# PNA.write('SYSTem:FPRESet')
PNA.write('SYSTem:UPRESet')
time.sleep(0.05)
PNA.write('OUTPut:STATe OFF')

# Initial setup for measurement
## Query the existing measurements
measurements = PNA.query('CALC1:PAR:CAT:EXTended?')

## If any measurements exist, delete them all
if measurements != 'NO CATALOG':
    PNA.write('CALC1:PARameter:DELete:ALL')
    
# create measurements
PNA.write(f'CALC1:MEASure1:DEFine \"{sparam}\"')
PNA.write(f'CALC1:MEASure2:DEFine \"{sparam}\"')

#set parameters for sweep

if segments:
    num_segments = len(segments)
    seg_data = ''.join([s for s in segments])
    PNA.write(f"SENSe1:SWEep:TYPE SEGment")
    PNA.write(f'SENSe1:SEGMent:LIST SSTOP, {num_segments}{seg_data}')
else:
    PNA.write("SENSe1:SWEep:TYPE LINear")
    PNA.write(f'SENSe1:SWEep:POINts {points}')
    PNA.write(f'SENSe1:FREQuency:CENTer {centerf}HZ')
    PNA.write(f'SENSe1:FREQuency:SPAN {span}HZ')

    PNA.write(f'SENSe1:SWEep:TIME:AUTO ON')
    
# print(PNA.query("SYST:CHAN:CAT?"))  # ask what channels are activated

PNA.write(f'SOUR1:POW1 {power}')
PNA.write(f'SENSe1:AVERage:STATe ON')
PNA.write(f'SENSe1:BANDwidth {if_bandwidth}HZ')

# configure ch 1 measurement 1
PNA.write(f'CALC1:PAR:MNUM 1')  # select ch 1, meas 1
PNA.write(f'DISPlay:WINDow1 ON')  # create window
PNA.write(f'DISPlay:MEAS1:FEED 1')  # display meas 1 on window 1
PNA.write(f'CALC1:CORRection:EDELay:TIME {edelay}NS')
PNA.write(f'CALC1:MEASure1:FORMat MLOGarithmic')

# configure ch 1 measurement 2
PNA.write(f'CALC1:PAR:MNUM 2')  # select ch 1, meas 2
PNA.write(f'DISPlay:WINDow2 ON')  # create window 2
PNA.write(f'DISPlay:MEAS2:FEED 2')  # display meas 2 on window 2
PNA.write(f'CALC1:CORRection:EDELay:TIME {edelay}NS')
PNA.write(f'CALC1:MEASure2:FORMat PHASe')

# autoscale for visibility on the display
PNA.write(f'DISPlay:WINDow1:TRACe1:Y:SCAle:AUTO')
PNA.write(f'DISPlay:WINDow2:TRACe1:Y:SCAle:AUTO')

# make sure to have averages as an integer
PNA.write(f'SENSe1:AVERage:Count {averages // 1}')


# %% get_data

# initiate display and turn on output
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
    print(f"      time elapsed: [{t_elapsed:1.0f}s]")
         
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


# %% read_data

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

# PNA.write('SYSTem:CHANnels:SINGLE')  # TODO: perhaps switch to using this cmd and INIT:IMM?
# PNA.write('INITiate:CONTinuous ON')
PNA.write('OUTPut:STATe OFF')


# %% # plot data

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


mosaic = "AACC\nBBCC"
fig, axes = plt.subplot_mosaic(mosaic, figsize=(10,5))
ax1, ax2, ax3 = axes["A"], axes["B"], axes["C"]

ax1.plot(freq/1e9, magn_lin, "r.")
ax2.plot(freq/1e9, phase_rad, "b.")
ax3.plot(real, imag, 'g.')

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
