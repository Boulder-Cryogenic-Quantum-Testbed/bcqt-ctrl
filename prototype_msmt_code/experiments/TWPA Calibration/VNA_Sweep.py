
# %%
"""
    TWPA Calibration - VNA Sweep

        Procedure:
        
            (0) Set initial TWPA pump frequency and pump power
            (1) Perform two identical VNA sweeps, one with and one without the TWPA pump
            (2) Repeat (1) for a new pump power but at the same frequency
            (3) Repeat (1)-(2) for a new frequency
            
        At the end, you will have a dataset that consists of two traces swept over power, and swept over frequency.
        
        You can either see the *raw* gain profile of the TWPA from creating a surface plot of just the "TWPA On" trace, or you
        can see the *effective* gain of the TWPA by creating a surface plot of the ratio/difference of the "TWPA On" and "TWPA Off"
        traces.
        
        Why do we need two traces? Because, generally we want the pump config that give the best SNR, not the most gain. It is possible 
        that a given config has more gain, but also has a higher noise floor. Thus, we want to have on/off traces so that we can compare
        the *effective* increase in SNR.
        
"""

from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time, sys


dstr = datetime.today().strftime("%m_%d_%I%M%p")
current_path = Path(".")
script_filename = Path(__file__).stem

msmt_code_path = Path(r"../..").resolve()
experiment_path = Path("..").resolve()
src_path = msmt_code_path / "src"
driver_path = msmt_code_path / "drivers"
instr_path = driver_path / "instruments"
data_path = current_path / "data" / script_filename / dstr 
csv_path = data_path / "raw_csvs"
dcm_path = data_path / "dcm_fits"

# all_paths = [current_dir, experiment_path, msmt_code_path, src_path, driver_path, instr_path, data_path, csv_path, dcm_path]

# grab all the variables in the local space with "path" in it and save into dict
all_paths_dict = {k:v for k, v in locals().items() if isinstance(v, Path)}

# %%

# make sure all paths exist, then append to $PATH
for path_name, path in all_paths_dict.items():
    path = path.resolve()  # convert relative Path objs to absolutes
    print(f"Checking if path [{path_name}] exists:  ['{path}']")
    print(f"             {str(path.exists()).upper()}\n")
    
    # ensure our data/fit storage paths exists
    if path.exists() is False and path in [csv_path.absolute(), data_path.absolute(), dcm_path.absolute()]:
        path.mkdir(parents=True, exist_ok=True)
        print(f"       ->  Created! Path now exists. [{path.exists() = }]")
    
    sys.path.append(str(path))


# %%
from VNA_Keysight import VNA_Keysight
from SG_Anritsu import SG_Anritsu
from SA_RnS_FSEB20 import SA_RnS_FSEB20

VNA_Keysight_InstrConfig = {
    "instrument_name" : "VNA_Keysight",
    # "rm_backend" : "@py",
    "rm_backend" : None,
    "instr_address" : 'TCPIP0::192.168.0.105::inst0::INSTR',
    # "instr_address" : 'TCPIP0::K-N5231B-57006.local::inst0::INSTR',
}


SA_RnS_InstrConfig = {
    "instrument_name" : "SA_RnS",
    # "rm_backend" : "@py",
    "rm_backend" : None,
    "instr_address" : 'GPIB::20::INSTR',      
}


SG_Anritsu_InstrConfig = {
    "instrument_name" : "SG_Anritsu",
    # "rm_backend" : "@py",
    "rm_backend" : None,
    "instr_address" : 'GPIB::7::INSTR',  # test instr
    # "instr_address" : 'GPIB::8::INSTR',  # twpa
}

# %% initialize instruments

PNA_X = VNA_Keysight(VNA_Keysight_InstrConfig, debug=True)
SIG_Generator = SG_Anritsu(SG_Anritsu_InstrConfig, debug=True)
SIG_Analyzer = SA_RnS_FSEB20(SA_RnS_InstrConfig, debug=True)

all_instr = [PNA_X, SIG_Generator, SIG_Analyzer]
# all_instr = [SIG_Generator, SIG_Analyzer]


# %%

# for instr in all_instr:
#     instr.check_instr_error_queue(print_output=True)
#     instr.print_console(instr.idn, prefix="self.write(*IDN?) ->")
#     instr.return_instrument_parameters(print_output=True)

# %%
SIG_Generator.set_freq(4.909e9)
SIG_Generator.set_power(-17)
SIG_Generator.set_output(True)

SIG_Analyzer.set_freq_center_Hz(5.0e9)
SIG_Analyzer.set_freq_span_Hz(1e9)
SIG_Analyzer.set_IF_bandwidth(5000)
# SIG_Analyzer.trigger_sweep()
# SIG_Analyzer.resource.write("*OPC")
SIG_Analyzer.arm_trigger()
print(SIG_Analyzer.send_cmd_and_wait("INIT:IMM"))

SIG_Generator.check_instr_error_queue(print_output=True)
SIG_Analyzer.check_instr_error_queue(print_output=True)

# # %%
 
freqCenterHz = SIG_Analyzer.get_freq_center_Hz()
freqSpan = SIG_Analyzer.get_freq_span_Hz()

traceStr = SIG_Analyzer.query_check(f'TRAC:DATA? TRACE{1}') 
traceData = [float(x) for x in traceStr.split(',')]
freqs = np.linspace(freqCenterHz - freqSpan / 2, freqCenterHz + freqSpan / 2, len(traceData));

# display(SIG_Generator.return_instrument_parameters())
# display(SIG_Analyzer.return_instrument_parameters())

from scipy.signal import find_peaks
peaks, _ = find_peaks(traceData, prominence=20)

fig, ax = plt.subplots(1, 1, figsize=(8,5))
ax.plot(freqs, traceData, '.-')
for idx in peaks:
    ax.plot(freqs[idx], traceData[idx], "o", fillstyle="none", markeredgewidth=2, markersize=10,
             label=f"{freqs[idx]/1e6:1.2f} MHz")
    
ax.set_title("Spectrum Analyzer Output")
fig.legend()

plt.show()

# %%
# %%
# *OPC? places a 1 on the output queue when operation is complete. *OPC raises 
# bit 0 in the event status register when operation is complete. Both outputs are used for client synchronization.
# For example, the client may wait until it receives 1 on the output queue from *OPC?