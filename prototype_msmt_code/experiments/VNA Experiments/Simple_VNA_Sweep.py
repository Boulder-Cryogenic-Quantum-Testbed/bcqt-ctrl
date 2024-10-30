
# %%
"""
    Test implementation of VNA driver
"""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time, sys

driver_path = str(Path(r"..\..\src\drivers").absolute())
sys.path.append(driver_path)  # lazy way to import drivers while in a subdir 

from VNA_Keysight import VNA_Keysight

VNA_Keysight_InstrConfig = {
    "instrument_name" : "VNA_Keysight",
    # "rm_backend" : "@py",
    "rm_backend" : None,
    "instr_address" : 'TCPIP0::192.168.0.105::inst0::INSTR',
    # "instr_address" : 'TCPIP0::K-N5231B-57006.local::inst0::INSTR',
}

PNA_X = VNA_Keysight(VNA_Keysight_InstrConfig, debug=True)


# %% plotting and data methods

## TODO: This should go in the ExptConfig object
Expt_Config = {
    "points" : 50,
    "fc" : 6445962657.0,
    "span" : 0.25e6,
    "if_bandwidth" : 200,
    "power" : -65,
    "edelay" : 72.9,
    "averages" : 300,
    "sparam" : 'S21',
    # "segment_type" : "homophasal",
    
    "segment_type" : "hybrid",
    "Noffres" : 3
}

if "all_dfs" not in locals().keys():
    all_dfs = {}

# %%

num_msmt = 5

for idx in range(num_msmt):
    
    Expt_Config["segments"] = PNA_X.compute_homophasal_segments(**Expt_Config)
    PNA_X.set_instr_params(Expt_Config)
    PNA_X.get_instr_params()
    PNA_X.setup_measurement()
    PNA_X.check_instr_error_queue()
    PNA_X.acquire_trace()
    freqs, magn_dB, phase_deg = PNA_X.return_data()

    # freqs, magn, phase = PNA_X.take_single_trace(Expt_Config)

    df, fig, axes = plot_data_with_pandas(freqs, magn_dB, phase_deg=phase_deg)

    title_str = str(f"{Expt_Config["span"]/1e6:1.2f}MHz_span_{Expt_Config["averages"]}_avgs_{Expt_Config["if_bandwidth"]}_IFBW_{Expt_Config["power"]}_dBm")
    fig = axes["A"].get_figure()
    fig.suptitle(title_str, size=16)
    fig.tight_layout()
    plt.show()

    all_dfs[title_str] = df

    Expt_Config["span"] += 0.5e6


# %%  plotting 

# plot all traces in their own plot
for key, df in all_dfs.items():
    freqs, magn_dB, phase_rad = unpack_df(df)
    df, fig, axes = plot_data_with_pandas(freqs, magn_dB, phase_rad=phase_rad)

# plot all complex circles by themselves
# fig, axes = plt.subplots(len(all_dfs), 1, figsize=(6, 32))
# for (key, df), ax in zip(all_dfs.items(), axes):
    
#     freqs, magn_dB, phase_rad = unpack_df(df)
#     df, fig, axes = plot_data_with_pandas(freqs, magn_dB, phase_rad=phase_rad, ax=ax)
    
#     print(freqs[magn_dB.argmin()])
#     ax.set_title(key)
    
# fig.tight_layout()
    

# %% run analysis with scresonators

