
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

## TODO: these should go in the DataProcessor object
def unpack_df(df): 
    freqs = df["Frequency"]
    magn_dB = df["S21 [dB]"]
    phase_rad = df["Phase [rad]"]
    return freqs, magn_dB, phase_rad

def plot_data_with_pandas(freqs, magn_dB, phase_deg=None, phase_rad=None, ax=None):

    # check phase input
    if phase_rad is None and phase_deg is not None:
        # degrees were given
        phase_rad = np.unwrap(np.deg2rad(phase_deg))
        
    elif phase_rad is not None and phase_deg is None:
        # radians were given
        pass
        
    elif phase_rad is None and phase_deg is None:
        # both are None
        raise ValueError("One of phase_rad and phase_deg must be given!")
    
    else:
        # both were given
        if phase_rad != np.deg2rad(phase_deg):
            raise ValueError(f"Both radians and degrees were given, but they don't match!\n  {phase_rad != np.deg2rad(phase_deg) = }")
        
    df = pd.DataFrame.from_dict(data={"Frequency":freqs, "S21 [dB]":magn_dB, "Phase [rad]":phase_rad}, orient="columns")
    
    ## convert dataset
    magn_lin = 10**(magn_dB/20)
    cmpl = magn_lin * np.exp(1j * phase_rad)
    real, imag = np.real(cmpl), np.imag(cmpl)

    # create a new plot or use one given as arg
    if ax is None:
        # # %% plot data
        mosaic = "AACC\nBBCC"
        fig, axes = plt.subplot_mosaic(mosaic, figsize=(10,5))
        ax1, ax2, ax3 = axes["A"], axes["B"], axes["C"]
    else:
        # only plot complex portion, then return early
        ax3 = ax
        fig = ax3.get_figure()
    
    ax3.plot(real, imag, 'g.')
    ax3.set_ylabel("Imag")
    ax3.set_title("Real vs Imag")
    ax3.axhline(0, linestyle=':', linewidth=1, color='k')
    ax3.axvline(0, linestyle=':', linewidth=1, color='k')
    ax3.set_aspect("equal")
    
    if ax is not None:
        fig.tight_layout()
        return df, fig, [ax3]

    # magn and phase
    ax1.plot(freqs/1e9, magn_lin, "r.")
    ax2.plot(freqs/1e9, phase_rad, "b.")

    ax1.set_title("Freq vs Magn")
    ax2.set_title("Freq vs Phase")

    ax1.set_xlabel("Frequency [GHz]")
    ax2.set_xlabel("Frequency [GHz]")
    ax3.set_xlabel("Real")

    ax1.set_ylabel("S21 [dB]")
    ax2.set_ylabel("Phase [Rad]")
    fig.tight_layout()

    
    return df, fig, axes



# %%

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

