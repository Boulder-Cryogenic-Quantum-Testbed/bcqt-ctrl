# %%
"""

    Fast Qi Tracking via VNA

"""

# %%

from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time, sys

dstr = datetime.today().strftime("%m_%d_%I%M%p")
current_dir = Path(".")
script_filename = Path(__file__).stem

# %% create folders and make path

# lazy way to import modules - just append to path... TODO: fix via proper __init__.py :)

msmt_code_path = Path(r"../../..").resolve()
experiment_path = Path("..").resolve()
src_path = msmt_code_path / "src"
driver_path = msmt_code_path / "drivers"
instr_path = driver_path / "instruments"
data_path = current_dir / "data" / script_filename / dstr 
csv_path = data_path / "raw_csvs"
dcm_path = data_path / "dcm_fits"

all_paths = [current_dir, experiment_path, msmt_code_path, src_path, driver_path, instr_path, data_path, csv_path, dcm_path]

# make sure all paths exist, then append to $PATH
for path in all_paths:
    path = path.resolve()  # convert relative Path objs to absolutes
    print(f"Checking if path exists:  ['{path}']")
    print(f"     {str(path.exists()).upper()}")
    
    # ensure our data/fit storage paths exists
    if path.exists() is False and path in [csv_path.absolute(), data_path.absolute(), dcm_path.absolute()]:
        path.mkdir(parents=True, exist_ok=True)
        print(f"       ->  Created! Path now exists. [{path.exists() = }]")
    
    sys.path.append(str(path))

# %%
from src.DataAnalysis import DataAnalysis
from VNA_Keysight import VNA_Keysight
from quick_helpers import unpack_df, plot_data_with_pandas
# from DataAnalysis import DataAnalysis

VNA_Keysight_InstrConfig = {
    "instrument_name" : "VNA_Keysight",
    "rm_backend" : "@py",
    # "rm_backend" : None,
    "instr_address" : 'TCPIP0::192.168.0.105::inst0::INSTR',
    # "instr_address" : 'TCPIP0::K-N5231B-57006.local::inst0::INSTR',
}

PNA_X = VNA_Keysight(VNA_Keysight_InstrConfig, debug=True)


if "all_dfs" not in locals().keys():
    all_dfs = {}
    
# %%

all_fcs =  [ 5.733901e9,
             5.773425e9,
             5.822726e9,
             5.863280e9,
            
             6.256667811e9,   # sucks, doesnt saturate
             6.306375544e9,   # saturates around -78 dBm
             6.360336e9,
             6.417038e9
            ]


Expt_Config = {
    "points" : 1000,
    "fc" : all_fcs[0],
    "span" : 0.25e6,
    "if_bandwidth" : 1000,
    "power" : -40,
    "edelay" : 76.36,
    "averages" : 1,
    "sparam" : 'S21',
    
    # "segment_type" : "homophasal",
    "segment_type" : "hybrid",
    
    "Noffres" : 5
}


# %%

num_msmt = 1

Expt_Config["segments"] = PNA_X.compute_homophasal_segments(**Expt_Config)
PNA_X.set_instr_params(Expt_Config)
PNA_X.get_instr_params()
PNA_X.setup_measurement()

for idx in range(num_msmt):
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


    Res_PowSweep_Analysis = DataAnalysis(None, dstr)

    
    # print(f"Fitting {filename}")
    
    power = Expt_Config["power"]
    time_end = Expt_Config["time_end"]
    
    try:
        # output_params, conf_array, error, init, output_path
        params, conf_intervals, err, init1, fig = Res_PowSweep_Analysis.fit_single_res(data_df=df, save_dcm_plot=False, plot_title=filename, save_path=dcm_path)
    except Exception as e:
        # print(f"Failed to plot DCM fit for {power} dBm -> {filename}")
        continue
        


# %%  plotting 
    

# %% run analysis with scresonators

