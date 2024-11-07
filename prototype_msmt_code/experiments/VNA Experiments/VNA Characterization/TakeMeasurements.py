# %%
"""
    Test implementation of VNA driver
"""

%load_ext autoreload
%autoreload 2

# %%

from pathlib import Path
from datetime import datetime
import sys
import matplotlib.pyplot as plt
import pandas as pd

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

all_paths = [current_dir, experiment_path, msmt_code_path, src_path, driver_path, instr_path]

# make sure all paths exist, then append to $PATH
for path in all_paths:
    path = path.resolve()  # convert relative Path objs to absolutes
    print(f"Checking if path exists:  ['{path}']")
    print(f"     {str(path.exists()).upper()}")
    sys.path.append(str(path))

# %%

import quick_helpers as qh
from VNA_Keysight import VNA_Keysight

VNA_Keysight_InstrConfig = {
    "instrument_name" : "VNA_Keysight",
    # "rm_backend" : "@py",
    "rm_backend" : None,
    "instr_address" : 'TCPIP0::192.168.0.105::inst0::INSTR',
}

PNA_X = VNA_Keysight(VNA_Keysight_InstrConfig, debug=True)

# %%

# %%

""" 
    most of the measurements for the characterization will be taken in the interactive window, so
        let's create some helper functions to speed up that process
        
    characterization workflow:
    
        1) Reset_VNA() will set the default configs and then call setup_s2p_measurement() to get 
            all four s-parameters ready for measurement
        2) Take_Data() will check for errors, perform measurement, and then plot for quick visualization
        3) Repeat steps 1-2 until happy with trace
             .
             .
        4) Archive_Results() will get the results from the VNA and save to a specified location with
            appropriate metadata.
"""

def Reset_VNA(VNA, config, **kwargs):
    """
        Resets the VNA to default configs, then passes any kwargs to the configs,
        and then sets up the s2p measurement.
    """
    VNA.check_instr_error_queue()
    VNA.set_instr_params(config)
    VNA.get_instr_params()
    
    
def Make_Measurement(VNA, **kwargs):
    """
        instruct VNA to do error checks, add configs, and perform measurement
    """
    VNA.setup_s2p_measurement()
    VNA.add_kwargs_and_filter_configs(**kwargs)
    VNA.check_instr_error_queue()
    VNA.acquire_trace()
    
def Acquire_Trace(VNA, plot_complex=True):
    """
        Should take zero time, only asks VNA to send the data it has
        
        
        returns:
            df = pandas dataframe of acquired data
                 eight columns of magn/phase scattering parameters
    
    """
    data_dict = VNA.return_data_s2p()
    
    all_dfs = {}
    for sparam, (freqs, magn_dB, phase_rad) in data_dict.items():
        #########################
        # plot data with helper function
        df, fig, axes = qh.plot_data_with_pandas(freqs, magn_dB, phase_rad=phase_rad, plot_complex=plot_complex)

        # add datetime to first row for archive
        first_row = {col : val for col, val in zip(df.columns, [datetime.now()]*len(df.columns))}
        datetime_row = pd.DataFrame(first_row, index=["datetime.now()"])
        df = pd.concat([datetime_row, df.iloc[:]])
        
        # title_str = str(f"{VNA.configs["f_span"]/1e6:1.2f}MHz_span_{VNA.configs["averages"]}_avgs_{VNA.configs["if_bandwidth"]}_IFBW_{VNA.configs["power"]}_dBm")
        title_str = sparam
        fig = axes["A"].get_figure()
        fig.suptitle(title_str, size=32)
        fig.tight_layout()
        
        # fc, span, ifbw, avg, power = Expt_Config["fc"], Expt_Config["span"], Expt_Config["if_bandwidth"], Expt_Config["averages"], Expt_Config["power"]

        all_dfs[sparam] = df
        
    
    
    return all_dfs
        #########################

def Archive_Data(VNA, all_dfs:list, expt_name:str, expt_category:str = '', save_dir:str = "./data"):
    # check if save_dir is a path or string
    if not isinstance(save_dir, Path):
        save_dir = Path(save_dir)
    
    # check if save_dir exists
    if not save_dir.exists():
        VNA.print_console(f"Creating directory {save_dir} under category {expt_category}")
        if expt_category not in save_dir:
            save_dir = save_dir / expt_category
        save_dir.mkdir(exist_ok=True, parents=True)
    
    # append number to end of filename and save to csv
    expt_no = len(save_dir.glob("*.csv")) + 1    
    filename = str(save_dir / f"{expt_name}_{expt_no:03d}.csv")
    VNA.print_console(f"Saving data as {filename.name}")
    
    df.to_csv(filename)
    
    return df

    #########################
# %% set default values

DefaultConfig = {
    "n_points" : 20001,
    "f_start" : 4e9,
    "f_stop" : 8e9,
    "if_bandwidth" : 5000,
    "power" : -20,
    "edelay" : 0,
    "averages" : 10,
    "sparam" : 'all',
    
    "segment_type" : "linear",
}

PNA_X.set_instr_params(DefaultConfig)

# %% example usage

Measurement_Configs = {
    "f_start" : 2e9,
    "f_stop" : 10e9,
    "n_pts" : 2001,
    "if_bw" : 15000,
    "power" : -20,
    
    # by default, sparam = 'all', edelay = 0, averages = 10
}

expt_category = "TestCode"
meas_name = "Attenuator_With_Thru"

Reset_VNA(PNA_X, Measurement_Configs)
Make_Measurement(PNA_X)
all_dfs = Acquire_Trace(PNA_X, plot_complex=False)
Archive_Data(PNA_X, all_dfs, expt_category, meas_name)


# %%
