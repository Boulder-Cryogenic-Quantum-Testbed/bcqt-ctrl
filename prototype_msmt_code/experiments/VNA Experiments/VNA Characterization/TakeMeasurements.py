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
experiment_path = Path("..").resolve().parent
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
    VNA.run_measurement()
    
def Acquire_Trace(VNA, plot_complex=True, track_min=True, title="", do_edelay_fit=True):
    """
        Should take zero time, only asks VNA to send the data it has
        
        
        returns:
            df = pandas dataframe of acquired data
                 eight columns of magn/phase scattering parameters
    
    """
    df = VNA.return_data_s2p()
    all_axes = qh.plot_s2p_df(df, plot_complex, track_min, title, do_edelay_fit=do_edelay_fit)
    
    return df, all_axes
        #########################

def Archive_Data(VNA, s2p_df:pd.DataFrame, meas_name:str, expt_category:str = '', save_dir:str = "./data", all_axes=None):
    # check if save_dir is a path or string
    if not isinstance(save_dir, Path):
        save_dir = Path(save_dir).absolute()
        
    timestamp = datetime.today().strftime("%m_%d_%I%M%p")
    file_dir = save_dir / expt_category / meas_name / timestamp
    
    # check if file_dir exists
    if not file_dir.exists():
        VNA.print_console(f"Creating category {expt_category}")
        VNA.print_console(f"    under save directory {save_dir}")
        file_dir.mkdir(exist_ok=True, parents=True)
    
    # append number to end of filename and save to csv
    expt_no = len(list(save_dir.glob("*.csv"))) + 1    
    filename = f"{meas_name}_{expt_no:03d}.csv"
    VNA.print_console(f"Saving data as {filename}")
    VNA.print_console(f"    under '{str(Path(*file_dir.parts[-6:]))}'")
    
    final_path = file_dir / filename
    print(final_path)
    s2p_df.to_csv(final_path)
    
    if all_axes is not None:
        for axes in all_axes:
            ax = axes[0]
            fig = ax.get_figure()
            title = fig.get_suptitle().replace(" - ","_") + ".png"
            fig_filename = file_dir / title
            fig.tight_layout()
            fig.savefig(fig_filename)
            plt.show()
            print(fig_filename)
            
    
    return filename, final_path.parent


    #########################
# %% set default values

DefaultConfig = {
    "n_points" : 20001,
    "f_start" : 4e9,
    "f_stop" : 8e9,
    "if_bandwidth" : 5000,
    "power" : -75,
    "edelay" : 0,
    "averages" : 3,
    "sparam" : ['S11', 'S22', 'S21'],  # do not want to measure S12 in VNA two port mode
    
    "segment_type" : "linear",
}

PNA_X.set_instr_params(DefaultConfig)

# %% example usage

Measurement_Configs = {
    "f_start" : 2e9,
    "f_stop" : 10e9,
    "n_pts" : 4001,
    "if_bw" : 5000,
    "power" : -60,
    "averages" : 3,
    
    # by default, sparam = 'all', edelay = 0, averages = 10
}

expt_category = "Circulators"
meas_name = "VNA_Purple_Cables"

Reset_VNA(PNA_X, Measurement_Configs)
Make_Measurement(PNA_X)

# %%
# TODO: stop this method from resetting display configs
all_dfs, all_axes = Acquire_Trace(PNA_X, plot_complex=False, track_min=False, title=meas_name, do_edelay_fit=True)
filename, filepath = Archive_Data(PNA_X, all_dfs, meas_name=meas_name, expt_category=expt_category, all_axes=all_axes)

# %%

