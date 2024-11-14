# %%
"""
    Test implementation of VNA driver
"""

# %load_ext autoreload
# %autoreload 2

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
    "n_points" : 1001,
    "f_start" : 2e9,
    "f_stop" : 10e9,
    "if_bandwidth" : 1000,
    "power" : -50,
    "edelay" : 0,
    "averages" : 2,
    # "sparam" : ['S12'],  # do not want to measure S12 in VNA two port mode
    "sparam" : ['S11', 'S22', 'S21', 'S12'],  # do not want to measure S12 in VNA two port mode
    # "sparam" : ['S12', 'S21'],  # do not want to measure S12 in VNA two port mode
    
    "segment_type" : "linear",
}

PNA_X = VNA_Keysight(VNA_Keysight_InstrConfig, debug=True)

PNA_X.set_instr_params(VNA_Keysight_InstrConfig)

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

def Archive_Data(VNA, s2p_df:pd.DataFrame, meas_name:str, expt_category:str = '', save_dir:str = "./data", all_axes=None):
    # check if save_dir is a path or string
    if not isinstance(save_dir, Path):
        save_dir = Path(save_dir).absolute()
        
    timestamp = datetime.today().strftime("%m_%d_%I%M%p")
    # file_dir = save_dir / expt_category / meas_name / timestamp
    file_dir = save_dir / expt_category / meas_name
    
    # check if file_dir exists
    if not file_dir.exists():
        VNA.print_console(f"Creating category {expt_category}")
        VNA.print_console(f"    under save directory {save_dir}")
        file_dir.mkdir(exist_ok=True, parents=True)
    
    # append number to end of filename and save to csv
    expt_no = len(list(file_dir.glob("*.csv"))) + 1    
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
            title = fig.get_suptitle().replace(" - ","_") + f"{expt_no:03d}.png"
            fig_filename = file_dir / title
            fig.tight_layout()
            fig.savefig(fig_filename)
            plt.show()
            print(fig_filename)
            
    
    return filename, final_path.parent

def Load_CSV(filepath):
    """
        convenience method - take csv filepath and return df, magns, phases, and freqs
    """
    
    if not isinstance(filepath, Path):
        filepath = Path(filepath).absolute()
    
    df = pd.read_csv(filepath, index_col=0)
    
    all_magns, all_phases = [], []
    
    for col in df:
        if "magn" in col.lower():
            series = df[col].values[1:]
            all_magns.append(series)
        if "phase" in col.lower():
            series = df[col].values[1:]
            all_phases.append(series)
        if "freq" in col.lower():
            freqs = df[col].values[1:]
    
    return df, all_magns, all_phases, freqs

    
    #########################
# %%

Measurement_Configs = {
    "f_start" : 2e9,
    "f_stop" : 10e9,
    "n_pts" : 10001,
    "if_bw" : 1000,
    "power" : -50,
    "averages" : 2,
    "sparam" : ['S11', 'S22', 'S21', 'S12'], 
    
    # by default, sparam = 'all', edelay = 0, averages = 10
}



# PNA_X.check_instr_error_queue()
# PNA_X.add_kwargs_and_filter_configs()

"""
    ResetVNA
        Resets the VNA to default configs, then passes any kwargs to the configs,
        and then sets up the s2p measurement.
"""
PNA_X.set_instr_params(Measurement_Configs)
PNA_X.get_instr_params()


"""
    PerformMeasurement
        instruct VNA to do error checks, add configs, and perform measurement
"""
PNA_X.setup_s2p_measurement()
PNA_X.run_measurement()


"""
    DownloadMeasurement
    Should take zero time, only asks VNA to send the data it has, does not measure
        df = dataframe of scattering parameters

"""
s2p_df = PNA_X.return_data_s2p()
all_axes = qh.plot_s2p_df(s2p_df)

"""
    Archive_Data(VNA, s2p_df, meas_name, expt_category, save_dir="./data", all_axes=None):
    return filename, final_path.parent
"""

expt_category = "HEMT_RT_Amplifiers"
meas_name = "HEMT_B"

filename, filepath = Archive_Data(PNA_X, s2p_df, meas_name=meas_name, expt_category=expt_category, all_axes=all_axes)



# %% manually load data

load_filepath = r"C:\Users\Lehnert Lab\GitHub\bcqt-ctrl\prototype_msmt_code\experiments\VNA Experiments\VNA Characterization\data_jin_characterization\S12_MXC\001_Input_Line24_to_Attenuators\Input_Line24_to_Attenuators_001.csv"

loaded_df, all_magns, all_phases, freqs = Load_CSV(load_filepath)

all_axes = qh.plot_s2p_df(loaded_df, plot_complex=True, track_min=False, title=meas_name, do_edelay_fit=False)
