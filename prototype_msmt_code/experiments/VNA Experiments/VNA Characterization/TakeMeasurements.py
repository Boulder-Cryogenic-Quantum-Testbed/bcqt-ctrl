# %%
"""
    Test implementation of VNA driver
"""

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
data_path = current_dir / "data" / script_filename / dstr 

data_path.mkdir(parents=True, exist_ok=True)

all_paths = [current_dir, experiment_path, msmt_code_path, src_path, driver_path, instr_path, data_path]

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

DefaultConfig = {
    "n_pts" : 20001,
    "f_start" : 4e9,
    "f_stop" : 8e9,
    "if_bandwidth" : 5000,
    "power" : -20,
    "edelay" : 0,
    "averages" : 10,
    "sparam" : 'all',
    
    "segment_type" : "linear",
}

# %%

def SetupMeasurement(VNA, data_dir=None):
    VNA.set_instr_params(DefaultConfig)
    VNA.get_instr_params()
    VNA.setup_s2p_measurement()
    
def TakeData(VNA):
    
    VNA.check_instr_error_queue()
    VNA.acquire_trace()
    
    freqs, magn_dB, phase_deg = VNA.return_data()
    
    #########################
    
    df, fig, axes = qh.plot_data_with_pandas(freqs, magn_dB, phase_deg=phase_deg)

    title_str = str(f"{Expt_Config["span"]/1e6:1.2f}MHz_span_{Expt_Config["averages"]}_avgs_{Expt_Config["if_bandwidth"]}_IFBW_{Expt_Config["power"]}_dBm")
    fig = axes["A"].get_figure()
    fig.suptitle(title_str, size=16)
    fig.tight_layout()
    
    # fc, span, ifbw, avg, power = Expt_Config["fc"], Expt_Config["span"], Expt_Config["if_bandwidth"], Expt_Config["averages"], Expt_Config["power"]

    #########################
    
    expt_no = len(data_dir.glob("*.csv")) + 1    # append number to end of filename
    filename = str(data_dir / f"{experiment_name}_{expt_no:03d}.csv")
    
    # want to add datetime to file, but must match dimensions, so just add it as first row... oops :)
    
    first_row = {col : val for col, val in zip(df.cols, [datetime.datetime()]*len(df.cols))}
    datetime_row = pd.DataFrame(first_row, index=["datetime.now()"])
    df2 = pd.concat([datetime_row, df.iloc[:]])
    df2.to_csv(filename)

    #########################

def Get_Data(VNA):
    
    
    
def Save_Measurement(VNA, experiment_name, csv_path, ):
    if not isinstance(data_dir, Path):
        data_dir = Path(data_dir)
    
    return 
#% %
