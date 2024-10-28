# %%
"""
    Test implementation of DataProcessor (eventually)
"""
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time, sys

current_dir = Path(".")
script_filename = Path(__file__).stem.replace("Analysis","")  # match measurement script

# lazy way to import modules - just append to path... TODO: fix :)
src_path = Path(r"..\..\src")
driver_path = src_path / "drivers"
data_dir = current_dir / "data"

assert data_dir.exists()

# make sure all paths exist, then append to $PATH
for path in [src_path, driver_path]:
    path = path.absolute()
    assert path.exists()
    sys.path.append(str(path))

# %% load data if experiment wasn't done

from DataAnalysis import DataAnalysis
import regex as re

data_path = "10_17_1230PM" 
csv_path = (data_dir / data_path / "VNA_PowerSweep") / "raw_csvs"
dcm_path = (data_dir / data_path / "VNA_PowerSweep") / "dcm_fits"

all_dfs = {}
for csv_filepath in csv_path.glob("*.csv"):
    csv_filename = csv_filepath.stem
    # print(f"Loading file:  '{csv_filename}'")
    
    df = pd.read_csv(csv_filepath, index_col=0)
    
    basic_configs = {
        "power" : float(re.search(r"-\d{1,2}_dBm", csv_filename).captures()[0].replace("_dBm","")),
        "dstr" : re.search(r"\d{1,2}_\d{1,2}_\d{1,4}[A,P]M", csv_filename).captures()[0],
        "time_end" : datetime.now(),
        "idx" : int(re.search(r"msmt_\d{1,2}", csv_filename).captures()[0].replace("msmt_","")),
    }
    
    all_dfs[csv_filename] = {"df" : df, "configs" : basic_configs, "time_end" : basic_configs["time_end"]} # TODO: DataProcessor needs to save config & init params

    dstr = basic_configs["dstr"]  # overwrite dstr from first cell

assert len(all_dfs) != 0
print(f"all_dfs populated with > {len(all_dfs)} < csv files.")

# %% helper methods

from quick_helpers import unpack_df, plot_data_with_pandas

# %% plot all traces in their own plot
for key, (df, Expt_Config, time_end) in all_dfs.items():
    
    freqs, magn_dB, phase_rad = unpack_df(df)
    df, fig, axes = plot_data_with_pandas(freqs, magn_dB, phase_rad=phase_rad)

    fig = axes["A"].get_figure()
    fig.suptitle(key, size=16)
    fig.tight_layout()
    plt.show()

# plot all complex circles by themselves
# chosen_plots = [x for x in all_dfs.keys() if '-75_dBm' in x]
chosen_plots = list(all_dfs.keys())
fig, axes = plt.subplots(len(chosen_plots), 1, figsize=(12, 3*len(chosen_plots)))
for idx, key in enumerate(chosen_plots):
    df, Expt_Config, time_end = all_dfs[key]
    ax = axes[idx]
    
    freqs, magn_dB, phase_rad = unpack_df(df)
    df, _, _ = plot_data_with_pandas(freqs, magn_dB, phase_rad=phase_rad, ax=ax)
    
    # ax.set_aspect("equal")
    ax.set_title(f"{key[-7:]}")
    
fig.tight_layout()


# plot all complex circles on one plot
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
for key, (df, Expt_Config, time_end) in all_dfs.items():
    if "-75" not in key:
        continue
    
    if "200" not in key:
        label = None
    else:
        label = key
    
    freqs, magn_dB, phase_rad = unpack_df(df)
    df, _, _ = plot_data_with_pandas(freqs, magn_dB, phase_rad=phase_rad, ax=ax, label=label)
    
    
# ax.set_title(key)
plt.legend()
fig.tight_layout()
plt.show()
    
    
# %% run analysis with scresonators
from DataAnalysis import DataAnalysis


init_params = [None]*4
processed_data = all_dfs
# processed_data = { key : {"df": df,     
#                           "Expt_Config" : {**Expt_Config,  
#                                            "init_params" : init_params, 
#                                            "time_end" : time_end  
#                                            }, 
#                           }
                        #   for key, (df, Expt_Config) in all_dfs.items() } 

Res_PowSweep_Analysis = DataAnalysis(processed_data, dstr)

    
fit_results = {}
for key, processed_data_dict in processed_data.items():
    df, Expt_Config, time_end = processed_data_dict.values()
    
    print(f"Fitting {key}")
    
    power = Expt_Config["power"]
    time_end = Expt_Config["time_end"]
    
    
    try:
        # output_params, conf_array, error, init, output_path
        params, conf_intervals, err, init1, fig = Res_PowSweep_Analysis.fit_single_res(data_df=df, save_dcm_plot=False, plot_title=key, save_path=dcm_path)
    except Exception as e:
        print(f"Failed to fit {key}  --> \n   {e}")
        continue
    
    # 1/Q = 1/Qi + cos(phi)/|Qc|
    # 1/Qi = 1/Q - cos(phi)/|Qc|
    # Qi = 1/(1/Q - cos(phi)/|Qc|)
    
    Q, Qc, fc, phi = params
    Q_err, Qi_err, Qc_err, Qc_Re_err, phi_err, fc_err = conf_intervals
    
    Qi = 1/(1/Q - np.cos(phi)/np.abs(Qc))
    
    params = [Q, Qi, Qc, fc, phi]
    
    parameters_dict = {
        "power" : power,
        "Q" : Q,
        "Q_err" : Q_err,
        "Qi" : Qi,
        "Qi_err" : Qi_err,
        "Qc" : Qc,
        "Qc_err" : Qc_err,
        "fc" : fc,
        "fc_err" : fc_err,
        "phi" : phi,
        "phi_err" : phi_err,
    }

    perc_errs = {
        "Q_perc" : Q_err / Q,
        "Qi_perc" : Qi_err / Qi,
        "Qc_perc" : Qc_err / Qc,
    }
    
    fit_results[key] = (df, Expt_Config, parameters_dict, perc_errs)
    
    plt.show()
    
# %%

all_param_dicts = {}
for key, (df, Expt_Config, parameters_dict, perc_errs) in fit_results.items():
    all_param_dicts[key] = parameters_dict
    
df_fit_results = pd.DataFrame.from_dict(all_param_dicts, orient="index").reset_index()
# df_fit_results.drop("phi_err", axis="columns", inplace=True)  


n_pows = df_fit_results["power"].nunique()
powers = df_fit_results["power"].unique()

for param in ["Q", "Qi", "Qc"]:
    
    fig, axes = plt.subplots(n_pows, 1, figsize=(10, 4*n_pows))
    
    # handle case where we have single plot
    if type(axes) != np.ndarray:
        fig.set_figheight(10)
        axes = [axes]
        
    for ax, power in zip(axes, powers):
        
        threshold_percentage = 20
        matching_powers = df_fit_results.loc[ df_fit_results["power"] == power]
        
        # param = "Q"
        dataset = matching_powers[param]
        dataset_err = matching_powers[f"{param}_err"]
        xvals = range(len(dataset))
        
        avg, std = np.average(dataset), np.std(dataset)
        
        failed_points = []
        for x, pt, pt_err in zip(xvals, dataset, dataset_err):
            perc_err = pt_err/pt * 100
            
            if perc_err > threshold_percentage:
                info_str = f"idx{x}, {power=}: {param}={pt:1.1f} +/- {pt_err:1.1f}  ({perc_err=:1.1f}% > {threshold_percentage=}%)"
                failed_points.append((x, pt, pt_err, info_str))
                pt_color = 'r' 
            else:
                pt_color = 'b'
                
            ax.errorbar(x=x, y=pt, yerr=pt_err, color=pt_color, markersize=6, capsize=3)
        
            ax.plot(x, pt, 'o', markersize=6, color=pt_color)
            
        
        ax.axhline(avg, linestyle='--', linewidth=1, color='k', label=f"Average {param} = {avg:1.1e}")
        
        ax.set_title(f"Power = {power} dBm")
        # ax.set_yscale("log")
        
        ax.set_xlabel("idx")
        ax.set_ylabel(f"{param} Values")
        ax.legend()


        display(f"[{power = }] Failed Points: ")
        for (x, pt, pt_err, info_str) in failed_points:
            print(info_str)


        fig.suptitle(f"Tracking {param} values over time, \n1 min/pt, \nthreshold = {threshold_percentage}% \n50 points per trace \n1 kHz IF_BW \n 1000 averages", size=18)
        fig.tight_layout()
        
    plt.show()

# %%
