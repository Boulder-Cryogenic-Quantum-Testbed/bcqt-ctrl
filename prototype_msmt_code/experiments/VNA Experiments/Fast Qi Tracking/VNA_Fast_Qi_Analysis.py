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
script_filename = Path(__file__).stem
data_dir = current_dir / "data" / "VNA_Fast_Qi_Measurement"
msmt_code_path = Path(r"../../..").resolve()
src_path = msmt_code_path / "src"

sys.path.append(str(msmt_code_path))
sys.path.append(str(msmt_code_path / "Experiments" / "VNA Experiments"))



# %% load data if experiment wasn't done

from src.DataAnalysis import DataAnalysis
import regex as re

# %%

# four stars -> "./data/VNA_Fast_Qi_Measurement/10_31_0950PM/raw_csvs"
all_datasets = [x for x in (current_dir / "data" ).glob("*/*/*/*") if "_parsed" not in str(x)]

for dataset_path in all_datasets:
    parsed_dataset_path = Path(f"{dataset_path}_parsed")
    parsed_dataset_path.mkdir(exist_ok=True)
    csv_list = sorted([x for x in dataset_path.glob("*.csv")])
    print(f"{len(csv_list)} csv files in {dataset_path.name}")

    csv_files_parsed = len(list(parsed_dataset_path.glob("*.csv")))
    csv_files_unparsed = len(list(dataset_path.glob("*.csv")))
    
    # check if all data has been parsed
    if csv_files_parsed == csv_files_unparsed:
        print(f"All data has been parsed, delete old directory now.")
    else:
        print(f"{csv_files_unparsed - csv_files_parsed} new files since last parse.")

dcm_path = (parsed_dataset_path / ".." / ".." / "dcm_fits").resolve()
csv_path = (parsed_dataset_path / "..").resolve()

# %% read all csvs and parse info from filename and csv contents

all_dfs = {}
for csv_filepath in csv_list:
    csv_filename = csv_filepath.stem
    
    # lazy lazy lazy
    set_str, msmt_str, _, span_str, _, power_str, avgs, _ = csv_filename.split("_")
    msmt_num = int(re.search(r"\d{1,6}", msmt_str)[0])
    
    new_filename = f"{set_str}_Msmt{msmt_num:05.0f}_span_{span_str}_power_{power_str}_{avgs}_avgs.csv"
    new_filepath = parsed_dataset_path / new_filename

    
    if new_filepath.exists():
        # print(f"Skipping {set_str}-{msmt_num}, parsed file exists")
        pass
    else:
        df = pd.read_csv(csv_filepath, index_col=0)
        
        basic_configs = {
            "filename" : csv_filename,
            "filepath" : csv_filepath.parent,
            "power" : int(power_str.replace("dBm","")),
            "span" : int(span_str.replace("kHz",""))*1e3,
            "set_num" : int(set_str[-1]),
            "msmt_num" : msmt_num,
            "avgs" : int(avgs),
            "dstr" : datetime.fromisoformat(df.iloc[0,0]).strftime("%m_%d_%I%M%p"),
            "timestamp" : datetime.fromisoformat(df.iloc[0,0]),
        }
        
        print(f"~~~Saving parsed csv for {set_str}-{msmt_num}")
        df.to_csv(new_filepath)

# %% go through parsed folder

csv_list = sorted([x for x in parsed_dataset_path.glob("*.csv")])

# copy pasted from above
all_dfs = {}
for csv_filepath in csv_list:
    csv_filename = csv_filepath.stem
    
    # lazy lazy lazy
    set_str, msmt_str, _, span_str, _, power_str, avgs, _ = csv_filename.split("_")
    msmt_num = int(re.search(r"\d{1,6}", msmt_str)[0])
    set_msmt = f"{set_str}-{msmt_num:05.0f}"
    
    new_filename = f"{set_str}_Msmt{msmt_num:05.0f}_span_{span_str}_power_{power_str}_{avgs}_avgs.csv"
    new_filepath = parsed_dataset_path / new_filename

    df = pd.read_csv(csv_filepath, index_col=0)
    
    basic_configs = {
        "set_msmt" : set_msmt,
        "filename" : csv_filename,
        "filepath" : csv_filepath.parent,
        "power" : int(power_str.replace("dBm","")),
        "span" : int(span_str.replace("kHz",""))*1e3,
        "set_num" : int(set_str[-1]),
        "msmt_num" : msmt_num,
        "avgs" : int(avgs),
        "dstr" : datetime.fromisoformat(df.iloc[0,0]).strftime("%m_%d_%I%M%p"),
        "timestamp" : datetime.fromisoformat(df.iloc[0,0]),
    }
    
    # we can remove the first row of datetime now
    df.drop(index="datetime.now()", inplace=True)
    
     # TODO: DataProcessor needs to save config & init params
    all_dfs[set_msmt] = {"df" : df, "configs" : basic_configs}

assert len(all_dfs) != 0
print(f"all_dfs populated with > {len(all_dfs)} < csv files.")


# %% helper methods

from quick_helpers import unpack_df, plot_data_with_pandas

make_plots = False

# %% plot all traces in their own plot
if make_plots is True and len(all_dfs) <= 25:
    for key, (df, csv_configs) in all_dfs.items():
        
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
        df, csv_configs = all_dfs[key]
        ax = axes[idx]
        
        freqs, magn_dB, phase_rad = unpack_df(df)
        df, _, _ = plot_data_with_pandas(freqs, magn_dB, phase_rad=phase_rad, ax=ax)
        
        # ax.set_aspect("equal")
        ax.set_title(f"{key[-7:]}")
        
    fig.tight_layout()


    # plot all complex circles on one plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    for key, (df, csv_configs) in all_dfs.items():
        freqs, magn_dB, phase_rad = unpack_df(df)
        df, _, _ = plot_data_with_pandas(freqs, magn_dB, phase_rad=phase_rad, ax=ax, label=label)
        
        
    # ax.set_title(key)
    plt.legend()
    fig.tight_layout()
    # plt.show()
    plt.close()
    
        
    
# %% run analysis with scresonators
from src.DataAnalysis import DataAnalysis
    

init_params = [None]*4
processed_data = all_dfs
# processed_data = { key : {"df": df,     
#                           "csv_configs" : {**csv_configs,  
#                                            "init_params" : init_params, 
#                                            "time_end" : time_end  
#                                            }, 
#                           }
                        #   for key, (df, csv_configs) in all_dfs.items() } 

Res_PowSweep_Analysis = DataAnalysis(processed_data)

fit_results = {}
for key, processed_data_dict in processed_data.items():
    df, csv_configs = processed_data_dict.values()
    
    print(f"Fitting {key}")
    
    power = csv_configs["power"]
    time_end = csv_configs["timestamp"]    
    
    # try:
    # output_params, conf_array, error, init, output_path
    params, conf_intervals, err, init1, fig = Res_PowSweep_Analysis.fit_single_res(data_df=df, save_dcm_plot=False, plot_title=key, save_path=dcm_path)
    # except Exception as e:
    #     print(f"Failed to fit {key}  --> \n   {e}")
    #     continue
    
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
    
    fit_results[key] = (df, csv_configs, parameters_dict, perc_errs)
    
    if make_plots is True:
        plt.show()
    else:
        plt.close()
        
    break
    
# %% get all fit results and organize into a single 

all_param_dicts = {}
for key, (df, csv_configs, parameters_dict, perc_errs) in fit_results.items():
    all_param_dicts[key] = parameters_dict
    
df_fit_results = pd.DataFrame.from_dict(all_param_dicts, orient="index").reset_index()
# df_fit_results.drop("phi_err", axis="columns", inplace=True)  


n_pows = df_fit_results["power"].nunique()
powers = df_fit_results["power"].unique()

for param in ["Q", "Qi", "Qc"]:
    
    if make_plots is True:
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
