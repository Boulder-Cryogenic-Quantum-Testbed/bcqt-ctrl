"1"

# %%
"""
    Test implementation of VNA driver
"""
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time, sys

dstr = datetime.today().strftime("%m_%d_%I%M%p")
current_dir = Path(".")
script_filename = Path(__file__).stem

# lazy way to import modules - just append to path... TODO: fix :)
src_path = Path(r"..\..\..\src")
driver_path = src_path / "drivers"
data_path = current_dir / "data" / dstr / script_filename 
csv_path = data_path / "raw_csvs"
dcm_path = data_path / "dcm_fits"

# make sure all paths exist, then append to $PATH
for path in [src_path, driver_path, data_path, csv_path, dcm_path]:
    print(f"Checking if path exists:  ['{path}']")
    path = path.absolute()  # convert relative Path objs to absolutes
    print(f"     {str(path.exists()).upper()}")
    
    # ensure our data/fit storage paths exists
    if path.exists() is False and path in [csv_path.absolute(), data_path.absolute(), dcm_path.absolute()]:
        path.mkdir(parents=True, exist_ok=True)
        print(f"       ->  Created! Path now exists. [{path.exists() = }]")
    
    sys.path.append(str(path))

# %% import VNA driver

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


# %%

all_fcs =  [ 5.7338e9,
             5.7738e9,
             5.8226e9,
             5.8632e9,
            
             6.256667811e9,   # sucks, doesnt saturate
             6.306375544e9,   # saturates around -78 dBm
             6.360e9,
             6.416e9
            ]

## TODO: This should go in the ExptConfig object
Expt_Config = {
    "points" : 42,
    
    "fc" : all_fcs[5],
    
    "span" : 0.3e6,
    "if_bandwidth" : 1000,
    "power" : -78,
    # "edelay" : 86.0089, # res 4
    "edelay" : 72.095, # res 5
    
    "averages" : 2000,
    "sparam" : 'S21',
    # "segment_type" : "homophasal",
    "segment_type" : "hybrid",
    "Noffres" : 8
}


# %%

num_msmts = 20
all_powers = [-78]

if "all_dfs" not in locals().keys():
    all_dfs = {}

init_params = [None]*4
fit_results = {}

for idx in range(num_msmts):
    for power in all_powers:
        
        Expt_Config["power"] = power
        Expt_Config["segments"] = PNA_X.compute_homophasal_segments(**Expt_Config)
        
        PNA_X.set_instr_params(Expt_Config)
        PNA_X.get_instr_params()
        PNA_X.setup_measurement()
        PNA_X.check_instr_error_queue()
        PNA_X.acquire_trace()
        freqs, magn_dB, phase_deg = PNA_X.return_data()

        # freqs, magn, phase = PNA_X.take_single_trace(Expt_Config)

        df, fig, axes = plot_data_with_pandas(freqs, magn_dB, phase_deg=phase_deg)

        title_str = str(f"{dstr}_msmt_{idx}_{Expt_Config["power"]}_dBm").replace(".","p")
        fig = axes["A"].get_figure()
        fig.suptitle(title_str, size=16)
        fig.tight_layout()
        plt.show()

        Expt_Config["time_end"]  = datetime.now()
        
        all_dfs[title_str] = (df, Expt_Config) 
        
        filename = rf"{csv_path}\{title_str}.csv"
        df.to_csv(filename)


        Res_PowSweep_Analysis = DataAnalysis(None, dstr)

        
        print(f"Fitting {filename}")
        
        power = Expt_Config["power"]
        time_end = Expt_Config["time_end"]
        
        try:
            # output_params, conf_array, error, init, output_path
            params, conf_intervals, err, init1, fig = Res_PowSweep_Analysis.fit_single_res(data_df=df, save_dcm_plot=False, plot_title=filename, save_path=dcm_path)
        except Exception as e:
            print(f"Failed to plot DCM fit for {power} dBm -> {filename}")
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
        
        fit_results[filename] = (df, Expt_Config, parameters_dict, perc_errs)
        
        plt.show()
    
    all_param_dicts = {}
    for key, (df, Expt_Config, parameters_dict, perc_errs) in fit_results.items():
        all_param_dicts[key] = parameters_dict
        
    df_fit_results = pd.DataFrame.from_dict(all_param_dicts, orient="index").reset_index()
    # df_fit_results.drop("phi_err", axis="columns", inplace=True)  


    n_pows = df_fit_results["power"].nunique()
    powers = df_fit_results["power"].unique()
    
    for param in ["Q", "Qi", "Qc"]:
        
        fig2, axes2 = plt.subplots(n_pows, 1, figsize=(10, 4*n_pows))
        
        # handle case where we have single plot
        if type(axes2) != list or len(axes2) == 1:
            fig2.set_figheight(10)
            axes2 = [axes2]
        
        for ax, power in zip(axes2, powers):
            
            threshold_percentage = 12
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


            fig2.suptitle(f"Tracking {param} values over time, \n1 min/pt, \nthreshold = {threshold_percentage}% \n50 points per trace \n1 kHz IF_BW \n 1000 averages", size=18)
            fig2.tight_layout()
            
        plt.show()

    

# %% run analysis with scresonators


# init_params = [None]*4
# processed_data = { key : {"df": df, 
#                           "Expt_Config" : {**Expt_Config,  
#                                            "init_params" : init_params, 
#                                            "time_end" : time_end},  # add init_params and timestamp to config
#                           } 
#                   for key, (df, Expt_Config, time_end) in all_dfs.items() }

# Res_PowSweep_Analysis = DataAnalysis(processed_data, dstr)

    
# fit_results = {}
# for key, processed_data_dict in processed_data.items():
#     df, Expt_Config = processed_data_dict.values()
    
#     print(f"Fitting {key}")
    
#     power = Expt_Config["power"]
#     time_end = Expt_Config["time_end"]
    
#     save_dcm_plot = True if power <= -70 else False
    
#     if save_dcm_plot is not True:
#         continue
    
#     # output_params, conf_array, error, init, output_path
#     params, conf_intervals, err, init1, fig = Res_PowSweep_Analysis.fit_single_res(data_df=df, save_dcm_plot=False, plot_title=key, save_path=dcm_path)

#     # 1/Q = 1/Qi + cos(phi)/|Qc|
#     # 1/Qi = 1/Q - cos(phi)/|Qc|
#     # Qi = 1/(1/Q - cos(phi)/|Qc|)
    
#     Q, Qc, fc, phi = params
#     Q_err, Qi_err, Qc_err, Qc_Re_err, phi_err, fc_err = conf_intervals
    
#     Qi = 1/(1/Q - np.cos(phi)/np.abs(Qc))
    
#     params = [Q, Qi, Qc, fc, phi]
    
#     parameters_dict = {
#         "power" : power,
#         "Q" : Q,
#         "Q_err" : Q_err,
#         "Qi" : Qi,
#         "Qi_err" : Qi_err,
#         "Qc" : Qc,
#         "Qc_err" : Qc_err,
#         "fc" : fc,
#         "fc_err" : fc_err,
#         "phi" : phi,
#         "phi_err" : phi_err,
#     }

#     perc_errs = {
#         "Q_perc" : Q_err / Q,
#         "Qi_perc" : Qi_err / Qi,
#         "Qc_perc" : Qc_err / Qc,
#     }
    
#     fit_results[key] = (df, Expt_Config, parameters_dict, perc_errs)
    
#     plt.show()
        
    
# %%

all_param_dicts = {}
for key, (df, Expt_Config, parameters_dict, perc_errs) in fit_results.items():
    all_param_dicts[key] = parameters_dict
    
df_fit_results = pd.DataFrame.from_dict(all_param_dicts, orient="index").reset_index()
# df_fit_results.drop("phi_err", axis="columns", inplace=True)  


n_pows = df_fit_results["power"].nunique()
powers = df_fit_results["power"].unique()

# for param in ["Q", "Qi", "Qc"]:
    
#     fig, axes = plt.subplots(n_pows, 1, figsize=(10, 4*n_pows))
    
#     # handle case where we have single plot
#     if type(axes) != list:
#         fig.set_figheight(10)
#         axes = [axes]
        
#     for ax, power in zip(axes, powers):
        
#         threshold_percentage = 20
#         matching_powers = df_fit_results.loc[ df_fit_results["power"] == power]
        
#         # param = "Q"
#         dataset = matching_powers[param]
#         dataset_err = matching_powers[f"{param}_err"]
#         xvals = range(len(dataset))
        
#         avg, std = np.average(dataset), np.std(dataset)
        
#         failed_points = []
#         for x, pt, pt_err in zip(xvals, dataset, dataset_err):
#             perc_err = pt_err/pt * 100
            
#             if perc_err > threshold_percentage:
#                 info_str = f"idx{x}, {power=}: {param}={pt:1.1f} +/- {pt_err:1.1f}  ({perc_err=:1.1f}% > {threshold_percentage=}%)"
#                 failed_points.append((x, pt, pt_err, info_str))
#                 pt_color = 'r' 
#             else:
#                 pt_color = 'b'
                
#             ax.errorbar(x=x, y=pt, yerr=pt_err, color=pt_color, markersize=6, capsize=3)
        
#             ax.plot(x, pt, 'o', markersize=6, color=pt_color)
            
        
#         ax.axhline(avg, linestyle='--', linewidth=1, color='k', label=f"Average {param} = {avg:1.1e}")
        
#         ax.set_title(f"Power = {power} dBm")
#         # ax.set_yscale("log")
        
#         ax.set_xlabel("idx")
#         ax.set_ylabel(f"{param} Values")
#         ax.legend()


#         display(f"[{power = }] Failed Points: ")
#         for (x, pt, pt_err, info_str) in failed_points:
#             print(info_str)


#         fig.suptitle(f"Tracking {param} values over time, \n1 min/pt, \nthreshold = {threshold_percentage}% \n50 points per trace \n1 kHz IF_BW \n 1000 averages", size=18)
#         fig.tight_layout()
        
#     plt.show()

# %%
