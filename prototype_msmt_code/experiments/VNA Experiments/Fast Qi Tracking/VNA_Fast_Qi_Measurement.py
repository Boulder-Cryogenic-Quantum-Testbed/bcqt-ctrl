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

all_fcs =  [ #5.733901e9,
             #5.773425e9,
             #5.822726e9,
             5.863280e9,
            
             #6.256667811e9,   # sucks, doesnt saturate
             #6.306375544e9,   # saturates around -78 dBm
             #6.360336e9,
             #6.417038e9
            ]


Expt_Config = {
    "points" : 100,
    # "fc" : all_fcs[1],
    "span" : 0.1e6,
    "if_bandwidth" : 500,
    "power" : -72,
    "edelay" : 76.36,
    "averages" : 2,
    "sparam" : 'S21',
    
    # "segment_type" : "homophasal",
    "segment_type" : "hybrid",
    
    "Noffres" : 15
}


# %%

sets = 100
num_msmt_per_set = 10000

start_time = time.time()

# measure all resonators once for X sets, with Y traces/resonator in each set
for set_idx in range(sets): 
    
    for res_idx, fc in enumerate(all_fcs):
    
        Expt_Config["segments"] = PNA_X.compute_homophasal_segments(**Expt_Config)
        Expt_Config["fc"] = fc
        
        PNA_X.set_instr_params(Expt_Config)
        PNA_X.get_instr_params()
        PNA_X.setup_measurement()
        
        fit_results = {}
        res_start_time = time.time()
        
        for idx in range(num_msmt_per_set):
        
            Expt_Config["time_start"] = datetime.now()
            
            # PNA_X.setup_measurement()
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

            SingleFit = DataAnalysis(None, dstr)

            fc, span, ifbw, avg, power = Expt_Config["fc"], Expt_Config["span"], Expt_Config["if_bandwidth"], Expt_Config["averages"], Expt_Config["power"]
            name = f"Set{set_idx+1}_Msmt{idx+1}_span_{span/1e3:1.0f}kHz_power_{power}dBm_{avg}_avgs"

            res_dir = csv_path / f"Res{res_idx}_{fc/1e6:1.0f}MHz"
            res_dir.mkdir(parents=True, exist_ok=True)
            filename = str(res_dir / f"{name}.csv")
            
            # want to add datetime, but must match dimensions, so just add it as first row... oops :)
            datetime_row = pd.DataFrame({"Frequency" : datetime.now(), "S21 [dB]" : datetime.now(), "Phase [rad]" : datetime.now()}, index=["datetime.now()"])
            df2 = pd.concat([datetime_row, df.iloc[:]])
            df2.to_csv(filename)

        #     try:
        #         # output_params, conf_array, error, init, output_path
        #         params, conf_intervals, err, init1, fig = SingleFit.fit_single_res(data_df=df, save_dcm_plot=False, plot_title=filename, save_path=dcm_path)
        #     except Exception as e:
        #         print(f"Failed to plot DCM fit for {power} dBm -> {filename = }")
        #         print(f"Error: {e}")
        #         continue

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
        #         "meas_time" : Expt_Config["time_start"]
        #     }

        #     perc_errs = {
        #         "Q_perc" : Q_err / Q,
        #         "Qi_perc" : Qi_err / Qi,
        #         "Qc_perc" : Qc_err / Qc,
        #     }
            
        
            # fit_results[filename] = (df, Expt_Config, parameters_dict, perc_errs)
            
        #     plt.close()
            
        
        # all_param_dicts = {}
        # for key, (df, Expt_Config, parameters_dict, perc_errs) in fit_results.items():
        #     all_param_dicts[key] = parameters_dict
        
        # fit_dict_name = str((csv_path / "..").resolve() / f"{fc/1e3:1.0f}MHz_all_fit_results.csv")
            
        # df_fit_results = pd.DataFrame.from_dict(all_param_dicts, orient="index").reset_index()
        # df_fit_results.to_csv(fit_dict_name)
        # df_fit_results.drop("phi_err", axis="columns", inplace=True)  

        end_time = time.time()
        
        resonator_elapsed_time = end_time - res_start_time
        total_elapsed_time = end_time - start_time 
        display(f"Finished Resonator [{res_idx}/{len(all_fcs)}]! \nSingle resonator elapsed time: {resonator_elapsed_time/60:1.1f} minutes.\nTotal measurement resonator elapsed time: {total_elapsed_time/60:1.1f} minutes.")
            



# %%  plotting 
    


# %% run analysis with scresonators


# def fake():
#         Res_PowSweep_Analysis = DataAnalysis(None, dstr)

        
#         print(f"Fitting {filename}")
        
#         power = Expt_Config["power"]
#         time_end = Expt_Config["time_end"]
        
#         try:
#             # output_params, conf_array, error, init, output_path
#             params, conf_intervals, err, init1, fig = Res_PowSweep_Analysis.fit_single_res(data_df=df, save_dcm_plot=False, plot_title=filename, save_path=dcm_path)
#         except Exception as e:
#             print(f"Failed to plot DCM fit for {power} dBm -> {filename}")
#             continue
        
#         # 1/Q = 1/Qi + cos(phi)/|Qc|
#         # 1/Qi = 1/Q - cos(phi)/|Qc|
#         # Qi = 1/(1/Q - cos(phi)/|Qc|)
        
#         Q, Qc, fc, phi = params
#         Q_err, Qi_err, Qc_err, Qc_Re_err, phi_err, fc_err = conf_intervals
        
#         Qi = 1/(1/Q - np.cos(phi)/np.abs(Qc))
        
#         params = [Q, Qi, Qc, fc, phi]
        
#         parameters_dict = {
#             "power" : power,
#             "Q" : Q,
#             "Q_err" : Q_err,
#             "Qi" : Qi,
#             "Qi_err" : Qi_err,
#             "Qc" : Qc,
#             "Qc_err" : Qc_err,
#             "fc" : fc,
#             "fc_err" : fc_err,
#             "phi" : phi,
#             "phi_err" : phi_err,
#         }

#         perc_errs = {
#             "Q_perc" : Q_err / Q,
#             "Qi_perc" : Qi_err / Qi,
#             "Qc_perc" : Qc_err / Qc,
#         }
        
#         fit_results[filename] = (df, Expt_Config, parameters_dict, perc_errs)
        
#         plt.show()
    
#     all_param_dicts = {}
#     for key, (df, Expt_Config, parameters_dict, perc_errs) in fit_results.items():
#         all_param_dicts[key] = parameters_dict
        
#     df_fit_results = pd.DataFrame.from_dict(all_param_dicts, orient="index").reset_index()
#     # df_fit_results.drop("phi_err", axis="columns", inplace=True)  


#     n_pows = df_fit_results["power"].nunique()
#     powers = df_fit_results["power"].unique()
    
#     for param in ["Q", "Qi", "Qc"]:
        
#         fig2, axes2 = plt.subplots(n_pows, 1, figsize=(10, 4*n_pows))
        
#         # handle case where we have single plot
#         if type(axes2) != list or len(axes2) == 1:
#             fig2.set_figheight(10)
#             axes2 = [axes2]
        
#         for ax, power in zip(axes2, powers):
            
#             threshold_percentage = 12
#             matching_powers = df_fit_results.loc[ df_fit_results["power"] == power]
            
#             # param = "Q"
#             dataset = matching_powers[param]
#             dataset_err = matching_powers[f"{param}_err"]
#             xvals = range(len(dataset))
            
#             avg, std = np.average(dataset), np.std(dataset)
            
#             failed_points = []
#             for x, pt, pt_err in zip(xvals, dataset, dataset_err):
#                 perc_err = pt_err/pt * 100
                
#                 if perc_err > threshold_percentage:
#                     info_str = f"idx{x}, {power=}: {param}={pt:1.1f} +/- {pt_err:1.1f}  ({perc_err=:1.1f}% > {threshold_percentage=}%)"
#                     failed_points.append((x, pt, pt_err, info_str))
#                     pt_color = 'r' 
#                 else:
#                     pt_color = 'b'
                    
#                 ax.errorbar(x=x, y=pt, yerr=pt_err, color=pt_color, markersize=6, capsize=3)
            
#                 ax.plot(x, pt, 'o', markersize=6, color=pt_color)
                
            
#             ax.axhline(avg, linestyle='--', linewidth=1, color='k', label=f"Average {param} = {avg:1.1e}")
            
#             ax.set_title(f"Power = {power} dBm")
#             # ax.set_yscale("log")
            
#             ax.set_xlabel("idx")
#             ax.set_ylabel(f"{param} Values")
#             ax.legend()


#             display(f"[{power = }] Failed Points: ")
#             for (x, pt, pt_err, info_str) in failed_points:
#                 print(info_str)


#             fig2.suptitle(f"Tracking {param} values over time, \n1 min/pt, \nthreshold = {threshold_percentage}% \n50 points per trace \n1 kHz IF_BW \n 1000 averages", size=18)
#             fig2.tight_layout()