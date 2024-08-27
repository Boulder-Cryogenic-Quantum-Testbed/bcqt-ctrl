
# %% 

"""
    organize_fits.py    
"""
# %%

%load_ext autoreload
%autoreload 2

%run setup_vna_measurement  

import sys, time, os, glob
import numpy as np
import matplotlib.pyplot as plt
import regex as re
import shutil

import misc_functions as mf
import helper_load as hl
import helper_misc as hm
import helper_fit as hf

# may need to open this file, save it and
# then close it for settings to apply

# %%

# TODO: create metadata file that shares these parameters with all user files in the folder

base_dir = os.path.basename(os.getcwd())

# %% to try and get rid of stupid style sheet issue
%run ../global_scripts/plot_settings

plt.subplots(1,1)
plt.plot()
plt.show()
plt.close()

# %%  copy all reports over to a new directory to filter manually

all_loss_tans = []
for line_num in range(6):
    # glob_str = f"Line{line_num}*/data/*/*GHz/all_fit_plots/tand*"
    glob_str = f"all_reports*"
    all_loss_tans.append(glob.glob(glob_str))
all_loss_tans = np.concatenate(all_loss_tans)

print(base_dir)
print(len(all_loss_tans))

# %%
for loss_tan_plot_filepath in all_loss_tans:
            
    line_folder, _, timestamp, res_folder, _, filename = loss_tan_plot_filepath.split("\\")
    
    freq_regex = r"\dp\d{3}"
    filename_freq = re.search(freq_regex, filename)[0]
    res_folder_freq = re.search(freq_regex, res_folder)[0]
    
    if filename_freq != res_folder_freq:
        # print(f"{filename_freq}    {res_folder_freq}")
        continue
    
    report_dir = f"all_reports\\by_resonator\\{res_folder}"
    final_report_dir = f"all_reports\\final"
    
    dst_filepath = f"{report_dir}\\{res_folder}_{timestamp}_tand.png"
    
    dst_dir = os.path.dirname(dst_filepath)
    
    all_dirs = [dst_dir, dst_dir, report_dir, dst_dir, final_report_dir]
    for dir_path in all_dirs:
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)
        
    print(f"{loss_tan_plot_filepath}\n --->   {dst_filepath}\n")
    shutil.copyfile(loss_tan_plot_filepath, dst_filepath)
    
# %% once finished filtering manually, this will move everything to a new directory

assert len(glob.glob(f"{final_report_dir}\\*")) == 0 
assert os.path.isdir(final_report_dir)

all_finished_plots = glob.glob(f"all_reports\\by_resonator\\*\\*.png")

for plot_filepath in all_finished_plots:
    plot_filename = os.path.basename(plot_filepath)
    dst_filepath = f"{final_report_dir}\\{plot_filename}"
    shutil.copyfile(plot_filepath, dst_filepath)

# %% now grab data files for the selected ones
# sort filenames by sample name

sample_names = ["MQC_anneal_01", "MQC_anneal_02", "MQC_thermal_02", "MQC_BOE_01", "MQC_thermal_01"]
organized_fits_dict = {}

for name in sample_names:
    selected_plots = glob.glob(f"{final_report_dir}\\{name}*")
    print(f"Found: {name} - {len(selected_plots)} files.")
    organized_fits_dict[name] = [os.path.basename(x) for x in selected_plots]
    
    
# %% move qiqcfc files into this directory from their original

for sample_name, filenames in organized_fits_dict.items():
    for filename in filenames:
        separations = os.path.basename(filename).split("_")
        freq_str = separations[3]
        timestamp = "_".join(separations[4:-1])
        print(sample_name, freq_str, timestamp)
        single_res_foldername = f"Line*\\*\\{timestamp}\\{sample_name}_{freq_str}\\qiqc*"
        glob_results = glob.glob(single_res_foldername)
        
        # keep only most recent one per directory
        if len(glob_results) > 1:
            glob_results = [glob_results[-1]]
        
        # print("  ", *glob_results, sep="  \n  ")
        
        for qiqcfile_path in glob_results:
            qiqcfile_name = os.path.basename(qiqcfile_path)
            dst_filepath = f"all_reports\\by_resonator_final\\{sample_name}_{freq_str}\\{qiqcfile_name}"
            shutil.copyfile(qiqcfile_path, dst_filepath)
            # print(f"  moving: {qiqcfile_name} to {dst_filepath}")



# %% inspect csv files with pandas
import pandas as pd 
# csv_dir = rf"E:\Cooldown_Data\Cooldown55\vna_measurements\all_reports\by_resonator_final\MQC_anneal_01_4p337GHz"
# csv_files = glob.glob(f"{csv_dir}\\*.csv")

csv_dir = rf"E:\Cooldown_Data\Cooldown55\vna_measurements\all_reports\by_resonator_final\*"
csv_files = glob.glob(f"{csv_dir}\\*.csv")

csv_dfs = {}
for csv_file in csv_files:
    res_name = os.path.basename(os.path.dirname(csv_file))
    qiqc_name = csv_file
    csv_name = (res_name, qiqc_name)
    df = pd.read_csv(csv_file, index_col=0)
    csv_dfs[csv_name] = df
    # display(df.head())

# %%
%run ../global_scripts/plot_settings

all_qiqc_files = glob.glob("Line*\\*\\*\\*\\*qiqc*.csv")

for csv_name, df in csv_dfs.items():
    
    res_name, qiqc_filepath = csv_name
    qiqc_file = os.path.basename(qiqc_filepath)
    file_save_dir = os.path.dirname(qiqc_filepath)
    
    # find original filepath so we can get timestamp from it
    print(f"Searching for {qiqc_file}")
    for filepath in all_qiqc_files:
        if qiqc_file in filepath and res_name in filepath:
            print(f"  {filepath}, \n  files = {len(temp)}")
            sample_name, _, timestamp, res_folder, csv_filename  = filepath.split("\\")
    
    # power_dBm = df["Power [dBm]"]
    navg = df["navg"]
    fc, fc_err = df["fc [GHz]"], df["fc error"]
    min_fc = min(fc)    
    
    Q, Q_perc_err = df["Q"], df["error"], 
    Qi, Qi_err = df["Qi"], df["Qi error"], 
    Qc, Qc_err = df["Qc"], df["Qc error"], 
    
    mosaic = "AA\nAA\nBB"
    fig, axes = plt.subplot_mosaic(mosaic)
    ax1, ax2 = axes["A"], axes["B"]
    # ax1, ax2, ax3 = axes
    
    ax1.errorbar(x=navg, y=Q, yerr=Q*Q_perc_err, fmt='bx', label="Q")
    ax1.errorbar(x=navg, y=Qi, yerr=Qi_err, fmt='gx', label="Qi")
    ax1.errorbar(x=navg, y=Qc, yerr=Qc_err, fmt='rx', label="Qc")
    fig.suptitle(f"{res_name}  -  Fit Results")
    ax1.set_title(f"{qiqc_file}          \n\nPower Dependent Q, Qi, Qc")
    ax1.set_xscale("log")
    ax1.set_xlabel("$n_{avg}$", size=16)
    ax1.set_ylabel("Q Value [a.u.]")
    ax1.legend()
    
    ax2.errorbar(x=navg, y=(fc-min_fc)*1e6, yerr=fc_err, fmt='rx', label=f"$f_c^{"{min}"} = {min_fc:1.6f}$")
    ax2.set_title("\n\nPower Dependence - Frequency Shift")
    ax2.set_ylabel("Freq Shift [MHz]")
    ax2.set_xscale("log")
    ax2.set_xlabel("$n_{avg}$")
    ax2.legend()
    
    fig.tight_layout()
    plt.savefig(f"{file_save_dir}\\qiqcfc_results_{res_folder}_{timestamp}")
    plt.close()


# %% if you need to remove all the fit result png's

# files = glob.glob(f"all_reports\\by_resonator_final\\*\\*fit_results*.png")
# files = glob.glob(f"all_reports\\by_resonator_final\\*\\qiqc*.csv")
# for file in files:
#     os.remove(file)

