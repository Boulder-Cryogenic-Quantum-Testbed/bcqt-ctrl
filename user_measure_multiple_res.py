# %% -*- encoding: utf-8 -*-

# %%

%load_ext autoreload
%autoreload 2

import sys, time, os, glob
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

%run ../setup_measurement
# %run ../global_scripts/misc_functions
 
from janis_ctrl import measure_multiple_resonators
# from janis_ctrl import JanisCtrl
import misc_functions as mf
import helper_plot as hp
import plot_settings

%load_ext autoreload
%autoreload 2

# report date & time for measurement
dstr = datetime.today().strftime(r'%b_%d_%H%M')
print(f"\n\nCurrent timestamp: {dstr}")


# %%  establish measurement variables and parameters

base_directory = os.path.basename(os.getcwd())
sample_name = base_directory[6:]  # get rid of "LineX_ from the beginning of folder name"


# record all resonant frequencies from VNA
all_fcs = [4.708899, 5.082241, 5.450325, 5.847405, 
           6.205773, #6.591279, 
           6.968852]

# choose which resonators we will measure
fcs = all_fcs

# separate spans for each res, and 
# delays for 4 and 4 resonators
spans = [0.5, 0.5, 0.5, 1.0,
         0.5, 0.5, 0.5]

delays = [86.61]*len(fcs)

# create strings for the filename frequencies and data directories 
freq_strs = [f'{fc:.3f}GHz'.replace('.', 'p') for fc in fcs]
data_dirs = [f'data\\{dstr}\\{sample_name}_{freq_str}' for freq_str in freq_strs]

display(data_dirs)

# %% estimate time

high_powers_floats = np.arange(-40, -54, -7) + 3
med_powers_floats = np.arange(-54, -69, -7) + 3 
low_powers_floats = np.arange(-72, -79, -6) + 3
ultra_low_powers_floats = np.arange(-84, -91, -6)  + 3   

power_tuple_dict = {"HPow"  : (  len(high_powers_floats),      7  ), 
                    "MPow"  : (   len(med_powers_floats),      52  ),  
                    "LPow"  : (   len(low_powers_floats),      496  ),
                    "ULPow" : ( len(ultra_low_powers_floats),  1233  )
                    }

mf.estimate_resonator_runtime(power_tuple_dict, num_res=len(fcs))

conds = [len(fcs) == len(spans), len(fcs) == len(delays)]
for cond in conds:
    assert cond
    
print(high_powers_floats, med_powers_floats, low_powers_floats, ultra_low_powers_floats)

# %% high power scan
high_powers = [int(x) for x in high_powers_floats]

HPow_num_avgs, HPow_IFBW_kHz, HPow_num_pts = 3, 1, 301
 
tStart_high = time.time()   
measure_multiple_resonators(fcs, spans, delays, high_powers,
                             ifbw = HPow_IFBW_kHz, npts = HPow_num_pts,  
                             sample_name = sample_name,
                             data_dirs = data_dirs, 
                             Nf = int(HPow_num_pts*0.8), # num of fractions of 2pi for homophasal, each fraction is 2 pts
                             Navg_init = HPow_num_avgs, 
                             Noffres = int(HPow_num_pts*0.2),  # only used for segmented and linear, in case of linear: Npts = 5*Noffres
                             offresfraction = 0.1,  # only used for segmented
                             segment_option="segmented",
                            #  segment_option="linear",
                            #  segment_option="homophasal",
                            #  segment_option="hybrid",
                             filename_suffix="",
                             verbose = False,
                             wait_time = 3,  # wait 3 seconds in between every measurement to avoid locking up
                             )
tEnd_high = time.time()


mf.print_text_block(tStart_high, tEnd_high, HPow_num_avgs, HPow_num_pts, HPow_IFBW_kHz, num_powers=len(high_powers), num_resonators=len(fcs))

for res_folder in data_dirs: 
    fig, ax = mf.plot_whole_directory(res_folder, "*\\*.csv", plot_dir=res_folder, 
                                        max_rows=5, verbose=False, plot_min=True, save_plot=True, show_plot=True)
        


# %% medium power scan
med_powers = [int(x) for x in med_powers_floats]

MPow_num_avgs, MPow_IFBW_kHz, MPow_num_pts = 80, 1.0, 301

tStart_med = time.time()   
measure_multiple_resonators(fcs, spans, delays, med_powers,
                             ifbw = MPow_IFBW_kHz, npts = MPow_num_pts,  
                             sample_name = sample_name,
                             data_dirs = data_dirs,
                             Nf = int(MPow_num_pts*0.8), # num of fractions of 2pi for homophasal, each fraction is 2 pts
                             Navg_init = MPow_num_avgs, 
                             Noffres = int(MPow_num_pts*0.2),  # only used for segmented and linear, in case of linear: Npts = 5*Noffres
                             offresfraction = 0.1,  # only used for segmented
                             segment_option="segmented",
                            #  segment_option="linear",
                            #  segment_option="homophasal",
                            #  segment_option="hybrid",
                             filename_suffix="",
                             verbose = False,
                             wait_time = 3,  # wait 3 seconds in between every measurement to avoid locking up
                             )
tEnd_med = time.time()

for res_folder in data_dirs: 
    fig, ax = mf.plot_whole_directory(res_folder, "*\\*.csv", plot_dir=res_folder, 
                                        max_rows=5, verbose=False, plot_min=True, save_plot=True, show_plot=True)
        

mf.print_text_block(tStart_med, tEnd_med, MPow_num_avgs, MPow_num_pts, MPow_IFBW_kHz, num_powers=len(med_powers), num_resonators=len(fcs))


# %% low power scan
low_powers = [int(x) for x in low_powers_floats]

LPow_num_avgs, LPow_IFBW_kHz, LPow_num_pts = 2400, 1.0, 101

tStart_low = time.time()   
measure_multiple_resonators(fcs, spans, delays, low_powers,
                             ifbw = LPow_IFBW_kHz, npts = LPow_num_pts,  
                             sample_name = sample_name,
                             data_dirs = data_dirs,
                             Nf = int(LPow_num_pts*0.8), # num of fractions of 2pi for homophasal, each fraction is 2 pts
                             Navg_init = LPow_num_avgs, 
                             Noffres = int(LPow_num_pts*0.2),  # only used for segmented and linear, in case of linear: Npts = 5*Noffres
                             offresfraction = 0.1,  # only used for segmented
                             segment_option="segmented",
                            #  segment_option="linear",
                            #  segment_option="homophasal",
                            #  segment_option="hybrid",
                             filename_suffix="",
                             verbose = False,
                             wait_time = 3,  # wait 3 seconds in between every measurement to avoid locking up
                             )
tEnd_low = time.time()

for res_folder in data_dirs: 
    fig, ax = mf.plot_whole_directory(res_folder, "*\\*.csv", plot_dir=res_folder, 
                                        max_rows=5, verbose=False, plot_min=True, save_plot=True, show_plot=True)
        

mf.print_text_block(tStart_low, tEnd_low, LPow_num_avgs, LPow_num_pts, LPow_IFBW_kHz, num_powers=len(low_powers), num_resonators=len(fcs))


# %% ultra low power scan
ultra_low_powers = [int(x) for x in ultra_low_powers_floats]

ULPow_num_avgs, ULPow_IFBW_kHz, ULPow_num_pts = 6000, 1.0, 101

tStart_ultra_low = time.time()   
measure_multiple_resonators(fcs, spans, delays, ultra_low_powers,
                             ifbw = ULPow_IFBW_kHz, npts = ULPow_num_pts,  
                             sample_name = sample_name,
                             data_dirs = data_dirs,
                             Nf = int(ULPow_num_pts*0.8), # num of fractions of 2pi for homophasal, each fraction is 2 pts
                             Navg_init = ULPow_num_avgs, 
                             Noffres = int(ULPow_num_pts*0.2),  # only used for segmented and linear, in case of linear: Npts = 5*Noffres
                             offresfraction = 0.1,  # only used for segmented
                             segment_option="segmented",
                            #  segment_option="linear",
                            #  segment_option="homophasal",
                            #  segment_option="hybrid",
                             filename_suffix="",
                             verbose = False,
                             wait_time = 3,  # wait 3 seconds in between every measurement to avoid locking up
                             )
tEnd_ultra_low = time.time()

for res_folder in data_dirs: 
    fig, ax = mf.plot_whole_directory(res_folder, "*\\*.csv", plot_dir=res_folder, 
                                        max_rows=5, verbose=False, plot_min=True, save_plot=True, show_plot=True)
        

    
mf.print_text_block(tStart_ultra_low, tEnd_ultra_low, ULPow_num_avgs, ULPow_num_pts, ULPow_IFBW_kHz, num_powers=len(ultra_low_powers), num_resonators=len(fcs))


# %% time analysis


# # %%
# all_data_files = [glob.glob(f"data\\*{dstr}*\\{os.path.basename(res_folderpath)}\\*.csv") for res_folderpath in data_dirs]

# for single_res_files in all_data_files:
#     hp.plot_multiple_resonators(single_res_files, debug=False, show_plot=False)







