# %%

%load_ext autoreload
%autoreload 2

import sys, time, os, glob
import numpy as np
import matplotlib.pyplot as plt
import regex as re

import misc_functions as mf
import helper_plot as hp
import plot_settings
 
from datetime import datetime
from janis_ctrl import measure_multiple_resonators

%run ../setup_measurement

# %%  establish measurement variables and parameters

# record date & time for measurement
dstr = datetime.today().strftime(r'%b_%d_%H%M')
print(f"\n\nCurrent folder timestamp: {dstr}\n\n")

# use folder name to get device name
base_dir = os.path.basename(os.getcwd())
line_num, sample_name = re.split(r"_", base_dir, maxsplit=1)  # use regex to split only at first _

# use VNA to record center frequencies
all_fcs = [4.708899, 5.082241, 
           5.450325, 5.847405, 
           6.205773, #6.591279, 
           6.968852]

# choose which resonators to measure, and spans/delays
fcs = all_fcs
# fcs = all_fcs[-1]   

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

power_tuple_dict = { 
                 # tuple name :  (num of power,  measurement duration in seconds)
                    "HPow"  : (  len(high_powers_floats),      7  ), 
                    "MPow"  : (   len(med_powers_floats),      52  ),  
                    "LPow"  : (   len(low_powers_floats),      496  ),
                    "ULPow" : ( len(ultra_low_powers_floats),  1233  )
                    }

mf.estimate_resonator_runtime(power_tuple_dict, num_res=len(fcs))

# use assert to make sure every freq has a span & delay
assert len(fcs) == len(spans), f"length of fcs =/= spans  ({len(fcs)} =/= {len(spans)})"
assert len(fcs) == len(delays), f"length of fcs =/= delays ({len(fcs)} =/= {len(delays)} )"
    
print(high_powers_floats, med_powers_floats, low_powers_floats, ultra_low_powers_floats)

# %% high power scan
high_powers = [int(x) for x in high_powers_floats]

HPow_num_avgs, HPow_IFBW_kHz, HPow_num_pts = 3, 1, 301
 
tStart_high = time.time()   
measure_multiple_resonators(fcs, spans, delays, high_powers, 
                            ifbw = HPow_IFBW_kHz, npts = HPow_num_pts,  
                            sample_name = sample_name, data_dirs = data_dirs, 
                            Nf = int(HPow_num_pts*0.8), Navg_init = HPow_num_avgs,  # Nf = homosphasal fractions
                            Noffres = int(HPow_num_pts*0.2),  offresfraction = 0.1, # variables for segmented
                            filename_suffix="", verbose = False, wait_time = 3, 
                            segment_option="segmented",
                            # segment_option="linear",
                            # segment_option="homophasal",
                            # segment_option="hybrid",
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
                            sample_name = sample_name, data_dirs = data_dirs,
                            Nf = int(MPow_num_pts*0.8), Navg_init = MPow_num_avgs,  # Nf = homosphasal fractions
                            Noffres = int(MPow_num_pts*0.2),  offresfraction = 0.1, # variables for segmented
                            filename_suffix="", verbose = False, wait_time = 3, 
                            segment_option="segmented",
                            # segment_option="linear",
                            # segment_option="homophasal",
                            # segment_option="hybrid",
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
                            sample_name = sample_name, data_dirs = data_dirs,
                            Nf = int(LPow_num_pts*0.8), Navg_init = LPow_num_avgs,  # Nf = homosphasal fractions
                            Noffres = int(LPow_num_pts*0.2),  offresfraction = 0.1, # variables for segmented
                            filename_suffix="", verbose = False, wait_time = 3, 
                            segment_option="segmented",
                            # segment_option="linear",
                            # segment_option="homophasal",
                            # segment_option="hybrid",
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
                            sample_name = sample_name, data_dirs = data_dirs,
                            Nf = int(ULPow_num_pts*0.8), Navg_init = ULPow_num_avgs,  # Nf = homosphasal fractions
                            Noffres = int(ULPow_num_pts*0.2),  offresfraction = 0.1, # variables for segmented
                            filename_suffix="", verbose = False, wait_time = 3, 
                            segment_option="segmented",
                            # segment_option="linear",
                            # segment_option="homophasal",
                            # segment_option="hybrid",
                            )
tEnd_ultra_low = time.time()

for res_folder in data_dirs: 
    fig, ax = mf.plot_whole_directory(res_folder, "*\\*.csv", plot_dir=res_folder, 
                                        max_rows=5, verbose=False, plot_min=True, save_plot=True, show_plot=True)

mf.print_text_block(tStart_ultra_low, tEnd_ultra_low, ULPow_num_avgs, ULPow_num_pts, ULPow_IFBW_kHz, num_powers=len(ultra_low_powers), num_resonators=len(fcs))
