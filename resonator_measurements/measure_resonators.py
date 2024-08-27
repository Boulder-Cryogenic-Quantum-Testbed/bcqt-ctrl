# %%

"""
    measure_resonators.py 
    
    Author: Jorge Ramirez
        based on "user_ctrl_segmented_homophasal.py" used in
        prior cooldowns to routinely measure resonators
        
"""
 
# %%

%load_ext autoreload
%autoreload 2

# usually I will put this setup script in parent directory, to 
# keep initialization consistent between all device measurements

%run ../setup_vna_measurement

import sys, glob
import time, os 
import numpy as np
import matplotlib.pyplot as plt
import regex as re

import misc_functions as mf
# import helper_plot as hp
 
from datetime import datetime
from janis_ctrl import measure_multiple_resonators


# %% to try and get rid of stupid style sheet issue
import plot_settings
plt.subplots(1,1)
plt.plot()
plt.show()

# %%  grab current time and format for use as dataset label

dstr = datetime.today().strftime(r'%b_%d_%H%M')
print(f"\n\nCurrent folder timestamp: {dstr}\n\n")

# use parent directory name to get device name
base_dir = os.path.basename(os.getcwd())
line_num, sample_name = re.split(r"_", base_dir, maxsplit=1)  # use regex to split only at first _

# %% determine measurement parameters 

# use VNA to record center frequencies and 
# the span for each resonator measurement.
all_fcs = [
            4.329693, 4.707040,
            5.085460, 5.451642,
            5.840113, 6.210005,
            6.592910, 6.966201
          ] # GHz

all_spans = [
              0.20, 0.20,
              0.20, 0.25,
              0.50, 0.35,
              0.25, 0.50
            ] # MHz

###################################################
##### choose here which resonators to measure #####
###################################################
# by default, measure all resonators
idxs = [
        # 0, #1, 
        2, #3,
        # 4, 
        # 5,
        # 6, 7,
        ]   

# or choose a specific subset  
# idxs = [4, 7]  
# idxs = [0, 1]  

fcs = [all_fcs[idx] for idx in idxs]   

###################################################
###################################################

# set electrical delay & span for each measurement
delays = [85.23]*len(fcs)  
spans = [all_spans[idx] for idx in idxs]

# assert that every freq has an associated span & delay
assert len(fcs) == len(spans), f"length of fcs =/= spans  ({len(fcs)} =/= {len(spans)})"
assert len(fcs) == len(delays), f"length of fcs =/= delays ({len(fcs)} =/= {len(delays)} )"
    
# create strings for the filename frequencies and data directories 
freq_strs = [f'{fc:.3f}GHz'.replace('.', 'p') for fc in fcs]
data_dirs = [f'data\\{dstr}\\{sample_name}_{freq_str}' for freq_str in freq_strs]

print(f"Running measurements, saving data in these directories:")
print(*[f'   "{dir_str}"' for dir_str in data_dirs], sep="\n")  # neat format

# add an ending to the filename
fname_suffix = "10dB_Atten"

# %% estimate runtime

# in case of multiple runs, might as well measure at diff powers
offset = 0  # must be positive!!
high_powers_floats = np.arange(-37, -53, -4) + offset
med_powers_floats = np.arange(-53, -69, -4) + offset
low_powers_floats = np.arange(-69, -87, -3) + offset 
ultra_low_powers_floats = np.arange(-87, -91, -3) + offset  # VNA stops at -90

power_tuple_dict = {  # comment a line if only measuring certain powers
                 # tuple name :  (num of power,  measurement duration in seconds)
                    "HPow"  : (  len(high_powers_floats),      33     ), 
                    "MPow"  : (   len(med_powers_floats),      203    ),  
                    "LPow"  : (   len(low_powers_floats),      555   ),
                    "ULPow" : ( len(ultra_low_powers_floats),  3657  )
                    }

mf.estimate_resonator_runtime(power_tuple_dict, num_res=len(fcs))
print(f"\nResonators to measure: \n  {fcs}\n")
print(high_powers_floats, med_powers_floats, low_powers_floats, ultra_low_powers_floats, sep="\n")

# %% measurement settings

# for a given offresfraction, x% of points should have (1-x)% of span
#   e.g. for 20 %, the npts dist is 10/80/10
#             and the freq dist is 40/20/40
offresfraction = 0.8

# %% high power scan

high_powers = [int(x) for x in high_powers_floats]

HPow_num_avgs, HPow_IFBW_kHz, HPow_num_pts = 1000, 2.0, 51
 
tStart_high = time.time()   
mf.check_valid_fridge_temp()
measure_multiple_resonators(fcs, spans, delays, high_powers, 
                            Navg_init=HPow_num_avgs,
                            ifbw = HPow_IFBW_kHz, npts = HPow_num_pts,  
                            sample_name = sample_name, data_dirs = data_dirs, 
                            offresfraction=offresfraction, Noffres=5,  # first and last x% have (1-x)% of the points
                            filename_suffix=fname_suffix, verbose = False, wait_time = 3, 
                            segment_option="segmented", is_segmented = True,
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

MPow_num_avgs, MPow_IFBW_kHz, MPow_num_pts = 6000, 2.0, 51

tStart_med = time.time()   
mf.check_valid_fridge_temp()
measure_multiple_resonators(fcs, spans, delays, med_powers, 
                            Navg_init=MPow_num_avgs,
                            ifbw = MPow_IFBW_kHz, npts = MPow_num_pts,  
                            sample_name = sample_name, data_dirs = data_dirs,
                            offresfraction=offresfraction, Noffres=5,  # first and last x% have (1-x)% of the points
                            filename_suffix=fname_suffix, verbose = False, wait_time = 3, 
                            segment_option="segmented", is_segmented = True,
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

LPow_num_avgs, LPow_IFBW_kHz, LPow_num_pts = 20000, 2.0, 51

tStart_low = time.time()   
mf.check_valid_fridge_temp()
measure_multiple_resonators(fcs, spans, delays, low_powers,
                            Navg_init=LPow_num_avgs,
                            ifbw = LPow_IFBW_kHz, npts = LPow_num_pts,  
                            sample_name = sample_name, data_dirs = data_dirs,
                            offresfraction=offresfraction, Noffres=5,  # first and last x% have (1-x)% of the points
                            filename_suffix=fname_suffix, verbose = False, wait_time = 3, 
                            segment_option="segmented", is_segmented=True,
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

ULPow_num_avgs, ULPow_IFBW_kHz, ULPow_num_pts = 75000, 1.0, 51

tStart_ultra_low = time.time()   
mf.check_valid_fridge_temp()

measure_multiple_resonators(fcs, spans, delays, ultra_low_powers,
                            Navg_init=ULPow_num_avgs,
                            ifbw = ULPow_IFBW_kHz, npts = ULPow_num_pts,  
                            sample_name = sample_name, data_dirs = data_dirs,
                            offresfraction=offresfraction, Noffres=5,  # first and last x% have (1-x)% of the points
                            filename_suffix=fname_suffix, verbose = False, wait_time = 3, 
                            segment_option="segmented", is_segmented=True
                            # segment_option="linear",
                            # segment_option="homophasal",
                            # segment_option="hybrid",
                            )
tEnd_ultra_low = time.time()

for res_folder in data_dirs: 
    fig, ax = mf.plot_whole_directory(res_folder, "*\\*.csv", plot_dir=res_folder, 
                                        max_rows=5, verbose=False, plot_min=True, save_plot=True, show_plot=True)

mf.print_text_block(tStart_ultra_low, tEnd_ultra_low, ULPow_num_avgs, ULPow_num_pts, ULPow_IFBW_kHz, num_powers=len(ultra_low_powers), num_resonators=len(fcs))

# %%

