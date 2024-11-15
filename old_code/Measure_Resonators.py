# %%

"""
    measure_resonators.py 
    
    Author: Jorge Ramirez
        based on "user_ctrl_segmented_homophasal.py" used in
        prior cooldowns to routinely measure resonators
        
"""

# %%
from pathlib import Path

# %load_ext autoreload
# %autoreload 2

# usually I will put this setup script in parent directory, to 
# keep initialization consistent between all device measurements

# %run ../global_scripts/setup_experiment
# %run ../global_scripts/setup_vna

setup_experiment_filepath = Path(".").absolute().parent / "global_scripts" / "setup_experiment.py"

with open(setup_experiment_filepath) as file:
    exec(file.read())

import time
import matplotlib.pyplot as plt

import misc_functions as mf
# import helper_plot as hp
 
from janis_ctrl import measure_multiple_resonators


# %% to try and get rid of stupid style sheet issue
import plot_settings
plt.subplots(1,1)
plt.plot()
plt.show()


# %%  grab current time and format for use as dataset label


# %% determine measurement parameters 

# assert that every freq has an associated span & delay
assert len(fcs) == len(spans), f"length of fcs =/= spans  ({len(fcs)} =/= {len(spans)})"
assert len(fcs) == len(delays), f"length of fcs =/= delays ({len(fcs)} =/= {len(delays)} )"
    
# create strings for the filename frequencies and data directories 
freq_strs = [f'{fc:.3f}GHz'.replace('.', 'p') for fc in fcs]
data_dirs = [f'data\\{dstr}\\{sample_name}_{freq_str}' for freq_str in freq_strs]

print(f"\nRunning measurements, saving data in these directories:")
print(*[f'   "{dir_str}"' for dir_str in data_dirs], sep="\n")  # neat format


# %% estimate runtime

# %% measurement settings

# for a given offresfraction, x% of points should have (1-x)% of span
#   e.g. for 20 %, the npts dist is 10/80/10
#             and the freq dist is 40/20/40
offresfraction = 0.8


# %% high power scan
high_powers = [int(x) for x in high_powers_floats]

if MEASURE_HPOW is True:
    try: 
        tStart_high = time.time()   
        mf.check_valid_fridge_temp()
        measure_multiple_resonators(fcs, spans, delays, high_powers, 
                                    Navg_init=HPow_num_avgs,
                                    ifbw = HPow_IFBW_kHz, npts = HPow_num_pts,  
                                    sample_name = sample_name, data_dirs = data_dirs, 
                                    offresfraction=offresfraction, Noffres=5,  # first and last x% have (1-x)% of the points
                                    filename_suffix=fname_suffix, verbose = False, wait_time = 3, 
                                    # segment_option="segmented", is_segmented = True,
                                    segment_option="linear",
                                    # segment_option="homophasal",
                                    # segment_option="hybrid",
                                    )
        tEnd_high = time.time() 


        mf.print_text_block(tStart_high, tEnd_high, HPow_num_avgs, HPow_num_pts, HPow_IFBW_kHz, num_powers=len(high_powers), num_resonators=len(fcs))

        for res_folder in data_dirs: 
            fig, ax = mf.plot_whole_directory(res_folder, "*\\*.csv", plot_dir=res_folder, 
                                                max_rows=5, verbose=False, plot_min=True, save_plot=True, show_plot=True)
    except Exception as e:
        display("Failed! Error:")
        print(e)


# %% medium power scan


if MEASURE_MPOW is True:
    try: 
        med_powers = [int(x) for x in med_powers_floats]


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
    
    except Exception as e:
        display("Failed! Error:")
        print(e)



# %% low power scan


if MEASURE_LPOW is True:
    try: 
        low_powers = [int(x) for x in low_powers_floats]

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
    
    except Exception as e:
        display("Failed! Error:")    
        print(e)

# %% ultra low power scan

if MEASURE_ULPOW is True:
    try: 
        ultra_low_powers = [int(x) for x in ultra_low_powers_floats]

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
    except Exception as e:
        display("Failed! Error:")
        print(e)
