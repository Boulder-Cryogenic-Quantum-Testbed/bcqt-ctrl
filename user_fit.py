 
# %%

%load_ext autoreload
%autoreload 2

import sys, time, os, glob
import numpy as np
import matplotlib.pyplot as plt
import regex as re

import misc_functions as mf
import helper_load as hl
import helper_misc as hm
import helper_fit as hf

import plot_settings

%run ../setup_measurement

# %%

# TODO: create metadata file that shares these parameters with all user files in the folder

# XXX: Change the sample name
base_dir = os.path.basename(os.getcwd())
line_num, sample_name = re.split(r"_", base_dir, maxsplit=1)  # use regex to split only at first _

print(f"{line_num}_{sample_name}")

# %% to try and get rid of stupid style sheet issue
import plot_settings
plt.subplots(1,1)
plt.plot()
plt.show()

# %%

# search_dir = 'best_datasets'        # hand picked datasets
search_dir = 'data'                   # all datasets
# search_dir = r'data\Jul_29_1931'    # specific dataset  

# use list comprehension to grab all glob items
alL_folders = [x for x in glob.glob(f"{search_dir}\\*GHz")]

# if looking through all datasets, filter through 
if search_dir == 'data':
    chosen_timestamp = -1
    all_timestamped_folders = [x for x in glob.glob(f"{search_dir}\\*") if 'all_plots' not in x]
    chosen_folder = all_timestamped_folders[chosen_timestamp]
    alL_folders = [x for x in glob.glob(f"{chosen_folder}\\*GHz")]

plot_dir = f"{search_dir}\\all_plots"
circle_plots_dir = f"{plot_dir}\\circle_plots"
power_plots_dir = f"{plot_dir}\\power_plots"
hm.check_and_make_dir(circle_plots_dir)
hm.check_and_make_dir(power_plots_dir)

display(alL_folders)

# %% e just the circle plots and regular data plots

# chosen_idx = [0, 1, 2, 3, 4, 5, 6, 7]
# chosen_idx = [-1, -2, -3]
chosen_idx = range(len(alL_folders))

chosen_resonators = [alL_folders[x] for x in chosen_idx]

for resonator_folder_path in chosen_resonators:

    all_datasets, all_dataset_paths = hl.load_files_in_dir(resonator_folder_path, key="*.csv", debug=False)   
    
    fig, ax = mf.plot_whole_directory(resonator_folder_path, "*\\*.csv", plot_dir=power_plots_dir,
                                        max_rows=5, verbose=False, plot_min=True, save_plot=False, show_plot=True)
        
    mf.plot_all_circles(resonator_folder_path, "*\\*.csv", plot_dir=circle_plots_dir, 
                                show_line=False, verbose=False, save_plot=False, show_plot=True)


                        
 # %%  individual folder loss tan fit 

# perform_loss_tan_fit = False
perform_loss_tan_fit = True
 
# chosen_idx = [0, 1, 2, 3, 4, 5, 6,]
# chosen_idx = [-1]
chosen_resonators = [alL_folders[x] for x in chosen_idx]

for resonator_folder_path in chosen_resonators:
    
    all_names, all_paths = hl.load_files_in_dir(resonator_folder_path, key="*.csv", blacklist='qiqc', debug=False)   

    powers_in = [hm.get_power_from_filename(x) for x in all_names if 'dB' in x] 

    # all should be within 10 mK anyway
    all_temperatures =  [int(hm.get_temperature_from_filename(fname)) for fname in all_names if 'qiqc' not in fname] 
    temperature = int(np.ceil(np.average(all_temperatures)))  
    
    # placeholder
    init_conds = [None]*len(all_names)
    # init_conds = [[5e5, 5e5, 0, np.pi/2] ]

    all_fits_save_dir = f"{resonator_folder_path}\\dcm_fits\\"
    qiqc_fit_save_dir = f"{plot_dir}"

    save_fit_dirs = [all_fits_save_dir, qiqc_fit_save_dir]
    for fit_dir in save_fit_dirs:
        hm.check_and_make_dir(fit_dir)
    
    try:
        hf.power_sweep_fit_drv(sample_name=sample_name,
                            atten=[0, -70], temperature=temperature,
                            powers_in=powers_in, all_paths=all_paths, 
                            # plot_from_file=False,
                            use_error_bars=True, temp_correction='', phi0=0.,
                            use_gauss_filt=False, use_matched_filt=False,
                            use_elliptic_filt=False, use_mov_avg_filt=False,
                            loss_scale=1e-6, preprocess_method='linear',
                            ds = {'QHP' : 1e5, 'nc' : 1e1, 'Fdtls' : 1e-6},
                            plot_twinx=False, plot_fit=perform_loss_tan_fit, QHP_fix=True, show_plots=True,
                            data_dir=resonator_folder_path, save_dcm_plot=True, manual_init_list=init_conds,
                            save_fit_dirs=save_fit_dirs, show_dbm=True)
        plt.show()
    except Exception as e:
        display(resonator_folder_path)
        print( "====================================================================================")
        print( "====================================================================================")
        print(f"=======================    Failed to fit:  {sample_name}    ========================")
        print( "====================================================================================")
        print( "====================================================================================")
        print("Error Message:  \n")
        print(e)   # TODO: add stack trace  
        print("\n\n")
            
# %% plot whole directory

# best_datasets_path = 'best_datasets'
# best_resonator_paths = glob.glob(f"{best_datasets_path}\\*GHz")

# for resonator_path in best_resonator_paths:
#     plot_dir_path = f"{resonator_path}"
#     mf.plot_whole_directory(resonator_path, search_str="\\*.csv", plot_dir=plot_dir_path, 
#                             save_plot=True, show_plot=False, verbose=True)

