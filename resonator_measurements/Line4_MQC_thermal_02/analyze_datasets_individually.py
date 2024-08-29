

"""
    analyze_datasets_individually.py 
    
    Author: Jorge Ramirez
        redid `analyze_data.py` but with the intent to track which data points came from what scan. 
        Necessary when taking multiple traces of a single resonator, since there may be systematic
        changes between measurements- not necessarily experimental systematic errors, but for example
        some kind of temperature change or fluctuation in nearby TLS contributions to resonator Q
        
"""
# %%

%load_ext autoreload
%autoreload 2

%run ../setup_vna_measurement  

import sys, time, os, glob
import numpy as np
import matplotlib.pyplot as plt
import regex as re

import misc_functions as mf
import helper_load as hl
import helper_misc as hm
import helper_fit as hf

# may need to open this file, save it and
# then close it for settings to apply
import plot_settings  

# %%

# XXX: Change the sample name
base_dir = os.path.basename(os.getcwd())
line_num, sample_name = re.split(r"_", base_dir, maxsplit=1)  # use regex to split only at first _

print(f"{line_num}_{sample_name}")


# %% stupid matplotlib bug....
# you need to open import `plot_settings` 
# save it with `ctrl+s`
# then import it again` 
plt.subplots(1,1)
plt.plot()
plt.show()

# %% use glob and sort to get all datasets

search_dir = 'all_datasets'  # `data` should contain folders with timestamps 

plot_dir = f"{search_dir}/all_plots"
circle_plots_dir = f"{plot_dir}/circle_plots"
power_plots_dir = f"{plot_dir}/power_plots"
hm.check_and_make_dir(circle_plots_dir)
hm.check_and_make_dir(power_plots_dir)


# %% first scenario:  hand-picked data

if search_dir == "best_data":
    
    pass


# %% second scenario:  timestamp folders of data scans

# if search_dir == "all_datasets":

all_timestamp_paths = [x for x in glob.glob(f"{search_dir}/*") if 'all_plots' not in x]
all_timestamp_paths = sorted(all_timestamp_paths)

# now that we have all timestamped folders, go through them and get all resonator sub-folders and
# save into a dictionary for easy access to the timestamp foldername and also each resonator folder

all_datasets = {}

for timestamp_folder in all_timestamp_paths:
    # all_datasets => "timestamp_folder" : [ array of resonator folders ]
    
    # use os.path.basename() to keep just the folder name, since that is equal to 
    # `timestamp_folder` by construction
    all_datasets[timestamp_folder] = sorted([os.path.basename(x) for x in glob.glob(f"{timestamp_folder}/*GHz")])
    
    
all_resonators = []
for dataset_directory, all_res_folders in all_datasets.items():
    for res_folder in all_res_folders:
        if res_folder not in all_resonators:
            all_resonators.append(res_folder)
        
    # print(f"Dataset Directory: {dataset_directory}\n   Datasets: {resonator_folder}\n")

# %%

print(f"Found {len(all_datasets)} datasets:", *list(all_datasets.keys()), sep="\n  ~~ ")
print(f"\n{len(all_resonators)} unique resonators found in {len(all_datasets)} datasets", *all_resonators, sep="\n  ~~ ")



# %%  begin analysis

# now we will work with all_datasets, regardless of where the datasets are

# goal: create a single "resonator" object for each resonator, and then store all datasets underneath
#          this way, we can keep track of the parameters, data, and notes on each dataset separately
#          effectively creating a "metadata" for all measurements


for dataset_directory, res_folders in all_datasets.items():
    print(f"\n\nLooking in: {dataset_directory}  ... ")
    print(f"  found {len(res_folders)} resonator datasets")
    print(f"  ")
    
    for single_res_path in res_folders:
        all_csv_filepaths = glob.glob(f"{dataset_directory}/{single_res_path}/*.csv")
        
        # filter out all items that are not data files... easiest way is that every data file
        # should have the sample name in its filename. most general case
        all_csv_filenames = [os.path.basename(x) for x in all_csv_filepaths if sample_name in os.path.basename(x)]
        
        # conveniently, this sorts from highest to lowest power 
        all_csv_filenames = sorted(all_csv_filenames)  
        
        for csv_file in all_csv_filenames:
            print(f"  {csv_file}")
    
    
    
    
    

    


















