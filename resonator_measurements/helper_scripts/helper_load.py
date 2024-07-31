# %%

'''
    helper_load.py
'''

# print("    loading helper_load.py")

# %%

import glob, os, sys, time

import pandas as pd
import regex as re
import numpy as np

# %%
def load_many_csvs_as_dataframes(search_dir, search_str="\\*.csv", column_headers=["Freq","Magn","Phase"], debug=False):
    data_dict = {}
    for item in glob.glob(search_dir + search_str):
        if debug: print("    ",  item_name)
        if ".csv" in item and "qiqc" not in item:
            item_name = item.replace(search_dir, "")
            resonator_name = os.path.dirname(item_name)
            if debug: print(resonator_name, "    ",  item_name)
            try:
                df = pd.read_csv(item, names=column_headers)
                data_dict[item_name] = df
            except Exception as e:
                print(rf"Failed to load {item}\n    {e}")
    return data_dict


def gen_dataframe_from_csv(filepath, **kwargs):
    filename = os.path.basename(filepath)
    df = pd.read_csv(filepath, names=["Frequency", "Magnitude", "Phase_Deg"])
    freq = df["Frequency"]
    
    if any(freq > 1e9):  # scale to GHz
        freq = freq/1e9
        
    phase_deg = df["Phase_Deg"]
    phase = np.deg2rad(phase_deg)
    df["Frequency"] = freq
    df["Phase_Rad"] = phase
    
    index = df.index
    index.name = filename
    
    return df


def load_files_in_dir(directory, key="*", blacklist='', debug=False):
    if directory[-1] != "\\":  #??? 
        directory = directory + "\\"
        
    if debug: 
        print(f"     directory + key = {directory + key}")
        print(f"   glob({directory+key}) = ")
        for val in glob.glob(directory + key): 
            print(val)
    
    filepaths = [fname for fname in glob.glob(directory + key) if blacklist not in fname]
    filenames = [os.path.basename(x) for x in glob.glob(directory + key) if blacklist not in x]
    
    if len(filepaths) == 0 or len(filenames) == 0:
        print("No files found. Check that the directory, key, and that the") 
        print("blacklist is correct using the debug=True argument")
    
    if debug:
        print(f"Chosen directory: {directory}")
        for file, path in zip(filenames, filepaths):
            print(f"   {path}")
            
    return filenames, filepaths




# %%