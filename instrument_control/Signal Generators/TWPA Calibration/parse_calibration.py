"""
calibration data anaylsis
show results of user_calibrate

Author: Jorge Ramirez
Date:   08/2024
                
"""
# %%

%load_ext autoreload
%autoreload 3

%run ../../../resonator_measurements/setup_measurement

import sys, os

from pathlib import Path
from datetime import datetime
import helper_misc as hm
import regex as re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data_dir = Path.cwd() / 'data' 
timestamp_folders = [x for x in data_dir.glob("*")]
pick_idx = -1  # just use the most recent measurement

data_folder_paths = [x for x in timestamp_folders[pick_idx].glob("*")]


measurement_folder_path = data_folder_paths[-1]


# %%
all_data_csvs = [x for x in measurement_folder_path.glob("*.csv")]
all_dataframes = {}
for csv_filepath in all_data_csvs:
    csv_filename = csv_filepath.stem
    
    # power is from filename, freq is from filename
    # assumes you swept TWPA freq first, then TWPA power
    twpa_freq = float(re.search(r"\d{1,2}p\d{1,3}", csv_filename)[0].replace("p","."))
    twpa_power = float(csv_filepath.parent.stem.replace("_dBm",""))

    meas_info = {
        "Folder Timestamp" : csv_filepath.parent.stem,
        # "Temperature" : hm.get_temperature_from_filename(csv_filename),
        # "VNA Power" : hm.get_power_from_filename(csv_filename),
        "TWPA Power" : twpa_power, 
        "TWPA Freq" : twpa_freq, 
    }
    
    df = pd.read_csv(csv_filepath, names=["Frequency [Hz]", "Magnitude [dBm]", "Phase [deg]"])
    df["Frequency [Hz]"] = df["Frequency [Hz]"] / 1e9
    df["Phase [rad]"] = np.deg2rad(df["Phase [deg]"])
    
    df.head()
    
    all_dataframes[csv_filename] = (df, meas_info)
    
# %% extract data
    
twpa_freq_array = []
vna_magn_array = []
for csv_filename, (df, meas_info) in all_dataframes.items():
    vna_magn_array.append(np.array(df["Magnitude [dBm]"]))
    twpa_freq_array.append(meas_info["TWPA Freq"])
    
    
vna_freq_array = np.array(df["Frequency [Hz]"])
twpa_freq_array = np.array(twpa_freq_array)
vna_magn_array = np.array(vna_magn_array)
print(vna_freq_array.shape, twpa_freq_array.shape, vna_magn_array.shape)

# %% create imshow plot
# x axis will be vna frequency
# z axis (color) will be vna response
# y axis will be twpa frequency or twpa power

mosaic = "AABB\nCCCC"
fig, axes = plt.subplot_mosaic(mosaic, figsize = (10,5))
ax1, ax2, ax3 = axes["A"], axes["B"], axes["C"]

x = vna_freq_array;   x_label = "VNA Frequency [GHz]"
y = twpa_freq_array;  y_label = "TWPA Frequency [GHz]"
c = vna_magn_array;    c_label = "VNA S21 Response [dB]"

extent = [x.min(), x.max(), y.min(), y.max()]

img = ax1.imshow(c, aspect='auto', extent=extent, 
                 origin='lower', cmap='rainbow', interpolation="None")
cbar = fig.colorbar(img, ax=ax1, label=c_label)

########
# zoom in on a section, used chatgpt for some code because I hate imshow
########

add_zoom_box = False

if add_zoom_box:
    # Define the zoomed region
    x_zoomed_min, x_zoomed_max = 4, 8          # Zoom range for x-axis, or vna frequency
    y_zoomed_min, y_zoomed_max = 7.9, 7.915     # Zoom range for y-axis, or twpa frequency

    # show zoomed region on first plot
    zoomed_box_coords = [ (x_zoomed_min, y_zoomed_min),
                        (x_zoomed_min, y_zoomed_max),
                        (x_zoomed_max, y_zoomed_max),
                        (x_zoomed_max, y_zoomed_min),
                        (x_zoomed_min, y_zoomed_min)  ]

    x_coords, y_coords = [x for x,y in zoomed_box_coords], [y for x,y in zoomed_box_coords], 
    ax1.plot(x_coords, y_coords, 'r-', linewidth=1)

    # Find the indices that correspond to the zoomed x and y ranges
    x_indices = np.where((x >= x_zoomed_min) & (x <= x_zoomed_max))[0]
    y_indices = np.where((y >= y_zoomed_min) & (y <= y_zoomed_max))[0]

    zoomed_c = c[np.min(y_indices):np.max(y_indices)+1, np.min(x_indices):np.max(x_indices)+1]

    # Create the zoomed plot using imshow
    zoomed_extent = [x_zoomed_min, x_zoomed_max, y_zoomed_min, y_zoomed_max]
    img2 = ax2.imshow(zoomed_c, aspect='auto', extent=zoomed_extent,
                    origin='lower', cmap='viridis', interpolation="None")

    cbar = fig.colorbar(img, ax=ax2, label=c_label)
else:
    fig.delaxes(ax2)
    fig.delaxes(ax3)
    
    
########
# show some VNA traces if the zoomed section
########

if len(y_indices) <= 5:
    for y_idx in y_indices:
        ax3.plot(x[x_indices], c[y_idx, x_indices], label=f"$f_{"{TWPA}"}$ = {y[y_idx]:1.3f} GHz")
    ax3.legend()
else:
    fig.delaxes(ax3)

# misc labels
ax1.set_xlabel(x_label)
ax2.set_xlabel(x_label)
ax3.set_xlabel(x_label)

ax1.set_ylabel(y_label)
ax2.set_ylabel(y_label)
ax3.set_ylabel("VNA S21 [dB]")

ax1.set_title("TWPA Frequency vs VNA Response Scan")
ax2.set_title("Zoomed & Rescaled red box")
ax3.set_title("Individual VNA Traces taken from red box")

fig.suptitle(f"TWPA Calibration - Power = {twpa_power}")
fig.tight_layout()
