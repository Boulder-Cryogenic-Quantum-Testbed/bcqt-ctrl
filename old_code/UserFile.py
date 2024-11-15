import sys, os
sys.path.append(r"E:\GitHub\bcqt-ctrl\resonator_measurements\helper_scripts")
sys.path.append(r"E:\GitHub\bcqt-ctrl\pna_control")

import misc_functions as mf
import regex as re
import numpy as np
from datetime import datetime
from pathlib import Path


# if this is giving you a syntax error,
# comment these two lines out. these are
# only commands for a jupyter notebook
%load_ext autoreload
%autoreload 3

###################################################
#########  VNA Frequencies to Measure  ############
###################################################

# center frequencies, in GHz
all_fcs = [
            4.0, 4.5,
            5.0, 5.5,
            6.5, 7.0,
            7.5, 8.0,
          ] # GHz

# spans, in MHz
all_spans = [
              500, 500,
              500, 500,
              500, 500,
              500, 500,
            ] # MHz



###################################################
#######  Which resonators will we measure?  #######
###################################################

# by default, measure all resonators
idxs = [
        0, 1, 
        2, 3,
        4, 5,
        6, 7,
        ]   

# or choose a specific subset
idxs = [7]  
# idxs = [0, 1, 2, 3, 4, 7, ]  

# finalize arrays for fcs and spans
fcs = [all_fcs[idx] for idx in idxs]   
spans = [all_spans[idx] for idx in idxs]

# set electrical delay & spans
delays = [85.92]*len(fcs)  

###################################################
#########  Sweep Parameters (per power)  ##########
###################################################

# see next block for guidance on choosing parameters
# np.arange( start, stop, step ) # in dBm  (NOT dB!)
high_powers_floats = np.arange(-37, -53, -4)
med_powers_floats = np.arange(-53, -69, -4)
low_powers_floats = np.arange(-69, -87, -3) 
ultra_low_powers_floats = np.arange(-87, -91, -3)  # VNA stops at -90!!!



###################################################
###############  Power Settings   #################
###################################################

# define measurement parameters

HPow_num_avgs, HPow_IFBW_kHz, HPow_num_pts =         1, 2.0, 50001

MPow_num_avgs, MPow_IFBW_kHz, MPow_num_pts =       200, 2.0, 51 

LPow_num_avgs, LPow_IFBW_kHz, LPow_num_pts =      2000, 2.0, 51

ULPow_num_avgs, ULPow_IFBW_kHz, ULPow_num_pts =  10000, 2.0, 51

#     generally, I try to organize this by time taken where
#  I want to spend ~x seconds/power for a given power i.e, 
#         ~10 seconds/power for high_power
#        ~100 seconds/power for medium_power
#        ~500 seconds/power for low_power
#       ~2000+ seconds/power for ultra_low_power
#
#     where I'll have something like 10/5/5/2 powers per level
#  in general, test your parameters with high/medium power first
#  and then you can let low power run. I wouldn't run an ultra
#  low power without inspecting the results of low power unless
#  it's an overnight measurement and I have nothing to lose
#
#  the worst feeling in the world is spending 2+ hours on a
#  ultra low power measurement and your span is incorrect

# quick estimate of how long a measurement will take, and note that
# it's formatted so that you can comment out a specific line
power_tuple_dict = {  
                 # tuple name :  (num of power,  duration of one power in seconds)
                    "HPow"  : (  len(high_powers_floats),      33     ), 
                    "MPow"  : (   len(med_powers_floats),      203    ),  
                    "LPow"  : (   len(low_powers_floats),      555   ),
                    #"ExamplePow" : (  comment and no syntax error, 500),
                    "ULPow" : ( len(ultra_low_powers_floats),  3657  ),
                    }

  
mf.estimate_resonator_runtime(power_tuple_dict, num_res=len(fcs))

print(f"\nResonators to measure: \n  {fcs}\n")
print(high_powers_floats, med_powers_floats, low_powers_floats, ultra_low_powers_floats, sep="\n")



###################################################
############  Miscellaneous Settings   ############
###################################################

# add an ending to the filename
fname_suffix = ""

# grab today's date and format for use in data collection
dstr = datetime.today().strftime(r'%b_%d_%H%M')
print(f"\n\nCurrent folder timestamp: {dstr}\n\n")

# use parent directory name to get device name
base_dir = os.path.basename(os.getcwd())
line_string = re.search(r"Line \d", base_dir)[0]
sample_name = base_dir.replace(f"{line_string} - ", "") 


###################################################
######  Finally, run measure_resonators.py  #######
###################################################

measure_resonators_filepath = Path(".").absolute().parent / "measure_resonators.py"

MEASURE_HPOW = True
MEASURE_MPOW = False
MEASURE_LPOW = False
MEASURE_ULPOW = False

# with open(measure_resonators_filepath) as script:
#     print(script)
#    exec(script.read())

# will only work if run as an interactive (or jupyter) notebook
# %run ../measure_resonators.py

##################################################
##################  Test VNA   ###################
##################################################

sys.path.append(r"E:\GitHub\bcqt-ctrl\pna_control")
import pyvisa
import pna_control as PNA 

test_vna_filepath = Path(".").absolute().parent / "example_vna_comm.py"

# with open(test_vna_filepath) as script:
  #  exec(script.read())
  
# will only work if run as an interactive (or jupyter) notebook
%run ../example_vna_comm.py