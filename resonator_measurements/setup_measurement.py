"""
# setup_measurement.py

### Purpose

To be ran at the beginning of every measurement

"""
#############################################################################
####################### generic imports and paths ###########################
#############################################################################

import sys, os
import matplotlib.pyplot as plt

print(f"Running run_config.py from {os.getcwd()}...")

sys.path.append(r'E:\GitHub\bcqt-ctrl')
sys.path.append(r'E:\GitHub\bcqt-ctrl\temperature_control')
sys.path.append(r'E:\GitHub\bcqt-ctrl\helper_scripts')
sys.path.append(r'E:\GitHub\scresonators')

global_scripts_dir = r"E:\Cooldown_Data\Cooldown55\global_scripts"
print(f"\nUsing {global_scripts_dir} as global script directory source.\n")

sys.path.append(f'{global_scripts_dir}')
sys.path.append(rf'{global_scripts_dir}\instruments')

# import plot_settings s

import plot_settings
#############################################################################
############################### attenuators #################################
#############################################################################

# print("Setting up attenuators ")

# sys.path.append("./scripts")

# from MiniCircuits_Attenuator import set_atten, read_atten
# qb_atten_IP = "192.168.137.101"  
# ro_atten_IP = "192.168.137.102"

# try:
#     set_atten(ro_atten_IP, 5) #-28dBm for 13dBm
#     read_atten(ro_atten_IP)

#     set_atten(qb_atten_IP, 0) #-28dBm for 13dBm
#     read_atten(qb_atten_IP)
# except Exception as e:
#     print(e)
  
print("\n~~ Finished running global_init.py! \n\n")

