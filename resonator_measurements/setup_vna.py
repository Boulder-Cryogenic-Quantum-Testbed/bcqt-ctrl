"""
# setup_vna_measurement.py

### Purpose

To be run at the beginning of every measurement

"""
#############################################################################
####################### generic imports and paths ###########################
#############################################################################

import sys, os

print(f"Running setup_vna_measurement.py from {os.getcwd()}...")

sys.path.append(r'E:\GitHub\bcqt-ctrl')
sys.path.append(r'E:\GitHub\bcqt-ctrl\temperature_control')
sys.path.append(r'E:\GitHub\bcqt-ctrl\resonator_measurements')
sys.path.append(r'E:\GitHub\bcqt-ctrl\resonator_measurements\helper_scripts')

sys.path.append(r'E:\GitHub\scresonators')

global_scripts_dir = r"E:\Cooldown_Data\Cooldown55\global_scripts"
print("\n\n#############################################################################")
print(f"~~~~~ Using {global_scripts_dir} as global script directory source. ~~~~~ ")
print("#############################################################################\n\n")

sys.path.append(f'{global_scripts_dir}')
sys.path.append(rf'{global_scripts_dir}\instruments')

import plot_settings
import matplotlib.pyplot as plt
import helper_misc as hm

#############################################################################
############################### attenuators #################################
#############################################################################

# print("Setting up attenuators ")

# sys.path.append("./scripts")

# from MiniCircuits_Attenuator import set_atten, read_atten
# qb_atten_IP = "192.168.0.118"  
# ro_atten_IP = "192.168.0.119"

# try:
#     set_atten(ro_atten_IP, 0)
#     read_atten(ro_atten_IP)
# except Exception as e:
#     print("Failed to connect to qubit attenuator at {qb_atten_IP=}")
#     print("Error: \n{e}")

# try:
#     set_atten(qb_atten_IP, 0)
#     read_atten(qb_atten_IP)
# except Exception as e:
#     print("Failed to connect to resonator attenuator at {ro_atten_IP=}")
#     print("Error: \n{e}")
  
#############################################################################
############################### cryoswitch  #################################
#############################################################################

print("Setting up Qphox cryoswitch controller")
cryoswitch_IP = "192.168.0.117"

sys.path.append(r'E:\GitHub\CryoSwitchController')

# from CryoSwitchController import Cryoswitch

# try:
#     switch = Cryoswitch(IP=cryoswitch_IP) ## -> CryoSwitch class declaration and USB connection
#     switch.start() ## -> Initialization of the internal hardware
    
#     switch.get_internal_temperature()
#     switch.get_pulse_history(pulse_number=5, port='A') ##-> Show the last 5 pulses send through on port A
#     switch.set_output_voltage(5) ## -> Set the output pulse voltage to 5V

# except Exception as e:
#     print(f"Failed to connect to Qphox cryoswitch controller at {cryoswitch_IP=}")
#     print(f"Error: \n{e}")


print("\n~~ Finished running global_init.py! \n\n")

