"""
# setup_measurement.py

08/16/24

### Purpose

    To be run at the beginning of *every* measurement

    e.g. global reference
        %run E:\GitHub\bcqt-ctrl\setup_qick_board
        
    or, if in Cooldown_Data\Cooldown_XX
        %run ..\..\setup_qick_board
"""
#############################################################################
####################### generic imports and paths ###########################
#############################################################################

import sys, os

print(f"Running setup_measurement.py from {os.getcwd()}...")

sys.path.append(r'E:\GitHub\bcqt-ctrl')
sys.path.append(r'E:\GitHub\bcqt-ctrl\temperature_control')
sys.path.append(r'E:\GitHub\bcqt-ctrl\resonator_measurements')
sys.path.append(r'E:\GitHub\bcqt-ctrl\resonator_measurements\helper_scripts')
sys.path.append(r'E:\GitHub\scresonators')

%run E:\GitHub\bcqt-ctrl\resonator_measurements\plot_settings
  
#############################################################################
##################################  TWPA  ###################################
#############################################################################




#############################################################################
##################################  HEMT  ###################################
#############################################################################




#############################################################################
############################### cryoswitch  #################################
#############################################################################


print("Setting up Qphox cryoswitch controller")
cryoswitch_IP = "192.168.0.117"

sys.path.append(r'E:\GitHub\CryoSwitchController')
try:
    switch = Cryoswitch(IP=cryoswitch_IP) ## -> CryoSwitch class declaration and USB connection
    switch.start() ## -> Initialization of the internal hardware
    
    switch.get_internal_temperature()
    switch.get_pulse_history(pulse_number=5, port='A') ##-> Show the last 5 pulses send through on port A
    switch.set_output_voltage(5) ## -> Set the output pulse voltage to 5V

except Exception as e:
    print(f"Failed to connect to Qphox cryoswitch controller at {qb_atten_IP=}")
    print(f"Error: \n{e}")
    
    
#############################################################################
#############################################################################
#############################################################################

print("\n~~ Finished running global_init.py! \n\n")

