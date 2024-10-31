"""
# setup_qick_board.py

08/27/24

### Purpose

    To be run at the beginning of *every* measurement or experiment, regardless
    of what instruments you are using
    
    Mainly standardizes paths from different machines, as well as ensuring the TWPA,
    HEMT, cryoswitch, and attenuators are connected (if available)

"""
#############################################################################
####################### generic imports and paths ###########################
#############################################################################

import sys, os, platform
from pathlib import Path

print(f"Running setup_measurement.py from {os.getcwd()}...")

# TODO: just scan the bcqt-ctrl folder for subdirs and append those
if platform.system() == 'Windows':  # assuming this is the new lab PC, 68714MCRAE
    github_folder = Path(r'E:\GitHub')
else:  # oops... basically only Jorge's mac
    github_folder = Path(r'/Users/jlr7/Library/CloudStorage/OneDrive-UCB-O365/GitHub')

# was this an awful idea? yes. but now I can use a single 'for' loop with sys.path.append
all_folders = [ github_folder / 'scresonators' ,
                github_folder / 'bcqt-ctrl' ,
                
                github_folder / 'bcqt-ctrl' / 'instrument_control',
                github_folder / 'bcqt-ctrl' / 'instrument_control' / 'CryoSwitchController',
                
                github_folder / 'bcqt-ctrl'  / 'temperature_control', 
                github_folder / 'bcqt-ctrl'  / 'resonator_measurements', 
                github_folder / 'bcqt-ctrl'  / 'resonator_measurements' / 'helper_scripts',
]

print(f"Appending '{github_folder}' and listed subfolders to system path...")
for subfolder in all_folders:
    print(f"    ...'{subfolder.name}'")
    sys.path.append(str(subfolder))

print(f"Detected operating system ['{platform.system()}]")

#############################################################################
##################################  TWPA  ###################################
#############################################################################




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

from CryoSwitchController import Cryoswitch 

try:
    
    #### BCQT setting - changed IP August 2nd 2024
    #### changed IP from default 192.168.1.101
    ####  to 192.168.0.117 throughout script
    
    switch = Cryoswitch(IP='192.168.0.117') ## -> CryoSwitch class declaration and USB connection

    switch.start() ## -> Initialization of the internal hardware
    
    switch.get_internal_temperature()
    switch.get_pulse_history(pulse_number=5, port='A') ##-> Show the last 5 pulses send through on port A
    switch.get_pulse_history(pulse_number=5, port='B') ##-> Show the last 5 pulses send through on port A
    switch.set_output_voltage(5) ## -> Set the output pulse voltage to 5V

    # switch.connect(port='A', contact=1) ## Connect contact 1 of port A to the common terminal
    # switch.disconnect(port='A', contact=1) ## Disconnects contact 1 of port A from the common terminal


except Exception as e:
    print(f"Failed to connect to Qphox cryoswitch controller at {}")
    print(f"Error: \n{e}")
    
    
#############################################################################
#############################################################################
#############################################################################


from MiniCircuits_Attenuator import set_atten, read_atten
qb_atten_IP = "192.168.0.118"  
ro_atten_IP = "192.168.0.119"

try:
    set_atten(ro_atten_IP, 0)
    read_atten(ro_atten_IP)
except Exception as e:
    print("Failed to connect to qubit attenuator at {qb_atten_IP=}")
    print(f"Error: \n{e}")

try:
    set_atten(qb_atten_IP, 0)
    read_atten(qb_atten_IP)
except Exception as e:
    print("Failed to connect to resonator attenuator at {ro_atten_IP=}")
    print(f"Error: \n{e}")
  
  

#############################################################################
##############################  attenuators  ################################
#############################################################################

  
print("\n~~ Finished running global_init.py! \n\n")

