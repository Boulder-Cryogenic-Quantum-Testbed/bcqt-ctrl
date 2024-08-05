"""
# setup_qick_measurement.py

### Purpose

To be ran at the top of every script using the following code

"""

#############################################################################
####################### generic imports and paths ###########################
#############################################################################

import sys, os
import matplotlib.pyplot as plt

print(f"Running setup_qick_measurement.py from {os.getcwd()}...")

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

#############################################################################
##################### connect to pyro 4 nameserver ##########################
#############################################################################

import Pyro4
from qick.pyro import make_proxy

Pyro4.config.SERIALIZER = "pickle"
Pyro4.config.PICKLE_PROTOCOL_VERSION=4

ns_host = "192.168.0.99"
ns_port = 8888
proxy_name = "BCQT_QICK"

print(" Connecting to server {}:{} as a client under proxy '{}'...".format(ns_host, ns_port, proxy_name))

ns = Pyro4.locateNS(host=ns_host, port=ns_port)
soc, soccfg = make_proxy(ns_host, ns_port, proxy_name)

try:
    print("\n*** installing pyro's excepthook")
    sys.excepthook = Pyro4.util.excepthook
    print("\n*** except hook installed")
except Exception as e:
    print("Failed ", e)


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
######################## power supply - flux bias ###########################
#############################################################################

# print("Setting up flux bias to current mode, output = 0 ")

# import pymeasure.instruments.yokogawa.yokogawa7651
# yoko = pymeasure.instruments.yokogawa.yokogawa7651.Yokogawa7651("GPIB0::2") #Enter the Yokogawa GPIB address
# yoko.apply_current(max_current=120e-3, compliance_voltage=1)
# yoko.source_current_range = 120e-3
# yoko.source_current = 0
# yoko.enable_source()

#############################################################################
############################### cryoswitch  #################################
#############################################################################


print("Setting up Qphox cryoswitch controller")
cryoswitch_IP = "192.168.0.117"

sys.path.append(r'E:\GitHub\CryoSwitchController')
try:
    switch = Cryoswitch(IP='192.168.0.117') ## -> CryoSwitch class declaration and USB connection
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

