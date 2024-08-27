"""
# setup_qick_measurement.py

08/16/24

### Purpose

    Initializes qick related instrumentation, specifically the qick board itself, its programmable attenuator,

    e.g. global reference
        %run E:\GitHub\bcqt-ctrl\setup_qick_board
        
    or, if in Cooldown_Data\Cooldown_XX
        %run ..\..\setup_qick_board
"""

#############################################################################
####################### generic imports and paths ###########################
#############################################################################

import sys, os

print(f"Running setup_qick_board.py from {os.getcwd()}...")

print(f"Starting with setup_measurement.py")

sys.path.append(r'E:\GitHub\bcqt-ctrl\resonator_measurements')

%run E:\GitHub\bcqt-ctrl\resonator_measurements\setup_measurement


#############################################################################
##################### connect to pyro 4 nameserver ##########################
#############################################################################

import Pyro4
from qick.pyro import make_proxy

Pyro4.config.SERIALIZER = "pickle"
Pyro4.config.PICKLE_PROTOCOL_VERSION=4

ns_host = "192.168.0.99"  # set by router DHCP
ns_port = 8888
proxy_name = "BCQT_QICK"

print(f"Connecting to server {ns_host}:{ns_port} as a client under proxy '{proxy_name}'...")

ns = Pyro4.locateNS(host=ns_host, port=ns_port)
soc, soccfg = make_proxy(ns_host, ns_port, proxy_name)

try:
    print("\n*** installing pyro's excepthook")
    sys.excepthook = Pyro4.util.excepthook
    print("\n*** except hook installed")
except Exception as e:
    print("\n*** except hook failed to install... error:\n",e)


#############################################################################
############################### attenuators #################################
#############################################################################

# print("Setting up attenuators ")


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

