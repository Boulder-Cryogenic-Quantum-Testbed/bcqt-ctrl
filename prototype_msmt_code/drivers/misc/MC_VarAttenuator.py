"""
    MiniCircuits Variable Attenuator
    
    Coded specifically for the RCDAT-8000-30 model

    Loosely based on their example python code on their website
    https://www.minicircuits.com/WebStore/pte_example_download.html?fam=Programmable%20Attenuator


    # # Single channel models
    # Set_1 = Get_HTTP_Result("SETATT=10.75")   	# Set attenuation
    # print ("Set Attenuation:", str(Set_1))

    # Read_1 = Get_HTTP_Result("ATT?")        	# Read attenuation
    # print ("Read Attenuation: ", str(Read_1))



"""

from urllib.request import urlopen
import sys

class MiniCircuits_RFSwitch():
    
    def __init__(self, device_address, timeout=5, debug=True):
        
        self.debug = debug
        self.device_address = device_address
        self.timeout = timeout
    
    
    def Get_HTTP_Result(self, CmdToSend):

        # Specify the IP address of the switch box
        CmdToSend = f"http:///{self.device_address}:{CmdToSend}"

        # Send the HTTP command and try to read the result
        try:
            HTTP_Result = urlopen(CmdToSend, timeout=self.timeout)
            PTE_Return = HTTP_Result.read()

            # The switch displays a web GUI for unrecognised commands
            if len(PTE_Return) > 100:
                print ("Error, command not found:", CmdToSend)
                PTE_Return = "Invalid Command!"

        # Catch an exception if URL is incorrect (incorrect IP or disconnected)
        except:
            print ("Error, no response from device; check IP address and connections.")
            PTE_Return = "No Response!"
            raise ConnectionError

        # Return the response
        return PTE_Return


    def Get_Model_Name(self):
        return self.Get_HTTP_Result("MN?")


    def Get_Serial_No(self):
        return self.Get_HTTP_Result("SN?")


    def Get_Attenuation(self):
        return self.Get_HTTP_Result("ATT?")        	
        

    def Set_Attenuation(self, attenuation: float):
        
        cmd = f"SETATT={attenuation:1.2f}"
        
        if self.debug is True:
            print(f"Sending {cmd = }")
        
        status = self.Get_HTTP_Result(cmd)   # Send switch command
        
        if self.debug is True:  # Print switch position
            print(f"Command sent -> new switch status {self.Get_HTTP_Result("SWPORT?")}")       
                    
        return status
