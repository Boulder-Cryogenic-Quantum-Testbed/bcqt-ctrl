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

class MC_VarAttenuator():
    
    def __init__(self, device_address, timeout=5, debug=True):
        
        self.debug = debug
        self.device_name = "Attenuator"
        self.device_address = device_address
        self.timeout = timeout
        
        
        if self.debug is True:
            print(self.Get_Model_Name())
            print(self.Get_Serial_No())
            
        print(self.Get_Attenuation())
    
    
    def Format_PTE_Return(self, PTE_Return):
        # example output:
        #   b'MN=RCDAT-8000-30'
        # split and return the parts on either side
        #   of the '=', and remove b and '
        
        PTE_Return = str(PTE_Return).replace("b'", "").replace("'","").strip()
        
        try:
            var, result = PTE_Return.split("=")
        except:
            var, result = self.device_name, PTE_Return
            
        
        if self.debug is True:
            print(var, result)
            
        return var, result
    
    def Get_HTTP_Result(self, CmdToSend):

        # Specify the IP address of the attenuator
        CmdToSend = f"http://{self.device_address}/:{CmdToSend}"

        # Send the HTTP command and try to read the result
        try:
            HTTP_Result = urlopen(CmdToSend, timeout=self.timeout)
            PTE_Return = HTTP_Result.read()

            # # The attenuator displays a web GUI for unrecognised commands
            # if len(PTE_Return) > 100:
            #     print ("Error, command not found:", CmdToSend)
            #     PTE_Return = "Invalid Command!"

        # Catch an exception if URL is incorrect (incorrect IP or disconnected)
        except Exception as e:
            print ("Error, no response from device; check IP address, command, and connections.")
            PTE_Return = "No Response!"
            print((f"{CmdToSend = }"))
            raise e

        var, result = self.Format_PTE_Return(PTE_Return)
        
        # Return the formatted response
        return var, result


    def Get_Model_Name(self):
        return self.Get_HTTP_Result("MN?")


    def Get_Serial_No(self):
        return self.Get_HTTP_Result("SN?")


    def Get_Attenuation(self):
        return self.Get_HTTP_Result("ATT?")        	
        

    def Set_Attenuation(self, attenuation: float):
        
        cmd = f"SETATT={attenuation:1.3f}"
        
        if self.debug is True:
            print(f"Sending {cmd = }")
        
        var, result = self.Get_HTTP_Result(cmd)   # Send command
        
        if self.debug is True:  # Print attenuator
            print(f"Set attenuation on {self.device_address} to\n    {self.Get_Attenuation()} dB")       
                
        if result == '1':
            result = f"Success: Atten={self.Get_Attenuation()[1]}"
        else:
            result = f"Failed to set {cmd}"
            
        return var, result


if __name__ == "__main__":
    
    ip_addr_1 = "192.168.0.113"  # atten #1
    ip_addr_2 = "192.168.0.114"  # atten #2
    
    atten_1 = MC_VarAttenuator(ip_addr_1)
    
    atten_1.Set_Attenuation(1.5)
    