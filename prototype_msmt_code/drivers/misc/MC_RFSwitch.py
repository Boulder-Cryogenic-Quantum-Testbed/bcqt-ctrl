"""
    MiniCircuits RF Switch
    
    Coded specifically for the RC-4SPDT-A18 model

    Loosely based on their example python code on their website
    
    https://www.minicircuits.com/WebStore/pte_example_download.html?fam=Mechanical%20Switch%20Box
    
    # TODO: could combine with other minicircuits drivers with an abstract class...
    #       probably not worth the effort?

"""

from urllib.request import urlopen
import sys

class MC_RFSwitch():
    
    def __init__(self, device_address="192.168.0.115", timeout=5, debug=True):
        
        self.debug = debug
        self.device_address = device_address
        self.timeout = timeout
        
    
        if self.debug is True:
            print(self.Get_Model_Name())
            print(self.Get_Serial_No())
            print(self.Get_Attenuation())
    
    
    
    def Get_HTTP_Result(self, CmdToSend):

        # Specify the IP address of the switch box
        CmdToSend = f"http://{self.device_address}:{CmdToSend}"

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


    def Set_Switch_State(self, switch: str, contact: int):
        # switch = A, B, C, D
        # contact = 0 for blue, 1 for red
        
        if len(switch) != 1:
            raise ValueError("Switch variable must ba a string of length 1 -> e.g. 'A', 'B', 'C', or 'D'")
        
        if contact != 0 or contact != 1:
            raise ValueError("Contact argument must be an integer - either '0' for blue, or '1' for red switch state!")
        
        cmd = f"SET{switch}={contact}"
        
        if self.debug is True:
            print(f"Sending {cmd = }")
        
        status = self.Get_HTTP_Result(cmd)   # Send switch command
        
        if self.debug is True:  # Print switch position
            print(f"Command sent -> new switch status {self.Get_HTTP_Result("SWPORT?")}")       
            
        return status


if __name__ == "__main__":
    
    ip_addr = "192.168.0.115"  
    
    atten = MC_RFSwitch(ip_addr, debug=True)
    
    pass