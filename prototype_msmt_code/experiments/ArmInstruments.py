"""
    
    ArmInstruments.py
    
    Tired of forgetting to turn the TWPA on or set the switch correctly?
    
    Use this script to check all instrument statuses.
    
    
    
    
    List of Instruments:

        Example Instr
            [#x] (nickname) - IP Address
                -> Device MAC Address
        
        Keysight VNA
            [#1] (PNA-X) - TCPIP0::192.168.0.105::inst0::INSTR'  
                -> 00-13-95-54-90-B5

        Anritsu Signal Generators
            [#1] (Test SigGen) - 'GPIB::8::INSTR'
                
            [#2] (TWPA SigGen) - 'GPIB::9::INSTR'
            
        Rohde & Schwarz Spectrum Analyzer
            [#1] (RnS_SA) - 'GPIB::20::INSTR'
            
        MiniCircuits RF Switch
            [#1] (Switch #1) - 192.168.0.115  
                -> Device Address = D0-73-7F-41-20-B6
        
        MiniCircuits Attenuators
            [#1] (Atten 1) - 192.168.0.113  
                -> Device Address = D0-73-7F-B7-10-32
                
            [#2] (Atten 2) - 192.168.0.114  
                -> Device Address = D0-73-7F-B7-10-34

"""


{ "name" : "Keysight VNA",
 }



