from abc import ABC, abstractmethod
from datetime import datetime
from pprint import pprint
from pathlib import Path
import time, sys

sys.path.append(str(Path("..").resolve()))

from BaseDriver import BaseDriver


class SG_Anritsu(BaseDriver):


    def __init__(self, InstrConfig_Dict, instr_resource=None, instr_address=None, debug=False, **kwargs):
        super().__init__(InstrConfig_Dict, instr_resource, instr_address, debug, **kwargs)
        
        if "suppress_warnings" in InstrConfig_Dict:
            self.suppress_warnings = InstrConfig_Dict["suppress_warnings"]
        else:
            self.suppress_warnings = False
        
          
    def read_check(self, fmt = str):
        return super().read_check(fmt=fmt)
    
    def write_check(self, cmd: str):
        return super().write_check(cmd=cmd)
    
    def query_check(self, cmd: str, fmt = str):
        return super().query_check(cmd=cmd, fmt=fmt)
    
    def check_instr_error_queue(self, print_output=False):
        return super().check_instr_error_queue(print_output)
    
    def return_instrument_parameters(self, print_output=False, old_output=False):
        
        instr_params = super().return_instrument_parameters()
            
        if old_output is True and print_output is True:
            power_setting = self.query_check(f'SOUR:POW:LEV:IMM:AMPL?', fmt=int)
            frequency_setting =  self.query_check(f'SOUR:FREQ:CW?', fmt=int)
            output_setting = bool(self.query_check('OUTP:STAT?', fmt=int))
            self.print_console()
            self.print_console(f"  self.return_instrument_parameters:")
            self.print_console(f"     Output = {output_setting}")
            self.print_console(f"     Frequency = {frequency_setting/1e9:1.2f} GHz")
            self.print_console(f"     Power = {power_setting} dBm")
            return power_setting, frequency_setting, output_setting
        
        elif old_output is False and print_output is True: 
            self.print_console(f"  self.return_instrument_parameters:")
            for method, val in instr_params:
                self.print_console(f"     {method} = {val}")
                
        return instr_params
    
    
    # ~~~~~~~~~~~~~~~~~~
    # ~~~  Parameters
    # ~~~~~~~~~~~~~~~~~~

    def get_output(self, print_output=False):
        output_status = bool(self.query_check('OUTP:STAT?', fmt=int))
        if print_output is True:
            self.print_console(f"Output is {output_status}")
        return output_status
            
    def set_output(self, setting: bool = False):
        output_status = self.get_output()
        if output_status == setting:
            self.print_console(f" <WARNING> Output already set to {setting}")
        else:
            self.print_console(f"Switching output from {str(output_status).upper()} to {str(setting).upper()}")
        
        self.write_check(f"OUTP:STAT {int(setting)}")
        
        
    # ~~~~~~~~~

    def get_power(self):
        return self.query_check(f'SOUR:POW:LEV:IMM:AMPL?', fmt=int)
        
    def set_power(self, power_dBm : float, override_safety=False):
        
        if override_safety is True and power_dBm > 20:
            self.print_console(f" <WTF?!> You have overridden the safety, AND you have sent a power greater than 20 dBm.")
            self.print_console(f"                      I've set this as a hardcoded limit, so you'll need to go to this script:")
            self.print_console(f"    {Path(".").absolute()} ")
            self.print_console(f"                      to change this. \n\n    Why is this the case? Well, there's a good chance the Anritsu isn't ")
            self.print_console(f"                      connected to enough attenuation, and this will significantly heat the fridge.")
            
            # comment this line out to disable the safety
            raise PermissionError
            
        elif override_safety is True and power_dBm > 0:
            self.print_console(f"\n\n")
            self.print_console(f" <WARNING> You are sending more than 0 dBm, triggering the code's safety.") 
            self.print_console(f"                        Since you have set override_safety=True, the setting has gone through") 
            self.print_console(f"                        Make sure that you did not drop a minus sign.")
            self.print_console(f"\n\n")
        
        elif power_dBm > 0:
            self.print_console(f"\n\n")
            self.print_console(f" <SAFETY> You are sending more than 0 dBm, triggering the code's safety.") 
            self.print_console(f"                        The command has been aborted. Check that you did not drop a minus sign. ")
            self.print_console(f"\n\n")
            raise ValueError
            
        # power is < 0 unless override_safety == True
        # send power change cmd 
        self.write_check(f'SOUR:POW:LEV:IMM:AMPL {power_dBm} dBm')
        
        power_setting = self.get_power()
        if self.debug: 
            self.print_console(f"Power set to {power_setting} dBm")
    
    # ~~~~~~~~~
        
    def get_freq(self):
        return self.query_check(f'SOUR:FREQ:CW?', fmt=int) 
    
    def set_freq(self, frequency : float, suppress_warnings=False):
        if self.suppress_warnings is False:
            if frequency <= 10e0:  # input is likely in GHz
                self.print_console(f"\n~~~~\nWarning! received input '{frequency}', which seems to be in GHz instead of MHz. \n    Please give value in MHz instead. \n\nSuppress future errors with the argument 'suppress_warnings=True'\n~~~~\n")
                raise ValueError("See printed error above")
            
            # duh, this is the correct one
            elif frequency <= 10e3 >= 10e0:  # input is likely in MHz
                self.print_console(f"\n~~~~\nWarning! received input '{frequency}', which seems to be in MHz instead of Hz. \n    Please give value in Hz instead. \n\nSuppress future errors with the argument 'suppress_warnings=True'\n~~~~\n")
                raise ValueError("See printed error above")
            
            elif frequency <= 10e6: # input is likely in KHz
                self.print_console(f"\n~~~~\nWarning! received input '{frequency}', which seems to be in KHz instead of MHz. \n    Please give value in MHz instead. \n\nSuppress future errors with the argument 'suppress_warnings=True'\n~~~~\n")
                raise ValueError("See printed error above")
                
            elif frequency <= 10e9: # input is likely in Hz, allow code to pass
                pass
                
        # send frequency change cmd
        self.write_check(f'SOUR:FREQ:CW {frequency} HZ') 
        
        frequency_status = self.get_freq()
        if self.debug: 
            self.print_console(f"Frequency set to {frequency_status} ** Hz **")
            
    
            
if __name__ == '__main__':
    
    # %load_ext autoreload
    # %autoreload 2 
    
    Anritsu_InstrConfig = {
        "instrument_name" : "TEST_ANRITSU",
        # "rm_backend" : None,
        "rm_backend" : "@py",
        "amplitude" : 0,
        # "instr_address" : "192.168.0.100",
        # "instr_address" : 'GPIB::8::INSTR',  # test instr
        "instr_address" : 'GPIB::9::INSTR',  # twpa
        
    }
    
    test_anritsu = SG_Anritsu(Anritsu_InstrConfig, debug=True)
    
    msg = test_anritsu.idn
    test_anritsu.set_output(False)
    test_anritsu.print_console(msg, prefix="self.write(*IDN?) ->")
    
    test_anritsu.print_class_members()
    
    # time.sleep(1)
    # test_anritsu.return_instrument_parameters(print_output=True)
    # test_anritsu.set_freq(2000)
    
    # time.sleep(1)
    # test_anritsu.return_instrument_parameters(print_output=True)
    # test_anritsu.set_freq(5000)
    
    # time.sleep(1)
    # test_anritsu.return_instrument_parameters(print_output=True)
    
    
    