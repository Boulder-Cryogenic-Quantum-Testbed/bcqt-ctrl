from abc import ABC, abstractmethod
from datetime import datetime
from pprint import pprint
import numpy as np
import pyvisa, time

# Abstract Base Class (ABC) for creating drivers for instruments
# class BaseDriver(ABC):
class BaseDriver(ABC):
    
    def __init__(self, InstrConfig_Dict, rm_backend=None, instr_resource=None, instr_address=None, debug=False, **kwargs):
        """
            rm_backend = "@py" or None, depending on if using pyvisa or pyvisa-py
        """
        
        self.debug = debug
        self.rm_backend = rm_backend
        self.instrument_name = InstrConfig_Dict["instrument_name"].upper()
        
        # Open the pyvisa resource manager 
        self.rm = self.open_pyvisa_backend()

        # pick between address and resource
        self.instr_address = InstrConfig_Dict["instr_address"] if "instr_address" in InstrConfig_Dict else None
        self.instr_resource = InstrConfig_Dict["instr_resource"] if "instr_resource" in InstrConfig_Dict else None
        self.instr_config = InstrConfig_Dict
        
        # now connect to instrument using address
        if self.instr_resource is None and self.instr_address is not None:
            # self.print_debug(f"first condition: self.instr_resource is None and self.instr_address is not None\n    {self.instr_resource = }\n    {self.instr_address = }")
            
            self.resource = self.open_pyvisa_resource()
            
        elif self.instr_resource is not None and self.instr_address is None:
            # self.print_debug(f"second condition: self.instr_resource is not None and self.instr_address is None\n    {self.instr_resource = }\n    {self.instr_address = }")
            
            self.resource = instr_resource
            
        else:
            raise ValueError(f"""
                             \n        Both input arguments "instr_resource" and "instr_address" are none, or both are not none. 
                             \n            {instr_address = }\n            {instr_resource = }
                             \n        Provide only one of these to allow connections to the instrument.""")
        
        # full reset of the instrument
        self.write_check("*CLS")    # clears status register and error queue of instrument
        self.write_check("*RST")    # resets to factory default state
        self.write_check("*ESE 1")  # resets the event status registry for *ESR? loops 
                                    #    -> (see the 'send_cmd_and_wait()' method )

        self.idn = self.query_check("*IDN?")
        
        self.model, self.model_no, _, _ = self.idn.split(",")
        
        self.set_default_attrs(**kwargs)
        
        # print instrument parameters
        self.print_debug(self.idn)
        
        self.print_debug(f"resource successfully opened for [{self.instrument_name}]")

        self.print_console("Initialization finished.")

    def __del__(self):
        """
        Deconstructor to free resources
        """
        if self.rm:
            self.rm.close()
            
    @abstractmethod
    def check_instr_error_queue(self, print_output=False):
        
        try:
            cmd = ':SYST:ERR?'
            err = self.resource.query(cmd)
                
            if print_output is True:
                self.print_debug(err)
                self.print_debug(f"checking instr error queue:    {err}")
            
            return err
        
        except Exception as e:
            if type(e) == pyvisa.InvalidSession:    # catch a stupid bug 
                time.sleep(0.25)
                self.handle_InvalidSession_error(cmd, e)
                self.check_instr_error_queue()
            
        
    
    @abstractmethod
    def write_check(self, cmd: str, check_errors: bool = True):
        """
        Writes a command `cmd` and checks for errors
        """
        
        try:
            self.resource.write(cmd)
        except pyvisa.InvalidSession as e:
            print(e)
            self.print_debug("Caught InvalidSession exception in write_check()")
            self.print_debug("Restarting backend and reopening resource)")
        
        err = self.check_instr_error_queue()
        
        if err is None:
            raise ValueError("Query error queue has returned None!?")
        
        # Check that there were no errors
        status, description, = err.split(',')
        status = int(status)
        
        assert not status, f'Error: {description}'
    
    @abstractmethod
    def read_check(self, fmt = str):
        """
        Sends a read command and (is going to) checks for errors
        """
        
        return self.resource.read_raw()
        
        
    @abstractmethod
    def query_check(self, cmd : str, fmt = str):
        """
        Sends a query command `cmd` and checks for errors
        """
        try:
            ret = self.resource.query(cmd)
            
        except Exception as e:  
            if type(e) == pyvisa.InvalidSession:    # catch a stupid bug 
                self.handle_InvalidSession_error(cmd, e)
                ret = self.resource.query(cmd)
                
            if type(e) == pyvisa.VisaIOError:   # likely a timeout
                self.handle_VisaIOError(cmd, e)
                raise e

        err = self.check_instr_error_queue()
        
        # self.print_debug(f"{err=}\n{ret=}")
        
        # Check that there were no errors
        status, description, = err.split(',')
        status = int(status)

        assert not status, f'Error: {description}'

        return fmt(ret)
    
    
    # TODO: identical to query_check, except for the actualy query cmd
    def query_ascii_values(self, cmd : str,container = np.array):
        """
        Sends a query command `cmd` and checks for errors, but
            returns via query_ascii_values
        """
        
        try:
            ret = self.resource.query_ascii_values(cmd, container=container)
            
        except Exception as e:  
            if type(e) == pyvisa.InvalidSession:    # catch a stupid bug 
                self.handle_InvalidSession_error(cmd, e)
                ret = self.resource.query(cmd)
                
            if type(e) == pyvisa.VisaIOError:   # likely a timeout
                self.handle_VisaIOError(cmd, e)
                raise e

        err = self.check_instr_error_queue()
        
        # self.print_debug(f"{err=}\n{ret=}")
        
        # Check that there were no errors
        status, description, = err.split(',')
        status = int(status)

        assert not status, f'Error: {description}'

        return ret
        
    @abstractmethod
    def return_instrument_parameters(self, print_output=False):
        
        # get all methods that start with "get_" and save as list
        all_get_methods = [method_name for method_name in dir(self) 
                           if callable(getattr(self, method_name)) 
                            and "get_" in method_name]
        
        # run through list and call each method using gettattr()
        # all_methods_and_results = []
        # for method in all_get_methods:
        #     # avoid infinite loop :)
        #     if "return_instrument_parameters" in method:
        #         continue
        #     self.print_debug(f"running {method}")
        #     result = getattr(self, method)()
        #     all_methods_and_results.append( (method, result) )
            
        # one liner w/ list comprehension
        all_methods_and_results = [(name, getattr(self, name)()) for name in all_get_methods if ("return_instrument_parameters" not in name and "__" not in name)]
        
        return all_methods_and_results
    
    # @abstractmethod
    def send_cmd_and_wait(self, cmd: str):
        """
            Instead of wrestling with *OPC?, this method uses the 
            Event Status Register (ESR) to let the instrument 
            "announce" when it has finished acquiring
            
            The advantage is that (1) it does not cause the code to 
            stop while waiting for the *OPC?, and more importantly, 
            (2) it lets the code decide where to place the waits
            
            based on flowchart from R&S guide on command synchronizing
            
            https://www.rohde-schwarz.com/us/driver-pages/remote-control/measurements-synchronization_231248.html
        """
        
        # write a 1 to the Event Status Register and then query it
        # to reset the ESR entirely
        self.write_check("*ESE 1") 
        self.query_check("*ESR?") 
        
        # now synchronize the instrument by sending the command
        # we want to synchronize. By adding *OPC at the end,
        # we tell the instrument that it needs to update the
        # ESR once it has finished that string of commands
        
        if not cmd.endswith("*OPC"):
            cmd += ";*OPC"
        
        # send command to be synchronized with *OPC
        self.write_check(cmd)

        # start querying every 0.1s (or every 1s after 1000 tries) to see if
        # the instrument is finished with its command
        try:
            self.print_console(f"Sending {cmd} and waiting:")
            count = 0
            check_if_finished = False
            while check_if_finished is False:
                self.print_console(f"    [{count}]")
                # at count = 10, 100, 1000, increase delay between queries
                # log10(10) = 1.0     =>  delay = 0.1s
                # log10(100) = 2.0    =>   delay = 1s
                # log10(1000) = 3.0   =>   delay = 10s
                #    ... etc
                #  Honestly, I think querying every 1s is slow enough for any scenario
                count += 1
                
                if count <= 100:
                    time.sleep(0.1)
                else:
                    time.sleep(1)
                    
                status = self.read_check("*STB?")
                
                # status should be something like "+1\n"
                if "1" in status: 
                    check_if_finished = True
                
                return True
            
        except Exception as e:
            self.print_console
            
        # return traceData
    
    ###############
    ### helpers ###
    ###############
    def handle_VisaIOError(self, cmd, err):
        self.print_console(f"Failed to run command '{cmd}', with error:    {err}")
        self.print_console(f"pyvisa.VisaIOError:    {err}")
        self.print_console(self.check_instr_error_queue())
    
    
    def handle_InvalidSession_error(self, cmd, err):
        self.print_console(f"Failed to run command '{cmd}', with error:    {err}")
        self.print_console(f"Caught InvalidSession exception in query_check()")
        self.print_console(f"Waiting one second and restarting backend/resource...")
        
        time.sleep(1)
        self.rm = self.open_pyvisa_backend()
        self.resource = self.open_pyvisa_resource()
    

    def print_class_members(self):
        """
        Prints all members in the class
        """
        self.print_console("Printing all object members: ")
        for k, v in self.__dict__.items():
            self.print_console(f'      {k} : {v}')

    def set_default_attrs(self, **kwargs):
        # Update the arguments and the keyword arguments
        # This will overwrite the above defaults with any user-passed kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)
            
            self.print_debug(f"setattr -> self.{k} = {v}")

    def open_pyvisa_backend(self):
        self.print_console(f"Initializing using backend `{"pyvisa-py" if self.rm_backend == "@py" else "pyvisa" }`")
        
        if self.rm_backend is not None:
            rm = pyvisa.ResourceManager(self.rm_backend)
        else:
            rm = pyvisa.ResourceManager()
        
        self.print_debug(f"pyvisa resource manager initialized")
        
        return rm
    
    def open_pyvisa_resource(self):
        resource = self.rm.open_resource(self.instr_address)  # Open the instrument object
        return resource
    
    def hard_reset(self):
        """ 
        Use low-level VISA Library commands to clear the device
        
        see: https://pyvisa.readthedocs.io/en/latest/api/visalibrarybase.html
        """
        
        return self.rm.visalib.clear(self.resource.session)
        
     
    def print_console(self, msg : str = "", prefix : str = None):
        # add prefix to distinguish this instrument from other instruments
        # by default, prefix is [INSTRUMENT_NAME]
        
        if prefix is None:
            msg = f"[{self.instrument_name}]  {msg}".strip()
        else:
            msg = f"[{self.instrument_name}]  {prefix} {msg}".strip()
                    
        print(msg)
            
    def print_debug(self, msg : str = ""):
        if self.debug is True:
            self.print_console(msg, prefix=" **[DEBUG]**  ")
   
  

if __name__ == '__main__':
    
    # %load_ext autoreload
    # %autoreload 2 
    
    Anritsu_InstrConfig = {
        "instrument_name" : "TEST_ANRITSU",
        "rm_backend" : None,
        "amplitude" : 0,
        # "instr_address" : "192.168.0.100",
        "instr_address" : 'GPIB::8::INSTR',  # test instr
        # "instr_address" : 'GPIB::9::INSTR',  # twpa
        
    }
    
    test_anritsu = BaseDriver(Anritsu_InstrConfig, debug=True)
    
    msg = test_anritsu.idn
    test_anritsu.print_console(msg, prefix="self.write(*IDN?) ->")
    
    test_anritsu.return_instrument_parameters(print_output=True)
    
    test_anritsu.print_class_members()
    
    
    
    SA_InstrConfig = {
        "instrument_name" : "R&S SA",
        "rm_backend" : None,
        "amplitude" : 0,
        # "instr_address" : "192.168.0.100",
        "instr_address" : 'GPIB::20::INSTR',  # test instr
        # "instr_address" : 'GPIB::9::INSTR',  # twpa
        
    }
    
    RnS_Instr = BaseDriver(SA_InstrConfig, debug=True)
    
    msg = RnS_Instr.idn
    RnS_Instr.print_console(msg, prefix="self.write(*IDN?) ->")
    
    # test_anritsu.return_instrument_parameters(print_output=True)  # doesnt work because those commands are for sig gen :)
    
    RnS_Instr.print_class_members()
    
    
    
    
    
    
    
    
    
    