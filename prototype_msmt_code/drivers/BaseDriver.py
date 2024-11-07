from abc import ABC, abstractmethod
from datetime import datetime
from pprint import pprint
import numpy as np
import pyvisa, time

# Abstract Base Class (ABC) for creating drivers for instruments
class BaseDriver():
# class BaseDriver(ABC):
    
    def __init__(self, InstrConfig_Dict, instr_resource=None, instr_address=None, debug=False, **kwargs):
        """
            rm_backend = "@py" or None, depending on if using pyvisa or pyvisa-py
        """
        
        self.debug = debug
        self.instr_config = InstrConfig_Dict        
        self.instrument_name = self.instr_config["instrument_name"].upper()
        self.rm_backend = self.instr_config["rm_backend"] if "rm_backend" in self.instr_config else None

        # pick between address and resource
        self.instr_address = self.instr_config["instr_address"] if "instr_address" in self.instr_config else None
        self.instr_resource = self.instr_config["instr_resource"] if "instr_resource" in self.instr_config else None

        # if self.debug is True:
        #     pyvisa.log_to_screen()
            
        # Open the pyvisa resource manager 
        self.open_pyvisa_backend()
        self.open_pyvisa_resource()
        
        # now connect to instrument using address
          
        # full reset of the instrument, write manually without checking

        self.write_check("*CLS")    # clears status register and error queue of instrument
        self.write_check("*RST")    # resets to factory default state
        self.write_check("*ESE 1")  # resets the event status registry for *ESR? loops 
                                    #    -> (see the 'send_cmd_and_wait()' method )

        self.idn = self.query_check("*IDN?")
        self.model, self.model_no, _, _ = self.idn.split(",")
        
        # set all defaults and announce identity
        self.set_default_attrs(**kwargs)
        self.print_console(f"Resource successfully opened for [{self.instrument_name}]")
        self.print_debug(self.idn)
        
        # if self.debug is True:
        #     self.debug_force_clear()

    def __del__(self):
        """
        Deconstructor to free resources
        """
        if self.rm:
            self.rm.close()
            
            
            
    
    ####################################################
    ##############  instrument operation  ##############
    ####################################################
    
    
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
        
        status, description = self.check_instr_error_queue()
        status = int(status)
        
        assert not status, f'Error: {description}'
        
    def query_check(self, cmd, fmt=str):
        try:
            ret = self.resource.query(cmd)
            return fmt(ret)
            
        except Exception as e:  
            if type(e) == pyvisa.InvalidSession:    # catch a stupid bug 
                self.handle_InvalidSession_error(cmd, e)
                ret = self.resource.query(cmd)
                return fmt(ret)
            
            if type(e) == pyvisa.VisaIOError or type(e) == ValueError:   # likely a timeout
                self.handle_VisaIOError(cmd, e)
                raise e
            
            
    def query_check_ascii(self, cmd : str, container = np.array):
        """
        Sends a query command `cmd` and checks for errors, but
            returns via query_ascii_values
        """
        
        try:
            ret = self.resource.query_ascii_values(cmd, container=container)
            return ret
            
        except Exception as e:  
            if type(e) == pyvisa.InvalidSession:    # catch a stupid bug 
                self.handle_InvalidSession_error(cmd, e)
                ret = self.resource.query(cmd)
                return ret
            if type(e) == pyvisa.VisaIOError:   # likely a timeout
                self.handle_VisaIOError(cmd, e)
                raise e
            
    def query_check_binary(self, cmd : str, container = np.array):
        """
        Sends a query command `cmd` and checks for errors, but
            returns via query_binary
        """
        return NotImplemented
        
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
    
    ####################################################
    ################  obj helpers  #####################
    ####################################################
    
    def print_class_members(self):
        """
            Prints all members in the object class
        """
        self.print_console("\nPrinting all object members: ")
        for k, v in self.__dict__.items():
            self.print_console(f'      {k} : {v}')

    def set_default_attrs(self, **kwargs):
        # Update the arguments and the keyword arguments
        # This will overwrite the above defaults with any user-passed kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)
            self.print_debug(f"setattr -> self.{k} = {v}")

    def open_pyvisa_backend(self):
        self.print_console(f"Initializing pyvisa backend `{self.rm_backend}`")
        if self.rm_backend is not None:
            rm = pyvisa.ResourceManager(self.rm_backend)
        else:
            rm = pyvisa.ResourceManager()
        self.print_debug(f"Pyvisa resource manager successfully initialized")
        self.rm = rm

    def open_pyvisa_resource(self):
        self.print_console(f"Opening resource using backend `{self.rm}`")
        if self.instr_resource is None and self.instr_address is not None:
            self.print_console(f"{self.instr_resource = }, initializing new one with {self.instr_address = }")
            if self.rm is None:
                self.print_debug(f"{self.rm = }, calling self.open_pyvisa_backend()")
                self.open_pyvisa_backend()
            else:
                self.print_debug(f"Found {self.rm = }, calling self.rm.open_resource()")
            resource = self.rm.open_resource(self.instr_address)  # Open the instrument object
            
        elif self.instr_resource is not None and self.instr_address is None:
            self.print_console(f"open_pyvisa_resource() was called, but already found instrument resource? {self.instr_resource = }")
            resource = self.instr_resource
            
        else:
            raise ValueError(f"""
                             \n        Both input arguments "instr_resource" and "instr_address" are none, or both are not none. 
                             \n            {self.instr_address = }\n            {self.instr_resource = }
                             \n        Provide only one the address or the resource to use this driver.""")
        self.resource = resource
    
    
    
    ####################################################
    ##############  instrument utilities  ##############
    ####################################################
    
    @abstractmethod
    def check_instr_error_queue(self, print_output=False):
        """
            uses standard SCPI cmd `:SYST:ERR?` to see if there are errors in the queue
        """
        cmd = ':SYST:ERR?'
        err = self.resource.query(cmd)
            
        if print_output is True:
            self.print_debug(err)
            self.print_debug(f"checking instr error queue:    {err}")
        
        if err is None:
            self.print_console("WARNING -> Query error queue has returned None!?")
            status, description = '0', 'Query error queue returned None'
        else:
            # Check that there were no errors
            status, description = err.split(',')
            
        return status, description
    
    def hard_reset(self):
        """ 
        Use low-level VISA Library commands to clear the device
        
        see: https://pyvisa.readthedocs.io/en/latest/api/visalibrarybase.html
        """
        return self.rm.visalib.clear(self.resource.session)
    
    
    
    ####################################################
    ##################  print methods  #################
    ####################################################
    
    def print_console(self, msg : str = "", prefix : str = None, **kwargs):
        # add prefix to distinguish this instrument from other instruments
        # by default, prefix is [INSTRUMENT_NAME]
        
        if prefix is None:
            new_msg = f"[{self.instrument_name}]  {msg.replace("\n","")}".strip()
        else:
            new_msg = f"[{self.instrument_name}]  {prefix} {msg.replace("\n","")}".strip()
        
        if msg[:1] == "\n":
            self.print_console()  # print an empty line with prefix
            print(new_msg, **kwargs)
        else:   
            print(new_msg, **kwargs)
            
    def print_debug(self, msg : str = "", **kwargs):
        if self.debug is True:
            self.print_console(msg, prefix=" **[DEBUG]**  ", **kwargs)
        
    def print_warning(self, msg, cmd, err, **kwargs):
        self.print_console(msg, prefix=" ****[WARNING]****  ", **kwargs)
        
    def print_error(self, msg, cmd, err):
        self.print_console(f"Failed to run command '{cmd}', with error:    {err}")
        self.print_console(msg)
        
        
    
    ####################################################
    ##################  error handlers  ################
    ####################################################
    
    def handle_VisaIOError(self, cmd, err):
        self.print_error("Caught VisAIOError exception:  ", cmd, err)
        self.print_console(f"Checking instrument error queue...")
        self.print_console(self.check_instr_error_queue())
    
    def handle_InvalidSession_error(self, cmd, err):
        self.print_warning("Caught InvalidSession exception:  ", cmd, err)
        self.print_console(f"Waiting one second and restarting backend/resource...")
        
        time.sleep(1)
        self.rm = self.open_pyvisa_backend()
        if self.rm is None: 
            self.handle_InvalidSession_error()
            
        time.sleep(1)
        self.resource = self.open_pyvisa_resource()
        if self.resource is None: 
            self.handle_InvalidSession_error()
    
        
    ####################################################
    ###############  debugging utilities  ##############
    ####################################################
    
    def debug_read(self, extra=""):
        self.print_debug(f"{extra} Attempting Read:         [{self.debug_writes}]", end="")
        result = self.resource.read()
        self.debug_writes -= 1
        print(f" ---> Success! [{self.debug_writes}]\n" +" "*10+result)
    
    def debug_write(self, cmd, extra="", count=True):
        self.print_debug(f"{extra} Attempting Write ({cmd}) [{self.debug_writes}]", end="")
        self.resource.write(cmd)
        if count is True:
            self.debug_writes += 1
        print(f" ---> Success! [{self.debug_writes}]")
        
    def debug_force_clear(self):
        self.debug_writes = 0
        self.print_debug("Reading until exception occurs!")
        try:
            i = 0
            while True:
                time.sleep(1)
                self.print_console(f" debug_force_clear -> {i}")
                self.debug_read()
                i+=1
        except Exception as e:
            self.debug_writes = 0
            print(f"\n\n{e}\n")
        
    def debug_queue_script(self, sleep_time=0.5, write_cmd_loop="*IDN?", init_cmd=None, test_read_mod=None, test_write_mod=None, test_cmd="*IDN?", ):
        """
            instructive for loop on how the instrument responds to
            a given number of reads and writes. mostly for my own
            learning and debugging why the session closes sometimes
            on write_checks() and query_checks()
        """
        
        if self.debug is not True:
            self.debug = True
            self.print_debug("Forcing debug mode on for debug_queue_script()")
            
        if sleep_time < 0.5:
            self.print_debug("Time between read/writes cannot be faster than 500ms")
            sleep_time = 0.5
            
        self.print_debug("    This script will alternate between sending reads and writes")
        self.print_debug("and will send an extra command at specified intervals.")
        self.print_debug("    The number at beginning of every line is the number of writes")
        self.print_debug("that have not been read. This persists between scripts!!")
        
        if hasattr(self, "debug_writes") is False:
            self.print_debug("\nThis is the first time this script is run- initializing debug_writes=0")
            self.debug_writes = 0
            
        self.print_debug("\nUse KeyboardInterrupt (Ctrl+C or Interrupt) to cancel loop\n\n")
        
        try:
            script_loops = 0
            script_extra_reads = 0
            script_extra_writes = 0
            print()
            if init_cmd is not None:
                time.sleep(sleep_time*2)
                self.debug_write(init_cmd, extra=">>[INIT]<<", count=False)
                                 
            while True:
                self.debug_write(write_cmd_loop)
                time.sleep(sleep_time)
                self.debug_read()
                        
                if test_write_mod is not None:
                    if script_loops % test_write_mod == 0:
                        time.sleep(sleep_time)
                        self.debug_write(test_cmd, extra=">>[EXTRA]<<")
                        script_extra_writes += 1
                
                if test_read_mod is not None:
                    if script_loops % test_read_mod == 0:
                        time.sleep(sleep_time)
                        self.debug_read(extra=">>[EXTRA]<<")
                        script_extra_reads += 1
                        
                script_loops += 1
                
                
        except KeyboardInterrupt:
            self.print_debug("\nCaught KeyboardInterrupt. Stats:")
            stats_dict = {k:v for k, v in locals().items() if "script" in k}
            for k, v in stats_dict.items():
                self.print_debug(f"   {k} = {v}")
            test_anritsu.resource.write("*RST")
            test_anritsu.debug_force_clear()
    
    
    ####################################################
    ####################  Finished!   ##################
    ####################################################
    
if __name__ == '__main__':
    
    """ 
        basically just for testing purposes... this script should never be run on its own, 
        but if you must, remove the "ABC" argument in the class definition at the top
    """
    
    # %load_ext autoreload
    # %autoreload 2 
    
    Anritsu_InstrConfig = {
        "instrument_name" : "TEST_ANRITSU",
        "rm_backend" : None,
        "amplitude" : 0,
        # "instr_address" : "192.168.0.100",
        "instr_address" : 'GPIB::7::INSTR',  # test instr
        # "instr_address" : 'GPIB::8::INSTR',  # twpa
        
    }
    
    test_anritsu = BaseDriver(Anritsu_InstrConfig, debug=True)
    test_anritsu.debug_queue_script(sleep_time=1.25, 
                                    init_cmd="*CLS", test_cmd="*IDN?", 
                                    test_write_mod=1, test_read_mod=None)
    
    
    
    
    # SA_InstrConfig = {
    #     "instrument_name" : "R&S SA",
    #     "rm_backend" : None,
    #     "amplitude" : 0,
    #     "instr_address" : "192.168.0.100",
        
    # }
    
    # RnS_Instr = BaseDriver(SA_InstrConfig, debug=True)
    
    # msg = RnS_Instr.idn
    # RnS_Instr.print_console(msg, prefix="self.write(*IDN?) ->")
    
    # # test_anritsu.return_instrument_parameters(print_output=True)  # doesnt work because those commands are for sig gen :)
    
    # RnS_Instr.print_class_members()
    