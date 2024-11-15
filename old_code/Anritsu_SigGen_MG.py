# -*- coding: utf-8 -*-
import pyvisa, os
import pna_control as pna


class AnritsuCtrl(object):
    """
    Driver for communicating to Anritsu signal generators with SCPI 
    """
    def __init__(self, anritsu_addr=None, vna_addr=None, 
                       rm_backend = None, suppress_warnings=False,
                       verbose=True, *args, **kwargs):
        """
        Class constructor
        
        'rm_backend' = None or '@py', use default pyvisa or use pyvisa-py
        
        """
        
        # define some defaults, using None to check if they were supplied
        default_anritsu_addr = 'GPIB::7::INSTR'
        default_vna_addr =  'TCPIP0::K-N5231B-57006.local::inst0::INSTR'
        
        anritsu_addr = default_anritsu_addr if anritsu_addr is None else anritsu_addr
        vna_addr = default_vna_addr if vna_addr is None else vna_addr
        
        if anritsu_addr != default_anritsu_addr:
            print(f"Not using default anritsu IP address {default_anritsu_addr}\n   - Instead using {anritsu_addr}")
            self.anritsu_addr = anritsu_addr
        else:
            self.anritsu_addr = default_anritsu_addr
            print(f"Using default address for Anritsu -> {default_anritsu_addr}")
        
        if vna_addr != default_vna_addr:
            print(f"Not using default VNA IP address {default_vna_addr}\n   - are you sure this is what you want?")
            self.vna_addr = vna_addr
        else:
            self.vna_addr = default_vna_addr
            print(f"Using default address for VNA -> {default_vna_addr}")
            
            

        # Open the pyvisa resource manager 
        if rm_backend is not None:
            self.rm = pyvisa.ResourceManager(rm_backend)
        else:
            self.rm = pyvisa.ResourceManager()

            
    def __del__(self):
        """
        Deconstructor to free resources
        """
        if self.rm:
            self.rm.close()

    def print_class_members(self):
        """
        Prints all members in the class
        """
        for k, v in self.__dict__.items():
            print(f'{k} : {v}')

    # ~~~~~~~~~~~~~~~~~~~~~
    # ~~~  Original Code
    # ~~~~~~~~~~~~~~~~~~~~~
    
    def vna_process(self, vna_dict : dict, suffix : str = None):
        """
        Performs a PNA measurement

        Parameters:
        ----------

        vna_dict :dict:     dictionary of inputs to the VNA
                            'sample_id'  : output file prefix
                            'centerf'    : center frequency [GHz]
                            'span'       : frequency span [MHz]
                            'temp'       : temperature [mK]
                            'avg'        : number of averages
                            'power'      : output power [dBm]
                            'edelay'     : electrical delay [ns]
                            'ifband_khz' : IF bandwidth [kHz]
                            'npts'       : number of sample points
                            'sparam'     : S-parameter 'S12', 'S21'
                            'cal_set'    : calibration set
                            'data_dir'   : directory to save data in
                            'output_filename' : custom filename
    
        """
        
        prefix = vna_dict['filename_prefix']
        
        # TODO: inconsistent with resonator measurement code, vs. 'output_filename'
        if vna_dict["make_filename"]:
            output_filename = f"{prefix}_{vna_dict["sample_id"]}_{vna_dict["temp"]}mK_{suffix}".replace("__","_").strip("_")
        else:
            output_filename = ''
            
        # Note: PNA power sweep assumes the outputfile has .csv as its last
        # four characters and removes them when manipulating strings and
        # directories
        # outputfile = sampleid+'_'+str(vna_dict['centerf'])+'GHz'
        
        pna.get_data(
                     sample_id = vna_dict['sample_id'],
                     centerf = vna_dict['centerf'],
                     span = vna_dict['span'],
                     temp = vna_dict['temp'],
                     averages = vna_dict['avg'],
                     power = vna_dict['power'],
                     edelay = vna_dict['edelay'],
                     ifband_khz = vna_dict['ifband_khz'],
                     points = vna_dict['npts'],
                     output_filename = output_filename,  # added prefixes and suffixes
                     sparam = vna_dict['sparam'],
                     cal_set = vna_dict['cal_set'],
                     data_dir = vna_dict['data_dir'],
                     filename_suffix = vna_dict['filename_suffix'],
                     instr_addr = self.vna_addr)

    def write_check(self, cmd : str):
        """
        Writes a command `cmd` and checks for errors
        """
        self.resource.write(cmd)
        err = self.resource.query(':SYST:ERR?')

        # Check that there were no errors
        status, description, = err.split(',')
        status = int(status)

        assert not status, f'Error: {description}'

    def read_check(self, cmd : str, fmt = float):
        """
        Sends a query command `cmd` and checks for errors
        """
        ret = self.resource.query(cmd)
        err = self.resource.query(':SYST:ERR?')

        # Check that there were no errors
        status, description, = err.split(',')
        status = int(status)

        assert not status, f'Error: {description}'

        return fmt(ret)
    
    def frequency_sweep(self, sweep_freqs : list, power : float,
            run_vna : bool = False, vna_dict : dict = None):
        """
        Performs sweep from sweep_freqs, at a fixed power

        Parameters:
        ----------

        sweep_freqs     :list:    list of frequencies [GHz]
        power           :float:   fixed power [dBm]
        run_vna         :bool:    run the VNA measurement
        vna_dict        :dict:    parameters to pass to VNA 

        """
        
        # Set the power
        self.write_check(f'SOUR:POW:LEV:IMM:AMPL {power} dBm')
        print(f'Sweeping frequencies {sweep_freqs} GHz at {power} dBm ...')

        fndigits = self.fndigits

        # Iterate over all frequencies
        for freq in sweep_freqs:
            print(f'Measuring with {freq} GHz ...')
            self.write_check(f'SOUR:FREQ:CW {freq} GHZ') 
            is_output_on = self.read_check('OUTP:STAT?', fmt=int)
            if not is_output_on:
                self.write_check('OUTP:STAT ON')
            if run_vna and vna_dict:
                fsuffix = f'Scan_{freq:.{fndigits}f}_GHz_{power}_dBm'
                fsuffix = fsuffix.replace('.', 'p')
                self.vna_process(vna_dict, suffix=fsuffix)

        # Turn off power at the end of the sweep
        self.write_check('OUTP:STAT OFF')

    def power_sweep(self, sweep_powers : list, freq : float,
            run_vna : bool = False, vna_dict : dict = None):
        """
        Performs sweep of power in sweep_powers, at a fixed frequency

        Parameters:
        ----------

        sweep_powers    :list:    list of powers [dBm]
        freq            :float:   fixed frequency [GHz]
        run_vna         :bool:    run the VNA measurement
        vna_dict        :dict:    parameters to pass to VNA 

        """
        
        # Set the power
        self.write_check(f'SOUR:FREQ:CW {freq} GHZ') 
        
        print(f'Sweeping powers {sweep_powers} dBm at {freq} GHz ...')
        fndigits = self.fndigits

        # Iterate over all frequencies
        for power in sweep_powers:
            print(f'Measuring with {power} dBm ...')
            self.write_check(f'SOUR:POW:LEV:IMM:AMPL {power} dBm')
            is_output_on = self.read_check('OUTP:STAT?', fmt=int)
            if not is_output_on:
                self.write_check('OUTP:STAT ON')
            if run_vna and vna_dict:
                fsuffix = f'Scan_{freq:.{fndigits}f}_GHz_{power}_dBm'
                fsuffix = fsuffix.replace('.', 'p')
                self.vna_process(vna_dict, suffix=fsuffix)

        # Turn off power at the end of the sweep
        self.write_check('OUTP:STAT OFF')

    def power_frequency_sweep_2d(self, 
                                 sweep_powers : list,
                                 sweep_freqs : list,
                                 sweep_order : str = 'power_frequency',
                                 run_vna : bool = False,
                                 vna_dict : dict = None):
        
        """
        Performs 2D power and frequency from two lists
        
        Parameters:
        ----------

        sweep_powers    :list:    list of powers [dBm]
        sweep_freqs     :list:    list of frequencies [GHz]
        sweep_order     :str:     'power_frequency'
        run_vna         :bool:    run the VNA measurement
        vna_dict        :dict:    parameters to pass to VNA 

        """
        
        if vna_dict["filename_suffix"] == '' or vna_dict["filename_suffix"] is None:
            add_filename_suffix = True
        else:
            add_filename_suffix = False
            
        base_dir = vna_dict["data_dir"]
        
        # Check for the sweep order flag
        if sweep_order == 'power_frequency':
            for power in sweep_powers:
                
                if add_filename_suffix:
                    vna_dict["filename_suffix"] = f"{power}_dBm"
                    
                vna_dict["data_dir"] = base_dir /  f"{power}_dBm"
                
                self.frequency_sweep(sweep_freqs, power,
                                     run_vna=run_vna, vna_dict=vna_dict)
                
        elif sweep_order == 'frequency_power':
            for freq in sweep_freqs:
                
                if add_filename_suffix:
                    vna_dict["filename_suffix"] = f"{freq}_GHz"
                    
                vna_dict["data_dir"] = base_dir / f"{freq}_GHz"
                
                self.power_sweep(sweep_powers, freq,
                                 run_vna=run_vna, vna_dict=vna_dict)
        else:
            raise ValueError(f'Sweep order {sweep_order} not recognized.')

        # finally, do a reference power_sweep
        
        vna_dict_ref = vna_dict
        vna_dict_ref["filename_suffix"] = f"REFERENCE_{power}dBm"
        vna_dict_ref["filename_suffix"] = f"REFERENCE_{power}dBm"

    # ~~~~~~~~~~~~~~~~~~
    # ~~~  Parameters
    # ~~~~~~~~~~~~~~~~~~

    def set_output(self):
        output_status = self.confirm_output()
        if output_status is True:
            print("--[ANRITSU] <WARNING> Output is already on! Instrument is configured to:")
            self.get_instrument_parameters(print_output=True)
            
    def get_output(self, print_output=False):
        output_status = bool(self.read_check('OUTP:STAT?', fmt=int))
        if print_output is True:
            print(f"Output is {output_status}")
        return output_status
            
    # ~~~~~~~~~

    def set_power(self, power_dBm : float, override_safety=False):
        
        if override_safety is True and power_dBm > 20:
            print(f"--[ANRITSU] <WTF?!> You have overridden the safety, AND you have sent a power greater than 20 dBm.")
            print(f"                      I've set this as a hardcoded limit, so you'll need to go to \n{os.getcwd()} to ")
            print(f"                      change this. Why is this the case? Well, there's a good chance the Anritsu isn't ")
            print(f"                      connected to enough attenuation, and this will significantly heat the fridge.")
            
            # comment this line out to disable the safety
            raise PermissionError
            
        elif override_safety is True and power_dBm > 0:
            print(f"\n\n")
            print(f"--[ANRITSU] <WARNING> You are sending more than 0 dBm, triggering the code's safety.") 
            print(f"                        Since you have set override_safety=True, the setting has gone through") 
            print(f"                        Make sure that you did not drop a minus sign.")
            print(f"\n\n")
        
        elif power_dBm > 0:
            print(f"\n\n")
            print(f"--[ANRITSU] <SAFETY> You are sending more than 0 dBm, triggering the code's safety.") 
            print(f"                        The command has been aborted. Check that you did not drop a minus sign. ")
            print(f"\n\n")
            raise ValueError
            
        # power is < 0 unless override_safety == True
        # send power change cmd 
        self.write_check(f'SOUR:POW:LEV:IMM:AMPL {power_dBm} dBm')
        
        if self.verbose: 
            print(f" --[ANRITSU] Power set to {power_dBm} dBm")
    
    def get_power(self):
        return self.read_check(f'SOUR:POW:LEV:IMM:AMPL?', fmt=int)
        
    # ~~~~~~~~~
        
    def set_freq(self, frequency : float, suppress_warnings=False):
        if self.suppress_warnings is False:
            if frequency <= 10e0:  # input is likely in GHz
                print(f"\n~~~~\nWarning! received input '{frequency}', which seems to be in GHz instead of MHz. \n    Please give value in MHz instead. \n\nSuppress future errors with the argument 'suppress_warnings=True'\n~~~~\n")
                raise ValueError
            
            # duh, this is the correct one
            elif frequency <= 10e3 >= 10e0:  # input is likely in MHz
                pass
            
            elif frequency <= 10e6: # input is likely in KHz
                print(f"\n~~~~\nWarning! received input '{frequency}', which seems to be in KHz instead of MHz. \n    Please give value in MHz instead. \n\nSuppress future errors with the argument 'suppress_warnings=True'\n~~~~\n")
                raise ValueError
                
            elif frequency <= 10e9: # input is likely in Hz
                print(f"\n~~~~\nWarning! received input '{frequency}', which seems to be in Hz instead of MHz. \n    Please give value in MHz instead. \n\nSuppress future errors with the argument 'suppress_warnings=True'\n~~~~\n")
                raise ValueError
                
        # send frequency change cmd
        self.write_check(f'SOUR:FREQ:CW {frequency} MHZ') 
        if self.verbose: 
            print(f" --[ANRITSU] Frequency set to {frequency} **MHz**")
            
    def get_freq(self):
        return self.read_check(f'SOUR:FREQ:CW?', fmt=int) 
            
    # ~~~~~~~~~
            
    def get_instrument_parameters(self, print_output=False):
        """
        returns result of three queries:
            (1) is the output on? 
            (2) what is the freq?
            (3) what is the amplitude?
            
        and optionally prints a message in the console
            
        based on too many measurements made with the TWPA off :)
        """
        
        # Set the power
        output_setting = self.get_output()
        frequency_setting = self.get_freq()
        power_setting = self.get_power()
        
        if print_output is True:
            print(f"--[ANRITSU]  Instrument status:")
            print(f"     Output = {output_setting}")
            print(f"     Frequency = {frequency_setting}")
            print(f"     Power = {power_setting}")
        
        return output_setting, frequency_setting, power_setting

    
    
    
    
    
    
    
    
    