from abc import ABC, abstractmethod
from datetime import datetime
from pprint import pprint
from pathlib import Path
import numpy as np
import time

from BaseDriver import BaseDriver



class VNA_Keysight(BaseDriver):
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~  Base Class Features
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def __init__(self, InstrConfig_Dict, rm_backend="@py", instr_resource=None, instr_address=None, debug=False, **kwargs):
        super().__init__(InstrConfig_Dict, rm_backend, instr_resource, instr_address, debug, **kwargs)
        
    def read_check(self, fmt = str):
        return super().read_check(fmt)
    
    def write_check(self, cmd: str):
        return super().write_check(cmd=cmd)
    
    def query_check(self, cmd, fmt = str):
        return super().query_check(cmd, fmt)
    
    def check_instr_error_queue(self, print_output=False):
        return super().check_instr_error_queue(print_output)
    
    def return_instrument_parameters(self, print_output=False):
        return super().return_instrument_parameters(print_output)
    
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~  get/set Instr Parameters
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    def set_instr_params(self, InstrConfig_Dict=None):
        
        if InstrConfig_Dict is None:
            configs = self.instr_config
        else:
            configs = InstrConfig_Dict
        
        self.configs = configs
        
        # TODO: do this better
        # self.configs.power = configs["power"] 
        # self.configs.fc = configs["fc"] 
        # self.configs.span = configs["span"] 
        # self.configs.points = configs["points"] 
        # self.configs.averages = configs["averages"] 
        # self.configs.if_bandwidth = configs["if_bandwidth"] 
        # self.configs.edelay = configs["edelay"] 
        # self.configs.sparam = configs["sparam"] 
        
    
    def get_instr_params(self):
        
        if hasattr(self, "configs") is not True:
            # TODO: make a warning feature
            self.print_console("get_instr_params called without having configured any parameters", prefix="[WARNING]")
            return None
        
        power = self.configs["power"]
        fc = self.configs["fc"]
        span = self.configs["span"]
        points = self.configs["points"]
        averages = self.configs["averages"]
        if_bandwidth = self.configs["if_bandwidth"]
        edelay = self.configs["edelay"]
        sparam = self.configs["sparam"]
        
        all_params = { 
                     "power" : power , 
                     "fc" : fc , 
                     "span" : span , 
                     "points" : points , 
                     "averages" : averages , 
                     "if_bandwidth" : if_bandwidth , 
                     "edelay" : edelay , 
                     "sparam" : sparam ,
                    }
        
        for k, v in all_params.items():
            self.print_console(f" {k} = {v}")
        
        return all_params
        
        
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~  Instr Methods
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def create_sweep(self):
        
        # if segments:
        #     num_segments = len(segments)
        #     seg_data = ''.join([s for s in segments])
        #     self.write_check(f"SENSe1:SWEep:TYPE SEGment")
        #     self.write_check(f'SENSe1:SEGMent:LIST SSTOP, {num_segments}{seg_data}')
        # else:
        #     self.write_check("SENSe1:SWEep:TYPE LINear")
        #     self.write_check(f'SENSe1:SWEep:POINts {points}')
        #     self.write_check(f'SENSe1:FREQuency:CENTer {fc}HZ')
        #     self.write_check(f'SENSe1:FREQuency:SPAN {span}HZ')

        #     self.write_check(f'SENSe1:SWEep:TIME:AUTO ON')
            
        pass
        
    def compute_homophasal_segments(self, fc, span, Noffres=None, points=None, segment_type='homophasal', **kwargs):
        """
            Computes segments needed to perform homophasal measurements
                "segments" are just strings that contain the parameters
                for each frequency "slice" we are splitting our x-axis into
                
                A guide for the meaning of each part of the string
                    example = ', 1, {Noffres}, {fstop*fscale}, {fb}',
                              ', 1, <# of pts>, <start freq>, <stop freq>
        """
        
        # conversion factor for MHz -> Hz
        fscale = 1 if fc >= 1e6 else 1e6
        
        # conversion factor for GHz -> Hz  (overwrites previous)
        fscale = 1 if fc >= 1e9 else 1e9
        
        # Estimate the number of linewidths per sweep
        Q = 20 * (fc / span) 

        # Compute the frequencies
        fstart = fc - span / 2
        fstop  = fc + span / 2
        
        # determine homophasal
        theta0 = np.pi / 32
        Nf = 30 if points is None else points
        theta = np.linspace(-np.pi + theta0, (np.pi - theta0), Nf + 2)
        freq = fc * (1 - 0.5 * np.tan(theta / 2) / Q)           
            
        if segment_type == 'homophasal':
            # homophasal for entire freq range
            segments = [f',1,2,{ff1*fscale},{ff2*fscale}'
                    for ff1, ff2 in zip(freq[0::2], freq[1::2])]
        
        elif segment_type == 'hybrid':
            assert Noffres is not None
            
            # split between homophasal and linear
            #   homophasal near resonance   freqs = [fa->fb]
            #   linear off resonance        freqs = [fstart->fa] and [fb->fstop]
            hsegments = [f',1,2,{ff1*fscale},{ff2*fscale}'
                    for ff1, ff2 in zip(freq[0::2], freq[1::2])][1:-1]
            fa = np.min(freq[1:-1]) * fscale
            fb = np.max(freq[1:-1]) * fscale

            segments = [f',1,{Noffres},{fstop*fscale}, {fb}',
                        *hsegments,
                        f',1,{Noffres},{fa},{fstart*fscale}']
        else:
            # simple linear sweep
            segments = [f',1,{points},{fstop*fscale}, {fstart*fscale}']
            
        return segments
    
    def setup_measurement(self, Expt_Config=None):
        
        '''
            set parameters for the PNA for the sweep (number of points, center
            frequency, span of frequencies, IF bandwidth, power, electrical delay and
            number of averages)

            XXX: Do not change this order:

            1.  Define a measurement
            2.  Turn on display
            3.  Set the number of points
            4.  Set the center frequency, span
            5.  Turn on sweep time AUTO
            6.  Set the electrical delay
            7.  Turn on interpolation
            8.  Set the calibration
            9.  Set the power
            10. Turn on averaging
            11. Set the IF bandwidth

        '''
        
        if Expt_Config is not None:
            self.print_console("Updating Expt_Config...")
            self.set_instr_params(Expt_Config)
        
        self.print_console("Initializing VNA...")
        self.write_check('*RST')
        self.write_check('*CLS')

        # self.write_check('SYSTem:FPRESet')
        self.write_check('SYSTem:UPRESet')
        time.sleep(0.05)
        self.write_check('OUTPut:STATe OFF')

        # Initial setup for measurement
        ## Query the existing measurements
        measurements = self.query_check('CALC1:PAR:CAT:EXTended?')

        ## If any measurements exist, delete them all
        if measurements != 'NO CATALOG':
            self.write_check('CALC1:PARameter:DELete:ALL')
            
        # create measurements
        self.write_check(f'CALC1:MEASure1:DEFine \"{self.configs["sparam"]}\"')
        self.write_check(f'CALC1:MEASure2:DEFine \"{self.configs["sparam"]}\"')

        
        if self.configs["segments"] is not None:
            num_segments = len(self.configs["segments"])
            seg_data = ''.join([s for s in self.configs["segments"]])
            self.write_check(f"SENSe1:SWEep:TYPE SEGment")
            self.write_check(f'SENSe1:SEGMent:LIST SSTOP, {num_segments}{seg_data}')
        else:
            self.write_check("SENSe1:SWEep:TYPE LINear")
            self.write_check(f'SENSe1:SWEep:POINts {self.configs["points"]}')
            self.write_check(f'SENSe1:FREQuency:CENTer {self.configs["fc"]}HZ')
            self.write_check(f'SENSe1:FREQuency:SPAN {self.configs["span"]}HZ')

            self.write_check(f'SENSe1:SWEep:TIME:AUTO ON')
        
        self.write_check(f'SOUR1:POW1 {self.configs["power"]}')
        self.write_check(f'SENSe1:AVERage:STATe ON')
        self.write_check(f'SENSe1:BANDwidth {self.configs["if_bandwidth"]}HZ')

        # configure ch 1 measurement 1
        self.write_check(f'CALC1:PAR:MNUM 1')  # select ch 1, meas 1
        self.write_check(f'DISPlay:WINDow1 ON')  # create window
        self.write_check(f'DISPlay:MEAS1:FEED 1')  # display meas 1 on window 1
        self.write_check(f'CALC1:CORRection:EDELay:TIME {self.configs["edelay"]}NS')
        self.write_check(f'CALC1:MEASure1:FORMat MLOGarithmic')

        # configure ch 1 measurement 2
        self.write_check(f'CALC1:PAR:MNUM 2')  # select ch 1, meas 2
        self.write_check(f'DISPlay:WINDow2 ON')  # create window 2
        self.write_check(f'DISPlay:MEAS2:FEED 2')  # display meas 2 on window 2
        self.write_check(f'CALC1:CORRection:EDELay:TIME {self.configs["edelay"]}NS')
        self.write_check(f'CALC1:MEASure2:FORMat PHASe')

        # autoscale for visibility on the display
        self.write_check(f'DISPlay:WINDow1:TRACe1:Y:SCAle:AUTO')
        self.write_check(f'DISPlay:WINDow2:TRACe1:Y:SCAle:AUTO')

        # make sure to have averages as an integer
        self.write_check(f'SENSe1:AVERage:Count {self.configs["averages"] // 1}')

    
    def acquire_trace(self):
            
        # initiate display and turn on output
        # self.write_check('OUTPut:STATe ON')
        
        # self.write_check('SENS1:SWE:MODE SINGle')  
        # self.query_check('*OPC?')  
        # self.write_check('INIT:IMM')  # just use INIT:IMM to trigger one sweep
        # self.write_check('FORMat ASCII')
        # self.write_check('DISPlay:WINDow1:Y:AUTO')
        # self.write_check('DISPlay:WINDow2:Y:AUTO')

        # initiate display and turn on output
        # self.write_check('OUTPut:STATe ON')
        # self.write_check('ABORT;INITIATE:IMMEDIATE')  # just use INIT:IMM to trigger one sweep
        # self.write_check('FORMat ASCII')
        # self.write_check('DISPlay:WINDow1:Y:AUTO')
        # self.write_check('DISPlay:WINDow2:Y:AUTO')
                
                
        self.write_check('OUTPut:STATe ON')
        self.write_check('INITiate:CONTinuous ON')
        self.write_check('FORMat ASCII')
        self.write_check('DISPlay:WINDow1:Y:AUTO')
        self.write_check('DISPlay:WINDow2:Y:AUTO')
        
        
        # check if the VNA has finished every second, in my experience the *OPC? or *WAI command isnt very reliable
        check = False
        tstart = time.time()
        
        while check is False:
            time.sleep(1)
            t_elapsed = time.time() - tstart
            print(f"      time elapsed: [{t_elapsed:1.0f}s]")
                
            # check_str is a string, "0" = busy or "1" = complete
            check_str = self.query_check('STAT:OPER:AVER1:COND?')[1]

            # once it is "1", print that we're finished
            if check_str != "0":
                print(f"\nTrace finished. Uploading now.")
                print(f"\n   Total time elapsed: {t_elapsed:1.0f} seconds")
                if t_elapsed >= 600:
                    print(f"                     = {t_elapsed/60:1.1f} minutes \n")
                
                # update the variable and let the while finish
                check = bool(check_str)


    def return_data(self):
                
        if self.configs["segments"] is not None:
            # Read the list of all segments
            freqs = np.array([])
            for s in self.configs["segments"]:
                ssplit = s.replace(" ", "").split(',')
                
                # int() doesnt want a string of a float like '12.0', so if it has  
                # a decimal point, turn it into a float first
                nf = int(ssplit[2]) if '.' not in ssplit[2] else int(float(ssplit[2]))  
                f1 = float(ssplit[3])
                f2 = float(ssplit[4])
                f = np.linspace(f1, f2, nf)
                freqs = np.hstack((freqs, f))
        else:
            gpoints = int(self.query_check(f'SENSe1:SWEep:POINts?'))
            freqs = np.linspace(float(self.query_check('SENSe1:FREQuency:START?')),
                    float(self.query_check('SENSe1:FREQuency:STOP?')), gpoints)
                        
        # read in magn
        self.write_check('CALC1:PAR:MNUM 1')  # select ch 1, meas 1
        self.write_check('CALC1:FORMat MLOG')
        magn = self.query_ascii_values('CALC1:DATA? FDATA', container=np.array)

        # read in phase
        self.write_check('CALC1:PAR:MNUM 2')  # select ch 1, meas 2
        self.write_check('CALC1:FORMat PHASe')
        self.write_check('DISPlay:WINDow2:Y:AUTO')
        phase = self.query_ascii_values('CALC1:DATA? FDATA', container=np.array)

        
        return freqs, magn, phase
    
    
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~  Instr Scripts
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def take_single_trace(self, Expt_Config = None):
        
        # TODO: separate "instr_config" and "expt_config"
        if self.instr_config is None and Expt_Config is not None: 
            self.instr_config = Expt_Config
        
        self.instr_config["segments"] = self.compute_homophasal_segments(**self.instr_config)
        self.set_instr_params(Expt_Config)
        self.get_instr_params()
        self.setup_measurement()
        self.check_instr_error_queue()
        self.acquire_trace()
        freqs, magn_dB, phase_deg = self.return_data()
        
        return freqs, magn_dB, phase_deg