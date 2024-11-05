from abc import ABC, abstractmethod
from datetime import datetime
from pprint import pprint
from pathlib import Path
import numpy as np
import time

from BaseDriver import BaseDriver


class SA_RnS_FSEB20(BaseDriver):

    def __init__(self, InstrConfig_Dict, instr_resource=None, instr_address=None, debug=False, **kwargs):
        super().__init__(InstrConfig_Dict, instr_resource, instr_address, debug, **kwargs)
          
    def read_check(self, fmt = str):
        return super().read_check(fmt=fmt)
    
    def write_check(self, cmd: str):
        return super().write_check(cmd=cmd)
    
    def query_check(self, cmd: str, fmt = str):
        return super().query_check(cmd=cmd, fmt=fmt)
    
    def check_instr_error_queue(self, print_output=False):
        return super().check_instr_error_queue(print_output)
    
    def return_instrument_parameters(self, print_output=False):
        return super().return_instrument_parameters(print_output)
    
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~  get/set Instr Parameters
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    def get_IF_bandwidth(self, fmt=float):
        return self.query_check('SENS:BAND?', fmt=fmt)

    def set_IF_bandwidth(self, IFBW):
        self.write_check(f'SENS:BAND {IFBW}')
        setBwHz = self.get_IF_bandwidth()
        if self.debug is True:
            self.print_console(f'    Set spectrum analyzer IF bandwidth = {setBwHz} Hz')
    
    
    # ~~~~

    def get_freq_center_Hz(self, fmt=float):
        return self.query_check('FREQ:CENT?', fmt=fmt)
        
    def set_freq_center_Hz(self, freq_center_Hz):
        self.write_check(f'FREQ:CENT {freq_center_Hz} Hz')
        setCenterFreqHz = self.get_freq_center_Hz()
        if self.debug is True:
            self.print_console(f'    Set spectrum analyzer center freq = {setCenterFreqHz} Hz')
    
    # ~~~~
    
    def get_freq_span_Hz(self, fmt=float):
        return self.query_check('FREQ:SPAN?', fmt=fmt)
    
    def set_freq_span_Hz(self, freq_span_Hz):
        self.write_check(f'FREQ:SPAN {freq_span_Hz}')
        setFreqSpanHz = self.get_freq_span_Hz()
        if self.debug is True:
            self.print_console(f'Set spectrum analyzer freq span = {setFreqSpanHz} Hz')
    

    # ~~~~
    
    def set_num_averages(self, fmt=int):
        return self.query_check('AVER:COUN?', fmt=fmt)

    def set_num_averages(self, numAvg):
        self.write_check(f'AVER:COUN {numAvg}')
        setNumAvg = self.set_num_averages()
        if self.debug is True:
            self.print_console(f'Set spectrum analyzer averaging = {setNumAvg}')
    

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~  Instr Methods
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    def toggle_continuous_sweep(self, sweep_mode):
        state = self.Bool2OnOff(state) 
        self.write_check(f'INIT:CONT {sweep_mode}' )
        continuousSweepState = self.query_check('INIT:CONT?')
        if self.debug is True:
            self.print_console(f'    Continuous sweep mode = {continuousSweepState}')
    

    def toggle_averaging(self, avg_mode):
        self.write_check(f'AVER {avg_mode}')
        setAvgState = self.query_check('AVER:COUN?')
        if self.debug is True:
            self.print_console(f'    Set spectrum analyzer averaging = {setAvgState}')

    def arm_trigger(self):
        status = self.write_check("INIT:CONT OFF")
        return status
    
    def trigger_sweep(self):
        
        # self.write_check('INIT:IMM')
        
        # cancel previous sweep, start new one, and then 
        # use *OPC? 
        self.query_check('INIT:IMM; *OPC?')
        
        ret = self.read_check(int)
        
        if ret == 0:
            self.print_console('    *OPC returned "0" - failed to sweep?')
        
        if self.debug is True:
            self.print_console('    Single sweep triggered')
        
        return ret
    
    def send_marker_to_max(self):
        self.write_check('CALC:MARK:MAX')
        if self.debug is True:
            self.print_console('    Marker set to max peak')


    def read_marker_freq_amp(self):
        markerFreqHz = self.query_check('CALC:MARK:X?')
        markerAmpldBm = self.query_check('CALC:MARK:Y?')
        
        if self.debug is True:
            self.print_console(f'    Marker: freq = {markerFreqHz} MHz\tamp = {markerAmpldBm} dBm')
        
        return markerFreqHz, markerAmpldBm
         
         
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~  Instr Scripts
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
    def take_single_trace(self, trace_num=1):
        
        """ leftover matlab code, not sure if useful """
        # self.write_check(':FORM ASCii'); # set the trace data to ascii format, slower but more readable than binary 
        
        # nBytes = 16 * nFreqPts; # it's roughly 13 bytes per data point, when pulling traces in ascii format, so this gives some buffer (ha ha)
        # if instr.gpibObj.InputBufferSize < nBytes
        #     if instr.verbose
        #         fprintf('SA input buffer size insufficient. Setting buffer size (requires closing and reopening gpib connection).')
            
            # fclose(instr.gpibObj);
            # instr.gpibObj.InputBufferSize = nBytes;
            # fopen(instr.gpibObj)
        
        # continuously check if SA has finished its trace
        check = False
        tstart = time.time()
        
        # while check is False:
        #     time.sleep(1)
        #     t_elapsed = time.time() - tstart
        #     print(f"      time elapsed: [{t_elapsed:1.0f}s]")
                
        #     # check_str is a string, "0" = busy or "1" = complete
        #     check_str = self.query_check('STAT:OPER:AVER1:COND?')[1]

        #     # once it is "1", print that we're finished
        #     if check_str != "0":
        #         print(f"\nTrace finished. Uploading now.")
        #         print(f"\n   Total time elapsed: {t_elapsed:1.0f} seconds")
        #         if t_elapsed >= 600:
        #             print(f"                     = {t_elapsed/60:1.1f} minutes \n")
                
        #         # update the variable and let the while finish
        #         check = bool(check_str)


        # trigger sweep then get data 
        self.trigger_sweep()
        
        traceData = self.query_check(f'TRAC:DATA? TRACE{trace_num}')
        
        
        # manually determine x axis
        freqCenterHz = self.get_freq_center_Hz()
        freqSpan = self.get_freq_span_Hz()
        freqs = np.linspace(freqCenterHz - freqSpan / 2, freqCenterHz + freqSpan / 2, len(traceData));
        
        return freqs, traceData
    
    
         
if __name__ == '__main__':
    
    # %load_ext autoreload
    # %autoreload 2 
    
    SA_RnS_InstrConfig = {
        "instrument_name" : "SA_RnS",
        "rm_backend" : None,
        "instr_address" : 'GPIB::20::INSTR',  
        
    }
    
    test_SA = SA_RnS_FSEB20(SA_RnS_InstrConfig, debug=True)
    test_SA.check_instr_error_queue()
    msg = test_SA.idn
    test_SA.print_console(msg, prefix="self.write(*IDN?) ->")
    
    test_SA.print_class_members()
    
    display(test_SA.query_check("*IDN?"))
    
    
    
    
    
    ##
    
    method_list = [func for func in dir(test_SA) if callable(getattr(test_SA, func)) and "get" in func and "__" not in func] 
    display(method_list)

    callable_method_list = [getattr(test_SA, func) for func in method_list]
    display(callable_method_list)

    for method in method_list:
        func = getattr(test_SA, method) 
        try:
            test_SA.print_console(f"testing:  {method}")
        except pyvisa.VisaIOError as e:
            print("failed: \n")
            print(e)
            
        result = func()
        display(result)
        