# -*- coding: utf-8 -*-
"""
Instrument control file for the ENA E8365A from Agilent, adapted from:

https://github.com/Wheeler1711/submm_python_routines/blob/main/submm/instruments/agilent_E8364A.py

"""
import pyvisa
import numpy as np
import time

class E8365A(object):

    def __init__(self, *args, address = None, Smn = 'S21', **kwargs):
        """
        Constructor to setup a VNA measurement with the E8365A
        """
        # Set the address
        if not address:
            self.GPIBaddress = 'GPIB0::20::INSTR'
        else:
            self.GPIBaddress = address

        # Start the resource manager to send commands to the VNA
        rm = pyvisa.ResourceManager("@py")
        self.obj = rm.open_resource(self.GPIBaddress)

        # Transfer from the VNA to control computer
        self.transfermode = 2 #%1 = asc, 2 = binary transfer
        self.obj.timeout = 30000 #30 seconds
        self.Smn = Smn

        # Update the args and kwargs to be class members
        self.__dict__.update(locals())
        for k, v in kwargs.itmes():
            setattr(self, k, v)
    
    def SetStat(self,bon):
    #set the state to 1=on or 0=off
        if bon:
            cmd_str = ':OUTP ON;'
            self.obj.write(cmd_str)
        else:
            cmd_str = ':OUTP OFF;'
            self.obj.write(cmd_str)
    
    def SetAttAuto(self, bon):
        if bon:
            cmd_str = 'SOUR:POW1:ATT:AUTO ON;'
            self.obj.write(cmd_str)
        else:
            cmd_str = 'SOUR:POW1:ATT:AUTO OFF;'
            self.obj.write(cmd_str)
    
    def SetAttenuation(self, att):
        cmd_str =  'SOUR:POW2:ATT %d;' % np.abs(np.round(att))
        self.obj.write(cmd_str)        

    def read_write_data(self, fc : float, fspan : float, power : float,
            filename : str = None, averfact : int = 1, points : int = 1601,
            ifbw = 1):
        """
        Reads data from VNA and writes the result to file
        """
        #get the measurement string and then select
        cmd_str = 'CALC:PAR:CAT?'
        answer = self.obj.query(cmd_str)   

        #MeaString = ['"' str(2:findstr(answer,',')-1) '"'];
        MeaString = answer.split(",")[0][1:]#str(2:findstr(answer,',')-1);
        #print(MeaString)
        cmd_str = 'CALC:PAR:SEL ' + MeaString +';'
        self.obj.write(cmd_str)
        self.obj.write('CALC:PAR:MOD ' + self.Smn +';' )
          
        N = points;
        cmd_str = 'SENS:SWE:POIN %d;' % points
        self.obj.write(cmd_str);
       
        #set sweeptime automatically
        self.obj.write('SENS:SWE:TIME:AUTO 1')
        
        #set bandwidth, kHz
        cmd_str = 'SENS:BWID %.2f HZ;' % ifbw * 1e3
        self.obj.write(cmd_str);
        
        #set twait
        sweeptime = float(self.obj.query('SENS:SWE:TIME?'))
        twait = sweeptime * averfact * 1.02;
        
        #set power and frequency
        self.obj.write('SENS:SWE:TYPE LIN;')
        cmd_str = 'SOUR:POW1 %.2f;' % power
        self.obj.write(cmd_str)
        cmd_str = 'SENS:FREQ:CENT %.9f GHz;' % fc
        self.obj.write(cmd_str)
        cmd_str = 'SENS:FREQ:SPAN %.9f GHz' % fspan
        self.obj.write(cmd_str)
        
        if self.transfermode == 1:
            self.obj.write('FORM:DATA ASCii,0')
        else: #(obj.transfermode==2)
            self.obj.write('FORM:DATA REAL,32')
            self.obj.write('FORM:BORD SWAP')
        
        if averfact > 0:
            cmd_str = 'SENS:AVER:COUN %d;' % averfact
            self.obj.write(cmd_str)
            self.obj.write('SENS:AVER ON;')
            
            cmd_str = 'SENS:SWE:GRO:COUN %d;' % averfact
            self.obj.write(cmd_str)
            self.obj.write('INIT:CONT ON')
            
            answer = self.obj.query('SENS:SWE:MODE GRO;*OPC?')
        
        self.obj.write('DISP:WIND:TRAC:Y:AUTO') #Autoscale display
        
        if self.transfermode == 1: #asc #probably doesn't work
            k = self.obj.query('CALC:DATA? SDATA;')
            print(k)
        else: #bin
            answer = np.asarray(
                    self.obj.query_binary_values(
                        'CALC:DATA? SDATA;'),dtype =  np.double)
                  
        # Read the center frequency and span from the VNA
        fc = float(self.obj.query('SENS:FREQ:CENT?;'))#in Hz, %E8364A
        fspan = float(self.obj.query('SENS:FREQ:SPAN?;'))#in Hz, %E8364A

        # Compute the frequency vector, assuming linearly-spaced points
        f = (np.linspace(-(N-1)/2,(N-1)/2,N)*fspan/(N-1) + fc)

        # Convert the output to a complex signal
        z = answer[::2] + 1j * answer[1::2]

        # Extract the magnitude and phase, as expected by scresonators
        magdB = 20 * np.log10(np.abs(z))
        angdeg = (180 / np.pi) * np.angle(z)
        
        # Write data to file
        if filename is not None:
            with open(filename, 'w') as fid:
                for freq, mag, ang in zip(f, magdB, angdeg):
                    fid.write(f'{freq},{mag},{ang}\n')

    def power_sweep(self, powers : list, avg : float, edelay : float,
            ifbw : float, points : float, sample_id : str,
            fc : float, fspan : float,
            attenuation : float = 0., use_adpt_avg : bool = True):
        """
        Performs a power sweep, similar to the pna_control driver
        """
        # Iterate over an arbitrary list or array of powers
        averages = avg
        dp = np.diff(powers)
        for i in range(len(powers)):
            print(f'{powers[i]} dBm {avgerages} averages')

            # Set the default, power-swept filename
            fname = self.get_formatted_filename(sample_id,
                    powers[i] + attenuation, temp, fc)

            # Read data from the VNA and write it to file
            self.read_write_data(fc, fspan, powers[i], filename=fname,
                    averfact=averages, points=points, ifbw=ifbw)

            # Adaptive averaging
            if use_adpt_avg and i < len(powers) - 1:
                averages *= ((10**dp[i] / 10)**0.5)

    def get_formatted_filename(self, sample_id : str, power : float, 
            temp : float, freq : float) -> str:
        """
        Returns the standard formatted filename
        """
        # Convert frequency to GHz
        fc = freq if freq < 1e9 else freq / 1e9
        filename = f'{sample_id}_{fc:.3f}GHz_{power:.0f}dB_{temp:.0f}mK'

        return filename.replace('.', 'p')
