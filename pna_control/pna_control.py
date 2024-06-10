# -*- encoding: utf-8 -*-
"""
Collection of functions defining control of the Keysight PNA instrument

TODO:
    * Collect all functions into a single class
    * Sends commands to the instrument with a socket or other connection
    * Commands of the SCPI variety using the pyvisa interface
    * Write a wrapper that uses the powersweep() function for legacy users

"""
import numpy as np
import pyvisa
import os
from os import path

import time
import pandas as pd

def pna_setup(pna, points: int, centerf: float, span: float, ifband_khz: float, power: float,
              edelay: float, averages: int, sparam : str = 'S21', cal_set : str = None,
              segments : list = None):
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
    # Send a preset command to the VNA and turn off the RF power
    print("  Reinitializing PNA")
    # XXX: Changed to user preset
    # pna.write('SYSTem:FPRESet')
    pna.write('SYSTem:UPRESet')
    time.sleep(0.01)
    pna.write('OUTPut:STATe OFF')

    # Initial setup for measurement
    ## Query the existing measurements
    measurements = pna.query('CALC1:PAR:CAT:EXTended?')

    ## If any measurements exist, delete them all
    if measurements != 'NO CATALOG':
        pna.write(f'CALC1:PARameter:DELete:ALL')
        
    # create measurements
    pna.write(f'CALC1:MEASure1:DEFine \"{sparam}\"')
    pna.write(f'CALC1:MEASure2:DEFine \"{sparam}\"')

    #set parameters for sweep

    if segments:
        num_segments = len(segments)
        seg_data = ''.join([s for s in segments])
        pna.write("SENSe1:SWEep:TYPE SEGment")
        pna.write(f'SENSe1:SEGMent:LIST SSTOP, {num_segments}{seg_data}')
    else:
        pna.write("SENSe1:SWEep:TYPE LINear")
        pna.write(f'SENSe1:SWEep:POINts {points}')
        pna.write(f'SENSe1:FREQuency:CENTer {centerf}GHZ')
        pna.write(f'SENSe1:FREQuency:SPAN {span}MHZ')

        pna.write(f'SENSe1:SWEep:TIME:AUTO ON')
        
    # print(pna.query("SYST:CHAN:CAT?"))  # ask what channels are activated

    if cal_set:
        pna.write(f'CALC1:CORRection:TYPE \'Full 2 Port(1,2)\'')
        pna.write('SENSe1:CORRection:INTerpolate:state ON')
        # XXX: This does not work!
        # cal_cmd = f'SENS1:CORR:CSET:ACT \'{cal_set}\',1'
        cal_cmd = f'SENS:CORR:CSET:ACT \'{cal_set}\',0'
        # print(f'cal_cmd: {cal_cmd}')
        pna.write(cal_cmd)

        # gpoints = pna.query(f'SENSe1:SWEep:POINts?')
        # assert int(gpoints) == points, f'VNA points ({gpoints}) != {points}.'

    pna.write(f'SOUR1:POW1 {power}')
    pna.write('SENSe1:AVERage:STATe ON')
    pna.write(f'SENSe1:BANDwidth {ifband_khz}KHZ')

    # configure ch 1 measurement 1
    pna.write('CALC1:PAR:MNUM 1')  # select ch 1, meas 1
    pna.write('DISPlay:WINDow1 ON')  # create window
    pna.write('DISPlay:MEAS1:FEED 1')  # display meas 1 on window 1
    pna.write(f'CALC1:CORRection:EDELay:TIME {edelay}NS')
    pna.write('CALC1:MEASure1:FORMat MLOGarithmic')
    
    
    # configure ch 1 measurement 2
    pna.write('CALC1:PAR:MNUM 2')  # select ch 1, meas 2
    pna.write('DISPlay:WINDow2 ON')  # create window 2
    pna.write('DISPlay:MEAS2:FEED 2')  # display meas 2 on window 2
    pna.write(f'CALC1:CORRection:EDELay:TIME {edelay}NS')
    pna.write('CALC1:MEASure2:FORMat PHASe')
    
    # autoscale for visibility on the display
    # pna.write('DISPlay:WINDow1:TRACe1:Y:SCAle:AUTO')
    # pna.write('DISPlay:WINDow2:TRACe1:Y:SCAle:AUTO')
    
    #ensure at least 10 averages are taken
    #if(averages < 10):
    #    averages = 10
    
    if(averages <= 1):
        print(f"Changing averages from {averages} to 3.")
        averages = 3

    # Convert averages to integer
    averages = averages//1
    pna.write('SENSe1:AVERage:Count {}'.format(averages))

def read_data(pna, points, sample_id, power, temp, centerf, segments : list = None,
        overwrite : bool = True, output_file = None,  data_dir = '.\\', seg_str = None):
    '''
    function to read in data from the pna and output it into a file
    '''
    
    #read in frequency
    cfreq = float(pna.query('SENSe1:FREQuency:CENTER?')) / 1e9
    
    # This obviates the need for points as an input
    if segments:
        # Read the list of all segments
        freq = np.array([])
        for s in segments:
            ssplit = s.split(',')
            nf = int(ssplit[2])
            f1 = float(ssplit[3])
            f2 = float(ssplit[4])
            f = np.linspace(f1, f2, nf)
            freq = np.hstack((freq, f))
    else:
        gpoints = int(pna.query(f'SENSe1:SWEep:POINts?'))
        freq = np.linspace(float(pna.query('SENSe1:FREQuency:START?')),
                float(pna.query('SENSe1:FREQuency:STOP?')), gpoints)

    # TODO: switch to using CALC:DATA SMEM instead of FDATA to avoid format issues
    # read in magn
    pna.write('CALC1:PAR:MNUM 1')  # select ch 1, meas 1
    pna.write('CALC1:FORMat MLOG')
    mag = pna.query_ascii_values('CALC1:DATA? FDATA', container=np.array)

    # read in phase
    pna.write('CALC1:PAR:MNUM 2')  # select ch 1, meas 2
    pna.write('CALC1:FORMat PHASe')
    pna.write('DISPlay:WINDow2:Y:AUTO')
    phase = pna.query_ascii_values('CALC1:DATA? FDATA', container=np.array)

    # save files
    # create a new directory for the output to be put into
    
    directory_name = timestamp_folder(os.getcwd() + '\\' + data_dir + '\\', centerf, sample_id)
    
    if not os.path.exists(directory_name):
        print(f'      directory_name: {directory_name}')
        print(f'      Does not exist. Making new directory.')
        os.mkdir(directory_name)
        
    #open output file and put data points into the file
    if output_file is not None:
        filename = output_file
    else:
        filename = name_datafile(sample_id, power, temp, cfreq, seg_str)
        
    if filename[-4:] != '.csv':
        filename += '.csv'
        
    final_savepath = directory_name + filename
    file = open(final_savepath, 'w')
    print("saving at: ", final_savepath)
    
    count = 0
    for i in freq:
        file.write(str(i)+','+str(mag[count])+','+str(phase[count])+'\n')
        count = count + 1
    file.close()

def get_data(centerf: float, 
             span: float, 
             temp: float, 
             output_file: str = None,
             averages: int = 100, 
             power: float = -30, 
             edelay: float = 76, 
             ifband_khz: float = 5, 
             points: int = 201, 
             sample_id: str = 'sample',
             # instr_addr : str = 'GPIB::16::INSTR', # If using GPIB
             # instr_addr : str = 'TCPIP0::69.254.35.52::islip0::INSTR1', # Old address from JILA lab
             instr_addr : str = 'TCPIP0::169.254.89.124::hislip0::INSTR',
             sparam : str = 'S21',
             cal_set : str = None,
             setup_only : bool = False,
             segments : list = None,
             seg_str : str = None,
             data_dir : str = '.\\'):
    '''
    function to get data and put it into a user specified file
    '''
    print(f"Connecting to {instr_addr}")
    #set up the PNA to measure s21 for the specific instrument GPIB0::16::INSTR
    rm = pyvisa.ResourceManager()
    keysight = rm.open_resource(instr_addr)

    # handle failure to open the GPIB resource #this is an issue when connecting
    # to the PNA-X from newyork rather than ontario
    # try:
        # keysight = rm.open_resource(instr_addr)

        ## Attempt to fix the timeout error in averaging command
        # keysight.timeout = None
        # keysight = rm.open_resource('GPIB0::16::INSTR')
        
    # except Exception as ex:
    #     print(f'\n----------\nException:\n{ex}\n----------\n')
    #     print(f'Trying GPIB address {GPIB_addr} ...')
    #     keysight = rm.open_resource(GPIB_addr)
        # keysight = rm.open_resource(instr_addr)
    print("  Setting up PNA.")
    pna_setup(keysight, points, centerf, span, ifband_khz, power, edelay, averages,
              sparam=sparam, cal_set=cal_set, segments=segments)

    if setup_only:
        return
    print(f"Beginning measurement for:\n   power = {power} dBm, averages = {averages}, IFBW = {ifband_khz} kHz")
    
    keysight.timeout = 30000
    
    # start taking data for S21
    keysight.write('INITiate:CONTinuous ON')
    keysight.write('OUTPut:STATe ON')
    # keysight.write('CALC1:PARameter:SELect \'M1\'')
    keysight.write('FORMat ASCII')
    keysight.write('DISPlay:WINDow1:Y:AUTO')
    keysight.write('DISPlay:WINDow2:Y:AUTO')

    #wait until the averages are done being taken then read in the data
    cnt = 0
    
    tstart = time.time()
    while(True):
        # print(keysight.query('STAT:OPER:AVER1:COND?'))
        time.sleep(1)
        print(f"      time elapsed: {time.time() - tstart:1.0f}s")
        if (keysight.query('STAT:OPER:AVER1:COND?')[1] != "0"):
            print(f"      Trace finished. Uploading now.")
            cnt += 1
            break
        
    keysight.write('DISPlay:WINDow1:Y:AUTO')
    keysight.write('DISPlay:WINDow2:Y:AUTO')
    
    print("  sending OPC?")
    keysight.query('*OPC?')
    print("  sending *WAI?")
    keysight.write('*WAI')
    time.sleep(3.0)
    print("  sending HOLD")
    keysight.write('SYSTem:CHANnels:HOLD')

    print("  Reading PNA Data.")
    read_data(keysight, points, sample_id, power, temp,
            centerf, segments=segments, seg_str=seg_str, data_dir=data_dir)

    print("  Finished reading, shutting off output.")
    keysight.write('SYSTem:CHANnels:RESume')
    keysight.write('OUTPut:STATe OFF')

def power_sweep(startpower: float, 
                endpower: float, 
                numsweeps: int, 
                centerf: float, 
                span: float, 
                temp: float, 
                averages: float = 100, 
                edelay: float = 76, 
                ifband_khz: float = 5, 
                points: int = 201, 
                sample_id: str = 'sample',
                sparam : str = 'S21',
                adaptive_averaging : bool = True,
                cal_set : str = None,
                setup_only : bool = False,
                segments : list = None, 
                seg_str : str = None,
                instr_addr : str = 'TCPIP0::K-N5222B-21927::hislip0,4880::INSTR',
                output_file : str = None,
                data_dir : str = '.\\'):

    '''
    run a power sweep for specified power range with a certain number of sweeps
    '''

    #create an array with the values of power for each sweep
    if np.isclose(startpower, endpower):
        print(f'  Running only one power {startpower} dBm ...')
        sweeps = [startpower]
        stepsize = 0
    else:
        sweeps = np.linspace(startpower, endpower, numsweeps)
        stepsize = sweeps[0]-sweeps[1]
    print(f'  Measuring {sparam} ...')

    # #write an output file with conditions
    # with open(directory_name+'/'+'conditions.csv',"w") as file:
    #     file.write('# Parameter, Value, Units\n')
    #     file.write(f'SPARAM, {sparam}, \n')
    #     file.write(f'CALSET, {cal_set}, \n')
    #     file.write(f'STARTPOWER, {startpower}, dB\n')
    #     file.write(f'ENDPOWER, {endpower}, dB\n')
    #     file.write(f'NUMSWEEPS, {numsweeps}, \n')
    #     file.write(f'CENTERF, {centerf}, GHz\n')
    #     file.write(f'SPAN, {span}, MHz\n')
    #     file.write(f'TEMP, {temp:.3f}, mK\n')
    #     file.write(f'STARTING AVERAGES, {averages}\n')
    #     file.write(f'EDELAY, {edelay}, ns\n')
    #     file.write(f'IFBAND_KHZ, {ifband_khz}, kHz\n')
    #     file.write(f'POINTS, {points}, \n')
    #     file.close()

    #run each sweep
    for power in sweeps:
        print(f'{power} dBm, {averages//1} averages ...')
        get_data(centerf, span, 
                 temp, output_file, 
                 averages, power,
                 edelay, ifband_khz, 
                 points,
                 sample_id, 
                 sparam=sparam, 
                 cal_set=cal_set, 
                 setup_only=setup_only, 
                 segments=segments,
                 seg_str=seg_str,
                 instr_addr=instr_addr,
                 data_dir = data_dir)
        
        if adaptive_averaging: 
            averages = averages * ((10**(stepsize/10))**0.5)
    print('Power sweep completed.')


def name_datafile(sample_id: str = None,
                  power: float = None,
                  temp: float = None,
                  freq: float = None,
                  seg_str: str = None) -> str:
    
    if sample_id is None:  sample_id = "MissingSampleID"
    if power is None:      power = 999
    if temp is None:       temp = -1
    if freq is None:       freq = 0
    if seg_str is None:    seg_str = ''
    
    # Check that the file does not have an extension, otherwise strip it
    # Use f-strings to make the formatting more compact
            
    filename = f'{sample_id}_{freq:.3f}GHz_{power:.0f}dB_{temp:.0f}mK_{seg_str}'
    filename = filename.replace('.','p')

    return filename
    
def timestamp_folder(data_dir: str = None, centerf = None, sample_id: str='powersweep') -> str:
    """Create a filename and directory structure to annotate the scan.

        Takes a root directory, appends scan type and timestamp.

        Args:
            dir: root directory for the scan
            meastype: type of measurements, eg: 'powersweep' 

        Returns:
            Formatted path eg. dir/5p51414GHz_HPsweep_200713_12_18_04/ 
    """
    now = time.strftime("%y%m%d", time.localtime())
    
    
    output = f'{sample_id}_{centerf:.3f}GHz'
    output = output.replace('.','p')
    
    if dir != None:
        output_path = data_dir  # bypass
    else:
        output_path = output + '/'
    return output_path

