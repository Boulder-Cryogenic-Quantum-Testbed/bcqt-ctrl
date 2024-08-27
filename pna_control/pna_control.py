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
from datetime import datetime
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
    print("Initializing PNA..")
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
    
    # if(averages <= 1):
        # print(f"Changing averages from {averages} to 3.")
        # averages = 3

    # Convert averages to integer
    averages = averages//1
    pna.write('SENSe1:AVERage:Count {}'.format(averages))

def read_data(pna, points, sample_id, power, temp, centerf, 
                segments : list = None,
                overwrite : bool = True, 
                output_filename = None, 
                output_filepath = None, 
                filename_suffix = None,
                verbose : bool = False):
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
            ssplit = s.replace(" ", "").split(',')
            
            # int() doesnt want a string of a float like '12.0', so if it has  
            # a decimal point, turn it into a float first
            nf = int(ssplit[2]) if '.' not in ssplit[2] else int(float(ssplit[2]))  
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
    if output_filepath is None:
        output_filepath = timestamp_folder(os.getcwd() + '\\', centerf, sample_id)
        
    if output_filename == None or output_filename == '':
        if verbose:  print(f"{output_filename=}, using name_datafile")
        output_filename = name_datafile(power, temp, cfreq, sample_id, filename_suffix)
        
    print(f'      File Directory: {output_filepath}')
    print(f"      Filename: {output_filename}")
    if not os.path.exists(output_filepath):
        print(f'\n      ~~!!  Directory does not exist. Making new directory.')
        os.makedirs(output_filepath)
        
    # use python to save data... line by line...
    file = open(f"{output_filepath}\\{output_filename}", 'w')
    
    count = 0
    for i in freq:
        file.write(str(i)+','+str(mag[count])+','+str(phase[count])+'\n')
        count = count + 1
    file.close()

def get_data(centerf: float, 
             span: float, 
             temp: float, 
             output_filename: str = None,
             averages: int = 100, 
             power: float = -30, 
             edelay: float = 76, 
             ifband_khz: float = 5, 
             points: int = 201, 
             sample_id: str = 'sample',
            #  instr_addr : str = 'GPIB::16::INSTR', # If using GPIB  7/18
             instr_addr : str = 'TCPIP0::169.254.89.124::hislip0::INSTR',
             sparam : str = 'S21',
             cal_set : str = None,
             setup_only : bool = False,
             segments : list = None,
             filename_suffix : str = None,
             data_dir : str = '.\\',
             verbose : bool = False):
    '''
    function to get data and put it into a user specified file
    '''
    if verbose: print(f"\nStarting new measurement. \nConnecting to {instr_addr}")
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
    #     keysight = rm.open_resource(instr_addr)
        
    if verbose: print("  Setting up PNA.")
    pna_setup(keysight, points, centerf, span, ifband_khz, power, edelay, averages,
              sparam=sparam, cal_set=cal_set, segments=segments)

    if setup_only:
        return
    
    print(f"\nMeasuring {centerf} GHz for parameters:\n   power = {power} dBm, averages = {averages}, IFBW = {ifband_khz} kHz")
    
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
        t_elapsed = time.time() - tstart
        print(f"      time elapsed: [{t_elapsed:1.0f}s]")
            
        if (keysight.query('STAT:OPER:AVER1:COND?')[1] != "0"):
            print(f"\nTrace finished. Uploading now.")
            print(f"\n   Total time elapsed: {t_elapsed:1.0f} seconds")
            if t_elapsed >= 600:
                print(f"                     = {t_elapsed/60:1.1f} minutes \n")
            cnt += 1
            break
        
    keysight.write('DISPlay:WINDow1:Y:AUTO')
    keysight.write('DISPlay:WINDow2:Y:AUTO')
    
    if verbose: print("  sending OPC?")
    keysight.query('*OPC?')
    if verbose: print("  sending *WAI?")
    keysight.write('*WAI')
    time.sleep(3.0)
    if verbose: print("  sending HOLD")
    keysight.write('SYSTem:CHANnels:HOLD')

    if verbose: print("  Reading PNA Data.")
    read_data(keysight, points, 
              sample_id, power, 
              temp, centerf, 
              output_filename=output_filename,
              segments=segments, output_filepath=data_dir, 
              filename_suffix=filename_suffix, verbose=verbose)

    if verbose:  print("  Finished reading, shutting off output.")
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
                output_filename : str = None, 
                adaptive_averaging : bool = True, 
                cal_set : str = None, 
                setup_only : bool = False, 
                segments : list = None, 
                filename_suffix : str = None, 
                instr_addr : str = 'TCPIP0::K-N5222B-21927::hislip0,4880::INSTR', 
                data_dir : str = '.\\',
                verbose : bool = False): 

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
    for idx, power in enumerate(sweeps):
        
        print(f'\n~~~~~~> [{power} dBm -> power ({idx+1}/{len(sweeps)})] <~~~~~~ ')
        get_data(centerf, 
                 span, 
                 temp, 
                 output_filename, 
                 averages, 
                 power,
                 edelay, 
                 ifband_khz, 
                 points,
                 sample_id, 
                 instr_addr,
                 sparam=sparam, 
                 cal_set=cal_set, 
                 setup_only=setup_only, 
                 segments=segments,
                 filename_suffix=filename_suffix,
                 data_dir = data_dir,
                 verbose = verbose)
        
        if adaptive_averaging: 
            averages = averages * ((10**(stepsize/10))**0.5)
    print('\nPower sweep completed.')


def name_datafile(power: float = None,
                  temp: float = None,
                  freq: float = None,
                  sample_id: str = None,
                  filename_suffix: str = None) -> str:
    
    if sample_id is None:  sample_id = "MissingSampleID"
    if power is None:      power = 999
    if temp is None:       temp = -99
    if freq is None:       freq = 0
    
    # suffix is .csv by default 
    if filename_suffix == "" or filename_suffix == None:  
        filename_suffix = ".csv"
        
    elif filename_suffix.startswith("_") is False:
        # add an underscore to the front if suffix is not empty
        filename_suffix = "_" + filename_suffix
    
    # make sure the filename ends with csv
    if filename_suffix.endswith("csv") is False:
        filename_suffix += ".csv"
    
    # debug
    # print(f"{type(output_filepath)}, {type(sample_id)}, {type(freq)}, {type(power)}, {type(temp)}, {type(filename_suffix)}")
    
    # Use f-strings to make the formatting more compact
    filename = f'{sample_id}_{freq:.3f}GHz_{power:.0f}dBm_{temp:.0f}mK'
    filename = filename.replace('.','p') # add suffix after using replace to avoid ".csv" -> "pcsv"
    filename += filename_suffix  
    
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
    # now = time.strftime("%y%m%d", time.localtime())
    now = datetime.today().strftime(r'%b_%d_%H%M')
    
    output = f'{now}\\{sample_id}_{centerf:.3f}GHz'
    output = output.replace('.','p')
    
    if data_dir is None:
        output_path = data_dir  # bypass
    else:
        output_path = output + '\\'
    
    return output_path

