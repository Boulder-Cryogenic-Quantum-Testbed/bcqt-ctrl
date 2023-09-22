# -*- encoding: utf-8 -*-
"""
PID Controller using the open loop control on the BlueFors
gas handling control system.

Author: Nick Materise
Date:   230913


Example:
    
    python janis_ctrl.py

"""

import simple_pid
import pyvisa
import glob
import datetime
import numpy as np
import sys
import time
import re
import threading
from typing import NamedTuple
sys.path.append(r'C:\Users\68707\Documents\bcqt\bcqt-ctrl\pna_control')
import pna_control as PNA
from lakeshore import Model372
from lakeshore import Model372HeaterOutputSettings, Model372OutputMode,\
Model372InputChannel, Model372ControlLoopZoneSettings,\
Model372MeasurementInputCurrentRange, Model372SensorExcitationMode,\
Model372MeasurementInputVoltageRange, Model372AutoRangeMode,\
Model372InputSensorUnits, Model372MeasurementInputResistance,\
Model372InputSetupSettings

class PnaData(NamedTuple):
    vna_centerf : float
    vna_span : float
    vna_edelay : float
    vna_points : int
    sparam : str
    vna_ifband : float
    vna_startpower : float
    vna_endpower : float
    vna_numsweeps : int
    setup_only : bool
    segments : list
    vna_averages : int
    adaptive_averaging : bool
    cal_set : str
    sample_name : str

class BlueForsCtrl(object):
    """
    Class that reads temperatures, pressures, flow rates, and controls
    the mixing chamber heater temperature to perform temperature sweeps
    and record S21 measurements of resonators.

    This controller also talks to the Lakeshore resistance bridge
    and uses the software from the vendor to retrieve temperatures.
    """
    def __init__(self, *args, **kwargs):
        """
        Class constructor
        """
        # Set the default VNA address
        self.vna_addr = 'TCPIP0::68707CRYOCNTRL::hislip_PXI10_CHASSIS1_SLOT1_INDEX0,4880::INSTR'
        self.lksh_addr = 'GPIB0::12::INSTR'

        # Set as True to start the PID controller, then set to False to allow
        # for updates to the PID values from the previous temperature set point
        self.is_pid_init = True
        self.pid_values = None
        self.bypass_bluefors = False
        self.baud_rate = 57600

        # Set the base temperature of the fridge here
        self.dstr = datetime.datetime.today().strftime('%y%m%d')

        # Set the relative temperature tolerance to 5 %
        self.Trel = 0.05
        self.temp_delay = 15. 
        self.thermalization_minutes = 30.

        # Default temperature log file
        self.enable_logging = True
        self.log_file_name = f'logs/temperature_log_{self.dstr}.txt'

        # Dictionary with the channels on the Lakeshore
        self.channel_dict = {'50K' : 1, '4K' : 2, 'still' : 5, 'MXC' : 6}
        self.Tscale_dict = {'K' : 1., 'mK' : 1e-3, 'uK' : 1e-6}

        # Update the arguments and the keyword arguments
        # This will overwrite the above defaults with the user-passed kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

        # Open the log file
        if self.enable_logging:
            self.fid = open(self.log_file_name, 'a')

        # Connect to the Lakeshore
        self.connect_to_lakeshore()

    def __del__(self):
        """
        Class destructor
        """
        self.close_lakeshore()
        self.close_rm()
        self.close_log_file()

    def close_log_file(self):
        """
        Closes the log file, if it exists
        """
        if hasattr(self, 'fid'):
            if self.fid is not None:
                self.fid.close()

    def close_rm(self):
        """
        Close the pyvisa Resource Manager
        """
        if hasattr(self, 'rm'):
            self.rm.close()

    def close_lakeshore(self):
        """
        Close the Lakeshore class object
        """
        if hasattr(self, 'lksh'):
            del self.lksh

    def reset_connections(self):
        """
        Resets the Resource Manager and Lakeshore connection to prevent hanging
        during long temperature sweeps
        """
        self.rm.close()
        del self.lksh
        del self.m372
        self.connect_to_lakeshore()

    def print_class_members(self):
        """
        Prints all members of the class
        """
        for k, v in self.__dict__.items():
            print(f'{k} : {v}')

    def connect_to_lakeshore(self):
        """
        Create a pyvisa connection to the Lakeshore 372, and call the Model372
        driver to control the Lakeshore
        """
        # pyvisa setup
        self.rm = pyvisa.ResourceManager()
        self.lksh = self.rm.open_resource(self.lksh_addr)

        # Lakeshore connection information
        self.m372 = Model372(self.baud_rate, connection=self.lksh)

    def get_pid_ctrl(self, Tset, sample_time=15):
        """
        Returns a simple_pid PID controller object
        """
        # Trying Ku = 70, Tu = 186 s (this is set by LS372 scanning cycle time)
        # 
        # go to https://en.wikipedia.org/wiki/Ziegler%E2%80%93Nichols_method
        # for info on Ku and Tu and how they relate to PID gain settings
        Ku = 70.
        # Tu = 186 #s -- this is close to Nyquist, 2 * 90 s update time
        Tu = 186. #s -- this is close to Nyquist, 2 * 90 s update time
        # Tu = 90#s
        Kp = 0.6 * Ku
        Ki = 1.2 * Ku / Tu
        Kd = 0.075 * Ku * Tu

        # Temperature setpoint must be in Kelvin!
        # sample_time is in seconds
        stime = self.sample_time if hasattr(self, 'sample_time') \
                else sample_time
        
        pid = simple_pid.PID(Kp, Ki, Kd, setpoint=Tset, sample_time=stime)

        # Set the maximum current
        T_base = 0.0085 #K (approx. base temperature)

        # Limits the current to 1/3 of the total range, Lakeshore 372?
        max_current = 1.33*8.373*(Tset-T_base)**(0.833) #mA
        pid.output_limits = (0, max_current)

        return pid

    def log_temperatures_to_file(self, channels : list,
                                 print_temp : bool = False):
        """
        Record the temperature to the log file, append if it already exists
        """
        # Records the temperature (s) to file
        Tlist = [self.get_temperature(ch, T_scale='K') for ch in channels]
        if self.fid is None:
            self.fid = open(self.log_file_name, 'a')
        packet = ','.join([f'{T:.4f}' for T in Tlist])
        if print_temp:
            print(packet)
        self.fid.write(packet)
        self.fid.write('\n')

    def get_heater_range(self, T_K : float) -> int:
        """
        Heater ranges as recommended by Jake
        """
        if T_K < 70e-3:
            return 5 # 3.16 mA
        elif T_K < 101e-3:
            return 6 # 10 mA
        elif T_K < 300e-3:
            return 7 # 31.6 mA
        else:
            return 8 # 100 mA

    def set_temperature(self, T_mK : float,
                        ramp_rate_seconds : float = 10.,
                        output_percent : float = 50., heater_id : int = 6,
                        heater_rate_kelvin : float = 0.01, 
                        zone : int = 1,
                        T_scale : str = 'mK',
                        relay_number : int = 1,
                        is_temperature_logged : bool = True,
                        use_auto_pid : bool = True):
        """

        Sets the temperature using the Lakeshore PID controller

        Code modified from the documentation here:

        https://lake-shore-python-driver.readthedocs.io/en/latest/model_372.html  

        Arguments:
        ---------

        T_mK                :float: temperature set point in mK
        ramp_rate_seconds   :float: PID ramp time in seconds
        output_percent      :float: manual heater output percent
        heater_id           :int:   output number of the heater (1) -- warmup
        heater_rate_kelvin  :float: rate of heating in Kelvin / minute

        """
        # PID control settings
        Tscale = self.Tscale_dict[T_scale]
        T = Tscale * T_mK
        print(f'Setting temperature to {T_mK} mK ...')
        Tmax = 1.05 * T

        # Use software PID to update hardware PID values
        # if use_auto_pid:
        #     pid = self.get_pid_ctrl(T)
        #     Iout = pid(T)
        #     P, I, D =  pid.components
        #     print(f'PID: ({P}, {I}, {D})')
        # else:
        #     P = 50.
        #     I = 5000.
        #     D = 2000.
        P = 50.
        I = 5000.
        D = 2000.

        # Set the heater range
        rng = self.get_heater_range(T)
        self.lksh.write(f'HTRRNG {rng}')

        # Set the output mode, PID, and ramp
        self.lksh.write(f'OUTMODE 0,5,{heater_id},1,0,1')
        self.lksh.write(f'PID 0,{P},{I},{D}')
        self.lksh.write(f'SETP {T*0.9:.6f}')
        self.lksh.write(f'RAMP 0,1,{heater_rate_kelvin:.6f}')
        self.lksh.write(f'SETP {T:.6f}')

        # Loop on the PID
        Tstr = self.lksh.query(f'RDGK? {heater_id}')
        TT = float(re.findall('\+(.*)\\r', Tstr)[0])
        print(f'np.abs(TT - T) / T: {np.abs(TT - T) / T}')
        counter = 0
        while 1: # (np.abs(TT - T) / T) > self.Trel:
            # Wait to print the temperature
            time.sleep(self.temp_delay)
            Tstr = self.lksh.query(f'RDGK? {heater_id}')
            TT = float(re.findall('\+(.*)\\r', Tstr)[0])
            print(f'{TT / Tscale:.3f} mK')
            if (np.abs(TT - T) / T) < self.Trel:
                break

            # Update the PID values based on the current temperature
            if use_auto_pid:
                pid = self.get_pid_ctrl(T)
                Iout = pid(TT)
                P, I, D = pid.components
                print(f'pid: {P}, {I}, {D}')
                self.lksh.write(f'PID 0,{P},{I},{D}')

            if is_temperature_logged:
                self.log_temperatures_to_file(['MXC', 'still'])

        # Waiting to thermalize
        print(f'Waiting {self.thermalization_minutes} minutes to thermalize ...')
        tic = time.perf_counter()
        dt = 0
        while dt < (self.thermalization_minutes * 60):
            # Update the time
            toc = time.perf_counter()

            # Wait to print
            time.sleep(self.temp_delay)
            Tstr = self.lksh.query(f'RDGK? {heater_id}')
            TT = float(re.findall('\+(.*)\\r', Tstr)[0])

            if use_auto_pid:
                pid = self.get_pid_ctrl(T)
                Iout = pid(TT)
                P, I, D = pid.components
                print(f'pid: {P}, {I}, {D}')
                self.lksh.write(f'PID 0,{P},{I},{D}')
            print(f'{TT / Tscale:.3f} mK')
            
            # Write to log file
            if is_temperature_logged:
                self.log_temperatures_to_file(['MXC', 'still'])

            # Update the time
            dt = toc - tic

    def get_temperature(self, channel : str = 'MXC', T_scale : str = 'mK'):
        """
        Returns the temperature of a given channel
        """
        # Convert strings to floats or ints
        Tscale = self.Tscale_dict[T_scale]
        ch_id = self.channel_dict[channel]

        # Enumerate the sensor settings
        sensor_settings_mxc = \
        Model372InputSetupSettings(Model372SensorExcitationMode.VOLTAGE,
                                   Model372MeasurementInputVoltageRange.RANGE_20_MICRO_VOLTS,
                                   Model372AutoRangeMode.CURRENT, False,
                                   Model372InputSensorUnits.KELVIN,
                                   Model372MeasurementInputResistance.RANGE_63_POINT_2_KIL_OHMS)
        sensor_settings_still = \
        Model372InputSetupSettings(Model372SensorExcitationMode.VOLTAGE,
                                   Model372MeasurementInputVoltageRange.RANGE_200_MICRO_VOLTS,
                                   Model372AutoRangeMode.CURRENT, False,
                                   Model372InputSensorUnits.KELVIN,
                                   Model372MeasurementInputResistance.RANGE_2_KIL_OHMS)

        # Collect all of the sensor settings in a dictionary
        settings_dict = {'MXC' : sensor_settings_mxc,
                         'still' : sensor_settings_still}

        # Configure the desired input
        self.m372.configure_input(ch_id, settings_dict[channel])

        # Get the sensor values
        sensor_readings = self.m372.get_all_input_readings(ch_id)

        # Return the temperature in T_scale

        return sensor_readings['kelvin'] / Tscale

    def sweep_temperature(self, T_list : list, pna_data : PnaData = None, 
                          delay_minutes : float = 30.,
                          interupt_tsweep : bool = False):
        """
        Perform a temperature sweep with a list of temperatures and take
        measurements at each temperature

        Arguments:
        ---------

        T_list          :list:  list of temperature set points in mK
        measurement     :str:   string option to perform a measurement 'Sij' or
                                None 
        delay_minutes   :float: thermalization time in minutes

        """
        sec2min = 60.
        print(f'pna_data:\n{pna_data}')
        for T in T_list:
            if interupt_tsweep:
                Tj = self.get_temperature(channel='MXC')
                # Perform the measurements
                if pna_data is not None:
                    PNA.power_sweep(pna_data.vna_startpower, pna_data.vna_endpower,
                        pna_data.vna_numsweeps, pna_data.vna_centerf, pna_data.vna_span, Tj,
                        pna_data.vna_averages, pna_data.vna_edelay, pna_data.vna_ifband,
                        pna_data.vna_points, pna_data.sample_name, sparam=pna_data.sparam, 
                        adaptive_averaging=pna_data.adaptive_averaging,
                        cal_set=pna_data.cal_set,
                        setup_only=pna_data.setup_only,
                        segments=pna_data.segments,
                        instr_addr=self.vna_addr)
            else:
                Tj = self.get_temperature(channel='MXC')
                print(f'T (current): {Tj:.1f} mK, T (target): {T:.1f} mK')
                self.set_temperature(T)
                Tj = self.get_temperature(channel='MXC', T_scale = 'mK')
                print(f'T (current): {Tj:.1f} mK, T (target): {T:.1f} mK')
                self.reset_connections()

                # Perform the measurements
                if pna_data is not None:
                    PNA.power_sweep(pna_data.vna_startpower, pna_data.vna_endpower,
                        pna_data.vna_numsweeps, pna_data.vna_centerf, pna_data.vna_span, Tj,
                        pna_data.vna_averages, pna_data.vna_edelay, pna_data.vna_ifband,
                        pna_data.vna_points, pna_data.sample_name, sparam=pna_data.sparam, 
                        adaptive_averaging=pna_data.adaptive_averaging,
                        cal_set=pna_data.cal_set,
                        setup_only=pna_data.setup_only,
                        segments=pna_data.segments,
                        instr_addr=self.vna_addr)

        # Set the heater to 0
        self.lksh.write(f'SETP 0')


def compute_segments(fc, span, p, pc, beta, fscale, Noffres, offresfraction,
        option='hybrid'):
    """
    Computes segments needed to perform homophasal measurements
    with inputs from the resonator frequencies, bandwidths, powers
    """
    # Estimate the number of linewidths per sweep
    power_fac = 1. # 0.5 * (np.tanh((4 / beta) * (p - pc) / pc) + 1)
    print(f'power_fac: {power_fac}')
    Q = 20 * (fc / span) * power_fac

    # Compute the frequencies
    fstart = fc - span / 2
    fstop  = fc + span / 2
    fa = fstart + offresfraction * span / 2
    fb = fstop  - offresfraction * span / 2
    if option == 'homophasal':
        theta0 = np.pi / 32
        Nf = 30
        theta = np.linspace(-np.pi + theta0, (np.pi - theta0), Nf + 2)
        freq = fc * (1 - 0.5 * np.tan(theta / 2) / Q)
        segments = [f',1,2,{ff1*fscale},{ff2*fscale}'
                for ff1, ff2 in zip(freq[0::2], freq[1::2])]
    elif option == 'hybrid':
        theta0 = np.pi / 32
        Nf = 20
        theta = np.linspace(-np.pi + theta0, (np.pi - theta0), Nf + 2)
        freq = fc * (1 - 0.5 * np.tan(theta / 2) / Q)

        np.set_printoptions(precision=4)

        # Homophasal, near resonance
        hsegments = [f',1,2,{ff1*fscale},{ff2*fscale}'
                for ff1, ff2 in zip(freq[0::2], freq[1::2])][1:-1]
        fap = np.min(freq[1:-1]) * fscale
        fbp = np.max(freq[1:-1]) * fscale

        segments = [f',1,{Noffres},{fstop*fscale}, {fbp}',
                    *hsegments,
                    f',1,{Noffres},{fap},{fstart*fscale}']
    else:
        segments = [f',1,5,{fstart*fscale},{fa*fscale}',
                    f',1,41,{fa*fscale},{fb*fscale}',
                    f',1,5,{fb*fscale},{fstop*fscale}']

    return segments

def measure_multiple_resonators_tsweep(fcs, spans, delays, powers, ifbw=1.,
                                       sparam='S21', npts=1001,
                                       adaptive_averaging=True, sample_name='',
                                       runtime=1., cal_set=None,
                                       start_delay=0., offresfraction=0.45,
                                       is_segmented=True, use_homophasal=None,
                                       Navg_init=None, Noffres=5, pc=-75.,
                                       beta=0.2, Tlist=None,
                                       thermalization_minutes=30.):
    """
    Measures multiple resonators sequentially
    """
    # Example inputs to run a temperature sweep
    # Iterate over a list of temperatures
    # 30 mK -- 300 mK, 10 mK steps
    if len(powers) < 2:
        p1 = powers[0]
        p2 = powers[0]
    else:
        p1 = powers[0]
        p2 = powers[-1]
    power_steps = len(powers)

    # Delay the start of a sweep by Nstart hours
    h2s = 3600.
    if start_delay > 0:
        print(f'Delaying start by {start_delay} hours ...')
        time.sleep(start_delay * h2s)

    for fc, span, delay in zip(fcs, spans, delays):

        # Create the BlueForsCtrl 
        print(f'Measuring {sample_name} at {fc} GHz ...')
        bfc = BlueForsCtrl(thermalization_minutes=thermalization_minutes)

        """
        Initial number of averages
        """
        # Only used if adaptive_averaging == False
        vna_averages = Navg_init if Navg_init else 1

        time_per_sweep = npts / (1e3 * ifbw)
        print(f'powers: {powers}')

        """
        Expected runtime for power sweep
        if using the estimated runtime option
        """
        total_time_hr = runtime
        if adaptive_averaging:
            if Navg_init:
                Navg_adaptive = Navg_init
                runtime_est = estimate_time_adaptive_averages(
                                time_per_sweep,
                                powers,
                                Navg_adaptive)
                print('\n---------------------------------------')
                print(f'\nEstimated run-time: {runtime_est} hr\n')
                print('---------------------------------------\n')

            else:
                Navg_adaptive = estimate_init_adaptive_averages(
                        time_per_sweep, 
                        powers,
                        total_time_hr)

            vna_averages = Navg_adaptive

        # Set the segment data
        if is_segmented:
            fscale = 1e9 if fc < 1e9 else 1.
            span *= 1e-3
            fstart = fc - span / 2
            fstop  = fc + span / 2
            fa = fstart + offresfraction * span / 2
            fb = fstop  - offresfraction * span / 2

            # segments = compute_homophasal_segments()
            p = p1
            segments = compute_segments(fc, span, p, pc, beta, 
                    fscale, Noffres, offresfraction, option=use_homophasal)
        else:
            segments = None

        pna_data = PnaData(fc, span, delay, npts, sparam, ifbw, p1, p2,
                           power_steps, False, segments, vna_averages,
                           adaptive_averaging, cal_set, sample_name)

        print(f'pna_data:\n{pna_data}')
        bfc.sweep_temperature(Tlist, pna_data=pna_data, 
                          delay_minutes = 0.)

        del bfc

