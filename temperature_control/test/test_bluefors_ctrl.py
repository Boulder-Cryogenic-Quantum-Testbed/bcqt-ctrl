# -*- coding : utf-8 -*-
"""
BlueFors Lakeshore control test file

Author: Nick Materise
Date:   230913

"""

import sys
# Need to test relative path, otherwise try the above
sys.path.append(r'./')
from bluefors_ctrl import BlueForsCtrl
from bluefors_ctrl import measure_multiple_resonators_tsweep
import numpy as np
import time

# Color printing
RED   = '\033[31m'
GREEN = '\033[32m'
CRST  = '\033[0m'

def check_test(ret, name):
    """
    Check if a test returned correct or not
    """

    # Print success
    if ret:
        print('>>> ' + name + GREEN + ' PASSED' + CRST)

    # Print failure
    else:
        print('>>> ' + name + RED + ' FAILED' + CRST)

def test_instantiate_class():
   bf = BlueForsCtrl()
   bf.print_class_members()
   return True

def test_get_temperature():
   bf = BlueForsCtrl()
   Tmxc = bf.get_temperature('MXC')
   Tstill = bf.get_temperature('still')
   print(f'Tmxc: {Tmxc:.1f} mK')
   print(f'Tstill: {Tstill:.1f} mK')
   return True

def test_set_temperature():
   bf = BlueForsCtrl(thermalization_minutes=0.)
   # Get the temperature
   Tmxc = bf.get_temperature('MXC')
   Tstill = bf.get_temperature('still')
   print(f'Tmxc: {Tmxc:.1f} mK')
   print(f'Tstill: {Tstill:.1f} mK')
   # Set the temperature to 10 mK
   bf.set_temperature(25.)
   Tmxc = bf.get_temperature('MXC')
   print(f'Tmxc: {Tmxc:.1f} mK')
   return True

def test_temperature_sweep():
    # List of temperatures to try
    Tlist = [30, 50] # , 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]

    # Set the center frequencies, spans, delays, powers
    fcs = [5.2100783, 6.5512265, 7.87484732811]
    spans = [0.5, 1., 0.2]
    delays = [61.32, 59.97, 60.05] 
    
    fcs = [7.87484732811]
    spans = [0.2]
    delays = [60.05] 

    powers = [-35]

    # Change the sample name
    sample_name = 'SPCAl4N_NIST_NOETCH'
    measure_multiple_resonators_tsweep(fcs, spans, delays, powers,
            ifbw=1., sparam='S21', npts=51,
            adaptive_averaging=False, sample_name=sample_name,
            runtime=0.1, cal_set = None, start_delay=0.,
            is_segmented=True, offresfraction=0.8, use_homophasal=None,
            Navg_init=None, Tlist=Tlist, thermalization_minutes=1.)

    return True
    

def run_tests(tests):
    """
    Runs all tests and reports successes and failures
    """
    ret_cnt = 0
    for t in tests:
        print('\n------------------------------------------')
        print(f'Testing {t} ...')
        print('------------------------------------------\n')
        ret = eval(f'{t}()')
        check_test(ret, t)
        if ret: ret_cnt += 1
        print('------------------------------------------')

    print('\n--------------------------------------------')
    print(f'|         {ret_cnt} of {len(tests)} tests passed.             |')
    if ret_cnt != len(tests):
        print(f'{len(tests)-ret_cnt} of {len(tests)} tests failed.')
    print('--------------------------------------------\n')
    

if __name__ == '__main__':
    tests = [# 'test_instantiate_class',
             # 'test_get_temperature'
             # 'test_set_temperature'
             'test_temperature_sweep'
             ]
    run_tests(tests)
