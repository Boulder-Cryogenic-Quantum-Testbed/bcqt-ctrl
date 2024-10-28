"""
# bcqt_helpers.py

### Purpose

contains several useful functions to keep our notebooks short

"""

import numpy as np
import pandas as pd
import os, sys, time, math, shutil
import numpy, pytz, scipy.optimize
import regex as re

import matplotlib.pyplot as plt
import scipy as sp
from matplotlib import gridspec
from glob import glob
from datetime import datetime, timedelta

# print(f"Running run_config.py from {os.getcwd()}...")
sys.path.append(r'E:\GitHub\bcqt-ctrl')
# sys.path.append(r'./instruments')
# sys.path.append(r'./experiments')
# sys.path.append(r'../scripts')
# sys.path.append(r'../notebooks')

import helper_load as hl
import helper_misc as hm
import plot_settings


def prep_report_directory(report_dir, dry_run=True, debug=False):
    
    if os.path.basename(report_dir) != "reports":
        print(f"""Error: parameter report_dir must end with 'reports' folder. Variables:
          {report_dir=}
          {os.path.basename(report_dir)=}
          {os.getcwd()=}""")
        raise ValueError
    
    debug = True if dry_run is True else debug

    all_res_folders = glob(rf"{report_dir}\*GHz")
    qiqcfc_folder = rf"{report_dir}\all_qiqcfc_csvs"
    loss_tan_folder = rf"{report_dir}\all_lost_tangents_plots"
    
    globs_and_folders = [("qiqcfc*.csv", qiqcfc_folder),
                         ("tand_vs_power*.png", loss_tan_folder)]
    
    hm.check_and_make_dir(qiqcfc_folder)
    hm.check_and_make_dir(loss_tan_folder)
    
    for res_folder in all_res_folders:
        print(f"\nChecking {res_folder}")
        for (glob_target, dst_folder) in globs_and_folders:
            files_to_copy = glob(f"{res_folder}\{glob_target}")
            if debug is True:
                print(f"  Found {len(files_to_copy)} files.")
            for filepath in files_to_copy:
                filename = os.path.basename(filepath)
                dst_filepath = rf"{dst_folder}\{filename}"
                
                if debug is True:
                    print(f"    Target: {filename},\n     src={os.path.dirname(filepath)}\n     dst={os.path.dirname(dst_filepath)}")
                    time.sleep(0.1)
                
                if dry_run is False:
                    shutil.copyfile(filepath, dst_filepath)
            
    
        
    
    

# TODO: WIP
# def check_files_for_power(pow_val, pow_list, file_list, print_output=False):
#     all_pass_flags = []
#     for power in pow_list:
#         if print_output is True:  print(power)
#         pow_bool_list = [True if str(power) in file else False for file in file_list]
#         if print_output is True:  print("  ", pow_bool_list)

#         pass_flag = any(pow_bool_list)
#         all_pass_flags.append(pass_flag)

#     return all_pass_flags

# check_files_for_power(-75, [-50, -60, -70], MPow_data_files, True)        


def clean_directory_of_qiqcfc_pngs(directory="best_datasets", dry_run=True):
    pwd = os.getcwd()
    print(f"Current directory: '{pwd}' \nTarget directory: '{directory}'")
    
    files_to_delete_pngs = glob(rf"{pwd}\{directory}\*\*.png")
    files_to_delete_pngs_2 = glob(rf"{pwd}\{directory}\*\*\*.png")
    files_to_delete_qiqcfc = glob(rf"{pwd}\{directory}\*\qiqcfc_vs_power*.csv")
    
    files_to_delete = files_to_delete_pngs + files_to_delete_pngs_2 + files_to_delete_qiqcfc
    print(len(files_to_delete))
    # print(files_to_delete[0])

    for idx, file in enumerate(files_to_delete):
        file_name = os.path.basename(file)
        file_dir = os.path.basename(os.path.dirname(file))
        print(f"[{idx}/{len(files_to_delete)}] Deleting:  '{file_name}'\n   from directory '{file_dir}'\n")
        time.sleep(0.01)
        
        if dry_run is False:
            os.remove(file)
            
    if dry_run is True:
        print(f"\n\nFinished dry run - {len(files_to_delete)} to be deleted")
    else:
        print(f"\n\nFinished - {len(files_to_delete)} deleted")
        
    # clean_directory_of_qiqcfc_pngs(dry_run=True)


def estimate_resonator_runtime(power_tuples_dict, num_res, print_output=True):
    
    """
    power_tuples_dict = { "power_label_1" : power_tuple_1,
                          "power_label_2" : power_tuple_2,
                            .
                            .
                        }
        e.g. "high_pow" : (num_high_powers, time_for_one_high_power),
    
    """
    
    if print_output is True: 
        print("Measurement Time (per resonator): ")
    
    all_res_pow_tuples = {}
    for pow_label, (num_pows, pow_time) in power_tuples_dict.items():
    
        total_pow_time = num_pows * pow_time
        all_res_pow_tuples[pow_label] = (num_pows, pow_time, total_pow_time)
        
        if print_output is True:
            print(f"      one {pow_label} power takes  ->  {pow_time} secs ({pow_time/60:1.2f} mins)" )
            print(f"          for {num_pows} powers  ->  {num_pows}*{pow_time} = {total_pow_time} ({total_pow_time/60:1.2f} mins)\n" )
        
    all_pow_times = [pow_tuple[2] for pow_tuple in all_res_pow_tuples.values()]
    single_res_time = sum(all_pow_times)
    
    total_time_secs = single_res_time * num_res
    
    for pow_label, (num_pows, pow_time, total_pow_time) in all_res_pow_tuples.items():
        print(f"         {pow_label} time = {num_pows} powers * {pow_time} seconds = {total_pow_time} seconds")
        print(f"            total_time_secs * {num_res} resonators = {total_pow_time*num_res} seconds ")
        print(f"                          = {total_pow_time*num_res/60:1.2f} minutes = {total_pow_time*num_res/3600:1.2f} hours\n")


    if print_output is True:
        print(f"\n    Total time for one resonator = {single_res_time} = {single_res_time/60:1.2f} mins = {single_res_time/3600:1.2f} hours")
        print(f"\n    Total Measurement Time:")
        print(f"                = {num_res} resonator(s) * {single_res_time:1.1f} secs")
        print(f"                                   = {total_time_secs} secs")
        print(f"                                   = {total_time_secs/60:1.2f} mins")
        print(f"                                   = {total_time_secs/3600:1.2f} hours\n")
        
    time_now = datetime.today()
    time_to_add = timedelta(seconds=total_time_secs)
    time_future = time_now + time_to_add
    time_str = datetime.strftime(time_future, "%b_%d at %I:%M:%S %p")
    print(f"Scan will finish at approximately: {time_str}  ({total_time_secs/60:1.0f} minutes from now)")
        
def print_text_block(tStart, tEnd, num_avgs, num_pts, IFBW_kHz, num_powers=1, num_resonators=None):
    tElapsed = (tEnd - tStart)/60
    tTraceTime = tElapsed/num_powers
    print("\n")
    print("======================================================")
    print(f"    Elapsed time = {tElapsed:1.2f} mins ({tElapsed*60:1.0f} seconds)")
    print(f"      secs/power = {tTraceTime:1.2f} mins/power ({tTraceTime*60:1.0f} seconds)")
    print(f"                     for:  {num_powers} power(s)")
    print(f"                           {num_avgs} averages")
    print(f"                           {IFBW_kHz:1.1f} KHz IF bandwidth")
    print(f"                           {num_pts} pre-segment points")
    if num_resonators is not None:
        print(f"                           {num_resonators} resonators")
    print("======================================================")
    print("\n")
    return


def report_shifts(dressed_freqs, bare_freqs):
    shifts = [int((f2 - f1)*1e6) for f2, f1 in zip(dressed_freqs, bare_freqs)]
    for n, (f2, f1, shift) in enumerate(zip(dressed_freqs, bare_freqs, shifts)):
        print(f"Resonator {n} at bare freq {f1:1.2f} GHz has shifted {shift} KHz to {f2:1.2f} GHz")
    return shifts

         
def find_nearest(a, a0):
    "Element in nd array `a` closest to the scalar value `a0`"
    idx = np.abs(a - a0).argmin()
    return a.flat[idx], idx


def plot_all_circles(dir_path, search_str="\\*\\*.csv", verbose=False,  show_plot=True, show_line=True,
                     save_plot=False, plot_dir=None, **kwargs):
    
    import plot_settings
    
    if verbose: print(dir_path+search_str); print(glob(dir_path+search_str))
    
    # first load all files in directory
    df_dict = hl.load_many_csvs_as_dataframes(search_dir=dir_path, search_str=search_str)

    if verbose is True:
        for label, df in df_dict.items():
            print(f"{label}: \n {df.head()}")
            
    # get all data into plots
    filenames = [os.path.basename(file).replace(".csv","") for file in df_dict.keys() if "qiqc" not in file]
    
    if verbose: print(filenames, df_dict.keys())
    
    if plot_dir is None:  # just take the directory of the files
        data_dir = os.path.dirname(dir_path)
        plot_dir = f"{data_dir}"
        sample_name = os.path.basename(os.path.dirname(plot_dir))
            
        if verbose is True:
            print(f"[??] data_dir = {data_dir}\n[???] plot_dir = {plot_dir}\n[????] sample_name = {sample_name}\n[?????] dir_path = {dir_path}")
    
    all_powers = []
    fig, ax = plt.subplots(1, 1, figsize=(12,12))
    for fname, df_key in zip(filenames, df_dict): 
        # don't you just love regex? <3 
        fname_freq = re.search(r"\dp\d{3}GHz", fname)[0]
        fname_temp = re.search(r"\d{1,2}mK", fname)[0]
        fname_power = re.search(r"-\d{1,3}dB", fname)[0]  # TODO: dBm not dB
        all_powers.append(float(fname_power.replace("dB","")))
        sample_name = re.sub(rf"_{fname_freq}_{fname_power}_{fname_temp}_homophasal", "", fname)
        
        fname = fname.replace("_homophasal","")
        
        df = df_dict[df_key]
        freq = df["Freq"]
            
        phase = df["Phase"]
        phase_rad = np.deg2rad(phase)
    
        magn_dBm = df["Magn"]
        magn_lin = 10**(magn_dBm/20)
        
        cmplx = magn_lin*np.exp(1j * phase_rad)
        
        # place info at the top
        ax.set_title(f"""{sample_name}
                         \nVNA Freq = {fname_freq}
                         \nVNA Power = {fname_power}
                         \nDR Temp = {fname_temp}""")
        
        real = np.real(cmplx)
        imag = np.imag(cmplx)
        
        # fmt_string = '-x' if show_line is True else 'x'
        fmt_string = "o"
        ax.plot(real, imag, fmt_string, label=fname_power)
        # ax.plot(real, imag, 'bo', markersize=2, markerfacecolor='none', label=fname_power)
        
    ax.set_xlabel("Real [a.u.]")
    ax.set_ylabel("Imag [a.u.]")
    
    ax.set_aspect('equal') 
    # ax.yaxis.tick_right()
    # ax.yaxis.set_label_position("right")
    
    ax.legend(bbox_to_anchor=(1.05, 1.0))
    
    ax.axhline(0, color='k', linestyle=':', linewidth=2)
    ax.axvline(0, color='k', linestyle=':', linewidth=2)
        
    # fig.suptitle(f"{sample_name}", fontsize=18)
    fig.tight_layout()
    
    filename = f"{sample_name}_{fname_freq}_all_resonance_circles.png"
    if verbose is True: print(f"Save path is: {plot_dir}\\{filename}")
    
    if save_plot is True:
        fig.savefig(f"{plot_dir}\\{filename}", format='png')
        
    if show_plot is True:
        plt.show()
    else:
        plt.close()
            
def plot_whole_directory(dir_path, search_str="\\*\\*.csv", max_rows=4, temp_threshold_mK=50, 
                         plot_min=True, verbose=False, show_plot=True, save_plot=False, plot_dir=None, plot_zero_lines=True):
    
    if verbose is True: 
        print(dir_path+search_str)
        print(glob(dir_path+search_str))
    
    # first load all files in directory
    df_dict = hl.load_many_csvs_as_dataframes(search_dir=dir_path, search_str=search_str)

    assert len(df_dict) != 0
    
    if verbose is True:
        for label, df in df_dict.items():
            print(f"{label}")
        print(f"{len(df_dict)=}")
            

    # get all data into plots 
    filenames = [os.path.basename(file).replace(".csv","") for file in df_dict.keys()]
    
    if verbose is True: print(filenames, df_dict.keys()) 
    
    if plot_dir is None:  # just take the directory of the files
        data_dir = os.path.dirname(dir_path)
        plot_dir = f"{data_dir}"
        sample_name = os.path.basename(os.path.dirname(plot_dir))
            
        if verbose is True:
            print(f"[?] data_dir = {data_dir}\n[??] plot_dir = {plot_dir}\n[???] sample_name = {sample_name}\n[????] dir_path = {dir_path}")
    
    ###### prepare the plots ######
    fig_axes_list = []
    
    # we want three columns for magn, phase, complex
    # max_rows is an input to this function
    num_datasets = len(df_dict)
    
    # first make the figs that have the max number of axes
    max_plots_per_fig = max_rows * 3
    fully_filled_figs = num_datasets//max_rows
    for row_idx in range(fully_filled_figs):
        fig, axes = make_n_plots(N=max_plots_per_fig, cols=3)
        fig_axes_list.append( (fig, axes) )
    
    # append the last plot that is all the leftover axes, so use modulo
    leftover_plots = num_datasets % max_rows
    if leftover_plots != 0:
        fig_axes_list.append( (make_n_plots(N=leftover_plots*3, cols=3)) )
    
    
    if verbose: print(f"{leftover_plots=}, {len(fig_axes_list)=}, {len(filenames)=}")
       
    # fig 0
    #  dataset 0 -> [0,1,2]
    #  dataset 1 -> [3,4,5]
    #    . . .
    #  dataset 7 -> [?,?,?]
    #
    # fig 1
    #  dataset 8 -> [0,1,2]
    #  dataset 9 -> [3,4,5]
    #    . . .
    #  dataset n -> [n-2, n-1, n]    
    
    ###### plot all data ######
    plot_idx, idx = 0, 0
    for fig_num, (fig, axes) in enumerate(fig_axes_list):
        
        plot_idx = 0
        for row_num in range(max_rows):
            
            fname_idx = row_num + fig_num*max_rows
            if fname_idx == num_datasets:
                break
            
            fname = filenames[fname_idx]
            df_key =  f"\\{fname}.csv"
            
            # remove any suffixes
            fname = fname.replace("_homophasal","").replace("_segmented","").replace("_hybrid","")
            
            # don't you just love regex? use it to get dataset parameters
            fname_power = re.search(r"-\d{1,3}dB", fname)[0]
            fname_freq = re.search(r"\dp\d{3}GHz", fname)[0]
            fname_temp = re.search(r"\d{1,2}mK", fname)[0]
            temp_int = int(''.join([s for s in fname_temp if s.isdigit()]))
            sample_name = re.sub(rf"_{fname_freq}_{fname_power}_{fname_temp}", "", fname)
            
            if verbose is True: 
                # print(f"{fname=}, {sample_name=}")
                print(f"{fig_num=}, {plot_idx=}, {row_num=}")
                print(f"len(axes) = {len(axes)}, {max_rows=}")
                
            
            # grab objects for all 3 plots
            ax1, ax2, ax3 = axes[plot_idx], axes[plot_idx+1], axes[plot_idx+2]
                    
            # take out data from dataframe
            df = df_dict[df_key]
            freq = df["Freq"]
                
            phase = df["Phase"]
            phase_rad = np.unwrap(np.deg2rad(phase))

            magn_dBm = df["Magn"]
            magn_lin = 10**(magn_dBm/20)
            
            cmplx = magn_lin*np.exp(1j * phase_rad)
            
            real = np.real(cmplx)
            imag = np.imag(cmplx)
                
            ax1.plot(freq, magn_dBm, 'ko', markersize=3, alpha=0.9, label="Magn")
            ax2.plot(freq, phase, 'ro', markersize=3, alpha=0.9, label="Phase")
            
            freq_min = freq[magn_dBm.argmin()]
            ax1.set_title(f"\nVNA Freq = {fname_freq}\nFreq Min = {freq_min/1e9:1.6f} GHz")
            
            if plot_min is True:
                ax1.axvline(freq_min, linestyle='--', linewidth=3, color='red', alpha=0.75)
                
            # ax1.legend(bbox_to_anchor=(0.55, 0.3))  # shift the legend a bit out of bounds
            # ax2.legend(bbox_to_anchor=(0.55, 0.3))
            
            if temp_int >= temp_threshold_mK:
                ax2.set_title(f"\nWARNING! \nDR Temp = {temp_int}mK !!", color='r', size=14)
            else:
                ax2.set_title(f"\nDR Temp = {temp_int}mK\n")
                
            ax1.set_ylabel("S21 [a.u.]")
            
            ax3.plot(real, imag, 'bo', markersize=2, markerfacecolor='none')
            
            ax3.set_xlabel("Real [a.u.]")
            ax3.set_ylabel("Imag [a.u.]")
            ax3.set_title(f"\nVNA Power = {fname_power}m\n")
            
            ax3.set_aspect('equal') 
            ax3.yaxis.tick_right()
            ax3.yaxis.set_label_position("right")
            
            if plot_zero_lines is True:
                ax3.axhline(0, color='k', linestyle=':', linewidth=2)
                ax3.axvline(0, color='k', linestyle=':', linewidth=2)
            
            fig.tight_layout()
            
            plot_idx += 3 
        
    # finalize each plot, show, and save
    for fig_num, (fig, axes) in enumerate(fig_axes_list):
        fig_title = f"{sample_name}_{fname_freq}_{fname_temp}_Fig{fig_num}"
        fig.suptitle(f"{fig_title}\n", fontsize=22)
        fig.tight_layout()
        
        filename = f"{fig_title}.png"
        if verbose is True: print(f"Save path is: {plot_dir}\\{filename}")
        
        if save_plot is True:
            fig.savefig(f"{plot_dir}\\{filename}", format='png')
            
        if show_plot is True:
            plt.show()
        else:
            plt.close()
            
    
    return fig, axes

def make_n_plots(N=None, cols=None, **kwargs):
    """
        creates a grid of N plots, with 'cols' number of columns
        and automatically sizes. passes any kwargs to plt.figure()
    
    """
    if all([N == None, cols == None]):
        print("No values passed, using demo values of 11 plots, 3 columns")
        N = 11
        cols = 3
    elif N is None:
        N = 11
        print("N was not passed, setting to default value N=11")
    elif cols is None:
        cols = 3
        print("cols was not passed, setting to default value cols=3")
    
    rows = int(math.ceil(N / cols))
    gs = gridspec.GridSpec(rows, cols)    

    if 'figsize' not in kwargs.keys():
        kwargs['figsize'] = (cols*3.5, rows*4.0)
        
    fig = plt.figure(**kwargs)
    axes = []
    for n in range(N):
        ax = fig.add_subplot(gs[n])
        axes.append(ax)
        

    fig.tight_layout()
    return fig, axes


def prep_cfgs(all_cfgs):
    """ 
    all_cfgs should be a list of dictionaries
    """
    # try: del del_target 
    # except: print("ResFreqQubitFreq_config does not exist, proceeding with initializing cfg.")  
    
    # all_dicts is a list of dicts so we unpack 
    # all keywords and values with comprehension
    # final_cfg = {k:v for list_item in args for (k,v) in list_item.items()}
    # for key in all_cfgs:
        # print(key)

    final_cfg = {}
    for cfg_dict in all_cfgs:   
        # print(cfg_dict)    
              
        # add "f_start" to cfg if it's specified by center+span
        for key, val in cfg_dict.items():
            if "center" in key:
                # print(key)
                span_key = key[0:2] + "span"
                start_key = key[0:2] + "start"
                step_key = key[0:2] + "step"
                center_key = key
                # print(start_key)
                start = cfg_dict[center_key] + cfg_dict[span_key] + cfg_dict[step_key]
                final_cfg[start_key] = start

        final_cfg.update(cfg_dict)

    return final_cfg


def gen_sweep_arrays(sweep_dict):
    """ arg should be a dict with "key" = var name, "value" = config dict """

    all_sweep_arrays = []
    for cfg_key, old_cfg_dict in sweep_dict.items():
        print(f"  Generating {cfg_key}")
        # all the params have some prefix, so 
        # just check every key to see if it contains
        # one of these parameters, and if it does
        # then just copy it without the prefix
        cfg_dict = old_cfg_dict.copy()  # can't update a dict as you loop over it...?
        for key, val in old_cfg_dict.items():
            if "center" in key: cfg_dict["center"] = val
            if "span" in key: cfg_dict["span"] = val
            if "start" in key: cfg_dict["start"] = val
            if "end" in key: cfg_dict["stop"] = val
            if "stop" in key: cfg_dict["stop"] = val
            if "step" in key: cfg_dict["step"] = val

        # now build the array but make sure the dict
        # only specified one method
        step = cfg_dict["step"]
        if "center" in cfg_dict and "start" not in cfg_dict:
            center = cfg_dict["center"]
            span = cfg_dict["span"]
            array = np.arange(center - span/2, center + span/2 + step, step)      

        elif "start" in cfg_dict and "center" not in cfg_dict:
            start = cfg_dict["start"]
            if "end" in cfg_dict:   stop = cfg_dict["end"]
            if "stop" in cfg_dict:  stop = cfg_dict["stop"]
            array = np.arange(start, stop + step, step)            

        elif "start" in cfg_dict and "center" in cfg_dict:
            print("error, both 'start' and 'center' found in dict")
            continue
        else:
            print("unknown error...? how did you manage this? spitting out all args, they should be dicts...")
            # for arg in args:
            #     print(type(arg), len(arg))
        
        # print(len(array))
        all_sweep_arrays.append(array)
    
    return all_sweep_arrays


def save_data(fpts, data, save_dir, title, verbose=False):
    tz = pytz.timezone('US/Eastern')
    today = datetime.now(tz)
    df = pd.DataFrame({'Frequency' : fpts})

    title = today.strftime("%m_%d_%H%M%p_")
    filename = save_dir + title + ".csv"
 
    try: 
        if type(data) == tuple & data.iscomplex():  # probably ("atten_XX", cmpl)
            if verbose: print("condition 1: type(data) == tuple & data[1].iscomplex()")
            df[data[0]] = data[1]  # e.g. "Atten_10" : [a+1j*b, c+1j*d, e+1j*f...n+1j*m] 
    
        elif type(data) == dict:
            if verbose: print("condition 2: type(data) == dict:")
            data_df = pd.DataFrame.from_dict(data)
            freq_df = pd.DataFrame(fpts, columns=["Frequency"])
            final_dict = pd.concat([freq_df, data_df], axis=1)

        else:
            if verbose: print("condition 3: else")
            df = pd.DataFrame(data)
            df.insert(loc=0, column='Frequency', value=fpts)

        df.to_csv(filename)
        print("Successfully saved to: {}".format(filename))

    except Exception as e:
        
        filename = './{}_Recovered_Data_{}.csv'.format(title, today.strftime("%m_%d_%H%M") )

        recovered_data = [*data.keys(), *data.values()]  # first row is gain_xxxx values, rest is data
        np.savetxt(filename, recovered_data, delimiter=',')
        print("failed to save.")
        print(e)


def fit_sin(tt, yy, debug=False):
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    tt = numpy.array(tt)
    yy = numpy.array(yy)
    ff = numpy.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
    Fyy = abs(numpy.fft.fft(yy))
    guess_freq = abs(ff[numpy.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    # guess_amp = numpy.std(yy) * 2.**0.5
    guess_amp = np.abs(max(yy) - min(yy))/2
    guess_offset = numpy.mean(yy)
    guess = numpy.array([guess_amp, 2.*numpy.pi*guess_freq, 0, guess_offset])
    
    # set A upper bound to 2*max of dataset, w bound to at least 10 kHz
    # guess_bounds = ([0, 0, 0, 0], [np.inf, 1000, 2*np.pi, np.inf])

    # disable ampl bound
    # guess_bounds = ([0, 0, -np.pi, -np.inf], [np.inf, np.inf, np.pi, np.inf])  

    # disable all bounds except positive A, w, phase = [0, 2pi]
    guess_bounds = ([max(yy)/10, 1, 0, 0], [max(yy), 1000, 2*np.pi, 1])
    
    print(f"{max(yy):1.2e}, {min(yy):1.2e}")

    # if debug == True:
    #     for (x0, guess_min, guess_max) in zip(guess, guess_bounds[0], guess_bounds[1]):
    #         print(f"{x0:.2f} in [{guess_min:.2f}, {guess_max:.2f}]")
    #         if (guess_max - x0) < 0 and guess_max != np.inf:
    #             print(f"  Error: {guess_max:.2f} - {x0:.2f} = {guess_max - x0:.2f}")
    #         if (x0 - guess_min) < 0:
    #             print(f"  Error: {guess_min:.2f} - {x0:.2f} = {guess_min - x0:.2f}")

    def sinfunc(t, A, w, p, c):  return A * numpy.sin(w*t + p) + c
    

    popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess, bounds=guess_bounds)
    A, w, p, c = popt
    f = w/(2.*numpy.pi)
    fitfunc = lambda t: A * numpy.sin(w*t + p) + c

    
    
    return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": numpy.max(pcov), "rawres": (guess,popt,pcov)}


def qick_hist(data=None, plot=True, ran=1.0):
    ig = data[0]
    qg = data[1]
    ie = data[2]
    qe = data[3]

    numbins = 200
    
    xg, yg = np.median(ig), np.median(qg)
    xe, ye = np.median(ie), np.median(qe)

    if plot==True:
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(16, 4))
        fig.tight_layout()

        axs[0].scatter(ig, qg, label='g', color='b', marker='*')
        axs[0].scatter(ie, qe, label='e', color='r', marker='*')
        axs[0].scatter(xg, yg, color='k', marker='o')
        axs[0].scatter(xe, ye, color='k', marker='o')
        axs[0].set_xlabel('I (a.u.)')
        axs[0].set_ylabel('Q (a.u.)')
        axs[0].legend(loc='upper right')
        axs[0].set_title('Unrotated')
        axs[0].axis('equal')

    """Compute the rotation angle"""
    theta = -np.arctan2((ye-yg),(xe-xg))
    
    """Rotate the IQ data"""
    ig_new = ig*np.cos(theta) - qg*np.sin(theta)
    qg_new = ig*np.sin(theta) + qg*np.cos(theta) 
    ie_new = ie*np.cos(theta) - qe*np.sin(theta)
    qe_new = ie*np.sin(theta) + qe*np.cos(theta)
    
    """New means of each blob"""
    xg, yg = np.median(ig_new), np.median(qg_new)
    xe, ye = np.median(ie_new), np.median(qe_new)
    
    #print(xg, xe)
    
    xlims = [xg-ran, xg+ran]
    ylims = [yg-ran, yg+ran]

    if plot==True:
        axs[1].scatter(ig_new, qg_new, label='g', color='b', marker='*')
        axs[1].scatter(ie_new, qe_new, label='e', color='r', marker='*')
        axs[1].scatter(xg, yg, color='k', marker='o')
        axs[1].scatter(xe, ye, color='k', marker='o')    
        axs[1].set_xlabel('I (a.u.)')
        axs[1].legend(loc='lower right')
        axs[1].set_title('Rotated')
        axs[1].axis('equal')

        """X and Y ranges for histogram"""
        
        ng, binsg, pg = axs[2].hist(ig_new, bins=numbins, range = xlims, color='b', label='g', alpha=0.5)
        ne, binse, pe = axs[2].hist(ie_new, bins=numbins, range = xlims, color='r', label='e', alpha=0.5)
        axs[2].set_xlabel('I(a.u.)')       
        
    else:        
        ng, binsg = np.histogram(ig_new, bins=numbins, range = xlims)
        ne, binse = np.histogram(ie_new, bins=numbins, range = xlims)

    """Compute the fidelity using overlap of the histograms"""
    contrast = np.abs(((np.cumsum(ng) - np.cumsum(ne)) / (0.5*ng.sum() + 0.5*ne.sum())))
    tind=contrast.argmax()
    threshold=binsg[tind]
    fid = contrast[tind]
    axs[2].set_title(f"Fidelity = {fid*100:.2f}%")

    return fid, threshold, theta
