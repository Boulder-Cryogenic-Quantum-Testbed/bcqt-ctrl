# %%

'''
    helper_plot.py
'''

# print("    loading helper_plot.py")

# %%

import glob, os, sys, time

import matplotlib.pyplot as plt
import pandas as pd
import regex as re
import numpy as np
import scipy as sp

import helper_misc as hm
import helper_fit as hf
import helper_load as hl

# %%
def plot_multiple_resonators(filepaths, column_headers=["Freq", "Magn", "Phase"], plot_config_dict=None, debug=False, **kwargs):
    
    if plot_config_dict is None:
        plot_config_dict = {
            "add_zero_lines" : True,
            "plot_complex" : True,
            "find_peaks" : False,
            "save_plot" : True,
            "plot_filepath" : None,
            "plot_filename" : None,
        }
    
    # add all kwargs to plotting dict
    plot_config_dict.update(kwargs)
    
    all_figs, all_axes = [], []
    
    # TODO: I should have just included the csv loading one by one, rather than all at once
    # dataframe_dict = hl.load_many_csvs_as_dataframes(search_dir)
    
    for csv_filepath in filepaths:
        
        resonator_folder = os.path.dirname(csv_filepath)
        plot_directory = f"{resonator_folder}\\data_plots"
        hm.check_and_make_dir(plot_directory)
        
        df = pd.read_csv(csv_filepath, names=column_headers)
        filename = os.path.basename(csv_filepath).replace(".csv","")
        
        plot_config_dict["plot_filepath"] = f"{plot_directory}"
        plot_config_dict["plot_filename"] = f"{filename}_plot.csv"
        plot_config_dict["plot_title"] = filename.replace(".csv","")
        
        freqs_GHz = df["Freq"] if any(df["Freq"] >= 1e9) else df["Freq"] * 1e9
        magn_dBm = df["Magn"]
        phase = df["Phase"]
        
        fig, axes = plot_S21_data(freqs_GHz, plot_config_dict, magn_dBm=magn_dBm, phase=phase, debug=debug)    
        
        all_figs.append(fig)
        all_axes.append(axes)            

    return all_figs, all_axes


# %%
def plot_S21_data(freq, plot_config_dict, debug=False, **kwargs):

    fig, axes = None, None
    
    keys = kwargs.keys()
    if debug: 
        print("    ~~ printing all method kwargs")
        # for key, value in kwargs.items():
        #     if type(value) is not bool:
        #         if len(value) < 10:
        #             print(key, value)
        #         else:
        #             print(key, type(value))
        #     else:
        #         print(key, value)

    
    ############ check method kwargs for data loading
    if any(freq >= 1e9):  # check if frequencies are not in GHz
        freq /= 1e9  # if so, assume Hz then divide by 1e9
    if "real" in keys: # real & imag -> cmpl
        if debug: print("    ~~ Received real & imag dataset")
        real = kwargs["real"]
        imag = kwargs["imag"]
        cmpl = real + 1j*imag
        del kwargs["imag"]
        del kwargs["real"]
    if "complex" in keys: # cmpl -> real & imag
        if debug: print("    ~~ Received complex dataset")
        cmpl = kwargs["complex"]
        real = np.real(cmpl)
        imag = np.imag(cmpl)
        del kwargs["complex"]
    if "magn_lin" in keys: # cmpl -> real & imag
        if debug: print("    ~~ Received magn_lin and phase dataset")
        magn_lin = kwargs["magn_lin"]
        magn_dBm = np.log10(magn_lin)*20
        phase = kwargs["phase"]
        # check if phase is in degrees
        if any(phase > 2.1*np.pi) or any(phase < -2.1*np.pi):
            if debug: print("hf.quick_plot_data:  Converting input phase from degrees to radians.")
            phase_deg = kwargs["phase"]
            phase = np.rad2deg(phase_deg)
        cmpl = magn_lin * np.exp(1j * phase)
        real = np.real(cmpl)
        imag = np.imag(cmpl)
    if "magn_dBm" in keys: # cmpl -> real & imag
            if debug: print("    ~~ Received magn_dBm and phase dataset")
            magn_dBm = kwargs["magn_dBm"]
            magn_lin = 10**(magn_dBm/20)
            phase = kwargs["phase"]
            
            # check if phase is in degrees
            if any(phase > 2.1*np.pi) or any(phase < -2.1*np.pi):
                if debug: print("hf.quick_plot_data:  Converting input phase from degrees to radians.")
                phase_deg = kwargs["phase"]
                phase = np.deg2rad(phase_deg)
            cmpl = magn_lin * np.exp(1j * phase)
            real = np.real(cmpl)
            imag = np.imag(cmpl)


    ############ check plot config dictionary
    plot_config_dict.update(**kwargs)
    config_keys = plot_config_dict.keys()
    if debug: 
        print("    ~~ printing all plot_config_dict items ")
        for key, value in plot_config_dict.items():
            if type(value) is not bool:
                if len(value) < 10:
                    print(key, value)
                else:
                    print(key, type(value))
            else:
                print(key, value)
    
    # TODO: use default matplotlib rcparams like a normal human being...   
    if "plot_title" in config_keys:
        fig_title = plot_config_dict["plot_title"]     
        
    if "figsize" in config_keys:
        if debug: print("    ~~ Received figsize")
        figsize = plot_config_dict["figsize"] 
    else:
        figsize = (8, 6)
        
    if "mosaic" in config_keys:
        if debug: print("    ~~ Received mosaic")
        mosaic = plot_config_dict["mosaic"]    
    else:
        if "plot_complex" in config_keys:
            mosaic = "AACCC\n BBCCC"
        else:
            mosaic = "AAA\n BBB"
            
    if "markersize" in config_keys:
        markersize = plot_config_dict["markersize"]    
    else:
        markersize = 3
        
    if "label_size" in config_keys:
        label_size = plot_config_dict["label_size"]    
    else:
        label_size = 12
        
    if "tick_label_size" in config_keys:
        tick_label_size = plot_config_dict["tick_label_size"]    
    else:
        tick_label_size = 12
        
    if "title_size" in config_keys:
        title_size = plot_config_dict["title_size"]    
    else:
        title_size = 16
        
    if "text_size" in config_keys:
        text_size = plot_config_dict["text_size"]    
    else:
        text_size = 12
        
    magn_lin = np.abs(cmpl)
    phase = np.unwrap(np.angle(cmpl))

    fig, axes_dict = plt.subplot_mosaic(mosaic, figsize=figsize, tight_layout=True)
    ax1, ax2 = axes_dict["A"], axes_dict["B"]
    axes = list(axes_dict.values())
    
    ax1.plot(freq, magn_lin, 'ko', markersize=markersize, alpha=0.9)
    ax1.set_xlabel("Frequency [GHz]", size=label_size)
    ax1.set_ylabel("S21 [a.u.]", size=label_size)
    ax1.set_title("Magnitude Data", size=title_size)
    
    ax2.plot(freq, phase, 'ro', markersize=markersize, alpha=0.9)
    ax2.set_xlabel("Frequency [GHz]", size=label_size)
    ax2.set_ylabel("Phase [rad]", size=label_size)
    ax2.set_title("Phase Data", size=title_size)
    
    if "find_peaks" in config_keys:
        pks_idx, _ = sp.signal.find_peaks(-1*magn_lin, distance=len(freq)*0.2, prominence=2)
        pk_freqs = [freq[idx] for idx in pks_idx]
        
        # TODO: maybe rainbow instead of red peaks? :)
        if len(pk_freqs) <= 5 and len(pk_freqs) != 0:
            for pk in pk_freqs:
                print(f"     > Resonance at:  {pk:1.9} GHz")
                ax1.plot(freq[pks_idx], magn_lin[pks_idx], 'ro', label=f"{pk:1.6f} GHz",
                    markerfacecolor='none', markersize=markersize, markeredgewidth=2)
            ax1.legend()
        elif len(pk_freqs) == 0:
            ax1.text(0.6, 0.9, "No peaks found", color='red', fontsize=text_size,
                         horizontalalignment='center', verticalalignment='center', transform = ax1.transAxes)
    
        else:
            ax1.text(0.6, 0.9, "Too many \npeaks found", color='red', fontsize=text_size,
                         horizontalalignment='center', verticalalignment='center', transform = ax1.transAxes)
    
    if "plot_complex" in config_keys:
        ax3 = axes_dict["C"]
        ax3.plot(real, imag, 'bo', markersize=markersize, markerfacecolor='none')
        ax3.set_xlabel("Real [a.u.]", size=label_size)
        ax3.set_ylabel("Imag [a.u.]", size=label_size,)
        ax3.set_title("Complex Data", size=title_size)
        ax3.set_aspect('equal') 
        ax3.yaxis.tick_right()
        ax3.yaxis.set_label_position("right")
      
    fig.suptitle(f"Raw Data Plot\n\n{fig_title}", size=title_size+2)
    
    for ax in axes:
        ax.tick_params(axis='x', labelsize=tick_label_size)
        ax.tick_params(axis='y', labelsize=tick_label_size)
        
    if "add_zero_lines" in config_keys:
        val = plot_config_dict["add_zero_lines"]
        if debug: print(f"    ~~ Received add_zero_lines: = {val}")
        if val and "plot_complex" in config_keys:
            ax3.axhline(0, linestyle=':', color='k')  
            ax3.axvline(0, linestyle=':', color='k')
    
    fig.tight_layout()
    
    if "show_plot" in config_keys:
        plt.show()
    else:
        plt.close()
        
    if "plot_filepath" in config_keys and "save_plot" in config_keys:
        
        # TODO: add error messages for save_plot = true and plot_filepath = false, etc
        plot_filepath = plot_config_dict["plot_filepath"]
        plot_filename = plot_config_dict["plot_filename"]
        
        if plot_filepath is not None:
            print(f"    Saving plot for {plot_filename} in {plot_filepath}")
            fig.savefig(f"{plot_filepath}\\{plot_filename}.png", format='png')
        
    return fig, axes
        
      
 
def plot_dataframe(list_of_df_dicts):
    ## Plot the internal and external quality factors separately
    fig_d, ax_d = plt.subplots(1, 1, tight_layout=True)
    
    for df in list_of_df_dicts:
        for idx, (name, qiqc_df) in enumerate(df.items()):
            # #debug plots
            # plt.plot(qiqc_df["navg"], qiqc_df["Qi"])
            # plt.plot(qiqc_df["Power [dBm]"], qiqc_df["Qi"])
            # print(qiqc_df["Power [dBm]"], qiqc_df["Qi"])
            
            fig_d, ax_d = plt.subplots(1, 1, tight_layout=True)
            label = os.path.basename(os.path.basename(os.path.dirname(name)))
            try:
                if label != prev_label:
                    # fig_d, ax_d = plt.subplots(1, 1, tight_layout=True)
                    print("label != prev_label")
                    print(label, prev_label)
            except:
                pass
            prev_label = label
            
            markers = ['o', 'd', '>', 's', '<', 'h', '^', 'p', 'v']
            colors  = plt.rcParams['axes.prop_cycle'].by_key()['color']
            powers = qiqc_df["navg"]
            Qi = qiqc_df["Qi"]
            Qi_err = np.asarray(qiqc_df['Qi error'])
            delta = 1 / Qi
            delta_err = Qi_err / Qi**2
            csize = 5
            
            ax_d.errorbar(powers, delta, yerr=delta_err, marker='d', ls='', color=colors[idx], 
                          ms=10, capsize=csize, label=label)
            ax_d.set_xscale("log")
            ax_d.legend()
            
            
    # return fig_d, ax_d 
  
  
  
def plot_data_file_dict(data_file_dict, plot_config_dict):
    # TODO: replace with plot_21_data
    for filename, filepath in data_file_dict.items():
        print(f"\n~~~> Loading {filename} from '{filepath}' ")
        fn = filepath + filename
        data = np.genfromtxt(fn, delimiter=',').T
        freqs = data[0] / 1e9
        magn = data[1]
        phase = data[2]
        cmpl = magn * np.exp(1j * phase)
        
        # if freqs[0] <= 4:  # crop the data arrays to get the peak hiding at 4.7
        #     peak_value = 4.783  # GHz
        #     peak_span = 0.0015  # GHz
        #     peak_idx = np.abs(freqs - peak_value).argmin()
            
        #     # calculate how many indices we need to have to capture peak_span
        #     full_span = freqs[-1] - freqs[0]
        #     peak_span_idx = (peak_span/2) // (full_span/len(freqs))  
        #     lower_idx = int(peak_idx - peak_span_idx)
        #     upper_idx = int(peak_idx + peak_span_idx)
            
        #     freqs = freqs[lower_idx:upper_idx]
        #     magn = magn[lower_idx:upper_idx]
        #     phase = phase[lower_idx:upper_idx]
        #     cmpl = magn * np.exp(1j * phase)
            
             
        
        plot_config_dict["plot_title"] = filename
        # plot_config_dict["mosaic"] = "AAACC\nBBBCC"
        plot_config_dict["figsize"] = (12,8)
        plot_config_dict["markersize"] = 2
        
        fig, axes = hf.quick_plot_S21_data(freqs, plot_config_dict, complex=cmpl) 

        span = (freqs[-1]-freqs[0])
        pks_idx, _ = sp.find_peaks(-1*magn, distance=len(freqs)*0.2, prominence=2)
        pk_freqs = [freqs[idx] for idx in pks_idx]
        
        # TODO: maybe rainbow instead of red peaks? :)
        if len(pk_freqs) <= 5:
            for pk in pk_freqs:
                print(f"     > Resonance at:  {pk:1.9} GHz")
                axes[0].plot(freqs[pks_idx], magn[pks_idx], 'ro', label=f"{pk:1.6f} GHz",
                    markerfacecolor='none', markersize=16, markeredgewidth=2)
            axes[0].legend()
        else:
            axes[0].text(0.2, 0.2, "Too many \npeaks found", color='red', fontsize=16,
                         horizontalalignment='center', verticalalignment='center', transform = axes[0].transAxes)
        
    # return fig, axes


  