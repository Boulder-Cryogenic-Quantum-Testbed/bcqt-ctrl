# quick_methods

# hopefully this isnt permanent :D

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## TODO: these should go in the DataProcessor object
def unpack_df(df): 
    freqs = df["Frequency"]
    magn_dB = df["S21 [dB]"]
    phase_rad = df["Phase [rad]"]
    return freqs, magn_dB, phase_rad


def plot_s2p_df(df, plot_complex=True, track_min=True):
    
    """
        assumes df was returned by the VNA driver's 'return_data_s2p' function
            if not, the format of the df is simply:
                  | Frequency | 'sparam' magn_dB | 'sparam' phase_rad | ...
            for however many sparam columns 
            
        this will also check if there is a row with the index "datetime.now()" with
            the value of that function in each entry to use as the time
    """
    
    # start by grabbing the first three letters of each column
    # if formatted appropriately, should be something like
    # [ 'Freq', 'S11', 'S11', 'S21, 'S21', .... ]
    # Then count all duplicates so we are left with all s-parameters
    # e.g. for a four param s2p we will have sparams = ['S11', 'S12', 'S21', 'S22']
    all_col_sparams = [col[:3] for col in df.columns]
    sparams = list(set([x for x in all_col_sparams if all_col_sparams.count(x) > 1]))

    # grab timestamp from datetime.now() index and drop it, 
    # with inplace=False to not affect the original df 
    datetime = df.iloc[0].values[0]
    timestamp = datetime.strftime("%m_%d_%I%M%p")
    fixed_df = df.drop(index='datetime.now()', inplace=False)

    freqs = fixed_df["Frequency"].values.astype(float)

    # now go through all the s-parameters and grab the magn and phase data using filter()
    for sparam in sparams:
        magn_and_phase = fixed_df.filter(like=sparam)
        magn_dB, phase_rad =  magn_and_phase.iloc[:,0].values.astype(float), magn_and_phase.iloc[:,1].values.astype(float)
        axes = plot_data_with_pandas(freqs, magn_dB=magn_dB, phase_rad=phase_rad, plot_complex=plot_complex, track_min=track_min, suptitle=timestamp)

    # subplot mosaic returns a dict....
    fig = list(axes.values())[0].get_figure()
    fig.suptitle(timestamp)
    fig.tight_layout()
    return axes
        

# TODO: add functionality for taking dataframes instead of individual np arrays
def plot_data_with_pandas(freqs, magn_dB, phase_deg=None, phase_rad=None, plot_complex=True, track_min=False, suptitle=None, **kwargs):

    if phase_rad is None and phase_deg is not None:
        # degrees were given
        phase_rad = np.unwrap(np.deg2rad(phase_deg))
    elif phase_rad is not None and phase_deg is None:
        # radians were given
        pass
    elif phase_rad is None and phase_deg is None:
        # both are None
        raise ValueError("One of phase_rad and phase_deg must be given!")
    else:
        # both were given
        if phase_rad != np.deg2rad(phase_deg):
            raise ValueError(f"Both radians and degrees were given, but they don't match!\n  {phase_rad != np.deg2rad(phase_deg) = }")
    
    ## convert dataset to linear, real, and imag
    magn_lin = 10**(magn_dB/20)
    cmpl = magn_lin * np.exp(1j * phase_rad)
    real, imag = np.real(cmpl), np.imag(cmpl)

    # create a new plot or use one given as arg
    # # %% plot data
    mosaic = "AACC\nBBCC"
    fig, axes = plt.subplot_mosaic(mosaic, figsize=(10,5))
    ax1, ax2, ax3 = axes["A"], axes["B"], axes["C"]
    
    ax3.plot(real, imag, 'o', markersize=6, **kwargs)
    ax3.set_ylabel("Imag")
    ax3.set_title("Real vs Imag")
    ax3.axhline(0, linestyle=':', linewidth=1, color='k')
    ax3.axvline(0, linestyle=':', linewidth=1, color='k')
    ax3.set_aspect("equal")
    
    freq_argmin = magn_lin.argmin()
    freq_min = freqs[freq_argmin]

    if track_min is True:
        new_freqs = (freqs - freq_min)/1e3
        freq_min_label = f"$f_{"{min}"}$ = {freq_min/1e6:1.3f} MHz"
        ax1.axvline(0, linestyle='--', color='k', linewidth=1)
        ax2.axvline(0, linestyle='--', color='k', linewidth=1)
        ax3.plot(real[freq_argmin], imag[freq_argmin], 'y*', markersize=8, label=freq_min_label)
        fig.legend()
    else:
        new_freqs = freqs/1e3
        
    # magn and phase
    ax1.plot(new_freqs, magn_lin, "r.", markersize=6, )
    ax2.plot(new_freqs, phase_rad, "b.", markersize=6, )

    ax1.set_title(f"Freq vs Magn [$f_{"{min}"}$ = {freq_min/1e9:1.6f} GHz]")
    ax2.set_title(f"Freq vs Phase [$f_{"{min}"}$ = {freq_min/1e9:1.6f} GHz]")

    ax1.set_xlabel("Frequency $\\Delta f$ [kHz]")
    ax2.set_xlabel("Frequency $\\Delta f$ [kHz]")
    ax3.set_xlabel("Real")

    ax1.set_ylabel("S21 [dB]")
    ax2.set_ylabel("Phase [Rad]")
    
    if plot_complex is False:
        fig.delaxes(ax3)
        axes = [ax1, ax2]
        
    if suptitle is not None:
        fig.suptitle(suptitle)
    
    fig.tight_layout()
    
    return axes
