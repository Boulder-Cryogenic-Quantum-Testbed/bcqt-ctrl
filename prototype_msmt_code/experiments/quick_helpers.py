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

def plot_data_with_pandas(freqs, magn_dB, phase_deg=None, phase_rad=None, ax=None, **kwargs):

    # check phase input
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
        
    df = pd.DataFrame.from_dict(data={"Frequency":freqs, "S21 [dB]":magn_dB, "Phase [rad]":phase_rad}, orient="columns")
    
    ## convert dataset
    magn_lin = 10**(magn_dB/20)
    cmpl = magn_lin * np.exp(1j * phase_rad)
    real, imag = np.real(cmpl), np.imag(cmpl)

    # create a new plot or use one given as arg
    if ax is None:
        # # %% plot data
        mosaic = "AACC\nBBCC"
        fig, axes = plt.subplot_mosaic(mosaic, figsize=(15,8))
        ax1, ax2, ax3 = axes["A"], axes["B"], axes["C"]
    else:
        # only plot complex portion, then return early
        ax3 = ax
        fig = ax3.get_figure()
        
    ax3.plot(real, imag, 'o', markersize=6, **kwargs)
    ax3.set_ylabel("Imag")
    ax3.set_title("Real vs Imag")
    ax3.axhline(0, linestyle=':', linewidth=1, color='k')
    ax3.axvline(0, linestyle=':', linewidth=1, color='k')
    ax3.set_aspect("equal")
    
    if ax is not None:
        fig.tight_layout()
        return df, fig, [ax3]

    freq_min_idx = magn_lin.argmin()
    freq_min = freqs[freq_min_idx]

    # magn and phase
    ax1.plot((freqs - freq_min)/1e3, magn_lin, "r.", markersize=6, )
    ax2.plot((freqs - freq_min)/1e3, phase_rad, "b.", markersize=6, )

    ax1.set_title(f"Freq vs Magn [$f_{"{min}"}$ = {freq_min/1e9:1.6f} GHz]")
    ax2.set_title(f"Freq vs Phase [$f_{"{min}"}$ = {freq_min/1e9:1.6f} GHz]")

    ax1.set_xlabel("Frequency $\\Delta f$ [kHz]")
    ax2.set_xlabel("Frequency $\\Delta f$ [kHz]")
    ax3.set_xlabel("Real")

    ax1.set_ylabel("S21 [dB]")
    ax2.set_ylabel("Phase [Rad]")
    fig.tight_layout()

    
    return df, fig, axes
