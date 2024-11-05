import uncertainties, sys
import matplotlib.pyplot as plt
import re as re
import numpy as np
import scipy as sp
import pandas as pd

from datetime import datetime
from pathlib import Path

# lazy... lab PC 1 and PC 2
scres_path_1 = Path(r"C:\Users\Lehnert Lab\GitHub\scresonators")
# scres_path_1 = Path(r"C:\Users\Lehnert Lab\GitHub\scresonators-andre")
scres_path_2 = Path(r"E:\GitHub\scresonators")
scres_path_3 = Path(r"/Users/jlr7/OneDrive - UCB-O365/GitHub/scresonators")

sys.path.append(str(scres_path_1))
sys.path.append(str(scres_path_2))
sys.path.append(str(scres_path_3))


import fit_resonator.resonator as res
import fit_resonator.fit as fsd


# DataResults Class
class DataAnalysis():
    def __init__(self, processed_data, dstr=None):
        """
            for now, processed_data is just a weird dictionary:
            
            The dictionary should have this form:
        
                dict_keys = filenames of given dataset
                
                dict_vals = tuple = (df, expt_config, init_params)
                            
                            df is a dataframe produced by measurement code, has columns
                            
                            cols = ( "Frequency [Hz]", "Magn [dB]", "Phase [Rad]" )
                            
                            fit_init_params = list of initial values for 
                                                [Q, Qc, w1, phi]
            
            so that we can loop over the resonator measurements using df_dict.items():
                for filename, data_tuple in df_dict.items():
                    df, init_p0 = data_tuple
        
        """
        self.data = processed_data
        
        if dstr is not None:
            self.dstr = dstr
        else:
            self.dstr = datetime.today().strftime("%m_%d_%I%M%p")
        
    def display_results(self):
        print(f"Displaying results: {self.processed_data}")
        
        
    #########################################
    #########################################
    #####  copy pasted from helper_fit  #####
    #########################################
    #########################################

    
    def fit_single_res(self, data_df : pd.DataFrame, save_dir : str = None, save_dcm_plot=False, save_path=None, manual_init=None, plot_title=""):
        """
            Fit a single resonator S21 measurement
        """
        
        # use pathlib to ensure save directory exists
        # TODO: move this to DataProcessor
        if save_dir is None:
            save_dir = Path(r"data").absolute()
            save_dir.mkdir(parents=True, exist_ok=True)
        elif type(save_dir) == Path:
            save_dir = save_dir.absolute()
            save_dir.mkdir(parents=True, exist_ok=True)
        else:
            save_dir = Path(save_dir).absolute()
            save_dir.mkdir(parents=True, exist_ok=True)
        
        # fit resonator compatability
        save_dir = str(save_dir)
        
        
        ## create scresonators Method object
        fit_type = 'DCM'
        MC_iteration = 10
        MC_rounds = 1e3
        MC_fix = []
        normalize_pts = 10

        myres = res.Resonator()
        print(res.__file__)
        print(myres)
        
        myres.preprocess_method = "circle"
        myres.normalize_pts = normalize_pts
        myres.plot = 'png'
        
        freqs = data_df["Frequency"].values
        magn_dB = data_df["S21 [dB]"].values
        phase_rad = data_df["Phase [rad]"].values
        magn_lin = 10 ** (magn_dB / 20)
        
        
        # print(freqs, magn_dB, phase_rad, magn_lin)
        # replaces myres.from_file()
        myres.data = res.ResonatorData(freqs, magn_dB, phase_rad, magn_lin)
        
        # Setup the method for fitting
        try: 
            myres.fit_method(fit_type, MC_iteration, MC_rounds=MC_rounds,
                              MC_fix=MC_fix, MC_step_const=0.3,
                              manual_init=manual_init, )
        except Exception as ex:
            print(f'Exception in fit_single_res:\n{ex}')
        
        fig = None
        
        params, conf_intervals, err, init1, fig = fsd.fit(myres) 
        if save_dcm_plot is True and fig is not None:
            
            if save_path is None:
                save_path = Path(rf".\{self.dstr}\dcm_fits")
            else:
                save_path = save_path / "dcm_fits"
                
                
            fig.suptitle(plot_title, fontsize=36)
            plt.show()
            
            try:
                save_path.mkdir(parents=True, exist_ok=True)
                file_path = str(save_path / f"{plot_title}.png")
                fig.savefig(file_path)
                
            except Exception as e:
                print(f"\n Failed to save dcm plot:  {e}")
        
        return params, conf_intervals, err, init1, fig
        
       
    def power_to_navg(self, power_dBm, Qi, Qc, fc, Z0_o_Zr=1.):
        """
        Converts power to photon number following Eq. (1) of arXiv:1801.10204
        and Eq. (3) of arXiv:1912.09119
        """
        # Physical constants, Planck's constant J s
        h = 6.62607015e-34
        hbar = 1.0545718e-34

        # Convert dBm to W
        Papp = 10**((power_dBm - 30) / 10) # * 1e-3
        # hb_wc2 = np.pi * h * (fc_GHz * 1e9)**2
        fscale = 1. if fc > 1e9 else 1e9
        fc_GHz = fc * fscale
        hb_wc2 = hbar * (2 * np.pi * fc_GHz)**2

        # Return the power as average number of photons
        Q = 1. / ((1. / Qi) + (1. / Qc))
        navg = (2. / hb_wc2) * (Q**2 / Qc) * Papp

        return navg


    def fit_qiqcfc_vs_power(self, df_dict : dict, show_plots : bool = False,
                            save_dir : str = None, save_dcm_plot : bool = False, ):
        """
            Takes several measurements of a given resonator, fits them using self.fit_single_res(),
            and then extracts qi/qc/fc vs power for the power sweep entirely.
        
            The dictionary should have this form:
            
                dict_keys = filename of given dataset
                dict_vals = tuple consisting of (df, fit_init_params)
                                df : dataframe produced by measurement code, has columns
                                        cols = ( "Frequency [Hz]", "Magn [dB]", "Phase [Rad]" )
            
            so that we can loop over the resonator measurements using df_dict.items():
                for filename, data_tuple in df_dict.items():
                    df, init_p0 = data_tuple

        """
        
        # leftover variables from importing
        phi0 = 0
        Npts = len(df_dict)
        Q, Qc, Qi, fc, navg = np.zeros(Npts), np.zeros(Npts), np.zeros(Npts), np.zeros(Npts), np.zeros(Npts)
        Qc_err, Qi_err, fc_err, errs = np.zeros(Npts), np.zeros(Npts), np.zeros(Npts), np.zeros(Npts)
        all_powers = np.zeros(Npts)
        
        # use pathlib to ensure save directory exists
        # TODO: move this to DataProcessor
        if save_dir is None:
            save_dir = Path(r".\data\qiqcfc_vs_power")
            save_dir.mkdir(parents=True, exist_ok=True).absolute()
        elif type(save_dir) == Path:
            save_dir.mkdir(parents=True, exist_ok=True).absolute()
        else:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True).absolute()
        
        # loop through all the data provided in df_dict
        for idx, (filename, data_tuple) in enumerate(df_dict.items()):
            df, instr_config, init_p0 = data_tuple
            
            power = instr_config["power"]
            all_powers[idx] = power
            
            params, err, conf_int, fig = self.fit_single_res(df, save_dir, save_dcm_plot=True, manual_init=init_p0, phi0=phi0)

            Qcj = params[1] * np.exp(1j*(params[3] + phi0))
            Qij = 1. / (1. / params[0] - np.real(1. / Qcj))

            # Total quality factor
            Q[idx] = params[0]

            fscale = 1e9 if params[2] > 1e9 else 1

            # Store the quality factors, resonance frequencies
            Qc[idx] = np.real(Qcj)
            Qi[idx] = Qij
            fc[idx] = params[2] / fscale
            errs[idx] = err
            navg[idx] = self.power_to_navg(power, Qi[idx], Qc[0], fc[0])

            # Store each quantity's 95 % confidence intervals
            Qi_err[idx] = conf_int[1]
            Qc_err[idx] = conf_int[2]
            fc_err[idx] = conf_int[5] / fscale

            print(f'navg: {navg[idx]} photons')
            print(f'Q: {Q[idx]} +/- {conf_int[0]}')
            print(f'Qi: {Qi[idx]} +/- {Qi_err[idx]}')
            print(f'Qc: {Qc[idx]} +/- {Qc_err[idx]}')
            print(f'fc: {fc[idx]} +/- {fc_err[idx]} GHz')
            print('-------------\n')
            
            if show_plots is False:
                plt.close('all')
            else:
                fig.show()

        # Save the final dataset to file
        # TODO: clean up this garbage
        df = pd.DataFrame(np.vstack((all_powers, navg, fc, Qi, Qc, Q,
                        errs, Qi_err, Qc_err, fc_err)).T,
                columns=['Power [dBm]', 'navg', 'fc [GHz]', 'Qi', 'Qc', 'Q',
                    'error', 'Qi error', 'Qc error', 'fc error'])
                                        
        # self.dstr = datetime.datetime.today().strftime('%y%m%d_%H_%M_%S')
        filename_csv = Path(rf'qiqcfc_vs_power_{self.dstr}.csv')
        save_filepath = save_dir / filename_csv
        
        # save in data directory
        df.to_csv(save_filepath.absolute())  

        return df
     

    def power_sweep_fit_drv(self, df_dict,
                            atten=[0, -60],
                            sample_name=None,
                            powers_in=None,
                            show_dbm=False,
                            temperature=15e-3,
                            do_delta_tls_fit=False,
                            show_plots=False,
                            save_dcm_plot=False, 
                            save_dir=None, 
                            QHP_fix=False,
                            plot_twinx=True,
                            manual_init_list=None):
        
        """
            fit power vs loss for a set of resonator data that sweeps power
        """
        
        # use pathlib to ensure save directory exists
        # TODO: move this to DataProcessor
        if save_dir is None:
            save_dir = Path(r".\data\power_sweep_fit")
            save_dir.mkdir(parents=True, exist_ok=True).absolute()
        elif type(save_dir) == Path:
            save_dir.mkdir(parents=True, exist_ok=True).absolute()
        else:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True).absolute()
        
        
        # self.dstr = datetime.datetime.today().strftime('%y_%m_%d')
        
        # fontsizes
        fsize = 20
        csize = 5

        # make array of all input powers in dBm
        all_powers = np.zeros(len(df_dict))
        for idx, (filename, data_tuple) in enumerate(df_dict.items()):
            df, instr_config, init_p0 = data_tuple
            power = instr_config["power"]
            all_powers[idx] = power

        # Plot the results after gathering all of the fits
        df = self.fit_qiqcfc_vs_power(df_dict, show_plots = show_plots,
                            save_dir = save_dir, save_dcm_plot = save_dcm_plot)
        
        # Extract the powers, quality factors, resonance frequencies, and 95 %
        # confidence intervals
        Qi = np.asarray(df['Qi'])
        Qc = df['Qc']
        Q  = df['Q']
        navg = df['navg']
        delta = 1. / Qi
        fc = df['fc [GHz]']
        Qi_err = np.asarray(df['Qi error'])
        Qc_err = df['Qc error']
        delta_err = Qi_err / Qi**2
        fc_err = df['fc error']

        # Add attenuation to powers
        powers += sum(atten)

        # method inside method to abuse scope??? oof
        def pdBm_to_navg_ticks(p):
            n = np.abs(self.power_to_navg(powers[0::2], Qi[0::2], Qc[0], fc[0]))
            labels = [r'$10^{%.2g}$' % x for x in np.log10(n)]
            print(f'labels:\n{labels}')
            return labels

        if do_delta_tls_fit:
            method_output = self.fit_delta_tls(Qi, T, fc[0], Qc[0], powers,\
                    display_scales={'QHP' : 1e4, 'nc' : 1e6, 'Fdtls' : 1e-6}, 
                    QHP_fix=QHP_fix, Qierr=Qi_err)
            
            Fdtls, nc, QHP, Fdtls_err, nc_err, QHP_err, delta_fit, delta_fit_str = method_output
            
            print('\n')
            print(f'F * d0_tls: {Fdtls:.2g} +/- {Fdtls_err:.2g}')
            print(f'nc: {nc:.2g} +/- {nc_err:.2g}')
            print('\n')

        
        ## Plot the resonance frequenices
        fig_fc, ax_fc = plt.subplots(1, 1, tight_layout=True)
        ax_fc.set_xlabel('Power [dBm]', fontsize=fsize)
        ax_fc.set_ylabel('Res Freq Shift From High Power [GHz]', fontsize=fsize)
        ax_fc_top = ax_fc.twiny()

        plot_kwargs = {
            "figsize" : (8,6),
        }
        
        ## Plot the internal and external quality factors separately
        fig_qc, ax_qc = plt.subplots(1, 1, tight_layout=True, **plot_kwargs)
        fig_qi, ax_qi = plt.subplots(1, 1, tight_layout=True, **plot_kwargs)
        fig_qiqc, ax_qiqc = plt.subplots(1, 1, tight_layout=True, **plot_kwargs)
        fig_d, ax_d = plt.subplots(1, 1, tight_layout=True, **plot_kwargs)

        if plot_twinx is True:
            powers = np.abs(self.power_to_navg(powers, Qi, Qc[0], fc[0]))

        # Plot with / without error bars
        markers = ['o', 'd', '>', 's', '<', 'h', '^', 'p', 'v']
        colors  = plt.rcParams['axes.prop_cycle'].by_key()['color']
        
        ax_fc.errorbar(powers, fc - fc[0] , yerr=fc_err, marker='o', ls='', ms=10,
                capsize=csize)
        ax_qc.errorbar(powers, Qc, yerr=Qc_err, marker='o', ls='', ms=10,
                capsize=csize)
        ax_qiqc.errorbar(powers, Qi, yerr=Qi_err, marker='h', ls='', ms=10,
                capsize=csize, color=colors[5],
                label=r'$Q_i$')
        ax_qiqc.errorbar(powers, Qc, yerr=Qc_err, marker='^', ls='', ms=10,
                capsize=csize, color=colors[6],
                label=r'$Q_c$')
        ax_qi.errorbar(powers, Qi, yerr=Qi_err, marker='o', ls='', ms=10,
                capsize=csize)
        
        ax_d.errorbar(powers, delta,
                yerr=delta_err, marker='d', ls='', color=colors[1],
                ms=10, capsize=csize)
        
        if do_delta_tls_fit:
            ax_d.plot(powers, delta_fit, ls='-', label=delta_fit_str,
                color=colors[1])
            
        if show_dbm:
            for x, y, text in zip(powers, delta, powers_in):
                ax_d.text(x, y, f"{text} dBm", size=12,  rotation=45, rotation_mode="anchor",
                        horizontalalignment="left", verticalalignment="bottom")

        ax_qc.set_ylabel(r'$Q_c$', fontsize=fsize)
        ax_qi.set_ylabel(r'$Q_i$', fontsize=fsize)
        ax_qiqc.set_ylabel(r'$Q_i, Q_c$', fontsize=fsize)

        ax_d.set_ylabel(r'$Q_i^{-1}$', fontsize=fsize)

        power_str = f'{atten[0]} dB ext, {atten[1]} dB int attenuation'
        ax_qc.set_title(power_str, fontsize=fsize)
        ax_fc.set_title(power_str, fontsize=fsize)
        ax_qi.set_title(power_str, fontsize=fsize)
        ax_qiqc.set_title(power_str, fontsize=fsize)
        ax_d.set_title(power_str, fontsize=fsize)

        # Set the second (top) axis labels
        if plot_twinx:
            ax_d_top  = ax_d.twiny()
            ax_qc_top = ax_qc.twiny()
            ax_qi_top = ax_qi.twiny()
            ax_qiqc_top = ax_qiqc.twiny()

            ax_qc.set_xlabel('Power [dBm]', fontsize=fsize)
            ax_qi.set_xlabel('Power [dBm]', fontsize=fsize)
            ax_qiqc.set_xlabel('Power [dBm]', fontsize=fsize)
            ax_d.set_xlabel('Power [dBm]', fontsize=fsize)

            ax_qc_top.set_xlabel(r'Power [$\left<{n}\right>$]', fontsize=fsize)
            ax_qi_top.set_xlabel(r'Power [$\left<{n}\right>$]', fontsize=fsize)
            ax_qiqc_top.set_xlabel(r'[$\left<{n}\right>$]', fontsize=fsize)
            ax_d_top.set_xlabel(r'$\left<{n}\right>$', fontsize=fsize)
            ax_fc_top.set_xlabel(r'Power [$\left<{n}\right>$]', fontsize=fsize)
        
            # Update the tick labels to display the photon numbers
            ax_qc.set_xticks(powers[0::2])
            ax_qi.set_xticks(powers[0::2])
            ax_qiqc.set_xticks(powers[0::2])
            ax_d.set_xticks(powers[0::2])
            ax_fc.set_xticks(powers[0::2])
            
            ax_qc_top.set_xticks(ax_qc.get_xticks())
            ax_qi_top.set_xticks(ax_qi.get_xticks())
            ax_qiqc_top.set_xticks(ax_qi.get_xticks())
            ax_d_top.set_xticks(ax_d.get_xticks())
            ax_fc_top.set_xticks(ax_fc.get_xticks())
            
            ax_qc_top.set_xbound(ax_qc.get_xbound())
            ax_qi_top.set_xbound(ax_qi.get_xbound())
            ax_qiqc_top.set_xbound(ax_qi.get_xbound())
            ax_d_top.set_xbound(ax_d.get_xbound())
            ax_fc_top.set_xbound(ax_fc.get_xbound())
            
            ax_qc_top.set_xticklabels(pdBm_to_navg_ticks(ax_qc.get_xticks()))
            ax_qi_top.set_xticklabels(pdBm_to_navg_ticks(ax_qi.get_xticks()))
            ax_qiqc_top.set_xticklabels(pdBm_to_navg_ticks(ax_qi.get_xticks()))
            ax_d_top.set_xticklabels(pdBm_to_navg_ticks(ax_d.get_xticks()))
            ax_fc_top.set_xticklabels(pdBm_to_navg_ticks(ax_fc.get_xticks()))


            for ax in [ax_d_top, ax_d]:
                for tick in ax.get_xticklabels():
                    tick.set_rotation(45)

        else:
            ax_qc.set_xscale('log')
            ax_qi.set_xscale('log')
            ax_qiqc.set_xscale('log')
            ax_fc.set_xscale('log')
            ax_d.set_xscale('log')
            
            
            # ax_qc.set_yscale('log')
            ax_qi.set_yscale('log')
            ax_qiqc.set_yscale('log')
            # ax_fc.set_yscale('log')
            # ax_d.set_yscale('log')

            # Turn on all ticks
            ax_qc.get_xaxis().get_major_formatter().labelOnlyBase = False
            ax_qi.get_xaxis().get_major_formatter().labelOnlyBase = False
            ax_qiqc.get_xaxis().get_major_formatter().labelOnlyBase = False
            ax_fc.get_xaxis().get_major_formatter().labelOnlyBase = False
            ax_d.get_xaxis().get_major_formatter().labelOnlyBase = False

            ax_qc.set_xlabel(r'Power [$\left<{n}\right>$]', fontsize=fsize)
            ax_qi.set_xlabel(r'Power [$\left<{n}\right>$]', fontsize=fsize)
            ax_qiqc.set_xlabel(r'[$\left<{n}\right>$]', fontsize=fsize)
            ax_fc.set_xlabel(r'Power [$\left<{n}\right>$]', fontsize=fsize)
            ax_d.set_xlabel(r'$\left<{n}\right>$', fontsize=fsize)


        # Set the legends
        qiqc_lbls, qiqc_hdls = ax_qiqc.get_legend_handles_labels()
        ax_qiqc.legend(qiqc_lbls, qiqc_hdls, fontsize=fsize)
        d_lbls, d_hdls = ax_d.get_legend_handles_labels()
        ax_d.legend(d_lbls, d_hdls, loc='upper right', fontsize=fsize)
            
        fc = instr_config["fc"]
        fc_str = f"{fc:1.3f}".replace(".","p")
        fsuffix = f"_{fc_str}GHz_{temperature}mK_{self.dstr}.png"
        
        
        fig_fc_title = 'fc_vs_power'+fsuffix
        fig_qc_title = 'qc_vs_power'+fsuffix
        fig_qiqc_title = 'qiqc_vs_power'+fsuffix
        fig_qi_title = 'qi_vs_power'+fsuffix
        fig_d_title = 'tand_vs_power'+fsuffix
        
        fig_title_size = 24
        fig_fc.suptitle(fig_fc_title.replace(".png",""), fontsize=fig_title_size)
        fig_qc.suptitle(fig_qc_title.replace(".png",""), fontsize=fig_title_size)
        fig_qiqc.suptitle(fig_qiqc_title.replace(".png",""), fontsize=fig_title_size)
        fig_qi.suptitle(fig_qi_title.replace(".png",""), fontsize=fig_title_size)
        fig_d.suptitle(fig_d_title.replace(".png",""), fontsize=fig_title_size)
        
        for fig in [fig_fc, fig_qc, fig_qc, fig_qiqc, fig_qi, fig_d]:
            fig.tight_layout()
            
            
        # save all figures, remembering save_dir is a pathlib object
        all_figs = [fig_fc, fig_qc, fig_qiqc, fig_qi, fig_d]
        all_titles = [fig_fc_title, fig_qc_title, fig_qiqc_title, fig_qi_title, fig_d_title]
        for fig, fig_title in zip(all_figs, all_titles):
            save_loc = save_dir / fig_title
            fig_fc.savefig( save_loc.absolute(), format='png')
    
        if show_plots is False:
            plt.close('all')



    def fit_delta_tls(self, Qi, T, fc, Qc, p, 
                    display_scales={'QHP' : 1e5, 'nc' : 1e7, 'Fdtls' : 1e-6}, 
                    QHP_fix=False, Qierr=None):
        
        """
            Fit the loss using the expression

            delta_tls = F * delta0_tls * tanh(hbar w_c / 2 kB T) (1 + <n> / nc)^-1/2

        """
        
        # Convert the inputs to the correct format
        h      = 6.626069934e-34
        hbar   = 1.0545718e-34
        kB     = 1.3806485e-23
        fc_GHz = fc if np.any(fc >= 1e9) else fc * 1e9
        TK     = T if T <= 400e-3 else T * 1e-3
        delta  = 1. / Qi
        hw0    = hbar * 2 * np.pi * fc_GHz
        hf0    = h * fc_GHz
        kT     = kB * TK

        # Extract the scale factors for the fit parameters
        QHP_max   = display_scales['QHP']
        nc_max    = display_scales['nc']
        Fdtls_max = display_scales['Fdtls']

        navg = np.abs(self.power_to_navg(p, Qi, Qc, fc))
        labels = [r'$10^{%.2g}$' % x for x in np.log10(navg)]
        print(f'<n>: {labels}')
        print(f'T: {TK} K')
        print(f'fc_GHz: {fc_GHz} Hz')

        def fitfun4(n, Fdtls, nc, QHP, beta):
            num = Fdtls * np.tanh(hw0 / (2 * kT))
            den = (1. + n / nc)**beta
            return num / den + 1. / QHP

        if QHP_fix:
            QHPidx = np.argmax(Qi)
            QHP = Qi[QHPidx]
            QHP_err = Qierr[QHPidx]

        def fitfun3(n, Fdtls, nc, beta):
            num = Fdtls * np.tanh(hw0 / (2 * kT))
            den = (1. + n / nc)**beta
            return num / den + 1. / QHP

        def fitfun(n, Fdtls, nc, dHP):
            num = Fdtls * np.tanh(hw0 / (2 * kT))
            den = np.sqrt(1. + n / nc)
            return num / den + dHP

        # Fit with Levenberg-Marquardt
        # x0 = [1e-6, 1e6, 1e4]
        # popt, pcov = sp.optimize.curve_fit(fitfun, navg, delta, p0=x0)
        #     F*d^0TLS,  nc,    dHP,  beta
        x0 = [     1e-6,  1e2,  np.max(Qi), 0.1]
        # x0 = [     3e-6,  1.2e6,  19600, 0.17]
        bounds = ((1e-10, 1e1,   1e3, 1e-3),\
                (1e-3,  1e10,  1e8, 1.))
        if QHP_fix:
            x0 = [1e-6,  1e2, 0.1]
            # bnds = ((1e-10, 1e1, 1e-3), (1e-3, 1e7, 1.))
            popt, pcov = sp.optimize.curve_fit(fitfun3, navg, 
                    delta, p0=x0) # , bounds=bnds)
            Fdtls, nc, beta = popt
            errs = np.sqrt(np.diag(pcov))
            Fdtls_err, nc_err, beta_err = errs
        else:
            x0 = [     1e-6,  1e2,  np.max(Qi), 0.1]
            popt, pcov = sp.optimize.curve_fit(fitfun4, navg, delta, p0=x0)
            Fdtls, nc, QHP, beta = popt
            errs = np.sqrt(np.diag(pcov))
            Fdtls_err, nc_err, QHP_err, beta_err = errs


        # Uncertainty formatting
        ## Rounding to n significant figures
        round_sigfig = lambda x, n \
                : round(x, n - int(np.floor(np.log10(abs(x)))) - 1)

        Fdtls_err = round_sigfig(Fdtls_err, 1)
        nc_err    = round_sigfig(nc_err, 1)
        QHP_err   = round_sigfig(QHP_err, 1)
        beta_err  = round_sigfig(beta_err, 1)

        ## Uncertainty objects
        Fdtls_un = uncertainties.ufloat(Fdtls, Fdtls_err)
        nc_un    = uncertainties.ufloat(nc, nc_err)
        QHP_un   = uncertainties.ufloat(QHP, QHP_err)
        beta_un  = uncertainties.ufloat(beta, beta_err)

        print(f'QHP: {QHP:.2f}+/-{QHP_err:.2f}')

        # Build a string with the results
        ## Formatted uncertainties
        Fdtls_latex = f'{Fdtls_un:L}'
        nc_latex = f'{nc_un:L}'
        QHP_latex = f'{QHP_un:L}'
        beta_latex = f'{beta_un:L}'

        ## Latex strings
        Fdtls_str = r'$F\delta^{0}_{TLS}: %s$'  % Fdtls_latex
        nc_str    = r'$n_c: %s$' % nc_latex
        QHP_str   = r'$Q_{HP}: %s$' % QHP_latex
        beta_str   = r'$\beta: %s$' % beta_latex
        delta_fit_str = Fdtls_str + '\n' + nc_str \
                + '\n' + QHP_str + '\n' + beta_str
        # delta_fit_str = Fdtls_str + '\n' + nc_str
        print(delta_fit_str)

        if QHP_fix:
            return Fdtls, nc, QHP, Fdtls_err, nc_err, QHP_err, \
                fitfun3(navg, *popt), delta_fit_str
                # fitfun4(nout, Fdtls, nc, QHP, beta), delta_fit_str, pout
                # fitfun(navg, Fdtls, nc, dHP), delta_fit_str
        else:
            return Fdtls, nc, QHP, Fdtls_err, nc_err, QHP_err, \
                fitfun4(navg, *popt), delta_fit_str
                # fitfun4(nout, Fdtls, nc, QHP, beta), delta_fit_str, pout
                # fitfun(navg, Fdtls, nc, dHP), delta_fit_str

