""" *** 1D heat conduction model ***
Intended for use in snow and ice and with a
Calculates heat conduction and the amount of superimposed ice that forms

This file contains a variety of functions that area called by heat_flux_1D.py
"""


import numpy as np
import matplotlib.pyplot as plt
import os


def plotting(T_evol, dt_plot, dt, y, D, slushatbottom, phi, days,
             t_final, t, refreeze_c, output_dir, m, iwc):

    plt.rcParams.update({'font.size': 28})
    fig, ax = plt.subplots(2, figsize=(24, 20), gridspec_kw={'height_ratios': [3, 1]})
    t_sel = np.arange(0, len(T_evol[0, :]), dt_plot/dt)
    n_t_sel = len(t_sel)
    colors = plt.cm.brg(np.linspace(0, 1, n_t_sel))

    for ni, i in enumerate(t_sel):
        day = int(np.floor(i * dt / 86400))
        ax[0].plot(T_evol[:, int(i)], y, color=colors[ni])
        # ax[0].plot([Tsurf[int(i)], T_evol[0, int(i)]], [0-dx/2, y[0]], color=colors[ni])
        # if bottom_boundary:
        #     ax[0].plot([T_evol[-1, int(i)], Tbottom], [y[-1], D+dx/2], color=colors[ni])
        ax[0].axhline(0, color='gray', ls=':')
        ax[0].axhline(D, color='gray', ls=':')
    ax[0].invert_yaxis()
    ax[0].set_xlabel('Temperature (°C)')
    ax[0].set_ylabel('Depth (m)')
    ax[0].set_ylim(ax[0].get_ylim()[0], ax[0].get_ylim()[1])
    if slushatbottom:
        ax[0].set_title('Temperature with snow depth')
        ax[0].axhspan(D, ax[0].get_ylim()[0], color='skyblue')
        ax[0].text(ax[0].get_xlim()[0] + 0.4, ax[0].get_ylim()[0] - 0.02, 'slush',
                   color='white', fontsize=40, fontweight='bold')
    else:
        ax[0].set_title('Temperature with ice depth')
        ax[0].axhspan(ax[0].get_ylim()[1], 0, color='skyblue')
        ax[0].text(ax[0].get_xlim()[0] + 0.4, 0 - 0.1, 'slush',
                   color='white', fontsize=40, fontweight='bold')

    if slushatbottom:
        ax[1].plot(t[:-1], phi[-1, :-1], color='Tab:blue')
        ax[1].set_title('Heat flux and superimposed ice formation at snow-slush interface')
    else:
        ax[1].plot(t[:-1], phi[0, :-1], color='Tab:blue')
        ax[1].set_title('Heat flux and superimposed ice formation at slush-ice interface')
    ax[1].set_xlabel('Days')
    steps = np.round(days/12)
    ax[1].set_xticks(np.arange(0, t_final + 1, 86400 * steps),
                     (np.arange(0, t_final + 1, 86400 * steps) / 86400).astype(int))
    ax[1].tick_params(axis='y', color='Tab:blue', labelcolor='Tab:blue')
    ax[1].set_ylabel('Heat flux (W m$^{-2}$)', color='Tab:blue')

    ax2 = ax[1].twinx()
    ax2.set_ylabel('S-imposed ice (mm)', color='Tab:orange')
    ax2.tick_params(axis='y', color='Tab:orange', labelcolor='Tab:orange')
    if slushatbottom:
        ax2.plot(t[:-1], refreeze_c[0, :-1], color='Tab:orange')
    else:
        ax2.plot(t[:-1], refreeze_c[1, :-1], color='Tab:orange')

    plt.tight_layout()

    if slushatbottom:
        direction = 'bottom-SI'
    else:
        direction = 'top-SI'


    if m == 1:
        plt.savefig(os.path.join(output_dir, '1D_heat_flux_' + str(int(days)) + 'd_'
                                 + str(int(dt)) + 's_iwc' + str(int(iwc)) + '_' +
                                 direction + '.png'))
    else:
        plt.savefig(os.path.join(output_dir, '1D_heat_flux_' + str(int(days)) + 'd_'
                                 + str(int(dt)) + 's_iwc' + str(int(iwc)) + '_' +
                                 direction + 'Tmultiplied_by_{:.1f}'.format(m) + '.png'))


def plotting_incl_measurements(T_evol, dt_plot, dt, y, D, slushatbottom, phi, days,
                               t_final, t, refreeze_c, output_dir, iwc, da, m, validation_dates):

    colors_validation = ['tab:red', 'tab:cyan', 'tab:purple', 'tab:orange', 'tab_pink']
    plt.rcParams.update({'font.size': 28})
    fig, ax = plt.subplots(2, figsize=(24, 20), gridspec_kw={'height_ratios': [3, 1]})
    t_sel = np.arange(0, len(T_evol[0, :]), dt_plot/dt)
    n_t_sel = len(t_sel)
    colors = plt.cm.brg(np.linspace(0, 1, n_t_sel))

    for ni, i in enumerate(t_sel):
        day = int(np.floor(i * dt / 86400))
        ax[0].plot(T_evol[:, int(i)], y, color=colors[ni])
        # ax[0].plot([Tsurf[int(i)], T_evol[0, int(i)]], [0-dx/2, y[0]], color=colors[ni])
        # if bottom_boundary:
        #     ax[0].plot([T_evol[-1, int(i)], Tbottom], [y[-1], D+dx/2], color=colors[ni])
        ax[0].axhline(0, color='gray', ls=':')
        ax[0].axhline(D, color='gray', ls=':')
    for nvd, vd in enumerate(validation_dates):
        ax[0].plot(da.sel(time=vd, method='nearest').values, da.z.values,
                   color=colors_validation[nvd], lw=3, ls='--', label=vd)
    ax[0].invert_yaxis()
    ax[0].set_xlabel('Temperature (°C)')
    ax[0].set_ylabel('Depth (m)')
    ax[0].set_ylim(ax[0].get_ylim()[0], ax[0].get_ylim()[1])
    ax[0].set_xlim(ax[0].get_xlim()[0], 0.5)
    ax[0].legend()
    if slushatbottom:
        ax[0].set_title('Temperature with snow depth')
        ax[0].axhspan(D, ax[0].get_ylim()[0], color='skyblue')
        ax[0].text(ax[0].get_xlim()[0] + 0.4, ax[0].get_ylim()[0] - 0.02, 'slush',
                   color='white', fontsize=40, fontweight='bold')
    else:
        ax[0].set_title('Temperature with ice depth')
        ax[0].axhspan(ax[0].get_ylim()[1], 0, color='skyblue')
        ax[0].text(ax[0].get_xlim()[0] + 0.4, 0 - 0.1, 'slush',
                   color='white', fontsize=40, fontweight='bold')

    if slushatbottom:
        ax[1].plot(t[:-1], phi[-1, :-1], color='Tab:blue')
        ax[1].set_title('Heat flux and superimposed ice formation at snow-slush interface')
    else:
        ax[1].plot(t[:-1], phi[0, :-1], color='Tab:blue')
        ax[1].set_title('Heat flux and superimposed ice formation at slush-ice interface')
    ax[1].set_xlabel('Date')
    # steps = np.round(days/12)
    # ax[1].set_xticks(np.arange(0, t_final + 1, 86400 * steps),
    #                  (np.arange(0, t_final + 1, 86400 * steps) / 86400).astype(int))
    ax[1].tick_params(axis='y', color='Tab:blue', labelcolor='Tab:blue')
    ax[1].set_ylabel('Heat flux (W m$^{-2}$)', color='Tab:blue')

    ax2 = ax[1].twinx()
    ax2.set_ylabel('S-imposed ice (mm)', color='Tab:orange')
    ax2.tick_params(axis='y', color='Tab:orange', labelcolor='Tab:orange')
    if slushatbottom:
        ax2.plot(t[:-1], refreeze_c[0, :-1], color='Tab:orange')
    else:
        ax2.plot(t[:-1], refreeze_c[1, :-1], color='Tab:orange')

    plt.tight_layout()

    if slushatbottom:
        direction = 'bottom-SI'
    else:
        direction = 'top-SI'

    if m == 1:
        plt.savefig(os.path.join(output_dir, 'test_1D_heat_flux_' + str(int(days)) + 'd_'
                                 + str(int(dt)) + 's_iwc' + str(int(iwc)) + '_' + direction + '_comp_meas' +'.png'))
    else:
        plt.savefig(os.path.join(output_dir, 'test_1D_heat_flux_' + str(int(days)) + 'd_'
                                 + str(int(dt)) + 's_iwc' + str(int(iwc)) + '_' + direction + '_comp_meas_' +
                                 'Tmultiplied_by_{:.1f}'.format(m) + '.png'))
