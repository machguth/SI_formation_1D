""" *** 1D heat conduction model ***
Intended for use in snow and ice and with a
Calculates heat conduction and the amount of superimposed ice that forms

This file contains a variety of functions that area called by heat_flux_1D.py
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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

    colors_validation = ['tab:red', 'tab:cyan', 'tab:purple', 'tab:orange', 'tab:pink']
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


def test_T_plotting1(T_evol, phi, refreeze_c, refreeze_c_mmice, rho_evol, iw_evol, D_evol,
                    t, melt, days, iwc, dt, n, dx, output_dir):

    colors = plt.cm.brg(np.linspace(0, 1, n))
    layer_depths = np.arange(n) * dx + dx

    fig, ax = plt.subplots(5, figsize=(24, 35),
                           gridspec_kw={'height_ratios': [2, 0.25, 0.9, 0.9, 0.9]}, sharex=True)

    # Use 5-day intervals for the x-ticks
    ax[4].xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax[4].xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))

    for nvd, vd in enumerate(layer_depths):
        ax[0].plot(t[:-1], T_evol[1 + nvd,:-1],
                   color=colors[nvd], lw=1, label='{:.2f}'.format(layer_depths[nvd]))  # ls='--',

    ax[0].plot(t[:-1], T_evol[0,:-1], label='$T_{surface}$', color='gray', lw=2)
    ax[0].set_title('Layer snow temperatures $T$')
    ax[0].set_ylabel('$T$ (°C)')

    ax[1].set_ylabel('$M$ (mm w.e. day$^{-1}$)')
    ax[1].set_title('Surface melt $M$')
    ax[1].plot(t[:-1], melt[:len(t[:-1])] * (86400 / dt) * 1000, color='gray')  # conv. to mm day^-1

    ax[2].plot(t[:-1], phi[-1, :-1], color='Tab:blue')
    ax[2].tick_params(axis='y', color='Tab:blue', labelcolor='Tab:blue')
    ax[2].set_ylabel('$\\phi$ (W m$^{-2}$)', color='Tab:blue')
    ax[2].set_title('Bottom heat flux $\\phi$ and superimposed ice formation $SIF$ at snow-slush interface')
    ax2a = ax[2].twinx()
    ax2b = ax[2].twinx()
    ax2a.set_ylabel('SIF (mm w.e.)', color='Tab:orange')
    ax2a.tick_params(axis='y', color='Tab:orange', labelcolor='Tab:orange')
    ax2a.plot(t[:-1], refreeze_c[0, :-1], color='Tab:orange')
    ax2b.plot(t[:-1], refreeze_c_mmice[0, :-1], color='Tab:orange')
    ax2b.set_ylabel('SIF (mm ice)', color='Tab:orange')
    ax2b.tick_params(axis='y', color='Tab:orange', labelcolor='Tab:orange')
    # right, left, top, bottom
    ax2b.spines['right'].set_position(('outward', 140))

    for nvd, vd in enumerate(layer_depths):
        ax[3].plot(t[:-1], rho_evol[nvd, :-1], color=colors[nvd], lw=1)
    ax[3].set_ylabel('$\\rho$ (kg m$^{-3}$)')
    ax[3].set_title('Layer snow densities $\\rho$')

    for nvd, vd in enumerate(layer_depths):
        ax[4].plot(t[:-1], iw_evol[nvd, :-1], color=colors[nvd], lw=1)
    ax[4].tick_params('x', rotation=45)
    ax[4].set_xlabel('Date')
    ax[4].set_ylabel('$W_l$ (kg m$^{-3}$)')
    ax[4].set_title('Layer water content $W_l$ and cumulative discharge $\Sigma D$ at snowpack bottom')
    ax4 = ax[4].twinx()
    ax4.set_ylabel('$\Sigma D$ (mm)', color='Tab:orange')
    ax4.tick_params(axis='y', color='Tab:orange', labelcolor='Tab:orange')
    ax4.plot(t[:-1], D_evol[:-1] * 1000, color='Tab:orange')  # convert to mm

    ax[0].legend(bbox_to_anchor=(1.2, 1.0), title='Depth (m)')
    fig.tight_layout()

    plt.savefig(os.path.join(output_dir, 'test_plot_D{:.2f}'.format(layer_depths[-1]) + 'm_' + str(int(days)) + 'd_'
                             + str(int(dt)) + 's_iwc' + str(int(iwc)) + '.png'))


def test_detail_plotting(T_evol, phi, refreeze_c, refreeze_c_mmice, rho_evol, iw_evol, D_evol,
                    t, melt, days, iwc, dt, n, dx, output_dir):

    colors = plt.cm.brg(np.linspace(0, 1, n))
    layer_depths = np.arange(n) * dx + dx

    sel_d_idx = [0, 1, 2, 3]  # indices of layers whose behaviour should be analysed

    tr = [3980, 4100]

    fig, ax = plt.subplots(5, figsize=(24, 35),
                           gridspec_kw={'height_ratios': [2, 0.25, 0.9, 0.9, 0.9]}, sharex=True)

    # Use 5-day intervals for the x-ticks
    ax[4].xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax[4].xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y %H:%M'))

    for nvd, vd in enumerate(layer_depths[sel_d_idx]):
        ax[0].plot(t[tr[0]:tr[1]], T_evol[1 + sel_d_idx[nvd], tr[0]:tr[1]],
                   color=colors[sel_d_idx[nvd]], lw=1, label='{:.2f}'.format(layer_depths[sel_d_idx[nvd]]))  # ls='--',

    ax[0].plot(t[tr[0]:tr[1]], T_evol[0,tr[0]:tr[1]], label='$T_{surface}$', color='gray', lw=2)
    ax[0].set_title('Layer snow temperatures $T$')
    ax[0].set_ylabel('$T$ (°C)')

    ax[1].set_ylabel('$M$ (mm w.e. day$^{-1}$)')
    ax[1].set_title('Surface melt $M$')
    ax[1].plot(t[tr[0]:tr[1]], melt[tr[0]:tr[1]] * (86400 / dt) * 1000, color='gray')  # conv. to mm day^-1

    for nvd, vd in enumerate(layer_depths[sel_d_idx]):
        ax[2].plot(t[tr[0]:tr[1]], phi[1 + sel_d_idx[nvd],tr[0]:tr[1]], color=colors[sel_d_idx[nvd]])
    ax[2].plot([t[tr[0]], t[tr[1]]], [0, 0], ls=':', color='gray')
    ax[2].tick_params(axis='y', color='Tab:blue', labelcolor='Tab:blue')
    ax[2].set_ylabel('$\\phi$ (W m$^{-2}$)', color='Tab:blue')
    ax[2].set_title('Heat flux $\\phi$')

    for nvd, vd in enumerate(layer_depths[sel_d_idx]):
        ax[3].plot(t[tr[0]:tr[1]], rho_evol[sel_d_idx[nvd], tr[0]:tr[1]], color=colors[sel_d_idx[nvd]], lw=1)
    ax[3].set_ylabel('$\\rho$ (kg m$^{-3}$)')
    ax[3].set_title('Layer snow densities $\\rho$')

    for nvd, vd in enumerate(layer_depths[sel_d_idx]):
        ax[4].plot(t[tr[0]:tr[1]], iw_evol[sel_d_idx[nvd], tr[0]:tr[1]], color=colors[sel_d_idx[nvd]], lw=1)
    ax[4].tick_params('x', rotation=45)
    ax[4].set_xlabel('Date')
    ax[4].set_ylabel('$W_l$ (kg m$^{-3}$)')
    ax[4].set_title('Layer water content $W_l$ and cumulative discharge $\Sigma D$ at snowpack bottom')
    ax4 = ax[4].twinx()
    ax4.set_ylabel('$\Sigma D$ (mm)', color='Tab:orange')
    ax4.tick_params(axis='y', color='Tab:orange', labelcolor='Tab:orange')
    ax4.plot(t[tr[0]:tr[1]], D_evol[tr[0]:tr[1]] * 1000, color='Tab:orange')  # convert to mm

    ax[0].legend(bbox_to_anchor=(1.2, 1.0), title='Depth (m)')
    fig.tight_layout()

    plt.savefig(os.path.join(output_dir, 'test_details_D{:.2f}'.format(layer_depths[-1]) + 'm_' + str(int(days)) + 'd_'
                             + str(int(dt)) + 's_iwc' + str(int(iwc)) + '.png'))
