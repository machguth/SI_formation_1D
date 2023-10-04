""" *** 1D heat conduction model ***
Intended for use in snow and ice and with a
Calculates heat conduction and the amount of superimposed ice that forms

This file contains a variety of functions that area called by heat_flux_1D.py
"""


import numpy as np
import matplotlib.pyplot as plt
import os
import sys


def tsurf_sine(days, t_final, dt, years, Tmean, Tamplitude):
    if days/365 != years:
        print('\n *** For use of sine curve multi-annual air temperatures, days needs to be set to 365 * years ***')
        sys.exit(0)
    x = np.linspace(0, 2 * np.pi * years, int(t_final / dt))
    T_surf = Tmean + np.sin(x) * Tamplitude
    return T_surf


def irrw(iwc, n, dx, rho, T0):
    iw = np.ones(n) * dx * 1000 * rho / 1000 * iwc / 100
    if (T0 < 0) & (iwc > 0):
        print('\n *** Warning: irreduible water content > 0 for negative T0. Setting irred. water content to 0. *** \n')
        iw *= 0
        iwc = 0
    return iw, iwc


def alpha_update(k, rho, Cp, n, iw):
    alpha = np.ones(n) * (k / (rho * Cp)) * (iw == 0)
    return alpha


def calc_closed(t, n, T, dTdt, alpha, dx, Tsurf, dt, T_evol, phi, k, refreeze, L, iw, rho, Cp):

    for j in range(0, len(t)-1):
        T[0] = Tsurf[j]  # Update temperature top layer according to temperature evolution (if one is prescribed)

        dTdt[:] = alpha * (-(T[1:-1] - T[0:-2]) / dx ** 2 + (T[2:] - T[1:-1]) / dx ** 2)

        T[1:-1] = T[1:-1] + dTdt * dt
        T_evol[:, j] = T
        phi[:, j] = k * (T[:-1] - T[1:]) / dx
        iw -= (-1) * phi[:-1, j] * dt / L * (phi[:-1, j] <= 0)  # *(phi[:-1, j] <= 0) otherwise iw created if phi > 0
        iw *= iw > 0  # check there is no negative iw
        alpha = alpha_update(k, rho, Cp, n, iw)
        refreeze[0, j] = (-1) * phi[-1, j] * dt / L  # [mm] refrozen water mm (w.e.) per time step, at bottom of domain
        refreeze[1, j] = phi[0, j] * dt / L  # [mm] refrozen water mm (w.e.) per time step, at top of domain

    return T_evol, phi, refreeze, iw


def calc_open(t, n, T, dTdt, alpha, dx, Tsurf, dt, T_evol, phi, k, refreeze, L, iw, rho, Cp):

    for j in range(0, len(t) - 1):
        T[0] = Tsurf[j]  # Update temperature top layer according to temperature evolution (if one is prescribed)
        T[-1] = T[-2]  # Update bottom temperature to equal the second-lowest grid cell

        dTdt[:] = alpha * (-(T[1:-1] - T[0:-2]) / dx ** 2 + (T[2:] - T[1:-1]) / dx ** 2)

        T[1:-1] = T[1:-1] + dTdt * dt
        T_evol[:, j] = T
        phi[:, j] = k * (T[:-1] - T[1:]) / dx
        iw -= (-1) * phi[:-1, j] * dt / L * (phi[:-1, j] <= 0)  # *(phi[:-1, j] <= 0) otherwise iw created if phi > 0
        iw *= iw > 0  # check there is no negative iw
        alpha = alpha_update(k, rho, Cp, n, iw)
        refreeze[0, j] = (-1) * phi[-1, j] * dt / L  # [mm] refrozen water mm (w.e.) per time step, at bottom of domain
        refreeze[1, j] = phi[0, j] * dt / L  # [mm] refrozen water mm (w.e.) per time step, at top of domain

    return T_evol, phi, refreeze


def plotting(T_evol, dt_plot, dt, y, D, slushatbottom, phi, days,
             t_final, t, refreeze_c, output_dir, iwc):

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

    plt.savefig(os.path.join(output_dir, '1D_heat_flux_' + str(int(days)) + 'd_'
                             + str(int(dt)) + 's_iwc' + str(int(iwc)) + '_' + direction + '.png'))


def plotting_incl_measurements(T_evol, dt_plot, dt, y, D, slushatbottom, phi, days,
                               t_final, t, refreeze_c, output_dir, iwc, da, validation_dates):

    plt.rcParams.update({'font.size': 28})
    fig, ax = plt.subplots(2, figsize=(24, 20), gridspec_kw={'height_ratios': [3, 1]})
    t_sel = np.arange(0, len(T_evol[0, :]), dt_plot/dt)
    n_t_sel = len(t_sel)
    colors = plt.cm.brg(np.linspace(0, 1, n_t_sel))

    for ni, i in enumerate(t_sel):
        day = int(np.floor(i * dt / 86400))
        ax[0].plot(T_evol[:, int(i)], y, color=colors[ni])
        for vd in validation_dates:
            ax[0].plot(da.z.values, da.sel(time=vd).values, color='tab:orange')
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

    plt.savefig(os.path.join(output_dir, '1D_heat_flux_' + str(int(days)) + 'd_'
                             + str(int(dt)) + 's_iwc' + str(int(iwc)) + '_' + direction + '_comp_meas.png'))

