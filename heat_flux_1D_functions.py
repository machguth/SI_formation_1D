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

# def irwc_old(iwc, n, dx, rho, T0):
#     iw = np.ones(n) * dx * 1000 * rho / 1000 * iwc / 100
#     if (T0 < 0) & (iwc > 0):
#         print('\n *** Warning: irreducible water cont. > 0 for negative T0. Setting irred. water content to 0. *** \n')
#         iw *= 0
#         iwc = 0
#     return iw, iwc

def irwc_init(iwc, irwc_max, dx, n, T0):
    iw = irwc_max * 1000 * dx
    if (T0 < 0) & (iwc > 0):
        print('\n *** Warning: irreducible water cont. > 0 for negative T0. Setting irred. water content to 0. *** \n')
        iw *= 0
        # iwc = 0
    return iw

def alpha_update(k, rho, Cp, n, iw):
    alpha = np.ones(n) * (k / (rho * Cp)) * (iw == 0)
    return alpha

# Calculate thermal conductivity of snow to ice based on Calonne et al. (2019), GRL
def k_update(T_evol, rho, a, rho_tr, k_ref_i, k_ref_a):
    T_evol += 273.15  # convert to K
    k_air = 1.5207E-11 * T_evol**3 - 4.8574E-08 * T_evol**2 + 1.0184E-04 * T_evol - 3.9333E-04
    k_ice = 9.828 * np.exp(-5.7 * 10**(-3) * T_evol)  # Based on Cuffex and Paterson (2010)

    k_ref_firn = 2.107 + 0.003618 * (rho - 917)
    k_ref_snow = 0.024 - 1.23 * 10**(-4) * rho + 2.5 * 10**(-6) * rho**2

    theta = 1 / (1 + np.exp((-2) * a * (rho - rho_tr)))
    # the below equation could be updated to also include the effect of water in the snow pack on k.
    # water is always at 0 °C and its effect could be added quite simply
    k = (1 - theta) * k_ice * k_air / (k_ref_i * k_ref_a) * k_ref_snow + \
        theta * k_ice / k_ref_i * k_ref_firn
    return k

def rho_por_irwc_max(rho, iwc):  # calculate the maximum amount of irreducible water content (IRWC) per layer
    # as function of rho using a fixed percentage of irreducible water content, expressed in % of the pore volume
    # This is based on Coléou and Lesaffre (1998), Annals of Glaciology, and Colbeck (1974), J. Glaciol.
    # While the previous study states that there is a weak dependency between porosity and IRWC expressed as % of
    # pore volume (IRWC% = -0.0508porosity + 0.0947; own calculation based on the data in Coléou and Lesaffre, 1998),
    # their data cover only a limited range of densities and hence here we use a simple fixed percentage (Colbeck 1974).
    porosity = 1 - rho / 917
    irwc_max = porosity * (iwc / 100)  # max potential irreducible water content
    # for rho > 873 kg m-3, no more pore space can be used. This addresses pore close-off density which is rather at
    # 830 kg m-3, but instead using 873 kg m-3 bcs. of this value corresponding to infiltration ice density as
    # measured by Machguth et al. (2016)
    irwc_max *= porosity > (1 - 873 / 917)  # irwc_max as fraction of 1 (where 1 represents total volume of layer)
    return porosity, irwc_max

def bucket_scheme(L, Cp, melt, iw, irwc_max, T_evol, rho, dx, j):
    # first calculate where irwc can be added
    irwc_existing = iw / 1000 / dx
    irwc_available = irwc_max - irwc_existing  # unit fraction of 1 (1 being total thickness dx of a layer)
    irwc_available *= (irwc_available > 0)  # to be sure available IRWC is nowhere below zero

    melt_f = melt / dx  # convert melt to a fraction of dx

    # calculate irreducible water distribution in all layers without a loop
    irwc_cs = np.cumsum(irwc_available)
    irwc_cs_m = irwc_cs - melt_f[j]
    irwc_cs_m_pos = irwc_cs_m * (irwc_cs_m > 0)
    irwc_added = irwc_available - irwc_cs_m_pos
    irwc_added = irwc_added * (irwc_added > 0)
    irwc_existing = irwc_added + irwc_existing
    bottom_water = np.sum(irwc_added) - melt_f[j]
    bottom_water *= (bottom_water > 0)

    # calculate the amount of refreezing
    Lh_release_layer = irwc_existing * 1000 * dx * L
    heat_capacity_layer = 1 * rho * dx * T_evol * Cp * (-1)  # 1 to represent the full layer, -1 bcs. T_evol negative
    # make sure layer is not warmed beyond 0 °C (in case Lh_release_layer > heat_capacity_layer)
    Lh_release = Lh_release_layer * (Lh_release_layer < heat_capacity_layer) + \
                 (Lh_release_layer - heat_capacity_layer) * (Lh_release_layer > heat_capacity_layer)
    refreezing = Lh_release / (1000 * dx * L)

    # calculate warming from latent heat release and adjust T_evol
    T_evol += Lh_release / (rho * dx * Cp)

    # calculate new IRWC after refreezing took place
    iw = irwc_existing - refreezing

    # calculate new rho after refreezing took place
    # pay attention that the refrozen water becomes ice and its volume grows by rho_water / rho_ice
    rho += 917 * refreezing * (1000 / 917)

    return iw, T_evol, rho, bottom_water

def calc_closed(t, n, T, dTdt, alpha, dx, Tsurf, dt, T_evol, phi, k, refreeze, L, iw, iwc, rho, Cp, melt,
                a, rho_tr, k_ref_i, k_ref_a):

    for j in range(0, len(t)-1):
        # calculation is for n + 2 layers. The n layers all have a thickness of D/n. The additional two layers are
        # the skin layer on top and the water saturated layer at the bottom
        porosity, irwc_max = rho_por_irwc_max(rho, iwc)
        T[0] = Tsurf[j]  # Update temperature top layer according to temperature evolution (if one is prescribed)

        dTdt[:] = alpha * (-(T[1:-1] - T[0:-2]) / dx ** 2 + (T[2:] - T[1:-1]) / dx ** 2)

        T[1:-1] = T[1:-1] + dTdt * dt
        T_evol[:, j] = T

        # To calculate heat transfer to the water layer at the bottom (not through the water),
        # an n+1 value of thermal conductivity is needed. For the moment, simply duplicate the lowermost k value
        phi[:, j] = np.append(k, k[-1]) * (T[:-1] - T[1:]) / dx

        # calculate the refreezing of irreducible water as an effect of heat conduction
        iw -= (-1) * phi[:-1, j] * dt / L * (phi[:-1, j] <= 0)  # *(phi[:-1, j] <= 0) otherwise iw created if phi > 0
        iw *= iw > 0  # check there is no negative iw

        # calculate superimposed ice formation at the bottom of the domain
        refreeze[0, j] = (-1) * phi[-1, j] * dt / L  # [mm] refrozen water mm (w.e.) per time step, at bottom of domain
        refreeze[1, j] = phi[0, j] * dt / L  # [mm] refrozen water mm (w.e.) per time step, at top of domain

        # calculate percolation as per bucket scheme
        # also update T_evol for warming from latent heat release where water percolates into layers with T_evol < 0 °C
        iw, T_evol[1:-1, j], rho, bottom_water = bucket_scheme(L, Cp, melt, iw, irwc_max, T_evol[1:-1, j], rho, dx, j)

        # update k and alpha for the next iteration
        k = k_update(T_evol[1:-1, 0], rho, a, rho_tr, k_ref_i, k_ref_a)
        alpha = alpha_update(k, rho, Cp, n, iw)

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
