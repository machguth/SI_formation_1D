""" *** 1D heat conduction model ***
Intended for use in snow and ice
Calculates heat conduction and the amount of superimposed ice that forms

Author: Horst Machguth horst.machguth@unifr.ch
        Andrew Tedstone andrew.tedstone@unil.ch
"""


import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import xarray as xr
from scipy import interpolate

import pdb

# CONSTANTS
Cp = 2090    # [J kg-1 K-1] Specific heat capacity of ice
L = 334000   # [J kg-1] Latent heat of water

def tsurf_sine(days, t_final, dt, years, Tmean, Tamplitude):
    if days/365 != years:
        print('\n *** For use of sine curve multi-annual air temperatures, days needs to be set to 365 * years ***')
        sys.exit(0)
    x = np.linspace(0, 2 * np.pi * years, int(t_final / dt))
    T_surf = Tmean + np.sin(x) * Tamplitude
    return T_surf


def irrw(iwc, n, dx, rho, T0):
    iw = np.ones(n) * dx * 1000 * rho / 1000 * iwc / 100
    if isinstance(T0, (int, float)):
        if (T0 < 0) & (iwc > 0):
            print('\n *** Warning: irreducible water cont. > 0 for negative T0. Setting irred. water content to 0. *** \n')
            iw *= 0
            iwc = 0
    elif isinstance(T0, np.ndarray):
        iw *= T0[1:-1] > 0
        # Does iwc also need changing?
    else:
        raise ValueError('Unknown T0 variable type.')

    return iw, iwc


def alpha_update(k, rho, Cp, n, iw):
    alpha = np.ones(n) * (k / (rho * Cp)) * (iw == 0)
    return alpha


def scale_initial_profile_10m(y, t_profile, t_profile_10m, t_scale_10m):
    """ Given an initial temperature profile and a 10 m temperature, rescale the T profile accordingly. 

    :param y: np.array of depth intervals over which the calculations are run
    :param t_profile: pd.Series of temperatures, index of depth (metres)
    :param t_10m: float of 10 metre depth temperature

    :returns: pd.Series adjusted temperatures
    """

    # prepare vector T_start of initial ice slab and firn temperatures
    function1d = interpolate.interp1d(t_profile.index, t_profile, fill_value='extrapolate')

    T_start = function1d(y).astype(float)
    T_start[T_start > 0] = 0  # make sure no positive values in initial temperatures
    T_start[0] = 0  # make sure top grid cell is at 0 °C

    T = T_start  # finally set the temperature distribution to T_start

    # multiply starting temperature with ratio between 10 m firn temperature in the initial T profile
    # and the 10 m firn T in the Grid cell that is simulated (T in grid cell according to Vandecrux et al., 2023)
    T *= t_scale_10m/t_profile_10m

    return T


def calc_flux(t, n, T, dTdt, alpha, dx, Tsurf, dt, T_evol, phi, k, refreeze, L, 
    iw, rho, Cp, bottom_boundary):
    """
    Run the heat flux calculations over the vertical domain for the provided timesteps.

    :param t: np.array, timestamps
    :param n:
    :param T:
    :param dTdt:
    :param alpha:
    :param dx: float, Desired layer thickness [m] 
    :param Tsurf: float, (Surface) temperature at top boundary [degC]
    :param dt: int, Timestep [s]
    :param T_evol:
    :param phi:
    :param k: float, Thermal conductivity of ice or snow:
    :param refreeze:
    :param L:
    :param iw: float, Irreducible water content [% of mass] ??
    :param rho: float, Density of snow or ice [kg m-3]
    :param Cp:
    :param bottom_boundary: 'open' or 'closed'

    :returns: (T_evol, phi, refreeze, iw)
    
    """

    for j in range(0, len(t)-1):
        T[0] = Tsurf[j]  # Update temperature top layer according to temperature evolution (if one is prescribed)
        
        if bottom_boundary == 'open':
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

    return T_evol, phi, refreeze, iw


def depth_intervals(D, dx, n=None):
    if n is None:
        n = int(np.round(D / dx))
    return np.linspace(-dx/2, D+dx/2, n+2)  


def run_calculation(D, dx, dt, t, T0, rho, k, iwc, por, Tbottom, Tsurf, slushatbottom,
    y=None):
    """
    Convenience function to run calculations through time.
    
    :param D: float, Thickness of snow pack or ice slab [m]
    :param dx: float, Desired layer thickness [m] 
    :param dt: int, Timestep [s]
    :param t: np.array, timestamps
    :param T0: np.array or float, initial temperature of all layers [degC]
    :param rho: float, Density of snow or ice [kg m-3]
    :param k: float, Thermal conductivity of ice or snow:
    :param iwc: float, Irreducible water content [% of mass]
    :param por: float, Porosity of snow where it is water-saturated []
    :param Tbottom: float or None, Bottom boundary condition, [None or degC]
    :param Tsurf: float, (Surface) temperature at top boundary [degC]
    :param slushatbottom: bool, if True, refreezing at bottom boundary; if False, refreezing at top [bool]
    :param y: None or np.array. If provided, an array of depth intervals to run calculations over. Then D, dx are ignored.

    :returns: (t, y, T_evol, refreeze, phi)
        y : depth intervals
        T_evol : temperatures
        refreeze : refreezing at bottom (0) and top (1)
        refreeze_c : cumulative refreezing at bottom (0) and top (1)
        phi : heat flux

    Usage notes:

        Tsurf can be a float or an array of length equal to ts.
    """

    if y is None:
        # vector of central points of each depth interval (=layer)
        y = depth_intervals(D, dx)
    else:
        assert isinstance(y, np.ndarray)
        dx = y[1] - y[0]

    n = len(y) - 2

    # Is Tsurf a single value or a time series?
    if isinstance(Tsurf, (float, int)):
        # If single value of Tsurf provided then extend it over the run duration
        Tsurf = np.ones(len(t)) * Tsurf

    if Tbottom is None:
        bottom_boundary = 'open'
        # Extend the Temperature array using T0
        if isinstance(T0, np.ndarray):
            Tbottom = T0[-1]
        else:
            Tbottom = T0
    else:
        bottom_boundary = 'closed'

    # Set up T and T_evol according to the inputs provided    
    if isinstance(T0, (float, int)):
        # If single initial temperature provided then extend it over depth (and time)

        # Vertical temperature profile at a given timestamp, with specified Tbottom
        T = np.append(np.insert(np.ones(n) * T0, 0, 0), Tbottom)
        T_evol = np.ones([n+2, len(t)]) * T0  # array of temperatures for each layer and time step
    else:
        T = T0
        # array of temperatures for each layer and time step
        T_evol = np.repeat([T], len(t), axis=0).T #np.ones([n+2, len(t)]) * T0  

    # derivative of temperature at each node
    dTdt = np.empty(n)  

    # array of the heat flux per time step, for each layer and time step
    phi = np.empty([n+1, len(t)])  
    
    refreeze = np.empty([2, len(t)])
    
    # Water per layer (irreducible water content) [mm w.e. m-2 or kg m-2]
    # This function also sets irreducible water content to 0 for all layers that have initial T < 0
    iw, iwc = irrw(iwc, n, dx, rho, T0)
    
    # Vector of thermal diffusivity [m2 s-1]
    alpha = alpha_update(k, rho, Cp, n, iw)
    
    # calculation of temperature profile over time
    T_evol, phi, refreeze, iw = calc_flux(t, n, T, dTdt, alpha, dx, Tsurf, dt, T_evol,
                                             phi, k, refreeze, L, iw, rho, Cp, bottom_boundary)
    
    # cumulative sum of refrozen water
    refreeze_c = np.cumsum(refreeze, axis=1)
    # and correct for the fact that water occupies only the pore space
    refreeze_c /= por
    
    return (y, T_evol, refreeze, refreeze_c, phi)


def temperatures_to_da(y, t, T_evol, save_path=None):
    # Xarray DataArray of all simulated temperatures
    da_to = xr.DataArray(
        data=T_evol.transpose(),
        dims=['time', 'z'],
        coords=dict(
            z=y,
            time=t
        ),
        attrs=dict(description="Simulated firn temperatures.", units='degree_Celsius'),
    )
    da_to.name = 'T'
    da_to = da_to.resample(time='1D').mean()
    da_to = da_to.coarsen(z=2, boundary='trim').mean()

    if save_path is not None:
        da_to.to_netcdf(save_path)

    return da_to


def refreeze_to_da(t, refreeze, slushatbottom, save_path=None):
    # Xarray DataArray of all simulated daily refreezing rates
    if slushatbottom:
        d = refreeze[0,:]
        w = 'bottom'
    else:
        d = refreeze[1,:]
        w = 'top'
    da_ro = xr.DataArray(
        data=d,
        dims=['time'],
        coords=dict(
            time=t
        ),
        attrs=dict(description="Simulated refreezing rates at the %s of the modelling domain." %w,
                   units='mm w.e. per time step',
                   long_name='Refreezing R refers to water that refreezes, does not include surrounding matrix'),
    )
    da_ro.name = 'R'
    da_ro = da_ro.resample(time='1D').sum()

    if save_path is not None:
        da_ro.to_netcdf(save_path)

    return da_ro


def plotting(T_evol, dt_plot, dt, y, D, slushatbottom, phi, days,
             t_final, t, refreeze_c, output_dir, m, iwc, save_to=None):

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


    if save_to is not None:
        plt.savefig(save_to)
    
    return
    

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
