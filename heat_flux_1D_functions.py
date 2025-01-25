""" *** 1D heat conduction model ***
Intended for use in snow and ice and with a
Calculates heat conduction and the amount of superimposed ice that forms

This file contains a variety of functions that area called by heat_flux_1D.py
"""


import numpy as np
import sys


def C_to_K(T):
    T += 273.15
    return T


def K_to_C(T):
    T -= 273.15
    return T


def tsurf_sine(days, t_final, dt, years, Tmean, Tamplitude):
    if days/365 != years:
        print('\n *** For use of sine curve multi-annual air temperatures, days needs to be set to 365 * years ***')
        sys.exit(0)
    x = np.linspace(0, 2 * np.pi * years, int(t_final / dt))
    T_surf = Tmean + np.sin(x) * Tamplitude
    return T_surf


def irwc_init(iwc, irwc_max, dx, n, T0):
    iw = irwc_max * 1000 * dx
    if (T0 < 0) & (iwc > 0):
        # print('\n *** Warning: irreducible water cont. > 0 for negative T0. Setting irred. water cont. to 0. *** \n')
        iw *= 0
        # iwc = 0
    return iw


def alpha_update(k, rho, Cp, n, iw):
    alpha = np.ones(n) * (k / (rho * Cp)) * (iw == 0)
    return alpha


# Calculate thermal conductivity of snow to ice based on Calonne et al. (2019), GRL
def k_update(T, rho, a, rho_tr, k_ref_i, k_ref_a):

    k_air = 1.5207E-11 * T**3 - 4.8574E-08 * T**2 + 1.0184E-04 * T - 3.9333E-04
    k_ice = 9.828 * np.exp(-5.7 * 10**(-3) * T)  # Based on Cuffex and Paterson (2010)

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


def bucket_scheme(L, Cp, melt, iw, irwc_max, T_in, rho, dx, j):

    # convert iw to unit fraction of 1 (1 being total thickness dx of a layer)
    irwc_existing = iw / 1000 / dx

    # calculate where irwc can be added
    irwc_available = irwc_max - irwc_existing  # [unit fraction of 1 (1 being total thickness dx of a layer)]
    irwc_available *= (irwc_available > 0)  # to be sure available IRWC is nowhere below zero

    melt_f = melt[j] / dx  # convert melt to a fraction of dx

    # calculate irreducible water distribution in all layers without a loop
    irwc_cs = np.cumsum(irwc_available)
    irwc_cs_m = irwc_cs - melt_f
    irwc_cs_m_pos = irwc_cs_m * (irwc_cs_m > 0)
    irwc_added = irwc_available - irwc_cs_m_pos
    irwc_added = irwc_added * (irwc_added > 0)
    irwc_existing = irwc_added + irwc_existing
    bottom_water = np.sum(irwc_added) - melt_f
    bottom_water *= (bottom_water > 0)

    # calculate the amount of refreezing
    pot_Lh_rel_layer = irwc_existing * 1000 * dx * L
    heat_capacity_layer = 1 * rho * dx * T_in * Cp * (-1)  # 1 to represent the full layer, -1 bcs. T_in negative
    # make sure layer is not warmed beyond 0 °C (in case pot_Lh_rel_layer > heat_capacity_layer)
    Lh_release = pot_Lh_rel_layer * (pot_Lh_rel_layer < heat_capacity_layer) + \
                 heat_capacity_layer * (pot_Lh_rel_layer > heat_capacity_layer)
    refreezing = Lh_release / (1000 * dx * L)

    # calculate warming from latent heat release and adjust T_in
    T_out = T_in + Lh_release / (rho * dx * Cp)

    # calculate new rho after refreezing took place
    # pay attention that the refrozen water becomes ice and its volume grows by rho_water / rho_ice
    rho += 917 * refreezing * (1000 / 917)

    # calculate new IRWC after refreezing took place  and convert back to [kg water per layer]
    iw = irwc_existing - refreezing
    iw *= 1000 * dx

    return iw, T_out, rho, bottom_water


def calc_closed(t, n, T, dTdt, alpha, dx, Tsurf, dt, T_evol, phi, k, refreeze, L, iw, iwc, rho, Cp, melt,
                a, rho_tr, k_ref_i, k_ref_a):

    for j in range(0, len(t)-1):
        # calculation is for n + 2 layers. The n layers all have a thickness of D/n. The additional two layers are
        # the skin layer on top and the water saturated layer at the bottom
        porosity, irwc_max = rho_por_irwc_max(rho, iwc)

        # Update temperature top layer according to surface temperature time series (if one is prescribed)
        T[0] = Tsurf[j]

        # calculate temperature change for all layers.
        # alpha is zero for all layers that contain water (and must be at 0 °C), hence they cannot be cooled down
        # by conduction. Before a layer can cool to below 0 °C, all pore water in the layer needs to refreeze first.
        # Refreezing is calculated below.
        dTdt[:] = alpha * (-(T[1:-1] - T[0:-2]) / dx ** 2 + (T[2:] - T[1:-1]) / dx ** 2)

        T[1:-1] = T[1:-1] + dTdt * dt

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
        iw, T[1:-1], rho, bottom_water = bucket_scheme(L, Cp, melt, iw, irwc_max, T[1:-1], rho, dx, j)

        # update k and alpha for the next iteration
        k = k_update(T[1:-1], rho, a, rho_tr, k_ref_i, k_ref_a)
        alpha = alpha_update(k, rho, Cp, n, iw)

        # finally record temperature profile for later establishing figures and writing output table
        T_evol[:, j] = T

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

