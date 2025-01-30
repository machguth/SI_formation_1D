""" *** Experimental 1D heat conduction model ****

- Basics:
    - Calculates heat conduction and the amount of superimposed ice (SI) that forms.
    - Intended for use in snow and ice.
    - Conduction is modelled as well as the effect of percolating meltwater through a bucket scheme.
    - While refreezing is simulated, there is no melt simulation. However, melt or an array of melt per time
      step can be prescribed
- SI formation and irreducible water content
    - It is assumed that there is infinite amount of slush available for refreezing.
    - SI formation can be simulated at the bottom of the domain.
    - SI simulation at the bottom when a snow cover is partially filled with slush and heat flow is from the slush.
      through the snow and towards the snow surface (where heat is lost to a cold atmosphere).
    - The irreducible water content is being specified as input parameter
    - The model can take irreducible water content inside the snowpack into account. It does so by assuming that each
      layer's irreducible water first needs to be frozen before heat conduction through that layer is possible.
      This means any presence of irreducible water strongly slows the progression of a cold wave.
- Boundary conditions:
    - Top and bottom boundary conditions can be defined.
    - Surface boundary conditions can be a constant temperature or reading an array of T_surf per time step.
- Numerics:
    - Simulation is 1D along a depth axis, the depth axis is divided into evenly spaced layers
    - Numerical time steps, resolution of the depth axis and parameter alpha (k / (rho * Cp)) need to match,
      the smaller layer spacing and the larger alpha, the shorter the time steps need to be chosen
    - thermal conductivity is a function of density, following Calonne et al. (2019)

    todo: irreducible water that is present after the snowpack has warmed to 0 °C does not yet add
    to snow density by the moment the snowpack cools down. Irreducible water gets modified in hf.bucket_scheme()
    and in hf.calc_closed(). Check whether both are needed, that they do not conflict and if both needed, that
    both modify the density.
    todo: bottom water does not yet work properly


"""

import numpy as np
import pandas as pd
import xarray as xr

import heat_flux_1D_functions as hf
import heat_flux_1D_plotting as hp
import datetime
import os
import warnings

warnings.filterwarnings("ignore")

time_start = datetime.datetime.now()

# ============================================== input ===================================================

# start and end date define the duration of the model run. The two variables are used also
# when there is no comparison to measurements.
start_date = '2022/07/06 14:15:00'  # '2022/07/05 18:30:00' # '2022/09/06 14:15:00'  #
end_date = '2022/08/31 23:30:00' # '2022/12/31 23:30:00'

print('Marcus, use versioning!')

D = 0.5  # [m] thickness of snow pack (top-down refreezing) or ice slab (bottom-up refreezing)
n = 10  # [] number of layers
T0 = -5  # [°C]  initial temperature of all layers
dx = D/n  # [m] layer thickness
# Thermal conductivity of ice or snow [W m-1 K-1]: now function of rho and T, following Calonne et al. (2019)
Cp = 2090  # [J kg-1 K-1] Specific heat capacity of ice
L = 334000  # [J kg-1] Latent heat of water
rho = 400  # [kg m-3] Initial density of the snow or ice - can be scalar or density profile of depth D with n elements
iwc = 5  # [% of pore volume] Max. possible irreducible water content in snow
por = 0.4  # [] porosity of snow where water saturated (slush) - Variable only used to convert SIF from m w.e. to m
dt = 300  # [s] numerical time step, needs to be a fraction of 86400 s

# The model calculates how much slush refreezes into superimposed ice (SI). Slush with refreezing can be
# currently assumed that there is always slush at the bottom

# Surface temperature (top boundary condition)
Tsurf = [0, -10, 1000, 3000, 5]  # [°C] scalar, file name or five element list specifying synthetic data
# bottom boundary condition.
Tbottom = 0  # [°C]

# Surface melt can be a scalar or an array of length equal number of time steps
# (at 300 s time steps, 6.95e-05 m w.e. per time step corresponds to ~20 mm w.e. melt per day)
melt = [6.95e-05, 0, 1000, 3000, 5]  # [m w.e. per time step] scalar, filename or 5 el. list specifying synthetic data

# parameters used to calculate k based on Calonne et al. (2019)
a = 0.02  # [m^3/kg]
rho_tr = 450  # [kg m^-3]
k_ref_i = 2.107  # [W m^-1 k^-1]
k_ref_a = 0.024  # [W m^-1 k^-1]

# output_dir = r'C:\horst\modeling\lateralflow'
output_dir = r'C:\Users\machguth\OneDrive - Université de Fribourg\modelling\1D_heat_conduction\test'
# output_dir = r'O:\test_1D_heat_conduction'
# output_dir = r'D:\modelling\1d_heat_transfer'

# ============================================== Preparations ===================================================
# check if output folder exists, if no create
isdir = os.path.isdir(output_dir)
if not isdir:
    os.mkdir(output_dir)

y = np.linspace(-dx/2, D+dx/2, n+2)  # vector of central points of each depth interval (=layer)
t = np.arange(pd.to_datetime(start_date), pd.to_datetime(end_date), np.timedelta64(dt, 's'))  # vector of time steps
days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
t_final = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).total_seconds()
# Vector of T for all layers (n + 2 (bottom and top)). Top set to zero, will then be updated each model step
T = np.append(np.insert(np.ones(n) * T0, 0, 0), Tbottom)
T_evol = np.ones([n+2, len(t)]) * T0  # array of temperatures for each layer and time step
dTdt = np.empty(n)  # derivative of temperature at each node
phi = np.empty([n+1, len(t)])  # array of the heat flux per time step, for each layer and time step
refreeze = np.empty([2, len(t)])
dt_plot = np.floor(len(t) / 40) * dt  # [s] time interval for which to plot temperature evolution


# ============================================== calculations ===================================================

# create the array of density values per layer
if isinstance(rho, int) or isinstance(rho, float):
    rho = np.ones(n) * rho
else:
    # here needs to be a function to read a density profile from a table and maybe to interpolate to the n layers
    pass

# initialize array to record evolution of rho
rho_evol = np.zeros([n, len(t)])

# Calculate initial porosity and irreducible water content
porosity, irwc_max = hf.rho_por_irwc_max(rho, iwc)

# Initialize water per layer (irreducible water content) [mm w.e. m-2 or kg m-2]
# This function also sets irreducible water content to 0 for all layers that have initial T < 0
# In contrast to previous version, the variable iwc is not changed as it now constitutes the maximum potential IRWC
iw = hf.irwc_init(iwc, irwc_max, dx, n, T0)

# initialize array to record evolution of irreducible water content and of cumulative discharge D
iw_evol = np.zeros([n, len(t)])
D_evol = np.zeros(len(t))

# Initial array of thermal conductivity
k = hf.k_update(hf.C_to_K(T_evol[1:-1, 0]), rho, a, rho_tr, k_ref_i, k_ref_a)

# Vector of thermal diffusivity [m2 s-1]
alpha = hf.alpha_update(k, rho, Cp, n, iw)

# create the array of surface temperatures (one entry per time step)
if isinstance(Tsurf, int):
    Tsurf = np.ones(len(t)) * Tsurf
elif isinstance(Tsurf, str):
    # read a file, maybe interpolate temporally
    pass
elif isinstance(Tsurf, list):  # in this case create synthetic test data
    Tsurf = hf.create_test_data(Tsurf[0], Tsurf[1], Tsurf[2], Tsurf[3], Tsurf[4])
else:
    pass
    # Tsurf = np.linspace(Tsurf[0:-1], Tsurf[1:], int(86400/dt))
    # Tsurf = Tsurf.flatten(order='F')

# create the array of melt (one entry per time step)
if isinstance(melt, int) or isinstance(melt, float):
    melt = np.ones(len(t)) * melt
elif isinstance(melt, str):
    # read a file, maybe interpolate temporally
    pass
elif isinstance(melt, list):
        melt = hf.create_test_data(melt[0], melt[1], melt[2], melt[3], melt[4])

# calculation of temperature profile over time
T_evol, phi, refreeze, iw_evol, rho_evol, D_evol = hf.calc_closed(t, n, T, dTdt, alpha, dx, Tsurf, dt,
                                                    T_evol, phi, k, refreeze, L, iw, iwc, iw_evol, D_evol,
                                                    rho, rho_evol, Cp, melt,
                                                    a, rho_tr, k_ref_i, k_ref_a)

# cumulative sum of refrozen water
refreeze_c = np.cumsum(refreeze, axis=1)
# and correct for the fact that water occupies only the pore space
refreeze_c /= por

# cumulative sum of bottom water
D_evol = np.cumsum(D_evol)

print('\nHeat flux at the top of the domain, end of model run: {:.3f}'.format(phi[0, -2]) + ' W m-2')
print('Heat flux at the bottom of the domain, end of model run: {:.3f}'.format(phi[-2, -2]) + ' W m-2')
print('(downward flux is positive, upward flux negative)\n')

time_end_calc = datetime.datetime.now()
print('runtime', time_end_calc - time_start)

# plotting
hp.plotting(T_evol, dt_plot, dt, y, D, True, phi, days,
            t_final, t, refreeze_c, output_dir, 1, iwc)

hp.test_T_plotting1(T_evol, phi, refreeze_c, rho_evol, iw_evol, D_evol, t, melt, days, iwc, dt, n, dx, output_dir)

# write output
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

# Xarray DataArray of all simulated daily refreezing rates
da_ro = xr.DataArray(
    data=refreeze[1,:],
    dims=['time'],
    coords=dict(
        time=t
    ),
    attrs=dict(description="Simulated refreezing rates at the top of the modelling domain.",
               units='mm w.e. per time step',
               long_name='Refreezing R refers to water that refreezes, does not include surrounding matrix'),
)
da_ro.name = 'R'

da_to = da_to.resample(time='1D').mean()
da_to = da_to.coarsen(z=2, boundary='trim').mean()

da_ro = da_ro.resample(time='1D').sum()

da_to.to_netcdf(path=output_dir + '/simulated_daily_T_evolution.nc')
da_ro.to_netcdf(path=output_dir + '/simulated_daily_refreezing.nc')

