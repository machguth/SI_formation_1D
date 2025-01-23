""" *** Experimental 1D heat conduction model ****

- Basics:
    - Calculates heat conduction and the amount of superimposed ice (SI) that forms.
    - Intended for use in snow and ice.
    - Only conduction is modelled.
    - While refreezing is simulated, there is no melt simulation. During the model run, no water (as slush or
      irreducible water content) is added. All water is present from the start of the model run.
- SI formation and irreducible water content
    - It is assumed that there is infinite amount of slush available for refreezing.
    - SI formation can be simulated both at the top or at the bottom of the domain.
    - SI formation at the top to investigate how much SI forms when slush is sitting on top of a cold ice slab and
      heat flux is from the slush into the cold ice slab.
    - SI simulation at the bottom when a snow cover is partially filled with slush and heat flow is from the slush.
      through the snow and towards the snow surface (where heat is lost to a cold atmosphere).
    - The irreducible water content is being specified as input parameter
    - The model can take irreducible water content inside the snowpack into account. It does so by assuming that each
      layer's irreducible water first needs to be frozen before heat conduction through that layer is possible.
      This means any presence of irreducible water strongly slows the progression of a cold wave.
- Boundary conditions:
    - Top and bottom boundary conditions can be defined.
    - Bottom boundary condition can be left open.
    - Boundary conditions can be a constant temperature, sine curve of air temperature over several years,
      or simple linear change in air temperature.
- Numerics:
    - Simulation is 1D along a depth axis, the depth axis is divided into evenly spaced layers
    - Numerical time steps, resolution of the depth axis and parameter alpha (k / (rho * Cp)) need to match,
      the smaller layer spacing and the larger alpha, the shorter the time steps need to be chosen

ToDo: thermal conductivity as function of density, e.g. following Oster and Albert (2022), Calonne et al. (2011, 2019)

"""

import numpy as np
import pandas as pd
import xarray as xr

import heat_flux_1D_functions as hf
import datetime
from scipy import interpolate
import os
import warnings

warnings.filterwarnings("ignore")

time_start = datetime.datetime.now()

# ============================================== input ===================================================
# Compare to measurements?
# If yes, then measured temperatures are automatically used as starting conditions
compare_to_measurements = False

# Use an initial Temperature profile as starting condition? If this option is chosen then
# compare_to_measurements is set to False but an intial profile is still being read and used
use_initial_T_profile = False

# The following four variables are only relevant if compare_to_measurements = True AND use_initial_T_profile = False
measured_T = r'C:\Users\machguth\OneDrive - Université de Fribourg\modelling\1D_heat_conduction\D6050043-logged_data(FS2)_v2.xlsx'
top_thermistor_height = 2.15  # (m) height top thermistor above slab - required to correct depth intervals
# Dates for which the measured T-profile will be plotted into the output figures
validation_dates = ['2022/07/06 14:15:00', '2022/09/04 16:00:00']  # ['2022/07/05 18:30:00', '2022/08/17 16:00:00']
# ['2022/07/05 18:30:00', '2022/08/24 00:00:00'] # '2022/08/01 00:00:00', '2022/08/24 00:00:00'
# sensitivity study: multiply initial temperatures with a factor m in order to test sensitivity
# to different initial slab temperatures. Multiplication is chosen because temperature at the
# snow-slab interface is 0°C, which is preserved in multiplication
m = 1.0  # []

# initial T profile, is only used if use_initial_T_profile = True
# Here used is the measured FS2 T-profile from 2022/07/06 14:15:00
initial_Tprofile = r'C:\Users\machguth\OneDrive - Université de Fribourg\modelling\1D_heat_conduction\D6050043-logged_data(FS2)_optimal_initial_Tprofile.xlsx'
# specify  height of the thermistor that is closest to the slab surface. Needed to discard all T measured above
# the slab. Any T above the slab (= snowpack T) will be set to 0 °C.
height_top_of_slab_thermistor = 2.15  # (m) Keep unchanged if FS2 T-profile from 2022/07/06 14:15:00
T10m_measured_FS2 = -10.06  # (°C) T in the init. profile at 10 m - Keep unchanged if FS2 T from 2022/07/06 14:15:00
T10m_local = -12  # (°C) Temperature at 10 m depth at any given grid cell, according to Vandecrux et al. (2023)

# start and end date define the duration of the model run. The two variables are used also
# when there is no comparison to measurements.
start_date = '2022/07/06 14:15:00'  # '2022/07/05 18:30:00' # '2022/09/06 14:15:00'  #
end_date = '2022/07/31 23:30:00' # '2022/12/31 23:30:00'

D = 1  # [m] thickness of snow pack (top-down refreezing) or ice slab (bottom-up refreezing)
n = 25  # [] number of layers
T0 = -5  # [°C]  initial temperature of all layers - ignored if compare_to_measurements or use_initial_T_profile
dx = D/n  # [m] layer thickness
k = 2.25  # [W m-1 K-1] Thermal conductivity of ice or snow: at rho 400 kg m-3 = 0.5; at rho 917 kg m-3 = 2.25
Cp = 2090  # [J kg-1 K-1] Specific heat capacity of ice
L = 334000  # [J kg-1] Latent heat of water
rho = 400  # [kg m-3] Density of the snow or ice - can be a skalar or a density profile of depth D with n elements
iwc = 7  # [% of pore volume] Irreducible water content in snow
por = 0.4  # [] porosity of snow where water saturated (slush) - Variable only used to convert SIF from m w.e. to m
dt = 150  # [s] numerical time step, needs to be a fraction of 86400 s

# The model calculates how much slush refreezes into superimposed ice (SI). Slush with refreezing can be
# prescribed either for the top or the bottom of the model domain (not both). Bottom is default (slushatbottom = True),
# if set to False, then slush and SI formation is assumed to happen at the top.
slushatbottom = True
# specify if the bottom boundary condition should be applied or not (if not, temperatures at the bottom can fluctuate
# freely). If there is no bottom boundary condition, bottom heat flux will equal zero
bottom_boundary = True

# -20  # [°C] boundary condition temperature top
# can either be a scalar (e.g. -20 °C) or an array of length days + 1
# Tsurf = np.linspace(-20, -0, days + 1)
# Tsurf = 'sine'
Tsurf = 0  # [°C] Top boundary condition
# bottom boundary condition, initial value of T-profile. Overwritten if compare_to_measurements or use_initial_T_profile
Tbottom = 0  # [°C]

melt = 7.72e-07  # Surface melt [mm w.e. per time step] can be a scalar or an array of length equal number of time steps

# parameters used to calculate k based on Calonne et al. (2019)
a = 0.02  # [m^3/kg]
rho_tr = 450  # [kg m^-3]
k_ref_i = 2.107  # [W m^-1 k^-1]
k_ref_a = 0.024  # [W m^-1 k^-1]

# output_dir = r'C:\horst\modeling\lateralflow'
output_dir = r'C:\Users\machguth\OneDrive - Université de Fribourg\modelling\1D_heat_conduction\test'
# output_dir = r'O:\test_1D_heat_conduction'

# ============================================== Preparations ===================================================
# check if output folder exists, if no create
isdir = os.path.isdir(output_dir)
if not isdir:
    os.mkdir(output_dir)

# make sure that use_initial_T_profile = True is only used with compare_to_measurements = False
if use_initial_T_profile:
    compare_to_measurements = False

y = np.linspace(-dx/2, D+dx/2, n+2)  # vector of central points of each depth interval (=layer)
t = np.arange(pd.to_datetime(start_date), pd.to_datetime(end_date), np.timedelta64(dt, 's'))  # vector of time steps
days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
t_final = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).total_seconds()
#t = np.arange(0, t_final, dt)  # vector of time steps
# T = np.ones(n) * T0  # vector of temperatures for each layer
# Vector of T for all layers (n + 2 (bottom and top)). Top set to zero, will then be updated each model step
T = np.append(np.insert(np.ones(n) * T0, 0, 0), Tbottom)
T_evol = np.ones([n+2, len(t)]) * T0  # array of temperatures for each layer and time step
dTdt = np.empty(n)  # derivative of temperature at each node
phi = np.empty([n+1, len(t)])  # array of the heat flux per time step, for each layer and time step
refreeze = np.empty([2, len(t)])
dt_plot = np.floor(len(t) / 40) * dt  # [s] time interval for which to plot temperature evolution

if compare_to_measurements:

    # read the thermistor string data
    df_mt = pd.read_excel(measured_T)
    df_mt['dateUTC'] = pd.to_datetime(df_mt['DateTime (UTC)'], format='%m.%d.%Y %H:%M')
    df_mt.set_index('dateUTC', inplace=True)

    # establish list of depth values
    depths = df_mt.columns[5:].values  # columns that contain depth values
    for ni, i in enumerate(depths):
        depths[ni] = float(i.split(' ')[0])

    # make sure depth axis is positive as depth axis of model is also positive
    # subtract height of top thermistor to adjust to positive depth below ice slab
    depths = depths * (-1) - top_thermistor_height

    # Xarray DataArray of all temperature measurements
    da = xr.DataArray(
        data=df_mt[df_mt.columns[5:]].to_numpy(),
        dims=['time', 'z'],
        coords=dict(
            z=depths,
            time=df_mt.index.values
        ),
        attrs=dict(description="thermistor data FS2, Greenland Ice Sheet."),
    )

    # prepare vector T_start of initial ice slab and firn temperatures
    function1d = interpolate.interp1d(da.z.values, da.sel(time=start_date, method='nearest').values,
                                      fill_value='extrapolate')
    T_start = function1d(y)
    T_start[T_start > 0] = 0  # make sure no positive values in initial temperatures
    T_start[0] = 0  # make sure top grid cell is at 0 °C

    T = T_start  # finally set the temperature distribution to T_start

    # Sensitivity study, multiply starting temperature with a certain factor, in order to test the influence
    # of different temperatures on SI formation
    T *= m

if use_initial_T_profile:
    # read the thermistor string data
    df_mt = pd.read_excel(initial_Tprofile)
    # df_mt['dateUTC'] = pd.to_datetime(df_mt['DateTime (UTC)'], format='%m.%d.%Y %H:%M')
    # df_mt.set_index('dateUTC', inplace=True)

    # establish list of depth values
    depths = df_mt.columns.values  # columns that contain depth values
    for ni, i in enumerate(depths):
        depths[ni] = float(i.split(' ')[0])

    # make sure depth axis is positive as depth axis of model is also positive
    # subtract height of top thermistor to adjust to positive depth below ice slab
    depths = depths * (-1) - height_top_of_slab_thermistor

    # prepare vector T_start of initial ice slab and firn temperatures
    function1d = interpolate.interp1d(depths, df_mt, fill_value='extrapolate')

    T_start = function1d(y)[0].astype(float)
    T_start[T_start > 0] = 0  # make sure no positive values in initial temperatures
    T_start[0] = 0  # make sure top grid cell is at 0 °C

    T = T_start  # finally set the temperature distribution to T_start

    # multiply starting temperature with ratio between 10 m firn temperature in the initial T profile
    # and the 10 m firn T in the Grid cell that is simulated (T in grid cell according to Vandecrux et al., 2023)
    T *= T10m_local/T10m_measured_FS2


# ============================================== calculations ===================================================

# create the array of density values per layer
if isinstance(rho, int) or isinstance(rho, float):
    rho = np.ones(n) * rho
else:
    # here needs to be a function to read a density profile from a table and maybe to interpolate to the n layers
    pass

# Calculate initial porosity and irreducible water content
porosity, irwc_max = hf.rho_por_irwc_max(rho, iwc)

# Initialize water per layer (irreducible water content) [mm w.e. m-2 or kg m-2]
# This function also sets irreducible water content to 0 for all layers that have initial T < 0
# In contrast to previous version, the variable iwc is not changed as it now constitutes the maximum potential IRWC
iw = hf.irwc_init(iwc, irwc_max, dx, n, T0)

# Initial array of thermal conductivity
k = hf.k_update(T_evol[1:-1, 0], rho, a, rho_tr, k_ref_i, k_ref_a)

# Vector of thermal diffusivity [m2 s-1]
alpha = hf.alpha_update(k, rho, Cp, n, iw)

# create the array of surface temperatures (one entry per time step)
if isinstance(Tsurf, int):
    Tsurf = np.ones(len(t)) * Tsurf
elif isinstance(Tsurf, str):
    Tsurf = hf.tsurf_sine(days, t_final, dt, years=5, Tmean=-20, Tamplitude=10)
else:
    Tsurf = np.linspace(Tsurf[0:-1], Tsurf[1:], int(86400/dt))
    Tsurf = Tsurf.flatten(order='F')

# create the array of melt (one entry per time step)
if isinstance(melt, int) or isinstance(melt, float):
    melt = np.ones(len(t)) * melt
else:
    # here needs to be function reading melt from a table and maybe to interpolate to the t time steps
    pass

# calculation of temperature profile over time
if bottom_boundary:
    T_evol, phi, refreeze, iw = hf.calc_closed(t, n, T, dTdt, alpha, dx, Tsurf, dt,
                                               T_evol, phi, k, refreeze, L, iw, iwc, rho, Cp, melt,
                                               a, rho_tr, k_ref_i, k_ref_a)
else:
    T_evol, phi, refreeze = hf.calc_open(t, n, T, dTdt, alpha, dx, Tsurf, dt, T_evol,
                                         phi, k, refreeze, L, iw, rho, Cp)

# cumulative sum of refrozen water
refreeze_c = np.cumsum(refreeze, axis=1)
# and correct for the fact that water occupies only the pore space
refreeze_c /= por

print('\nHeat flux at the top of the domain, end of model run: {:.3f}'.format(phi[0, -2]) + ' W m-2')
print('Heat flux at the bottom of the domain, end of model run: {:.3f}'.format(phi[-2, -2]) + ' W m-2')
print('(downward flux is positive, upward flux negative)\n')

time_end_calc = datetime.datetime.now()
print('runtime', time_end_calc - time_start)

# plotting
if compare_to_measurements:
    hf.plotting_incl_measurements(T_evol, dt_plot, dt, y, D, slushatbottom, phi, days,
                                  t_final, t, refreeze_c, output_dir, iwc, da, m, validation_dates)
else:
    hf.plotting(T_evol, dt_plot, dt, y, D, slushatbottom, phi, days,
                t_final, t, refreeze_c, output_dir, m, iwc)

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

if m == 1:
    da_to.to_netcdf(path=output_dir + '/simulated_daily_T_evolution.nc')
    da_ro.to_netcdf(path=output_dir + '/simulated_daily_refreezing.nc')
else:
    da_to.to_netcdf(path=output_dir + '/simulated_daily_T_evolution_Tmultiplied_by_{:.1f}'.format(m) + '.nc')
    da_ro.to_netcdf(path=output_dir + '/simulated_daily_refreezing_Tmultiplied_by_{:.1f}'.format(m) + '.nc')
