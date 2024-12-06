# # 1D heat conduction calculation: minimum working example
#
# This example runs bottom-up heat flux/SIF calculations using a single T profile (here from FS2) as initialisation.
#

# +
import numpy as np
import pandas as pd
import xarray as xr
import datetime
from scipy import interpolate
import os
import warnings
import datetime as dt
import matplotlib.pyplot as plt

import sys
sys.path.append('/Users/atedston/scripts/SI_formation_1D')
import heat_flux_1D_functions as hf

# -

warnings.filterwarnings("ignore")

# Period to run calculations for
calc_start = '2022-07-06 14:15:00'
calc_end = '2022-12-31 23:30:00'

#output_dir = os.path.join(os.environ['WORKDIR'], '1D_heat_flux_modelling')
output_dir = '/scratch/'

# +
# MODEL VARIABLES

# Grid and time parameters
slab_depth = 12      # [m] thickness of snow pack or ice slab
param_dx = 0.05      # [m] desired layer thickness; might be adjusted
param_dt = 100       # [s] numerical time step, needs to be a fraction of 86400 s

# Vertical profile
param_rho = 920      # [kg m-3] Density of the snow or ice
param_k = 2.25       # [W m-1 K-1] Thermal conductivity of ice or snow: at rho 400 kg m-3 = 0.5; at rho=917 kg m-3: 2.25
param_iwc = 0        # [% of mass] Irreducible water content in snow
param_por = 0.4      # [] porosity of the snow where it is water saturated
param_Tbottom = None # [°C or None] bottom boundary condition, if None then temperature at bottom allowed to vary freely.
param_Tsurf = 0      # [deg C] Temperature at upper boundary

# The model calculates how much slush refreezes into superimposed ice (SI). Slush with refreezing can be
# prescribed either for the top or the bottom of the model domain (not both). Bottom is default (slushatbottom = True),
# if set to False, then slush and SI formation is assumed to happen at the top.
param_slushatbottom = False

# +
# Initial T profile information
# Filename of initial T profile (to be adjusted for given 10 m temperature)
#initial_Tprofile_fn =  os.path.join(os.environ['WORKDIR'], '1D_heat_flux_modelling/bottom-up_R2', 'D6050043-logged_data(FS2)_optimal_initial_Tprofile.xlsx')
initial_Tprofile_fn = '/Users/atedston/scratch/rlim_retention_scratch/1D_heat_flux_modelling/bottom-up_R2/D6050043-logged_data(FS2)_optimal_initial_Tprofile.xlsx'

# specify  height of the thermistor that is closest to the slab surface. Needed to discard all T measured above
# the slab. Any T above the slab (= snowpack T) will be set to 0 °C.
height_top_of_slab_thermistor = 2.15  # (m) 
# At T string installation in May 2022, there was 0.8 m snowpack overlying the ice slab.
# Thus, approximate 10 m temperature is located at (10 m slab depth - 10)

T10m_measured_FS2 = -10.06  # this is the temperature at 10 m depth in the T-profile 
                            # (depth below surface, not depth below slab) . 
                            # Used to scale with T10m for any grid in Vandecrux et al. 

# +
# Load the profile
df_mt = pd.read_excel(initial_Tprofile_fn)

depths = df_mt.columns.values  # columns that contain depth values
for ni, i in enumerate(depths):
    depths[ni] = float(i.split(' ')[0])
# ...this works by reference, so df_mt can now be transposed directly

# Transpose depth-as-columns to depth-as-rows, and squeeze to Series
df_mt = df_mt.T.squeeze()

# # make sure depth axis is positive as depth axis of model is also positive
# # subtract height of top thermistor to adjust to positive depth below ice slab
df_mt.index = df_mt.index * -1 - height_top_of_slab_thermistor

# +
# Get depth intervals, we need to interpolate and scale the temperature profile over these.
y = hf.depth_intervals(slab_depth, param_dx)

# Create the timesteps
t = pd.date_range(calc_start, calc_end, freq='{dt}s'.format(dt=param_dt))

# Plotting ...
# [s] time interval for which to plot temperature evolution
dt_plot = np.floor(len(t) / 40) * param_dt  
# Final timestep
t_final =( t[-1] - t[0]).total_seconds()
# Number of days in run
days = ( t[-1] - t[0]).days   
if param_slushatbottom:
    direc = 'bottom'
else:
    direc = 'top'

# +
# Run calculations
y, T_evol, refreeze, refreeze_c, phi = hf.run_calculation(
    slab_depth, param_dx, param_dt, t, 
    T10m_measured_FS2, param_rho, param_k, param_iwc, param_por, 
    param_Tbottom, param_Tsurf, param_slushatbottom,
    y=y)

# Basic diagnostics
print('\nHeat flux at the top of the domain, end of model run: {:.3f}'.format(phi[0, -2]) + ' W m-2')
print('Heat flux at the bottom of the domain, end of model run: {:.3f}'.format(phi[-2, -2]) + ' W m-2')
print('(downward flux is positive, upward flux negative)\n')

# refreezing.
da_ro = hf.refreeze_to_da(t, refreeze, param_slushatbottom)

# temperatures.
da_t = hf.temperatures_to_da(y, t, T_evol)


# -
### Regular plot                       
fig = hf.plotting(T_evol, dt_plot, param_dt, y, slab_depth, param_slushatbottom, phi, days,
                    t_final, t, refreeze_c, param_iwc)

# +
### Plot against observed values

validation_dates = ['2022/07/06 14:15:00', '2022/09/04 16:00:00']

measured_T = r'C:\Users\machguth\OneDrive - Université de Fribourg\modelling\1D_heat_conduction\D6050043-logged_data(FS2)_v2.xlsx'

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


save_to = '1D_heat_flux_{d}d_{dt}s_iwc{iwc}_depth{D}m_{direc}-SI_comp_meas.png'.format(d=days, dt=param_dt, iwc=param_iwc, D=slab_depth, direc=direc)
save_to = os.path.join(output_dir, save_to)
                         
fig = hf.plotting_incl_measurements(T_evol, dt_plot, param_dt, y, slab_depth, param_slushatbottom, phi, days,
                    t_final, t, refreeze_c, param_iwc, 
                    da, validation_dates,
                    save_to)
# -



