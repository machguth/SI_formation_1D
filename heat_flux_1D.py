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
      layer's irreducible water first needs to be frozen before heat conduction through that layer is possible. This means
      any presence of irreducible water strongly slows the progression of a cold wave.
- Boundary conditions:
    - Top and bottom boundary conditions can be defined.
    - Bottom boundary condition can be left open.
    - Boundary conditions can be a constant temperature, sine curve of air temperature over several years,
      or simple linear change in air temperature.
- Numerics:
    - Simulation is 1D along a depth axis, the depth axis is divided into evenly spaced layers
    - Numerical time steps, resolution of the depth axis and parameter alpha (k / (rho * Cp)) need to match,
      the smaller layer spacing and the larger alpha, the shorter the time steps need to be chosen

"""

import numpy as np
import heat_flux_1D_functions as hf
import datetime

time_start = datetime.datetime.now()

# ============================================== input ===================================================

days = 10  # [days] time period for which to simulate
D = 1.8  # [m] thickness of snow pack
n = 50  # [] number of layers
T0 = 0  # [°C]  initial temperature of all layers
dx = D/n  # [m] layer thickness
k = 0.5  # [W m-1 K-1] Thermal conductivity of ice at rho approx. 400 kg m-3 = 0.5; for ice 2.25
Cp = 2090  # [J kg-1 K-1] Specific heat capacity of ice
L = 334000  # [J kg-1] Latent heat of water
rho = 400  # [kg m-3] Density of the snow or ice
iwc = 0  # [% of mass] Irreducible water content in snow
por = 0.4  # [] porosity of the snow where it is water saturated
t_final = 86400 * days  # [s] end of model run
dt = 300  # [s] numerical time step, needs to be a fraction of 86400 s

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
Tsurf = -20  # [°C] Top boundary condition
Tbottom = 0  # [°C] bottom boundary condition

output_dir = r'C:\horst\modeling\lateralflow'
# output_dir = r'D:\modelling\lateralflow'

# ============================================== Preparations ===================================================

y = np.linspace(-dx/2, D+dx/2, n+2)  # vector of central points of each depth interval (=layer)
t = np.arange(0, t_final, dt)  # vector of time steps
# T = np.ones(n) * T0  # vector of temperatures for each layer
# Vector of T for all layers (n + 2 (bottom and top)). Top set to zero, will then be updated each model step
T = np.append(np.insert(np.ones(n) * T0, 0, 0), Tbottom)
T_evol = np.ones([n+2, len(t)]) * T0  # array of temperatures for each layer and time step
dTdt = np.empty(n)  # derivative of temperature at each node
phi = np.empty([n+1, len(t)])  # array of the heat flux per time step, for each layer and time step
refreeze = np.empty([2, len(t)])
dt_plot = np.floor(len(t) / 40) * dt  # [s] time interval for which to plot temperature evolution

# ============================================== calculations ===================================================

# Water per layer (irreducible water content) [mm w.e. m-2 or kg m-2]
# This function also sets irreducible water content to 0 for all layers that have initial T < 0
iw, iwc = hf.irrw(iwc, n, dx, rho, T0)

# Vector of thermal diffusivity [m2 s-1]
alpha = hf.alpha_update(k, rho, Cp, n, iw)

# create the array of surface temperatures
if isinstance(Tsurf, int):
    Tsurf = np.ones(len(t)) * Tsurf
elif isinstance(Tsurf, str):
    Tsurf = hf.tsurf_sine(days, t_final, dt, years=5, Tmean=-20, Tamplitude=10)
else:
    Tsurf = np.linspace(Tsurf[0:-1], Tsurf[1:], int(86400/dt))
    Tsurf = Tsurf.flatten(order='F')

# calculation of temperature profile over time
if bottom_boundary:
    T_evol, phi, refreeze, iw = hf.calc_closed(t, n, T, dTdt, alpha, dx, Tsurf, dt,
                                               T_evol, phi, k, refreeze, L, iw, rho, Cp)
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
hf.plotting(T_evol, dt_plot, dt, y, D, slushatbottom, phi, days,
            t_final, t, refreeze_c, output_dir, iwc)
