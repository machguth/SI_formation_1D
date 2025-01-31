""" *** 1D heat conduction model ***
Intended for use in snow and ice and with a
Calculates heat conduction and the amount of superimposed ice that forms

This file contains functions to write output which are being called by heat_flux_1D.py and heat_flux_1D_v2.py
"""

import xarray as xr


def write_output(T_evol, refreeze, output_dir, y, t):
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

