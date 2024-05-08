"""
Calculating Greenland SI formation
Plotting SI data from Greenland
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import AutoMinorLocator
from scipy import stats

# to temporarily get rid of silly minimalistic way of displaying pd data frames
# pd.set_option("display.max_rows", None, "display.max_columns", None)

# ###################################################################################################


# establish dataframe which contains information on infiltration ice content
def compact_stratigraphy(strat, ice_percent):
    ii = pd.DataFrame(columns=['topdepth', 'bottomdepth', 'unit'])
    top = 0
    count = 0
    for c in strat.index:
        # add condition for last element
        if c == np.max(strat.index):
            ii.loc[count, 'topdepth'] = top
            ii.loc[count, 'bottomdepth'] = c
            ii.loc[count, 'unit'] = strat.loc[c, ice_percent]
        elif strat.loc[c, ice_percent] != strat.loc[c+1, ice_percent]:
            ii.loc[count, 'topdepth'] = top
            ii.loc[count, 'bottomdepth'] = c
            ii.loc[count, 'unit'] = strat.loc[c, ice_percent]
            top = c
            count += 1
    return ii


# Plot 1: creates a subplot which compares stratigraphy in year n against stratigraphy in year n+1
def stake_plots(ns, s, yrs, df, depth_fig, rho_IE):
    print(str(yrs[0]) + '/' + str(yrs[1]) + ': plotting ' + s)
    ax[ns].set_title(s + '\n' + str(yrs[0]) + '     ' + str(yrs[1]), pad=40)
    ax[ns].plot(df[str(yrs[0]) + ' density'], df.index / 100, color='black', lw=2, zorder=25,
                label='$\\rho$ meas.' if ns == 0 else '')
    ax[ns].plot(df[str(yrs[1]) + ' density'] + 1000, df.index / 100, color='black', lw=2, zorder=25)
    ax[ns].plot(df[str(yrs[0]) + ' density_final'], df.index / 100, color='black', ls='--', lw=2, zorder=20,
                label='$\\rho$ estimated' if ns == 0 else '')
    ax[ns].plot(df[str(yrs[1]) + ' density_final'] + 1000, df.index / 100, color='black', ls='--', lw=2, zorder=20)
    for nyr, yr in enumerate(yrs):  # nyr is 0 or 1
        yr = str(yr)
        # create simplified material column which only distinguished snow, firn and ice
        df[yr + ' material_simple'] = df[yr + ' material'].str.split(",", n=1).str[0]
        # set %ice to 100 wherever the material is ice and there is no lower percentage indicated
        df.loc[(df[yr + ' % ice'].isnull()) & (df[yr + ' material_simple'] == 'ice'), yr + ' % ice'] = 100
        # set %ice to 0 wherever there is no percentage yet
        df.loc[df[yr + ' % ice'].isnull(), yr + ' % ice'] = 0

        xmin, xmax = nyr * 0.5, 0.5 + nyr * 0.5  # xmin and xmax so strat. plots over half the width of subplot

        # plotting ice lenses: establish a stratigraphy that allows plotting using axhspan
        iiyr = compact_stratigraphy(df, yr + ' % ice')
        for niy, iy in enumerate(iiyr.loc[iiyr.unit > 0].index):
            ax[ns].axhspan(iiyr.loc[iy, 'topdepth'] / 100, iiyr.loc[iy, 'bottomdepth'] / 100,
                           xmin=xmin, xmax=xmax, color='cornflowerblue', zorder=10,
                           alpha=iiyr.loc[iy, 'unit'] / 100,
                           label='ice' if ns == 0 and niy == 0 and nyr == 0 else '')
        # plotting firn: establish a stratigraphy that allows plotting using axhspan
        iiyr = compact_stratigraphy(df, yr + ' material_simple')
        for niy, iy in enumerate(iiyr.loc[iiyr.unit == 'firn'].index):
            ax[ns].axhspan(iiyr.loc[iy, 'topdepth'] / 100, iiyr.loc[iy, 'bottomdepth'] / 100,
                           xmin=xmin, xmax=xmax, color='lightgray', zorder=0, alpha=0.9,
                           label='firn' if ns == 0 and niy == 0 and nyr == 0 else '')
        # plotting snow: establish a stratigraphy that allows plotting using axhspan
        iiyr = compact_stratigraphy(df, yr + ' material_simple')
        for niy, iy in enumerate(iiyr.loc[iiyr.unit == 'snow'].index):
            ax[ns].axhspan(iiyr.loc[iy, 'topdepth'] / 100, iiyr.loc[iy, 'bottomdepth'] / 100,
                           xmin=xmin, xmax=xmax, color='lightgray', zorder=0, alpha=0.4,
                           label='snow' if ns == 0 and niy == 0 and nyr == 0 else '')
        # fill below the bottom of cores until bottom of plot with hatched conflowerblue
        bottom = np.max(df.loc[df[yr + ' material_simple'].notnull()].index) / 100
        if bottom < depth_fig:
            ax[ns].axhspan(bottom, depth_fig, facecolor='cornflowerblue', edgecolor='white', hatch='//',
                           xmin=xmin, xmax=xmax, linewidth=0.0, alpha=1, zorder=5,
                           label='ice extrapol.' if ns == 0 and nyr == 0 else '')
            ax[ns].plot([rho_IE + nyr * 1000, rho_IE + nyr * 1000], [bottom, depth_fig],
                        color='black', ls='--', lw=2, zorder=20)

    ax[ns].set_ylim([0, depth_fig])
    ax[ns].set_xlim([0, 2000])
    if ns == 0:
        ax[ns].set_ylabel('Depth (m)')
    ax[ns].set_xlabel('Density (g cm$^{-3}$)')
    ax[ns].invert_yaxis()
    minor_locator = AutoMinorLocator(2)
    ax[ns].yaxis.set_minor_locator(minor_locator)
    ax[ns].set_yticks(np.arange(0, 3.1, 0.5))
    ax[ns].set_xticks(np.arange(0, 2001, 500), labels=['0', '0.5', '1', '', ''])
    ax2 = ax[ns].twiny()
    ax2.set_xlim([0, 2000])
    ax2.set_xticks(np.arange(0, 2001, 500), labels=['', '', '0', '0.5', '1'])
    ax[ns].yaxis.set_ticks_position('both')
    ax[ns].tick_params(axis='both', which='major', top=True, length=7)
    ax[ns].tick_params(axis='both', which='minor', length=4)

# Plot 2: creates a subplot which compares stratigraphy in year n against stratigraphy in year n+1
def Drho_plots(ns, s, yrs, df, depth_fig, dfcy):

    SI_top = int(np.round(dfcy.loc[s, str(yrs[1]) + ' slab below surface (m)'] * 100))
    SI_bottom = int(np.round((dfcy.loc[s, 'delta surface height (m)'] +
                             dfcy.loc[s, str(yrs[0]) + ' slab below surface (m)']) * 100))

    print(str(yrs[0]) + '/' + str(yrs[1]) + ': plotting D rho ' + s)

    ax[ns].set_title(s + '\n' + str(yrs[0]) + '     ' + str(yrs[1]), pad=40)
    ax[ns].plot(df['delta_density_final_filled'], df.index / 100, color='black', lw=2, zorder=25,
                label='$\\Delta \\rho$' if ns == 0 else '')
    ax[ns].plot(df.loc[SI_top:SI_bottom, 'delta_density_final_filled'],
                df.loc[SI_top:SI_bottom].index / 100, color='red', lw=2, zorder=30,
                label='SI $\\Delta \\rho$' if ns == 0 else '')

    for nyr, yr in enumerate(yrs):  # nyr is 0 or 1
        yr = str(yr)
        # create simplified material column which only distinguished snow, firn and ice
        df[yr + ' material_simple'] = df[yr + ' material'].str.split(",", n=1).str[0]
        # set %ice to 100 wherever the material is ice and there is no lower percentage indicated
        df.loc[(df[yr + ' % ice'].isnull()) & (df[yr + ' material_simple'] == 'ice'), yr + ' % ice'] = 100
        # set %ice to 0 wherever there is no percentage yet
        df.loc[df[yr + ' % ice'].isnull(), yr + ' % ice'] = 0

        xmin, xmax = nyr * 0.5, 0.5 + nyr * 0.5  # xmin and xmax so strat. plots over half the width of subplot

        # plotting ice lenses: establish a stratigraphy that allows plotting using axhspan
        iiyr = compact_stratigraphy(df, yr + ' % ice')
        for niy, iy in enumerate(iiyr.loc[iiyr.unit > 0].index):
            ax[ns].axhspan(iiyr.loc[iy, 'topdepth'] / 100, iiyr.loc[iy, 'bottomdepth'] / 100,
                           xmin=xmin, xmax=xmax, color='cornflowerblue', zorder=10,
                           alpha=iiyr.loc[iy, 'unit'] / 100,
                           label='ice' if ns == 0 and niy == 0 and nyr == 0 else '')
        # plotting firn: establish a stratigraphy that allows plotting using axhspan
        iiyr = compact_stratigraphy(df, yr + ' material_simple')
        for niy, iy in enumerate(iiyr.loc[iiyr.unit == 'firn'].index):
            ax[ns].axhspan(iiyr.loc[iy, 'topdepth'] / 100, iiyr.loc[iy, 'bottomdepth'] / 100,
                           xmin=xmin, xmax=xmax, color='lightgray', zorder=0, alpha=0.9,
                           label='firn' if ns == 0 and niy == 0 and nyr == 0 else '')
        # plotting snow: establish a stratigraphy that allows plotting using axhspan
        iiyr = compact_stratigraphy(df, yr + ' material_simple')
        for niy, iy in enumerate(iiyr.loc[iiyr.unit == 'snow'].index):
            ax[ns].axhspan(iiyr.loc[iy, 'topdepth'] / 100, iiyr.loc[iy, 'bottomdepth'] / 100,
                           xmin=xmin, xmax=xmax, color='lightgray', zorder=0, alpha=0.4,
                           label='snow' if ns == 0 and niy == 0 and nyr == 0 else '')
        # fill below the bottom of cores until the bottom of the plot with hatched conflowerblue
        bottom = np.max(df.loc[df[yr + ' material_simple'].notnull()].index) / 100
        if bottom < depth_fig:
            ax[ns].axhspan(bottom, depth_fig, facecolor='cornflowerblue', edgecolor='white', hatch='//',
                           xmin=xmin, xmax=xmax, linewidth=0.0, alpha=1, zorder=5,
                           label='ice extrapol.' if ns == 0 and nyr == 0 else '')

    # mark depth range of SI formation
    ax[ns].axhspan(SI_top/100, SI_bottom/100, fill=False, color='red', hatch='..', alpha=0.4, zorder=40,
                   label='SI formation' if ns == 0 else '')

    ax[ns].set_ylim([0, depth_fig])
    ax[ns].set_xlim([-1000, 1000])
    if ns == 0:
        ax[ns].set_ylabel('Depth (m)')
    ax[ns].set_xlabel('$\\Delta$ Density (g cm$^{-3}$)')
    ax[ns].invert_yaxis()
    minor_locator = AutoMinorLocator(2)
    ax[ns].yaxis.set_minor_locator(minor_locator)
    ax[ns].set_yticks(np.arange(0, 3.1, 0.5))
    ax[ns].set_xticks(np.arange(-1000, 1001, 500), labels=['-1', '-0.5', '0', '0.5', '1'])
    ax[ns].yaxis.set_ticks_position('both')
    ax[ns].tick_params(axis='both', which='major', top=True, length=7)
    ax[ns].tick_params(axis='both', which='minor', length=4)


# Plot 3: Linear regression between two parameters
def linear_regression(ivar, dvar, xlab, ylab, name, slope, intercept, r2, p_value, outfolder,
                      outfilename_head, outfilename_v):
    line = slope * ivar + intercept
    # ----- plotting -----
    plt.rcParams.update({'font.size': 28})
    fig, ax = plt.subplots(figsize=(16, 12))  #
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

    if intercept >= 0:
        ax.plot(ivar, line, ls=':', lw=1, color='blue',
                label='y={:.2f}x+{:.2f}; R2={:.2f}; p={:.3f}'.format(slope, intercept, r2, p_value))
    else:
        ax.plot(ivar, line, ls=':', lw=1, color='blue',
                label='y={:.2f}x-{:.2f}; R2={:.2f}; p={:.3f}'.format(slope, np.abs(intercept), r2, p_value))

    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    # ax.yaxis.set_major_locator(MultipleLocator(200))
    ax.tick_params(which='major', length=8)
    ax.tick_params(which='minor', length=3)

    im1 = ax.scatter(ivar, dvar, c='tab:orange', s=200)

    ax.legend(frameon=False)
    plt.tight_layout()

    plt.savefig(outfolder + outfilename_head + name + outfilename_v + '.png')
    plt.close()

# ###################################################################################################
# user = 'machguth'
user = 'machg'

# define data and variables   2021: r'C:/Users/' + user + '/switchdrive/_current/SI/',
SI = {2021: r'C:/Users/' + user + '/switchdrive/_current/SI/_SI_stakes_2021_v20230629.xlsx',
      2022: r'C:/Users/' + user + '/switchdrive/_current/SI/_SI_stakes_2022_v20230628.xlsx',
      2023: r'C:/Users/' + user + '/switchdrive/_current/SI/_SI_stakes_2023_v20230828.xlsx'}

depth_fig = 3  # [m] maximum depth displayed in figures of stratigraphy comparisons

plt_params = {'font.size': 22, 'hatch.linewidth': 1.5}  # some parameters for all plots

rho_IE = 897.8  # [kg m-3] density of infiltration ice, is simply used to draw extrapolate density to bottom of plot

outfilename_head = 'SI-stats_'

# outfilename_v = '_test'
outfilename_v = '_v20231212'

outfolder = r'C:/Users/' + user + '/switchdrive/_current/SI/'

# ######################################## calculation and plotting ##################################################
# check if output folder exists, if no create
isdir = os.path.isdir(outfolder)
if not isdir:
    os.mkdir(outfolder)

# "xlsx_files" dictionary of pandas ExcelFile objects
xlsx_files = {k: None for k in SI}
for i in xlsx_files:
    xlsx_files[i] = pd.ExcelFile(SI[i])

# "overview" dictionary of all annual overview tables from the xlsx spread sheets
overview = {k: None for k in SI}
for i in xlsx_files:
    overview[i] = xlsx_files[i].parse('overview', header=1).sort_values('stake')  # .reset_index()
    overview[i].set_index('stake', inplace=True)

# "stakes" dictionary which contains for each year the names of all stake locations with measurements
stakes = {k: None for k in SI}
for i in overview:
    s = overview[i].index
    stakes[i] = s.str.rsplit("_", n=1).str[0].tolist()

# dictionary "cs" contains for each pair of years (e.g. 2021-2022) the names of the stake locations
# with measurements for both individual years n and n+1
# dictionary "cy" contains for each pair of years (e.g. 2021-2022) the two years involved
# "cs" and "cy" dictionaries are aech one element smaller than the other dictionaries
# because for n years of measurements, the number of possible comparisons is n-1
cs, cy = {}, {}
y = list(stakes.keys())
for n in range(len(stakes)-1):
    cy[n] = y[n:n+2]
    cs[n] = sorted(list(set(stakes[y[n]]).intersection(stakes[y[n+1]])))

# create list that will contain overview dataframes, one for each pair of years
dfcy = []
for n in cy:
    # initiate overview table with information about SI formation from year n to year n+1
    dfcy_cols = ['location', str(cy[n][0]) + ' lat', str(cy[n][0]) + ' lon', str(cy[n][0]) + ' Z (m a.s.l.)',
                 str(cy[n][1]) + ' lat', str(cy[n][1]) + ' lon', str(cy[n][1]) + ' Z (m a.s.l.)',
                 str(cy[n][0]) + ' stake above surface (m)',
                 str(cy[n][1]) + ' stake above surface (m)',
                 'delta surface height (m)', str(cy[n][0]) + ' slab below surface (m)',
                 str(cy[n][1]) + ' slab below surface (m)', 'SI growth (m)',
                 'climatic SMB (m w.e.)', 'SI based on delta measured rho (m w.e.)',
                 'SI in SMB (%)',
                 str(cy[n][0]) + ' mean rho before SI formation',
                 str(cy[n][1]) + ' mean rho after SI formation']
    dfcy.append(pd.DataFrame(columns=dfcy_cols, index=cs[n]))

# prepare dictionary for statistics on firn density prior and after SI formation
# this dataframe will contain all density values, only for the depth intervals where SI has formed
# between year n and n+1
dd = {'site': [], 'rho_before_SI_formation': [], 'rho_after_SI_formation': []}

for n in cy:  # Loop over the pairs of years (e.g. 2021-2022)

    # one plot per pair of years is created that contains subplots for each stake location
    plt.rcParams.update({'font.size': 22, 'hatch.linewidth': 1.5})  # plt.rcParams['hatch.linewidth'] = 1
    fig, ax = plt.subplots(nrows=1, ncols=len(cs[n]), figsize=(len(cs[n]) * 4.4, 15))

    # fill overview data frame
    dfcy[n].location = cs[n]
    yr0, yr1 = cy[n][0], cy[n][1]
    t0, t1 = [s + '_' + str(yr0) for s in cs[n]], [s + '_' + str(yr1) for s in cs[n]]
    dfcy[n][str(yr0) +' lat'] = overview[yr0].loc[t0, 'N'].array
    dfcy[n][str(yr0) + ' lon'] = overview[yr0].loc[t0, 'E'].array
    dfcy[n][str(yr0) +' Z (m a.s.l.)'] = overview[yr0].loc[t0, 'Z'].array
    dfcy[n][str(yr1) +' lat'] = overview[yr1].loc[t1, 'N'].array
    dfcy[n][str(yr1) + ' lon'] = overview[yr1].loc[t1, 'E'].array
    dfcy[n][str(yr1) +' Z (m a.s.l.)'] = overview[yr1].loc[t1, 'Z'].array
    dfcy[n][str(yr0) + ' stake above surface (m)'] = overview[yr0].loc[t0, 'total above surface (m) '].array
    dfcy[n][str(yr1) + ' stake above surface (m)'] = overview[yr1].loc[t1, 'OLD STAKE total above surface (m) '].array
    dfcy[n]['delta surface height (m)'] = dfcy[n][str(yr0) + ' stake above surface (m)'] - \
                                       dfcy[n][str(yr1) + ' stake above surface (m)']
    dfcy[n][str(yr0) + ' slab below surface (m)'] = overview[yr0].loc[t0, 'stake (total) above ice slab (m)'].array - \
                                                 overview[yr0].loc[t0, 'total above surface (m) '].array
    dfcy[n][str(yr1) + ' slab below surface (m)'] = overview[yr1].loc[t1, 'stake (total) above ice slab (m)'].array - \
                                                 overview[yr1].loc[t1, 'total above surface (m) '].array

    # calculate the amount of SI formation
    dfcy[n]['SI growth (m)'] = dfcy[n][str(yr0) + ' slab below surface (m)'] - \
                               dfcy[n][str(yr1) + ' slab below surface (m)'] +\
                               dfcy[n]['delta surface height (m)']

    # create data frames of shifted depth profiles and stratigraphies
    dict_cs = {k: None for k in cs[n]}
    for ns, s in enumerate(dict_cs):  # loop over individual stakes

        # read the data
        # an exception is needed here for SIS_FS3_2022 where two cores were drilled and analyzed
        if s == 'SIS_FS3' and cy[n][0] == 2022:
            df0 = xlsx_files[cy[n][0]].parse(s + '_core2', header=0)
        else:
            df0 = xlsx_files[cy[n][0]].parse(s, header=0)
        if s == 'SIS_FS3' and cy[n][1] == 2022:
            df1 = xlsx_files[cy[n][1]].parse(s + '_core2', header=0)
        else:
            df1 = xlsx_files[cy[n][1]].parse(s, header=0)

        df0.set_index('depth (cm)', inplace=True)
        df1.set_index('depth (cm)', inplace=True)

        dict_cs[s] = pd.DataFrame(columns=['depth (cm)',
                                           str(cy[n][0]) + ' material',
                                           str(cy[n][1]) + ' material',
                                           str(cy[n][0]) + ' % ice',
                                           str(cy[n][1]) + ' % ice',
                                           str(cy[n][0]) + ' density',
                                           str(cy[n][1]) + ' density',
                                           str(cy[n][0]) + ' density_final',
                                           str(cy[n][1]) + ' density_final',
                                           str(cy[n][0]) + ' density_final_filled',
                                           str(cy[n][1]) + ' density_final_filled',
                                           'delta_density_final_filled'
                                           ])

        # make sure code can also deal with the situation that the slab shrinks due to extreme melt
        shift = int(np.round(dfcy[n].loc[s, 'delta surface height (m)'] * 100))
        if shift >= 0:
            length = np.max([df0.index.max() + 1 + shift, df1.index.max() + 1])
            dict_cs[s]['depth (cm)'] = np.arange(1, length)
            dict_cs[s].set_index('depth (cm)', inplace=True)
            dict_cs[s].loc[shift:np.max(df0.index)+shift-1, str(cy[n][0]) + ' material'] = df0['material'].array
            dict_cs[s].loc[0:np.max(df1.index), str(cy[n][1]) + ' material'] = df1['material'].array
            dict_cs[s].loc[shift:np.max(df0.index)+shift-1, str(cy[n][0]) + ' % ice'] = df0['% ice'].array
            dict_cs[s].loc[0:np.max(df1.index), str(cy[n][1]) + ' % ice'] = df1['% ice'].array
            dict_cs[s].loc[shift:np.max(df0.index)+shift-1, str(cy[n][0]) + ' density'] = df0['density'].array
            dict_cs[s].loc[0:np.max(df1.index), str(cy[n][1]) + ' density'] = df1['density'].array
            dict_cs[s].loc[shift:np.max(df0.index)+shift-1, str(cy[n][0]) + ' density_final'] = \
                df0['density_final'].array
            dict_cs[s].loc[0:np.max(df1.index), str(cy[n][1]) + ' density_final'] = df1['density_final'].array
            dict_cs[s].loc[:, str(cy[n][0]) + ' density_final_filled'] = dict_cs[s][str(cy[n][0]) + ' density_final']
            dict_cs[s].loc[:, str(cy[n][1]) + ' density_final_filled'] = dict_cs[s][str(cy[n][1]) + ' density_final']
        else:
            length = np.max([df0.index.max() + 1, df1.index.max() + 1 - shift])
            dict_cs[s].index = np.arange(1, length)
            dict_cs[s].set_index('depth (cm)', inplace=True)
            dict_cs[s].loc[-shift:np.max(df1.index) - shift, str(cy[n][1]) + ' material'] = df1['material']
            dict_cs[s].loc[0:np.max(df0.index), str(cy[n][0]) + ' material'] = df0['material']
            dict_cs[s].loc[-shift:np.max(df1.index) - shift, str(cy[n][1]) + ' % ice'] = df1['% ice']
            dict_cs[s].loc[0:np.max(df0.index), str(cy[n][0]) + ' % ice'] = df0['% ice']
            dict_cs[s].loc[-shift:np.max(df1.index) - shift, str(cy[n][1]) + ' density'] = df1['density']
            dict_cs[s].loc[0:np.max(df0.index), str(cy[n][0]) + ' density'] = df0['density']
            dict_cs[s].loc[-shift:np.max(df1.index) - shift, str(cy[n][1]) + ' density_final'] = df1['density_final']
            dict_cs[s].loc[0:np.max(df0.index), str(cy[n][0]) + ' density_final'] = df0['density_final']
            dict_cs[s].loc[:, str(cy[n][0]) + ' density_final_filled'] = df0['density_final']
            dict_cs[s].loc[:, str(cy[n][1]) + ' density_final_filled'] = df1['density_final']

        # add fill-densities into column 'density_final'
        # for proper calculation of m w.e. of SMB and SI, the column "year density_final_filled" needs to contain a
        # density of zero for depth range that were aboth the surface in year n ...
        if shift >= 0:
            dict_cs[s].loc[0:shift-1, str(cy[n][0]) + ' density_final_filled'] = 0
        else:
            dict_cs[s].loc[0:-shift-1, str(cy[n][1]) + ' density_final_filled'] = 0
        # ... and a density of ice for all depth range that was below the bottom of the core in year n+1
        length_row = np.max(dict_cs[s].loc[dict_cs[s][str(cy[n][0]) + ' density_final_filled'].notnull(),
                            str(cy[n][0]) + ' density_final_filled'].index)
        dict_cs[s].loc[length_row+1:np.max(dict_cs[s].index), str(cy[n][0]) + ' density_final_filled'] = rho_IE
        length_row = np.max(dict_cs[s].loc[dict_cs[s][str(cy[n][1]) + ' density_final_filled'].notnull(),
                            str(cy[n][1]) + ' density_final_filled'].index)
        dict_cs[s].loc[length_row+1:np.max(dict_cs[s].index), str(cy[n][1]) + ' density_final_filled'] = rho_IE
        # however, make sure that unmeasured (= depth range of unknown stratigraphy and density) are set to NaN
        # unmeasured araes are makred as 'unmeasured' in column 'year material'
        dict_cs[s].loc[dict_cs[s][str(cy[n][0]) + ' material'] == 'unmeasured',
                       str(cy[n][0]) + ' density_final_filled'] = np.nan
        dict_cs[s].loc[dict_cs[s][str(cy[n][1]) + ' material'] == 'unmeasured',
                       str(cy[n][1]) + ' density_final_filled'] = np.nan

        # calculate change in density over entire depth range
        dict_cs[s].loc[:, 'delta_density_final_filled'] = \
            dict_cs[s].loc[:, str(cy[n][1]) + ' density_final_filled'] - \
            dict_cs[s].loc[:, str(cy[n][0]) + ' density_final_filled']

        # calculate SMB in m w.e. and append to overview table
        if shift >= 0:
            smb_reference = dfcy[n].loc[s, str(cy[n][0]) + ' slab below surface (m)'] * 100 + shift
            dfcy[n].loc[s, 'climatic SMB (m w.e.)'] = \
                np.mean(dict_cs[s].loc[1:smb_reference, 'delta_density_final_filled'].array.astype(float)) \
                * smb_reference / 100 / 1000

        # calculate SI formation in m w.e. and append to overview table
        SI_range = sorted([int(np.round(dfcy[n].loc[s, str(cy[n][0]) + ' slab below surface (m)'] * 100 + shift)),
                           int(np.round(dfcy[n].loc[s, str(cy[n][1]) + ' slab below surface (m)'] * 100))])
        dfcy[n].loc[s, 'SI based on delta measured rho (m w.e.)'] = \
            np.mean(dict_cs[s].loc[SI_range[0]:SI_range[1], 'delta_density_final_filled'].array.astype(float)) \
            * (SI_range[1] - SI_range[0]) / 100 / 1000
        dfcy[n].loc[s, 'SI in SMB (%)'] = 100 * dfcy[n].loc[s, 'SI based on delta measured rho (m w.e.)'] / \
                                       dfcy[n].loc[s, 'climatic SMB (m w.e.)']

        # only for the depth interval of SI formation between years 1 and 2:
        # calculate mean density (rho) of firn/snow before SI formation (neglecting any areas of density = 0, which means
        # no snow yet) and rho after the SI formation

        # initially set all zero densities to np.nan
        t0 = dict_cs[s].loc[SI_range[0]:SI_range[1], str(cy[n][0]) + ' density_final_filled']
        t1 = dict_cs[s].loc[SI_range[0]:SI_range[1], str(cy[n][1]) + ' density_final_filled']
        t0[t0 == 0] = np.nan
        t1[t1 == 0] = np.nan

        # append values to the delta_density dictionary ['site', 'rho_before_SI_formation', 'rho_after_SI_formation']
        dd['site'] += len(t0) * [s + '_' + str(cy[n][0]) + '-' + str(cy[n][1])]
        dd['rho_before_SI_formation'] += list(t0.values)
        dd['rho_after_SI_formation'] += list(t1.values)

        # then calculate the mean densities, thereby ignoring all NaN values
        dfcy[n].loc[s, str(cy[n][0]) + ' mean rho before SI formation'] = np.nanmean(t0.array.astype(float))
        dfcy[n].loc[s, str(cy[n][1]) + ' mean rho after SI formation'] = np.nanmean(t1.array.astype(float))

    # **********  create the plots  ************
    plt.rcParams.update(plt_params)
    # plot 1: density and stratigraphy comparisons
    fig, ax = plt.subplots(nrows=1, ncols=len(cs[n])+1, figsize=((len(cs[n])+1) * 4.4, 15))
    for ns, s in enumerate(dict_cs):
        # one plot per pair of years is created that contains subplots for each stake location
        stake_plots(ns, s, cy[n], dict_cs[s], depth_fig, rho_IE)
    # finalize the figure and save to output
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    ax[len(cs[n])].legend(lines, labels, borderaxespad=0)
    ax[len(cs[n])].axis("off")
    plt.tight_layout()  # pad=0.1
    plt.savefig(outfolder + outfilename_head + '_SI-strat-comp_' + str(cy[n][0]) + '-' + str(cy[n][1])
                + outfilename_v + '.png')
    plt.close()

    # plot 2: delta density and SI areas highlighted
    fig, ax = plt.subplots(nrows=1, ncols=len(cs[n])+1, figsize=((len(cs[n])+1) * 4.4, 15))
    for ns, s in enumerate(dict_cs):
        # one plot per pair of years is created that contains subplots for each stake location
        Drho_plots(ns, s, cy[n], dict_cs[s], depth_fig, dfcy[n])
    # finalize the figure and save to output
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    ax[len(cs[n])].legend(lines, labels, borderaxespad=0)
    ax[len(cs[n])].axis("off")
    plt.tight_layout()  # pad=0.1
    plt.savefig(outfolder + outfilename_head + '_D-rho_' + str(cy[n][0]) + '-' + str(cy[n][1]) + outfilename_v + '.png')
    plt.close()

    # write data frames to output
    dfcy[n].to_excel(outfolder + outfilename_head + 'overview_' + str(cy[n][0]) + '-' + str(cy[n][1]) +
                     outfilename_v + '.xlsx')
    with pd.ExcelWriter(outfolder + outfilename_head + 'per_stake_comp_' + str(cy[n][0]) + '-' +
                        str(cy[n][1]) + outfilename_v + '.xlsx') as writer:
        for s in dict_cs:
            dict_cs[s].to_excel(writer, sheet_name=s)

# create dataframe for statistics on firn density prior and after SI formation
# this dataframe  contains all density values, only for the depth intervals where SI has formed
# between year n and n+1
delta_density = pd.DataFrame(dd)

# plot 3: some experiments with linear regressions
# first summarize all dfcy data in one table
# append the years back onto the stake names
for ny, y in enumerate(dfcy):
    y['SI stake'] = y.index + '_' + str(cy[ny][0]) + '-' + str(cy[ny][1])
    y.rename(columns={str(cy[ny][0]) + ' lat': '1st year lat',
                      str(cy[ny][1]) + ' lat': '2nd year lat',
                      str(cy[ny][0]) + ' lon': '1st year lon',
                      str(cy[ny][1]) + ' lon': '2nd year lon',
                      str(cy[ny][0]) + ' Z (m a.s.l.)': '1st year Z (m a.s.l.)',
                      str(cy[ny][1]) + ' Z (m a.s.l.)': '2nd year Z (m a.s.l.)',
                      str(cy[ny][0]) + ' stake above surface (m)': '1st year stake above surface (m)',
                      str(cy[ny][1]) + ' stake above surface (m)': '2nd year stake above surface (m)',
                      str(cy[ny][0]) + ' slab below surface (m)': '1st year slab below surface (m)',
                      str(cy[ny][1]) + ' slab below surface (m)': '2nd year slab below surface (m)'}, inplace=True)
dfca = pd.concat(dfcy, axis=0)
dfca.set_index('SI stake',inplace=True)

# calculate and plot the linear regressions
mask = ~np.isnan(dfca['climatic SMB (m w.e.)'].astype(float).array)

ivar = dfca.loc[mask, 'SI growth (m)'].astype(float)
dvar = dfca.loc[mask, 'SI based on delta measured rho (m w.e.)'].astype(float)
slope, intercept, r_value, p_value, std_err = stats.linregress(ivar, dvar)
linear_regression(ivar, dvar, 'SI growth (m)', 'SI based on $\\Delta \\rho$ (m w.e.)',
                  '_linregress_SIm-SImwe', slope, intercept, r_value**2, p_value, outfolder, outfilename_head,
                  outfilename_v)

ivar = dfca.loc[mask, '1st year slab below surface (m)'].astype(float)
dvar = dfca.loc[mask, 'SI growth (m)'].astype(float)
slope, intercept, r_value, p_value, std_err = stats.linregress(ivar, dvar)
linear_regression(ivar, dvar, '1st year slab below surface (m)', 'SI growth (m)',
                  '_linregress_InitialSlabdepth-SIgrowth', slope, intercept, r_value**2, p_value, outfolder,
                  outfilename_head, outfilename_v)

ivar = dfca.loc[mask, 'SI based on delta measured rho (m w.e.)'].astype(float)
dvar = dfca.loc[mask, 'climatic SMB (m w.e.)'].astype(float)
slope, intercept, r_value, p_value, std_err = stats.linregress(ivar, dvar)
linear_regression(ivar, dvar, 'SI based on $\\Delta \\rho$ (m w.e.)', 'climatic SMB (m w.e.)',
                  '_linregress_SImwe-SMB', slope, intercept, r_value**2, p_value, outfolder, outfilename_head,
                  outfilename_v)

# write the combined data frame to output
dfca.to_excel(outfolder + outfilename_head + 'overview_all_years_' + str(np.min(list(SI))) + '_to_'
              + str(np.max(list(SI))) + outfilename_v + '.xlsx')

# write the delta-density data frame to output
delta_density.to_excel(outfolder + outfilename_head + 'delta-density.xlsx')
