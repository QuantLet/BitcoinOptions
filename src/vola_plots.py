import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp
import matplotlib.dates as dates

from mpl_toolkits.mplot3d import Axes3D  
from mpl_toolkits.mplot3d import Axes3D

import pdb

def surface_plot(sub, day, xvar = 'moneyness', yvar = 'tau', zvar = 'mark_iv', xlabel = 'Moneyness', ylabel = 'Time-to-Maturity', zlabel = 'IV', out_dir = 'out/vola/'):
    # Fix Date and create 3d plot on
    # Moneyness, Tau, PK
    

    fig = plt.figure(figsize=(12, 7)) 
    ax = fig.gca(projection='3d')   # set up canvas for 3D plotting

    # Filter
    #sub = sub.loc[(sub.is_call == 1)]

    X = sub[xvar].tolist()
    Y = sub[yvar].tolist()
    Z = sub[zvar].tolist()
    
    # Interpolation
    # 30 days out must exist on the tau grid: that is 30/365
    tau_day = 1/365
    year = 1
    year_reps = 365

    plotx,ploty, = np.meshgrid(np.linspace(np.min(X),np.max(X),year_reps),\
                        np.linspace(tau_day, year, year_reps))
    
    plotz = interp.griddata((X,Y),Z,(plotx,ploty),method='linear')

    surf = ax.plot_surface(plotx,ploty,plotz,cstride=3,rstride=3,cmap=plt.cm.coolwarm, antialiased = True, linewidth = 0.5) 
    #surf = ax.plot_surface(X, Y, Z, cmap=plt.cm.coolwarm, linewidth=0.5,antialiased=True)  # creates 3D plot
    ax.view_init(30, 30)
    ax.set_xlabel(xlabel)  
    ax.set_ylabel(ylabel)  
    ax.set_zlabel(zlabel)  
    ax.set_zlim(0, 2)

    ax.xaxis.set_pane_color([1.0, 1.0, 1.0, 0])
    ax.yaxis.set_pane_color([1.0, 1.0, 1.0, 0])
    ax.zaxis.set_pane_color([1.0, 1.0, 1.0, 0])
    
    fig.colorbar(surf)
    surf.set_clim(vmin=0, vmax = 2)

    fname = str(out_dir) + pd.to_datetime(day).strftime('%Y-%m-%d') + '.png'
    plt.savefig(fname,transparent=False)

    return plotx, ploty, plotz


out_dir = 'out/3dplots/vola'
data_dir = 'out/filtered/'


data_files = os.listdir(data_dir)

dfs = []
for f in data_files:

    # Load File
    if 'filtered_' in f:
        curr_dat = pd.read_csv(data_dir + '/' + f)

        # Decompose Filename
        splt = f.split('_')
        date = pd.Timestamp(splt[1][:10])
        curr_dat['date'] = date

        sub = curr_dat.reset_index()
        strike, tau, iv = surface_plot(sub.loc[(sub.is_call == 1)], date)
