"""
Create 3D Pricing Kernel Plots

Visualization: https://stackoverflow.com/questions/35259285/plotting-three-lists-as-a-surface-plot-in-python-using-mplot3d

"""
from mpl_toolkits.mplot3d import Axes3D  
import matplotlib.pyplot as plt
import pdb
import os
import pandas as pd
import numpy as np
import scipy.interpolate as interp
import matplotlib.dates as dates

out_dir = 'out/3dplots'
data_dir = 'out/movies'


data_files = os.listdir(data_dir)

dfs = []
for f in data_files:

    # Load File
    if 'btc_pk' in f:
        curr_dat = pd.read_csv(data_dir + '/' + f)

        # Decompose Filename
        splt = f.split('_')
        tau = float(splt[2])
        date = pd.Timestamp(splt[3].split('.')[0])
        curr_dat['tau'] = tau
        curr_dat['date'] = date

        # Clusters for Tau: 1, 2, 4, 8 Weeks
        # Constant
        curr_dat['nweeks'] = 0
        floatweek = 1/52
        curr_dat['nweeks'][(curr_dat['tau'] <= floatweek)] = 1
        curr_dat['nweeks'][(curr_dat['tau'] > floatweek) & (curr_dat['tau'] <= 2 * floatweek)] = 2
        curr_dat['nweeks'][(curr_dat['tau'] > 2 * floatweek) & (curr_dat['tau'] <= 3 * floatweek)] = 3
        curr_dat['nweeks'][(curr_dat['tau'] > 3 * floatweek) & (curr_dat['tau'] <= 4 * floatweek)] = 4
        curr_dat['nweeks'][(curr_dat['tau'] > 4 * floatweek) & (curr_dat['tau'] <= 8 * floatweek)] = 8

        dfs.append(curr_dat)

bound = pd.concat(dfs).reset_index()

# Fix Tau and create 3d plot on
# Moneyness, Observation Date, PK
for d in bound['nweeks'].unique():

    sub = bound.loc[bound['nweeks'] == d].reset_index()
    sub['date'] = pd.to_datetime(sub['date'])

    # Create a spacing where the first date is 1, ..., until last date is n
    # use date range, then find the respective index :)
    mindate = sub['date'].min()
    maxdate = sub['date'].max()
    date_rng = pd.date_range(mindate, maxdate)
    sub['date_idx'] = list(map(lambda x: np.where(x == date_rng)[0][0], sub['date']))

    for tgt, tgtlabel in zip(['pk', 'spdy'], ['Pricing Kernel', 'State Price Density']):

        fig = plt.figure(figsize=(12, 7)) 
        ax = fig.gca(projection='3d')   # set up canvas for 3D plotting

        sub = sub.loc[(sub[tgt] >= 0) & (sub[tgt] <= 10)]

        X = sub['m'].tolist()
        Y = sub['date_idx'].tolist()
        Z = sub[tgt].tolist()
        
        # As a helper for plotz
        linsteps = 100
        plotx,ploty, = np.meshgrid(np.linspace(np.min(X),np.max(X),linsteps),np.linspace(np.min(Y),np.max(Y),linsteps))


        #https://stackoverflow.com/questions/60589494/need-help-for-3d-plot-by-datetime-series-in-matplotlib
        
        plotz = interp.griddata((X,Y),Z,(plotx,ploty),method='linear')
        surf = ax.plot_surface(plotx,ploty,plotz,cstride=3,rstride=3,cmap=plt.cm.coolwarm, antialiased = True, linewidth = 0.5) 
        #surf = ax.plot_surface(X, Y, Z, cmap=plt.cm.coolwarm, linewidth=0.5,antialiased=True)  # creates 3D plot
        ax.view_init(30, 30)
        ax.set_xlabel('Moneyness')  
        ax.set_ylabel('Observation Date')  
        ax.set_zlabel(tgtlabel)  
        ax.set_zlim(0, 10)
	
        ax.xaxis.set_pane_color([1.0, 1.0, 1.0, 0])
        ax.yaxis.set_pane_color([1.0, 1.0, 1.0, 0])
        ax.zaxis.set_pane_color([1.0, 1.0, 1.0, 0])

        nsteps = 8
        ysteps = round(len(date_rng) / nsteps)
        yl = []
        val = 0
        for i in range(nsteps - 1):
            yl.append(date_rng[val].strftime('%Y-%m-%d'))
            val += ysteps
        yl.append(date_rng[-1].strftime('%Y-%m-%d'))
        ax.set_yticklabels(yl)
        
        fig.colorbar(surf)
        surf.set_clim(vmin=0, vmax = 5)

        fname = 'out/transparent3dplots/SortedByTau/' + tgtlabel + '_over_time_nweeks=' + str(d) + '.png'
        plt.savefig(fname,transparent=True)


"""
# Fix Date and create 3d plot on
# Moneyness, Tau, PK
for d in bound['date'].unique():

    sub = bound.loc[bound['date'] == d]

    fig = plt.figure(figsize=(12, 7)) 
    ax = fig.gca(projection='3d')   # set up canvas for 3D plotting


    sub = sub.loc[(sub['pk'] >= 0) & (sub['pk'] <= 10)]

    X = sub['m'].tolist()
    Y = sub['tau'].tolist()
    Z = sub['pk'].tolist()
    
    plotx,ploty, = np.meshgrid(np.linspace(np.min(X),np.max(X),50),\
                           np.linspace(np.min(Y),np.max(Y),50))
    
    plotz = interp.griddata((X,Y),Z,(plotx,ploty),method='linear')

    surf = ax.plot_surface(plotx,ploty,plotz,cstride=3,rstride=3,cmap=plt.cm.coolwarm, antialiased = True, linewidth = 0.5) 
    #surf = ax.plot_surface(X, Y, Z, cmap=plt.cm.coolwarm, linewidth=0.5,antialiased=True)  # creates 3D plot
    ax.view_init(30, 30)
    ax.set_xlabel('Moneyness')  
    ax.set_ylabel('Time-to-Maturity')  
    ax.set_zlabel('Pricing Kernel')  
    ax.set_zlim(0, 10)
    
    fig.colorbar(surf)
    surf.set_clim(vmin=0, vmax = 5)

    fname = 'out/3dplots/' + pd.to_datetime(d).strftime('%Y-%m-%d') + '.png'
    plt.savefig(fname,transparent=True)


"""
