"""
Generate Plots from main.py Output

- out/estimated_densities.csv Show SPD over time

Just use a normal, multiple plot for each day first
Then maybe for each instrument over time



    # 3D Stuff: Use to track a single instrument over time
    plt.figure()
    ax = plt.subplot(projection='3d')

    for i in range(3):

        ax.plot(x[i], np.ones(len(x[0])) * (i + 1), z[i], color='r', label = 'testmeup')

    plt.yticks(range(len(y)), y)
    ax.set_xlabel('Price')
    ax.set_zlabel('SPD')

    plt.show()

"""

import os
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from helpers import load_dict
import numpy as np
import pdb

# 2D Stuff: Use to show different SPDs/EPKs on a single day
def pricingkernel_plot(x, pk, date, tau, idx):
    '''
    Plots all the Pks on a single day

    '''


    fig=plt.figure()
    ax=fig.add_subplot(111)

    for i in idx:
        #plt.plot(x[i], s[i], label = str('spd') + str(tau[i]))
        #plt.plot(x[i], z[i], label = str('hd') + str(tau[i]))
        #pdb.set_trace()
        plt.plot(x[i], pk[i], label = str('Tau: ') + str(tau[i]))
        #pdb.set_trace()
        currdate = date[i]
        currtau = tau[i]
        currmaturity = maturitydate[i]
    
    plt.xlabel('Moneyness')
    plt.ylabel('Pricing Kernel')
    plt.title(str('Pricing Kernel on ') + str(currdate))
    #plt.ylim(0, 0.1)

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    #  Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig('pricingkernel/plots/pricingkernel' + str(currdate) + '.png')
    #plt.draw()
    plt.show()

def plot_pk_over_time(x, pk, date, tau, idx, maturitydate):
    '''
    Track all PKs instrumentwise (for a fixed maturitydate)

    '''
    
    fig=plt.figure()
    ax=fig.add_subplot(111)

    for i in idx:
        #plt.plot(x[i], s[i], label = str('spd') + str(tau[i]))
        #plt.plot(x[i], z[i], label = str('hd') + str(tau[i]))
        #pdb.set_trace()
        plt.plot(x[i], pk[i], label = str(tau[i])) #  + maturitydate[i]
        #currdate = date[i]
        #currtau = tau[i]
    
    plt.xlabel('Moneyness')
    plt.ylabel('Pricing kernel')
    plt.title(str('Pricing Kernel for fixed Maturity ') + str(maturitydate[i])) # maturitydate is all the same

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    #  Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title = 'Time to Maturity (Tau)')
    #plt.savefig('pricingkernel/plots/pk_dynamics' + str(maturitydate) + '.png')
    #plt.draw()
    plt.show()

    return None



est_spd_file = 'out/estimated_densities.csv'  #'out/estimated_densities.csv'
if os.path.isfile(est_spd_file):
    spds = load_dict(est_spd_file)
    print(spds)

    x, y, z, s = [], [], [], []
    date, tau = [], []
    pk = [] # pricingkernel
    maturitydate = []

    for key, (moneyness, gdom, gstar, spd) in spds.items():
            x.append(moneyness)
            y.append(key)
            z.append(gstar)
            s.append(spd)
            pk.append(spd/gstar)
            
            # Extract Date and Tau from Dict Key
            ysplit = str(key).split('_')
            date.append(ysplit[0]) # Day when instrument is traded
            tau.append(ysplit[1]) # Time to maturity as Decimal
            maturitydate.append(ysplit[2]) # Maturitydate of Instrument
            

    # Filter for different dates and run this whole thing
    unique_dates = np.unique(date)
    unique_maturities = np.unique(maturitydate)
    
    for i in range(len(unique_dates)): 
        # Tracking all Maturities which are traded on one day
        idx = [index for index, element in enumerate(date) if element == unique_dates[i]]
        if len(idx) > 1:
            pricingkernel_plot(x, pk, date, tau, idx)

    for i in range(len(unique_maturities)): 
        # Find the index where all maturitydates are the same
        # So that we can compare instruments which have equal Maturity, 
        # Meaning we can track one fixed instrument over time
        idx = [index for index, element in enumerate(maturitydate) if element == unique_maturities[i]]

        if len(idx) > 1:
            plot_pk_over_time(x, pk, date, tau, idx, maturitydate)

         
