# Calculate IV vs Real Vola and visualize
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import json
import collections
import datetime
import numpy as np
import pandas as pd
import pdb

from pymongo.uri_parser import _TLSINSECURE_EXCLUDE_OPTS

from brc import BRC

def compute_vola(x, window):
    # Annualizing daily SD should be ok aswell

    #window = 21  # trading days in rolling windoww
    dpy = 365 #252  # trading days per year
    ann_factor = dpy / window

    df = pd.DataFrame({'price' : x})
    df['log_rtn'] = np.log(df['price']).diff()

    # Var Swap (returns are not demeaned)
    df['real_var'] = np.square(df['log_rtn']).rolling(window).sum() * ann_factor
    df['real_vol'] = np.sqrt(df['real_var'])

    # Classical (returns are demeaned, dof=1)
    #df['real_var'] = df['log_rtn'].rolling(window).var() * ann_factor
    #df['real_vol'] = np.sqrt(df['real_var'])
    return df


if __name__ == '__main__':
    brc = BRC()
    dat = brc._mean_iv(do_sample = False, write_to_file = True) # Execute

    # Order dict by keys and then plot IV, Vola over Time
    ord = collections.OrderedDict(dat)


    # @Todo: Filter for volume
    # Dump each element in a dict and save as JSON
    asks = []
    bids = []
    underlying = []
    dates = []
    s = sorted(ord.items())
    for ele in s:
        print(ele)
        currkey, currvalue = ele[0], ele[1]
        #dates.append(currkey)
        dates.append(datetime.datetime.strptime(currkey, '%Y-%m-%d'))
        asks.append(currvalue['ask'])
        bids.append(currvalue['bid'])
        underlying.append(currvalue['underlying'])

    dailyvola = compute_vola(underlying, 7) #* 100
    rollingvola = compute_vola(underlying, 21)
        
    # Display only Monthly Dates
    locator = mdates.MonthLocator()  # every month
    # Specify the format - %b gives us Jan, Feb...
    fmt = mdates.DateFormatter('%b')

    df = pd.DataFrame({'date': dates,
                        'asks': [a/100 for a in asks],
                        'bids': [b/100 for b in bids],
                        'daily': dailyvola['real_vol'],
                        'rolling': rollingvola['real_vol']})
    
    sub = df.loc[(df['date'] >= pd.Timestamp('2021-01-01')) & (df['date'] <= pd.Timestamp('2021-12-31'))]
    print(sub['date'].unique())
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(sub['date'], sub['asks'], label = 'Ask IV')
    plt.plot(sub['date'], sub['bids'], label = 'Bid IV')
    plt.plot(sub['date'], sub['daily'], label = '7 Day \nVolatility')
    plt.plot(sub['date'], sub['rolling'], label = '21 Day \nVolatility')
    plt.title('Implied vs. Realized Volatility')
    #plt.ylabel('IV')
    plt.xlabel('Time')
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    #  Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Only Months
    X = plt.gca().xaxis
    X.set_major_locator(locator)
    # Specify formatter
    X.set_major_formatter(fmt)

    plt.savefig('/Users/julian/src/up/spd/pricingkernel/plots/iv_comparison.png', transparent = True)
    plt.show()
