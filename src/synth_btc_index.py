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

#from brc import BRC

if __name__ == '__main__':
    """
    brc = BRC()
    dat = brc.synth_btc(do_sample = False, write_to_file = True)

    # Order dict by keys and then plot IV, Vola over Time
    ord = collections.OrderedDict(dat)

    # @Todo: Filter for volume
    # Dump each element in a dict and save as JSON
    underlying = []
    dates = []
    s = sorted(ord.items())
    for ele in s:
        print(ele)
        currkey, currvalue = ele[0], ele[1]
        #dates.append(currkey)
        dates.append(datetime.datetime.strptime(currkey, '%Y-%m-%d-%H-%M'))
        underlying.append(currvalue['underlying'])
    """

    df = pd.read_csv('Data/DeribitIndex.csv')
    df = df.rename(columns = {'Price': 'Index'})
    df = df.sort_values('Date')

    # Display only Yearly Dates
    locator = mdates.YearLocator()  # every month
    # Specify the format - %b gives us Jan, Feb...
    fmt = mdates.DateFormatter('%Y')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    #plt.plot(dates, underlying)
    plt.plot(df['Date'], df['Index'], color = 'black')
    #plt.title('Synthetic BTC Index')
    plt.xlabel('Time')

    # Only Months
    X = plt.gca().xaxis
    X.set_major_locator(locator)
    # Specify formatter
    #X.set_major_formatter(fmt)

    plt.savefig('/Users/julian/src/spd/pricingkernel/plots/synthetic_btc_index2.png', transparent = True)
    plt.show()
