# Calculate IV vs Real Vola and visualize
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import json
import collections
import datetime
import numpy as np
import pandas as pd

from pymongo.uri_parser import _TLSINSECURE_EXCLUDE_OPTS

from brc import BRC

if __name__ == '__main__':
    brc = BRC()
    dat = brc._summary_stats_preprocessed(do_sample = False, write_to_file = True)

    # Order dict by keys and then plot IV, Vola over Time
    #ord = collections.OrderedDict(dat)
