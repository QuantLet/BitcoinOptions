from doctest import DocFileSuite
from lib2to3.pgen2.pgen import DFAState
import sshtunnel
import pymongo
import pprint
import datetime
import pandas as pd
import os

# For Deribit
import asyncio
import websockets
import json

# Avg IV per Day
from bson.json_util import dumps # dump Output
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import numpy as np

import pdb

def trisurf(x, y, z, xlab, ylab, zlab, filename, blockplots):
    # This is the data we have: vola ~ tau + moneyness
    # https://stackoverflow.com/questions/9170838/surface-plots-in-matplotlib
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel('Moneyness')
    ax.set_ylabel('Tau')
    ax.set_zlabel('IV')
    #x, y = np.meshgrid(x, y)

    df = pd.DataFrame({'x': x, 'y': y, 'z': z})
    surf = ax.plot_trisurf(df.x, df.y, df.z, cmap=plt.cm.jet, linewidth=0.1)
    fig.colorbar(surf, shrink=0.5, aspect=5)   
    plt.title('Empirical Vola Smile') 
    plt.savefig(filename)
    plt.draw()
    #plt.show(block = blockplots)

def vola_surface_interpolated(df, out_path = 'out/volasurface/', moneyness_min = 0.7, moneyness_max = 1.3):


    # Adjust date
    df['date_short'] = df['date'].apply(lambda x: datetime.datetime.strftime(x, '%Y-%m-%d'))
    
    # Adjust mark iv
    df['mark_iv'] = df['mark_iv']/100

    print('Using Calls only for Vola Surface. Restricting Moneyness to ', moneyness_min, ' - ',moneyness_max)
    for d in df['date_short'].unique():
        
        sub = df.loc[df['date_short'] == d]

        fig = plt.figure(figsize=(12, 7)) 
        ax = fig.gca(projection='3d')   # set up canvas for 3D plotting

        sub = sub.loc[(sub['mark_iv'] >= 0) & (sub['mark_iv'] <= 3) & (sub['is_call'] == 1) & (sub['moneyness'] >= moneyness_min) & (sub['moneyness'] <= moneyness_max)]

        X = sub['moneyness'].tolist()
        Y = sub['tau'].tolist()
        Z = sub['mark_iv'].tolist()
        
        plotx,ploty, = np.meshgrid(np.linspace(np.min(X),np.max(X),50),\
                            np.linspace(np.min(Y),np.max(Y),50))
        
        plotz = interp.griddata((X,Y),Z,(plotx,ploty),method='linear')

        surf = ax.plot_surface(plotx,ploty,plotz,cstride=3,rstride=3,cmap=plt.cm.coolwarm, antialiased = True, linewidth = 0.5) 
        #surf = ax.plot_surface(X, Y, Z, cmap=plt.cm.coolwarm, linewidth=0.5,antialiased=True)  # creates 3D plot
        ax.view_init(30, 30)
        ax.set_xlabel('Moneyness')  
        ax.set_ylabel('Time-to-Maturity')  
        ax.set_zlabel('IV')  
        ax.set_zlim(0, 3)
        
        fig.colorbar(surf)
        surf.set_clim(vmin=0, vmax = 3)

        fname = out_path + pd.to_datetime(d).strftime('%Y-%m-%d') + '.png'
        plt.savefig(fname,transparent=True)


def decompose_instrument_name(_instrument_names, tradedate, round_tau_digits = 4):
    """
    Input:
        instrument names, as e.g. pandas column / series
        in this format: 'BTC-6MAR20-8750-C'
    Output:
        Pandas df consisting of:
            Decomposes name of an instrument into
            Strike K
            Maturity Date T
            Type (Call | Put)
    """
    try:
        _split = _instrument_names.str.split('-', expand = True)
        _split.columns = ['base', 'maturity', 'strike', 'is_call'] 
        
        # call == 1, put == 0 in is_call
        _split['is_call'] = _split['is_call'].replace('C', 1)
        _split['is_call'] = _split['is_call'].replace('P', 0)

        # Calculate Tau; being time to maturity
        #Error here: time data '27MAR20' does not match format '%d%b%y'
        _split['maturitystr'] = _split['maturity'].astype(str)
        # Funny Error: datetime does recognize MAR with German AE instead of A
        maturitydate        = list(map(lambda x: datetime.datetime.strptime(x, '%d%b%y') + datetime.timedelta(hours = 8), _split['maturitystr'])) # always 8 o clock
        reference_date      = tradedate.dt.date #list(map(lambda x: x.dt.date, tradedate))#tradedate.dt.date # Round to date, else the taus are all unique and the rounding creates different looking maturities
        Tdiff               = pd.Series(maturitydate).dt.date - reference_date #list(map(lambda x: x - reference_date, maturitydate))
        Tdiff               = Tdiff[:len(maturitydate)]
        sec_to_date_factor   = 60*60*24
        _Tau                = list(map(lambda x: (x.days + (x.seconds/sec_to_date_factor)) / 365, Tdiff))#Tdiff/365 #list(map(lambda x: x.days/365, Tdiff)) # else: Tdiff/365
        _split['tau']       = _Tau
        _split['tau']       = round(_split['tau'], round_tau_digits)

        # Strike must be float
        _split['strike'] =    _split['strike'].astype(float)

        # Add maturitydate for trading simulation
        _split['maturitydate_trading'] = maturitydate
        _split['days_to_maturity'] = list(map(lambda x: x.days, Tdiff))

        print('\nExtracted taus: ', _split['tau'].unique(), '\nExtracted Maturities: ',_split['maturity'].unique())

    except Exception as e:
        print('Error in Decomposition: ', e)
    finally:
        return _split


# Todo: connection doesnt stop atm
class BRC:
    def __init__(self):
        
        self.MONGO_HOST = '35.205.115.90'
        self.MONGO_DB   = 'cryptocurrency'
        self.MONGO_USER = 'winjules2'
        self.MONGO_PASS = ''
        self.PORT = 27017
        #self.server_started = self._start()
        #if self.server_started:
        print('\nGoing Local')
        self.client = pymongo.MongoClient('localhost', 27017) 
        self.db = self.client[self.MONGO_DB]
        self.collection_name = 'deribit_orderbooks'#'threemonths'
        self.collection = self.db[self.collection_name]#['deribit_orderbooks']
        print('using collection: ', self.collection_name)
        self._generate_stats()
        #self._mean_iv(do_sample = False, write_to_file = False)
        self.update_db = self.client['cryptocurrency_updated']
        self.update_collection = self.update_db['deribit_orderbooks_updated_20220427']

    def _server(self):
        self.server = sshtunnel.SSHTunnelForwarder(
            self.MONGO_HOST,
            ssh_username=self.MONGO_USER,
            ssh_password=self.MONGO_PASS,
            remote_bind_address=('127.0.0.1', self.PORT)
            )
        return self.server

    def _start(self):
        #self.server = self._server()
        #self.server.start()
        return True

    def _stop(self):
        #self.server.stop()
        return True

    def _filter_by_timestamp(self, starttime, endtime):        
        """
        Example:
        starttime = datetime.datetime(2020, 4, 19, 0, 0, 0)
        endtime = datetime.datetime(2020, 4, 20, 0, 0, 0)
        """
        ts_high     = round(endtime.timestamp() * 1000)
        ts_low      = round(starttime.timestamp() * 1000)
        return ts_high, ts_low

    def _generate_stats(self):
        print('\n Established Server Connection')
        print('\n Available Collections: ', self.db.collection_names())
        print('\n Size in GB: ', self.db.command('dbstats')['dataSize'] * 0.000000001) 
        print('\n Object Count: ', self.collection.count())
        
        # Get first and last element:
        last_ele = self.collection.find_one(
        sort=[( '_id', pymongo.DESCENDING )]
        )

        first_ele = self.collection.find_one(
            sort = [('_id', pymongo.ASCENDING)]
        )
        
        self.first_day = datetime.datetime.fromtimestamp(round(first_ele['timestamp']/1000))
        self.last_day  = datetime.datetime.fromtimestamp(round(last_ele['timestamp']/1000))
        self.first_day_timestamp = first_ele['timestamp']
        self.last_day_timestamp  = last_ele['timestamp']

        print('\n First day: ', self.first_day, ' \n Last day: ', self.last_day)

    def synth_btc(self, do_sample, write_to_file):
        """
        Extract high frequency prices of Deribit synthetic btc price
        """
        print('extracting synth index')
        if do_sample:
            pipeline = [
                {
                    "$sample": {"size": 120000},
                },

                {
                    '$group':{"_id": { "$dateToString": { "format": "%Y-%m-%d-%H-%M", "date": {'$toDate': '$timestamp' } }},
                            'avg_btc_price': {'$avg': '$underlying_price'}
                        }
                }

            ]
        else:
            pipeline = [
                {
                    '$group':{"_id": { "$dateToString": { "format": "%Y-%m-%d-%H-%m", "date": {'$toDate': '$timestamp' } }},
                            'avg_btc_price': {'$avg': '$underlying_price'}
                        }
                }

            ]
        
        print('Pumping the Pipeline')
        synth_per_minute = self.collection.aggregate(pipeline)

        # Save Output as JSON
        a = list(synth_per_minute)
        j = dumps(a, indent = 2)
        
        if write_to_file:
            # Dump each element in a dict and save as JSON
            fname = "/Users/julian/src/up/spd/out/synth_btc_per_minute.JSON"
            print('Writing Output to ', fname)
            out = {}
            for ele in a:
                out[ele['_id']] = {'underlying': ele['avg_btc_price']}

            with open(fname,"w") as f:
                json.dump(out, f)
        else:
            out = json.loads(j)

        return out


    def _mean_iv(self, do_sample = False, write_to_file = False):
        """
        Task: 
            Select Average IV (for bid and ask) and group by day!

        Paste in the pipeline to have a sample for debugging
            {
                "$sample": {"size": 10},
            },
        """
        print('init mean iv')

        if do_sample:
            pipeline = [
                {
                    "$sample": {"size": 120000},
                },

                # Try to Subset / WHERE Statement
                {'$match': {'bid_iv': {"$gt": 0.02}}},

                {
                    '$group':{"_id": { "$dateToString": { "format": "%Y-%m-%d", "date": {'$toDate': '$timestamp' } }},
                            'avg_ask': {'$avg': '$ask_iv'},
                            'avg_bid': {'$avg': '$bid_iv'},
                            'avg_btc_price': {'$avg': '$underlying_price'}

                        }
                }

            ]
        else:
            pipeline = [

                # Bid cannot be 0 in IV. This would distort the mean.
                {'$match': {'bid_iv': {"$gt": 0.02}}},

                {
                    '$group':{"_id": { "$dateToString": { "format": "%Y-%m-%d", "date": {'$toDate': '$timestamp' } }},
                            'avg_ask': {'$avg': '$ask_iv'},
                            'avg_bid':{'$avg': '$bid_iv'},
                            'avg_btc_price': {'$avg': '$underlying_price'}

                        }
                }

            ]
        
        print('Pumping the Pipeline')
        avg_iv_per_day = self.collection.aggregate(pipeline)
        #print(list(avg_iv_per_day))

        # Save Output as JSON
        a = list(avg_iv_per_day)
        j = dumps(a, indent = 2)

        if write_to_file:
            # Dump each element in a dict and save as JSON
            fname = "/Users/julian/src/spd/out/volas_per_day.JSON"
            print('Writing Output to ', fname)
            out = {}
            for ele in a:
                out[ele['_id']] = {'ask': ele['avg_ask'],
                                    'bid': ele['avg_bid'],
                                    'underlying': ele['avg_btc_price']}

            with open(fname,"w") as f:
                json.dump(out, f)
        else:
            out = json.loads(j)

        return out
    
    def load_update_collection(self, do_sample = False):
        """
        Load all elements of the update_collection
        """
        if do_sample:
            print('using a sample')
            documents = list(self.update_collection.find().limit(10000))
        else:
            documents = list(self.update_collection.find({}, no_cursor_timeout=True))

        collection_data = []
        for doc in documents:
            collection_data.append(doc)

        dat = pd.DataFrame(collection_data).drop_duplicates().dropna()
        return dat

    def plot_iv_surface(self, moneyness_min = 0.9, moneyness_max = 1.1, min_rows = 100):
        """

        """

        ivdat = self.load_update_collection(do_sample = True)

        # Prepare date string
        ivdat['date_short'] = ivdat['date'].apply(lambda x: datetime.datetime.strftime(x, '%Y-%m-%d'))
        
        # Adjust mark iv
        ivdat['mark_iv'] = ivdat['mark_iv']/100

        # Filter
        filtered = ivdat[(ivdat['moneyness'] > moneyness_min) & (ivdat['moneyness'] < moneyness_max)]

        # Loop over dates
        unique_dates = filtered['date_short'].unique()
        for currdate in unique_dates:
            print(currdate)
            sub = filtered[(filtered['date_short'] == currdate)]

            if sub.shape[0] > min_rows:
                # Plot
                trisurf(sub['moneyness'], sub['tau'], sub['mark_iv'], 'moneyness', 'tau', 'vola', 'ivsurfaces/empirical_vola_smile_' + currdate, False)

        return None

    def _summary_stats_preprocessed(self, otm_thresh = 0.7, itm_thresh = 1.3, do_sample = True, write_to_file = True):
        """
        Task: 
            Summary Statistics for:
                Moneyness, Implied Volatility, Amount of Calls, Amount of Puts
                Per time-to-maturity in weeks (1,2,4,8)


        Paste in the pipeline to have a sample for debugging
            {
                "$sample": {"size": 10},
            },

        # Regex like Query
        db.users.find({'name': {'$regex': 'sometext'}})

        db.deribit_orderbooks.findOne({'instrument_name': {'$regex': '(?<=\-)(.*?)(?=\-)'}})
        """
        print('Pumping the Pipeline')

        """
        time_variable = 'timestamp' # timestamp

        collection_data = []
        documents = list(self.update_collection.find({time_variable:{'$exists': True}}).sort(time_variable).limit(1000))
        for doc in documents:
            collection_data.append(doc)

        while True:
            ids = set(doc['_id'] for doc in documents)
            cursor = self.update_collection.find({time_variable: {'$gte': documents[-1][time_variable]}})
            documents = list(cursor.limit(1000).sort(time_variable))
            if not documents:
                break  # All done.
            for doc in documents:
                # Avoid overlaps
                if doc['_id'] not in ids:
                    collection_data.append(doc)
        """

        dat = self.load_update_collection(do_sample = False)

        #dat = pd.DataFrame(collection_data)
        assert(dat.shape[0] != 0)
        df  = dat[['_id', 'strike', 'is_call', 'tau', 'mark_iv', 'date', 'moneyness']]    
        #df['date_short'] = df['date'].apply(lambda x: datetime.datetime.strftime(x, '%Y-%m-%d'))

        # Clusters for Tau: 1, 2, 4, 8 Weeks
        df['nweeks'] = 0
        floatweek = 1/52
        df['nweeks'][(df['tau'] <= floatweek)] = 1
        df['nweeks'][(df['tau'] > floatweek) & (df['tau'] <= 2 * floatweek)] = 2
        df['nweeks'][(df['tau'] > 2 * floatweek) & (df['tau'] <= 3 * floatweek)] = 3
        df['nweeks'][(df['tau'] > 3 * floatweek) & (df['tau'] <= 4 * floatweek)] = 4
        df['nweeks'][(df['tau'] > 4 * floatweek) & (df['tau'] <= 8 * floatweek)] = 8

        # Table 1
        # Amount of OTM and ITM Options
        table1 = df[(df['moneyness'] < itm_thresh) & (df['moneyness'] > otm_thresh)].describe()

        # Moneyness and implied volatility of valid 50ETF options.
        table2 = df[['mark_iv', 'moneyness', 'nweeks']].groupby(['nweeks']).describe()

        # Table3 of Bitcoin gross returns
        #table3 = df[[]] # last per day
        # Just calculate this from the actual time series of index returns


        # options used for constructing implied volatility curves.
        table4 = df[['is_call', 'nweeks']].groupby(['nweeks']).describe()

        if write_to_file:
            table1.to_csv('summary_statistics_table1.csv')
            table2.to_csv('summary_statistics_table2.csv')
            table4.to_csv('summary_statistics_table4.csv')

        try:
            vola_surface_interpolated(df, out_path = 'out/volasurface/', moneyness_min = otm_thresh, moneyness_max = itm_thresh)
            vola_surface_interpolated(df, out_path = 'out/volasurface/restricted/', moneyness_min = 0.95, moneyness_max = 1.05)
        except Exception as e:
            print(e)
            #pdb.set_trace()

        return None

    
    def _summary_stats(self, otm_thresh, itm_thresh, do_sample = True, write_to_file = False):
        """
        Task: 
            Summary Statistics for:
                Moneyness, Implied Volatility, Amount of Calls, Amount of Puts
                Per time-to-maturity in weeks (1,2,4,8)


        Paste in the pipeline to have a sample for debugging
            {
                "$sample": {"size": 10},
            },

        # Regex like Query
        db.users.find({'name': {'$regex': 'sometext'}})

        db.deribit_orderbooks.findOne({'instrument_name': {'$regex': '(?<=\-)(.*?)(?=\-)'}})
        """
        print('init mean iv')
        # db.deribit_orderbooks.regexFind({'ext':{'input':'$instrument_name', 'regex':'(?<=\-)(.*?)(?=\-)'})}

        
        print('Pumping the Pipeline')
        #cursor = self.collection.find(no_cursor_timeout=True)
        # Save Output as JSON
        #collection_data = [document for document in cursor]

        collection_data = []
        documents = list(self.collection.find().sort('timestamp').limit(1000))
        for doc in documents:
            collection_data.append(doc)

        while True:
            ids = set(doc['_id'] for doc in documents)
            cursor = self.collection.find({'timestamp': {'$gte': documents[-1]['timestamp']}})
            documents = list(cursor.limit(1000).sort('timestamp'))
            if not documents:
                break  # All done.
            for doc in documents:
                # Avoid overlaps
                if doc['_id'] not in ids:
                    collection_data.append(doc)


        dat = pd.DataFrame(collection_data)

        assert(dat.shape[0] != 0)

        # Convert dates, utc
        dat['date'] = list(map(lambda x: datetime.datetime.fromtimestamp(x/1000), dat['timestamp']))
        dat_params  = decompose_instrument_name(dat['instrument_name'], dat['date'])
        dat         = dat.join(dat_params)

        # Drop all spoofed observations - where timediff between two orderbooks (for one instrument) is too small
        dat['timestampdiff'] = dat['timestamp'].diff(1)
        dat = dat[(dat['timestampdiff'] > 2)]

        dat['interest_rate'] = 0 # assumption here!
        dat['index_price']   = dat['index_price'].astype(float)

        # To check Results after trading 
        dates                       = dat['date']
        dat['strdates']             = dates.dt.strftime('%Y-%m-%d') 
        maturitydates               = dat['maturitydate_trading']
        dat['maturitydate_char']    = maturitydates.dt.strftime('%Y-%m-%d')

        # Calculate mean instrument price
        bid_instrument_price = dat['best_bid_price'] * dat['underlying_price'] 
        ask_instrument_price = dat['best_ask_price'] * dat['underlying_price']
        dat['instrument_price'] = (bid_instrument_price + ask_instrument_price) / 2

        # Prepare for moneyness domain restriction (0.8 < m < 1.2)
        dat['moneyness']    = round(dat['strike'] / dat['index_price'], 2)
        df                  = dat[['_id', 'index_price', 'strike', 'interest_rate', 'maturity', 'is_call', 'tau', 'mark_iv', 'date', 'moneyness', 'instrument_name', 'days_to_maturity', 'maturitydate_char', 'timestamp', 'underlying_price', 'instrument_price']]    
        
        # Select Tau and Maturity (Tau is rounded, prevent mix up!)
        unique_taus = df['tau'].unique()
        unique_maturities = df['maturity'].unique()
        
        # Save Tau-Maturitydate combination
        #tau_maturitydate[curr_day.strftime('%Y-%m-%d')] = (unique_taus,)
        
        unique_taus.sort()
        unique_taus = unique_taus[(unique_taus > 0) & (unique_taus < 0.25)]
        print('\nunique taus: ', unique_taus,
                '\nunique maturities: ', unique_maturities)

        # Clusters for Tau: 1, 2, 4, 8 Weeks
        df['nweeks'] = 0
        floatweek = 1/52
        df['nweeks'][(df['tau'] <= floatweek)] = 1
        df['nweeks'][(df['tau'] > floatweek) & (df['tau'] <= 2 * floatweek)] = 2
        df['nweeks'][(df['tau'] > 2 * floatweek) & (df['tau'] <= 3 * floatweek)] = 3
        df['nweeks'][(df['tau'] > 3 * floatweek) & (df['tau'] <= 4 * floatweek)] = 4
        df['nweeks'][(df['tau'] > 4 * floatweek) & (df['tau'] <= 8 * floatweek)] = 8

        # Table 1
        # Amount of OTM and ITM Options
        table1 = df[(df['moneyness'] < itm_thresh) & (df['moneyness'] > otm_thresh)]

        # Moneyness and implied volatility of valid 50ETF options.
        table2 = df[['mark_iv', 'moneyness', 'nweeks']].groupby(['nweeks']).describe()

        # Table3 of Bitcoin gross returns is calculated in the .Rmd

        # options used for constructing implied volatility curves.
        table4 = df[['is_call', 'nweeks']].groupby(['nweeks']).describe()

        if write_to_file:
            table1.to_csv('summary_statistics_table1.csv')
            table2.to_csv('summary_statistics_table2.csv')
            table4.to_csv('summary_statistics_table4.csv')

        return None

    # Deribit
    def create_msg(self, _tshigh, _tslow):
        # retrieves constant interest rate for time frame
        self.msg = \
        {
        "jsonrpc" : "2.0",
        "id" : None,
        "method" : "public/get_funding_rate_value",
        "params" : {
            "instrument_name" : "BTC-PERPETUAL",
            "start_timestamp" : _tslow,
            "end_timestamp" : _tshigh
            }
        }
        return None

    async def call_api(self):
        async with websockets.connect('wss://test.deribit.com/ws/api/v2') as websocket:
            await websocket.send(json.dumps(self.msg))
            while websocket.open:
                print(self.msg)
                response = await websocket.recv()
                # do something with the response...
                self.response = json.loads(response)
                self.historical_interest_rate = round(self.response['result'], 4)
                return None

    def _run(self, starttime, endtime, download_interest_rates, download_historical_iv, insert_to_db):
        
        #server_started = self._start()
        
    
        try:

            download_starttime = datetime.datetime.now()

            ts_high, ts_low = self._filter_by_timestamp(starttime, endtime)


            # Insert sampling here!
            #res = self.collection.find({ "$and": [{'timestamp': {"$lt": ts_high}},
           #                     {'timestamp': {"$gte": ts_low}}]}, no_cursor_timeout=True)#.sort('timestamp')
            """
            pipe = [ 
                {
                    "$sample": {"size": 2000000},
                },    
                {"$match": { "$and": [{'timestamp': {"$lt": ts_high}},
                                {'timestamp': {"$gte": ts_low}}]}
                }#.sort('timestamp')
                ]
            
            res = self.collection.aggregate(pipe,  allowDiskUse=True)
            """
            _filter = { "$and": [{'timestamp': {"$lt": ts_high}},
                                {'timestamp': {"$gte": ts_low}}]}
            res = self.collection.find(_filter)#.sort('timestamp')

            #nresults = res.count()
            #if nresults == 0:
            #    raise ValueError('No DB results returned, proceeding with the next day.')
            #else:
            #    print('DB count for current day', nresults)

            out = []
            for doc in res:
                out.append(doc)

                
            if insert_to_db:
                # Dump result into collection pytest
                # So queries run faster for large DBs
                self.target_collection  = self.db[insert_to_db] # Insert results in there if activated, see param insert_to_db
                #ins                     = self.target_collection.insert_many(out)
                insagain = self.target_collection.update_many(_filter, )
                #print(ins.inserted_ids)

            if download_interest_rates:
                # This is the 8hour funding rate
                try:
                    self.create_msg(ts_high, ts_low)
                    asyncio.get_event_loop().run_until_complete(self.call_api())
                except Exception as e:
                    print('Error while downloading from Deribit: ', e)
                    print('Proceeding with interest rate of 0')
                    self.historical_interest_rate = 0
            else:
                self.historical_interest_rate = 0

            download_endtime = datetime.datetime.now()
            # Got to change this to subtr.
            print('\nDownload Time: ', download_endtime - download_starttime)
            print('\nDisconnecting Server')

            return out, self.historical_interest_rate

        except Exception as e:
            print('Error: ', e)
            print('\nDisconnecting Server within error handler')
            self._stop()
            self.client.close()


if __name__ == '__main__':
    brc = BRC()
    #dat, interest_rate = brc._run(datetime.datetime(2022, 1, 1, 0, 0, 0),
    #               datetime.datetime(2022, 4, 1, 0, 0, 0),
    #                False, False, '')
    #print(len(dat))
    brc._summary_stats_preprocessed()
    #brc.plot_iv_surface()
    #d = brc._summary_stats(1.1, 0.9)

    #d = pd.DataFrame(dat)
    #d.to_csv('data/orderbooks_test.csv')
    #print('here')   