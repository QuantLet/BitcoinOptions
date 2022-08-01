# Get NOK/USD ...

import quandl
import pdb

def get_btc_prices(ticker):

    api_key = '51s_oiWJMf5t6tiUtG8F'
    quandl.ApiConfig.api_key = api_key

    qdat = quandl.get(ticker)
    qdat.reset_index(inplace = True)
    qdat = qdat.rename(columns = {'Value': 'Adj.Close'})
    return qdat

usd_kroner_ticker = "FED/RXI_N_B_NO"
usd_gbp_ticker = "CUR/GBP"

usd_kroner = get_btc_prices(usd_kroner_ticker)
usd_gbp = get_btc_prices(usd_gbp_ticker)

pdb.set_trace()

print("done")