library('readxl')
library('quantmod') # For SPX Spot

# Mapping Table for Input

# Column ... X
# 1 - date
# 2 - IV
# 3 - 1 for Call, 0 for Put
# 4 - Tau 
# 5 - Strike
# 6 - assigned meanwhile as put2call[,6]+ put2call[,7] -  put2call[,5]*exp(- mean(put2call[,8])* tau);
# 7 - Spot
# 8 - Probably Interest Rate
# 9 - assigned meanwhile Moneyness: Spot/Strike

extract_tau = function(raw_dat){
    # Fix cell
    tau_cell = raw_dat[1,1]

    # Extract Date from 9-Feb-21 (24d); CSize 100
    tau_d = regmatches(tau_cell, gregexpr("(?<=\\().*?(?=\\))", tau_cell, perl=T))[[1]]

    # Remove d
    tau = round(as.numeric(gsub('d', '', tau_d)) / 365, 2)

    return(tau)

}

prepare_conf_data_from_bbt = function(f = "data/spx_20210126.xlsx", end_date = as.Date('2021-01-26')){

    # Params
    #f = "data/spx_20210126.xlsx"
    #end_date = as.Date('2021-01-26')

    # Read and extract Tau
    raw_dat = read_excel(f, skip = 1)
    dat = raw_dat[-1,]

    tau = extract_tau(raw_dat)

    # Columns and indicate if call or put
    coln = c('Strike', 'Ticker', 'Bid', 'Ask', 'Last', 'IV', 'Vol', 'Moneyness', 'is_call')
    calls = dat[,1:8]
    calls$is_call = 1L
    names(calls) = coln

    puts = dat[,9:16]
    names(puts) = coln
    puts$is_call = 0L

    df = rbind(calls, puts)


    # Need Underlying
    underlying_ticker = '^GSPC'
    spot = getSymbols(underlying_ticker, src = 'yahoo')

    gspc_hist = GSPC[end_date >= index(GSPC), 'GSPC.Adjusted']
    gspc_df = data.frame(index(gspc_hist), gspc_hist)
    names(gspc_df) = c('Date', 'Adj.Close')
    last_spot = as.numeric(tail(gspc_hist, 1))

    # Assign constant variables
    df$Spot = last_spot
    df$Tau = tau
    df$Date = end_date
    df$Zeros = 0
    df$IR = 0.001

    # Scale
    df$IV = df$IV/100
    df$Moneyness = df$Moneyness/100

    # Sort and Select
    ord = c('Date', 'IV', 'is_call', 'Tau', 'Strike', 'Zeros', 'Spot', 'IR', 'Moneyness')

    # Save 
    df_fname = paste0('tmp/confidence_band_from_bbt_', tau, '.csv')
    gspc_fname = paste0('tmp/gspc_hist_', end_date, '.csv')

    write.csv(df[,ord], file = df_fname, row.names = FALSE)
    write.csv(gspc_df, file = gspc_fname, row.names = FALSE)

    #out = list(df[,ord], gspc_hist)
    out = list(df_fname, gspc_fname)
    return(out)

}