library('readxl')
library('quantmod') # For SPX Spot
library('data.table')
library('tidyverse')

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

extract_regex = function(raw_dat, end_date, days_in_year = 365){
    # Fix the cell and get Option Type, Strike, Maturity, Tau
    cell = raw_dat
    strike_call = regmatches(cell, gregexpr("C(\\d+)", cell, perl=T))[[1]]
    strike_put = regmatches(cell, gregexpr("P(\\d+)", cell, perl=T))[[1]]

    if(length(strike_call) > 0){
        # Remove C
        strike = as.numeric(gsub('C', '', strike_call))
        out = list(strike, 'call')
    }else if(length(strike_put) > 0){
        # Remove P
        strike = as.numeric(gsub('P', '', strike_put))
        out = list(strike, 'put')
    }else{
        stop("Error in calculating Tau")
    }

    # Now get Maturity, Tau
    extr_date = regmatches(cell, gregexpr("(\\d+) (\\d+) (\\d+)", cell, perl=T))[[1]]
    maturity_date = as.Date(extr_date, format = '%m %d %y')
    diff_in_days = as.numeric(maturity_date - end_date)
    tau = round(diff_in_days / days_in_year, 2)

    out[[3]] = tau
    out[[4]] = maturity_date

    return(out)

}

prepare_conf_data_from_bbt = function(dat, bbt, end_date){

    # Params
    #f = "data/spx_20210126.xlsx"
    #end_date = as.Date('2021-01-26')

    # Extract Option Type (P|C) and Strike from names
    extr = extract_regex(bbt, end_date)
    strike = extr[[1]]
    option_type = extr[[2]]
    tau = extr[[3]]
    mat = extr[[4]]

    # Columns and indicate if call or put
    #coln = c('Strike', 'Ticker', 'Bid', 'Ask', 'Last', 'IV', 'Vol', 'Moneyness', 'is_call')
    coln = c('Date', 'Maturity', 'Days_until_Maturity','Underlying', 'Spot','IV', 'Bid', 'Ask', 'Volume', 'IV_Ask', 'IV_Bid', 'Zeros')
    
    df = dat
    names(df) = coln

    df$Date = as.Date(df$Date)
    df$Underlying = as.numeric(df$Underlying)
    df$Spot = as.numeric(df$Spot)
    df$IV = as.numeric(df$IV)
    df$Bid = as.numeric(df$Bid)
    df$Ask = as.numeric(df$Ask)
    df$Volume = as.numeric(df$Volume)
    df$IV_Ask = as.numeric(df$IV_Ask)
    df$IV_BID = as.numeric(df$IV_Bid)

    df$Strike = strike
    df$is_call = ifelse(option_type == 'call', 1L, 0)
    df$Tau = as.numeric(mat - df$Date)/365

   if(length(na.omit(df$Volume)) == 0){
       return(NULL)}


    # Need Underlying
    underlying_ticker = '^GSPC'
    spot = getSymbols(underlying_ticker, src = 'yahoo')

    gspc_hist = GSPC[end_date >= index(GSPC), 'GSPC.Adjusted']
    gspc_df = data.frame(index(gspc_hist), gspc_hist)
    names(gspc_df) = c('Date', 'Adj.Close')
    #last_spot = as.numeric(tail(gspc_hist, 1))
    write.csv(gspc_df, file = 'data/gspc.csv', row.names = FALSE, quote = FALSE)

    phys_m = merge(gspc_df, df, by = 'Date', all.y = TRUE)

    # Assign constant variables
    df$Spot = phys_m$Adj.Close
    df$Zeros = 0
    df$IR = 0.001

    # Scale
    df$IV = df$IV/100
    df$Moneyness = df$Spot/df$Strike

    # Sort and Select
    ord = c('Date', 'IV', 'is_call', 'Tau', 'Strike', 'Zeros', 'Spot', 'IR', 'Moneyness')
    out = df[,ord]

    # Save 
    #df_fname = paste0('tmp/confidence_band_from_bbt_', tau, '.csv')
    #gspc_fname = paste0('tmp/gspc_hist_', end_date, '.csv')

    #write.csv(df[,ord], file = df_fname, row.names = FALSE)
    #write.csv(gspc_df, file = gspc_fname, row.names = FALSE)

    #out = list(df[,ord], gspc_hist)
    #out = list(df_fname, gspc_fname)
    return(out)

}

# The Spreadsheet has always 11 Columns for Puts and Calls
# The Strike K is in the top left corner (instrument name)
# So take blocks of 11

f = "data/SPX US July Option Chain 2000-6000K Call and Put - seperate.xlsx"
end_date = as.Date('2021-07-18')

# Read and extract Tau
#raw_dat = read_excel(f, skip = 3)
#dat = raw_dat[-1,]#read_excel(f, skip = 5)
 
sheets <- readxl::excel_sheets(f)
sheets = sheets[-1]

outl = lapply(sheets, function(z, enddate = end_date){
    dat = readxl::read_excel(f, sheet = z, skip = 5)
    out = try(prepare_conf_data_from_bbt(dat, z, enddate))
    if(!'try-error' %in% class(out)){
        return(out)
    }else{
        return(NULL)
    }
})

out_filtered = Filter(Negate(is.null), outl)
namev = names(out_filtered[[1]])

# Combine in one DF
m = out_filtered %>% reduce(full_join, by = namev)
m_final = m[!is.na(m$IV),]
o = order(m_final[,1])
write.csv(m_final[o,], file = 'data/SP500_OMON_sep.csv', row.names = FALSE)
