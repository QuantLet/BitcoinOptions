
#setwd('/home/lax/src/repos/spd/')
# -----------------
# Libraries loading
# -----------------

# List of libraries to be used
lib <- list("truncnorm", "MASS", "MCMCpack", "ggplot2", "readr", "Rlab")

# Installing or calling the libraries
invisible(lapply(lib, function(x){
  result <- library(x, logical.return=T, character.only =T)
  if(result == F) install.packages(x)
  library(x, character.only =T)
  #print(paste0(x, " loaded"))
}))
rm(lib)

# Get svcj function
source("src/svcj_model.R")



svcj_physical = function(mu, mu_y, sigma_y, lambda, alpha, beta, rho, sigma_v, rho_j, mu_v, kappa, theta, n){
    # Simulate Physical Density as in 
    # SVCJ Option App by Ivan Perez and Change of Measure Script

    # Create Matrices to collect output
    V    = matrix(0, nrow = n, ncol  = 1)  # Volatility of log return
    Y    = matrix(0, nrow = n, ncol  = 1)  # Log return
    #Jv   = matrix(0, nrow = iter, ncol  = n)  # Jumps in volatility
    #Jy   = matrix(0, nrow = iter, ncol  = n)  # Jumps in log return
    V[1,1] = mu_v  # Initial value of volatility = mean of volatilty

    # Const Helpers
    rho_sigma_prod = rho * sigma_v
    sigma_v_sq = sigma_v^2

    # Calculate RVs as vectors
    
    # Notation:
    # Subscript y or s for stock, v for vola
    # time indices are omitted
    
    # Jump Indicator
    J_s = rbern(n = n, prob = lambda)  # Bernoulli distributed random value with lambda = 0.051 for determining whether a jump exists
    J_v = rbern(n = n, prob = lambda)

    # Jumps Sizes
    Z_v       = rexp(n = n, rate = 1/mu_v)  # Exponential distributed random value jump size in volatility
    Z_y       = rnorm(n = n, mean = mu_y + rho_j * mu_v, sd = sigma_y)  # Jump size of log return
    
    # Expectation of Jump in returns
    E_gamma = mean(Z_y)

    # Errors
    epsilon_mat = mvrnorm(n = n, mu = c(0,0), Sigma = matrix(c(1, rho_sigma_prod, rho_sigma_prod, sigma_v_sq), nrow = 2))
    

    # 1 day periods
    Delta = 1/365 # Choose Delta according to Belaygorod05, Chapter 3.2
    #print('@Todo: Muessen die Delta subscripts noch runterskaliert werden auf Tagesbasis? E.g. mu = mu/365')
    # Problem with Euler Scheme: Vola can be negative for large Deltas
    for(i in 2:n){

        epsilon_s = try(epsilon_mat[i,1])
        if('try-error' %in% class(epsilon_s)){
            print(str(epsilon_s))
            print(i)
        }
        
        epsilon_v = epsilon_mat[i,2]
        
        # Scalars
        #mu_bar = (exp(mu_y + 0.5 * sigma_y) / (1 - mu_y * rho_j)) - 1

        # Calculate log diff in Stock price and Vola diff
        d_log_s = (mu - lambda * E_gamma) * Delta + sqrt(V[i-1, 1] * Delta) * epsilon_s + Z_y[i] * J_s[i]
        d_v     = kappa * (theta - V[i-1,1]) * Delta + sqrt(V[i-1, 1] * Delta) * epsilon_v + Z_v[i] * J_v[i]
    
        # Collect results
        Y[i,1] = d_log_s
        
        # If Vola is nonnegative add, else return to mean:
        vola_i = V[i-1, 1] + d_v
        if(vola_i > 0){
            V[i,1] = vola_i
        }else{
            print('vola would be negative, reverting to mean...')
            V[i,1] = mu_v
        }

        #print(vola_i)
    # Just lapply this n times and process in apply function below :)
    }
    return(unlist(Y)) # log diff prices
}

execute_svcj_sim = function(ndays = 10, n = 2000, iter = 5000, startdate = as.Date('2020-01-01'), startvalue = 0L, do_plot = TRUE, prices = NULL, cached_svcj_parameters = FALSE){

    # @ Todo: this is failing in main proabably because n is too small!
    # so just subset after n days and run for 1000 days anyways :)

    # Load price feed!
    # @Todo Check if daily data is ok
    #tdat = read.csv('data/BTCUSDT.csv', stringsAsFactors = FALSE)
    if(is.null(prices)){
        print('loading btc returns')
        btc_price_time_series_file = 'data/BTC_USD_Quandl.csv'
        tdat = read.csv(btc_price_time_series_file, stringsAsFactors = FALSE)
        prices = data.frame(list('Adj.Close' = tdat$Adj.Close,
                            'date' = as.Date(tdat$Date)))
        rm(tdat)
    }

    # Fit SVCJ
    # Some initial set up parameters: 
    #n      = 1000  # Number of observaions in each simulation (i.e number of future days)
    #iter = 5000 # Number of price paths
    #startdate = as.Date('2017-12-01')
    #do_plot = False
    p = prices$Adj.Close[prices$date <= startdate]
    print(summary(p))
    #browser()
    if(cached_svcj_parameters){
        #print('loading svcj params')
        load('svcj_results_R.RDATA')
        assign('btc_svcj', svcj_results)
    }else{
        print('Fitting SVCJ...')
        btc_svcj <- svcj_model(p, N = iter, n = n)
    }
    
    print(btc_svcj$parameters)
    #save(btc_svcj,file="Data/btc_svcj_own.Rda")


    # --------------------------------------
    # Physical Density Simulation
    # --------------------------------------

    crypto = 'btc'
    svcj_results = list('btc' = btc_svcj)

    simulated_returns_long = list()
    parameters          = svcj_results[[crypto]]$parameters
    mu                  = parameters[1,2]
    mu_y                = parameters[2,2]  
    sigma_y             = parameters[3,2]  
    lambda              = parameters[4,2]
    alpha               = parameters[5,2]
    beta                = parameters[6,2]
    rho                 = parameters[7,2]
    sigma_v             = parameters[8,2]
    rho_j               = parameters[9,2]
    mu_v                = parameters[10,2]
    kappa               = -(alpha/beta)
    theta               = alpha/kappa

    init_price <- prices$Adj.Close[prices$date == startdate] 
    print(paste0('init price in svcj sim: ', init_price))

    out = lapply(1:iter, function(z, startprice = init_price){
        sim_prices = svcj_physical(mu, mu_y, sigma_y, lambda, alpha, beta, rho, sigma_v, rho_j, mu_v, kappa, theta, ndays)
        #browser()
        #@Todo make sure overall_rets cant be crazier than a loss of 1
        overall_rets = (cumprod(1+sim_prices) - 1)
        lastprice  = startprice + (startprice * tail(overall_rets, 1))
        rets       = (lastprice - startprice)/startprice
        #compounded_rets = rets*ndays/365
        #print('@Todo: Check compounded rets')
        return(rets)
    })
    #browser()
    
    # Return list of simulated returns
    return(out)

}

#o = execute_svcj_sim(ndays = 10, n = 5000, iter = 5000, startdate = as.Date('2020-01-01'), startvalue = 0L, do_plot = FALSE, prices = NULL, cached_svcj_parameters = TRUE)