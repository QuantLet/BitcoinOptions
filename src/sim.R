
# Functions and Libraries for the Historical Density Estimation

library("sde")
library("stats")
library("rugarch")


# Debug
#setwd('/home/lax/src/repos/spd')
#tdat = read.csv('data/BTCUSDT.csv')
#startvalue = 10000
#days = 100

simple_brownian = function(r, startvalue, days, sigma){
    #https://stackoverflow.com/questions/36463227/geometrical-brownian-motion-simulation-in-r
    #print(paste0('HD Startvalue: ', startvalue))
    # for constant sigma: constant: sd(diff(log(tdat)))

    f0 = startvalue
    mu = r
    T = 1
    t = seq(1/days,T,by=1/days)
    n = length(t)
    f = f0*exp(cumsum((mu-sigma*sigma/2)*T/n + sigma*sqrt(T/n)*rnorm(n)))
    #diff = f - f0
    #print('todo: create correlated jumps!!')
    # like this or as in hilpisch
    # https://math.stackexchange.com/questions/446093/generate-correlated-normal-random-variables

    #plot(t,f,type="l", main = paste0('Bitcoin Price Simulation for ', days, ' days and start value ', startvalue, ' and interest rate ', r),
    #    ylab = 'Bitcoin Price in USD')
    return(f)
}

brownian <- function(n_times){

    # Brownian Motion for Milstein Scheme

    # n is length of time series
    .sd = 1/n_times

    # Choose sd as 1/n
    x <- x.new <- x.new.p <- vector()
    for(i in 1:n_times){
        # Initialize variables
        x <- rnorm(1, 0, 1/n_times)
        # concatenate variables 
        # to increase the vector size
        x.new <- c(x.new,x)
        # sum the vector numbers
        x.new.p <- cumsum(x.new)
        # plot the model
        #plot(x.new.p,type="b",
        #     main=paste("Brownian motion simulation in R\nTime =",i,sep=" "),
        #     xlab="x coordinates")
    }
    return(x.new.p)
}

diffusion_derivative = function(sigma){
    # Calculate the derivative of the estimated diffusion coefficient 
    # in order to perform Milstein Scheme

    deriv_est = list()
    for(i in 1:length(sigma$x)){
        delta_s = sigma$x[i+1] - sigma$x[i]
        deriv_est[[i]] = (sigma$y[i+1] - (sigma$y[i] - delta_s)) / delta_s
    }
    return(unlist(deriv_est))
}

milstein = function(s, r, startvalue){

    # Milstein Scheme as on page 203
    # Interest rate taken from Deribit
    #print(paste0('HD Startvalue: ', startvalue))

    n = length(s)
    delta_t = 1/n
    diffusion = sde::ksdiff(s, n = length(s))
    diffusion_deriv = diffusion_derivative(diffusion)
    #print(diffusion_deriv)
    #delta_brownian = brownian(n)

    s_est = c()
    s_est[1] = startvalue
    for(i in 2:n){
        #if(i == 511) browser()
        delta_brownian = rnorm(1, 0, 1) # thought it should be 1/n but not sure!
        delta_brownian_delta_time = delta_brownian^2 - delta_t

        # At least 0
        s_est[i] = max(s[i-1] + r * s[i-1] * delta_t + diffusion$y[i-1] * delta_brownian + 
                0.5 * s[i-1] * diffusion_deriv[i-1] * delta_brownian_delta_time, 0)

    }

        #plot(s_est, main = paste0('Milstein Scheme'))
        return(s_est)
}


gauss_kernel = function(u){
    return(1/sqrt(2 * pi) * exp(-0.5 * u^2))
}

gen_p_hat = function(simulation_return, dom = NULL){
    # p_hat as on page 202

    rets = unlist(simulation_return)
    d = density(rets) # should be able to use this too!
    h = d$bw
    m = length(simulation_return)
    
    if(is.null(dom)){
        dom = seq(-1, 1, 0.001)
    }
    
    p_hat = c()
    for(i in 1:length(dom)){
        fac = 1/(m*h)
        u = dom[i]
        gauss_result = c()
        for(k in 1:m){
            gauss_result[k] = gauss_kernel((rets[k] - u)/h)
        }
        p_hat[i] = fac * sum(gauss_result) 
        rm(gauss_result)
    }

    if(length(dom) > 1){
        # Comparison between R default density and this hot p_hat estimation!
        par(mfrow = c(2,1))
        png('R_Density_vs_own_Kernel.png')
        plot(d, main = paste0('R Density vs. Own Density'))
        plot(dom, p_hat)
        dev.off()
        par(mfrow = c(1,1))
    }

    return(list('p_hat' = p_hat, 'dom' = dom)) # must return dom, too!
}
