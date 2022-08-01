import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr


class HistoricalDensity:
    """
    ObservationDate: Fix a Date on which we observe Option Prices 
    StartDate: Fix a time frame that starts before the Observation Date
    timediff = ObservationDate - StartDate
    Look timediff in the past:
        Rets: Calculate returns for each maturity
        Calculate the bandwith for Kernel Estimation via cross-validation # https://rdrr.io/cran/kedd/man/h.ccv.html
        For each unique tau (in the column), calculate the returns from each day to ObservationDate
        Learn the variance of the process
    Now look timediff in the future



        hdens = list()
        band  = vector(length = length(tau_day))

        band_ccv = function() {
        for (i in 1:length(tau_day)) {
            band[i] = h.ccv(daxT[, i], kernel = "biweight")$h
        }
        return(band)
        }

        h = band_ccv()
        for (i in 1:length(tau_day)) {
        hdens[[i]] = density(daxT[, i], bw = h[[i]], na.rm = T, kernel = "biweight")
        }

            def estimate_bandwidth(self, tau_day, rets):
        
        KernSmooth = importr('kedd')
        rhccv = ro.r['h.ccv']
        
        r_tau_day = ro.FloatVector(tau_day)
        band = ro.FloatVector(length = len(r_tau_day))
        
        for i in range(len(tau_day)):
            band[i] = rhccv(rets[, i], kernel = 'biweight'])$h
        return band

        def estimate_density(self):
        stats       = importr('stats')
        r_rets      = ro.Matrix(rets)

    """

    def __init__(self):
        print('starting')
        self.dat = pd.read_csv('pricingkernel/BTCUSDT.csv')
        self.prices = self.dat['Adj.Close']

    def gausskernel(self, u):
        return 1/(2 * np.pi) * np.exp(-0.5 * u**2)

    def newton_raphson_f(self, h, n):
        return np.log(n) / (n * h)

    def newton_raphson_f_deriv(self, h, n):
        return (-np.log(n)) / (n * h**2)

    def estimate_bandwidth(self, n, error = 0.01):
        """
        Use Newton-Raphson to estimate the correct bandwidth for Florens-Zmirou Estimator.

        First Condition:
        ln(N) / (N * H) --> 0

        Second Condition:
        N * h**4 --> 0

        For Newton-Raphson:

        f(h ; N) = ln(N) / (N * H)
        f'(h; N) = -ln(N) * N / (N*h)**2

        """
        
        # Guess initial value:
        initial_h, h = 1/n, 1/n#0.01, 0.01 # try to fulfill second condition
        curr_error = 10
        iteration_counter = 1

        # Mean Value Theorem: Check if zero exists

        
        # First condition
        while abs(curr_error) > error:
            print('iteration: ', iteration_counter)
            
            f = self.newton_raphson_f(h, n)
            curr_error = f
            print('current error: ', curr_error)

            f_deriv = self.newton_raphson_f_deriv(h, n)
            print('\nf: ', f, '\nf_deriv: ', f_deriv, '\nh before: ', h)
            
            h = h - (f/f_deriv)
            print('\nh after: ', h)

            iteration_counter += 1
            if iteration_counter > 500:
                raise(ValueError('error in newton raphson'))
        
        # Second Condition
        # Confirm proposed h
        second_condition = n * h**4
        print('Second Condition: ', second_condition)

        return h

    # use indicator for florens-zmirou

    def florens_zmirou(self):
        """
        See HU Proof
        Florens-Zmirou (1993) estimator of the diffusion coefficient sigma
        Computational Finance p.201
        x: prices, np.array
        """
        S = self.prices
        n = len(S)
        h = self.estimate_bandwidth(n)
        
        sigma_hat = []
        domain = np.arange(round(min(S)), round(max(S)), 50)

        for s in domain:

            upper = []
            lower = []
            for i in range(n-1):
                k = self.gausskernel((S[i] - s)/h) # @Todo: check kernel input, dimensions should be false!
                o = k * (1/n) * (S[i + 1] - S[i])**2
                upper.append(o)
                lower.append(k)
            sigma_hat.append(sum(upper) / sum(lower))
        return sigma_hat

        

if __name__ == '__main__':
    hd = HistoricalDensity()
    fz = hd.florens_zmirou()
    #h = hd.estimate_bandwidth(10000)
    """
    n = 12391
    dom = np.arange(-1000, 1000, 1)
    f_out = []
    f_deriv_out = []
    for i in dom:
        f = hd.newton_raphson_f(i, n)
        f_deriv = hd.newton_raphson_f_deriv(i, n)
        f_out.append(f)
        f_deriv_out.append(f_deriv)
    plt.plot(dom, f_out)
    plt.plot(dom, f_deriv_out)
    plt.show()
    """