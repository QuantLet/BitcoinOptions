# Merton Jump Diffusion Model, 1976
# Yves Hilpisch - Python for Finance p. 285 ff.
# Stochastic Differential Equation on page 285
# Euler Discretization Scheme page 286

import numpy as np
import matplotlib.pyplot as plt

class JumpDiffusion():
    """
    SDE: 
    dS_t = (r - r_j)S_t dt + sigma S_t dZ_t + J_t S_t dN_t
    where
    S_t index level at date t
    r constant riskless rhot rate
    r_j defined as lambda * (e^{mu_j + delta**2/2 - 1}) drift correctionf or jump tomaintain risk neutrality
    sigma constant volatility of S
    Z_t Standard Brownian motion
    J_t Jump at date ti with distribution:
    log(1+J_t) approx. N(log(1+mu_j) - delta**2/2, delta**2)
    where N is the cumulative distribution function fo a standard normal random variable
    """
    def __init__(self):
        self.S0 = 100.0
        self.r = 0.05
        self.sigma = 0.2
        self.lamb = 0.75
        self.mu = -0.6
        self.delta = 0.25
        self.T = 1.0

    def _simulate(self):
        # We need three sets of independent random numbers in order to 
        # simulate the jump diffusion
        # Input: tdat, r, startvalue, days, sigma)
        
        M = 10 # Maturity # Default: 50
        I = 1 # Number of Paths # Default: 10000
        dt = self.T/M
        rj = self.lamb * (np.exp(self.mu + 0.5 * self.delta**2) - 1)
        S = np.zeros((M+1, I))
        S[0] = self.S0
        sn1 = np.random.standard_normal((M+1, I))
        sn2 = np.random.standard_normal((M+1, I))
        poi = np.random.poisson(self.lamb * dt, (M+1, I))
        for t in range(1, M+1, 1):
            S[t] = S[t-1] * (np.exp((self.r - rj - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * sn1[t]) + (np.exp(self.mu + self.delta * sn2[t]) - 1) * poi[t])
            S[t] = np.maximum(S[t], 0)
        return S

    def _run(self, plot = True):
        S = self._simulate()
        
        if plot:
            # Histogram
            plt.hist(S[-1], bins = 50)
            plt.xlabel('value')
            plt.ylabel('frequency')
            plt.grid(True)
            plt.show()

            # Paths
            plt.plot(S[:, :10], lw = 1.5)
            plt.xlabel('time')
            plt.ylabel('index level')
            plt.grid(True)
            plt.show()

if __name__ == '__main__':
    jd = JumpDiffusion()
    jd._run()

