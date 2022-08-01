import numpy as np
from scipy.stats import invwishart, truncnorm
from scipy.stats import beta as betaDist
from math import floor
import pdb
import pandas as pd
import datetime
import matplotlib.pyplot as plt


import numpy.random as npr  
import scipy as sp

# https://github.com/QuantLet/SVCJ/tree/master/

class SVCJ():
    def __init__(self, prices, n, N):
        print('Powering SVCJ Model...')
        self.annual_trading_days = 365
        self.n = n
        self.N = N

        # Prior distribution of hyperparameter values
        self.a = 0
        self.A = 25 # prior for mu
        self.b = np.array([0,0])#, nrow = 2, ncol = 1) # prior for alpha and beta
        self.B = np.diag([1,1]) #diag(2) # prior for alpha and beta
        self.c = 2.5 # prior for sigma2_v
        self.C = 0.1 # prior for sigma2_v
        self.d = 10 # prior for mu_v
        self.D = 20 # prior for mu_v
        self.e = 0 
        self.E = 100 # prior for mu_y
        self.f = 100 # prior for sigma2_y
        self.F = 40 # prior for sigma2_y
        self.g = 0
        self.G = 4
        self.k = 2 # prior for lambda
        self.K = 30 # prior for lambda
        self.tmp = [] #vector()

    def annualized_returns(self, prices):
        '''
        Converts list of prices to annualized returns
        https://stackoverflow.com/questions/10226551/whats-the-most-pythonic-way-to-calculate-percentage-changes-on-a-list-of-numbers/10226927
        '''
        p = np.array(prices, dtype = float)
        self.rets = (np.diff(p) / p[:-1])
        self.annualized_returns = self.rets * np.sqrt(self.annual_trading_days)
        return self.annualized_returns
    
    def test_rho_acceptance(self, rho, p, i, rhoprop, acceptsumrho, rhosum, rho2sum):
        pot_x = min(p, 1)
        u = np.random.uniform(low = 0, high = 1, size = 1)
        #rho = 0 # init

        if pot_x > u:
            rho = rhoprop[0]
            if i > self.n:
                acceptsumrho += 1

        if i > self.n:
            #print('trying to update rhosum')
            #print('rhoprop: ', rhoprop)
            #print('p: ', p)
            #print('pot_x vs u: ', pot_x, u)

            # += bug with ndarray
            rhosum = rhosum + rho
            rho2sum = rho2sum + rho ** 2

        return rho, acceptsumrho, rhosum, rho2sum

    def test_simple_acceptance(self, i, mV, mVsum, mV2sum):
        
        if i > self.n:
            mVsum += mV
            mV2sum += mV ** 2
        return mVsum, mV2sum


    def propose_rho(self, rho, rhoprop, Y, Z, m, J, X, s2V, alpha, V0tmp, V, beta, XV, T):
        """
        There was some T for TRUE in this code, its deleted as its bullshit
        """
        
        p1 = (np.sqrt(1 - rho ** 2) / np.sqrt(1 - rhoprop ** 2)) #** T
        p2 = -1 / (2 * (1 - rhoprop ** 2)) 
        p3 = (Y - Z * m - J * X - (rhoprop / s2V ** 0.5)) * (V - alpha - V0tmp * (1 + beta) - J * XV)**2
        p4 = V0tmp + 1 / (2 * (1 - rho ** 2)) 
        # Check if p5 really should be the same as p3
        p5 = (Y - Z * m - J * X - (rhoprop / s2V ** 0.5)) * (V - alpha - V0tmp * (1 + beta) - J * XV)**2 

        p = p1 * np.exp(sum(p2 * p3 / p4 * p5))
        #pdb.set_trace()
        return p

    def update_XV(self, J, Jindex, V, V0, alpha, beta, Y, Z, m, mJ, mV, X, rho, s2V, rhoJ, s2J, XV):
        """
        iterate over t
        """
        # @Todo check index
        for t in Jindex[0]:
            eV = V[t] - V[t-1] - alpha - beta * V0
            eY = Y[t] - Z[t] * m - X[t]
            H = (1/((1 - rho ** 2) * s2V * V[t-1]) + rhoJ ** 2 / s2J) ** (-1)
            h = H * ((eV - rho * np.sqrt(s2V) * eY) / ((1 - rho ** 2) * s2V * V[t-1]) + rhoJ * (X[t] - mJ)/s2J - 1/mV)
            if (h + 5 * np.sqrt(H) > 0):
                # so truncnorm should only be able to deliver positive results
                # but apparently it must be overwritten by loc 
                a_ = 0
                b_ = h + 5 * np.sqrt(H)
                XV[t] = truncnorm(a = a_, b = b_, loc = h, scale = np.sqrt(H)).rvs()
                truncnorm(-10, 0)
            else:
                XV[t] = 0
            if (np.isposinf(XV[0]) or np.isneginf(XV[0]) or np.isnan(XV[0])):
                XV[t] = 0

            if XV[t] < 0:
                #pdb.set_trace()
                # @Todo see comment above, result must be nonnegative but is overwritten
                # https://stackoverflow.com/questions/53125437/fitting-data-using-scipy-truncnorm
                #print('overwriting XV[t] due to negativity')
                XV[t] = 0
        return XV

    def update_X(self, Jindex, X, V, alpha, beta, Y, Z, m, mJ,XV, rho, rhoJ, s2J, s2V):
        
        for t in Jindex[0]:    
        #for t in Jindex:
            # @Todo check if this t index is correctly iterated over
            #print('current t: ', t)
            eV = V[t] - V[t-1] - alpha - beta * V[t-1] - XV[t]
            eY = Y[t] - Z[t] * m
            L = (1/ ((1 - rho ** 2) * V[t-1]) + 1/s2J) ** (-1)
            #pdb.set_trace()
            l = L * ( (eY - (rho / np.sqrt(s2V)) * eV) / ((1 - rho ** 2) * V[t-1]) + (mJ + rhoJ * XV[t]) / s2J)
            X[t] = np.random.normal(l, np.sqrt(L), size = 1)
            return X

    def draw_v(self, i, dfV, V, Y, Vsum2, stdevV):
        epsilon = np.random.standard_t(df = dfV, size = len(Y))
        mv = 0 # mean of t dist with parameter dfV
        v = dfV / (dfV-2) # Variance of t dist 
        epsilon = (stdevV/np.sqrt(v)) * epsilon

        if i == floor(self.n/2):
            self.Vindex1 = np.where(Vsum2 > np.quantile(Vsum2, 0.925))
            self.Vindex2 = np.where((Vsum2 > np.quantile(Vsum2, 0.75)) & (Vsum2 < np.quantile(Vsum2, 0.925)))
            self.Vindex3 = np.where((Vsum2 > np.quantile(Vsum2, 0.25)) & (Vsum2 < np.quantile(Vsum2, 0.025)))
            self.Vindex4 = np.where(Vsum2 < np.quantile(Vsum2, 0.025))
        if i >= floor(self.n / 2):
            epsilon[self.Vindex1] = 1.35 * epsilon[self.Vindex1]
            epsilon[self.Vindex2] = 1.25 * epsilon[self.Vindex2]
            epsilon[self.Vindex3] = 0.75 * epsilon[self.Vindex3]
            epsilon[self.Vindex4] = 0.65 * epsilon[self.Vindex4]
        
        Vprop = V + epsilon

        return Vprop, epsilon  

    def test_j(self, Y, Z, m, X, V, V0tmp, alpha, beta, XV, lda, rho, s2V):
        eY1 = Y - Z*m - X
        eY2 = Y - Z*m
        eV1 = V - V0tmp - alpha - beta*V0tmp - XV
        eV2 = V - V0tmp - alpha - beta*V0tmp
        s2V_inv = s2V ** (-1)
        p1 = lda*np.exp( -0.5 * ( ((eY1 - (rho/s2V_inv)*eV1)**2)/((1-rho**2)*V0tmp) + (eV1**2)/(s2V*V0tmp) ) )
        p2 = (1 - lda) * np.exp( -0.5 * ( ((eY2 - (rho/s2V_inv)*eV2)**2)/((1-rho**2)*V0tmp) + (eV2**2)/(s2V*V0tmp) ) )
        p = p1/(p1 + p2)

        if np.all(np.isnan(p)):
            print('p is na')
            pdb.set_trace()
        return p

    def fit(self, prices, N):
        '''
        N is sample size
        n 
        '''
        Y = self.annualized_returns(prices)
        T = len(Y)
        #self.ones = np.ones(len(self.annualized_returns))

                # Starting values
        # m=mu
        m=self.a 
        msum=0 
        m2sum=0

        # kappa=-alpha/beta, the theta in equation 1
        kappa = 0 
        kappasum = 0 
        kappa2sum = 0     

        # alpha in equation 3
        alpha = self.b[0]
        alphasum = 0 
        alpha2sum = 0

        # beta in eq.3
        beta = self.b[1] 
        betasum = 0 
        beta2sum = 0

        # sigma_v in eq.3
        s2V = self.C/(self.c - 2)
        s2Vsum = 0
        s2V2sum = 0

        # the relation between w1 ad w2
        rho = 0 
        rhosum = np.array([0])#0 
        rho2sum = np.array([0])#0

        # mu_v, the param in expoential distr. of Z_v
        # (jump size in variance)
        mV = self.D/(self.d-2) 
        mVsum = 0
        mV2sum = 0         

        # mu_y, the mean of jump size in price Z_y
        mJ = self.e 
        mJsum = 0 
        mJ2sum = 0

        # sigma_Y, the variance of jump size in price Z_y 
        s2J = self.F/(self.f - 2) 
        s2Jsum = 0 
        s2J2sum = 0

        # rho param in the jump size of price
        rhoJ = self.g 
        rhoJsum = 0 
        rhoJ2sum = 0

        # jump intensity
        lda = 0
        ldasum = 0
        lda2sum = 0

        # Initial values for variance_t
        V = 0.1*(Y - np.mean(Y))**2 + 0.9*np.var(Y)
        Vsum = 0
        Vsum2 = 0

        # J = data(2:end,3);
        J = abs(Y) - np.mean(Y) > 2 * np.std(Y)
        Jsum = 0

        # the jump size in volatility, Z_t^y
        XV = np.random.exponential(scale = 1/mV, size = len(Y)) #rexp(length(Y), rate = 1/mV)
        XVsum = 0

        # the jump size in price
        X = np.random.normal(loc = (mJ + XV * rhoJ), scale = s2J**0.5, size = len(Y))    #rnorm(n = length(Y), mean = (mJ+XV*rhoJ), sd = s2J**0.5)
        Xsum = 0
        stdevrho = 0.01
        dfrho = 6.5
        stdevV = 0.9
        dfV = 4.5
        acceptsumV = np.zeros(len(V))   #rep(0,len(V))
        acceptsumrho = 0
        acceptsums2V = 0
        Z = np.ones(len(Y))    #rep(1, len(Y))
        
        # Param Matrix
        ncol = 10
        test = np.zeros(shape = (N, ncol))
        print('length of test: ', N)

        for i in range(0, (N-1)):
            Rho = 1/(1 - rho**2)
            V0 = V[0]

            # Draw m(i+1)
            V0tmp = np.append(V0, V[:-1])
            Q = (Y - X*J - rho / s2V ** 0.5 * (V - V0tmp) * (1 + beta) - alpha - J*XV) / V0tmp ** 0.5
            W = (1/V0tmp) ** 0.5

            A_inv = self.A**(-1)
            As = (A_inv + 1 / (1 - rho ** 2) * np.dot(np.transpose(W), W))**(-1)
            as1 = As * (A_inv * self.a + Rho * np.dot(np.transpose(W), Q))

            m = np.random.normal(loc = as1, scale = As ** 0.5, size = 1)

            if i > self.n:
                msum += m
                m2sum += m ** 2
            

            # Expected Return
            eY = Y - Z * m - X * J

            # Expected Variance
            eV = V - V0tmp - XV*J
            Q = (eV - rho * np.sqrt(s2V) * eY) / V0tmp ** 0.5
            #pdb.set_trace()
            #  test = np.matrix([1/V0tmp**0.5, V0tmp**0.5])
            # np.dot(test,np.transpose(test))
            W = np.transpose(np.matrix([1/V0tmp**0.5, V0tmp**0.5])) #np.array(1/V0tmp**0.5, V0tmp**0.5)
            B_inv = np.linalg.inv(self.B)#np.invert(self.B)
            
            # Rho must be Rho[0] if it is ndarray or shapes are going to misalign
            if isinstance(Rho,np.ndarray):
                print('ndarray')
                Bs = (B_inv + (Rho[0]/s2V) * np.dot(np.transpose(W), W)) ** (-1)
                bs = np.dot(Bs, np.transpose(np.dot(B_inv, self.b) + (Rho[0]/s2V) * np.dot(np.transpose(W), Q)))
            else:
                Bs = (B_inv + (Rho/s2V) * np.dot(np.transpose(W), W)) ** (-1)
                bs = np.dot(Bs, np.transpose(np.dot(B_inv, self.b) + (Rho/s2V) * np.dot(np.transpose(W), Q)))
            
            temp = np.random.multivariate_normal(mean = bs.ravel().tolist()[0], cov = Bs, size = 1)
            alpha = temp[0][0]
            beta = temp[0][1]
            kappa = -alpha/beta
           
            if i > self.n:
                alphasum += alpha
                alpha2sum += alpha ** 2
                betasum += beta
                beta2sum += beta ** 2
                kappasum += kappa
                kappa2sum += kappa ** 2

            # s2V
            cs = self.c + len(Y)
            Cs = self.C + sum( (V - V0tmp - alpha - beta * V0tmp - XV * J)**2 / V0tmp)
            s2Vprop = invwishart(cs, Cs).rvs()

            q1 = ((V - V0tmp) * (1 + beta) - alpha - J*XV) ** 2
            q2 = s2Vprop * V0tmp
            q3 = s2V * V0tmp
            q = np.exp(-0.5 * sum((q1/q2) - (q1/q3)))

            # Check out how almost the same is being done for rho. Put this in a function
            p1 = ((V - V0tmp) * (1 + beta) - alpha - J*XV - rho * s2Vprop ** 0.5 * (Y - Z * m - J * X)) ** 2 
            p2 = (1 - rho ** 2) * s2Vprop * V0tmp
            p3 = ((V - V0tmp) * (1 + beta) - alpha - J * XV - rho * s2V ** 0.5 * (Y - Z * m - J * X)) ** 2
            p4 = (1 - rho ** 2) * s2V * V0tmp
            p = np.exp(-0.5 * sum(p1/p2 - p3/p4))

            # @todo: check if p and q are supposed to be the same here
            
            
            pot_x = min(p/q, 1)
            if np.isnan(pot_x):
                x = 1
            else:
                x = pot_x

            u = np.random.uniform(0, 1, size = 1)
            
            if x > u:
                s2V = s2Vprop
                if i > self.n:
                    acceptsums2V += 1

            if i > self.n:
                s2Vsum += s2V
                s2V2sum += s2V ** 2

            # Rho
            # Draw a candidate for rho (i + 1)
            # draw rhoc from a t distribution with 8 df and std of 0.26666
            rt = np.random.standard_t(df = dfrho, size = len([rho]))
            rhoprop = rho + stdevrho * rt
            if abs(rhoprop) < 1:
                p = self.propose_rho(rho, rhoprop, Y, Z, m, J, X, s2V, alpha, V0tmp, V, beta, XV, T)
                rho, acceptsumrho, rhosum, rho2sum = self.test_rho_acceptance(rho, p, i, rhoprop, acceptsumrho, rhosum, rho2sum)
                #print('updating rho: ', rho)
                #print('rhosum: ', rhosum)
            #if isinstance(rho, np.ndarray):
            #    print('here')
            #    rho = rho[0]


            # mV
            ds = self.d + 2 * T
            Ds = self.D + 2 * sum(XV) # Ds becomes negative, then invwishart fails
            #print('ds, Ds: ', ds, Ds)
            #if(sum(XV) < 0):
            #    print('sum(XV): ', sum(XV))
            #    pdb.set_trace()
                
            mV = invwishart(ds, Ds).rvs()
            
            mVsum, mV2sum = self.test_simple_acceptance(i, mV, mVsum, mV2sum)
            
            # mJ
            Es = 1/(len(Y) / s2J + 1/self.E)
            es = Es * (sum((X - XV * rhoJ) / s2J) + self.e/self.E)
            mJ = np.random.normal(loc = es, scale = Es ** 0.5, size = 1)
            mJsum, mJ2sum = self.test_simple_acceptance(i, mJ, mJsum, mJ2sum)

            # s2Y
            fs = self.f + len(Y)
            Fs = self.F + sum((X - mJ - rhoJ * XV) ** 2)
            s2J = invwishart(fs, Fs).rvs()
            s2Jsum, s2J2sum = self.test_simple_acceptance(i, s2J, s2Jsum, s2J2sum)
            
            # rhoJ
            Gs = (sum(XV ** 2) / s2J + 1/self.G) ** (-1)
            gs = Gs * (sum((X - mJ) * XV)/ s2J + self.g/self.G)
            rhoJ = np.random.normal(loc = gs, scale = Gs ** 0.5, size = 1)
            rhoJsum, rhoJ2sum = self.test_simple_acceptance(i, rhoJ, rhoJsum, rhoJ2sum)
            
            # lambda
            ks = self.k + sum(J)
            Ks = self.k + len(Y) - sum(J)
            lda = betaDist(ks, Ks).rvs()
            ldasum, lda2sum = self.test_simple_acceptance(i, lda, ldasum, lda2sum)
            
            # J
            """
            eY1 = Y - Z * m - X
            EY2 = Y - Z * m
            eV1 = V - V0tmp - alpha - beta * V0tmp - XV
            eV2 = eV1 + XV
            jp1 = (eY1 - (rho / np.sqrt(s2V)) * eV1) ** 2
            jp2 = ((1 - rho ** 2) * V0tmp)
            jp3 = (eV1 ** 2) / (s2V * V0tmp)
            p1 = lda * np.exp(-0.5 * (jp1 / jp2 + jp3))
            p2 = (1 - lda) * np.exp( - 0.5 * ( ((EY2 - (rho/np.sqrt(s2V)) *eV2)**2)/ ((1-rho**2) * V0tmp) + (eV2 ** 2) / (s2V * V0tmp)  ))
            p = p1/(p1 + p2)
            """
            #
            p = self.test_j(Y, Z, m, X, V, V0tmp, alpha, beta, XV, lda, rho, s2V)
            #print('p: ', p)
            #print('compare_p: ', compare_p)
            #pdb.set_trace()

            # print('watching p: ', p)
            u = np.random.uniform(0, 1, size = len(Y))
             #@Todo aaaaha X is none because J is fucked because p is just a list of nan...
            J = np.less(u, p) #u < p
            if i > self.n:
                Jsum = Jsum + J
            Jindex = np.where(J == 1)
            not_Jindex = np.where(J != 1)
            if(len(Jindex) == len(Y)):
                print('gonna have a fucked up X here')
                pdb.set_trace()
            n_exp = len(Y) - sum(J)
            XV[not_Jindex] = np.random.exponential(scale = 1/mV, size = n_exp)
            if len(Jindex) > 0:
                XV = self.update_XV(J, Jindex, V, V0, alpha, beta, Y, Z, m, mJ, mV, X, rho, s2V, rhoJ, s2J, XV)
                #print('XV: ', XV)
                #if XV < 0:
                #    print('xv < 0')
                #    pdb.set_trace()
            if i > self.n:
                XVsum += XVsum + XV
            if X is None:
                #print('x is none')
                pdb.set_trace()
            
            x_mean = mJ + rhoJ * XV[not_Jindex]
            x_sd = np.sqrt(s2J)
            # @Todo check len of X to be replaced here
            #pdb.set_trace()
            X[not_Jindex] = np.random.normal(loc = x_mean, scale = x_sd, size = len(not_Jindex[0]))
            if len(Jindex) > 0:
                X = self.update_X(Jindex, X, V, alpha, beta, Y, Z, m, mJ,XV, rho, rhoJ, s2J, s2V)
                        
            if X is None:
                print('x is none part 2')
                pdb.set_trace()
            
            if i > self.n:
                Xsum = Xsum + X


            # Draw V
            Vprop, epsilon = self.draw_v(i, dfV, V, Y, Vsum2, stdevV)
            # p1, p2...

            j = 0
            try:
                # Found Error: X is None
                # because in update_X Jindex is an empty array
                p1 = max(0, np.exp( -0.5 * ( ( Y[j+1] - Z[j+1]*m[0]  - J[j+1]*X[j+1] - rho / s2V**0.5 *(V[j+1] - Vprop[j] - alpha - Vprop[j] * beta - J[j+1]*XV[j+1] ) )**2/( (1 - rho**2) * Vprop[j] ) +
                                    ( Y[j] - Z[j]*m[0]  - J[j]*X[j] - rho / s2V**0.5 *(Vprop[j] - V0 - alpha - V0 * beta - J[j]*XV[j]))**2/( (1 - rho**2) * V0 ) +
                                    ( V[j+1] - Vprop[j] - alpha - Vprop[j] * beta - J[j+1]*XV[j+1] )**2/( s2V * Vprop[j] ) +
                                    ( Vprop[j] - V0 - alpha - V0 * beta - J[j]*XV[j])**2/( s2V * V0 ) ) ) / Vprop[j])
                p2 = max(0, np.exp( -0.5 * ( ( Y[j+1] - Z[j+1]*m[0]  - J[j+1]*X[j+1] - rho / s2V**0.5 *(V[j+1] - V[j] - alpha - V[j] * beta - J[j+1]*XV[j+1] ) )**2/( (1 - rho**2) * V[j] ) +
                                    ( Y[j] - Z[j]*m[0]  - J[j]*X[j] -rho / s2V**0.5 *(V[j] - V0 - alpha - V0 * beta - J[j]*XV[j]) )**2/( (1 - rho**2) * V0 ) +
                                    ( V[j+1] - V[j] - alpha - V[j] * beta - J[j+1]*XV[j+1])**2/( s2V * V[j] ) +
                                    ( V[j] - V0 - alpha - V0 * beta - J[j]*XV[j])**2/( s2V * V0 ) ) ) / V[j])
            except Exception as e:
                print(e)
                pdb.set_trace()
            # @Todo: check if p2 is integer
            if p2 != 0:
                acceptV = min(p1/p2, 1)
            elif p1 > 0:
                acceptV = 1
            else:
                acceptV = 0

            u = np.random.uniform(size = len(Y))
            if u[j] < acceptV:
                V[j] = Vprop[j]
                if i > self.n:
                    acceptsumV[j] += 1


            for j in range(1, len(Y) - 1):
                #pdb.set_trace()
                #print('j: ', j)
                p1 = max(0,np.exp( -0.5 * ( ( Y[j+1] - Z[j+1]*m[0]  - J[j+1]*X[j+1] - rho / s2V**0.5 *(V[j+1] - Vprop[j] - alpha - Vprop[j] * beta - J[j+1]*XV[j+1]) )**2/( (1 - rho**2) * Vprop[j] ) +
                               ( Y[j] - Z[j]*m[0]  - J[j]*X[j] - rho / s2V**0.5 *(Vprop[j] - V[j-1] - alpha - V[j-1] * beta - J[j]*XV[j] ) )**2/( (1 - rho**2) * V[j-1] ) +
                               ( V[j+1] - Vprop[j] - alpha - Vprop[j] * beta - J[j+1]*XV[j+1])**2/( s2V * Vprop[j] ) +
                               ( Vprop[j] - V[j-1] - alpha - V[j-1] * beta - J[j]*XV[j])**2/( s2V * V[j-1] ) ) ) / Vprop[j])
                p2 = max(0,np.exp( -0.5 * ( ( Y[j+1] - Z[j+1]*m[0]  - J[j+1]*X[j+1] - rho / s2V**0.5 *(V[j+1] - V[j] - alpha - V[j] * beta - J[j+1]*XV[j+1]) )**2/( (1 - rho**2) * V[j] ) +
                               ( Y[j] - Z[j]*m[0] - J[j]*X[j] - rho / s2V**0.5 *(V[j] - V[j-1] - alpha - J[j]*XV[j] - V[j-1] * beta ) )**2/( (1 - rho**2) * V[j-1] ) +
                               ( V[j+1] - V[j] - alpha - V[j] * beta - J[j+1]*XV[j+1])**2/( s2V * V[j] ) +
                               ( V[j] - V[j-1] - alpha - V[j-1] * beta - J[j]*XV[j])**2/( s2V * V[j-1] ) ) ) / V[j])
  


                if p2 != 0:
                    acceptV = min(p1/p2, 1)
                elif p1 > 0:
                    acceptV = 1
                else:
                    acceptV = 0

                u = np.random.uniform(size = len(Y))
                if u[j] < acceptV:
                    V[j] = Vprop[j]
                    if i > self.n:
                        acceptsumV[j] += 1
            
            if i > self.n:
                Vsum += V

            if i > floor(self.n / 2) - 100 or i < floor(self.n / 2):
                Vsum2 += V
            
            
            # Collect Results
            test[i,] = [m, mJ, s2J, lda, alpha, beta, rho, s2V, rhoJ, mV] 
            # @Todo: Check parameter evolution in test
            #print(test[i,])
            print('entering next iteration: ', i)

        # @ Todo: Probbo in _sd: mJ2sum is a negative number to the power of 0.5 --> NAN
        _parameter = ["mu", "mu_y", "sigma_y", "lambda", "alpha", "beta", "rho", "sigma_v", "rho_j", "mu_v"]
        _mean = [msum[0]/(N-self.n), mJsum[0]/(N-self.n), s2Jsum/(N-self.n),ldasum/(N-self.n),alphasum/(N-self.n),betasum/(N-self.n),rhosum[0]/(N-self.n),s2Vsum/(N-self.n),rhoJsum[0]/(N-self.n),mVsum/(N-self.n)]
        _sd = [(m2sum[0]/(N-self.n)-(msum[0]/(N-self.n))**2)**0.5, (mJ2sum[0]/(N-self.n)-(mJsum[0]/(N-self.n))**2)**0.5, (s2J2sum/(N-self.n)-(s2Jsum/(N-self.n))**2)**0.5, (lda2sum/(N-self.n)-(ldasum/(N-self.n))**2)**0.5,(alpha2sum/(N-self.n)-(alphasum/(N-self.n))**2)**0.5, (beta2sum/(N-self.n)-(betasum/(N-self.n))**2)**0.5, (rho2sum[0]/(N-self.n)-(rhosum[0]/(N-self.n))**2)**0.5,(s2V2sum/(N-self.n)-(s2Vsum/(N-self.n))**2)**0.5,(rhoJ2sum[0]/(N-self.n)-(rhoJsum[0]/(N-self.n))**2)**0.5,(mV2sum/(N-self.n)-(mVsum/(N-self.n))**2)**0.5]

        parameters = pd.DataFrame({'params' : _parameter, 'mean' : _mean,'sd' : _sd})
        parameters.to_csv('svcjparams.csv')

        # Summary
        jump_vol = XVsum/(N-self.n) * Jsum/(N-self.n)
        jump_price = Xsum/(N-self.n) * Jsum/(N-self.n)
        vol = Vsum/(N-self.n)
        sig = vol ** 0.5
        resid = (Y[1:] - msum[0]/(N-self.n) - jump_price[1:]) / sig[:-1]
        resid = np.append(resid, np.nan)

        evo = pd.DataFrame({'jump_volatility' : jump_vol, 'jump_price' : jump_price, 'volatility' : vol, 'residuals' : resid})
        evo.to_csv('evo.csv')

        
        # Plot Parameter Evolution
        fig = plt.figure()
        ax = fig.add_subplot(111)
        nparams = len(_parameter) - 1 #np.shape(test)[1] - 1
        for s in range(nparams): 
            curr_paramname = _parameter[s]
            print(curr_paramname)
            plt.plot(test[:(N-2),s], label = curr_paramname) # skip the zeros
        # Activate and Move Legend
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        #  Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title('SVCJ Parameter Evolution')
        plt.show()
        pdb.set_trace()
        return parameters, prices, V

    def extract_param(self, params, paramname):
        return params['mean'][(params['parameter'] == paramname)].values[0]

    def simulate(self, params, s0, startdate):
        #, prices, parameters, ndays, N, startvalue
        # SVCJ parameters
        params['mean'] = params['mean'].astype(float)
        mu      = self.extract_param(params, 'mu')#params['mean'][(params['params'] == 'mu')][0] #0.042
        r       = mu
        mu_y    = self.extract_param(params, 'mu_y') #-0.0492
        sigma_y = self.extract_param(params, 'sigma_y')#2.061
        l       = self.extract_param(params, 'lambda')# 0.0515
        alpha   = self.extract_param(params, 'alpha')#0.0102
        beta    = self.extract_param(params, 'beta')#-0.188
        rho     = self.extract_param(params, 'rho')#0.275
        sigma_v = self.extract_param(params, 'sigma_v')#0.007
        rho_j   = self.extract_param(params, 'rho_j')#-0.210
        mu_v    = self.extract_param(params, 'mu_v')#0.709
        v0      = self.extract_param(params, 'mu_y')#0.19**2 
        kappa   = 1-beta
        theta   = alpha / kappa

        dt      = 1/360.0 # dt
        m       = int(360.0 * (1/dt)/360.0) # time horizon in days
        n       = 100000

        T      = m * dt
        t      = np.arange(0,T+dt, dt)

        w      = npr.standard_normal([n,m])
        w2     = rho * w + sp.sqrt(1-rho**2) * npr.standard_normal([n,m])
        z_v    = npr.exponential(mu_v, [n,m])
        z_y    = npr.standard_normal([n,m]) * sigma_y + mu_y + rho_j * z_v
        dj     = npr.binomial(1, l * dt, size=[n,m])
        s      = np.zeros([n,m+1])
        v      = np.zeros([n,m+1])

        #s0     = 6500
        k      = 8000
        s[:,0] = s0 # initial CRIX level, p. 20
        v[:,0] = v0

        for i in range(1,m+1):
            v[:,i] = v[:,i-1] + kappa * (theta - np.maximum(0,v[:,i-1])) * dt + sigma_v * sp.sqrt(np.maximum(0,v[:,i-1])) * w2[:,i-1] + z_v[:,i-1] * dj[:,i-1]
            s[:,i] = s[:,i-1] * (1 + (r - l * (mu_y + rho_j * mu_v)) * dt + sp.sqrt(v[:,i-1] * dt) * w[:,i-1]) + z_v[:,i-1] * dj[:,i-1]


        plt.plot(np.transpose(s[:10000]))
        plt.xlabel('Days Ahead')
        plt.ylabel('Bitcoin Price in USD')
        plt.title(str('Bitcoin Price Simulation using SVCJ \nStarted on ') + str(startdate))
        plt.savefig('svcj_price_path_long.png',transparent=T)
        #plt.show()

        plt.plot(np.transpose(s[:1000]))
        plt.xlabel('Days Ahead')
        plt.ylabel('Bitcoin Price in USD')
        plt.title(str('Bitcoin Price Simulation using SVCJ \nStarted on ') + str(startdate))
        plt.savefig('svcj_price_path_short.png',transparent=T)
        #plt.show()

        return s, v


if __name__ == '__main__':

    startdate = '2020-03-01' # Startdate for algo
    firstdate = '2018-01-01' # First date we accept in the BTCUSDT data

    prices_raw = pd.read_csv('data/BTC_USD_Quandl.csv')
    pr = prices_raw['Adj.Close'][(prices_raw['Date'] >= firstdate) & (prices_raw['Date'] <= startdate) & (prices_raw['Adj.Close'] > 0)]
    # Reverse prices...
    p = pr.iloc[::-1].to_list()
    params = pd.read_csv('svcj_params.csv')
    
    # Calculate Vola per day in order to get V0
    _N = 1000
    s = SVCJ(prices = p, n = 100, N = _N)
    s.fit(p, _N)
    
    s0 = p[-1]
    
    sim_s, sim_v = s.simulate(params, s0, startdate)
    print("done")