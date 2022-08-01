from math import tau
from scipy import optimize # Optimize Thetas
import pandas as pd
import numpy as np
import pdb

from helpers import load_estimated_densities, load_dict
from functools import reduce # To merge list of data frames
import matplotlib.pyplot as plt

# R
from rpy2 import robjects
import rpy2.robjects.packages as rpackages

class KernelSmoother:
    """
    Maria Grith Style
    M contains spd, hd, moneyness


    def update_parameters(self):
        for i in range(4): # Update each of the four parameters
            # Integrate the following expression
            theta_vary = self.theta2 * u + self.theta3
            K_t(theta_vary) - self.theta1 * #...


    def K_t(self, theta_vary):
        '''
        Return Linear Combination of Pricing Kernel PK 
        Thetas are updated iteratively, they vary. NOT self.theta1 and so on
        '''
        pk = self.M.spd / self.M.hd
        pk_lin = self.theta1 * pk + self.theta4
        return pk_lin

    def initialize_K0(self):
        '''
        u is the same for every PK
        PK list of pricing Kernels, all evaluated at common moneyness u,
        e.g. interval [0.7, 1.3]
        There are T pricing kernels
        '''
        PK_lin = self.theta2 * self.PK[0] + self.theta3 # linearly moved u
        #k_out = []
        #for i in range(len(u)):
        #    k_out.append(self.PK[0](u_lin))
        #return sum(k_out) * (1/T)
        return PK_lin


    """
    def __init__(self, taurange):
        x, z, s, pk, date, tau, maturitydate = load_estimated_densities(fname='out/real_estimated_densities.csv', allowed_taurange=taurange)#load_estimated_densities(allowed_taurange=[0.0163, 0.0165])#load_estimated_densities(allowed_taurange=[0.018, 0.02])
        self.M = x
        self.T = len(x)
        self.t = list(range(self.T)) # Discrete Time Steps
        self.K0 = None
        self.PK = pk #List of Pricing Kernels for each Point in Time
        self.tau = tau
        self.tau_in_days = np.unique([round(float(t) * 365) for t in self.tau])[0]
        self.maturitydate = maturitydate
        if self.tau_in_days == 1:
            self.day_str = ' Day'
        else:
            self.day_str = ' Days'
        #self.d = load_dict('out/estimated_densities.csv')

        self.base = rpackages.importr("base") 
        self.sm = rpackages.importr("sm")
        self.r = robjects.r

        # Parameter Initialization
        self.theta1 = [0] * self.T   # Vector over length of T
        self.theta2 = [0] * self.T
        self.theta3 = [0] * self.T
        self.theta4 = [0] * self.T

        # Theta1 and Theta2 are initially 1, the others are 0 as in
        # Haerdle and Marron 1990
        self.theta1[0] = 1 
        self.theta2[0] = 1

        # @ Todo: Filter input for similar taus!

    def _normalize_parameters(self):
        #Normalization: Mean of all Parameters Theta_ti must be 1
        self.theta1 = self.theta1/sum(self.theta1)
        self.theta2 = self.theta2/sum(self.theta2)
        self.theta3 = self.theta3/sum(self.theta3)
        self.theta4 = self.theta4/sum(self.theta4)

    def _w(self, u, a, b, i, theta2, theta3):
        """
        Calculates Integral Weights
        Boundary Interval [a,b]
        u Moneyness
        t is Point in Time T 
        """
        #out = []
        #for _u in u[i]:#range(len(u)):
        prel = (u[i] - theta3)/theta2
        
        # Indicator Function
        if prel > a and prel < b:
            out = 0
            #out.append(prel)
        else:
            # Append one so that the product does not vanish
            #out.append(1)
            out = 1 
        return out
        # Take product of all constituents
        #p = 1
        #for i in range(len(out)):
        #    p = p * i
        
        #return p

    def initialize_K0(self, theta2, theta3):
        '''
        u is the same for every PK
        PK list of pricing Kernels, all evaluated at common moneyness u,
        e.g. interval [0.7, 1.3]

        '''
        smoothened_dfs = []
        est_dfs = []
        for i in range(len(self.PK)):
            df = pd.DataFrame({'PK' : self.PK[i], 'M' : self.M[i]})

            # Shift the domain
            # For the initialization, this is 1 * u + 0
            lin_shift = (theta2[i] * df['M'] + theta3[i])

            # Smoothen up
            kt = df['PK'].tolist()
            t = df['M'].tolist()
            # How to use h.select from r?
            h_bw = self.sm.h_select(robjects.FloatVector(t), robjects.FloatVector(kt))
            sm_reg         = self.sm.sm_regression(**{'x' : robjects.FloatVector(t),
                                    'y' : robjects.FloatVector(kt),
                                    'h' : robjects.FloatVector(h_bw),
                                    'eval.points' : robjects.FloatVector(lin_shift),
                                    'model' : "none",
                                    'poly_index' : 1,
                                    'display' : "none"})


            # Assign Y hat
            df['sm_estimate'] = sm_reg.rx2('estimate') #sm_reg['estimate']
            df['lin_shift'] = lin_shift
            smoothened_dfs.append(df)
            est_dfs.append(df[['M', 'sm_estimate']])
        
        # Mean Kt Curve is initial Reference Curve K0
        # names are still missing
        # and still got to calculate the mean!
        K0 = reduce(lambda x, y: pd.merge(x, y, on = 'M', how = 'outer'), est_dfs)
        
        # Rowwise Mean
        # @Todo exclude irrelevant columns for the mean calculation!!
        df['mean'] = df.drop(['M'], axis = 1).mean(axis = 1)
        #lt.plot(df['M'], df['mean'])
        #plt.show()
        out_df = df[['M', 'mean']]
        return out_df

    def mse(self, thetas, K0, i):
        """
        K0 is initialized reference curve, 
        or updated one in the following steps j

        get K_hat_t(theta_2 * u + theta_3)

        Then calculate MSE between the two.
        """
        df = pd.DataFrame({'PK' : self.PK[i], 'M' : self.M[i]})

        # Unpack to-be-optimized argument
        theta1 = thetas[0]
        theta2 = thetas[1]
        theta3 = thetas[2]
        theta4 = thetas[3]

        # Ensure restrictions
        theta1 = abs(theta1)
        theta3 = abs(theta3)
        #theta2 = theta2
        #theta4 = theta4

        #sm.t = (t.reference - theta2)/theta3  # time adjustment
        lin_shift = theta2 * self.M[i] + theta3
        kt   = self.PK[i].tolist()
        t    = self.M[i].tolist()
        h_bw = self.sm.h_select(robjects.FloatVector(t), robjects.FloatVector(kt))
        sm_reg         = self.sm.sm_regression(**{'x' : robjects.FloatVector(lin_shift),
                        'y' : robjects.FloatVector(kt),
                        'h' : robjects.FloatVector(h_bw),
                        'eval.points' : robjects.FloatVector(lin_shift),
                        'model' : "none",
                        'poly_index' : 1,
                        'display' : "none"})
        df['sm_estimate'] = sm_reg.rx2('estimate')
        df['lin_shift'] = lin_shift


        # Get The Difference to K0**(r-1)
        # @Todo check if mean is correct here
        df['ref_point'] = theta1 * K0['mean'] - theta4
        
        # Mean Squared Error
        df['mse'] = (df['sm_estimate'] - df['ref_point']) ** 2

        w = self._w(t, 0, 10, i, theta2, theta3)
        #du = df['M'].diff()# impute NAs! #np.diff().tolist()

        df['integral'] = df['mse'] #* w #* du # (du) missing here

        return sum(df['integral'])


    def optimize_theta(self, thetas, K0, i):
        # Minimize MSE with respect to Thetas

        theta1 = thetas[0][i]
        theta2 = thetas[1][i]
        theta3 = thetas[2][i]
        theta4 = thetas[3][i]

        theta_candidates = [theta1, theta2, theta3, theta4] # initial guess
        _args = (K0, i)  # , theta1, theta2, theta3, theta4; initial guess proabbaly excluded from rest of the arguments
        theta_est = optimize.minimize(self.mse, x0 = theta_candidates, args = _args)

        print(theta_est.message)
        print(theta_est.x)

        # Before vs afterwards        
        print(self.mse(theta_candidates, K0, i), self.mse(theta_est.x, K0, i))

        return theta_est.x


    # 2D Stuff: Use to show different SPDs/EPKs on a single day
    def pricingkernel_plot(self):
        '''
        Plots all the Pks on a single day
        self.pricingkernel_plot()
        '''

        fig=plt.figure()
        ax=fig.add_subplot(111)

        for i in range(self.T):
            if i == 0:
                lab = str('Maturity \n') + str(self.maturitydate[i])
            else:
                lab = str(self.maturitydate[i])
            plt.plot(self.M[i], self.PK[i], label = lab)
            #pdb.set_trace()
            #currdate = date[i]
            #currtau = self.tau[i]
            #currmaturity = maturitydate[i]
        
        plt.xlabel('Moneyness')
        plt.ylabel('Pricing Kernel')
        plt.title(str('Pricing Kernels for different Maturities \nTime to Maturity in ') + str(self.tau_in_days) + str(self.day_str))

        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        #  Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig('pricingkernel/plots/pricingkernel' + str(self.tau_in_days) + str('long') + '.png', transparent = True)
        plt.draw()
        #plt.show()

    def ara(self, K):
        """
        ara(u) = d log K(u) / du
        􏰀 ARA(u)>0, risk averseness
        􏰀 ARA(u)=0, risk neutrality
        􏰀 ARA(u)<0, risk proclivity

        Check if this yields the same results as

        −θt1 K′ 􏰅u−θt3 􏰆 θt2 0 θt2
        ARAt(u) = 􏰅u−θt3 􏰆 . θt1K0 θt2 + θt4
        """
        k_inv = K.iloc[::-1]
        u = k_inv.iloc[1:]['M'] # first observation is lost due to diff function
        ara = (-1) * np.diff(np.log(k_inv['mean'])) / np.diff(k_inv['M'])

        plt.plot(u, ara)
        plt.show()

    def _plot_thetas(self, theta_tracker):
        # Plot thetas
        last_thetas = theta_tracker[-1]
        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.xaxis.get_major_locator().set_params(integer=True)
        plt.plot(last_thetas[0], label = 'theta 1')
        plt.plot(last_thetas[1], label = 'theta 2')
        plt.plot(last_thetas[2], label = 'theta 3')
        plt.plot(last_thetas[3], label = 'theta 4')
        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        #  Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlabel('Pricing Kernel #')
        plt.ylabel('Theta')
        plt.savefig('pricingkernel/plots/thetas' + str(self.tau_in_days) + '.png', transparent = True)
        return None

    def _run(self):
        
        # If there are only two curves, just subtract the Kernel from the interpolation
        if len(self.M) == 2:
            print('use 2.1 estimation of SIM equation (4)')

        # Output Collection
        K0_tracker = []
        theta_tracker = []

        # Initial K0 must be looped over all PKs and M, 
        # basically Mean Kernel
        theta1, theta2, theta3, theta4 = [1]*self.T, [1]*self.T, [0]*self.T, [0]*self.T
        K0 = self.initialize_K0(theta2, theta3) 
        self.pricingkernel_plot()

        for j in range(4):
            for i in range(self.T):
                
                print(j, i)

                # Optimize Parameters
                thetas = [theta1, theta2, theta3, theta4]
                thetas_est = self.optimize_theta(
                                                thetas,
                                                K0,
                                                i)

                # Collect Parameters
                theta1[i] = thetas_est[0]
                theta2[i] = thetas_est[1]
                theta3[i] = thetas_est[2]
                theta4[i] = thetas_est[3]

            # Normalize Parameters
            theta1 = theta1/sum(theta1)
            theta2 = theta2/sum(theta2)
            theta3 = theta3 - np.mean(theta3) #theta3/sum(theta3)
            theta4 = theta4 - np.mean(theta4) #theta4/sum(theta4)#
                
            # Collect Output
            K0_tracker.append(K0)
            theta_tracker.append([theta1, theta2, theta3, theta4])

            # Restart
            # Update K0**(j-1)
            # like initialize_K0, but with updated thetas
            K0 = self.initialize_K0(theta2, theta3)

        # Absolute Risk Aversion
        #self.ara(K0_tracker[-1])
        #pdb.set_trace()

        # Check K0tracker
        if np.mean(abs(K0_tracker[-1]['mean'] - K0_tracker[-2]['mean'])) < 0.001:
           print('K0s are very close, only plotting last one!')
           K0_tracker = [K0_tracker[-1]]


        # Plot Convergence of PKs
        fig=plt.figure()
        ax=fig.add_subplot(111)
        for k in range(len(K0_tracker)):
            KX = K0_tracker[k]
            if len(K0_tracker) > 1:
                lab = str('Iteration ') + str(k)
            else:
                #lab = str('Tau: ') + str(np.unique(self.tau)[0])
                lab = ''

            plt.plot(KX['M'], KX['mean'], label = lab)

        ptitle = 'Shape Invariant Pricing Kernel \nTime to Maturity in ' + str(self.tau_in_days) + str(self.day_str)
        plt.title(ptitle)
        plt.xlabel('Moneyness')
        plt.ylabel('Pricing Kernel')

        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        #  Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig('pricingkernel/plots/pricingkernel_nth_iteration' + str(self.tau_in_days) + str('long') + '.png', transparent = True)
        plt.draw()

        self._plot_thetas(theta_tracker)


if __name__ == '__main__':
    errors = []
    taus = load_estimated_densities(allowed_taurange = [0, 1], only_taus = True)
    
    #tau = 0.0219
    #KS = KernelSmoother([float(tau)-0.0001, float(tau)+0.0001])
    #KS._run()
    
    
    unique_taus = np.sort(np.unique(taus))
    for tau in unique_taus:
        try:
            KS = KernelSmoother([float(tau)-0.0001, float(tau)+0.0001])
            KS._run()

        except Exception as e:
            errors.append(e)
            print(e)
    print('done')