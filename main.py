# Set US locale for datetime.datetime.strptime (Conflict with MAER/MAR)
import platform
import locale
from numpy.lib.function_base import insert
plat = platform.system()
if plat == 'Darwin':
    print('Using Macos Locale')
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
else:
    locale.setlocale(locale.LC_ALL, 'en_US.utf8')

import pandas as pd
import datetime
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np 
import time
import sys
import pdb
import pickle
import gc
import csv
import os
import math

from statsmodels.nonparametric.kernel_regression import KernelReg as nadarajawatson
from scipy import stats
from scipy.integrate import simps
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from src.brc import BRC
from src.strategies import IronCondor
from rpy2 import robjects
from src.helpers import save_dict, load_dict
import rpy2.robjects.packages as rpackages
from src.quandl_data import get_btc_prices
from sklearn.neighbors import KernelDensity
from mpl_toolkits import mplot3d


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

def create_regressors(_newm, _tau, m_0, t_0):
    # SET m_0 and t_0 as index variables for a loop!
    # Create matrix of regressors
    #m_0 = min(_newm)
    #t_0 = min(_tau)
    # as on page 181
    x = pd.DataFrame({  'a': np.ones(_newm.shape[0]),
                        'b':_newm - m_0, 
                        'c':(_newm - m_0)**2,
                        'd': (_tau - t_0), 
                        'e': (_tau - t_0)**2,
                        'f': (_newm - m_0)*(_tau - t_0)})
    return x

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

def plot_volasmile(m2, sigma, sigma1, sigma2, mindate, plot_ident, blockplots):
    fig=plt.figure()
    ax=fig.add_subplot(111)
    plt.plot(m2, sigma, label = 'Implied \nVolatility')
    plt.plot(m2, sigma1, label = '1st Derivative')
    plt.plot(m2, sigma2, label = '2nd Derivative')
    plt.xlabel('Moneyness')
    plt.ylabel('Implied Volatility')
    plt.title('Fitted Volatility Smile on 2020-' + str(mindate.month) + '-' + str(mindate.day))
    
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    #  Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig('pricingkernel/plots/fitted_vola_smile' + plot_ident)
    plt.draw()
    #plt.show(block = blockplots)

def plot_spd(spd, mindate, tau, plot_ident):
    fig2 = plt.figure()
    plt.scatter(spd['x'], spd['y'])
    plt.title('SPD on 2020-' + str(mindate.month) + '-' + str(mindate.day) + ' and tau: ' + str(np.unique(tau)))
    plt.savefig('pricingkernel/plots/SPD-' + plot_ident)
    plt.draw()
    #plt.show(block = blockplots)

def gaussian_kernel(M, m, h_m, T, t, h_t):
    u_m = (M-m)/h_m
    u_t = (T-t)/h_t
    return stats.norm.cdf(u_m) * stats.norm.cdf(u_t)

def gaussian_kernel_mod(M, m, h_m, sigma_m, T, t, h_t, sigma_t):
    u_m = (M-m)/h_m
    u_t = (T-t)*(sigma_m/sigma_t)/h_t
    return stats.norm.cdf(u_m) * stats.norm.cdf(u_t)

def gaussian_kernel_squared(M, m, h_m):
    u_m = (M-m)/h_m
    phi = stats.norm.cdf(u_m)**2
    return phi

def epanechnikov(M, m, h_m, T, t, h_t):
    u_m = (M-m)/h_m
    u_t = (T-t)/h_t
    return (3/4) * (1-u_m)**2 * (3/4) * (1-u_t)**2

def k(X, d):
    out = []
    n = len(X)
    for x in X:
        if abs(x) < d:
            o = (1/(2*d)) * (1 + (1/(2*n))) * (1- (x-d)**(2*n))
        else:
            o = 0
        out.append(o)
    return out

def epanechnikov2(M, m, h_m, T, t, h_t):
    u_m = (M-m)/h_m
    u_t = (T-t)/h_t

    l = k(u_m, 0.1)
    r = k(u_t, 0.1)

    return [a*b for a,b in zip(l, r)]
    

def new_epanechnikov(X):
    out = []
    for x in X:
        if abs(x) <= 1:
            k = (3/4) * (1-x**2)
        else:
            k = 0
        out.append(k)
    return out

def smoothing_rookley(df, m, t, h_m, h_t, kernel=gaussian_kernel, extend = False, boot = None):
    # M = np.array(df.M)
    # Before
    M = np.array(df.moneyness)
    if boot is None:
        y = np.array(df.mark_iv)
    else:
       # print('using bootstrapped IV')
        y = np.array(boot)

    # After Poly extension
    if extend:
        print('Extending Moneyness and IV in smoothing technique!')
        M, y = extend_polynomial(M, y)
    T = [df.tau.values[0]] * len(M) #np.array(df.tau)
    n = len(M)

    X1 = np.ones(n)
    X2 = M - m
    X3 = (M-m)**2
    X4 = T-t
    X5 = (T-t)**2
    X6 = X2*X4
    X = np.array([X1, X2, X3, X4, X5, X6]).T

    # the kernel lays on M_j - m, T_j - t
    #ker = new_epanechnikov(X[:,5])
    ker = kernel(M, m, h_m, T, t, h_t)
    #test = gausskernel(X[:,5])
    W = np.diag(ker)

    # Compare Kernels
    # This kernel gives too much weight on far-away deviations
    #plt.scatter(M, ker, color = 'green')
    #plt.scatter(M, X[:,5], color = 'red')
    #plt.vlines(m, ymin = 0, ymax = 1)
    #plt.show()

    XTW = np.dot(X.T, W)

    beta = np.linalg.pinv(np.dot(XTW, X)).dot(XTW).dot(y)

    # This is our estimated vs real iv 

    #iv_est = np.dot(X, beta)
    #plt.scatter(df.moneyness, df.mark_iv, color = 'red')
    #plt.scatter(df.moneyness, iv_est, color = 'green')
    #plt.vlines(m, ymin = 0, ymax = 1)
    #plt.title('est vs real iv and current location')
    #plt.show()
    

    return beta[0], beta[1], 2*beta[2]

def extend_polynomial(x, y):
    """
    Extend Smile, first and second derivative so that spd exists completely for large tau
    x = M_std
    y = first
    """
    polynomial_coeff=np.polyfit(x,y,2)
    xnew=np.linspace(0.6,1.4,100)
    ynew=np.poly1d(polynomial_coeff)
    #plt.plot(xnew,ynew(xnew),x,y,'o')
    #plt.title('interpolated smile')
    #plt.show()
    return xnew, ynew(xnew)

def rookley_unique_tau(df, h_m, h_t=0.01, gridsize=149, kernel='epak', boot_iv = None):
    # gridsize is len of estimated smile

    if kernel=='epak':
        kernel = epanechnikov
    elif kernel=='gauss':
        kernel = gaussian_kernel
    else:
        print('kernel not know, use epanechnikov')
        kernel = epanechnikov

    num = gridsize
    
    M_min, M_max = min(df.moneyness), max(df.moneyness)
    M = np.linspace(M_min, M_max, gridsize)
    M_std_min, M_std_max = min(df.moneyness), max(df.moneyness)
    M_std = np.linspace(M_std_min, M_std_max, num=num)


    # if all taus are the same
    tau_min, tau_max = min(df.tau[(df.tau > 0)]), max(df.tau) # empty sequence for tau precision = 3
    tau = np.linspace(tau_min, tau_max, gridsize)

    x = zip(M_std, tau)
    sig = np.zeros((num, 3)) # fill

    for i, (m, t) in enumerate(x):
        sig[i] = smoothing_rookley(df, m, t, h_m, h_t, kernel, extend = False, boot = boot_iv)

    smile = sig[:, 0]
    first = sig[:, 1] #/ np.std(df.moneyness)
    second = sig[:, 2] #/ np.std(df.moneyness)

    #plt.plot(df.moneyness, df.mark_iv, 'ro', ms=3, alpha=0.3, color = 'green')
    #plt.title('moneyness vs iv: base for interpolation')
    #plt.show()

    #plt.plot(M_std, smile, 'ro', ms=3, alpha=0.3, color = 'red')
    #plt.plot(M_std, first)
    #plt.plot(M_std, second)
    #plt.title('no interpolation')
    #plt.show()

    #plt.plot(M_std, smile)
    #M_std, smile = extend_polynomial(M_std, smile)
    #M_std, first = extend_polynomial(M_std, first)
    #M_std, second = extend_polynomial(M_std, second)

    S_min, S_max = min(df.index_price), max(df.index_price)
    K_min, K_max = min(df.strike), max(df.strike)
    S = np.linspace(S_min, S_max, gridsize)
    K = np.linspace(K_min, K_max, gridsize)

    return smile, first, second, M, S, K, M_std, tau    


def rookley(df, h_m, h_t=0.01, gridsize=149, kernel='epak'):
    # gridsize is len of estimated smile

    if kernel=='epak':
        kernel = epanechnikov
    elif kernel=='gauss':
        kernel = gaussian_kernel
    else:
        print('kernel not know, use epanechnikov')
        kernel = epanechnikov

    num = gridsize
    #tau = df.tau.iloc[0]
    M_min, M_max = min(df.moneyness), max(df.moneyness)
    M = np.linspace(M_min, M_max, gridsize)
    M_std_min, M_std_max = min(df.moneyness), max(df.moneyness)
    M_std = np.linspace(M_std_min, M_std_max, num=num)

    # if all taus are the same
    tau_min, tau_max = min(df.tau[(df.tau > 0)]), max(df.tau)
    tau = np.linspace(tau_min, tau_max, gridsize)

    x = zip(M_std, tau)
    sig = np.zeros((num, 3)) # fill

    # TODO: speed up with tensor instead of loop
    for i, (m, t) in enumerate(x):
        sig[i] = smoothing_rookley(df, m, t, h_m, h_t, kernel)

    smile = sig[:, 0]
    first = sig[:, 1] #/ np.std(df.moneyness)
    second = sig[:, 2] #/ np.std(df.moneyness)

    #plt.plot(df.moneyness, df.mark_iv, 'ro', ms=3, alpha=0.3, color = 'green')
    #plt.plot(df.moneyness, smile, 'ro', ms=3, alpha=0.3, color = 'red')
    #plt.plot(df.moneyness, first)
    #plt.plot(df.moneyness, second)
    #plt.show()

    S_min, S_max = min(df.index_price), max(df.index_price)
    K_min, K_max = min(df.strike), max(df.strike)
    S = np.linspace(S_min, S_max, gridsize)
    K = np.linspace(K_min, K_max, gridsize)

    return smile, first, second, M, S, K, M_std, tau

def compute_spd(sigma, sigma1, sigma2, tau, s, m2, r_const, mindate, plot_ident):

    # SPDBL
    # Scalars
    #tau = np.mean(tau)
    #s = np.mean(s)
    r = r_const * len(s)
    #r = np.mean(r)
    #r = 0

    # now start spdbl estimation
    st = np.sqrt(tau)
    ert = np.exp(r * tau)
    rt = r * tau
    # error should be here in the length of m
    d1 = (np.log(m2) + tau * (r + 0.5 * (sigma ** 2))) / (sigma * st)
    d2 = d1 - (sigma * st)
    f = stats.norm.cdf(d1, 0, 1) - (stats.norm.cdf(d2, 0, 1)/(ert * m2))

    # First derivative of d1
    d11 = (1/(m2*sigma*st)) - (1/(st*(sigma**2))) * ((np.log(m2) + tau * r) * sigma1) + 0.5 * st * sigma1
    
    # First derivative of d2 term
    d21 = d11 - (st * sigma1)
    
    # Second derivative of d1 term
    d12 = -(1/(st * (m2**2) * sigma)) - sigma1/(st * m2 * (sigma**2)) + sigma2 * (0.5 * st - (np.log(m2) + rt)) / (st * sigma**2) + sigma1 * (2 * sigma1 * (np.log(m2) + rt)) / (st * sigma**3) - 1/(st * m2 * sigma**2)

    # Second derivative of d2 term
    d22 = d12 - (st * sigma2)

    f1 = (stats.norm.pdf(d1, 0, 1) * d11) + (1/ert) * ((-stats.norm.pdf(d2, 0, 1) * d21)/m2 + stats.norm.cdf(d2, 0, 1) / m2**2)
    
    # f2 = dnorm(d1, mean = 0, sd = 1) * d12 - d1 * dnorm(d1, mean = 0, sd = 1) * (d11^2) - (1/(ert * m) * dnorm(d2, mean = 0, sd = 1) * d22) + ((dnorm(d2, mean = 0, sd = 1) * d21)/(ert * m^2)) + (1/(ert * m) * d2 * dnorm(d2, mean = 0, sd = 1) * (d21^2)) - (2 * pnorm(d2, mean = 0, sd = 1)/(ert * (m^3))) + (1/(ert * (m^2)) * dnorm(d2, mean = 0, sd = 1) * d21)
    f2 = stats.norm.pdf(d1, 0, 1) * d12 - d1 * stats.norm.pdf(d1, 0, 1) * d11**2            - (1/(ert * m2) * stats.norm.pdf(d2, 0, 1) * d22) + ((stats.norm.pdf(d2, 0, 1)*d21)/(ert * m2**2))      + (1/(ert * m2) * d2 * stats.norm.pdf(d2, 0, 1)) * d21**2 -(2 * stats.norm.cdf(d2, 0, 1)/(ert * m2**3))     + (1/(ert * m2**2)) * stats.norm.pdf(d2, 0, 1) *d21
   
    # recover strike price
    x = s/m2
    c1 = -(m2**2) * f1
    c2 = s * (1/x**2) * ((m2**2) * f2 + 2 * m2 * f1)

    # Calculate the quantities of interest
    cdf = ert * c1 + 1
    fstar = ert * c2
    delta = f + s + f1/x
    gamma = 2 * f1 / x + s * f2 / (x**2)
    #print('\ndelta: ', delta, '\ngamma:', gamma)

    #plt.plot(fstar)
    #plt.show()

    spd = pd.DataFrame({'x': np.mean(s)/m2, 
                        'y': fstar,
                        'm': m2}) # Moneyness

    #plot_spd(spd, mindate, tau, plot_ident)

    return spd  

def wild_bootstrap(x, n_boot):
    """
    Wild Bootstrap
    """
    n = len(x)

    # Golden Section Bootstrap
    sq = np.sqrt(5)
    a = (1-sq)/2
    b = a + sq
    c = (5 + sq)/10

    n_draws = n * n_boot
    unif = np.random.uniform(0, 1, n_draws)
    mat = unif.reshape(n, n_boot)
    smaller = np.where(mat < c, 0, a)
    greater = np.where(mat > c, 0, b)
    out = (greater + smaller).T * x.values

    return out.T


def kde2D(x, y, h, xbins=100j, ybins=100j, likelihood = False, **kwargs): 
    """Build 2D kernel density estimate (KDE)."""

    # create grid of sample locations (default: 100x100)
    xx, yy = np.mgrid[x.min():x.max():xbins, 
                      y.min():y.max():ybins]

    xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
    xy_train  = np.vstack([y, x]).T

    kde_skl = KernelDensity(bandwidth=h, **kwargs)
    kde_skl.fit(xy_train)

    # score_samples() returns the log-likelihood of the samples
    if likelihood:
        z = kde_skl.score_samples(xy_train)
        return xy_train[:,:1].flatten(), xy_train[:,1:2].flatten(),  z
    else:
        z = np.exp(kde_skl.score_samples(xy_train))
        return xy_train[:,:1].flatten(), xy_train[:,1:2].flatten(),  z


def fisher_information_prep(sub):
    x = sub.strike
    y = sub.instrument_price
    h = sub.shape[0] ** (-1/9)

    xx, yy, zz = kde2D(x, y, h, likelihood = True)

    # Derivatives of Likelihood Function
    # Respect to y
    dz_dy = np.gradient(zz, yy)
    dzdz_dydy = np.gradient(dz_dy, yy)

    # Respect to z
    dz_dx = np.gradient(zz, xx)
    dzdz_dxdx = np.gradient(dz_dx, xx)

    # Gradient (dz_dx, dz_dy)
    dz_dx_max_loc = np.nanargmax(dz_dx)
    xx_opt = zz[dz_dx_max_loc]

    dz_dy_max_loc = np.nanargmax(dz_dy)
    yy_opt = zz[dz_dy_max_loc]
    
    # Gradient
    grad = np.array([[xx_opt], [yy_opt]])

    # Hessian

    # Get other partial derivatives
    # Check if they are teh same
    dzdz_dydx = np.gradient(dz_dy, xx)
    dzdz_dxdy = np.gradient(dz_dx, yy)

    # Evaluate at max
    A = [dzdz_dxdx[np.nanargmax(dzdz_dxdx)], 
        dzdz_dxdy[np.nanargmax(dzdz_dxdy)]]

    B = [dzdz_dydx[np.nanargmax(dzdz_dydx)], 
        dzdz_dydy[np.nanargmax(dzdz_dydy)]]
    hessian = np.array([A,B])

    # Estimated Variance Contribution
    v_hat = np.dot(grad, np.transpose(grad))

    # Esimated J Contribution
    j_hat = hessian

    return grad, hessian, v_hat, j_hat

def plot_kde(sub):
    x = sub.strike
    y = sub.instrument_price
    h = sub.shape[0] ** (-1/9)

    xx, yy, zz = kde2D(x, y, h)

    # Derivatives of Kernel Density --> BUT WE NEED DERIVATIVES OF THE LIKELIHOOD FUNCTION!
    dz_dy_first = np.diff(zz)/np.diff(yy)
    dz_dy_second = np.diff(dz_dy_first)/(np.diff(yy[1:]))**2
    dz_dy_second = np.pad(dz_dy_second, (1,0))
    first_deriv = np.gradient(zz, yy)
    second_deriv = np.gradient(first_deriv, yy)
    
    fig, ax = plt.subplots()
    #plt.scatter(yy, zz, label = 'kd')
    plt.scatter(yy, first_deriv, label = 'first deriv')
    plt.scatter(yy, second_deriv, label = 'second deriv')
    plt.scatter(yy[1:], dz_dy_first, label = 'first')
    plt.scatter(yy[1:], dz_dy_second, label ='second')
    plt.savefig('derivatives.png')

    #fig = plt.figure()
    #ax = fig.add_subplot(projection='3d')
    #    ax.contour(xx,yy,zz)
    col = pd.DataFrame({'x': xx, 'z':zz}).sort_values('x')
    
    fig, ax = plt.subplots()
    plt.plot(col['x'], col['z'])

    #fig, ax = plt.figure(), plt.subplot(111)
    #plt.pcolormesh(xx, yy, zz)
    #plt.scatter(x, y, s=2, facecolor='white')
    plt.savefig('kdeplot.png')
    return xx, yy, zz

def resample(sub):
    """
    Goal: Resampling as in W2s Bootstrap Section
    Get (X,Y) = (Strike, Price)
    So get the bivariate distribution first
    """
    n = sub.shape[0]
    h = n ** (-1/9)
    est_std = sub.std()
    strike_std = est_std['strike']
    price_std = est_std['instrument_price']

    dom = np.linspace(0, 1, 1000)
    strike_domain = np.linspace(0, 100000, 1001)
    price_domain = strike_domain#np.linspace(0, 100000, 1000001)
    X = np.array(sub.strike)
    Y = np.array(sub.instrument_price)
    
    z = zip(Y, X)
    #sig = np.zeros((num, 3)) # fill
    out = []

    og_sample = sub[['strike', 'instrument_price']]
    x = np.array(og_sample['strike'])
    y = np.array(og_sample['instrument_price'])

    a = np.vstack([x.ravel(),y.ravel()]).T
    kd = KernelDensity(kernel='gaussian', bandwidth = h).fit(a)
    s = kd.score_samples(a)
    z = np.exp(s)

    fig, ax = plt.subplots()
    #plt.pcolormesh([x, y], z)
    plt.scatter(x, y, z)
    plt.savefig('kdsample.png')

    # TODO: speed up with tensor instead of loop
    for i, (y, x) in enumerate(z):
        gk = gaussian_kernel_mod(strike_domain, x, h, strike_std, price_domain, y, h, price_std)
        s = sum(gk)
        o = s * (strike_std / (n * h * h * price_std))


        out.append(gk)
    
    return None

def filter_sub(_df,  mindate, maxdate, tau):
    # Subset

    # Only calls, tau in [0, 0.25] and fix one day (bc looking at intra day here)
    # (_df['is_call'] == 1) & --> Dont need to filter for calls because that happens in the R script
    # Also need to consider put-call parity there
    sub = _df[(_df['date'] > mindate) & (_df['date'] < maxdate) & (_df['moneyness'] >= 0.7) & (_df['moneyness'] < 1.3) &(_df['mark_iv'] > 0)]# &

    if tau > 0:
        sub = sub[sub['tau'] == tau]

    nrows = sub.shape[0]
    if nrows == 0:
        raise(ValueError('Sub is empty'))
   
    sub['moneyness'] = round(sub['moneyness'], 3)
    sub['index_price'] = round(sub['index_price'], 2)

    sub = sub.drop_duplicates()
    
    print(sub.describe())

    if nrows > 50000:
        print('large df, subsetting')
        sub = sub.sample(10000)
        print(sub.describe())

    return sub

def log_option_stats(df, currdate, vnames = ['moneyness', 'mark_iv', 'tau']):
    """
    Need Maturity in Weeks, Min, Max, Mean, Num for OTM and ITM options respectively
    Maturity, Moneyness, IV
    """
    print('logging')
    itm = df[(df['moneyness'] < 1)]
    otm = df[(df['moneyness'] >= 1)]
    
    summarystats = {}
    for ele in [itm, otm]:
        g = ele.groupby(['nweeks'])

        for v in vnames:
            keyname = str(ele) + '-' + str(currdate) + '-' + str(v)
            summarystats[keyname] = g[v].describe()





def spdbl(_df, mindate, maxdate, tau, r_const, blockplots, bootstrap, physical_density = None):
    """
    computes spd according to Breeden and Litzenberger 1978
    """
    plot_ident = '2020-' + str(mindate.month) + '-' + str(mindate.day) + '-' + str(tau) + '.png'
    
    sub = filter_sub(_df, mindate, maxdate, tau)

    # Log Descriptive Stats 
    # Table 1 


    tau = sub['tau'].astype(float)
    #m = float(sub['index_price']/sub['strike'] # Spot price corrected for div divided by k ((s-d)/k); moneyness of an option; div always 0 here
    r = sub['interest_rate'].astype(float)
    s = sub['index_price'].astype(float)
    k = sub['strike'].astype(float)
    m = k / s
    div = 0
    
    # Forward price
    F = s * np.exp((r-div) * tau)
    K = m * F # k capital
    newm =(s * np.exp(-div*tau))/K

    sigma = []
    sigma1 = []
    sigma2 = []

    n_rows = sub.shape[0]
    h = n_rows ** (-1 / 9)
    print('Choosing Bandwidth h: ', h)
    sigma, sigma1, sigma2, M, S, K, M_std, newtau = rookley_unique_tau(sub, h, gridsize = n_rows)#rookley(sub, h) 

    # Empirical Vola Smile
    # @Todo: Conflict with new tau being just zeros
    #trisurf(newm, tau, vola, 'moneyness', 'tau', 'vola', 'pricingkernel/plots/empirical_vola_smile_' + plot_ident, blockplots)

    # Projected Vola Smile
    plot_volasmile(M, sigma, sigma1, sigma2, mindate, plot_ident, blockplots)  

    # tau is too long here!!
    spd = compute_spd(sigma, sigma1, sigma2, newtau,  S, M, r_const, mindate, plot_ident)

    if bootstrap:
        # Require physical density
        digits = 2 # For Rounding
        spd = merge_spd_physical(spd, physical_density['gdom'], physical_density['gdom_moneyness'], physical_density['gstar'], digits)

        print("bootstrap")
        n_bootstrap_samples = 100
        eps = sigma - sub['mark_iv']
        eps_cent = eps - np.mean(eps)
        boot_eps = wild_bootstrap(eps, n_bootstrap_samples)

        # Add bootstrapped samples to the estimated IV surfaces
        # 100 columns 
        sigma_samples = (np.array([sigma] * n_bootstrap_samples).T * boot_eps).T

        # Calculate Pricing Kernel
        spd['pk'] = spd['spdy']/spd['gy']

        # Bivariate Density Estimation
        #tt = plot_kde(sub)

        # Replace in sub
        z = []
        v, j = [], []
        for i in range(n_bootstrap_samples):
            sample_sub = sub.sample(frac=0.5, replace = False)

            # Prepare Fisher from resampled sub
            grad, hessian, v_hat, j_hat = fisher_information_prep(sample_sub)
            if np.isnan(v_hat).any() or np.isnan(j_hat).any() or np.isinf(v_hat).any() or np.isinf(j_hat).any():
                print('quitting loop')
                continue
            else:
                v.append(v_hat)
                j.append(j_hat)

            sigma_b, sigma1_b, sigma2_b, M_b, S_b, K_b, M_std_b, newtau_b = rookley_unique_tau(sample_sub, h, gridsize = n_rows)# # boot_iv = sigma_samples[i]
            spd_b = compute_spd(sigma_b, sigma1_b, sigma2_b, newtau_b,  S_b, M_b, r_const, mindate, plot_ident)
            
            # Ensure same length
            spd_b['m'] = round(spd_b['m'], digits)
            spd_b_g = spd_b.groupby(['m'])['y'].sum().reset_index()
            spd_b = spd_b_g[(spd_b_g.m > 0.8) & (spd_b_g.m < 1.2)]

            # Merge bootstrapped spd and spd dfs
            comb = spd.merge(spd_b, on ='m', how = 'inner')

            # Need to include pk_std here
            # Find Gumbel-distributed max of sample
            sample_maximum = max((abs(comb['spdy'] - comb['y']))) #times pk_std 
            print(sample_maximum)
            z.append(sample_maximum)
        
        v_n_theta = sum(v)
        # J still has some annoying inf values in there!
        j_n_theta = sum(j)
        j_inv = np.linalg.inv(j_n_theta)

        test = np.dot(np.dot(j_inv, v_n_theta), j_inv)


        # Find Quantiles
        alpha = 0.1
        alpha_half = alpha/2
        tmlow = np.quantile(z, alpha_half)
        tmup = np.quantile(z, 1-alpha_half)

        print('\ntmup: ', tmup, '\ntmlow: ', tmlow)

        spd['cb_up'] = spd['pk'] + tmup
        spd['cb_dn'] = spd['pk'] - tmup

    return spd, sub

def classify_options(dat):
    """
    Classify Options to prepare range trading
    """
    dat['option_type'] = ''
    dat['option_type'][(dat['moneyness'] < 0.9)]                                = 'FOTM Put'
    dat['option_type'][(dat['moneyness'] >= 0.9) & (dat['moneyness'] < 0.95)]   = 'NOTM Put'
    dat['option_type'][(dat['moneyness'] >= 0.95) & (dat['moneyness'] < 1)]     = 'ATM Put'
    dat['option_type'][(dat['moneyness'] >= 1) & (dat['moneyness'] < 1.05)]     = 'ATM Call'
    dat['option_type'][(dat['moneyness'] >= 1.05) & (dat['moneyness'] < 1.1)]   = 'NOTM Call'
    dat['option_type'][(dat['moneyness'] >= 1.1)]                               = 'FOTM Call'
    return dat

def verify_density(s):
    # Check if integral is 1
    rev = s.iloc[::-1]
    area = simps(y = rev['y'], x = rev['x'])
    print('Area under SPD: ', area)
    return area

def option_pnl(underlying_at_maturity, strikes, premium_per_strike, call = True, long = True):

    premia = sum(premium_per_strike)
    profit = []
    if call:
        for strike in strikes:
            profit.append(max(underlying_at_maturity - strike, 0))
    else:
        for strike in strikes:
            profit.append(max(strike - underlying_at_maturity, 0))
    
    p = sum(profit)
    if isinstance(p, np.ndarray):
        p = p.astype(float)[0]
    if long:
        return p, -1 * premia
    else:
        return -1 * p, premia

def hist_iv(df):
    sub = df[df.mark_iv > 0]
    sub = sub.sort_values('maturitydate_char')
    #o = sub.groupby(['maturitydate_char', 'instrument_name', 'date']).mean()['mark_iv']
    o = sub.groupby(['maturitydate_char', 'instrument_name']).mean()['mark_iv']
    return o.to_frame()

def plot_atm_iv_term_structure(df, currdate, bw, ymin = 0, ymax = 0.85):

    datestr = currdate.strftime("%Y-%m-%d")

    # Set namesmw
    fname = 'termstructure/term_structure_atm_options_on_-' + datestr + str(bw) + '.png'
    titlename = 'IV Term Structure of ATM' + ' Options on '+ datestr

    df.mark_iv = df.mark_iv / 100 # Rescale 
    call_sub = df[(df.mark_iv > 0) & (df.is_call == 1) & (df.moneyness <= 1.1) & (df.moneyness >= 0.9)]
    put_sub = df[(df.mark_iv > 0) & (df.is_call == 0) & (df.moneyness <= 1.1) & (df.moneyness >= 0.9)]
    
    # nimm einfach mal n tau ueber das man die iv plotten kann
    fig = plt.figure()
    ax = plt.subplot(111) #plt.axis()
    call_sub.groupby(["tau"])['mark_iv'].mean().plot()
    #put_sub.groupby(["tau"])['mark_iv'].mean().plot(label = 'Put IV')

    ax.set_ylabel('Implied Volatility')
    ax.set_xlabel('Tau')
    plt.title(titlename)
    plt.ylim(ymin, ymax)

    # Shrink current axis by 20%
    #box = ax.get_position()
    #ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
    #  Put a legend to the right of the current axis
    #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.savefig(fname, transparent = True)
    #plt.show(block = False)
    plt.draw()

    return None

def real_vola(df):
    o = df.groupby(['maturitydate_char']).std()['index_price'] * np.sqrt(252)
    return o.to_frame()

def iv_vs_real_vola(realized_vola, historical_iv):
    plt.plot(realized_vola)
    plt.plot(hist_iv)

def merge_spd_physical(spd, gdom, gdom_moneyness, gstar, round_digits):
    g = pd.DataFrame({'x' : gdom, 'gy' : gstar})
    s = spd.rename(columns = {'y' : 'spdy'})
    merged = s.merge(g, on='x', how = 'inner')
    merged['m'] = round(merged['m'], 2)
    out = merged.sort_values('m') 

    return out[(out.m > 0.8) & (out.m < 1.2)]

def prepare_confidence_band_data(df):

            # Old
                # Save for Confidence Band 
                    # Column 1 - Day
                    # Column 2 - Month
                    # Column 3 - Year
                    # Column 4 - Call Indicator
                    # Column 5 - Time Maturity in Days
                    # Column 6 - Strike
                    # Column 7 - IV
                    # Column 8 - Spot
                    # Column 9 - Risk Free Rate
                    # Column 10 - 
                    # Column 11 - Forward
                # 3	1	1997	0	49	3025	165.8	2861.706	0.031465	0.150026951	2863.26
                # Day Month Year Put Days Strike IV     Spot         Risk Free    Tau in Days (255)       Forward

    """
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
    """
    b = df[['date', 'mark_iv', 'is_call', 'tau', 'strike', 'underlying_price', 'interest_rate', 'moneyness']]
    b['placeholder'] = 0
    b = b.reset_index(drop=True)
    b['interest_rate'] = 0.001
    b['forward'] = 1

    # Reorder
    df_c = b[['date', 'mark_iv', 'is_call', 'tau', 'strike', 'placeholder', 'underlying_price', 'interest_rate', 'moneyness']]

    unique_maturities = df.days_to_maturity.unique()[0]
    fname = str('tmp/confidence_band_input_') + str(unique_maturities) + str('.csv')
    df_c.to_csv(fname, index = False)
    return fname

def bootstrap(conf_fname, phys_fname, rdate, tau, simmethod, r, out_dir, bw):
                            
    try:
        # MAKE SURE TO USE DIFFERENT 2ND INPUT FOR SP500!!
        moneyness, spd, epk, spdlo, spdup, epk_BS, tau = r.bootstrap_epk(robjects.StrVector([conf_fname]), 
                                                            robjects.StrVector([phys_fname]),
                                                            robjects.StrVector(rdate),
                                                            robjects.StrVector([out_dir]),
                                                            robjects.FloatVector([bw])
                                                            )
        # Can recover PD here as SPD/EPK
        spd_df = pd.DataFrame({'m': moneyness,
                                'spdy': spd,
                                'cb_up': spdup,
                                'cb_dn': spdlo,
                                'pk': epk})

        return spd_df, tau

    except Exception as e:
        print('Sim failed, proceeding with the next one... - ', simmethod)
        print('exception: ', e)
                

def plot_epks(spd_btc, spd_sp500, btc_tau, sp500_tau, simmethod, curr_date, bw):
    
    if abs(btc_tau - sp500_tau) <= 0.03:

        # Plot Bootstrap Pricing Kernel
        fig = plt.figure()
        ax = plt.subplot(111)
        plt.plot(spd_btc['m'], spd_btc['pk'], color = 'red')
        plt.plot(spd_btc['m'], spd_btc['cb_up'], color = 'orange', linestyle='--', dashes=(5, 5))
        plt.plot(spd_btc['m'], spd_btc['cb_dn'], color = 'orange', linestyle='--', dashes=(5, 5))
        plt.plot(spd_sp500['m'], spd_sp500['pk'], color = 'green')
        plt.plot(spd_sp500['m'], spd_sp500['cb_up'], color = 'blue', linestyle='--', dashes=(5, 10))
        plt.plot(spd_sp500['m'], spd_sp500['cb_dn'], color = 'blue', linestyle='--', dashes=(5, 10))
        plt.ylim(0, 10)
        plt.xlim(0.9, 1.1)
        plt.xlabel('Moneyness')

        fname = 'pricingkernel/plots/' + str(bw) + '/bootstrap_epk_' + 'tau-deribit:' + str(btc_tau) + '_sp:' + str(sp500_tau) + '_date-' + str(curr_date) + '.png'
        print(fname)
        plt.savefig(fname, transparent = True)


def hockeystick(df, tau, curr_date, bw):
    """


    """

    ord = df.sort_values('strike')
    calls = ord[(ord['is_call'] == 1)]
    puts = ord[(ord['is_call'] == 0)]

    fig = plt.figure()
    plt.subplot(121)
    plt.scatter(calls['strike'], calls['instrument_price'], color = 'blue')
    plt.subplot(122)
    plt.scatter(puts['strike'], puts['instrument_price'], color = 'green')

    #ax = plt.axes(projection='3d')
    #ax.scatter3D(df['strike'], df['tau'], df['instrument_price'])
    #plt.ylim(0, 10)
    #plt.xlim(0.9, 1.1)
    #plt.xlabel('Moneyness')

    fname = 'hockeysticks/_bw-' + str(bw) + '_tau-' + str(tau) + '_date-' + str(curr_date) + '.png'
    print(fname)
    plt.savefig(fname, transparent = True)

def term_structure(df, curr_date, bw):
    """
    ATM Call Prices for different clusters of Tau
    Weeks until Maturity: 1, 2, 4, 8
    """
    # Last price is spot of highest timestamp
    max_ts_idx = df['timestamp'].idxmax()
    last_spot = df.loc[max_ts_idx]['index_price']
    
    # Find Closest Strike to last Spot
    strikes = df['strike'].unique()
    dist = abs(last_spot - strikes)
    min_dist_idx = np.argmin(dist)
    atm_strike = strikes[min_dist_idx]

    # Get Term Structure
    atm = df[df['strike'] == atm_strike]
    atm['iv'] = atm['mark_iv'].astype(float) / 100
    
    fig = plt.figure()
    plt.subplot(111)
    plt.scatter(atm['nweeks'], atm['iv'], color = 'blue')
    plt.ylim(0, 0.85)

    #ax = plt.axes(projection='3d')
    #ax.scatter3D(df['strike'], df['tau'], df['instrument_price'])
    #plt.ylim(0, 10)
    #plt.xlim(0.9, 1.1)
    #plt.xlabel('Moneyness')

    fname = 'termstructure/_bw-' + str(bw) + '_date-' + str(curr_date) + '.png'
    print(fname)
    plt.savefig(fname, transparent = True)

    return None

def run(curr_day):
    print("Entering Main Loop")
    
    r_bandwidth = 0.08

    profit = {}
    errors = []
    realized_vola = []
    historical_iv = []

    # Collect Trading Summary
    trading = pd.DataFrame()

    # Initiate R Objects
    base = rpackages.importr("base") 
    r = robjects.r
    r.source('physicaldensity.R')
    r.source('src/confidence_bands.R')
    r.source('epk_bootstrap_main.R')

    # Read final prices
    #tdat = pd.read_csv('data/BTC_USD_Quandl.csv')#pd.read_csv('data/BTCUSDT.csv')
    #tdat = tdat.sort_values('Date', ascending = True) # ascending
    tdat = get_btc_prices()
    tdat[::-1].to_csv('Data/BTC_USD_Quandl.csv', index = False)

    # Initiate BRC instance to query data. First and Last day are stored.
    brc = BRC()

    # Simulation Methods
    simmethods = ['oldschool'] # ['brownian', 'svcj']

    # Tau-maturitydate combination dict
    tau_maturitydate = {}

    if curr_day < brc.last_day:
        print(curr_day)
        
        try:
            # make sure days are properly set
            curr_day_starttime = curr_day.replace(hour = 0, minute = 0, second = 0, microsecond = 0)
            curr_day_endtime = curr_day.replace(hour = 23, minute = 59, second = 59, microsecond = 0)

            # Debug
            #curr_day_starttime = datetime.datetime(2020, 4, 5, 0, 0, 0)
            #curr_day_endtime = datetime.datetime(2020, 4, 5, 23, 59, 59)

            print('\nStarting Simulation from ', curr_day_starttime, ' to ', curr_day_endtime)
            
            dat, int_rate = brc._run(starttime = curr_day_starttime,
                                        endtime   = curr_day_endtime, 
                                        download_interest_rates = False,
                                        download_historical_iv = False,
                                        insert_to_db = '') # insert_to_db = '' if not desired to save collection
            dat = pd.DataFrame(dat)

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
            
                        
            ## Isolate vars
            df['mark_iv'] = df['mark_iv'] / 100
            df['mark_iv'][(df['mark_iv'] < 0.01)] = 0
            vola = df['mark_iv'].astype(float)/100


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

            
            print('here')
            print('starting to update')
            for row in df.iterrows():
                record = row[1].to_dict()
                _id = record['_id']
                del record['_id']
                #print(_id)
                #result = db.testcollection.replace_one({'ItemId': record.get('ItemId'), 'ParentID': record.get('ParentID')}, record, upsert=True)
                #print(f'{"Replaced: " if result.modified_count == 1 else ""}{"Inserted: " if result.upserted_id is not None  else ""} {record}')
               
                tst = brc.update_collection.insert_one(record)
                brc.update_collection.update_one({
                #brc.db['updatetest'].update_one({
                    '_id': _id#record.get('_id')
                    },{
                    '$set': {
                        'moneyness': record.get('moneyness'),
                        'strike': record.get('strike'),
                        'date': record.get('date'),
                        'nweeks': record.get('nweeks'),
                        'tau': record.get('tau')
                    }
                    }, upsert=True)
            
            #term_structure(df, curr_day, r_bandwidth)
            plot_atm_iv_term_structure(df, curr_day, r_bandwidth)
           
            f = filter_sub(df, curr_day_starttime, curr_day_endtime, 0)
            f.to_csv('out/filtered_' + str(curr_day_starttime) + '.csv')
            #trisurf(f['moneyness'], f['tau'], f['mark_iv'], 'moneyness', 'tau', 'vola', 'pricingkernel/plots/empirical_vola_smile_' + curr_day_starttime.strftime('%Y-%m-%d'), False)


        except Exception as e:
            print('Download or Processing failed!\n')
            print(e)  

        # As taus are ascending, once we do not find one instrument for a specific taus it is unlikely to find one for the following
        # as the SPDs degenerate with higher taus.
        for tau in unique_taus:
            try:
                #r.XFGSPDcb2()
                print(tau)

                sub = filter_sub(df, curr_day_starttime, curr_day_endtime, tau)

                # Prepare Confidence Band Calculation for the whole Day
                conf_fname = prepare_confidence_band_data(sub)

                # last one on day which we observed
                observation_price = sub['index_price'].tail(1) 

                # need at least one day for the physical density, which is fixed in there!
                time_to_maturity_in_days = sub.days_to_maturity.unique()[0]
                rdate = base.as_Date(curr_day.strftime('%Y-%m-%d'))
                rdate_f = base.format(rdate, '%Y-%m-%d')

                # Only continue if spd fulfills conditions of a density
                # and we have more than 1 day until maturity
                if time_to_maturity_in_days > 1:

                    for simmethod in simmethods:

                        print(conf_fname)
                        
                        # For Mongo deribit_orderbooks
                        spd_btc, tau_btc = bootstrap(conf_fname, 'data/BTC_USD_Quandl.csv', rdate_f, tau, simmethod, r, 'out/deribit/', r_bandwidth)
                        
                        if spd_btc is not None:

                            # Saving Moneyness, SPD, PK, Confidence Bands
                            spd_btc.to_csv('out/movies/btc_pk_' + str(tau) + '_' + str(curr_day_starttime) + '.csv')
                            
                            # Also save data for Vola Smile and Term Structure
                            sub.to_csv('out/movies/sub_' + str(tau) + '_' + str(curr_day_starttime) + '.csv')
                        
                else:
                    print('SPD is not a valid density, proceeding with the next one')

            except Exception as e:
                print('error: ', e)
                errors.append(e)
                with open("out/errors.txt", "wb") as fp:   #Pickling
                    pickle.dump(errors, fp)

        #finally:
        #    curr_day += datetime.timedelta(1)


if __name__ == '__main__':
    print('starting non multi main...')

    brc = BRC()
    
    # Start in Q1 2021 for exactly one year
    # Paper Start
    startdate = datetime.datetime(2021,4,1)
    curr_date = startdate
    # Paper End
    enddate = datetime.datetime(2022,4,1)
    run_dates = [curr_date]
    
    while curr_date < enddate:
        curr_date += datetime.timedelta(1)
        run_dates.append(curr_date)

    for d in run_dates:
        try:
            print(d)
            run(d)
        except Exception as e:
            print('error in : ', e)
        
