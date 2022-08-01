import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pdb

from src.helpers import save_dict, load_dict

class IronCondor:
    """
    Trade one long call spread and one long put spread
    Name of this Strategy?
    """

    def __init__(self, s, gdom, gstar, tau, date, dat, tdat, time_to_maturity_in_days, simmethod, tradedirection = 'long'):
        self.tradedirection = tradedirection
        self.simmethod      = simmethod 
        self.s              = s
        self.time_to_maturity_in_days = time_to_maturity_in_days
        self.gdom   = gdom
        self.gstar  = gstar
        self.tau    = tau
        self.date   = date
        self.dat    = dat
        self.tdat   = tdat
        self.allowed_position_size  = 1
        self.positions              = ['long_call', 'short_call', 'long_put', 'short_put']
        self.max_price = max(self.dat['underlying_price'])
        self.settlement_states      = np.arange(0, self.max_price * 1.5, 100).tolist() # make this as long as the prices!
        self.payoff_per_state       = []
        self.curr_day               = date
        
        print('Trading ' + self.tradedirection + ' Iron Condor ' + self.simmethod)
        unique_maturities = self.dat[(self.dat.tau == self.tau)]['maturitydate_char'].unique()
        print('unique maturities in strategy: ', unique_maturities)
     
        if len(unique_maturities) != 1:
            print('maturity conflict : ', unique_maturities)
            # If this occurs, then probably tdat is not updated to the extent of our orderbooks
            #pdb.set_trace()
            raise ValueError('must not happen, maturity too long')
        self.maturity = unique_maturities[0] # need this for dict key

        """
        sorted_by_time = self.dat[self.dat.tau == self.tau].sort_values('timestamp')
        self.underlying_price_at_emission = sorted_by_time['underlying_price'].iloc[-1]
        self.settlement = self.tdat[(self.tdat.Date == self.maturity)]

        print('Contracts settling on: ', self.settlement['Date'].iloc[-1], ' for ', self.settlement['Adj.Close'].iloc[-1])
        self.settlement_price = self.settlement['Adj.Close'].iloc[-1]

        """

        # Alternatively: choose self.dat.settlement_price * self.dat.index_price

    def plot_densities_vs_strategy(self, m):

        # Sort by x to prevent overlapping lines
        #m = m.sort_values('x')

        # Build Plot
        fig = plt.figure(figsize=(19.20,9.83))
        #ax = plt.subplot(111)
        #plt.plot(1/m['m'], m['gy'], label = 'Physical Density ' + str(self.simmethod), linewidth = 3)
        plt.plot(1/m['m'], m['spdy']/m['gy'], label = 'State Price Density', linewidth = 3)
        plt.title('Pricing Kernel')
        plt.xlabel('Moneyness')

        plt.savefig('inversepk.png')

        ax2 = plt.subplot(122)
        plt.plot(m['m'], m['gy'], label = 'Physical Density ' + str(self.simmethod), linewidth = 3)
        plt.plot(m['m'], m['spdy'], label = 'State Price Density', linewidth = 3)
        plt.fill_between(m['m'], m['gy'], m['spdy'], where = m['long_call'] == 1, color = 'g', label = 'Long Call')
        plt.fill_between(m['m'], m['gy'], m['spdy'], where = m['short_call'] == 1, color = 'm', label = 'Short Call')
        plt.fill_between(m['m'], m['gy'], m['spdy'], where = m['long_put'] == 1, color = 'k', label = 'Long Put')
        plt.fill_between(m['m'], m['gy'], m['spdy'], where = m['short_put'] == 1, color = 'c', label = 'Short Put')
        plt.title('Trading Strategy')
        plt.xlabel('Moneyness')
        plt.suptitle('State-Price Density vs. Physical Density' + '\nMaturity in ' + str(int(round(self.tau * 365))) + ' Days, observed on ' + str(self.date))

        # Shrink current axis by 20%
        box = ax2.get_position()
        ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        #  Put a legend to the right of the current axis
        ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        fname = 'pricingkernel/plots/spd_vs_physical_density_' + 'tau-' + str(self.tau) + '_date-' + str(self.date) + str(self.simmethod) +'.png'
        print(fname)
        plt.savefig(fname, transparent = False)
        plt.draw()

    def plot_densities_vs_payoff(self, m, moneyness_lim = [0.5, 1.5]):

        # Payoff Function for Plot
        self.moneyness_per_state = []
        for state in self.settlement_states:
            self.payoff_per_state.append(self.spread_payoff(state))
            #self.moneyness_per_state.append(self.settlement_price/state)

        # For Payoff Plot
        #settlement_moneyness = self.settlement['Adj.Close'].iloc[-1] / self.settlement_states
        #m_min = max(min(settlement_moneyness), 0.5)
        #m_max = min(max(settlement_moneyness), 1.5)
        #self.settlement_moneyness = np.arange(m_min, m_max, len(self.payoff_per_state))

        # set order in order to prevent wiggly lines
        m = m.sort_values('x')

        # Build Plot
        fig = plt.figure(figsize=(19.20,9.83))
        ax = plt.subplot(121)
        plt.plot(m['m'], m['gy'], label = 'Physical Density', linewidth = 3)
        plt.plot(m['m'], m['spdy'], label = 'State Price Density', linewidth = 3)
        plt.title('Density Comparison')
        plt.xlabel('Moneyness')

        ax2 = plt.subplot(122)
        plt.plot(self.settlement_states, self.payoff_per_state, label = 'Payoff Function', color = 'green')
        plt.title('Payoff per Settlement Price')
        plt.xlabel('Bitcoin Price')
        #plt.xlim = moneyness_lim

        # Make sure we get no spelling error (days / day)
        days_to_go = int(round(self.tau * 365))
        if days_to_go == 1:
            upper = 'State-Price Density vs. Physical Density' + '\nMaturity in ' + str(days_to_go) + ' day, observed on ' + str(self.date)
        else:
            upper = 'State-Price Density vs. Physical Density' + '\nMaturity in ' + str(days_to_go) + ' days, observed on ' + str(self.date)
        plt.suptitle(upper)
        # Shrink current axis by 20%
        box = ax2.get_position()
        ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        #  Put a legend to the right of the current axis
        ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        fname = 'pricingkernel/plots/densities_vs_payoff_' + 'tau-' + str(self.tau) + '_date-' + str(self.date) + str(self.simmethod) + '.png'
        print(fname)
        plt.savefig(fname, transparent = False)
        plt.draw()

    def plot_bootstrap_pricingkernel(self, m, moneyness_lim = [0.5, 1.5]):

        # set order in order to prevent wiggly lines
        m = m.sort_values('x')
        m['pk'] = m['spdy'] / (m['gy'] * 10000)
        m['pk_cup'] = m['cup'] / (m['gy'] * 10000)
        m['pk_clo'] = m['clo'] / (m['gy'] * 10000)

        # Build Plot
        fig = plt.figure(figsize=(19.20,9.83))
        ax = plt.subplot(111)
        #plt.plot(m['m'], m['gy'], label = 'Physical Density', linewidth = 3)
        #plt.plot(m['m'], m['spdy'], label = 'State Price Density', linewidth = 3)
        plt.plot(m['m'], m['pk'], label = 'Pricing Kernel', linewidth = 3)
        plt.plot(m['m'], m['pk_cup'], label = 'Upper Confidence Band', linewidth = 3)
        plt.plot(m['m'], m['pk_clo'], label = 'Lower Confidence Band', linewidth = 3)
        #plt.title('Density Comparison')
        plt.xlabel('Moneyness')

        # Make sure we get no spelling error (days / day)
        days_to_go = int(round(self.tau * 365))
        if days_to_go == 1:
            upper = 'Bootstrap Pricing Kernel' + '\nMaturity in ' + str(days_to_go) + ' Day, observed on ' + str(self.date)
        else:
            upper = 'Bootstrap Pricing Kernel' + '\nMaturity in ' + str(days_to_go) + ' Days, observed on ' + str(self.date)
        plt.suptitle(upper)
        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        #  Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        fname = 'pricingkernel/plots/bootstrap_pk_' + 'tau-' + str(self.tau) + '_date-' + str(self.date) + str(self.simmethod) + '.png'
        print(fname)
        plt.savefig(fname, transparent = False)
        plt.draw()

        
    def build_strategy(self, m):
        og_m = m[['spdy', 'gy']]
        m = m.fillna(0)
        m = m.sort_values(by = 'x')
        center_idx = m.loc[m['gy'].idxmax(), 'x']
        center_comparison = m[m['gy'] == max(m['gy'])].x.values[0]
        min_quantile = 0 #m.spdy[m.spdy != 0].quantile(0.005)
        
        # Long Call
        m['long_call'] = 0
        m['long_call'][(m['spdy'] < m['gy']) & (m.x > center_idx) & (m['spdy'] > min_quantile)] = 1

        # Short Call
        m['short_call'] = 0
        m['short_call'][(m['spdy'] > m['gy']) & (m.x > center_comparison) & (m['spdy'] > min_quantile)] = 1
        
        # Long Put
        m['long_put'] = 0
        m['long_put'][(m['spdy'] < m['gy']) & (m.x < center_comparison) & (m['spdy'] > min_quantile)] = 1

        # Short Put
        m['short_put'] = 0
        m['short_put'][(m['spdy'] > m['gy']) & (m.x < center_idx) & (m['spdy'] > min_quantile)] = 1

        del m['spdy']
        del m['gy']

        return og_m.join(m)

    def find_instruments_in_range(self, df, strategy_range, posi):
        """
        For Calls choose minimum strike
        For Puts choose maximum strike
        So that the strikes converge
        """
        sub = df[(df.strike >= min(strategy_range)) & (df.strike <= max(strategy_range))]#.head(self.allowed_position_size)
        if 'call' in posi:
            return sub[(sub.strike == min(sub.strike))]
        else:
            return sub[(sub.strike == max(sub.strike))]

    def spread_payoff(self, settlement_price):
        """
        Default: Long
        For short need to rename build_strategy

        """

        # call spread and put spread
        # dont use self.settlement_price as default because we wanna have a complete payoff function for different settlement states
        k1 = self.short_put_instruments_price.index[0]
        k2 = self.long_put_instruments_price.index[0]
        k3 = self.long_call_instruments_price.index[0]
        k4 = self.short_call_instruments_price.index[0]

        #print(direction)
        if self.tradedirection == 'long':
            
            self.call_spread_payoff = max(settlement_price - k3, 0) - max(settlement_price - k4, 0)
            self.put_spread_payoff = max(k2 - settlement_price, 0) - max(k1 - settlement_price, 0)
            self.call_premium = self.short_call_instruments_price.values[0] - self.long_call_instruments_price.values[0] 
            self.put_premium = self.short_put_instruments_price.values[0] - self.long_put_instruments_price.values[0]

        elif self.tradedirection == 'short':
            # Reverse long and short instruments
            self.call_spread_payoff = - max(settlement_price - k3, 0) + max(settlement_price - k4, 0)
            self.put_spread_payoff = - max(k2 - settlement_price, 0) + max(k1 - settlement_price, 0)
            self.call_premium = - self.short_call_instruments_price.values[0] + self.long_call_instruments_price.values[0] 
            self.put_premium = - self.short_put_instruments_price.values[0] + self.long_put_instruments_price.values[0]
        else: 
            raise ValueError('Wrong Trading Direction!')

        overall_payoff = self.call_spread_payoff + self.put_spread_payoff + self.call_premium + self.put_premium

        return overall_payoff

    def run(self):
        # Based on the Density Comparison, create a trading rule

        # Also: Find regions to long/short the instruments
        # Long Call where SPD < HD and right side of center
        # Short Call where SPD > HD and right side of center
        # Long Put where SPD < HD and left side of center
        # Short Put where SPD > HD and left side of center
        g = pd.DataFrame({'x' : self.gdom, 'gy' : self.gstar})
        s = self.s.rename(columns = {'y' : 'spdy'})
        g['x'] = round(g['x'])
        s['x'] = round(s['x'])

        # Combine in order to compare which density is larger at which points
        m = s.merge(g, on = 'x', how = 'inner')
       # m = self.build_strategy(m)
        m = m.sort_values('x') # Prevent fuzzy graphs
        
        # Getting a bunch of NAs from the merge, gotta replace with zeros
        # So that its usable in global_plots.py
        # Fix the probbos where they arise

        #pdb.set_trace()
        save_dict(m['m'], m['x'], m['gy'], m['spdy'], 
                self.curr_day, self.tau, self.maturity, 'out/estimated_densities.csv')

        self.plot_bootstrap_pricingkernel(m)

        """
        # Trading
        calls = self.dat[(self.dat.is_call == 1) & (self.dat.tau == self.tau)]#  & (self.dat.days_to_maturity == self.time_to_maturity_in_days)
        puts = self.dat[(self.dat.is_call == 0)  & (self.dat.tau == self.tau)]

        # Construct payoff function
        for posi in self.positions:
            strategy_range = m['x'][m[posi] == 1]
            
            if len(strategy_range) == 0:
                raise ValueError('No Strike found')

            if posi == 'long_call':
                # Select instruments and Buy for the mean on a particular day.
                strategy_range = m['x'][m[posi] == 1]
                self.long_call_instruments       = self.find_instruments_in_range(calls, strategy_range, posi)
                self.long_call_instruments_price = self.long_call_instruments.groupby('strike')['instrument_price'].mean()

            elif posi == 'long_put':

                self.long_put_instruments       = self.find_instruments_in_range(puts, strategy_range, posi)
                self.long_put_instruments_price = self.long_put_instruments.groupby('strike')['instrument_price'].mean()

            elif posi == 'short_call':

                self.short_call_instruments         = self.find_instruments_in_range(calls, strategy_range, posi)
                self.short_call_instruments_price = self.short_call_instruments.groupby('strike')['instrument_price'].mean()

            elif posi == 'short_put':

                self.short_put_instruments       = self.find_instruments_in_range(puts, strategy_range, posi)
                self.short_put_instruments_price = self.short_put_instruments.groupby('strike')['instrument_price'].mean()

            else:
                raise ValueError('must not happen')

        if len(self.long_call_instruments_price) > 0 and len(self.short_call_instruments_price) > 0 and len(self.long_put_instruments_price) > 0 and len(self.short_put_instruments_price) > 0:
            
            # Pf log
            print('Constructing Portfolio on ', self.date)
            print('Buying Calls @ ', self.long_call_instruments_price)
            print('Selling Calls @ ', self.short_call_instruments_price)
            print('Buying Puts @ ', self.long_put_instruments_price)
            print('Selling Puts @ ', self.short_put_instruments_price)
            print('BTC Price at Contract Emission: ', self.underlying_price_at_emission)
            print('Settlement Price: ', self.settlement_price)

            # Actual Payoff
            self.overall_payoff = self.spread_payoff(self.settlement_price)

            print('call spread payoff: ', self.call_spread_payoff,
                    '\n put spread payoff: ', self.put_spread_payoff,
                    '\n call premium: ', self.call_premium,
                    '\n put premium: ', self.put_premium,
                    '\n overall payoff: ', self.overall_payoff)
        
            # Density Comparison and Strategy
            #self.plot_densities_vs_strategy(m)
            #self.plot_densities_vs_payoff(m)
        else:
            print('Some instrument not found, exiting without trade')

        """

        
