# Reconstruct trading from nohup.out
import csv
import re
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np

"""
RECONSTRUCT TRADING FROM LOG
"""

def extract_performance(line):
    return float(line[0].split(':')[1]) 

nohup_out_file = 'out/nohup_main.out'

# Collect info and subrtract delivery fee for options
# 0.015% - this fee can never be higher than 12.5% of the option's value.
delivery_fee = 0.00015

with open(nohup_out_file) as f:
    out_dict = {}
    out_all = {}
    c = csv.reader(f, delimiter='\n', skipinitialspace=True)
    line = ''
    target = list(c)
    for i in range(len(target)):
        try:
            line = target[i]
            #print(line)
            result = next((True for line in line if 'SVCJ.png' in line), False)
            if result is True:
                print('Found SVCJ in PNGname')

                # Reconstruct overall payoff from string
                # ['overall payoff:  99.11172356292109'] --> taking right part
                perf                = extract_performance(target[i-1]) # overall payoff
                put_premium         = extract_performance(target[i-2])
                call_premium        = extract_performance(target[i-3])
                put_spread_payoff   = extract_performance(target[i-4])
                call_spread_payoff  = extract_performance(target[i-5])
                #fees = delivery_fee * (abs(call_spread_payoff) + abs(put_spread_payoff))
                put_fee = abs(put_spread_payoff) * delivery_fee
                call_fee = abs(call_spread_payoff) * delivery_fee

                # @todo: how to get relative returns?
                trading_info = [put_premium, call_premium, put_spread_payoff - put_fee, call_spread_payoff - call_fee]
                print(trading_info)
            
                # Reconstruct date and tau of instrument
                rawkey = re.findall('_tau-(\d+.+\d)', line[0])[0]
                tau, date = rawkey.split('_date-')
                currkey = date + str('-') + tau # First date then tau so we can order according to date

                out_dict[currkey] = perf - put_fee - call_fee# real performance only
                out_all[currkey] =  trading_info# complete picture


        except Exception as e:
            print(e)

print(sorted(out_dict.keys()))

# PNL Distribution
payoff = [value for value in out_dict.values()]
dens = gaussian_kde(payoff)
xs = np.linspace(min(payoff), max(payoff), 100)
plt.plot(xs,dens(xs))
plt.xlabel('Absolute Profit')
plt.ylabel('Probability Density')
plt.savefig('pricingkernel/plots/pnl_distribution.png')
#plt.show()


plt.hist(out_dict.values(), bins = xs)
plt.savefig('pricingkernel/plots/pnl_histogram.png')
#plt.show()

# Risk free rate = 0
sharpe_ratio = np.mean(payoff)/(np.sqrt(np.var(payoff)))
print('sharpe ratio of the strategy: ', round(sharpe_ratio, 4))

with open('svcj_profit.csv', 'w') as f:  # You will need 'wb' mode in Python 2.x
    w = csv.DictWriter(f, out_dict.keys())
    w.writeheader()
    w.writerow(out_dict)