# Plot Deribit Underlying

# Calculate IV vs Real Vola and visualize
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import pdb

dat = pd.read_csv('DAta/DeribitUnderlying.csv')
dat['Date'] = pd.to_datetime(dat['Date'])#.sort_values('Date')

"""
# Display only Monthly Dates
locator = mdates.MonthLocator()  # every month
# Specify the format - %b gives us Jan, Feb...
fmt = mdates.DateFormatter('%Y-%b')
"""
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(dat['Date'], dat['Price'], color = 'black')
#plt.plot(sub['date'], sub['rolling'], label = '21 Day \nVolatility', color = 'orange')
#plt.title('Implied vs. Realized Volatility')
#plt.ylabel('IV')
#plt.xlabel('Time')
# Shrink current axis by 20%
#box = ax.get_position()
#ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
#  Put a legend to the right of the current axis
#ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

"""
# Only Months
X = plt.gca().xaxis
X.set_major_locator(locator)
# Specify formatter
X.set_major_formatter(fmt)
"""
plt.savefig('/Users/julian/src/up/spd/DeribitUnderlying.png', transparent = True)
#plt.show()

# Check Leverage Effect
vola = pd.read_csv("dailY_vola.csv")

vola['Date'] = pd.to_datetime(vola['date'])
mm = vola.merge(dat, on = 'Date', how = 'inner')
print(mm.corr(method = 'pearson'))
pdb.set_trace()
# Volatility Premium
mm.mean()