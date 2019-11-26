import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
"""
data = np.arange(10)
plt.plot(data)
"""
"""
fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax3.plot(np.random.randn(50).cumsum(),'g--')
_ = ax1.hist(np.random.randn(100),bins = 20,color= 'g',alpha =0.51)
ax2.scatter(np.arange(30),np.arange(30)+3*np.random.randn(30))
"""
"""
fig, axes = plt.subplots(2,2,sharex=True,sharey=True)#include same axis x,y
for i in range(2):
	for j in range(2):
		axes[i,j].hist(np.random.randn(500),bins =50,color ='k',alpha =0.5)
plt.subplots_adjust(wspace=0,hspace=0)
"""
"""
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(np.random.randn(1000).cumsum())
ticks = ax.set_xticks([0,250,500,750,1000])
labels = ax.set_xticklabels(['one','two','three','four','five'],rotation =0,fontsize ='small')
ax.set_title('My first matplotlib plot')
ax.set_xlabel('Stages')
"""
"""
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(np.random.randn(1000).cumsum(),'y',label = 'one')
ax.plot(np.random.randn(1000).cumsum(),'g--',label = 'two')
ax.plot(np.random.randn(1000).cumsum(),'r.',label = 'three')
ax.legend(loc = 'best')
"""
"""
fig = plt.figure(figsize=(12, 6)); ax = fig.add_subplot(1, 1, 1)
rect = plt.Rectangle((0.2, 0.75), 0.4, 0.15, color='k', alpha=0.3)
circ = plt.Circle((0.7, 0.2), 0.15, color='b', alpha=0.3)
pgon = plt.Polygon([[0.15, 0.15], [0.35, 0.4], [0.2, 0.6]],
                   color='g', alpha=0.5)
ax.add_patch(rect)
ax.add_patch(circ)
ax.add_patch(pgon)
"""
"""
from datetime import datetime
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
data = pd.read_csv('spx.csv', index_col=0, parse_dates=True)
spx = data['SPX']
spx.plot(ax=ax, style='k-')

crisis_data = [
    (datetime(2007, 10, 11), 'Peak of bull market'),
    (datetime(2008, 3, 12), 'Bear Stearns Fails'),
    (datetime(2008, 9, 15), 'Lehman Bankruptcy')
]
for date, label in crisis_data:
    ax.annotate(label, xy=(date, spx.asof(date) + 75),
                xytext=(date, spx.asof(date) + 225),
                arrowprops=dict(facecolor='black', headwidth=4, width=2,
                                headlength=4),
                horizontalalignment='left', verticalalignment='top')

ax.set_xlim(['1/1/2007', '1/1/2011'])
ax.set_ylim([600, 1800])
ax.set_title('Important dates in the 2008-2009 financial crisis')
plt.savefig('figpath.png',dpi=400,bbox_inches = 'tight')
plt.rc('figure',figsize=(0,1))
"""
"""
df = pd.DataFrame(np.random.randn(10,4).cumsum(0),columns = ['A','B','C','D'],index = np.arange(0,100,10))
df.plot()
"""
"""
fig,axes = plt.subplots(2,1)
data = pd.Series(np.random.rand(16),index = list('abcdefghijklmnop'))
data.plot.bar(ax = axes[0], color = 'y',alpha = 0.7)
data.plot.barh(ax = axes[1],color = 'k',alpha =0.7)
"""
"""
df = pd.DataFrame(abs(np.random.randn(6,4)),index = ['one','two','three','four','five','six'],columns = pd.Index(['A','B','C','D'],name = 'Genus'))
df.plot.barh(stacked = True ,alpha = 0.5)
"""
"""
tips = pd.read_csv('tips.csv')
party_counts = pd.crosstab(tips['day'],tips['size'])#使用crosstab组成表格
print(party_counts)
party_counts = party_counts.loc[:,2:5]
party_pcts = party_counts.div(party_counts.sum(1),axis=0)
print(party_pcts)
party_pcts.plot.bar()
"""
"""
tips = pd.read_csv('tips.csv')
tips['tip_pct'] = tips['tip']/(tips['total_bill']-tips['tip'])
print(tips)
#sns.barplot(x = 'tip_pct',y = 'day',hue = 'time' ,data = tips ,orient = 'h')
sns.set(style = 'whitegrid')
#tips['tip_pct'].plot.hist(bins = 50)
tips['tip_pct'].plot.density()
"""
"""
comp1 = np.random.normal(0,1,size = 200)
comp2 = np.random.normal(10,2,size = 200)
values = pd.Series(np.concatenate([comp1,comp2]))
sns.distplot(values,bins=100,color='b')
"""
macro = pd.read_csv('macrodata.csv')
data = macro[['cpi', 'm1', 'tbilrate', 'unemp']]
trans_data = np.log(data).diff().dropna()
print(trans_data[-5:])
sns.regplot('m1', 'unemp', data=trans_data)
plt.title('Changes in log %s versus log %s' % ('m1', 'unemp'))
sns.pairplot(trans_data,diag_kind = 'kde',plot_kws = {'alpha':0.3})






plt.show()