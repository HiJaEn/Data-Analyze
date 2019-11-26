#5.1	pandas数据结构介绍 --Series
import pandas as pd
import numpy as np
from pandas import Series,DataFrame
obj = pd.Series([4,7,-5,3])
print(obj)

obj2 = pd.Series([4,7,-5,3],index = ['d','b','a','c'])
print(obj2,'\n',obj2.dtype)
print(obj2.index)

print(obj2['a'])
obj2['d'] = 6
print(obj2[['d','b','a','c']])
print(obj2[obj2>0])
print(obj2*2)
print(np.exp(obj2))

print('b' in obj2)
print('s' in obj2)

sdata = {'Ohio':3500,'Texas':7100,'Oregon':16000,'Utah':5000}
obj3 = pd.Series(sdata)
print(obj3)

states = ['California','Ohio','Oregon','Texas']
obj4 = pd.Series(sdata,index = states)
print(obj4)

print(pd.isnull(obj4))
print(pd.notnull(obj4))
print(obj3+obj4)

obj = pd.Series([4,7,-5,3])
obj.index = ['Bob','Steve','Jeff','Ryan']
print(obj)

#DataFrame
data = {'State':['Ohio','Ohio','Ohio','Nevada','Nevada','Nevada'],
		'Year':[2000,2001,2002,2001,2002,2003],
		'Pop':[1.5,1.7,3.6,2.4,2.9,3.2]
		}
frame = pd.DataFrame(data)
print(frame)
frame = pd.DataFrame(data,columns = ['Year','Pop','State'])
print(frame)
print(frame.head())
frame2 = pd.DataFrame(data,columns = ['Year','Pop','State','Debt'],index = ['one','two','three','four','five','six'])
print(frame2)
print(frame2.index)
print(frame2['State'])
print(frame2.loc['three'])

#frame2['Debt'] = 16.5
#print(frame2)
#frame2['Debt'] = np.arange(6.0)
#print(frame2)

val = pd.Series([-1.2,-1.5,-1.7],index = ['two','four','five'])
frame2['Debt'] = val
print(frame2)

frame2['eastern'] = frame2.State == 'Ohio'
print(frame2)

del frame2['eastern']
print(frame2)
print(frame2.columns)

pop = {'Nevada':{2001:2.4,2002:2.9},
		'Ohio':{2000:1.5,2001:1.7,2002:3.6}}
frame3 = pd.DataFrame(pop,columns = ['Ohio','Nevada'],index = [2000,2001,2002])
print(frame3)
#print(frame3.T)

frame4 =pd.DataFrame(pop,index = [2001,2002,2003])
#print(frame4)

pdata = {'Ohio':frame3['Ohio'][:-1],
		'Nevada':frame3['Nevada'][:2]}
print(pd.DataFrame(pdata))

frame3.index.name = 'year';
frame3.columns.name = 'state'
print(frame3)

print(frame3.values)
print(frame2.values)

obj = pd.Series(range(3),index = ['a','b','c'])
index = obj.index
print(obj,index)
print(index[:2])
#index[1] = 'd'

#5.2.1	重建索引
obj = pd.Series([4.5,7.2,-5.3,3.6],index = ['d','b','a','c'])
print(obj)
obj2 = obj.reindex(['a','b','c','d','e'])
print(obj2)

obj3 = pd.Series(['blue','purple','yellow'],index = [0,2,4])
print(obj3)
obj3.reindex(range(6), method = 'ffill')


frame = pd.DataFrame(np.arange(9).reshape((3,3)),index = ['a','c','d'],columns = ['Ohino','Texas','California'])
print(frame)


#frame2 =frame.reindex = (['a','b','c','d'])#Error
#print(frame2)

frame2 = frame.reindex (['a','b','c','d'])
print(frame2)

#使用columns关键字引索
states = ['Texas','Utah','California']
frame2 = frame.reindex(columns = states)
print(frame2)

frame3 = frame.loc[['a','b','c','d'],states]
print(frame3)

#5.2.2	轴向上删除条目
obj = pd.Series(np.arange(5.),index = ['a','b','c','d','e'])
new_obj = obj.drop('c')
print(new_obj)
new_obj = obj.drop(['d','c'])
print(new_obj)

data = pd.DataFrame(np.arange(16).reshape(4,4),index = ['Ohid','Colorado','Utah','New_York'],columns = ['ones','two','three','four'])
print(data)
data1 = data.drop(['Colorado','Ohid'])
print(data1)
data2 = data.drop('two',axis = 1)
print(data2)
data3 = data.drop(['two','four'],axis = 'columns')
print(data3)

print(obj)
obj.drop('c',inplace =True)
print(obj)


#5.2.3	引索，选择，过滤
obj = pd.Series(np.arange(4),index = ['a','b','c','d'])
print(obj)
obj['b':'c'] = 5
print(obj)

data = pd.DataFrame(np.arange(16).reshape(4,4),index = ['Ohio','Colorado','Utah','New_York'],columns = ['one','two','three','four'])
print(data)
print(data['two'])
print(data[['one','two']])
print(data[:2])

print(data[data['three']>5])
print(data<5)

data[data<5] = 0
print(data)

print(data.loc['Colorado',['one','two','three']])
print(data.iloc[2,[3,0,1]])
print(data.iloc[2])
print(data.iloc[[1,2],[3,0,1]])

print(data.loc[:'Utah','two'])
print(data.iloc[:,:3][data.three > 5])

#5.2.4	整数索引
ser = pd.Series(np.arange(3.))
print(ser)
ser2 = pd.Series(np.arange(3.),index = ['a','b','c'])
print(ser2)
print(ser2[-2])
print(ser[:1])
print(ser.loc[:1])
print(ser.iloc[:1])

#	整数和数据对齐
df1 = pd.DataFrame(np.arange(12).reshape((3,4)),columns = list('abcd'))
df2 = pd.DataFrame(np.arange(20,).reshape((4,5)),columns = list('abcde'))
print(df1)
print(df2)
print(df1.add(df2,fill_value =0))
print(df2.rdiv(1))

arr = np.arange(12).reshape((3,4))
print(arr)
print(arr - arr[0])#广播

frame = pd.DataFrame(np.arange(12).reshape((4,3)),columns = list('bcd'),index = ['Utah','Ohio','Texas','Oregon'])
series = frame.iloc[0]
print(frame)
print(series)
print(frame - series)
series3 = frame['c']
print(series3)
print(frame.sub(series3,axis = 0))

frame = pd.DataFrame(np.random.randn(4,3),columns = list('bde'),index = ['Utah','Ohio','Texas','Oregon'])
print(frame)
print(np.abs(frame))

f = lambda x: x.max()-x.min()
print(frame.apply(f))
print(frame.apply(f,axis = 'columns'))

def f(x):
	return pd.Series([x.min(),x.max()],index = ['min','max'])
print(frame.apply(f))

format = lambda x: '%.2f' % x
#print(frame.applymap(format))

#排序和排名
frame = pd.DataFrame(np.arange(8).reshape((2,4)),index = ['three','one'],columns = ['d','a','b','c'])
#print(frame.sort_index())
#print(frame.sort_index(axis = 1))
print(frame.sort_index(axis =1,ascending = False))

obj = pd.Series([4,7,-3,2])
#print(obj.sort_values())
obj = pd.Series([4,np.nan,7,np.nan,-3,2])
#print(obj.sort_values())

frame = pd.DataFrame({'b':[4,7,-3,2],'a':[0,1,0,1]})
print(frame)
print(frame.sort_values(by = 'b'))
print(frame.sort_values(by = ['a','b']))

obj = pd.Series([7,-5,7,4,2,0,4])
obj.rank()
print(obj.rank(method = 'first'))
print(obj.rank(ascending = False ,method ='max'))

frame = pd.DataFrame({'b':[4.3,7,-3,2],'a':[0,1,0,1],'c':[-2,5,8,-2.5]})
frame.rank(axis = 'columns')

#含有重复标签的轴索引
df = pd.DataFrame(np.random.randn(4,3),index = ['a','a','b','b'])
print(df)
print(df.loc['b'])
print(df.index.is_unique)

#描述性统计的概率与计算
df = pd.DataFrame([[1.4,np.nan],[7.1,-4.5],[np.nan,np.nan],[0.75,-1.3]],index = ['a','b','c','d'],columns = ['one','two'])
print(df)
print(df.sum)
print(df.sum(axis = 'columns'))
print(df.describe())

'''
#相关性和协方差
import pandas_datareader.data as web
import datetime
start = datetime.datetime(2017,11,24)
end =datetime.date.today
all_data = {ticker:web.get_data_yahoo(ticker)
			for ticker in ['AAPL','IBM','MSFT','GOOG']
			}
price = pd.DataFrame({ticker:data['Adj Close']
			for ticker,data in all_data.items()})
volume = pd.DataFrame({ticker:data['Volume']
			for ticker,data in all_data.items()})
returns = price.pct_change()
print(returns.tail())
print(returns.corr())
print(returns.cov())
print(returns.corrwith(returns.IBM))
print(returns.corrwith(volume))
'''

#	唯一值 计数  成员属性
obj = pd.Series(['c','a','d','a','a','b','b','c','c'])
uniques = obj.unique()
print(uniques)
print(obj.value_counts())
