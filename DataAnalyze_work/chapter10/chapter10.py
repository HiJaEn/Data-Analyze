#分组聚合
#10.1--GroupBy机制
import pandas as pd
import numpy as np
df = pd.DataFrame({'key1':['a','a','b','b','a'],
					'key2':['one','two','one','two','one'],
					'data1':np.random.randn(5),
					'data2':np.random.randn(5)})
print(df)

grouped = df['data1'].groupby(df['key1'])#根据Key的标签计算data1列的均值
print(grouped)#groupeds是一个GroupBy对象grouped.mean()
print(grouped.mean())#发生了什么

means = df['data1'].groupby([df['key1'],df['key2']]).mean()
print(means) 

print(means.unstack())

states = np.array(['Ohio','California','California','Ohio','Ohio'])
years = np.array([2005,2005,2006,2005,2006])
print(df['data1'].groupby([states,years]).mean())
print(df['data1'].groupby([states,years]).mean().unstack())

print(df.groupby('key1').mean())
print(df.groupby(['key1','key2']).mean())

#10.1.1--遍历各分组
for name,group in df.groupby('key1'):#groupBy支持迭代，会生成一个包含组名和数据块的2维元组序列
	print(name)
	print(group)

for(k1,k2),group in df.groupby(['key1','key2']):#
	print((k1,k2))
	print(group)


pieces = dict(list(df.groupby('key1')))#选择在任何一块数据上进行想要的操作，使用一行代码计算出数据块的字典
print(pieces['b'])

print(df.dtypes)
grouped = df.groupby(df.dtypes,axis = 1)
for dtype,group in grouped:
	print(dtype)
	print(group)

#10.1.2--选则一列或者所有列的子集
data2means = df.groupby(['key1','key2'])[['data2']].mean()
print(data2means)
s_grouped = df.groupby(['key1','key2'])['data2'].mean()
print(s_grouped)
#10.1.3--使用字典和Series分组
people = pd.DataFrame(np.random.randn(5,5),columns = ['a','b','c','d','e'],index = ['Joe','Steve','Wes','Jim','Travis'])
people.iloc[2:3,[1,2]] = np.nan
print(people)

#10.1.3--使用字典和Series分组
#根据这个字典构造传给groupby的数组
mapping = {'a':'red','b':'red','c':'blue','d':'blue','e':'red','f':'orange'}
by_column = people.groupby(mapping,axis=1)
print(by_column.sum())
#Series也有相同的功能，可以视为固定大小的映射
map_series = pd.Series(mapping)
print(map_series)
print(people.groupby(map_series,axis = 1).count())

#10.1.4--使用函数分组
print(people)
print(people.groupby(len).sum())
key_list = ['one','one','one','two','two']
print(people.groupby([len,key_list]).min())

#10.1.5--根据索引层级分组
columns = pd.MultiIndex.from_arrays([['US','US','US','JP','JP'],[1,3,5,1,3]],names=['city','tenor'])
print(columns)
hier_df = pd.DataFrame(np.random.randn(4,5),columns = columns)
print(hier_df)
print(hier_df.groupby(level = 'city',axis = 1).count())

#10.2--数据聚合
print(df)
grouped = df.groupby('key1')
quan = grouped['data1'].quantile(0.9)
print(quan)

def peak_to_peak(arr):
	return arr.max()-arr.min()
print(grouped.agg(peak_to_peak))

#应用--通用拆分-应用-联合
tips = pd.read_csv('tips.csv')
print(tips)
tips['tip_pct'] = tips['tip']/(tips['total_bill']-tips['tip'])
def top(df,n=5,column = 'tip_pct'):
	return df.sort_values(by=column)[-n:]
print(top(tips,n=56))
top5 = tips.groupby('smoker').apply(top)
print(top5)

tops = tips.groupby(['smoker','day']).apply(top,n=1,column = 'total_bill')
print(tops)

