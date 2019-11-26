import pandas as pd
import numpy as np
df = pd.read_csv('examples/ex1.csv')
print(df)
df = pd.read_table('examples/ex1.csv',sep = ',')
print(df)
df = pd.read_csv('examples/ex2.csv',header = None)
print(df)

names = ['a','b','c','d','message']
df = pd.read_csv('examples/ex2.csv',names = names,index_col = 'message')
print(df)

result = pd.read_table('examples/ex3.txt',sep = '\s+')
print(result) #由于列名的数量比数据的列数少一个，因此read_table推断第一列作为DataFrame的索引

df = pd.read_csv('examples/ex4.csv',skiprows = [0,2,3])
print(df)

result =pd.read_csv('examples/ex5.csv')
print(result)

print(pd.isnull(result))

result = pd.read_csv('examples/ex5.csv',na_values = ['NULL'])
print(result)

sentinels = {'message':['foo','NA'],'something':['two']}
result = pd.read_csv('examples/ex5.csv',na_values = sentinels)
print(result)

pd.options.display.max_rows = 10
result = pd.read_csv('examples/ex6.csv')
print(result)
result = pd.read_csv('examples/ex6.csv',nrows = 5)
print(result)

#6.1.1	分块读入文本文件	
chunker = pd.read_csv('examples/ex6.csv',chunksize = 1000)
tot = pd.Series([])
for piece in chunker:
	tot = tot.add(piece['key'].value_counts(),fill_value=0)
tot = tot.sort_values(ascending = False)

#6.1.2	将数据写入文本格式
data = pd.read_csv('examples/ex5.csv')
data.to_csv('examples/ex5output.csv')

import sys
data.to_csv(sys.stdout,sep = '|')
data.to_csv(sys.stdout,na_rep = 'NULL')

data.to_csv(sys.stdout,index=False,header=False,na_rep = 'null')
data.to_csv(sys.stdout,index = False,columns = ['a','b','c'],na_rep = 'null')

datas =pd.date_range('2/12/2019',periods=7)
ts = pd.Series(np.arange(7),index = datas)
ts.to_csv('examples/tsoutput.csv')