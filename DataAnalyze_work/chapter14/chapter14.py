import json
path = 'example.txt'
records = [json.loads(line) for line in open(path)]

time_zones = [rec['tz'] for rec in records if 'tz' in rec]#获取到time_zones
print(len(time_zones))

def  get_counts(sequence):#获取空的
	counts = {}
	for x in sequence:
		if x in counts:
			counts[x] +=1
		else:
			counts[x] =1
	return counts

counts = get_counts(time_zones)#输入time_zones
print(counts['America/New_York'])

from collections import Counter
counts = Counter(time_zones)
print(counts.most_common(10))

import pandas as pd
frame = pd.DataFrame(records)
print(frame)
frame.info()
print(frame['tz'][:10])

tz_counts = frame['tz'].value_counts()
print(tz_counts[:10])
clean_tz = frame['tz'].fillna('missing')#缺失部分填充为missing
clean_tz[clean_tz == ''] = 'Unknow'#将布尔值为True的值赋值为Unknow
tz_counts = clean_tz.value_counts()
print(tz_counts[:10])

#利用绘图库为数据生成一张图片
import seaborn as sns
import matplotlib.pyplot as plt
subset = tz_counts[:10]
#subset.plot(kind = 'bar',rot = 0)
#sns.barplot(y = subset.index ,x =subset.values)
plt.show()

#分离字符串信息
print(frame['a'][51][:50])
results = pd.Series([x.split()[0] for x in frame.a.dropna()])#将表Frame中的‘a’这一列缺失的数据(NAN)进行过滤，然后取剩下的部分，split函数没有添加参数进去就会以‘ ’为间隔进行分割
import numpy as np
cframe = frame[frame.a.notnull()]#移除缺失数据，过滤的是空('')元素
cframe['os'] = np.where(cframe['a'].str.contains('Windows'),'Windows','Not Windows') 
print(cframe['os'])
by_tz_os = cframe.groupby(['tz','os'])
agg_counts = by_tz_os.size().unstack().fillna(0)
print(agg_counts)
#选择最常见的时区
indexer = agg_counts.sum(1).argsort()#默认表示按列相加，1表示按行相加
print(indexer)
count_subset = agg_counts.take(indexer[-10:])
print(count_subset)
#生成条形堆积图
#count_subset.plot(kind = 'barh',stacked = False)
#归一化
count_subset = count_subset.stack()
print(count_subset)
count_subset.name = 'total'
count_subset = count_subset.reset_index()#加上了计数的一行
print(count_subset)
count_subset[:10]

def normal_total(group):
	group['normal_total'] = group.total / group.total.sum()
	return group

results = count_subset.groupby('tz').apply(normal_total)#新添加一行
print(results)
#sns.barplot(x='normal_total',y='tz',hue = 'os',data = results)




import pandas as pd
names1880 = pd.read_csv('babynames/yob1880.txt',names = ['name','sex','birth'])
print(names1880)
print(names1880.groupby('sex').birth.sum())#使用按性别列出的出生总和作为当年的出生总数

years = range(1880,2011)
pieces = []
columns = ['name','sex','births']
for year in years:
	path = 'babynames/yob%d.txt'%year
	frame = pd.read_csv(path,names = columns)#这儿的name的意思是什么
	print(frame)

	frame['year'] = year
	pieces.append(frame)
names = pd.concat(pieces,ignore_index = True)#不想保留从read_csv返回的原始行号
print(names)

total_birth = names.pivot_table('births',index='year',columns='sex',aggfunc=sum)
print(total_birth)
#total_birth.plot(title = 'Total births by sex and year')

def add_prop(group):
	group['prop'] = group.births/group.births.sum()
	return group
names = names.groupby(['year','sex']).apply(add_prop)#增加对groupby的理解
print(names)
print(names.groupby(['year','sex']).prop.sum())

#每个性别/年份组合的前100名
def get_top1000(group):
	return group.sort_values(by = 'births',ascending = False)[:1000]

grouped = names.groupby(['year','sex'])
top1000 = grouped.apply(get_top1000)
top1000.reset_index(inplace = True,drop = True)
print(top1000)

#分析名字趋势
boys = top1000[top1000.sex == 'M']
girls = top1000[top1000.sex == 'F']
total_births = top1000.pivot_table('births',index = 'year',columns = 'name',aggfunc = sum)
print(total_births.info())
subset = total_births[['John','Harry','Mary','Marilyn']]
#subset.plot(subplots = True,figsize =(12,10),grid =False,title = "Number of births per year")

#计量命名多样性的增加
df = boys[boys.year == 2010]
prop_cumsum = df.sort_values(by = 'prop',ascending = False).prop.cumsum()
print(prop_cumsum)
print(prop_cumsum.values.searchsorted(0.5)+1)

df = boys[boys.year == 1900]
prop_cumsum = df.sort_values(by = 'prop',ascending = False).prop.cumsum()
print(prop_cumsum)
print(prop_cumsum.values.searchsorted(0.5)+1)


def get_quantile_count(group,q=0.5):
	group = group.sort_values(by = 'prop',ascending = False)
	return group.prop.cumsum().searchsorted(q)+1

diversity = top1000.groupby(['year','sex']).apply(get_quantile_count)
diversity = diversity.unstack('sex')
print(diversity)
#diversity.plot()

#最后一个字母
get_last_letter = lambda x:x[-1]
last_letters = names.name.map(get_last_letter)
last_letters.name = 'last_letters'
table = names.pivot_table('births',index = last_letters,columns = ['sex','year'],aggfunc = sum)
subtable = table.reindex(columns = [1910,1960,2010],level = 'year')
print(subtable)
subtable.sum()
letter_prop = subtable/subtable.sum()
print(letter_prop)

import matplotlib.pyplot as plt
#fig,axes = plt.subplots(2,1,figsize=(10,8))
#letter_prop['M'].plot(kind = 'bar',rot =0,ax = axes[0],title ='Male')
#letter_prop['F'].plot(kind = 'bar',rot =0,ax = axes[1],title ='Female',legend =False)#legend = False关闭图例


letter_prop = table/table.sum()
dny_ns = letter_prop.loc[['d','n','y'],'M'].T
dny_ns.head()
#dny_ns.plot()

#男孩名变成女孩名
all_names  = pd.Series(top1000.name.unique())
print(all_names)
lesley_like = all_names[all_names.str.lower().str.contains('lesl')]
print(lesley_like)
filtered = top1000[top1000.name.isin(lesley_like)]
print(filtered)
filtered.groupby('name').births.sum()
table = filtered.pivot_table('births',index='year',columns = 'sex',aggfunc = 'sum')
table = table.div(table.sum(1),axis=0)
print(table)
table.plot(style = {'M':'y-','F':'g--'})


plt.show()