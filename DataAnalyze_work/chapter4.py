import numpy as np
import time
import matplotlib.pyplot as plt
import pylab

data = {i : np.random.randn() for i in range(7)}
print(data)



my_arr = np.arange(10000)
my_list = list(range(10000))
starttime = time.perf_counter()
my_arr = my_arr*2
#print(my_arr)
endtime = time.perf_counter()
print(str((endtime-starttime)*1000)+"ms")
starttime = time.perf_counter()
my_list = [x*2 for x in my_list] 
#print(my_list)
endtime = time.perf_counter()
print(str((endtime-starttime)*1000)+"ms")



data = np.random.randn(2,3)
data_1 = data*10
data_2 = data+data
print(data,"\n",data_1,"\n",data_2)
print(data.shape,data.dtype)


data = [6,7.5,88,0,1]
data_1 = np.array(data)
print(data_1)
print(data_1.dtype)

data = [[1,2,3,4],[5,6,7,8]]
data_1 = np.array(data)
print(data_1)
print(data_1.dtype)

data = np.arange(10)
print(data,data.dtype)

arr1 = np.array([1,2,3],dtype = np.float64)
arr2 = np.array([1,2,3],dtype = np.int32)
print(arr1,arr1.dtype)
print(arr2,arr2.dtype)

arr = np.array([1,2,3,4])
float_arr = arr.astype(np.float64)
print(float_arr,float_arr.dtype)

arr = np.array([3.5,-1.2]) 
arr = arr.astype(np.int64)
print(arr,arr.dtype)

numeric_strings = np.array(['1.25','-9.6','42'],dtype=np.string_)
numeric = numeric_strings.astype(np.float64)
print(numeric,numeric.dtype)

int_array = np.arange(10)
calibers = np.array([.22,.3,.3,3.5],np.float64)
float_array = int_array.astype(calibers.dtype)
print(calibers)
print(float_array,float_array.dtype)

arr = np.array([[1.2,2.2,3.3],[4.,3,4.5]])
arr_1 =arr*arr
print(arr_1)

arr = np.arange(10)
arr_copy = arr[5:8].copy() 
arr_copy[0] = 1
print(arr,arr_copy)

arr =np.arange(10)
arr_slice = arr[5:8]
arr_slice[0] = 1
print(arr,arr_slice)

arr = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(arr[2],arr[2][0])

arr2d = np.array([[1,2,3],[4,5,6],[7,8,9]])
arr_0_1 = arr2d[:,0:1]
print(arr_0_1)
arr_2_3 = arr2d[:2,1:] #左闭右开
print(arr_2_3)

names = np.array(['A','B','A','D','F'])
data = np.random.randn(5,4)
print(data,'\n',data.shape)
print(names == 'A')
BobsData_1 = data[names == 'A']
BobsData_2 = data[names=='A',2:3]
print(BobsData_1,'\n',BobsData_2)

mask = (names == 'A') | (names == 'B')
print(mask)
print(data[mask])

i = list(range(8))
print(i)
arr = np.empty((8,4))
for i in range(8):
	arr[i] = i
print(arr)

arr = np.empty((2,3,2))
print(arr)
arr  = np.arange(32).reshape((8,4))
print(arr[[0,1,2,3,4,5,6,7],[3,3,3,3,3,3,3,3]])

arr = np.arange(4).reshape((2,2))
print(arr)
arr_arrt = np.dot(arr,arr.T)
print(arr_arrt)


#4.2	通用函数：快速的逐元素数组函数
arr = np.arange(10)
arr_1 = np.sqrt(arr)
print(arr,'\n',arr_1)

x = np.random.randn(8)
y = np.random.randn(8)
z = np.maximum(x,y)
print(x,'\n',y,'\n','\n',z)

arr = np.random.randn(7)*5
remainder,whole_part = np.modf(arr)
print(arr,'\n',remainder,'\n',whole_part)
arr = np.abs(arr)
print(arr)
arr_1 = np.sqrt(arr)
print(arr_1)

#4.3	使用数组进行面向数组编程
points = np.arange(-5,5,0.01)
print(points)
xs,ys = np.meshgrid(points,points)
print(xs)
print(ys)

z = np.sqrt(xs ** 2+ys ** 2)
print(z)
#plt.
#imshow(z,cmap = plt.cm.gray);
#pylab.show()
#print(plt.colorbar())
#plt.title("Image plot of $\sqrt{x^2,y^2}$ for a grid of values")

arr = np.random.randn(4,4)
judge = arr>0
print(judge)
arr_1 = np.where(arr>0,2,-2)
print(arr_1)
arr_2 = np.where(arr>0,2,arr)
print(arr_2)

print(arr_1)
arr_mean = arr_1.mean()
arr_sum =arr_1.sum()
print(arr_mean)
print(arr_sum)

arr = np.random.randn(5,4)
arr_1 = arr.mean(axis = 1)#perform
print(arr_1)
arr_2 = arr.sum(axis = 0)#low
print(arr_2)

arr = np.array([0,1,2,3,4,5,6,7])
cumsum = arr.cumsum()
print(arr)
print(cumsum)

arr = np.array([[0,1,2],[3,4,5],[6,7,8]])
cumsum_arr = arr.cumsum(axis = 0)
cumprod_arr = arr.cumprod(axis =1 )
print(cumsum_arr,'\n',cumprod_arr)

arr = np.random.randn(100)
judge = arr>0
number_judge = judge.sum()
print(number_judge)

arr = np.ones((5,5))
print(arr)
judge = arr>0
print(judge)
check_any = judge.all()
check_all = judge.any()
print(check_any,check_all)

arr = np.random.randn(5,3)
arr_sort = arr.sort(1)
print(arr)
print(arr_sort)#out = None

large_arr = np.random.randn(1000)
print(large_arr)
large_arr.sort()
print(large_arr)
print(large_arr[int(0.05*len(large_arr))])

names = np.array(['Bob','Joe','Will','Bob','Will','Joe','Joe'])
onces = np.unique(names)
print(onces,'\n',onces.dtype)

values = np.array([3,3,3,2,3,2,1,4])
isexits = np.in1d(values,[1,4])
print(isexits,'\n',isexits.dtype)

x = np.array([[1,2,3],[4,5,6]])
y = np.array([[6,23],[-1,6],[3,4]])
print(x.dot(y))#equal to np.dot(x,y) 
print(np.dot(x,y))

arr_1 = np.ones(3)
print(arr_1)
print(np.dot(x,np.ones(3)))


import random
position = 0
walk = [position]
steps = 1000
for i in range(steps):
	step = 1 if random.randint(0,1) else -1
	position +=step
	walk.append(position)
#plt.plot(walk[:100])
#pylab.show()

nsteps = 1000
draws = np.random.randint(0,2,size = nsteps)
print(draws.shape)
steps = np.where(draws>0,1,-1)
walk = steps.cumsum()
#plt.plot(walk[:1000])
#pylab.show()
print(walk)
print(walk.min())
print(walk.max())

nwalks = 4
nsteps = 10000
draws = np.random.randint(0,2,size=(nwalks,nsteps))
steps = np.where(draws>0,1,-1)
walks = steps.cumsum(1)
print(walks)
walks = walks.T 
plt.plot(walks)
pylab.show()
print(walks,walks.shape)
print(walks.max())