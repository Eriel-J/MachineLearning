#-*- coding:utf-8 -* 
import numpy as np
#用列表或元组创建
a = np.array([[1,2,3],[4,5,6]])
b = np.array([1,2],dtype=complex)

#类似内置函数range
c = np.arange(24).reshape(2,3,4)

#等差,等比数组
d = np.linspace(0,1,10,endpoint=False)
print (np.logspace(0,4,3,base=2))

#创建特殊数组
print (np.zeros((2,3)))
print (np.zeros_like(a))
print (np.ones((2,3),dtype=np.int16)) #全1
print (np.empty((2,3)))
print (np.eye(3)) #单位阵

#从字符串,函数,文件等创建
s ='abcdef'
print (np.fromstring(s,dtype=np.int8))
print (np.fromfunction(lambda i,j:(i+1)*(j+1), (9,9)))
#fromfile从二进制文件中创建,tofile写入文件


if __name__ == '__main__':
    c = np.arange(24).reshape(2, 3, 4)
    print(c)
    print(np.any(c[:,] == 23))