# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 16:28:48 2024

张冉
"""
'''张冉 车辆2102 202104060915
'''
import math
from matplotlib import pyplot
import time
N=10   
pi=math.pi  
x=[]
y=[]
time1=time.time()
for i in range(N):
    x.append(round((-1+(2/N)*(N-i)),3))

def function(x1):
    return math.sin(pi*x1)

for i in range(len(x)):
    y.append(function(x[i]))


def lagrange_interploate(x1):
    P=[]
    L_n=0
    for i in range(len(x)):
        numerator=1
        denominator=1
        for j in range(len(x)):
            if j!=i:
                numerator*=(x1-x[j])
                denominator*=(x[i]-x[j])
        P.append(numerator/denominator)

    for i in range(len(x)):
        L_n+=y[i]*P[i]

    return round(L_n,3)
x1=[]
y1=[]  
y2=[]
for i in range(1000):
    x1.append(round((-1 + (2 / 1000) * (1000 - i)), 3))   
for i in range(1000):
    y1.append(lagrange_interploate(x1[i]))   
for i in range(1000):
    y2.append(function(x1[i]))  
time2=time.time()

pyplot.plot(x1,y1)
pyplot.show()

error=[]
for i in range(1000):
    error.append(abs(y1[i]-y2[i]))

print("耗时：",time2-time1)
print(max(error))

