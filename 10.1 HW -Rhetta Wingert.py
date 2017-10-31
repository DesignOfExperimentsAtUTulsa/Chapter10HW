# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 09:38:53 2017

@author: redwi
"""
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy.stats import t
alpha = 0.05
x = [[1,10], [1,15], [1,15], [1,20], [1,20], [1,20], [1,25], [1,25], [1,28],[1,30]]
y = [[160], [171], [175], [182], [184], [181],[188], [193], [195], [200]]

     #Blocking matrix multiplication
xTx = np.matmul(np.transpose(x) , x)
#print('x transpose times x\n', xTx)
inv_xTx = np.linalg.inv(xTx)
print('Inverse x transpose times x\n', inv_xTx)
xTy = np.matmul(np.transpose(x), y)
#print('x transpose times y\n', (xTy))
bHat = np.matmul(inv_xTx, xTy)
n = len(y)
length = len(bHat)
k = length-1
p = length
#printing and preparing for any number of variables

print('The least Square fit is y = ', bHat[0], ' + ')
for i in range(1, k):
    print(bHat[i],'* x',i,' + ')
print(bHat[k],'* x',k)

print('ANOVA')
#Significance of Regression
#SSE
yTy = np.matmul(np.transpose(y), y)
bTxT = np.matmul(np.transpose(bHat), np.transpose(x))
bTxTy = np.matmul(bTxT, y)
SSE = yTy - bTxTy
print('SSE = ', SSE)
#SSR

ysqrd = ((sum(sum(var) for var in y))**2)/n
print(ysqrd)
SSR = bTxTy - ysqrd
#print(bTxTy, '-', ysqrd)
print('SSR = ', SSR)
#SST
SST = yTy-ysqrd
print('SST = ', SST)

MSR = SSR/k
MSE = SSE /(n-k-1)
print('MSR = ', MSR)
print( 'MSE = ', MSE)
f0 = MSR/MSE
Rsqrd = SSR/SST

print('f0 = ', f0)
print('R^2 = ', Rsqrd)

#Comfidence intervals
dof = n-p
sigma_sqrd = (SSE/dof)
#print('sigma hat =' , sigma_sqrd)
a = 1
CI_lower = bHat[a] - abs(t.ppf(alpha/2,dof))*math.sqrt(sigma_sqrd*inv_xTx[a][a])
CI_upper = bHat[a] + abs(t.ppf(alpha/2,dof))*math.sqrt(sigma_sqrd*inv_xTx[a][a])
#print('cjj = ', inv_xTx[a][a])
#print(math.sqrt(sigma_sqrd*inv_xTx[a][a]))
print('CI interval', CI_lower, CI_upper)
#Lengths of matrix
yBar = np.mean(y)
xShape = np.shape(x)

#predicted y values
yHat =[]
residuals = []
x1 = []
x2= []

#getting all the data out of the matrix
for row in range(xShape[0]):
    predicted_y = bHat[0]*x[row][0] + bHat[1]*x[row][1]
    yHat.append(predicted_y)
    resid = y[row] - predicted_y
    resid_float = float(resid)
    residuals.append(resid_float)
    x1.append(x[row][1])

plt.plot(x1,y, 'bo')
plt.plot(x1, bHat[0]+bHat[1]*x1)
plt.ylabel('Strength')
plt.xlabel('% Hardwood')
#plt.plot(x1, residuals, 'bo')
#plt.axhline(0, color='black')



