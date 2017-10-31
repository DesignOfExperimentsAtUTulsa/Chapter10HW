# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 14:35:37 2017

@author: Keagan Clement
@title: Chapter 10 Homework
"""

from PyQt5.QtWidgets import QFileDialog,QWidget
import numpy as np
from scipy import linalg as la
from scipy import stats as st
from matplotlib import pyplot as plt

def performRegression(dataFileName = None):
    # If no data file provided, use a file dialog
    # Response should be in first column; factors in subsequent columns
    if(dataFileName == None):
        dataFileName = QFileDialog.getOpenFileName(QWidget())[0]
    headerRow = 1 
    #load data file into memory as a list of lines  
    with open(dataFileName,'r') as dataFile:  
        dataLines = dataFile.readlines()
#    print("Opened {}".format(dataFileName))
#    print(dataLines[1:10])
    colHeaders = dataLines[headerRow].strip().split(',')
    
    # Find n, k, and p (assume simple linear fit with no quadratics or interactions)
    n = len(dataLines) - headerRow - 1
    k = len(dataLines[headerRow+1].strip().split(',')) - 1
    p = k+1
    
    # Fill y vector with data from file 
    y = np.array([])
    for row in range(headerRow+1, len(dataLines)):
        y = np.append(y, float((dataLines[row].strip().split(','))[0]))
    
    # Fill x array with data from file
    ones = [1] * n
    xList = [ones]
    for i in range(k):
        col = []
        for j in range(n):
            col.append(float((dataLines[j+headerRow+1].strip().split(','))[i+1]))
        xList.append(col)
    x = np.array(xList)
    
    # Calculate parameters
    beta = np.array(la.lstsq(x.T,y)[0])
    
    # Calculate the sum squares and the f-statistic
    SSE = y.dot(y.T) - beta.T.dot(x).dot(y)
    SSR = beta.T.dot(x).dot(y) - (1/n) * np.sum(y) ** 2
    F0 = (SSR / float(k)) / (SSE / float(n-k-1))
    Fa = st.f.ppf(0.95,k,n-k-1)
    if (F0 > Fa):
        print('Regression is significant')
    else:
        print('Regression is not significant')
        
    # Calculate standard error
    var = SSE / (n-p)
    cMat = la.inv(x.dot(x.T))
    c = []
    se = []
    for i in range(p):
        c.append(cMat.item((i,i)))
        se.append(np.sqrt(var * c[i]))
        
    # Calculate confidence intervals for parameters
    tTable = st.t.ppf(0.975,n-p)
    lowB = []
    highB = []
    for i in range(p):
        lowB.append(beta[i] - tTable * se[i])
        highB.append(beta[i] + tTable * se[i])
        
    # Print results
    
    
    # Plot regression and residuals if there is only one factor
    if(k == 1):
        # Regression plot
#        plt.subplots(nrows=2,ncols=1,figsize=(8,10.5),sharex=False,sharey=False)
        
        xMin = min(x[1])
        xMax = max(x[1])
        xx = np.linspace(xMin,xMax,100)
        yy = beta[0] + beta[1] * xx
        
        plt.subplot(212)
        plt.plot(xx,yy,label='Regression')
        plt.scatter(x[1],y,label='Data',color='red')
        plt.xlabel(colHeaders[1])
        plt.ylabel(colHeaders[0])
        plt.title('Plot of %s vs. %s with linear regression' % (colHeaders[0],colHeaders[1]))
        plt.legend(loc='lower right')
        plt.show()
        
        # Residuals plot
        plt.subplot(211)
        res = []
        for i in range(n):
            res.append(y[i] - (beta[0] + beta[1] * x[1][i]))
        plt.scatter(x[1],res,color='red')
        plt.plot(xx,np.mean(res)*np.ones(len(xx)))
        plt.xlabel(colHeaders[1])
        plt.ylabel('Residuals')
        plt.title('Plot of Residuals vs. %s' % (colHeaders[1]))
        
        print(res)
    
    
if __name__ == '__main__':
    fileName = "problem6.csv"
    performRegression(None)
    
    
    