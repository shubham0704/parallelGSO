
# coding: utf-8

from __future__ import division
import numpy as np
from scipy.optimize import minimize
from numpy import abs, cos, exp, mean, pi, prod, sin, sqrt, sum
import math,pylab

### Rosenbrock (Multimodal) Function
def rosen(x):
    """
    The Rosenbrock's function of N variables
    f(x) =  100*(x_i - x_{i-1}^2)^2 + (1- x_{1-1}^2)
    """
    x = np.asarray_chkfinite(x)
    n = len(x)
    a = 1
    b = 100
    score = 0
    for i in range(0, n-1):
        score += b * ((x[i+1] - x[i]**2)**2) + (a - x[i]) ** 2
    return score

### Rastrigin Function
def rastrigin( x ):  # rast.m
    x = np.asarray_chkfinite(x)
    n = len(x)
    return 10*n + sum( x**2 - 10 * cos( 2 * pi * x ))

### Ackley Function
def ackley( x, a=20, b=0.2, c=2*pi ):
    x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
    n = len(x)
    s1 = sum( x**2 )
    s2 = sum( cos( c * x ))
    return -a*exp( -b*sqrt( s1 / n )) - exp( s2 / n ) + a + exp(1)

### Weierstrass Function
def wer(a, x, e): # a is givin number, e is error tolerance

    Sum1 = 0
    Sum2 = 0
    k = 1

    while(True):
        #sine of pi times k to the a times x over pi times k to the a
        Sum1 = math.sin((math.pi)*pow(k,a)*(x))/((math.pi)*(pow(k,a)))
        Sum2 = Sum1 + math.sin((math.pi)*pow((k+1),a)*(x))/((math.pi)*pow((k+1),a))

        if (abs(Sum2-Sum1) < e):
            break
        else:
            k+=1
    return Sum1

def append(x0, xf, n):

    xl = [] #list containing x values
    yl = [] #corresponding y values
    dx = (xf-x0)/n #length of each subinterval

    for i in range (0, (n+1)):
        xval = x0 + (i * dx)
        yval = wer(a, xval, e) #ERROR HERE

        xl.append(xval)
        yl.append(yval)
        #print i,':',xl
    return xl, yl


### Non-Continuous Rastrigin Function

def nonContinuousRastrigin(x):
    y=[]
    for i in range(len(x)):
        temp=0
        if(abs(x[i]) <= 0.5):
            temp = x[i]
        else:
             temp = (2*x[i]+0.5)/2
        y.append(temp)
    y = np.asarray_chkfinite(y)
    n=len(x); A=10.
    return 10*n + sum( y**2 - 10 * cos( 2 * pi * y ))

## Unimodal Functions

### Bochevsky 2-dimensional for xi [-100, 100]
def bohachevsky(x):
    x = np.asarray_chkfinite(x)
    # X[1] X[2]
    foo = x[0]**2 + 2 * x[1]**2 - 0.3 * cos(3*pi*x[0]) - 0.4 * cos(4*pi*x[1]) + 0.7
    return foo

### Matyas 2-dimensional for x,y [-10, 10]

def Matyas(x):
    x = np.asarray_chkfinite(x)
    foo = 0.26*(x[0]**2 + x[1]**2) - 0.48 * x[0] * x[1]
    return foo

### n-dimensional xi [-1,1]
def exponential(x):
    x = np.asarray_chkfinite(x)
    foo = -exp(-0.5 * sum(x**2))
    return foo

### n-dimensional xi [-1,1]
def powellsumfcn(x):
    x = np.asarray_chkfinite(x)
    n = len(x)

    scores = 0
    absX = abs(x)
    for i in range(0,n):
        scores = scores + (absX[i] ** (i+1))
    return scores

### n-dimensional xi [-100, 100]
def schfewel_220(x):
    x = np.asarray_chkfinite(x)
    foo = sum(abs(x))
    return foo

### Sum of squares function xi [-10, 10] n-dimensional
def sum_of_squares(x):
    x = np.asarray_chkfinite(x)
    foo = sum([(i+1) * (x[i] ** 2) for i in range(0,len(x))])
    return foo

### n-dimensional xi [-100, 100]
def schwefel_222(x):
    x = np.asarray_chkfinite(x)
    foo = sum(abs(x)) + np.prod(abs(x))
    return foo

### Griewangk Function n-dimensional xi [-600, 600]
def griewank( x, fr=4000 ):
    x = np.asarray_chkfinite(x)
    n = len(x)
    j = np.arange( 1., n+1 )
    s = sum( x**2 )
    p = prod( cos( x / sqrt(j) ))
    return s/fr - p + 1

### Zakharov Function xi [-5, 10]
def zakharov( x ):  # zakh.m
    x = np.asarray_chkfinite(x)
    n = len(x)
    j = np.arange( 1., n+1 )
    s2 = sum( j * x ) / 2
    return sum( x**2 ) + s2**2 + s2**4

### Sphere Function
def sphere( x ):
    x = np.asarray_chkfinite(x)
    return sum( x**2 )

# ### Shifted Rotated High Conditioned Elliptic Function
# def srhe()

class BenchMark():
	def __init__(self):
		pass


### Bounds for different funcitons:
 

# - ackley._bounds       = [-15, 30]
# - dixonprice._bounds   = [-10, 10]
# - griewank._bounds     = [-600, 600]
# - levy._bounds         = [-10, 10]
# - michalewicz._bounds  = [0, pi]
# - perm._bounds         = ["-dim", "dim"]  # min at [1 2 .. n]
# - powell._bounds       = [-4, 5]  # min at tile [3 -1 0 1]
# - powersum._bounds     = [0, "dim"]  # 4d min at [1 2 3 4]
# - rastrigin._bounds    = [-5.12, 5.12]
# - rosenbrock._bounds   = [-2.4, 2.4]  # wikipedia
# - schwefel._bounds     = [-500, 500]
# - sphere._bounds       = [-5.12, 5.12]
# - sum2._bounds         = [-10, 10]
# - trid._bounds         = ["-dim**2", "dim**2"]  # fmin -50 6d, -200 10d
# - zakharov._bounds     = [-5, 10]
# 
# - ellipse._bounds      =  [-2, 2]
# - logsumexp._bounds    = [-20, 20]  # ?
# - nesterov._bounds     = [-2, 2]
# - powellsincos._bounds = [ "-20*pi*dim", "20*pi*dim"]
# - randomquad._bounds   = [-10000, 10000]
# - saddle._bounds = [-3, 3]
