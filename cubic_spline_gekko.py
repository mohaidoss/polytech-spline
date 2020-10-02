import numpy as np
import scipy as scp
import matplotlib.pyplot as plt
from math import *
from gekko import GEKKO

def f(x):
    return (cos(x))**2

f_x = np.vectorize(f)

X = np.linspace(0,6,7)
Y = f_x(X)


print(Y)


a = X[0]
b = X[-1]

m = GEKKO()
m.x = m.Param(value = np.linspace(a,b))
m.y = m.Var()
m.options.IMODE=2
m.cspline(m.x,m.y,X,Y)
m.solve(disp=False)
#help(m.cspline)


plt.plot(X,Y,'bo',label='data')
plt.plot(m.x.value,m.y.value,'r--',label='cubic spline')
plt.legend(loc='best')
plt.show()

