import numpy as np
import scipy as scp
import matplotlib.pyplot as plt
from gekko import GEKKO

a = 0
b = 5
n = 5

points = np.empty((n + 1,2)) #Ensemble de points
xn = np.linspace(a,b,n+1)  #subdivision des abscisses de a à b
yn = np.array([0.1,0.2,0.3,0.5,1.0,0.9])
points[:,0] = xn

m = GEKKO()
m.x = m.Param(value = np.linspace(-1,6))
m.y = m.Var()
m.options.IMODE=2
m.cspline(m.x,m.y,xn,yn)
m.solve(disp=False)
#help(m.cspline)

p = GEKKO()
p.x = p.Var(value=1,lb=0,ub=5)
p.y = p.Var()
p.cspline(p.x,p.y,xn,yn)
p.Obj(-p.y)
p.solve(disp=False)

plt.plot(xn,yn,'bo',label='data')
plt.plot(m.x.value,m.y.value,'r--',label='cubic spline')
plt.plot(p.x.value,p.y.value,'ko',label='maximum')
plt.legend(loc='best')
plt.show()
