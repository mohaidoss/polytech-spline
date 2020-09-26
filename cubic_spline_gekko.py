import numpy as np
import scipy as scp
import matplotlib.pyplot as plt
from gekko import GEKKO

a = 0
b = 5
n = 5

points = np.empty((n + 1,2)) #Ensemble de points
data = np.array([[0,0.1],[1,0.4],[2,0.2],[3,-0.2],[4,0.1],[5,0.5],[6,0.6]])  #Ensemble de points
xn = data[:,0]  #abscisses des points
yn = data[:,1]  #ordonnées des points 

m = GEKKO()
m.x = m.Param(value = np.linspace(-1,7))
m.y = m.Var()
m.options.IMODE=2
m.cspline(m.x,m.y,xn,yn)
m.solve(disp=False)
#help(m.cspline)


plt.plot(xn,yn,'bo',label='data')
plt.plot(m.x.value,m.y.value,'r--',label='cubic spline')
plt.legend(loc='best')
plt.show()
