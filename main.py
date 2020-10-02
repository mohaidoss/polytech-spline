import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from math import * 


#Fonction cos²(x)
def f(x) :
    return (cos(x))**2
#Dérivée
def fp(x):
    return -2*cos(x)*sin(x)

f_x = np.vectorize(f)
fp_x = np.vectorize(fp)

#Ensemble de points, et dérivé
x = np.linspace(1,6,6)
y = f_x(x)
yp = fp_x(x)

#Système d'équations en forme matricielle A*X = Y
A = np.matrix([[x[0]**3,x[0]**2,x[0],1],[x[1]**3,x[1]**2,x[1],1],[3*x[0]**2,2*x[0],1,0],[3*x[1]**2,2*x[1],1,0]])

Y = np.array([y[0],y[1],yp[0],yp[1]])
Y = Y.reshape(4,1)

#Résolution en utilisant la méthode QR, A = Q.R
Q,R = np.linalg.qr(A)

#B = tQ.Y
B = np.matmul(Q.transpose(),Y)


#Coeff de S0 (Squeeze pour convertir la matrice 4*1 en vecteur)
#R.X = B
X = np.squeeze(np.asarray(linalg.solve_triangular(R,B,0)))
print(X)


#Polynome S0
poly = np.poly1d(X)

a = np.linspace(1,2,50)
b = poly(a)

#Plotting de S0
plt.plot(x,y,'bo',label='data')
plt.plot(a,b,'r--',label='cubic')
plt.legend(loc='best')
plt.show()