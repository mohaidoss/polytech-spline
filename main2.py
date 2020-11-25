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

#Ensemble de points
x = np.linspace(1,6,6)
n = 5
y = f_x(x)
#La dérivé à chaque point
yp = fp_x(x)

#Calcul des variations entre les x_i
h = np.zeros(5)
for i in range(5):
    h[i] = x[i+1] - x[i]

Q = np.matrix(np.zeros((6,4)))
for i in range(4):
    Q[i,i] = 1/h[i]
    Q[i+1,i] = -1/h[i] - 1/h[i+1]
    Q[i+2,i] = 1/h[i+1]

T = np.matrix(np.zeros((4,4)))
T[0,0] = 2*(h[0] + h[1])
T[1,0] = h[1]
T[n-2,n-2] = 2*(h[n-2] + h[n-1])
T[n-3,n-2] = h[n-2]
for i in range(1,n-2):
    T[i-1,i] = 2*(h[i])
    T[i,i] = 2*(h[0] + h[1])
    T[i+1,i] = h[i+1]
T = 1/3*T

p1=np.matmul(Q.transpose(),Q)
#Param_tre de lissage p
p = 0.5

A = p1 + p*T
Y = p * np.matmul(Q.transpose(),y)
Y = Y.reshape(n-1,1)

#Resolution avec la methode QR, A*S_c = Y

Qs,Rs = np.linalg.qr(A)

#B = tQs.Y
B = np.matmul(Qs.transpose(),Y)

#Ensemble des coeff S_c (Squeeze pour convertir la matrice (n-1)*1 en vecteur)
#R.c = B
S_c = np.squeeze(np.asarray(linalg.solve_triangular(Rs,B,0)))
print(S_c)

