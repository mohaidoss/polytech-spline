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
#Initialisation de la matrice Q
Q = np.matrix(np.zeros((n+1,n-1)))
for i in range(n-1):
    Q[i,i] = 1/h[i]
    Q[i+1,i] = -1/h[i] - 1/h[i+1]
    Q[i+2,i] = 1/h[i+1]
#Initialisation de la matrice T
T = np.matrix(np.zeros((n-1,n-1)))
#Définition de la première colonne
T[0,0] = 2*(h[0] + h[1])
T[1,0] = h[1]
#Définition de la dernière colonne
T[n-2,n-2] = 2*(h[n-2] + h[n-1])
T[n-3,n-2] = h[n-2]
#Remplir les autres colonnes
for i in range(1,n-2):
    T[i-1,i] = h[i]
    T[i,i] = 2*(h[0] + h[1])
    T[i+1,i] = h[i+1]
#Division par 3
T = 1/3*T
sigma2 = np.matmul(np.diag(np.ones(n+1)),np.diag(np.ones(n+1)))


p1=np.matmul(Q.transpose(),sigma2)
#Param_tre de lissage p
p = 5555555

A = np.matmul(p1,Q) + p*T
Y = p * np.matmul(Q.transpose(),y)
Y = Y.reshape(n-1,1)


#Resolution avec la methode QR, A*S_c = Y

Qs,Rs = np.linalg.qr(A)

#B = tQs.Y
B = np.matmul(Qs.transpose(),Y)

#Ensemble des coeff S_c (Squeeze pour convertir la matrice (n-1)*1 en vecteur)
#R.S_c = B
S_c = np.squeeze(np.asarray(linalg.solve_triangular(Rs,B,0)))

S_a = np.squeeze(np.asarray(y-np.matmul(sigma2.dot(Q),S_c)/p))
print(S_a)

S_c = np.append(S_c,0)
S_c = np.insert(S_c,0,0)
print(S_c)
S_d = []
for i in range(n):
    S_d = S_d + [(S_c[i+1]-S_c[i])/(3*h[i])]
print(S_d)

S_b = []
for i in range(n):
    S_b = S_b + [ -1*S_c[i]*h[i] - S_d[i]*h[i]*h[i] + (S_a[i+1]-S_a[i])/h[i]]
print(S_b)

def fs(a,n):
    return S_d[n]*(a-x[n])**3 + S_c[n]*(a-x[n])**2 + S_b[n]*(a-x[n]) + S_a[n]
f_sol = np.vectorize(fs)


for i in range(n):

    #Polynome Si
    poly = np.poly1d([S_d[i],S_c[i],S_b[i],S_a[i]])
    print(poly)
    a = np.linspace(x[i],x[i+1],50)
    b = f_sol(a,i)
    plt.plot(a,b,'r--',label='cubic')

#Plotting de S0
plt.plot(x,y,'bo',label='data')
plt.legend(loc='best')
plt.show()

