import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from math import * 

x = np.linspace(1,6,6)
def f(x) :
    return (cos(x))**2

def fp(x):
    return -2*cos(x)*sin(x)

f_x = np.vectorize(f)
fp_x = np.vectorize(fp)

y = f_x(x)
yp = fp_x(x)


A = np.matrix([[x[0]**3,x[0]**2,x[0],1],[x[1]**3,x[1]**2,x[1],1],[3*x[0]**2,2*x[0],1,0],[3*x[1]**2,2*x[1],1,0]])
Y = np.array([y[0],y[1],yp[0],yp[1]])
Y = Y.reshape(4,1)


Q,R = np.linalg.qr(A)

B = np.matmul(Q.transpose(),Y)

X = linalg.solve_triangular(R,B,0)



print(x)
print(Y)

print("______________________________________")
print(Q)
print(R)
print("______________________________")
print(B)
