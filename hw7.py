import numpy as np
from math import *
import matplotlib
import matplotlib.pyplot as plt


nt = 551
delt = 1/(nt-1)
nx = 11
delx = 1/(nx-1)

x = np.zeros(nx)
t = np.zeros(nt)
F = np.zeros((nx-1,nt))
exact = np.zeros((nx-1,nt))
C = np.zeros((nx-2,nt))
U = np.zeros((nx,nt))


def f(x, t):
  return ((pi**2)-1)*exp(-t)*sin(pi*x)

# Integral of f from xi -h/2 to xi + h/2
def quadrature(x, h, t):
  a = x - h/2
  b = x + h/2
  #print('evaluating integral from {} to {}'.format(a,b))
  x1 = -0.57735026918963
  x2 = 0.57735026918963
  sum = (f(((b-a)*x1+a+b)/2, t) + f(((b-a)*x2+a+b)/2, t))*(b-a)/2
  return sum
  


t = np.linspace(0,1,nt)
x = np.linspace(0,1,nx)

m = np.array([
  [4,1,0,0,0,0,0,0,0],
  [1,4,1,0,0,0,0,0,0],
  [0,1,4,1,0,0,0,0,0],
  [0,0,1,4,1,0,0,0,0],
  [0,0,0,1,4,1,0,0,0],
  [0,0,0,0,1,4,1,0,0],
  [0,0,0,0,0,1,4,1,0],
  [0,0,0,0,0,0,1,4,1],
  [0,0,0,0,0,0,0,1,4]
])

M = delx/6*m

a = np.array([
  [2,-1,0,0,0,0,0,0,0],
  [-1,2,-1,0,0,0,0,0,0],
  [0,-1,2,-1,0,0,0,0,0],
  [0,0,-1,2,-1,0,0,0,0],
  [0,0,0,-1,2,-1,0,0,0],
  [0,0,0,0,-1,2,-1,0,0],
  [0,0,0,0,0,-1,2,-1,0],
  [0,0,0,0,0,0,-1,2,-1],
  [0,0,0,0,0,0,0,-1,2]
])

A = 1/delx*a


for n in range(nt):
  for i in range(1, nx-1):
    x1 = x[i-1]+delx/2
    F[i,n] = quadrature(x[i], delx, t[n])
    exact[i,n] = exp(-t[n]*sin(pi*x[i]))

F = F[1:,:]
exact = exact[1:,:]

b = delt*F[:,n]-delt*A*C[:,n]+C[:,n]
# Forward Euler time discretization
for n in range(nt-1):
  C[:,n+1] = np.linalg.solve(M, delt*F[:,n]-delt*np.matmul(A, C[:,n]))+C[:,n]
  #print(C[:,n+1])

U[1:-1] = C

fig, ax = plt.subplots()
im = ax.imshow(U, aspect='auto')
fig.tight_layout()
fig.colorbar(im)
plt.show()