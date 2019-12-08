# Galerkin Finite Element Method for Non Homogenous Heat Equation
# Author: Henry Rossiter

import numpy as np
from math import pi, exp, cos, sin
import matplotlib
import matplotlib.pyplot as plt
import sys

# 'forward' or 'backward' Euler time discretization
time_discretization = 'backward'

# Number of t-steps
nt = 512
delt = 1/(nt-1)

# Number of x-steps
nx = 11
delx = 1/(nx-1)

# t & x mesh
t = np.linspace(0,1,nt)
x = np.linspace(0,1,nx)

# Allocate matrices
F = np.zeros((nx-1,nt))
C = np.zeros((nx-2,nt))
U = np.zeros((nx,nt))

# source function (RHS of equation)
def f(x, t):
  return ((pi**2)-1)*exp(-t)*sin(pi*x)

# Integral of f on [xi-h/2, xi+h/2]
# Uses two point Gaussian Quadrature
def quadrature(x, h, t):
  a = x - h/2
  b = x + h/2
  x1 = -0.57735026918963
  x2 = 0.57735026918963
  return (f(((b-a)*x1+a+b)/2, t) + f(((b-a)*x2+a+b)/2, t))*(b-a)/2



# Build mass matrix
m = np.zeros((nx-2, nx-2))
for i in range(nx-2):
  m[i][i] = 4
for i in range(nx-3):
  m[i][i+1] = 1
  m[i+1][i] = 1
M = delx/6*m

# Build A matrix
a = np.zeros((nx-2, nx-2))
for i in range(nx-2):
  a[i][i] = 2
for i in range(nx-3):
  a[i][i+1] = -1
  a[i+1][i] = -1
A = 1/delx*a

# Approximate F at each meshpoint
for n in range(nt):
  for i in range(1, nx-1):
    F[i,n] = quadrature(x[i], delx, t[n])

# Discard first vector of F
F = F[1:,:]

# Perform time discretization
if time_discretization == 'forward':
  # Forward Euler time discretization
  for n in range(nt-1):
    C[:,n+1] = np.linalg.solve(M, delt*F[:,n]-delt*np.matmul(A, C[:,n]))+C[:,n]
elif time_discretization == 'backward':
  # Backward Euler time discretization
  for n in range(nt-1):
    C[:,n+1] = np.linalg.solve(delt*M+A, F[:,n+1]+delt*np.matmul(M, C[:,n]))
else:
  print('Time discretization must be either `forward` or `backward`')
  sys.exit(0)

# Solution U is C with BC's enforced
U[1:-1] = C

#Calculate exact solution on mesh
exact = np.zeros((1000,1000))
xx = np.linspace(0,1,1000)
tt = np.linspace(0,1,1000)
for n in range(len(tt)):
  for i in range(len(xx)):
    exact[i,n] = exp(-tt[n])*sin(pi*xx[i])

# fig, ax = plt.subplots()
# im = ax.imshow(U, aspect='auto')
# fig.tight_layout()
# ax.set(xlabel='t', ylabel='x', title='Numerical Solution, dt={}, dx={}'.format(delt, delx))
# fig.colorbar(im)
# plt.show()

fig, ax = plt.subplots()
ax.plot(x, U[:,-1], label='Numerical Solution')
ax.plot(xx, exact[:,-1], label='Analytical Solution')
ax.set(xlabel='x', ylabel='Heat', title='Solution at t=1. dt=1/{}, dx=1/{}'.format(nt-1, nx-1))
ax.legend()
ax.grid()
plt.show()