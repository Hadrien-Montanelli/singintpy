#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 15:39:49 2021

Copyright 2021 by Hadrien Montanelli.

Note: This is the code displayed in Appendix D of [1].

[1] H. Montanelli, M. Aussal and H. Haddar, Computing weakly singular and 
near-singular integrals in high-order boundary elements, SISC (2022).
"""

# Imports:
import numpy as np
from scipy.optimize import minimize
from singint2ex import singint2ex

# Map for transplanted Gauss:
def confmap(z, mu, nu):
    a = np.arcsinh((1 - mu)/nu)
    b = np.arcsinh((1 + mu)/nu)
    g = mu + nu * np.sinh((a + b)*(z - 1)/2 + a)
    dg = nu * (a + b)/2 * np.cosh((a + b)*(z - 1)/2 + a)
    return g, dg

# Step 1 - Mapping back:
a, b, c = 0.6, 0.7, 0.5                
Fx = lambda x: x[0] + 2*(2*a-1)*x[0]*x[1]
Fy = lambda x: x[1] + 2*(2*b-1)*x[0]*x[1]
Fz = lambda x: 4*c*x[0]*x[1]
F = lambda x: np.array([Fx(x), Fy(x), Fz(x)])                            # map 
J1 = lambda x: np.array([1 + 2*(2*a-1)*x[1], 2*(2*b-1)*x[1], 4*c*x[1]])  # Jacobian (1st col)
J2 = lambda x: np.array([2*(2*a-1)*x[0], 1 + 2*(2*b-1)*x[0], 4*c*x[0]])  # Jacobian (2nd col)
x0 = F([0.5, 1e-4]) + 1e-4*np.array([0, 0, 1])                           # singularity

# Step 2 - Locating the singularity:
e = lambda x: F(x) - x0                                            
E = lambda x: np.linalg.norm(e(x))**2                                    # cost function
dE = lambda x: 2*np.array([e(x) @ J1(x), e(x) @ J2(x)])                  # gradient
x0h = minimize(E, np.zeros(2), method='BFGS', jac=dE, tol=1e-12).x       # minimization
h = np.linalg.norm(F(x0h) - x0)                                       

# Step 3 - Taylor & 2D Gauss quadrature:  
n = 10; t, w = np.polynomial.legendre.leggauss(n)                        # 1D wts/pts
W = 1/8*np.outer(w*(1+t), w)                                             # 2D wts
X = np.array([1/2*np.outer(1-t, np.ones(n)), 1/4*np.outer(1+t, 1-t)])    # 2D pts
psi = lambda x: np.linalg.norm(np.cross(J1(x), J2(x), axis=0), axis=0)
tmp = lambda x,i: F(x)[i] - x0[i]
nrm = lambda x: np.sqrt(sum(tmp(x,i)**2 for i in range(3)))
tmp0 = lambda x,i: J1(x0h)[i]*(x[0]-x0h[0]) + J2(x0h)[i]*(x[1]-x0h[1])
nrm0 = lambda x: np.sqrt(sum(tmp0(x,i)**2 for i in range(3)))
f = lambda x: psi(x)/nrm(x) - psi(x0h)/nrm0(x)                           # regularized integrand
I = np.sum(W * f(X))                                                     # 2D Gauss
    
# Steps 4 & 5 - Continuation & 1D (transplanted) Gauss quadrature:
s1, s2, s3 = x0h[1], np.sqrt(2)/2*(1-x0h[0]-x0h[1]), x0h[0]              # Distances
dr1, dr2, dr3 = 1/2, np.sqrt(2)/2, 1/2
tmp = lambda t,r,i: (J1(x0h)[i]*r(t)[0] + J2(x0h)[i]*r(t)[1])**2
g, dg = confmap(t, -1 + 2*x0h[0], 2*s1)                       
          
r = lambda t: np.array([-x0h[0] + (t+1)/2,  -x0h[1]])                    # edge r1
nrm = lambda t: np.sqrt(tmp(t,r,0) + tmp(t,r,1) + tmp(t,r,2))
f = lambda t: (np.sqrt(nrm(t)**2 + h**2) - h)/nrm(t)**2
I += psi(x0h) * s1 * dr1 * (dg * w @ f(g))                               # 1D transplanted Gauss

r = lambda t: np.array([1 - x0h[0] - (t+1)/2, -x0h[1] + (t+1)/2])        # edge r2
nrm = lambda t: np.sqrt(tmp(t,r,0) + tmp(t,r,1) + tmp(t,r,2))
f = lambda t: (np.sqrt(nrm(t)**2 + h**2) - h)/nrm(t)**2
I += psi(x0h) * s2 * dr2 * (w @ f(t))                                    # 1D Gauss

r = lambda t: np.array([-x0h[0], 1 - x0h[1] - (t+1)/2])                  # edge r3
nrm = lambda t: np.sqrt(tmp(t,r,0) + tmp(t,r,1) + tmp(t,r,2))
f = lambda t: (np.sqrt(nrm(t)**2 + h**2) - h)/nrm(t)**2
I += psi(x0h) * s3 * dr3 * (w @ f(t))                                    # 1D Gauss

# Check error:
Iex = singint2ex(0.5, 1e-4, 1e-4)
error = np.abs(I - Iex)/np.abs(Iex)
print(f'Error (semi-a): {error:.1e}')