#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 16:02:54 2021

Copyright 2021 by Hadrien Montanelli.
"""

"""
Note: This code reproduces the numerical experiments of [1, Sec. 4].

[1] H. Montanelli, M. Aussal and H. Haddar, Computing weakly singular and 
near-singular integrals in high-order boundary elements, submitted.
"""

# Imports:
from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np
from sauterschwab import sauterschwab
from singint import map_func, map_jac, singint
from singintex import singintex
import os
os.environ['PATH'] += ':/usr/local/texlive/2017/bin/x86_64-darwin'
import time

# Plot options:
rc_fonts = {
    "text.usetex": True,
    'text.latex.preview': True,
    'mathtext.default': 'it',
    'mathtext.fontset': 'cm',
    'text.latex.preamble': [r'\usepackage{amsmath}'],
    'font.size': 12
}
rcParams.update(rc_fonts)
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

# %% Integral over a curved triangle (for a given quadrature size):

# Define a curved triangle:
a, b, c = 0.6, 0.7, 0.5
a1 = np.array([0, 0, 0])
a2 = np.array([1, 0, 0])
a3 = np.array([0, 1, 0])
a4 = np.array([0.5, 0, 0])
a5 = np.array([a, b, c])
a6 = np.array([0, 0.5, 0])
A = np.vstack((a1, a2, a3, a4, a5, a6))

# Source point x0:
u0, v0, dz0 = 0.2, 0.4, 1e-4
x0 = map_func(A)([u0, v0]) + dz0*np.array([0, 0, 1])

# Compute integral with singint2:
n = 10
p = 1
start = time.time()
I = singint(A, x0, n, p)
end = time.time()
print(f'Time:  {end-start:.4f}s')

# Compare with exact integral:
Iex = singintex(u0, v0, dz0)
error = np.abs(I - Iex)/np.abs(Iex)
print(f'Error: {error:.1e}')

# %% Integral over a curved triangle (convergence with quadrature size):

# Define a curved triangle:
a, b, c = 0.6, 0.7, 0.5
a1 = np.array([0, 0, 0])
a2 = np.array([1, 0, 0])
a3 = np.array([0, 1, 0])
a4 = np.array([0.5, 0, 0])
a5 = np.array([a, b, c])
a6 = np.array([0, 0.5, 0])
A = np.vstack((a1, a2, a3, a4, a5, a6))

# Source point x0:
u0, v0, dz0 = 0.5, 0.5, 0
x0 = map_func(A)([u0, v0]) + dz0*np.array([0, 0, 1])

# Compute integral with singint for increasing quadrature sizes:
Iex = singintex(u0, v0, dz0)
nn = np.array([2, 5, 10, 20, 50, 100, 200])
error1 = []
error2 = []
error3 = []
for n in nn:
    start = time.time()
    transplanted1 = [True]
    I1 = singint(A, x0, n, -1, transplanted1)
    error1.append(np.abs(I1 - Iex)/np.abs(Iex))
    transplanted2 = [True, True]
    I2 = singint(A, x0, n, 0, transplanted2)
    error2.append(np.abs(I2 - Iex)/np.abs(Iex))
    transplanted3 = [True, True, True]
    I3 = singint(A, x0, n, 1, transplanted3)
    error3.append(np.abs(I3 - Iex)/np.abs(Iex))
    end = time.time()
    print(f'Time:  {end-start:.4f}s') 
    
# Plot:
fig, ax = plt.subplots()
label1 = '$T_{-1}$ regularization'
label2 = '$T_0$ regularization'
label3 = '$T_1$ regularization'
plt.loglog(nn**2, error1, '.-', color='tab:blue', label=label1)
plt.loglog(nn**2, error2, 'x-', color='tab:red', label=label2)
plt.loglog(nn**2, error3, 'o-', color='tab:brown', label=label3, mfc='none')
if (u0 == 0.2):
    plt.loglog(nn**2, 4e-1/np.array(nn**2)**1, '--', color='tab:blue', linewidth=1.25)
    plt.loglog(nn**2, 4e-1/np.array(nn**2)**1.5, '--', color='tab:red', linewidth=1.25)
    plt.loglog(nn**2, 4e-1/np.array(nn**2)**2, '--', color='tab:brown', linewidth=1.25)
    plt.text(3.25e4, 2.5e-5, '$N^{-1}$', color='tab:blue')
    plt.text(3.25e4, 1.5e-7, '$N^{-1.5}$', color='tab:red')
    plt.text(3.25e4, 8e-10, '$N^{-2}$', color='tab:brown') 
if (u0 == 0.5):
    plt.loglog(nn**2, 3e-2/np.array(nn**2)**1, '--', color='tab:blue', linewidth=1.25)
    plt.loglog(nn**2, 1e-2/np.array(nn**2)**1.5, '--', color='tab:red', linewidth=1.25)
    plt.loglog(nn**2, 6e-2/np.array(nn**2)**2, '--', color='tab:brown', linewidth=1.25)
    plt.text(3.25e4, 1.5e-6, '$N^{-1}$', color='tab:blue')
    plt.text(3.25e4, 1.5e-8, '$N^{-1.5}$', color='tab:red')
    plt.text(3.25e4, 1.5e-10, '$N^{-2}$', color='tab:brown')
plt.xlim(1e0, 1e5)
plt.xlabel('Quadrature size $N$')
plt.ylabel('Relative error')
title1 = r'$\boldsymbol{x}_0 =$ ' + f'$F(${u0:1.0e}, {v0:1.0e}$)$'
if (dz0 != 0):
    title1 += f' + {dz0:1.0e}' + r'$\boldsymbol{z}$'
plt.title(title1)
ax.legend(loc='lower left', handletextpad=0.5)
plt.grid(1)
plt.savefig('../images/convergence_2D.pdf', dpi=256)

# %% Integral over two curved triangles (convergence with quadrature size):

# Define a curved triangle:
a, b, c = 0.5, 0.5, 1
a1 = np.array([0, 0, 0])
a2 = np.array([1, 0, 0])
a3 = np.array([0, 1, 0])
a4 = np.array([0.5, 0, 0])
a5 = np.array([a, b, c])
a6 = np.array([0, 0.5, 0])
A = np.vstack((a1, a2, a3, a4, a5, a6))

# Map and Jacobian:
F = map_func(A)
J = map_jac(A)

# 'Exact' integral is computed with the method of Sauter & Schwab:
start = time.time()
Iex = sauterschwab(20, a, b, c)
end = time.time()
print(f'Time:  {end-start:.4f}s') 

# Compute integral with singint for increasing quadrature sizes:
mm = np.arange(4, 16, 2)
error1 = []
error2 = []
error3 = []
for m in mm:
    start = time.time()
    x, w = np.polynomial.legendre.leggauss(m)
    I1 = 0
    I2 = 0
    I3 = 0
    for i in range(m):
        for j in range(m):
            aij = (1 - x[i])/2
            bij = (1 + x[i])*(1 - x[j])/4
            wij = 1/8*w[i]*(1 + x[i])*w[j]
            yij = np.array([aij, bij])
            Nij = np.linalg.norm(np.cross(J[0](yij), J[1](yij)))
            I1 += wij * Nij * singint(A, F(yij), m-1, -1)
            I2 += wij * Nij * singint(A, F(yij), m-1, 0)
            I3 += wij * Nij * singint(A, F(yij), m-1, 1)
    error1.append(np.abs(I1 - Iex)/Iex)
    error2.append(np.abs(I2 - Iex)/Iex)
    error3.append(np.abs(I3 - Iex)/Iex)
    end = time.time()
    print(f'Time:  {end-start:.4f}s') 

# Plot:
fig, ax = plt.subplots()
M = mm**4
label1 = '$T_{-1}$ regularization'
label2 = '$T_0$ regularization'
label3 = '$T_1$ regularization'
plt.loglog(M, error1, '.-', color='tab:blue', label=label1)
plt.loglog(M, error2, 'x-', color='tab:red', label=label2)
plt.loglog(M, error3, 'o-', color='tab:brown', label=label3, mfc='none')
plt.loglog(M, 2.5e-1/M**0.5, '--', color='tab:blue', linewidth=1.25)
plt.loglog(M, 2e-1/M**0.75, '--', color='tab:red', linewidth=1.25)
plt.loglog(M, 8e-1/M, '--', color='tab:brown', linewidth=1.25)
plt.xlim(1e2, 1e5)
plt.xlabel('Quadrature size $M$')
plt.ylabel('Relative error')
ax.legend(loc='lower left', handletextpad=0.5)
plt.text(4.5e4, 1.5e-3, r'$M^{-0.5}$', color='tab:blue')
plt.text(4.5e4, 1.25e-4, r'$M^{-0.75}$', color='tab:red')
plt.text(4.5e4, 1.5e-5, r'$M^{-1}$', color='tab:brown')
plt.grid(1)
plt.savefig('../images/convergence_4D.pdf', dpi=256)