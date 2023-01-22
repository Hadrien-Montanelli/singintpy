#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 15:39:49 2021

Copyright 2021 by Hadrien Montanelli.
"""
import numpy as np

def sauterschwab(n, a, b, c):
    """
    Compute a singular integral over two identical quadratic triangles with 
    the method of Sauter & Schwab [2, Sec. 5.2.1]. The triangle is the one 
    used in the epxeriments of [1, Sec. 4]. (The map F below that defines the 
    triangle differs from that of [1] to match the reference triangle of [2].)
    Mathematically, it computes the integral 
    
            I = int_T int_T dS(x)dS(y)/|x-y| 
            
    over two identical quadratic triangles T.
    
    Inputs
    ------
    n : int
        The quadrature size in each direction.
        
    a, b, c : float
        The parameters of the quadratic triangle used in [1, Sec. 4].
        
    Output
    ------
    I : float
        The value of the integral.
        
    References
    ----------
    [1] H. Montanelli, M. Aussal and H. Haddar, Computing weakly singular and 
    near-singular integrals in high-order boundary elements, SISC (2022).
    [2] S. Sauter and C. Schwab, Boundary Element Methods, Springer, 2011.
    """
    # Mapping and Jacobian:
    l1 = 2*(2*a - 1)
    l2 = 2*(2*b - 1)
    F1 = lambda x1, x2: -l1*x2**2 + l1*x1*x2 + x1 - x2
    F2 = lambda x1, x2: -l2*x2**2 + l2*x1*x2 + x2
    F3 = lambda x1, x2: 4*c*(x1 - x2)*x2
    J1 = lambda x1, x2: (4*c*(x1 - x2))**2
    J2 = lambda x1, x2: (4*c*x2)**2
    J3 = lambda x1, x2: (1 + l2*(x1 - x2) + l1*x2)**2
    J = lambda x1, x2: np.sqrt(J1(x1,x2) + J2(x1,x2) + J3(x1,x2))

    # Integrand:
    G1 = lambda x1, x2, y1, y2: (F1(x1,x2) - F1(y1,y2))**2
    G2 = lambda x1, x2, y1, y2: (F2(x1,x2) - F2(y1,y2))**2
    G3 = lambda x1, x2, y1, y2: (F3(x1,x2) - F3(y1,y2))**2
    G = lambda x1, x2, y1, y2: 1/np.sqrt(G1(x1,x2,y1,y2) + G2(x1,x2,y1,y2) + G3(x1,x2,y1,y2))
    k = lambda x1, x2, y1, y2: G(x1, x2, y1, y2) * J(x1, x2) * J(y1, y2)

    # Modified integrand:
    k1t = lambda e1, e2, e3, xi: k(xi, xi*(1 - e1 + e1*e2), xi*(1 - e1*e2*e3), xi*(1 - e1))
    k1 = lambda e1, e2, e3, xi: xi**3 * e1**2 * e2 * k1t(e1, e2, e3, xi)
    k2t = lambda e1, e2, e3, xi: k(xi*(1 - e1*e2*e3), xi*(1 - e1), xi, xi*(1 - e1 + e1*e2))
    k2 = lambda e1, e2, e3, xi: xi**3 * e1**2 * e2 * k2t(e1, e2, e3, xi)
    k3t = lambda e1, e2, e3, xi: k(xi, xi*e1*(1 - e2 + e2*e3), xi*(1 - e1*e2), xi*e1*(1 - e2))
    k3 = lambda e1, e2, e3, xi: xi**3 * e1**2 * e2 * k3t(e1, e2, e3, xi)
    k4t = lambda e1, e2, e3, xi: k(xi*(1 - e1*e2), xi*e1*(1 - e2), xi, xi*e1*(1 - e2 + e2*e3))
    k4 = lambda e1, e2, e3, xi: xi**3 * e1**2 * e2 * k4t(e1, e2, e3, xi)
    k5t = lambda e1, e2, e3, xi: k(xi*(1 - e1*e2*e3), xi*e1*(1 - e2*e3), xi, xi*e1*(1 - e2))
    k5 = lambda e1, e2, e3, xi: xi**3 * e1**2 * e2 * k5t(e1, e2, e3, xi)
    k6t = lambda e1, e2, e3, xi: k(xi, xi*e1*(1 - e2), xi*(1 - e1*e2*e3), xi*e1*(1 - e2*e3))
    k6 = lambda e1, e2, e3, xi: xi**3 * e1**2 * e2 * k6t(e1, e2, e3, xi)

    # 4D Gauss-Legendre quadrature:
    [x, w] = np.polynomial.legendre.leggauss(n)
    x = 1/2*(1 + x)
    w = 1/2*w
    I1 = 0
    I2 = 0
    I3 = 0
    I4 = 0
    I5 = 0
    I6 = 0
    for i1 in range(n):
        for i2 in range(n):
            for i3 in range(n):
                for i4 in range(n):
                    I1 += w[i1]*w[i2]*w[i3]*w[i4]*k1(x[i1], x[i2], x[i3], x[i4])
                    I2 += w[i1]*w[i2]*w[i3]*w[i4]*k2(x[i1], x[i2], x[i3], x[i4])
                    I3 += w[i1]*w[i2]*w[i3]*w[i4]*k3(x[i1], x[i2], x[i3], x[i4])
                    I4 += w[i1]*w[i2]*w[i3]*w[i4]*k4(x[i1], x[i2], x[i3], x[i4])
                    I5 += w[i1]*w[i2]*w[i3]*w[i4]*k5(x[i1], x[i2], x[i3], x[i4])
                    I6 += w[i1]*w[i2]*w[i3]*w[i4]*k6(x[i1], x[i2], x[i3], x[i4])
    
    # Assemble and return:
    return  I1 + I2 + I3 + I4 + I5 + I6