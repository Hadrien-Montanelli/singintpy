#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 15:39:49 2021

Copyright 2021 by Hadrien Montanelli.
"""
import numpy as np
from numpy.linalg import norm
from scipy.optimize import minimize

def singint(A, x0, n, p, trans=[True, True, True], optim='BFGS', quad='numerical'):
    """
    Compute a singular or near-singular integral over a quadratic triangle via
    the algorithm presented in [1]. Mathematically, it computes the integral 
    
            I = int_T dS(x)/|x-x0|
    
    over a quadratic triangle T.
        
    Inputs
    ------
    A : numpy.ndarray
        The six points that define the quadratic triangle as a 6x3 matrix.
        
    x0 : numpy.ndarray
        The singularity as a 3x1 vector.
        
    n : int
        The quadrature size; 2D integrals will use n*n points, 1D integrals
        will use 10*n points.
    
    p : int
        The regularization order; -1, 0, or 1 for T_{-1}, T_0 or T_1 
        regularization.
    
    trans : list
        A list of bool variables that specify to if one wants to use
        transplanted quadrature for T_{-1}, T_0 and T_1 regularization.
        
    optim : str
        The method used for locating the singularity ('BFGS' or 'Newton').
        
    quad : str
        The method for computing the integral of T_{-1} ('numerical or 'exact').
        
    Output
    ------
    I : float
        The value of the integral.
        
    References
    ----------
    [1] H. Montanelli, M. Aussal and H. Haddar, Computing weakly singular and 
    near-singular integrals in high-order boundary elements, SISC (2022).
    """
    # Tolerance for optimization and near-singularities:
    tol = 1e-12
            
    # Step 1: Map back to reference triangle.
    F = map_func(A)
    J = map_jac(A)
    H = map_hess(A)
    
    # Step 2: Locating the singularity or near-singularity.
    x0h = locate_sing(x0, F, J, H, optim, tol)
    eh = F(x0h) - x0
    h = norm(eh)
    if (h > tol):
        eh /= h
    else:
        h = 0
        eh = np.zeros(3)
    
    # Step 3: Compute regularized part with 2D quadrature.
    Ireg = compute_Ireg(x0, x0h, h, eh, F, J, H, n, p)

    # Step 4: Integrate Taylor terms.
    scl = 10
    In1 = compute_In1(x0h, h, J, scl*n, quad, trans, tol)
    if (p == -1):
        I = In1 + Ireg
    elif (p > -1):
        I0 = compute_I0(x0h, h, eh, J, H, scl*n, trans, tol)
        if (p == 0):
            I = In1 + I0 + Ireg
        elif (p == 1):
            I1 = compute_I1(x0h, h, eh, J, H, scl*n, trans, tol)
            I = In1 + I0 + I1 + Ireg
            
    return I

def locate_sing(x0, F, J, H, method, tol):
    
    # Cost function and gradient:
    e = lambda x: F(x) - x0
    cost_func = lambda x: norm(e(x))**2
    cost_grad = lambda x: 2*np.array([e(x) @ J[0](x), e(x) @ J[1](x)])
    
    # BFGS:
    if (method == 'BFGS'):
        res = minimize(cost_func,  np.zeros(2), method='BFGS', jac=cost_grad, 
                   options={'disp': False}, tol=tol)
        x0h = res.x
  
        return x0h
        
    # Gradient descent followed by Newton's method:
    if (method == 'Newton'):
        cost_hess = lambda x: 2*np.array([[e(x)@H[0](x) + J[0](x)@J[0](x), 
                                           e(x)@H[1](x) + J[0](x)@J[1](x)], 
                                          [e(x)@H[1](x) + J[0](x)@J[1](x), 
                                           e(x)@H[2](x) + J[1](x)@J[1](x)]])
        x0h = np.zeros(2)
        G0 = np.ones_like(x0h)
        c = 1e-4
        beta = 1e-3
        tau = 0.5
        itr_max = 1000
        itr = 0
        while (norm(G0) > tol and itr < itr_max):
            G0 = cost_grad(x0h)
            H0 = cost_hess(x0h)
            l = np.min(np.linalg.eig(H0)[0]);
            if (l < 0):
                H0 += max(0, beta - l)*np.eye(2);
            p = -np.linalg.solve(H0, G0)
            alpha = 1
            if (norm(G0) > 1e-2):
                while (cost_func(x0h + alpha*p) > cost_func(x0h) + c*alpha*(p @ G0)):
                    alpha *= tau
            x0h = x0h + alpha*p
            itr += 1
 
        return x0h
        
def compute_Ireg(x0, x0h, h, eh, F, J, H, n, p):
    # Note: the variable v belows correponds to psi in [1].
    
    # Singular integrand:
    X0 = np.zeros([3, n, n])
    X0[0] = x0[0]
    X0[1] = x0[1]
    X0[2] = x0[2]
    tmp = lambda x: norm(np.cross(J[0](x), J[1](x), axis=0), axis=0)
    nrm = lambda x: norm(F(x) - X0, axis=0)
    sing_func = lambda x: tmp(x)/nrm(x)
    
    # Tn1 term:
    X0h = np.zeros([2, n, n])
    X0h[0] = x0h[0]
    X0h[1] = x0h[1]
    tmp0 = norm(np.cross(J[0](x0h), J[1](x0h)))
    J0 = np.vstack((J[0](x0h), J[1](x0h))).T
    nrm0 = lambda x: np.sqrt(norm(mul_func(J0, x - X0h, n), axis=0)**2 + h**2)
    Tn1 = lambda x: tmp0/nrm0(x)
    
    # T0 term:
    if (p > -1):
        
        # Jacobian and Hessian:
        J1 = J[0](x0h)
        J2 = J[1](x0h)
        H11 = H[0](x0h)
        H12 = H[1](x0h)
        H22 = H[2](x0h)
        
        # Compute v and its first derivatives:
        P = np.cross(J1, J2)
        dP1 = np.cross(H11, J2) + np.cross(J1, H12)
        dP2 = np.cross(H12, J2) + np.cross(J1, H22)
        v = norm(P)
        dv1 = 1/v * P @ dP1
        dv2 = 1/v * P @ dP2
        
        # Compute the coefficients:
        a1 = eh @ H11
        a2 = 2 * (eh @ H12)
        a3 = eh @ H22
        c1 = J1 @ H11
        c2 = 2 * (J1 @ H12) + J2 @ H11
        c3 = 2 * (J2 @ H12) + J1 @ H22
        c4 = J2 @ H22
        
        # Term with v':
        dx1 = lambda x: x[0] - x0h[0]
        dx2 = lambda x: x[1] - x0h[1]
        dv = lambda x: dv1*dx1(x) + dv2*dx2(x)
        T0v = lambda x: dv(x)/nrm0(x)
        
        # Terms with a's:
        Ta1 = lambda x: (-h*v/2*a1) * dx1(x)**2/nrm0(x)**3
        Ta2 = lambda x: (-h*v/2*a2) * dx1(x)*dx2(x)/nrm0(x)**3
        Ta3 = lambda x: (-h*v/2*a3) * dx2(x)**2/nrm0(x)**3
        Ta = lambda x: Ta1(x) + Ta2(x) + Ta3(x)
        
        # Terms with c's:
        Tc1 = lambda x: (-v/2*c1) * dx1(x)**3/nrm0(x)**3
        Tc2 = lambda x: (-v/2*c2) * dx1(x)**2*dx2(x)/nrm0(x)**3
        Tc3 = lambda x: (-v/2*c3) * dx1(x)*dx2(x)**2/nrm0(x)**3
        Tc4 = lambda x: (-v/2*c4) * dx2(x)**3/nrm0(x)**3
        Tc = lambda x: Tc1(x) + Tc2(x) + Tc3(x) + Tc4(x)
        
        # Assemble:
        T0 = lambda x: T0v(x) + Ta(x) + Tc(x)
                   
    # T1 term:
    if (p > 0):

        # Compute second derivatives of v:
        dP11 = 2 * np.cross(H11, H12)
        dP12 = np.cross(H11, H22) + np.cross(H12, H12)
        dP22 = 2 * np.cross(H12, H22)
        dv11 = 1/v**2 * (v * dP1 @ dP1 + v * P @ dP11 - dv1 * P @ dP1)
        dv12 = 1/v**2 * (v * dP1 @ dP2 + v * P @ dP12 - dv2 * P @ dP1)
        dv22 = 1/v**2 * (v * dP2 @ dP2 + v * P @ dP22 - dv2 * P @ dP2)

        # Compute the coefficients:
        d1 = 1/4 * (H11 @ H11)
        d2 = H11 @ H12
        d3 = 1/2 * (H11 @ H22) + H12 @ H12
        d4 = H22 @ H12
        d5 = 1/4 * (H22 @ H22)
        e1 = -1/2 * a1*dv1
        e2 = -1/2 * (a2*dv1 + a1*dv2)
        e3 = -1/2 * (a3*dv1 + a2*dv2)
        e4 = -1/2 * a3*dv2
        f1 = 3/8 * v * a1**2
        f2 = 3/8 * v * 2*a1*a2
        f3 = 3/8 * v * (a2**2 + 2*a1*a3)
        f4 = 3/8 * v * 2*a2*a3
        f5 = 3/8 * v * a3**2
        g1 = -1/2 * (c1*dv1 + d1*v)
        g2 = -1/2 * (c2*dv1 + c1*dv2 + d2*v)
        g3 = -1/2 * (c3*dv1 + c2*dv2 + d3*v)
        g4 = -1/2 * (c4*dv1 + c3*dv2 + d4*v)
        g5 = -1/2 * (c4*dv2 + d5*v)
        h1 = 3/8 * v * c1**2
        h2 = 3/8 * v * 2*c1*c2
        h3 = 3/8 * v * (c2**2 + 2*c1*c3)
        h4 = 3/8 * v * (2*c1*c4 + 2*c2*c3)
        h5 = 3/8 * v * (c3**2 + 2*c2*c4)
        h6 = 3/8 * v * 2*c3*c4
        h7 = 3/8 * v * c4**2
    
        # Tem with v'':
        dx1 = lambda x: x[0] - x0h[0]
        dx2 = lambda x: x[1] - x0h[1]
        d2v = lambda x: 1/2*dv11*dx1(x)**2 + 1/2*dv22*dx2(x)**2
        T1v = lambda x: (d2v(x) + dv12*dx1(x)*dx2(x))/nrm0(x)

        # Terms with e's:
        Te1 = lambda x: (h*e1) * dx1(x)**3/nrm0(x)**3
        Te2 = lambda x: (h*e2) * dx1(x)**2*dx2(x)/nrm0(x)**3
        Te3 = lambda x: (h*e3) * dx1(x)*dx2(x)**2/nrm0(x)**3
        Te4 = lambda x: (h*e4) * dx2(x)**3/nrm0(x)**3
        Te = lambda x: Te1(x) + Te2(x) + Te3(x) + Te4(x)
        
        # Terms with f's:
        Tf1 = lambda x: (h**2*f1) * dx1(x)**4/nrm0(x)**5
        Tf2 = lambda x: (h**2*f2) * dx1(x)**3*dx2(x)/nrm0(x)**5
        Tf3 = lambda x: (h**2*f3) * dx1(x)**2*dx2(x)**2/nrm0(x)**5
        Tf4 = lambda x: (h**2*f4) * dx1(x)*dx2(x)**3/nrm0(x)**5
        Tf5 = lambda x: (h**2*f5) * dx2(x)**4/nrm0(x)**5
        Tf = lambda x: Tf1(x) + Tf2(x) + Tf3(x) + Tf4(x) + Tf5(x)
        
        # Terms with g's:
        Tg1 = lambda x: g1 * dx1(x)**4/nrm0(x)**3
        Tg2 = lambda x: g2 * dx1(x)**3*dx2(x)/nrm0(x)**3
        Tg3 = lambda x: g3 * dx1(x)**2*dx2(x)**2/nrm0(x)**3
        Tg4 = lambda x: g4 * dx1(x)*dx2(x)**3/nrm0(x)**3
        Tg5 = lambda x: g5 * dx2(x)**4/nrm0(x)**3
        Tg = lambda x: Tg1(x) + Tg2(x) + Tg3(x) + Tg4(x) + Tg5(x)
        
        # Terms with h's:
        Th1 = lambda x: h1 * dx1(x)**6/nrm0(x)**5
        Th2 = lambda x: h2 * dx1(x)**5*dx2(x)/nrm0(x)**5
        Th3 = lambda x: h3 * dx1(x)**4*dx2(x)**2/nrm0(x)**5
        Th4 = lambda x: h4 * dx1(x)**3*dx2(x)**3/nrm0(x)**5
        Th5 = lambda x: h5 * dx1(x)**2*dx2(x)**4/nrm0(x)**5
        Th6 = lambda x: h6 * dx1(x)*dx2(x)**5/nrm0(x)**5
        Th7 = lambda x: h7 * dx2(x)**6/nrm0(x)**5
        Th15 = lambda x: Th1(x) + Th2(x) + Th3(x) + Th4(x) + Th5(x)
        Th67 = lambda x: Th6(x) + Th7(x)
        Th = lambda x: Th15(x) + Th67(x)
        
        # Assemble:
        T1 = lambda x: T1v(x) + Te(x) + Tf(x) + Tg(x) + Th(x)
        
    # Regularized integrand:    
    if (p == -1):
        reg_func = lambda x: sing_func(x) - Tn1(x)
    elif (p == 0):
        reg_func = lambda x: sing_func(x) - Tn1(x) - T0(x)
    elif (p == 1):
        reg_func = lambda x: sing_func(x) - Tn1(x) - T0(x) - T1(x)
        
    # 2D Gauss quadrature:
    x, w = np.polynomial.legendre.leggauss(n)
    W = 1/8 * np.outer(w*(1 + x), w)
    x1 = 1/2 * np.outer(1 - x, np.ones(n))
    x2 = 1/4 * np.outer(1 + x, 1 - x)
    X = np.array([x1, x2])
    Ireg = np.sum(W * reg_func(X))
        
    return Ireg

def compute_In1(x0h, h, J, n, quad, trans, tol):
    # Note: the variable v belows correponds to psi in [1].
    
    # Exact quadrature:
    if (quad == 'exact'):
        
        # Compute triangle data:
        J = np.vstack((J[0](x0h), J[1](x0h))).T
        a1 = J @ np.array([0, 0])
        a2 = J @ np.array([1, 0])
        a3 = J @ np.array([0, 1])
        tau1 = a2 - a1
        tau2 = a3 - a2
        tau3 = a1 - a3
        tau1 = tau1/norm(tau1)
        tau2 = tau2/norm(tau2)
        tau3 = tau3/norm(tau3)
        n = np.cross(a2 - a1, a3 - a2) 
        A = 0.5*norm(n)
        n /= (2*A)
        nu1 = np.cross(tau1, n)
        nu2 = np.cross(tau2, n)
        nu3 = np.cross(tau3, n)
        nu1 = nu1/norm(nu1)
        nu2 = nu2/norm(nu2)
        nu3 = nu3/norm(nu3)
        
        # Antiderivative:
        if (h < tol):
            R = lambda t,s,h: np.arcsinh(t/s)
        else:
            R0 = lambda t,s,h: np.arcsinh(t/np.sqrt(s**2 + h**2))
            R1 = lambda t,s,h: np.arctan(t/s) 
            tmp0 = lambda t,s,h: s**2 + h**2 + 1j*t*s
            tmp1 = lambda t,s,h: h * np.sqrt(s**2 + h**2 + t**2)
            tmp2 = lambda t,s,h: np.arctanh(tmp0(t,s,h)/tmp1(t,s,h))
            R2 = lambda t,s,h: np.imag(tmp2(t,s,h) - 1j*np.pi/2*sign_func(t))
            R = lambda t,s,h: R0(t,s,h) - h/s * R1(t,s,h) - h/s * R2(t,s,h)
            
        # Integrate on each edge:
        In1 = 0
        s1 = (a1 - J @ x0h) @ nu1
        s2 = (a2 - J @ x0h) @ nu2
        s3 = (a3 - J @ x0h) @ nu3
        if (abs(s1) > tol):
            t1p = (a2 - J @ x0h) @ tau1
            t1m = (a1 - J @ x0h) @ tau1
            In1 += s1 * (R(t1p, s1, h) - R(t1m, s1, h))
        if (abs(s2) > tol):
            t2p = (a3 - J @ x0h) @ tau2
            t2m = (a2 - J @ x0h) @ tau2
            In1 += s2 * (R(t2p, s2, h) - R(t2m, s2, h))
        if (abs(s3) > tol):
            t3p = (a1 - J @ x0h) @ tau3
            t3m = (a3 - J @ x0h) @ tau3
            In1 += s3 * (R(t3p, s3, h) - R(t3m, s3, h))
    
        # Scale by area:
        In1 *= norm(np.cross(J[:,0], J[:,1]))/(2*A)
            
    # Numerical quadrature:
    if (quad == 'numerical'):
        
        # Points and weight for 1D Gauss quadrature on [-1,1]:
        t0, w0 = np.polynomial.legendre.leggauss(n)
            
        # Jacobian:
        J1 = J[0](x0h)
        J2 = J[1](x0h)
        
        # Compute v:
        P = np.cross(J1, J2)
        v = norm(P)
    
        # Integrals on each edge:
        s1 = x0h[1]
        s2 = np.sqrt(2)/2*(1 - x0h[0] - x0h[1])
        s3 = x0h[0]
        In1 = 0
        if (abs(s1) > tol):
 
            # Transplanted Gauss quadrature if near-vertex:
            if (abs(s1) <= 1e-2 and trans[0] == True):
                t, scl = conf_func(t0, -1 + 2*x0h[0], 2*s1)
                w = w0 * scl
            else:
                t = t0
                w = w0
                
            # Paramatretization and norm:
            r1 = lambda t: -x0h[0] + (t + 1)/2 
            r2 = lambda t: -x0h[1]
            nrm0 = lambda t: (J1[0]*r1(t) + J2[0]*r2(t))**2
            nrm1 = lambda t: (J1[1]*r1(t) + J2[1]*r2(t))**2
            nrm2 = lambda t: (J1[2]*r1(t) + J2[2]*r2(t))**2
            nrm = lambda t: np.sqrt(nrm0(t) + nrm1(t) + nrm2(t))
            
            # Term with v:
            tmp = lambda t: (np.sqrt(nrm(t)**2 + h**2) - h)/nrm(t)**2
            g = lambda t: v*tmp(t)
            In1 += s1/2 * (w @ g(t))
         
        if (abs(s2) > tol):
            
            # Transplanted Gauss quadrature if near-vertex:
            if (s2 <= 1e-2 and trans[0] == True):
                t, scl = conf_func(t0, -1 + 2*x0h[1], 2*s2)
                w = w0 * scl
            else:
                t = t0
                w = w0
                
            # Paramatretization and norm:
            r1 = lambda t: 1 - x0h[0] - (t + 1)/2
            r2 = lambda t: -x0h[1] + (t + 1)/2
            nrm0 = lambda t: (J1[0]*r1(t) + J2[0]*r2(t))**2
            nrm1 = lambda t: (J1[1]*r1(t) + J2[1]*r2(t))**2
            nrm2 = lambda t: (J1[2]*r1(t) + J2[2]*r2(t))**2
            nrm = lambda t: np.sqrt(nrm0(t) + nrm1(t) + nrm2(t))
            
            # Term with v:
            tmp = lambda t: (np.sqrt(nrm(t)**2 + h**2) - h)/nrm(t)**2
            g = lambda t: v*tmp(t)
            In1 += np.sqrt(2)*s2/2 * (w @ g(t))
      
        if (abs(s3) > tol):
        
            # Transplanted Gauss quadrature if near-vertex:
            if (s3 <= 1e-2 and trans[0] == True):
                t, scl = conf_func(t0, -1 + 2*(1 - x0h[1]), 2*s3)
                w = w0 * scl
            else:
                t = t0
                w = w0
            
            # Paramatretization and norm:
            r1 = lambda t: -x0h[0]
            r2 = lambda t: 1 - x0h[1] - (1 + t)/2
            nrm0 = lambda t: (J1[0]*r1(t) + J2[0]*r2(t))**2
            nrm1 = lambda t: (J1[1]*r1(t) + J2[1]*r2(t))**2
            nrm2 = lambda t: (J1[2]*r1(t) + J2[2]*r2(t))**2
            nrm = lambda t: np.sqrt(nrm0(t) + nrm1(t) + nrm2(t))
            
            # Term with v:
            tmp = lambda t: (np.sqrt(nrm(t)**2 + h**2) - h)/nrm(t)**2
            g = lambda t: v*tmp(t)
            In1 += s3/2 * (w @ g(t))
 
    return In1
        
def compute_I0(x0h, h, eh, J, H, n, trans, tol):
    # Note: the variable v belows correponds to psi in [1].
    
    # Points and weight for 1D Gauss quadrature on [-1,1]:
    t0, w0 = np.polynomial.legendre.leggauss(n)
        
    # Jacobian and Hessian:
    J1 = J[0](x0h)
    J2 = J[1](x0h)
    H11 = H[0](x0h)
    H12 = H[1](x0h)
    H22 = H[2](x0h)
    
    # Compute v and its first derivatives:
    P = np.cross(J1, J2)
    v = norm(P)
    dv1 = 1/v * (P @ (np.cross(H11, J2) + np.cross(J1, H12)))
    dv2 = 1/v * (P @ (np.cross(H12, J2) + np.cross(J1, H22)))

    # Compute the coefficients:
    a1 = eh @ H11
    a2 = 2 * (eh @ H12)
    a3 = eh @ H22
    A = np.array([a1, a2, a3])
    c1 = J1 @ H11
    c2 = 2 * (J1 @ H12) + J2 @ H11
    c3 = 2 * (J2 @ H12) + J1 @ H22
    c4 = J2 @ H22
    C = np.array([c1, c2, c3, c4])
    
    # Integrals on each edge:
    s1 = x0h[1]
    s2 = np.sqrt(2)/2*(1 - x0h[0] - x0h[1])
    s3 = x0h[0]
    I0 = 0
    if (abs(s1) > tol):
        
        # Transplanted Gauss quadrature if near-vertex:
        if (s1 <= 1e-2 and trans[1] == True):
            t, scl = conf_func(t0, -1 + 2*x0h[0], 2*s1)
            w = w0 * scl
        else:
            t = t0
            w = w0
 
        # Paramatretization and norm:
        r1 = lambda t: -x0h[0] + (t + 1)/2 
        r2 = lambda t: -x0h[1]
        nrm0 = lambda t: (J1[0]*r1(t) + J2[0]*r2(t))**2
        nrm1 = lambda t: (J1[1]*r1(t) + J2[1]*r2(t))**2
        nrm2 = lambda t: (J1[2]*r1(t) + J2[2]*r2(t))**2
        nrm = lambda t: np.sqrt(nrm0(t) + nrm1(t) + nrm2(t))
        
        # Term with v':
        tmp0 = lambda t: nrm(t)*np.sqrt(nrm(t)**2 + h**2) 
        tmp1 = lambda t: -h**2*np.arcsinh(nrm(t)/h) if h > 0 else 0
        tmp2 = lambda t: 2*nrm(t)**3
        tmp = lambda t: (tmp0(t) + tmp1(t))/tmp2(t)
        dv = lambda t: dv1*r1(t) + dv2*r2(t)
        g = lambda t: dv(t)*tmp(t)
        I0 += s1/2 * (w @ g(t))
 
        # Terms with a's:
        tmp0 = lambda t: nrm(t)**2 
        tmp1 = lambda t: 2*h*(h - np.sqrt(nrm(t)**2 + h**2))
        tmp2 = lambda t: nrm(t)**4*np.sqrt(nrm(t)**2 + h**2)
        tmp = lambda t: (tmp0(t) + tmp1(t))/tmp2(t)
        for k in range(3):
            g = lambda t: r1(t)**(3-k-1)*r2(t)**k*tmp(t)
            I0 += (-h*v/2*A[k]) * s1/2 * (w @ g(t))
            
        # Terms with c's:
        tmp0 = lambda t: (3*h**2*nrm(t) + nrm(t)**3)/np.sqrt(nrm(t)**2 + h**2)
        tmp1 = lambda t: -3*h**2*np.arcsinh(nrm(t)/h) if h > 0 else 0
        tmp2 = lambda t: 2*nrm(t)**5
        tmp = lambda t: (tmp0(t) + tmp1(t))/tmp2(t)
        for k in range(4):
            g = lambda t: r1(t)**(4-k-1)*r2(t)**k*tmp(t)
            I0 += (-v/2*C[k]) * s1/2 * (w @ g(t))
        
    if (abs(s2) > tol):
        
        # Transplanted Gauss quadrature if near-vertex:
        if (s2 <= 1e-2 and trans[1] == True):
            t, scl = conf_func(t0, -1 + 2*x0h[1], 2*s2)
            w = w0 * scl
        else:
            t = t0
            w = w0
            
        # Paramatretization and norm:
        r1 = lambda t: 1 - x0h[0] - (t + 1)/2
        r2 = lambda t: -x0h[1] + (t + 1)/2
        nrm0 = lambda t: (J1[0]*r1(t) + J2[0]*r2(t))**2
        nrm1 = lambda t: (J1[1]*r1(t) + J2[1]*r2(t))**2
        nrm2 = lambda t: (J1[2]*r1(t) + J2[2]*r2(t))**2
        nrm = lambda t: np.sqrt(nrm0(t) + nrm1(t) + nrm2(t))
        
        # Term with v':
        tmp0 = lambda t: nrm(t)*np.sqrt(nrm(t)**2 + h**2) 
        tmp1 = lambda t: -h**2*np.arcsinh(nrm(t)/h) if h > 0 else 0
        tmp2 = lambda t: 2*nrm(t)**3
        tmp = lambda t: (tmp0(t) + tmp1(t))/tmp2(t)
        dv = lambda t: dv1*r1(t) + dv2*r2(t)
        g = lambda t: dv(t)*tmp(t)
        I0 += np.sqrt(2)*s2/2 * (w @ g(t))
        
        # Terms with a's:
        tmp0 = lambda t: nrm(t)**2 
        tmp1 = lambda t: 2*h*(h - np.sqrt(nrm(t)**2 + h**2))
        tmp2 = lambda t: nrm(t)**4*np.sqrt(nrm(t)**2 + h**2)
        tmp = lambda t: (tmp0(t) + tmp1(t))/tmp2(t)
        for k in range(3):
            g = lambda t: r1(t)**(3-k-1)*r2(t)**k*tmp(t)
            I0 += (-h*v/2*A[k]) * np.sqrt(2)*s2/2 * (w @ g(t))
            
        # Terms with c's:
        tmp0 = lambda t: (3*h**2*nrm(t) + nrm(t)**3)/np.sqrt(nrm(t)**2 + h**2)
        tmp1 = lambda t: -3*h**2*np.arcsinh(nrm(t)/h) if h > 0 else 0
        tmp2 = lambda t: 2*nrm(t)**5
        tmp = lambda t: (tmp0(t) + tmp1(t))/tmp2(t)
        for k in range(4):
            g = lambda t: r1(t)**(4-k-1)*r2(t)**k*tmp(t)
            I0 += (-v/2*C[k]) * np.sqrt(2)*s2/2 * (w @ g(t))
            
    if (abs(s3) > tol):
        
        # Transplanted Gauss quadrature if near-vertex:
        if (s3 <= 1e-2 and trans[1] == True):
            t, scl = conf_func(t0, -1 + 2*(1 - x0h[1]), 2*s3)
            w = w0 * scl
        else:
            t = t0
            w = w0
        
        # Paramatretization and norm:
        r1 = lambda t: -x0h[0]
        r2 = lambda t: 1 - x0h[1] - (1 + t)/2
        nrm0 = lambda t: (J1[0]*r1(t) + J2[0]*r2(t))**2
        nrm1 = lambda t: (J1[1]*r1(t) + J2[1]*r2(t))**2
        nrm2 = lambda t: (J1[2]*r1(t) + J2[2]*r2(t))**2
        nrm = lambda t: np.sqrt(nrm0(t) + nrm1(t) + nrm2(t))
        
        # Term with v':
        tmp0 = lambda t: nrm(t)*np.sqrt(nrm(t)**2 + h**2) 
        tmp1 = lambda t: -h**2*np.arcsinh(nrm(t)/h) if h > 0 else 0
        tmp2 = lambda t: 2*nrm(t)**3
        tmp = lambda t: (tmp0(t) + tmp1(t))/tmp2(t)
        dv = lambda t: dv1*r1(t) + dv2*r2(t)
        g = lambda t: dv(t)*tmp(t)
        I0 += s3/2 * (w @ g(t))
        
        # Terms with a's:
        tmp0 = lambda t: nrm(t)**2 
        tmp1 = lambda t: 2*h*(h - np.sqrt(nrm(t)**2 + h**2))
        tmp2 = lambda t: nrm(t)**4*np.sqrt(nrm(t)**2 + h**2)
        tmp = lambda t: (tmp0(t) + tmp1(t))/tmp2(t)
        for k in range(3):
            g = lambda t: r1(t)**(3-k-1)*r2(t)**k*tmp(t)
            I0 += (-h*v/2*A[k]) * s3/2 * (w @ g(t))
            
        # Terms with c's:
        tmp0 = lambda t: (3*h**2*nrm(t) + nrm(t)**3)/np.sqrt(nrm(t)**2 + h**2)
        tmp1 = lambda t: -3*h**2*np.arcsinh(nrm(t)/h) if h > 0 else 0
        tmp2 = lambda t: 2*nrm(t)**5
        tmp = lambda t: (tmp0(t) + tmp1(t))/tmp2(t)
        for k in range(4):
            g = lambda t: r1(t)**(4-k-1)*r2(t)**k*tmp(t)
            I0 += (-v/2*C[k]) * s3/2 * (w @ g(t))
    
    return I0
    
def compute_I1(x0h, h, eh, J, H, n, trans, tol):
    # Note: the variable v belows correponds to psi in [1].
    
    # Points and weight for 1D Gauss quadrature on [-1,1]:
    t0, w0 = np.polynomial.legendre.leggauss(n)
        
    # Jacobian and Hessian:
    J1 = J[0](x0h)
    J2 = J[1](x0h)
    H11 = H[0](x0h)
    H12 = H[1](x0h)
    H22 = H[2](x0h)
    
    # Compute v and its 2nd derivatives:
    P = np.cross(J1, J2)
    dP1 = np.cross(H11, J2) + np.cross(J1, H12)
    dP2 = np.cross(H12, J2) + np.cross(J1, H22)
    dP11 = 2 * np.cross(H11, H12)
    dP12 = np.cross(H11, H22) + np.cross(H12, H12)
    dP22 = 2 * np.cross(H12, H22)
    v = norm(P)
    dv1 = 1/v * P @ dP1
    dv2 = 1/v * P @ dP2
    dv11 = 1/v**2 * (v * dP1 @ dP1 + v * P @ dP11 - dv1 * P @ dP1)
    dv12 = 1/v**2 * (v * dP1 @ dP2 + v * P @ dP12 - dv2 * P @ dP1)
    dv22 = 1/v**2 * (v * dP2 @ dP2 + v * P @ dP22 - dv2 * P @ dP2)
    
    # Compute the coefficients:
    a1 = eh @ H11
    a2 = 2 * (eh @ H12)
    a3 = eh @ H22
    c1 = J1 @ H11
    c2 = 2 * (J1 @ H12) + J2 @ H11
    c3 = 2 * (J2 @ H12) + J1 @ H22
    c4 = J2 @ H22
    d1 = 1/4 * (H11 @ H11)
    d2 = H11 @ H12
    d3 = 1/2 * (H11 @ H22) + H12 @ H12
    d4 = H22 @ H12
    d5 = 1/4 * (H22 @ H22)
    e1 = -1/2 * a1*dv1
    e2 = -1/2 * (a2*dv1 + a1*dv2)
    e3 = -1/2 * (a3*dv1 + a2*dv2)
    e4 = -1/2 * a3*dv2
    E = np.array([e1, e2, e3, e4])
    f1 = 3/8 * v * a1**2
    f2 = 3/8 * v * 2*a1*a2
    f3 = 3/8 * v * (a2**2 + 2*a1*a3)
    f4 = 3/8 * v * 2*a2*a3
    f5 = 3/8 * v * a3**2
    F = np.array([f1, f2, f3, f4, f5])
    g1 = -1/2 * (c1*dv1 + d1*v)
    g2 = -1/2 * (c2*dv1 + c1*dv2 + d2*v)
    g3 = -1/2 * (c3*dv1 + c2*dv2 + d3*v)
    g4 = -1/2 * (c4*dv1 + c3*dv2 + d4*v)
    g5 = -1/2 * (c4*dv2 + d5*v)
    G = np.array([g1, g2, g3, g4, g5])
    h1 = 3/8 * v * c1**2
    h2 = 3/8 * v * 2*c1*c2
    h3 = 3/8 * v * (c2**2 + 2*c1*c3)
    h4 = 3/8 * v * (2*c1*c4 + 2*c2*c3)
    h5 = 3/8 * v * (c3**2 + 2*c2*c4)
    h6 = 3/8 * v * 2*c3*c4
    h7 = 3/8 * v * c4**2
    H = np.array([h1, h2, h3, h4, h5, h6, h7])
    
    # Integrals on each edge:
    s1 = x0h[1]
    s2 = np.sqrt(2)/2*(1 - x0h[0] - x0h[1])
    s3 = x0h[0]
    I1 = 0
    if (abs(s1) > tol):
        
        # Transplanted Gauss quadrature if near-vertex:
        if (s1 <= 1e-2 and trans[2] == True):
            t, scl = conf_func(t0, -1 + 2*x0h[0], 2*s1)
            w = w0 * scl
        else:
            t = t0
            w = w0
            
        # Paramatretization and norm:
        r1 = lambda t: -x0h[0] + (t + 1)/2 
        r2 = lambda t: -x0h[1]
        nrm0 = lambda t: (J1[0]*r1(t) + J2[0]*r2(t))**2
        nrm1 = lambda t: (J1[1]*r1(t) + J2[1]*r2(t))**2
        nrm2 = lambda t: (J1[2]*r1(t) + J2[2]*r2(t))**2
        nrm = lambda t: np.sqrt(nrm0(t) + nrm1(t) + nrm2(t))
        
        # Term with v'':
        tmp0 = lambda t: 2*h**3 - 2*h**2*np.sqrt(nrm(t)**2 + h**2)
        tmp1 = lambda t: nrm(t)**2*np.sqrt(nrm(t)**2  + h**2)
        tmp2 = lambda t: 3*nrm(t)**4
        tmp = lambda t: (tmp0(t) + tmp1(t))/tmp2(t)
        d2v = lambda t: 1/2*(dv11*r1(t)**2 + dv22*r2(t)**2) + dv12*r1(t)*r2(t)
        g = lambda t: d2v(t)*tmp(t)
        I1 += s1/2 * (w @ g(t))
        
        # Terms with e's:
        tmp0 = lambda t: (3*h**2*nrm(t) + nrm(t)**3)/np.sqrt(nrm(t)**2 + h**2)
        tmp1 = lambda t: -3*h**2*np.arcsinh(nrm(t)/h) if h > 0 else 0
        tmp2 = lambda t: 2*nrm(t)**5
        tmp = lambda t: (tmp0(t) + tmp1(t))/tmp2(t)
        for k in range(4):
            g = lambda t: r1(t)**(4-k-1)*r2(t)**k*tmp(t)
            I1 += (h*E[k]) * s1/2 * (w @ g(t))
            
        # Terms with f's:
        tmp0 = lambda t: 8*h**4 + 12*h**2*nrm(t)**2 + 3*nrm(t)**4
        tmp1 = lambda t: -8*h*(nrm(t)**2 + h**2)**(3/2)
        tmp2 = lambda t: 3*nrm(t)**6*(nrm(t)**2 + h**2)**(3/2)
        tmp = lambda t: (tmp0(t) + tmp1(t))/tmp2(t)
        for k in range(5):
            g = lambda t: r1(t)**(5-k-1)*r2(t)**k*tmp(t)
            I1 += (h**2*F[k]) * s1/2 * (w @ g(t))
            
        # Terms with g's:
        tmp0 = lambda t: -8*h**4 - 4*h**2*nrm(t)**2 + nrm(t)**4
        tmp1 = lambda t: 8*h**3*np.sqrt(nrm(t)**2 + h**2)
        tmp2 = lambda t: 3*nrm(t)**6*np.sqrt(nrm(t)**2 + h**2)
        tmp = lambda t: (tmp0(t) + tmp1(t))/tmp2(t)
        for k in range(5):
            g = lambda t: r1(t)**(5-k-1)*r2(t)**k*tmp(t)
            I1 += G[k] * s1/2 * (w @ g(t))
            
        # Terms with h's (1):
        tmp0 = lambda t: -16*h**6 - 24*h**4*nrm(t)**2 
        tmp1 = lambda t: -6*h**2*nrm(t)**4 + nrm(t)**6 
        tmp2 = lambda t: 3*nrm(t)**8*(nrm(t)**2 + h**2)**(3/2)
        tmp = lambda t: (tmp0(t) + tmp1(t))/tmp2(t)
        for k in range(7):
            g = lambda t: r1(t)**(7-k-1)*r2(t)**k*tmp(t)
            I1 += H[k] * s1/2 * (w @ g(t))
            
        # Terms with h's (2):
        tmp0 = lambda t: 16*h**5*np.sqrt(nrm(t)**2 + h**2)
        tmp1 = lambda t: 16*h**3*nrm(t)**2*np.sqrt(nrm(t)**2 + h**2)
        tmp2 = lambda t: 3*nrm(t)**8*(nrm(t)**2 + h**2)**(3/2)
        tmp = lambda t: (tmp0(t) + tmp1(t))/tmp2(t)
        for k in range(7):
            g = lambda t: r1(t)**(7-k-1)*r2(t)**k*tmp(t)
            I1 += H[k] * s1/2 * (w @ g(t))
            
    if (abs(s2) > tol):
        
        # Transplanted Gauss quadrature if near-vertex:
        if (s2 <= 1e-2 and trans[2] == True):
            t, scl = conf_func(t0, -1 + 2*x0h[1], 2*s2)
            w = w0 * scl
        else:
            t = t0
            w = w0
            
        # Paramatretization and norm:
        r1 = lambda t: 1 - x0h[0] - (t + 1)/2
        r2 = lambda t: -x0h[1] + (t + 1)/2
        nrm0 = lambda t: (J1[0]*r1(t) + J2[0]*r2(t))**2
        nrm1 = lambda t: (J1[1]*r1(t) + J2[1]*r2(t))**2
        nrm2 = lambda t: (J1[2]*r1(t) + J2[2]*r2(t))**2
        nrm = lambda t: np.sqrt(nrm0(t) + nrm1(t) + nrm2(t))
        
        # Term with v'':
        tmp0 = lambda t: 2*h**3 - 2*h**2*np.sqrt(nrm(t)**2 + h**2)
        tmp1 = lambda t: nrm(t)**2*np.sqrt(nrm(t)**2  + h**2)
        tmp2 = lambda t: 3*nrm(t)**4
        tmp = lambda t: (tmp0(t) + tmp1(t))/tmp2(t)
        d2v = lambda t: 1/2*(dv11*r1(t)**2 + dv22*r2(t)**2) + dv12*r1(t)*r2(t)
        g = lambda t: d2v(t)*tmp(t)
        I1 += np.sqrt(2)*s2/2 * (w @ g(t))
        
        # Terms with e's:
        tmp0 = lambda t: (3*h**2*nrm(t) + nrm(t)**3)/np.sqrt(nrm(t)**2 + h**2)
        tmp1 = lambda t: -3*h**2*np.arcsinh(nrm(t)/h) if h > 0 else 0
        tmp2 = lambda t: 2*nrm(t)**5
        tmp = lambda t: (tmp0(t) + tmp1(t))/tmp2(t)
        for k in range(4):
            g = lambda t: r1(t)**(4-k-1)*r2(t)**k*tmp(t)
            I1 += (h*E[k]) * np.sqrt(2)*s2/2 * (w @ g(t))
            
        # Terms with f's:
        tmp0 = lambda t: 8*h**4 + 12*h**2*nrm(t)**2 + 3*nrm(t)**4
        tmp1 = lambda t: -8*h*(nrm(t)**2 + h**2)**(3/2)
        tmp2 = lambda t: 3*nrm(t)**6*(nrm(t)**2 + h**2)**(3/2)
        tmp = lambda t: (tmp0(t) + tmp1(t))/tmp2(t)
        for k in range(5):
            g = lambda t: r1(t)**(5-k-1)*r2(t)**k*tmp(t)
            I1 += (h**2*F[k]) * np.sqrt(2)*s2/2 * (w @ g(t))
            
        # Terms with g's:
        tmp0 = lambda t: -8*h**4 - 4*h**2*nrm(t)**2 + nrm(t)**4
        tmp1 = lambda t: 8*h**3*np.sqrt(nrm(t)**2 + h**2)
        tmp2 = lambda t: 3*nrm(t)**6*np.sqrt(nrm(t)**2 + h**2)
        tmp = lambda t: (tmp0(t) + tmp1(t))/tmp2(t)
        for k in range(5):
            g = lambda t: r1(t)**(5-k-1)*r2(t)**k*tmp(t)
            I1 += G[k] * np.sqrt(2)*s2/2 * (w @ g(t))
            
        # Terms with h's (1):
        tmp0 = lambda t: -16*h**6 - 24*h**4*nrm(t)**2 
        tmp1 = lambda t: -6*h**2*nrm(t)**4 + nrm(t)**6 
        tmp2 = lambda t: 3*nrm(t)**8*(nrm(t)**2 + h**2)**(3/2)
        tmp = lambda t: (tmp0(t) + tmp1(t))/tmp2(t)
        for k in range(7):
            g = lambda t: r1(t)**(7-k-1)*r2(t)**k*tmp(t)
            I1 += H[k] * np.sqrt(2)*s2/2 * (w @ g(t))
            
        # Terms with h's (2):
        tmp0 = lambda t: 16*h**5*np.sqrt(nrm(t)**2 + h**2)
        tmp1 = lambda t: 16*h**3*nrm(t)**2*np.sqrt(nrm(t)**2 + h**2)
        tmp2 = lambda t: 3*nrm(t)**8*(nrm(t)**2 + h**2)**(3/2)
        tmp = lambda t: (tmp0(t) + tmp1(t))/tmp2(t)
        for k in range(7):
            g = lambda t: r1(t)**(7-k-1)*r2(t)**k*tmp(t)
            I1 += H[k] * np.sqrt(2)*s2/2 * (w @ g(t))
            
    if (abs(s3) > tol):
        
        # Transplanted Gauss quadrature if near-vertex:
        if (s3 <= 1e-2 and trans[2] == True):
            t, scl = conf_func(t0, -1 + 2*(1 - x0h[1]), 2*s3)
            w = w0 * scl
        else:
            t = t0
            w = w0
        
        # Paramatretization and norm:
        r1 = lambda t: -x0h[0]
        r2 = lambda t: 1 - x0h[1] - (1 + t)/2
        nrm0 = lambda t: (J1[0]*r1(t) + J2[0]*r2(t))**2
        nrm1 = lambda t: (J1[1]*r1(t) + J2[1]*r2(t))**2
        nrm2 = lambda t: (J1[2]*r1(t) + J2[2]*r2(t))**2
        nrm = lambda t: np.sqrt(nrm0(t) + nrm1(t) + nrm2(t))
        
        # Term with v'':
        tmp0 = lambda t: 2*h**3 - 2*h**2*np.sqrt(nrm(t)**2 + h**2)
        tmp1 = lambda t: nrm(t)**2*np.sqrt(nrm(t)**2  + h**2)
        tmp2 = lambda t: 3*nrm(t)**4
        tmp = lambda t: (tmp0(t) + tmp1(t))/tmp2(t)
        d2v = lambda t: 1/2*(dv11*r1(t)**2 + dv22*r2(t)**2) + dv12*r1(t)*r2(t)
        g = lambda t: d2v(t)*tmp(t)
        I1 += s3/2 * (w @ g(t))
        
        # Terms with e's:
        tmp0 = lambda t: (3*h**2*nrm(t) + nrm(t)**3)/np.sqrt(nrm(t)**2 + h**2)
        tmp1 = lambda t: -3*h**2*np.arcsinh(nrm(t)/h) if h > 0 else 0
        tmp2 = lambda t: 2*nrm(t)**5
        tmp = lambda t: (tmp0(t) + tmp1(t))/tmp2(t)
        for k in range(4):
            g = lambda t: r1(t)**(4-k-1)*r2(t)**k*tmp(t)
            I1 += (h*E[k]) * s3/2 * (w @ g(t))
            
        # Terms with f's:
        tmp0 = lambda t: 8*h**4 + 12*h**2*nrm(t)**2 + 3*nrm(t)**4
        tmp1 = lambda t: -8*h*(nrm(t)**2 + h**2)**(3/2)
        tmp2 = lambda t: 3*nrm(t)**6*(nrm(t)**2 + h**2)**(3/2)
        tmp = lambda t: (tmp0(t) + tmp1(t))/tmp2(t)
        for k in range(5):
            g = lambda t: r1(t)**(5-k-1)*r2(t)**k*tmp(t)
            I1 += (h**2*F[k]) * s3/2 * (w @ g(t))
            
        # Terms with g's:
        tmp0 = lambda t: -8*h**4 - 4*h**2*nrm(t)**2 + nrm(t)**4
        tmp1 = lambda t: 8*h**3*np.sqrt(nrm(t)**2 + h**2)
        tmp2 = lambda t: 3*nrm(t)**6*np.sqrt(nrm(t)**2 + h**2)
        tmp = lambda t: (tmp0(t) + tmp1(t))/tmp2(t)
        for k in range(5):
            g = lambda t: r1(t)**(5-k-1)*r2(t)**k*tmp(t)
            I1 += G[k] * s3/2 * (w @ g(t))
            
        # Terms with h's (1):
        tmp0 = lambda t: -16*h**6 - 24*h**4*nrm(t)**2 
        tmp1 = lambda t: -6*h**2*nrm(t)**4 + nrm(t)**6 
        tmp2 = lambda t: 3*nrm(t)**8*(nrm(t)**2 + h**2)**(3/2)
        tmp = lambda t: (tmp0(t) + tmp1(t))/tmp2(t)
        for k in range(7):
            g = lambda t: r1(t)**(7-k-1)*r2(t)**k*tmp(t)
            I1 += H[k] * s3/2 * (w @ g(t))
            
        # Terms with h's (2):
        tmp0 = lambda t: 16*h**5*np.sqrt(nrm(t)**2 + h**2)
        tmp1 = lambda t: 16*h**3*nrm(t)**2*np.sqrt(nrm(t)**2 + h**2)
        tmp2 = lambda t: 3*nrm(t)**8*(nrm(t)**2 + h**2)**(3/2)
        tmp = lambda t: (tmp0(t) + tmp1(t))/tmp2(t)
        for k in range(7):
            g = lambda t: r1(t)**(7-k-1)*r2(t)**k*tmp(t)
            I1 += H[k] * s3/2 * (w @ g(t))
         
    return I1
        
def sign_func(x):
    if (x >= 0): return 1 
    else: return -1
    
def conf_func(z, mu, nu):
    a = np.arcsinh((1 - mu)/nu)
    b = np.arcsinh((1 + mu)/nu)
    g = mu + nu * np.sinh((a + b)*(z - 1)/2 + a)
    dg = nu * (a + b)/2 * np.cosh((a + b)*(z - 1)/2 + a)
    return g, dg
    
def mul_func(J, X, n):
    M = np.zeros([3, n, n])
    for i in range(n):
        for j in range(n):
            M[:, i, j] = J @ np.array([X[0][i,j], X[1][i,j]])
    return M
        
def map_func(A):    
    U = basis_func()
    Fx = lambda x: sum(U[k](x)*A[k,0] for k in range(6))
    Fy = lambda x: sum(U[k](x)*A[k,1] for k in range(6))
    Fz = lambda x: sum(U[k](x)*A[k,2] for k in range(6))
    F = lambda x: np.array([Fx(x), Fy(x), Fz(x)])
    return F

def map_jac(A):
    G = basis_grad()
    J1x = lambda x: sum(G[k](x)[0]*A[k,0] for k in range(6))
    J1y = lambda x: sum(G[k](x)[0]*A[k,1] for k in range(6))
    J1z = lambda x: sum(G[k](x)[0]*A[k,2] for k in range(6))
    J1 = lambda x: np.array([J1x(x), J1y(x), J1z(x)])
    J2x = lambda x: sum(G[k](x)[1]*A[k,0] for k in range(6))
    J2y = lambda x: sum(G[k](x)[1]*A[k,1] for k in range(6))
    J2z = lambda x: sum(G[k](x)[1]*A[k,2] for k in range(6))
    J2 = lambda x: np.array([J2x(x), J2y(x), J2z(x)])
    return J1, J2

def map_hess(A):
    H = basis_hess()
    H11x = lambda x: sum(H[k](x)[0,0]*A[k,0] for k in range(6))
    H11y = lambda x: sum(H[k](x)[0,0]*A[k,1] for k in range(6))
    H11z = lambda x: sum(H[k](x)[0,0]*A[k,2] for k in range(6))
    H11 = lambda x: np.array([H11x(x), H11y(x), H11z(x)])
    H12x = lambda x: sum(H[k](x)[0,1]*A[k,0] for k in range(6))
    H12y = lambda x: sum(H[k](x)[0,1]*A[k,1] for k in range(6))
    H12z = lambda x: sum(H[k](x)[0,1]*A[k,2] for k in range(6))
    H12 = lambda x: np.array([H12x(x), H12y(x), H12z(x)])
    H22x = lambda x: sum(H[k](x)[1,1]*A[k,0] for k in range(6))
    H22y = lambda x: sum(H[k](x)[1,1]*A[k,1] for k in range(6))
    H22z = lambda x: sum(H[k](x)[1,1]*A[k,2] for k in range(6))
    H22 = lambda x: np.array([H22x(x), H22y(x), H22z(x)])
    return H11, H12, H22
    
def basis_func():
    l1 = lambda x: 1 - x[0] - x[1]
    l2 = lambda x: x[0]
    l3 = lambda x: x[1]
    u1 = lambda x: l1(x)*(2*l1(x) - 1)
    u2 = lambda x: l2(x)*(2*l2(x) - 1)
    u3 = lambda x: l3(x)*(2*l3(x) - 1)
    u4 = lambda x: 4*l1(x)*l2(x)
    u5 = lambda x: 4*l2(x)*l3(x)
    u6 = lambda x: 4*l1(x)*l3(x)
    return u1, u2, u3, u4, u5, u6

def basis_grad():
    G1 = lambda x: np.array([4*x[0] + 4*x[1] - 3, 4*x[0] + 4*x[1] - 3])
    G2 = lambda x: np.array([4*x[0] - 1, 0])
    G3 = lambda x: np.array([0, 4*x[1] - 1])
    G4 = lambda x: np.array([-8*x[0] - 4*x[1] + 4, -4*x[0]])
    G5 = lambda x: np.array([4*x[1], 4*x[0]])
    G6 = lambda x: np.array([-4*x[1], -4*x[0] - 8*x[1] + 4])
    return G1, G2, G3, G4, G5, G6

def basis_hess():
    H1 = lambda x: np.array([[4, 4], [4, 4]])
    H2 = lambda x: np.array([[4, 0], [0, 0]])
    H3 = lambda x: np.array([[0, 0], [0, 4]])
    H4 = lambda x: np.array([[-8, -4], [-4, 0]])
    H5 = lambda x: np.array([[0, 4], [4, 0]])
    H6 = lambda x: np.array([[0, -4], [-4, -8]])
    return H1, H2, H3, H4, H5, H6