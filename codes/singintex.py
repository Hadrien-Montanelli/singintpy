#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 19:24:38 2021

Copyright 2021 by Hadrien Montanelli.
"""
def singintex(u0, v0, dz0):
    """
    # Outputs the exact value (computed in Mathematica) of the singular and
    # near-singular integrals used in the numerical experiments of [1, Sec. 4].
    # These values are computed in the singintex.nb file.
        
    # Inputs
    # ------
    # u0, v0, dz0 : float
    #     Position of the singularity is x0 = F(u0, v0) + dz0*z.

    # Output
    # ------
    # Iex : float
    #     The exact value of the integral.
        
    # References
    # ----------
    # [1] H. Montanelli, M. Aussal and H. Haddar, Computing weakly singular and 
    # near-singular integrals in high-order boundary elements, submitted.
    # """
    # Point near the center:
    if (u0 == .2) and (v0 == .4) and (dz0 == 0): # singularity 
        Iex = 3.240017458404107
    if (u0 == .2) and (v0 == .4) and (dz0 == 1e-4): # near-singularity 
        Iex = 3.239493851850319
    if (u0 == .2) and (v0 == .4) and (dz0 == 1e-3): # near-singularity 
        Iex = 3.234785969247374
    if (u0 == .2) and (v0 == .4) and (dz0 == 1e-2): # near-singularity 
        Iex = 3.188154928666069
    if (u0 == .2) and (v0 == .4) and (dz0 == 1e-1): # near-singularity 
        Iex = 2.762090628529887
    if (u0 == .2) and (v0 == .4) and (dz0 == 1e0): # near-singularity 
        Iex = 0.8597262637971332
    if (u0 == .2) and (v0 == .4) and (-dz0 == 1e-4): # near-singularity 
        Iex = 3.239500821128147
                        
    # Point near the a1-a2 vertex:
    if (u0 == .5) and (v0 == 1e-1) and (dz0 == 0): # singularity 
        Iex = 3.018547440468339
    if (u0 == .5) and (v0 == 1e-1) and (dz0 == 1e-4): # near-singularity 
        Iex = 3.018116377195088
    if (u0 == .5) and (v0 == 1e-1) and (dz0 == 1e-3): # near-singularity 
        Iex = 3.014240460722516
    if (u0 == .5) and (v0 == 1e-2) and (dz0 == 0): # singularity 
        Iex = 2.44181568875291
    if (u0 == .5) and (v0 == 1e-2) and (dz0 == 1e-4): # near-singularity 
        Iex = 2.441683569912414
    if (u0 == .5) and (v0 == 1e-3) and (dz0 == 0): # singularity 
        Iex = 2.310786384193376
    if (u0 == .5) and (v0 == 1e-3) and (dz0 == 1e-4): # near-singularity 
        Iex = 2.310927133672147
    if (u0 == .5) and (v0 == 1e-4) and (dz0 == 0): # singularity 
        Iex = 2.290532510026764
    if (u0 == .5) and (v0 == 1e-4) and (dz0 == 1e-4): # near-singularity 
        Iex = 2.290950009889399
    if (u0 == .5) and (v0 == 1e-4) and (dz0 == 1e-3): # near-singularity 
        Iex = 2.294299358958351
    if (u0 == .5) and (v0 == 1e-4) and (dz0 == 1e-2): # near-singularity 
        Iex = 2.309420005131348
    if (u0 == .5) and (v0 == 1e-4) and (dz0 == 1e0): # near-singularity 
        Iex = 0.9161025842305255
    if (u0 == .5) and (v0 == 1e-5) and (dz0 == 0): # singularity 
        Iex = 2.287793848213499
    if (u0 == .5) and (v0 == 1e-5) and (dz0 == 1e-4): # near-singularity 
        Iex = 2.288438219470312
    if (u0 == .5) and (v0 == 1e-6) and (dz0 == 0): # singularity 
        Iex = 2.287448672711318
    if (u0 == .5) and (v0 == 1e-6) and (dz0 == 1e-4): # near-singularity 
        Iex = 2.288172343092934
    if (u0 == .5) and (v0 == 1e-7) and (dz0 == 0): # singularity 
        Iex = 2.287407024410108
    if (u0 == .5) and (v0 == 1e-7) and (dz0 == 1e-4): # near-singularity 
        Iex = 2.288145597976325
    if (u0 == .5) and (v0 == 1e-10) and (dz0 == 0): # singularity 
        Iex = 2.287401524368497
    if (u0 == .5) and (v0 == 1e-10) and (dz0 == 1e-4): # near-singularity 
        Iex = 2.288142627543715
    if (u0 == .5) and (v0 == 0) and (dz0 == 0): # singularity 
        Iex = 2.287401516483698
    if (u0 == .5) and (v0 == 0) and (dz0 == 1e-4): # near-singularity 
        Iex = 2.288142624570126
    if (u0 == .5) and (v0 == 0) and (dz0 == 1e-2): # near-singularity 
        Iex = 2.308036432953315
        
    # Point near the a2-a3 vertex:
    if (u0 == .5) and (v0 == 0.5-1e-1) and (dz0 == 0): # singularity 
        Iex = 2.910980479568196
    if (u0 == .5) and (v0 == 0.5-1e-2) and (dz0 == 0): # singularity 
        Iex = 2.328357836622938
    if (u0 == .5) and (v0 == 0.5-1e-3) and (dz0 == 0): # singularity 
        Iex = 2.207283607389093
    if (u0 == .5) and (v0 == 0.5-1e-4) and (dz0 == 0): # singularity 
        Iex = 2.18893530823417
    if (u0 == .5) and (v0 == 0.5-1e-5) and (dz0 == 0): # singularity 
        Iex = 2.186474998717380
    if (u0 == .5) and (v0 == 0.5-1e-6) and (dz0 == 0): # singularity 
        Iex = 2.186166390068990
    if (u0 == .5) and (v0 == 0.5-1e-7) and (dz0 == 0): # singularity 
        Iex = 2.186129270984043
    if (u0 == .5) and (v0 == 0.5-1e-10) and (dz0 == 0): # singularity 
        Iex = 2.186124380996982
    if (u0 == .5) and (v0 == 0.5) and (dz0 == 0): # singularity 
        Iex = 2.186124374013931
        
    # Point near the a3-a1 vertex:
    if (u0 == 1e-1) and (v0 == 0.5) and (dz0 == 0): # singularity 
        Iex = 3.001704553910470
    if (u0 == 1e-2) and (v0 == 0.5) and (dz0 == 0): # singularity 
        Iex = 2.455261667489161
    if (u0 == 1e-3) and (v0 == 0.5) and (dz0 == 0): # singularity 
        Iex = 2.333635120399480
    if (u0 == 1e-4) and (v0 == 0.5) and (dz0 == 0): # singularity 
        Iex = 2.314977231319161
    if (u0 == 1e-5) and (v0 == 0.5) and (dz0 == 0): # singularity 
        Iex = 2.312463818083347
    if (u0 == 1e-6) and (v0 == 0.5) and (dz0 == 0): # singularity 
        Iex = 2.312147733122362
    if (u0 == 1e-7) and (v0 == 0.5) and (dz0 == 0): # singularity 
        Iex = 2.31210965045024
    if (u0 == 1e-10) and (v0 == 0.5) and (dz0 == 0): # singularity 
        Iex = 2.312104626952236
    if (u0 == 0) and (v0 == 0.5) and (dz0 == 0): # singularity 
        Iex = 2.312104619763527
        
    return Iex