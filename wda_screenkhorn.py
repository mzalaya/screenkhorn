#!/usr/bin/env python
# coding: utf-8

__author__ = 'Mokhtar Z. Alaya'

"""
Dimension reduction with Screened optimal transport.
The script is adapted from ot/dr.py in the POT toolbox.

"""
from time import process_time as time

from scipy import linalg
import autograd.numpy as np
from pymanopt.manifolds import Stiefel
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent, TrustRegions

import ot 
from screenkhorn import Screenkhorn

def wda_screenkhorn(X, y, p=2, reg=1, k=10, solver=None, maxiter=1000, verbose=1, P0=None, **kwargs):
    # noqa
    mx = np.mean(X)
    X -= mx.reshape((1, -1))

    # data split between classes
    d = X.shape[1]
    xc = ot.dr.split_classes(X, y)
    # compute uniform weighs
    wc = [np.ones((x.shape[0]), dtype=np.float64) / x.shape[0] for x in xc]

    def cost(P):
        # wda loss
        loss_b = 0
        loss_w = 0

        for i, xi in enumerate(xc):
            xi = np.dot(xi, P)
            for j, xj in enumerate(xc[i:]):
                xj = np.dot(xj, P)
                M = ot.dr.dist(xi, xj)
                M = M / M.max()
                # screenkhorn
                p_n = kwargs.get('p_n', 2) # keep only 50% of points
                p_m = kwargs.get('p_m', 2) # keep only 50% of points
                n_budget = int(np.ceil(M.shape[0] / p_n))
                m_budget = int(np.ceil(M.shape[1] / p_m))
                screenkhornWDA = Screenkhorn(wc[i], wc[j + i], M, reg, n_budget, m_budget, verbose=False)
                G = screenkhornWDA.lbfgsb()
                if j == 0:
                    loss_w += np.sum(G * M)
                else:
                    loss_b += np.sum(G * M)

        # loss inversed because minimization
        return loss_w / loss_b

    # declare manifold and problem
    manifold = Stiefel(d, p)
    problem = Problem(manifold=manifold, cost=cost)

    # declare solver and solve
    if solver is None:
        solver = SteepestDescent(maxiter=maxiter, logverbosity=verbose, maxtime=float('inf'),mingradnorm=1e-8, 
                        minstepsize=1e-16)
    elif solver in ['tr', 'TrustRegions']:
        solver = TrustRegions(maxiter=maxiter, logverbosity=verbose)

    Popt = solver.solve(problem, x=P0)
    def proj(X):
        return (X - mx.reshape((1, -1))).dot(Popt[0])

    return Popt, proj