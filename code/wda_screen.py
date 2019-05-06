# -*- coding: utf-8 -*-
"""
Dimension reduction with optimal transport
"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#
# License: MIT License

from scipy import linalg
# import numpy as np
import autograd.numpy as np
from pymanopt.manifolds import Stiefel
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent, TrustRegions

from new_formulation import Screenkhorn

def dist(x1, x2):
    """ Compute squared euclidean distance between samples (autograd)
    """
    x1p2 = np.sum(np.square(x1), 1)
    x2p2 = np.sum(np.square(x2), 1)
    return x1p2.reshape((-1, 1)) + x2p2.reshape((1, -1)) - 2 * np.dot(x1, x2.T)


def Sinkhorn(w1, w2, M, reg, k):
    """Sinkhorn algorithm with fixed number of iteration (autograd)
    """
    K = np.exp(-M / reg)
    ui = np.ones((M.shape[0],))
    vi = np.ones((M.shape[1],))
    for i in range(k):
        vi = w2 / (np.dot(K.T, ui))
        ui = w1 / (np.dot(K, vi))
    G = ui.reshape((M.shape[0], 1)) * K * vi.reshape((1, M.shape[1]))
    return G


def split_classes(X, y):
    """split samples in X by classes in y
    """
    lstsclass = np.unique(y)
    return [X[y == i, :].astype(np.float64) for i in lstsclass]


def fda(X, y, p=2, reg=1e-16):
    mx = np.mean(X)
    X -= mx.reshape((1, -1))

    # data split between classes
    d = X.shape[1]
    xc = split_classes(X, y)
    nc = len(xc)

    p = min(nc - 1, p)

    Cw = 0
    for x in xc:
        Cw += np.cov(x, rowvar=False)
    Cw /= nc

    mxc = np.zeros((d, nc))

    for i in range(nc):
        mxc[:, i] = np.mean(xc[i])

    mx0 = np.mean(mxc, 1)
    Cb = 0
    for i in range(nc):
        Cb += (mxc[:, i] - mx0).reshape((-1, 1)) * \
              (mxc[:, i] - mx0).reshape((1, -1))

    w, V = linalg.eig(Cb, Cw + reg * np.eye(d))

    idx = np.argsort(w.real)

    Popt = V[:, idx[-p:]]

    def proj(X):
        return (X - mx.reshape((1, -1))).dot(Popt)

    return Popt, proj


def wda_sinkhorn(X, y, p=2, reg=1, k=10, solver=None, maxiter=100, verbose=0, P0=None):
    mx = np.mean(X)
    X -= mx.reshape((1, -1))

    # data split between classes
    d = X.shape[1]
    xc = split_classes(X, y)
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
                M = dist(xi, xj)
                M = M/M.max()
                G = Sinkhorn(wc[i], wc[j + i], M, reg, k)
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
        solver = SteepestDescent(maxiter=maxiter, logverbosity=verbose)
    elif solver in ['tr', 'TrustRegions']:
        solver = TrustRegions(maxiter=maxiter, logverbosity=verbose)

    Popt = solver.solve(problem, x=P0)

    def proj(X):
        return (X - mx.reshape((1, -1))).dot(Popt)

    return Popt, proj


def wda_screenkhorn(X, y, p=2, reg=1, k=10, solver=None, maxiter=100, verbose=1, P0=None, **kwargs):
    # noqa
    mx = np.mean(X)
    X -= mx.reshape((1, -1))

    # data split between classes
    d = X.shape[1]
    xc = split_classes(X, y)
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
                M = dist(xi, xj)
                M = M/M.max()
                # screenkhorn
                p_n = kwargs.get('p_n', 2) # keep only 50% of points
                p_m = kwargs.get('p_m', 2) # keep only 50% of points
                n_budget = int(np.ceil(M.shape[0] / p_n))
                m_budget = int(np.ceil(M.shape[1] / p_m))
                screenkhornWDA = Screenkhorn(wc[i], wc[j + i], M, reg, n_budget, m_budget, verbose=False)
                G = screenkhornWDA.lbfgsb()[2]
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
        solver = SteepestDescent(maxiter=maxiter, logverbosity=verbose)
    elif solver in ['tr', 'TrustRegions']:
        solver = TrustRegions(maxiter=maxiter, logverbosity=verbose)

    Popt = solver.solve(problem, x=P0)

    def proj(X):
        # print("HERE")
        # print(Popt)
        # print(Popt.shape)
        # import time
        # time.sleep(6)
        return (X - mx.reshape((1, -1))).dot(Popt[0])

    return Popt, proj