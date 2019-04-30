# -*- coding: utf-8 -*-
"""
    screenkhorn: solver for screening Sinkhorn via dual projection

"""
__author__ = 'Mokhtar Z. Alaya'

import numpy as np

np.random.seed(3946)
from scipy.optimize import fmin_l_bfgs_b
from time import time


class Screenkhorn:

    def __init__(self, a, b, K, reg, N, M):

        tic_initial = time()
        self.a = np.asarray(a, dtype=np.float64)
        self.b = np.asarray(b, dtype=np.float64)
        self.K = np.asarray(K, dtype=np.float64)
        self.reg = reg
        n = K.shape[0]
        m = K.shape[1]
        self.N = N
        self.M = M

        # Gibbs Kernel K
        self.K = np.exp(-self.K / reg)
        self.C = self.K

        # Full I = {1, ..., n} and J ={1, ..., m}, i.e. Sinkhorn problem
        if self.N == n and self.M == m:

            # I, J
            self.I = list(range(n))
            self.J = list(range(m))

            # epsilon
            self.epsilon = 0.0
            self.fact_scale = 1.0

            # Restricted Sinkhron
            self.K_IJ_p = (1 / self.a).reshape(-1, 1) * self.K
            self.cst_u = 0.
            self.cst_v = 0.

            # box BFGS
            self.bounds_u = [(0.0, np.inf)] * n
            self.bounds_v = [(0.0, np.inf)] * m

        else:

            # Sum of rows and columns of K
            K_sum_cols = self.K.sum(axis=1)
            K_sum_rows = self.K.T.sum(axis=1)

            # Minimum of K
            K_min = self.K.min()

            # Ordered marginals
            a_sort = np.sort(a)[::-1]
            b_sort = np.sort(b)[::-1]

            aK_sort = np.sort(a / K_sum_cols)[::-1]
            bK_sort = np.sort(b / K_sum_rows)[::-1]

            # epsilon_u and epsilon_v
            epsilon_u = aK_sort[self.N - 1: self.N].mean()
            epsilon_v = bK_sort[self.M - 1: self.M].mean()

            print(epsilon_u, epsilon_v)

            # self.epsilon = np.sqrt((np.sqrt(epsilon_u * epsilon_v)))
            self.epsilon = (epsilon_u * epsilon_v) ** (1 / 4)
            self.fact_scale = np.sqrt(epsilon_v / epsilon_u)

            # I, J
            self.I = np.where(self.a >= (self.epsilon ** 2 / self.fact_scale) * K_sum_cols)[0].tolist()
            self.J = np.where(self.b >= (self.epsilon ** 2 * self.fact_scale) * K_sum_rows)[0].tolist()

            # Check the cardinals of I and J
            print('|I_active| = %s \t |J_active| = %s \t |I_active| + |J_active| = %s' \
                  % (len(self.I), len(self.J), len(self.I) + len(self.J)))

            # Bounds in  LBFGS
            self.bounds_u = [(self.fact_scale * a_sort[self.I][-1] / (self.epsilon * self.fact_scale * (m - len(self.J)) \
                                                                      + len(self.J) * (b_sort[self.J][0] / (
                                self.epsilon * n * K_min))), \
                              a_sort[self.I][0] / (self.epsilon * m * K_min))] * len(self.I)

            self.bounds_v = [((b_sort[self.J][-1] / self.fact_scale) / (
                        (self.epsilon * (n - len(self.I)) / self.fact_scale) \
                        + len(self.I) * (a_sort[self.I][0] / (self.epsilon * m * K_min))), \
                              b_sort[self.J][0] / (self.epsilon * n * K_min))] * len(self.J)

        # Ic, Jc
        self.Ic = list(set(list(range(n))) - set(self.I))
        self.Jc = list(set(list(range(m))) - set(self.J))

        self.a_I = self.a[self.I]
        self.b_J = self.b[self.J]

        self.a_Ic = self.a[self.Ic]
        self.b_Jc = self.b[self.Jc]

        self.K_IJ = self.K[np.ix_(self.I, self.J)]
        self.K_IcJ = self.K[np.ix_(self.Ic, self.J)]
        self.K_IJc = self.K[np.ix_(self.I, self.Jc)]

        self.vec_eps_IJc = self.fact_scale * self.epsilon * (self.K_IJc * np.ones(len(self.Jc)).reshape((1, -1))).sum(
            axis=1)
        self.vec_eps_IcJ = (self.epsilon / self.fact_scale) * (np.ones(len(self.Ic)).reshape((-1, 1)) * self.K_IcJ).sum(
            axis=0)

        # Restricted Sinkhron
        if self.N != n or self.M != m:
            # self.K_IJ_p = (1 / (self.fact_scale * self.a[self.I])).reshape(-1, 1) * self.K_IJ
            # self.cst_u = np.divide(self.fact_scale * self.epsilon * self.K_IJc.sum(axis=1), self.fact_scale * self.a_I)

            self.cst_u = self.fact_scale * self.epsilon * self.K_IJc.sum(axis=1)
            self.cst_v = self.epsilon * self.K_IcJ.sum(axis=0) / self.fact_scale

        self.toc_initial = time() - tic_initial

    def _projection_u(self, u):

        u[np.where(u <= self.epsilon / self.fact_scale)] = self.epsilon / self.fact_scale
        return u

    def _projection_v(self, v):

        v[np.where(v <= self.epsilon * self.fact_scale)] = self.epsilon * self.fact_scale
        return v

    def objective(self, u_param, v_param):

        part_IJ = u_param @ self.K_IJ @ v_param - self.fact_scale * self.a_I @ np.log(u_param) \
                  - (1. / self.fact_scale) * self.b_J @ np.log(v_param)
        part_IJc = u_param @ self.vec_eps_IJc
        part_IcJ = self.vec_eps_IcJ @ v_param
        psi_epsilon = part_IJ + part_IJc + part_IcJ
        return psi_epsilon

    def grad_objective(self, u_param, v_param):

        # gradients of Psi_epsilon w. r. t. u and v
        grad_u = self.K_IJ @ v_param + self.vec_eps_IJc - self.fact_scale * self.a_I / u_param
        grad_v = self.K_IJ.T @ u_param + self.vec_eps_IcJ - self.b_J / (self.fact_scale * v_param)
        return grad_u, grad_v

    def restricted_sinkhorn(self, usc, vsc, max_iter=10):
        cpt = 1

        while (cpt < max_iter):
            K_IJ_transpose = self.fact_scale * (self.K_IJ.T @ usc + self.cst_v)

            vsc = self.b_J / K_IJ_transpose

            KIJ_v = self.K_IJ @ vsc + self.cst_u
            usc = np.divide(self.fact_scale * self.a_I, KIJ_v)

            cpt += 1
        usc = self._projection_u(usc)
        vsc = self._projection_v(vsc)

        return usc, vsc

    def _bfgspsot(self, theta):
        u = theta[:len(self.I)]
        v = theta[len(self.I):]
        # objective value
        f = self.objective(u, v)
        # gradient
        g_u, g_v = self.grad_objective(u, v)
        g = np.hstack([g_u, g_v])
        return f, g

    def lbfgsb(self):

        (n, m) = self.C.shape

        u0 = np.full(len(self.I), 1. / len(self.I) + self.epsilon)
        v0 = np.full(len(self.J), 1. / len(self.J) + self.epsilon)

        u, v = self.restricted_sinkhorn(u0, v0, max_iter=3)

        # params of bfgs
        theta0 = np.hstack([u, v])
        maxiter = 10000  # max number of iterations
        maxfun = 1000  # max  number of function evaluations
        pgtol = 1e-09  # final objective function accuracy

        obj = lambda theta: self._bfgspsot(theta)
        bounds = self.bounds_u + self.bounds_v

        theta, _, d = fmin_l_bfgs_b(func=obj,
                                    x0=theta0,
                                    bounds=bounds,
                                    # m=2,
                                    factr=1e5,
                                    maxfun=maxfun,
                                    pgtol=pgtol,
                                    maxiter=maxiter)

        usc = theta[:len(self.I)]
        vsc = theta[len(self.I):]

        usc_full = np.full(n, self.epsilon / self.fact_scale)
        vsc_full = np.full(m, self.epsilon * self.fact_scale)
        usc_full[self.I] = usc
        vsc_full[self.J] = vsc
        Psc = usc_full.reshape((-1, 1)) * self.K * vsc_full.reshape((1, -1))

        return usc_full, vsc_full, Psc, d
