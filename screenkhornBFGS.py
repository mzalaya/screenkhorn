# -*- coding: utf-8 -*-
"""
    screenkhorn: solver for screening Sinkhorn via dual projection

"""
__author__ = 'Mokhtar Z. Alaya'

import numpy as np
np.random.seed(3946)
from scipy import optimize, linalg
from scipy.optimize import fmin_l_bfgs_b
from time import time
from tqdm import tqdm
import warnings
import logging


class Screenkhorn:

    def __init__(self, a, b, C, reg, N, M):

        tic_initial = time()
        self.a = np.asarray(a, dtype=np.float64)
        self.b = np.asarray(b, dtype=np.float64)
        self.C = np.asarray(C, dtype=np.float64)
        self.reg = reg
        n = C.shape[0]
        m = C.shape[1]
        self.N = N
        self.M = M

        # K
        self.K = np.empty_like(self.C)
        np.divide(self.C, - self.reg, out=self.K)
        np.exp(self.K, out=self.K)

        # Test
        if self.N == n and self.M == m:

            # I, J
            self.I = list(range(n))
            self.J = list(range(m))

            # epsilon
            self.epsilon_u = 0.0
            self.epsilon_v = 0.0

            # restricted Sinkhron
            self.K_IJ_p = (1 / self.a).reshape(-1, 1) * self.K
            self.cst_u = 0.
            self.cst_v = 0.

            # box BFGS
            self.bounds_u = [(0.0, np.inf)] * n
            self.bounds_v = [(0.0, np.inf)] * m

        else:

            # sum of rows and columns of K
            K_sum_cols = self.K.sum(axis=1)
            K_sum_rows = self.K.T.sum(axis=1)

            # K_min
            K_min = self.K.min()

            #
            a_sort = np.sort(a)[::-1]
            b_sort = np.sort(b)[::-1]

            # epsilon

            ## 1st choice

            # ab_sort = np.sort(np.hstack([np.sqrt(np.divide(self.a, K_sum_cols)),\
            #                              np.sqrt(np.divide(self.b, K_sum_rows))]))[::-1]
            #
            # self.epsilon = ab_sort[self.N + self.M - 1: self.N + self.M].mean()
            # self.epsilon_u = self.epsilon
            # self.epsilon_v = self.epsilon

            ## 2sd  choice

            K_sum_cols_sort = np.sort(K_sum_cols)
            K_sum_rows_sort = np.sort(K_sum_rows)

            self.epsilon = (np.sqrt(np.divide(a_sort, K_sum_cols_sort))[self.N - 1]+\
                                      np.sqrt(np.divide(b_sort, K_sum_rows_sort))[self.M - 1])/2

            self.epsilon_u = np.sqrt(np.divide(a_sort, K_sum_cols_sort))[self.N - 1]
            self.epsilon_v = np.sqrt(np.divide(b_sort, K_sum_rows_sort))[self.M - 1]


            #print("Epsilon is %s\n" % self.epsilon)
            print("\n Epsilon_u is %s\n" % self.epsilon_u)
            print(" Epsilon_v is %s\n" % self.epsilon_v)

            # I, J

            self.I = np.where(self.a >= self.epsilon_u**2 * K_sum_cols)[0].tolist()
            self.J = np.where(self.b >= self.epsilon_v**2 * K_sum_rows)[0].tolist()


            print('|I_active| = %s \t |J_active| = %s \t |I_active| + |J_active| = %s'\
              %(len(self.I), len(self.J), len(self.I) + len(self.J)))


            # LBFGS box
            self.bounds_u = [(a_sort[self.I][-1] / (self.epsilon_v * (m - len(self.J)) \
                                                    + len(self.J) * (
                                                                b_sort[self.J][0] / (self.epsilon_u * n * K_min))), \
                              a_sort[self.I][0] / (self.epsilon_v * m * K_min))] * len(self.I)

            self.bounds_v = [(b_sort[self.J][-1] / (self.epsilon_u * (n - len(self.I)) \
                                                    + len(self.I) * (
                                                                a_sort[self.I][0] / (self.epsilon_v * m * K_min))), \
                              b_sort[self.J][0] / (self.epsilon_u * n * K_min))] * len(self.J)

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
        #
        # self.K_IcJc = self.K[np.ix_(self.Ic, self.Jc)]
        # self.part_IcJc = self.epsilon_u * self.epsilon_v * self.K_IcJc.sum() \
                        # - np.log(self.epsilon_u) * self.a_Ic.sum() - np.log(self.epsilon_v) * self.b_Jc.sum()

        self.vec_eps_IJc = self.epsilon_v * (self.K_IJc * np.ones(len(self.Jc)).reshape((1, -1))).sum(axis=1)
        self.vec_eps_IcJ = self.epsilon_u * (np.ones(len(self.Ic)).reshape((-1, 1)) * self.K_IcJ).sum(axis=0)

        # self.vec_eps_IJc = self.epsilon_v * self.K_IJc @ np.ones(len(self.Jc))
        # self.vec_eps_IcJ = self.epsilon_u * np.ones(len(self.Ic)) @ self.K_IcJ

        # restricted Sinkhron
        if self.N != n or self.M != m:

            self.K_IJ_p = (1 / self.a[self.I]).reshape(-1, 1) * self.K_IJ
            self.cst_u = np.divide(self.epsilon_u * self.K_IJc.sum(axis=1), self.a_I)
            self.cst_v = self.epsilon_v * self.K_IcJ.sum(axis=0)


        self.toc_initial = time() - tic_initial


    def _projection(self, u, epsilon):

        u[np.where(u <= epsilon)] = epsilon
        return u

    def objective(self, u_param, v_param):

        part_IJ = u_param @ self.K_IJ @ v_param - self.a_I @ np.log(u_param) - self.b_J @ np.log(v_param)
        part_IJc = u_param @ self.vec_eps_IJc
        part_IcJ = self.vec_eps_IcJ @ v_param
        psi_epsilon = part_IJ + part_IJc + part_IcJ #+ self.part_IcJc
        return psi_epsilon

    def grad_objective(self, u_param, v_param):

        # gradients of Psi_epsilon w. r. t. u and v
        grad_u = self.K_IJ @ v_param + self.vec_eps_IJc - self.a_I / u_param
        grad_v = self.K_IJ.T @ u_param + self.vec_eps_IcJ - self.b_J / v_param
        return grad_u, grad_v

    def restricted_sinkhorn(self, usc, vsc, max_iter=100):
        cpt = 1

        while (cpt < max_iter):

            K_IJ_transpose_u = self.K_IJ.T @ usc + self.cst_v
            vsc = self.b_J / K_IJ_transpose_u
            KIJ_v = self.K_IJ_p @ vsc + self.cst_u
            usc = np.divide(1., KIJ_v)
            cpt += 1

        usc = self._projection(usc, self.epsilon_u)
        vsc = self._projection(vsc, self.epsilon_v)

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

        (n ,m) = self.C.shape

        u0 = np.full(len(self.I), 1. / len(self.I) + self.epsilon_u)
        v0 = np.full(len(self.J), 1. / len(self.J) + self.epsilon_v)

        u, v = self.restricted_sinkhorn(u0, v0, max_iter=3)

        # params of bfgs
        theta0 = np.hstack([u, v])
        maxiter = 10000 # max number of iterations
        maxfun = 10000 # max  number of function evaluations
        pgtol = 1e-09 # final objective function accuracy

        obj = lambda theta: self._bfgspsot(theta)
        bounds = self.bounds_u + self.bounds_v

        theta, _, d = fmin_l_bfgs_b(func=obj,
                                      x0=theta0,
                                      bounds=bounds,
                                      m=2,
                                      factr=1e5,
                                      maxfun=maxfun,
                                      pgtol=pgtol,
                                      maxiter=maxiter)

        usc = theta[:len(self.I)]
        vsc = theta[len(self.I):]

        usc_full = np.full(n, self.epsilon_u)
        vsc_full = np.full(m, self.epsilon_v)
        usc_full[self.I] = usc
        vsc_full[self.J] = vsc
        Psc = usc_full.reshape((-1, 1)) * self.K * vsc_full.reshape((1, -1))

        return usc_full, vsc_full, Psc, d