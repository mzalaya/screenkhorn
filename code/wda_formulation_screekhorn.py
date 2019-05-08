# -*- coding: utf-8 -*-
"""
    screenkhorn: solver for screening Sinkhorn via dual projection

"""
__author__ = 'Mokhtar Z. Alaya'

import numpy as np
# import autograd.numpy as NP
from scipy.optimize import fmin_l_bfgs_b
from time import time

class Screenkhorn:

    def __init__(self, a, b, C, reg, n_b, m_b,verbose=True,uniform=True):

        tic_initial = time()

        # In wda_screen.py autograd package is used, we then have to change some arrays from "ArrayBox" type to "np.array".
        if isinstance(C,np.ndarray) == False:
            C = C._value

        self.a = np.asarray(a, dtype=np.float64)
        self.b = np.asarray(b, dtype=np.float64)
        self.C = np.asarray(C, dtype=np.float64)

        
        self.reg = reg
        n = C.shape[0]
        m = C.shape[1]
        self.n_b = n_b
        self.m_b = m_b
        self.verbose = verbose
        self.uniform = uniform
        # K
        print(time() - tic_initial)

        self.K = np.empty_like(self.C)
        np.divide(self.C, - self.reg, out=self.K)
        np.exp(self.K, out=self.K)

        # Test
        if self.n_b == n and self.m_b == m:

            # I, J
            self.I = list(range(n))
            self.J = list(range(m))

            # epsilon
            self.epsilon = 0.0
            # scale factor
            self.fact_scale = 1.0

            # restricted Sinkhron
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

            if not self.uniform:
                a_sort = np.sort(a)
                b_sort = np.sort(b)
            else:
                a_sort,b_sort = a,b
                
            aK_sort = np.sort(a / K_sum_cols)[::-1]
            bK_sort = np.sort(b / K_sum_rows)[::-1]

            epsilon_u_square = aK_sort[self.n_b - 1]
            epsilon_v_square = bK_sort[self.m_b - 1]

            self.epsilon = (epsilon_u_square * epsilon_v_square)**(1/4)
            self.fact_scale = (epsilon_v_square / epsilon_u_square)**(1/2)
            

            # print("Epsilon = %s" % self.epsilon)
            # print("Scaling factor = %s" % self.fact_scale)

            # I, J

            self.I = np.where(self.a >= (self.epsilon**2 / self.fact_scale) * K_sum_cols)[0].tolist()
            self.J = np.where(self.b >= self.epsilon**2 * self.fact_scale * K_sum_rows)[0].tolist()
            
            if self.verbose:
                print('|I_active| = %s \t |J_active| = %s \t |I_active| + |J_active| = %s\n'\
                      %(len(self.I), len(self.J), len(self.I) + len(self.J)))

            # LBFGS box

            self.bounds_u = [(
                                max(self.fact_scale * a_sort[self.I][-1] / (self.fact_scale  * self.epsilon * (m - len(self.J)) \
                                                    + max(self.fact_scale * self.epsilon, len(self.J) * (
                                                                b_sort[self.J][0] / (self.epsilon * n * K_min)))), self.epsilon / self.fact_scale), \
                                max(self.epsilon / self.fact_scale, a_sort[self.I][0] / (self.epsilon * m * K_min))

                             )] * len(self.I)

            self.bounds_v = [(
                                max((1. / self.fact_scale) * b_sort[self.J][-1] / ( (self.epsilon / self.fact_scale) * (n - len(self.I)) \
                                                    + max((self.epsilon / self.fact_scale), len(self.I) * (
                                                                a_sort[self.I][0] / (self.epsilon * m * K_min)))), self.epsilon * self.fact_scale), \
                                max(self.epsilon * self.fact_scale, b_sort[self.J][0] / (n * self.epsilon * K_min))

                            )] * len(self.J)

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
        # self.K_IcJc = self.K[np.ix_(self.Ic, self.Jc)]

        self.vec_eps_IJc = self.epsilon * self.fact_scale * (self.K_IJc * np.ones(len(self.Jc)).reshape((1, -1))).sum(axis=1)
        self.vec_eps_IcJ = (self.epsilon / self.fact_scale) * (np.ones(len(self.Ic)).reshape((-1, 1)) * self.K_IcJ).sum(axis=0)
        # self.vec_eps_IcJc = self.epsilon**2 * self.K_IcJc.sum()

        # restricted Sinkhron
        if self.n_b != n or self.m_b != m:
            self.cst_u = self.fact_scale * self.epsilon * self.K_IJc.sum(axis=1)
            self.cst_v = self.epsilon * self.K_IcJ.sum(axis=0) / self.fact_scale


        self.toc_initial = time() - tic_initial



    def _projection(self, u, epsilon):

        u[np.where(u <= epsilon)] = epsilon
        return u

    def objective(self, u_param, v_param):

        part_IJ = u_param @ self.K_IJ @ v_param - self.fact_scale * self.a_I @ np.log(u_param)\
                  - (1. / self.fact_scale) * self.b_J @ np.log(v_param)
        part_IJc = u_param @ self.vec_eps_IJc
        part_IcJ = self.vec_eps_IcJ @ v_param
        psi_epsilon = part_IJ + part_IJc + part_IcJ
        return psi_epsilon #+ self.vec_eps_IcJc

    def grad_objective(self, u_param, v_param):

        # gradients of Psi_epsilon w. r. t. u and v
        grad_u = self.K_IJ @ v_param + self.vec_eps_IJc - self.fact_scale * self.a_I / u_param
        grad_v = self.K_IJ.T @ u_param + self.vec_eps_IcJ - (1. / self.fact_scale) * self.b_J / v_param
        return grad_u, grad_v

    def restricted_sinkhorn(self, usc, vsc, max_iter=5):
        cpt = 1

        while (cpt < max_iter):

            K_IJ_v = self.K_IJ.T @ usc + self.cst_v
            vsc = np.divide(self.b_J, self.fact_scale * K_IJ_v)

            KIJ_u = self.K_IJ @ vsc + self.cst_u
            usc = np.divide(self.fact_scale * self.a_I, KIJ_u)

            cpt += 1

        usc = self._projection(usc, self.epsilon / self.fact_scale)
        vsc = self._projection(vsc, self.epsilon * self.fact_scale)

        return usc, vsc

    def _bfgspost(self, theta):
        u = theta[:len(self.I)]
        v = theta[len(self.I):]
        # objective value
        f = self.objective(u, v)
        # gradient
        g_u, g_v = self.grad_objective(u, v)
        g = np.hstack([g_u, g_v])
        return f, g

    def lbfgsb(self):

        (n, m) = self.K.shape

        u = np.full(len(self.I), (1. / len(self.I)) + self.epsilon / self.fact_scale)
        v= np.full(len(self.J), (1. / len(self.J)) + self.epsilon * self.fact_scale)

        u, v = self.restricted_sinkhorn(u, v, max_iter=3)

        # params of bfgs
        theta0 = np.hstack([u, v])
        maxiter = 10000 # max number of iterations
        #maxfun = 1000 # max  number of function evaluations
        pgtol = 1e-09 # final objective function accuracy

        obj = lambda theta: self._bfgspost(theta)
        bounds = self.bounds_u + self.bounds_v

        theta, _, d = fmin_l_bfgs_b(func=obj,
                                      x0=theta0,
                                      bounds=bounds,
                                      #maxfun=maxfun,
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