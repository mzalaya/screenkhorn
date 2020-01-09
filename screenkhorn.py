#!/usr/bin/env python
# coding: utf-8

__author__ = 'Mokhtar Z. Alaya' 

import numpy as np
import bottleneck
from scipy.optimize import fmin_l_bfgs_b
from time import time

class Screenkhorn:
    """
    Screenkhorn: solver of screening Sinkhorn algorithm for discrete regularized optimal transport (OT)

    Parameters
    ----------
    a : `numpy.ndarray`, shape=(ns,) 
        samples weights in the source domain

    b : `numpy.ndarray`, shape=(nt,) 
        samples weights in the target domain

    C : `numpy.ndarray`, shape=(ns, nt)
        Cost matrix

    reg : `float`
        Level of the entropy regularisation

    ns_budget: `int`, default=None
        Number budget of points to be keeped in the source domain
        If it is None then 50% of the target sample points will be keeped

    nt_budget: `int`, default=None
        Number budget of points to be keeped in the target domain
        If it is None then 50% of the target sample points will be keeped

    uniform: `bool`, default=True
        If `True`, a_i = 1 /ns and b_j = 1 / nt

    restricted: `bool`, default=True
         If `True`, a warm-start initialization for the  LBFGSB solver
         using a Sinkhorn-like with at most 5 iterations

    verbose: `bool`, default=True
        If `True`, dispaly informations along iterations

    Dependency
    ----------
    To gain more efficiency, screenkhorn needs to call the "Bottleneck" package (https://pypi.org/project/Bottleneck/) in the screening pre-processing step.
    If Bottleneck isn't installed, the following error message appears:
    "Bottleneck module doesn't exist. Install it from https://pypi.org/project/Bottleneck/"

    Returns
    -------
    Psc : numpy.ndarray`, shape=(ns, nt)
        Screened optimal transportation matrix for the given parameters 

    """
    # check if bottleneck module exists
    try:
        import bottleneck
        print("tototototototo")
    except ImportError as e:
        print("Bottleneck module doesn't exist. Install it from https://pypi.org/project/Bottleneck/")

    def __init__(self, a, b, C, reg, ns_budget=None, nt_budget=None, verbose=True,
                 uniform=True, restricted=True, one_init=False):

        tic_initial = time()

        self.a = np.asarray(a, dtype=np.float64)
        self.b = np.asarray(b, dtype=np.float64)

        # if autograd package is used, we then have to change
        # some arrays from "ArrayBox" type to "np.array".
        if isinstance(C, np.ndarray) == False:
            C = C._value

        self.C = np.asarray(C, dtype=np.float64)
        self.reg = reg
        ns = C.shape[0]
        nt = C.shape[1]
        self.ns_budget = ns_budget
        self.nt_budget = nt_budget
        self.verbose = verbose
        self.uniform = uniform
        self.restricted = restricted
        self.one_init = one_init

        # by default, we keep only 50% of the sample data points
        if self.ns_budget is None:
            self.ns_budget = int(np.floor(0.5 * ns))
        if self.nt_budget is None:
            self.nt_budget = int(np.floor(0.5 * nt))

        # calculate the Gibbs kernel
        self.K = np.empty_like(self.C)
        np.divide(self.C, - self.reg, out=self.K)
        np.exp(self.K, out=self.K)

        # screening test (see Lemma 1 in the paper)
        if self.ns_budget == ns and self.nt_budget == nt:
            # I, J
            self.I = np.ones(ns, dtype=bool)
            self.J = np.ones(nt, dtype=bool)
            # epsilon
            self.epsilon = 0.0
            # scale factor
            self.fact_scale = 1.0
            # restricted Sinkhron
            self.cst_u = 0.
            self.cst_v = 0.
            # box constraints in LBFGS
            self.bounds_u = [(0.0, np.inf)] * ns
            self.bounds_v = [(0.0, np.inf)] * nt
            #
            self.K_IJ = self.K
            self.a_I = self.a
            self.b_J = self.b
            self.K_IJc = []
            self.K_IcJ = []

        else:
            if self.verbose:
                print(time() - tic_initial)
            # sum of rows and columns of K
            K_sum_cols = self.K.sum(axis=1)
            K_sum_rows = self.K.sum(axis=0)

            if self.uniform:
                if ns / self.ns_budget < 4:
                    aK_sort = np.sort(K_sum_cols)
                    epsilon_u_square = a[0] / aK_sort[self.ns_budget - 1]
                    print(epsilon_u_square)
                else :
                    aK_sort = bottleneck.partition(K_sum_cols, ns_budget-1)[ns_budget-1]
                    epsilon_u_square = a[0] / aK_sort
                    
                if nt / self.ns_budget < 4:
                    bK_sort = np.sort(K_sum_rows)
                    epsilon_v_square = b[0]/bK_sort[self.nt_budget - 1]
                else:
                    bK_sort = bottleneck.partition(K_sum_rows, nt_budget-1)[nt_budget-1]
                    epsilon_v_square =b[0] / bK_sort  
            else:
                aK = a / K_sum_cols
                bK = b / K_sum_rows
                
                aK_sort = np.sort(aK)[::-1]
                epsilon_u_square = aK_sort[self.ns_budget - 1]
            
                bK_sort = np.sort(bK)[::-1]
                epsilon_v_square = bK_sort[self.nt_budget - 1]
            # I, J
            self.I = self.a >=  epsilon_u_square * K_sum_cols
            self.J = self.b >=  epsilon_v_square * K_sum_rows
             
            if sum(self.I) != self.ns_budget:
                print("test error", sum(self.I), self.ns_budget)
                if self.uniform:
                    aK = a / K_sum_cols
                    aK_sort = np.sort(aK)[::-1]
                epsilon_u_square = aK_sort[self.ns_budget - 1:self.ns_budget+1].mean()
                self.I = self.a >=  epsilon_u_square * K_sum_cols
            
            if sum(self.J) != self.nt_budget:
                print("test error", len(self.J), self.nt_budget)
                if self.uniform:
                    bK = b / K_sum_rows
                    bK_sort = np.sort(bK)[::-1]
                epsilon_v_square = bK_sort[self.nt_budget - 1:self.nt_budget+1].mean()
                self.J = self.b >= epsilon_v_square * K_sum_rows
                
            self.epsilon = (epsilon_u_square * epsilon_v_square)**(1/4)
            self.fact_scale = (epsilon_v_square / epsilon_u_square)**(1/2)
            
            if self.verbose:
                print("Epsilon = %s\n" % self.epsilon)
                print(time() - tic_initial)
            
            if self.verbose:
                print('|I_active| = %s \t |J_active| = %s \t |I_active| + |J_active| = %s'\
                      %(len(self.I), len(self.J), len(self.I) + len(self.J)))

            
            # Ic, Jc: complementary sets of I and J
            self.Ic = ~self.I
            self.Jc = ~self.J
           #
            self.K_IJ = self.K[np.ix_(self.I, self.J)]
            self.K_IcJ = self.K[np.ix_(self.Ic, self.J)]
            self.K_IJc = self.K[np.ix_(self.I, self.Jc)]
            # K_min
            K_min = self.K_IJ.min()
            if K_min == 0:
                K_min = np.finfo(float).tiny  

            # a_I,b_J,a_Ic,b_Jc
            self.a_I = self.a[self.I]
            self.b_J = self.b[self.J]
            if not self.uniform:
                self.a_I_min = self.a_I.min()
                self.a_I_max = self.a_I.max()
                self.b_J_max = self.b_J.max()
                self.b_J_min = self.b_J.min()
            else:
                self.a_I_min = self.a_I[0]
                self.a_I_max = self.a_I[0]
                self.b_J_max = self.b_J [0]
                self.b_J_min = self.b_J[0]
            
            # box constraints in L-BFGS-B (see Proposition 1 in the paper)
            self.bounds_u = [(max(self.a_I_min / (self.epsilon * (nt - self.nt_budget) \
                                                    + self.nt_budget * (self.b_J_max / (self.epsilon *  self.fact_scale * ns * K_min))), \
                                  self.epsilon / self.fact_scale), \
                              self.a_I_max / (self.epsilon * nt * K_min))] * self.ns_budget

            self.bounds_v = [(max(self.b_J_min / (self.epsilon * (ns - self.ns_budget) \
                                                    + self.ns_budget * (self.fact_scale * self.a_I_max / (self.epsilon * nt * K_min))), \
                                  self.epsilon * self.fact_scale), \
                              self.b_J_max / (self.epsilon * ns * K_min))] * self.nt_budget
         
        if self.verbose:
                print("Time of initialization %s:" %(time() - tic_initial))

        
        self.vec_eps_IJc = self.epsilon * self.fact_scale \
                           * (self.K_IJc * np.ones(nt - self.nt_budget).reshape((1, -1))).sum(axis=1)
        self.vec_eps_IcJ = (self.epsilon / self.fact_scale) \
                           * (np.ones(ns - self.ns_budget).reshape((-1, 1)) * self.K_IcJ).sum(axis=0)

        # restricted Sinkhron
        if self.ns_budget != ns or self.ns_budget != nt:
            self.cst_u = self.fact_scale * self.epsilon * self.K_IJc.sum(axis=1)
            self.cst_v = self.epsilon * self.K_IcJ.sum(axis=0) / self.fact_scale

        if not self.one_init:
            u0 = np.full(self.ns_budget, (1. / self.ns_budget) + self.epsilon / self.fact_scale)
            v0 = np.full(self.nt_budget, (1. / self.nt_budget) + self.epsilon * self.fact_scale)
        else:
            print('one initialization')
            u0 = np.full(self.ns_budget, 1.)
            v0 = np.full(self.nt_budget, 1.)
        
        if self.restricted:
            self.u0, self.v0 = self._restricted_sinkhorn(u0, v0, max_iter=5)
        else:
            print('no restricted')
            self.u0 = u0
            self.v0 = v0

        if self.verbose :
            print(time() - tic_initial)

        self.toc_initial = time() - tic_initial

    def update(self, C):
        """
       we use this function to gain more efficiency in OTDA experiments
        """
        self.C = np.asarray(C, dtype=np.float64)
        nt = C.shape[0]
        ns = C.shape[1]
        self.K = np.exp(-self.C / self.reg)

        # sum of rows and columns of K
        K_sum_cols = self.K.sum(axis=1)
        K_sum_rows = self.K.sum(axis=0)
                      
        if self.uniform:
            if ns / self.ns_budget < 4:
                aK_sort = np.sort(K_sum_cols)
                epsilon_u_square = self.a[0] / aK_sort[self.ns_budget - 1]
            else :
                aK_sort = bottleneck.partition(K_sum_cols, self.ns_budget-1)[self.ns_budget-1]
                epsilon_u_square = self.a[0] / aK_sort
                
            if nt / self.nt_budget < 4:
                bK_sort = np.sort(K_sum_rows)
                epsilon_v_square = self.b[0] / bK_sort[self.nt_budget - 1]
            else:
                bK_sort = bottleneck.partition(K_sum_rows, self.nt_budget-1)[self.nt_budget-1]
                epsilon_v_square = self.b[0] / bK_sort
                
        else:
            aK = self.a / K_sum_cols
            bK = self.b / K_sum_rows
            
            aK_sort = np.sort(aK)[::-1]
            epsilon_u_square = aK_sort[self.ns_budget - 1]
            bK_sort = np.sort(bK)[::-1]
            epsilon_v_square = bK_sort[self.nt_budget - 1] 
        
        # I, J
        self.I = self.a >= epsilon_u_square * K_sum_cols
        self.J = self.b >= epsilon_v_square * K_sum_rows
                      
        if sum(self.I) != self.ns_budget:
            if self.uniform:
                aK = self.a / K_sum_cols
            aK_sort = np.sort(aK)[::-1]
            epsilon_u_square = aK_sort[self.ns_budget - 1:self.ns_budget+1].mean()
            self.I = self.a >= epsilon_u_square * K_sum_cols
            
        if sum(self.J) != self.nt_budget:
            if self.uniform:
                bK = self.b / K_sum_rows
            bK_sort = np.sort(bK)[::-1]
            epsilon_v_square = bK_sort[self.nt_budget - 1:self.nt_budget+1].mean()
            self.J = self.b >= epsilon_v_square * K_sum_rows

        self.epsilon = (epsilon_u_square * epsilon_v_square)**(1/4)
        self.fact_scale = (epsilon_v_square / epsilon_u_square)**(1/2)
        
        if self.verbose:
            print("Epsilon = %s\n" % self.epsilon)
            print("Scaling factor = %s\n" % self.fact_scale)
            
         # Ic, Jc
        self.Ic = ~self.I
        self.Jc = ~self.J
        if self.verbose:
            print('|I_active| = %s \t |J_active| = %s \t |I_active| + |J_active| = %s'\
                  %(sum(self.I), sum(self.J), sum(self.I) + sum(self.J)))
        # K_min  
        self.K_IJ = self.K[np.ix_(self.I, self.J)]
        self.K_IcJ = self.K[np.ix_(self.Ic, self.J)]
        self.K_IJc = self.K[np.ix_(self.I, self.Jc)]
        
        K_min = self.K_IJ.min()
        if K_min == 0:
            K_min = np.finfo(float).tiny
        # a_I,b_J,a_Ic,b_Jc
        self.a_I = self.a[self.I]
        self.b_J = self.b[self.J]
        if not self.uniform:
                self.a_I_min = self.a_I.min()
                self.a_I_max = self.a_I.max()
                self.b_J_max = self.b_J.max()
                self.b_J_min = self.b_J.min()
        else:
                self.a_I_min = self.a_I[0]
                self.a_I_max = self.a_I[0]
                self.b_J_max = self.b_J [0]
                self.b_J_min = self.b_J[0]

        # box constraints in LBFGS solver (see Proposition 1 in the paper)
        self.bounds_u = [(max(self.a_I_min / (self.epsilon * (nt - self.nt_budget) \
                                                    + self.nt_budget * (self.b_J_max / (self.epsilon *  self.fact_scale * ns * K_min))), \
                                  self.epsilon / self.fact_scale), \
                              self.a_I_max / (self.epsilon * nt * K_min))] * self.ns_budget

        self.bounds_v = [(max(self.b_J_min / (self.epsilon * (ns - self.ns_budget) \
                                                    + self.ns_budget * (self.fact_scale * self.a_I_max / (self.epsilon * nt * K_min))), \
                                  self.epsilon * self.fact_scale), \
                              self.b_J_max / (self.epsilon * ns * K_min))] * self.nt_budget

        self.vec_eps_IJc = self.epsilon * self.fact_scale \
                           * (self.K_IJc * np.ones(nt-self.nt_budget).reshape((1, -1))).sum(axis=1)
        self.vec_eps_IcJ = (self.epsilon / self.fact_scale) \
                           * (np.ones(ns-self.ns_budget).reshape((-1, 1)) * self.K_IcJ).sum(axis=0)

        # pre-calculed constans for restricted Sinkhron
        if self.ns_budget != ns or self.ns_budget != nt:
            self.cst_u = self.fact_scale * self.epsilon * self.K_IJc.sum(axis=1)
            self.cst_v = self.epsilon * self.K_IcJ.sum(axis=0) / self.fact_scale

        if not self.one_init:
            u0 = np.full(self.ns_budget, (1. / self.ns_budget) + self.epsilon / self.fact_scale)
            v0 = np.full(self.nt_budget, (1. / self.nt_budget) + self.epsilon * self.fact_scale)
        else:
            u0 = np.full(self.ns_budget, 1.)
            v0 = np.full(self.nt_budget, 1.)
        
        if self.restricted:
            self.u0, self.v0 = self._restricted_sinkhorn(u0, v0, max_iter=5)
        else:
            print('no_restricted')
            self.u0 = u0
            self.v0 = v0

    def _projection(self, u, epsilon):
        u[u <= epsilon] = epsilon
        return u

    def _objective(self, u_param, v_param):
        part_IJ = u_param @ self.K_IJ @ v_param\
                  - self.fact_scale * self.a_I @ np.log(u_param) - (1. / self.fact_scale) * self.b_J @ np.log(v_param)
        part_IJc = u_param @ self.vec_eps_IJc
        part_IcJ = self.vec_eps_IcJ @ v_param
        psi_epsilon = part_IJ + part_IJc + part_IcJ
        return psi_epsilon

    def _grad_objective(self, u_param, v_param):
        # gradients of Psi_epsilon wrt u and v
        grad_u = self.K_IJ @ v_param + self.vec_eps_IJc - self.fact_scale * self.a_I / u_param
        grad_v = self.K_IJ.T @ u_param + self.vec_eps_IcJ - (1. / self.fact_scale) * self.b_J / v_param
        return grad_u, grad_v

    def _restricted_sinkhorn(self, usc, vsc, max_iter=5):
        """
        Restricted Sinkhorn as a warm-start initialized point for LBFGSB
        """
        cpt = 1
        while (cpt < max_iter):
            K_IJ_v = self.K_IJ.T @ usc + self.cst_v
            vsc = self.b_J / (self.fact_scale * K_IJ_v)
            KIJ_u = self.K_IJ @ vsc + self.cst_u
            usc = (self.fact_scale * self.a_I) / KIJ_u
            cpt += 1

        usc = self._projection(usc, self.epsilon / self.fact_scale)
        vsc = self._projection(vsc, self.epsilon * self.fact_scale)

        return usc, vsc

    def _bfgspost(self, theta):
        u = theta[:self.ns_budget]
        v = theta[self.ns_budget:]
        # objective value
        f = self._objective(u, v)
        # gradient
        g_u, g_v = self._grad_objective(u, v)
        g = np.hstack([g_u, g_v])
        return f, g

    def lbfgsb(self):

        (ns, nt) = self.C.shape
        
        # params of LBFGSB
        theta0 = np.hstack([self.u0, self.v0])
        maxiter = 10000 # max number of iterations
        maxfun = 10000 # max  number of function evaluations
        pgtol = 1e-06 # final objective function accuracy

        bounds = self.bounds_u + self.bounds_v # constraint bounds
        obj = lambda theta: self._bfgspost(theta)

        theta, _, d = fmin_l_bfgs_b(func=obj,
                                      x0=theta0,
                                      bounds=bounds,
                                      maxfun=self.maxfun,
                                      pgtol=self.pgtol,
                                      maxiter=self.maxiter)

        usc = theta[:self.ns_budget]
        vsc = theta[self.ns_budget:]

        usc_full = np.full(ns, self.epsilon / self.fact_scale)
        vsc_full = np.full(nt, self.epsilon * self.fact_scale)
        usc_full[self.I] = usc
        vsc_full[self.J] = vsc
        Psc = usc_full.reshape((-1, 1)) * self.K * vsc_full.reshape((1, -1))

        return Psc
