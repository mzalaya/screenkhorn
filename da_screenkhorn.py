#!/usr/bin/env python
# coding: utf-8

__author__ = 'Mokhtar Z. Alaya'

"""
Domain adaptation with Screened optimal transport.
The script is adapted from ot/da.py in the POT toolbox.
"""

import numpy as np
import scipy.linalg as linalg
from time import process_time as time

# POT
from ot.da import BaseTransport

# SCREENKHORN
from screenkhorn import Screenkhorn

def screenkhorn_lpl1_mm(a, labels_a, b, M, reg, eta=0.1, numItermax=10,
                     numInnerItermax=200, stopInnerThr=1e-9, verbose=False,
                     log=False, **kwargs):
    """
    Solve the entropic regularization optimal transport problem with nonconvex
    group lasso regularization
    """
    p = 0.5
    epsilon = 1e-3

    indices_labels = []
    classes = np.unique(labels_a)
    for c in classes:
        idxc, = np.where(labels_a == c)
        indices_labels.append(idxc)

    p_n = kwargs.get('p_n', 1)
    p_m = kwargs.get('p_m', 1)
    one_init =  kwargs.get('one_init', False)

    ns_budget = int(np.ceil(M.shape[0] / p_n))
    nt_budget = int(np.ceil(M.shape[1] / p_m))
    
    screenkhorn = Screenkhorn(a, b, M, reg, ns_budget=n_budget, nt_budget=nt_budget, verbose=False, one_init=one_init)
    transp = screenkhorn.lbfgsb()
    
    W = np.ones(M.shape)
    for (i, c) in enumerate(classes):
        majs = np.sum(transp[indices_labels[i]], axis=0)
        majs = p * ((majs + epsilon)**(p - 1))
        W[indices_labels[i]] = majs
     
    for cpt in range(numItermax-1):
        
        Mreg = M + eta * W
    
        screenkhorn.update(Mreg)
        transp = screenkhorn.lbfgsb()[2]
        W = np.ones(M.shape)
        for (i, c) in enumerate(classes):
            majs = np.sum(transp[indices_labels[i]], axis=0)
            majs = p * ((majs + epsilon)**(p - 1))
            W[indices_labels[i]] = majs

    return transp

class ScreenkhornTransport(BaseTransport):

    """Domain Adapatation OT method based on SCREENKHORN Algorithm
    """

    def __init__(self, reg_e=1., max_iter=1000,
                 tol=10e-9, verbose=False, log=False,
                 metric="sqeuclidean", norm=None,one_init=False,
                 distribution_estimation=distribution_estimation_uniform,
                 out_of_sample_map='ferradans', limit_max=np.infty):

        self.reg_e = reg_e
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.log = log
        self.metric = metric
        self.norm = norm
        self.limit_max = limit_max
        self.distribution_estimation = distribution_estimation
        self.out_of_sample_map = out_of_sample_map
        self.one_init=one_init

    def fit(self, Xs=None, ys=None, Xt=None, yt=None, **kwargs):
        """Build a coupling matrix from source and target sets of samples
        (Xs, ys) and (Xt, yt)
        """

        super(ScreenkhornTransport, self).fit(Xs, ys, Xt, yt)

        # coupling estimation

        # screenkhorn
        dec_ns = kwargs.get('dec_ns', 2)# keep only 50% of points
        dec_nt = kwargs.get('dec_nt', 2)# keep only 50% of points

        ns_budget = int(np.ceil(self.cost_.shape[0] / dec_ns))
        nt_budget = int(np.ceil(self.cost_.shape[1] / dec_nt))

        screenkhorn = Screenkhorn(a=self.mu_s, b=self.mu_t, C=self.cost_, reg=self.reg_e,
                                  ns_budget=ns_budget, nt_budget=nt_budget, one_init=self.one_init_init,
                                  verbose = False)
        returned_ = screenkhorn.lbfgsb()

        if self.log:
            self.coupling_, self.log_ = returned_
        else:
            self.coupling_ = returned_
            self.log_ = dict()

        return self


class ScreenkhornLpl1Transport(BaseTransport):

    """Domain Adapatation OT method based on SCREENKHORN algorithm + LpL1 class regularization.
    """

    def __init__(self, reg_e=1., reg_cl=0.1,
                 max_iter=10, max_inner_iter=200, log=False,
                 tol=10e-9, verbose=False, one_init= False,
                 metric="sqeuclidean", norm=None,
                 distribution_estimation=distribution_estimation_uniform,
                 out_of_sample_map='ferradans', limit_max=np.infty):

        self.reg_e = reg_e
        self.reg_cl = reg_cl
        self.max_iter = max_iter
        self.max_inner_iter = max_inner_iter
        self.tol = tol
        self.log = log
        self.verbose = verbose
        self.metric = metric
        self.norm = norm
        self.distribution_estimation = distribution_estimation
        self.out_of_sample_map = out_of_sample_map
        self.limit_max = limit_max
        self.one_init = one_init

    def fit(self, Xs, ys=None, Xt=None, yt=None, **kwargs):
        """Build a coupling matrix from source and target sets of samples
        (Xs, ys) and (Xt, yt)
        """
        # check the necessary inputs parameters are here
        if check_params(Xs=Xs, Xt=Xt, ys=ys):

            super(ScreenkhornLpl1Transport, self).fit(Xs, ys, Xt, yt)

            dec_ns = kwargs.get('dec_ns', 2)# keep only 50% of points
            dec_nt = kwargs.get('dec_nt', 2)# keep only 50% of points
            returned_ = screenkhorn_lpl1_mm(
                a=self.mu_s, labels_a=ys, b=self.mu_t, M=self.cost_, one_init= self.one_init,
                reg=self.reg_e, eta=self.reg_cl, numItermax=self.max_iter,
                numInnerItermax=self.max_inner_iter, stopInnerThr=self.tol,
                verbose=self.verbose, log=self.log, dec_ns=dec_ns, dec_nt=dec_nt)

        # deal with the value of log
        if self.log:
            self.coupling_, self.log_ = returned_
        else:
            self.coupling_ = returned_
            self.log_ = dict()
        return self
