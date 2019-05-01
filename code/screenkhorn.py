# -*- coding: utf-8 -*-
"""
    screenkhorn: solver for screening Sinkhorn via dual projection

    `math`:
                        min                   u^T K v - a^T log(u) - b^T log(v)
        (u,v) in  (C^n_epsilon x C^m_epsilon)

        where

        C^n_epsilon := {u in R^n: u_i >= epsilon} and C^m_epsilon := {v in R^m: v_j >= epsilon}

"""
__author__ = 'Mokhtar Z. Alaya'

import numpy as np
from scipy import optimize, linalg
from scipy.optimize import fmin_l_bfgs_b

from ot import sinkhorn
import cvxpy as cvx
from numba import jit
from time import time
from datetime import datetime
from tqdm import tqdm
import warnings
import logging


class Screenkhorn:
    """
    Attributes
    ----------
    a : `numpy.ndarray`, shape=(n,)  (read-only)
    b : `numpy.ndarray`, shape=(m,) (read-only)
    M : `numpy.ndarray`, shape=(n,m) (read-only)
    reg : `float`
          Level of regularization
    epsilon : `float`
              Level of thresholding
    """

    def __init__(self, a, b, M, reg, epsilon):
        self.a = np.asarray(a, dtype=np.float64)
        self.b = np.asarray(b, dtype=np.float64)
        self.M = np.asarray(M, dtype=np.float64)
        self.reg = reg  # astype(np.float64)
        self.epsilon = epsilon  # .astype(np.float64)

    @staticmethod
    def _complementary(I, r):
        if I is None:
            raise ValueError('I should be not None')
        Ic = []
        for i in range(r):
            if i not in I:
                Ic.append(i)
        return Ic

    @staticmethod
    def _subvector(I, u):
        return u[np.ix_(I)]

    @staticmethod
    def _submatrix(K, I, J):
        return K[np.ix_(I, J)]

    def _projection(self, u):
        u_proj = u.copy()
        u_proj[np.where(u_proj <= self.epsilon)] = self.epsilon
        return u_proj

    def objective(self, u_param, v_param, I, J, K):
        """
        Parameters
        ----------
        u_param : `numpy.ndarray`, shape=(n,)
        v_param : `numpy.ndarray`, shape=(m,)
        I : `list of int`
        J : `list of int`

        Returns
        -------
        output : objective function \Psi_epsilon(u,v)

        """

        (n, m) = self.M.shape
        # K = np.empty_like(self.M)
        # np.divide(self.M, - self.reg, out=K)
        # np.exp(K, out=K)

        Ic = self._complementary(I, n)
        Jc = self._complementary(J, m)

        # submarginals mu_{I}, nu_{J}, mu_{I^c}, nu_{J^c}
        a_I = self._subvector(I, self.a)
        a_Ic = self._subvector(Ic, self.a)
        b_J = self._subvector(J, self.b)
        b_Jc = self._subvector(Jc, self.b)

        # submatrices K_{IJ}, K_{IJ^c}, K_{I^cJ}, K_{I^cJ^c}
        K_IJ = self._submatrix(K, I, J)
        K_IJc = self._submatrix(K, I, Jc)
        K_IcJ = self._submatrix(K, Ic, J)
        K_IcJc = self._submatrix(K, Ic, Jc)

        part_IJ = (u_param.T @ K_IJ @ v_param - a_I.T @ np.log(u_param) - b_J.T @ np.log(v_param)).sum()
        part_IJc = (self.epsilon * u_param.T @ K_IJc @ np.ones(len(Jc))).sum()
        part_IcJ = (self.epsilon * np.ones(len(Ic)).T @ K_IcJ @ v_param).sum()
        part_IcJc = (self.epsilon**2 * K_IcJc.sum() - np.log(self.epsilon) * (a_Ic.sum() + b_Jc.sum())).sum()

        psi_epsilon = part_IJ + part_IJc + part_IcJ + part_IcJc
        return psi_epsilon

    def grad_objective(self, u_param, v_param, I, J, K):

        (n, m) = self.M.shape

        # K = np.empty_like(self.M)
        # np.divide(self.M, - self.reg, out=K)
        # np.exp(K, out=K)

        Ic = self._complementary(I, n)
        Jc = self._complementary(J, m)

        # submarginals mu_{I}, nu_{J}
        a_I = self._subvector(I, self.a)
        b_J = self._subvector(J, self.b)

        # submatrices K_{IJ}, K_{IJ^c}, K_{I^cJ}
        K_IJ = self._submatrix(K, I, J)
        K_IJc = self._submatrix(K, I, Jc)
        K_IcJ = self._submatrix(K, Ic, J)

        # gradients of Psi_epsilon w. r. t. u and v
        grad_u = K_IJ @ v_param + self.epsilon * K_IJc @ np.ones(len(Jc)) - a_I / u_param
        grad_v = K_IJ.T @ u_param + self.epsilon * K_IcJ.T @ np.ones(len(Ic)) - b_J / v_param
        return grad_u, grad_v

    def proj_grad(self, usc0, vsc0, I, J, max_iter=1000, tol=1e-9, step_size=1e-3,
                       backtracking=True, backtracking_factor=0.5, max_iter_backtracking=20,
                       trace=False, verbose=False) -> optimize.OptimizeResult:

        """
        Proximal Gradient Descent
        Parameters
        ----------
        initial : `np.ndarray`, shape=(card(I),) shape(card(J),)
                   initial point
        step_size : `float`, default=None
                    Initial step size used for learning.

        max_iter: `int`, default=100
                   Maximum number of iterations of the solver.

        tol: `float`, default=1e-9
              The tolerance of the solver (iterations stop when the stopping criterion is below it).
              If not reached it does ``max_iter`` iterations

        verbose: `bool`, default=False
                  If `True`, we verbose things, otherwise the solver does not print anything (but records information
                  in history anyway)

        Returns
        -------
        output : u_sc :`numpy.ndarray`, shape=(n,)
                       u_sc_Ic = log(epsilon) . mathbf{1}_{|I^c|}

                 v_sc : `numpy.ndarray`, shape=(m,)
                       v_sc_Jc = log(epsilon) .  mathbf{1}_{|J^c|}

                 history : `dictionary`
                        A dictionary of the history values
        """
        (n, m) = self.M.shape

        if usc0 is None:
            usc = np.full(len(I), self.epsilon)
        else:
            usc = np.array(usc0[I], copy=True)

        if vsc0 is None:
            vsc = np.full(len(J), self.epsilon)
        else:
            vsc = np.array(vsc0[J], copy=True)

        success = False
        trace_usc = []
        trace_vsc = []
        trace_obj = []
        trace_time = []
        start_time = datetime.now()

        for k in range(max_iter):
        # for k in tqdm(range(max_iter)):
            # Compute gradient and step size.

            current_step_size = step_size
            grad_usc, grad_vsc = self.grad_objective(usc, vsc, I, J)

            usc_new = self._projection(usc - current_step_size * grad_usc)
            vsc_new = self._projection(vsc - current_step_size * grad_vsc)

            incr_usc = usc_new - usc
            incr_vsc = vsc_new - vsc

            if backtracking:
                if not max_iter_backtracking > 0:
                    raise ValueError('Backtracking iterations need to be greater than 0')

                objval = self.objective(usc, vsc, I, J)
                objval_new = self.objective(usc_new, vsc_new, I, J)
                for _ in range(max_iter_backtracking):
                    if objval_new <= objval + np.hstack([grad_usc, grad_vsc]) @ np.hstack([incr_usc, incr_vsc]) \
                            + np.hstack([incr_usc, incr_vsc]).dot((np.hstack([incr_usc, incr_vsc]))) / (
                            2.0 * current_step_size):
                        # Step size found.
                        break
                    else:
                        # Backtracking, reduce step size.
                        current_step_size *= backtracking_factor
                        usc_new = self._projection(usc - current_step_size * grad_usc)
                        vsc_new = self._projection(vsc - current_step_size * grad_vsc)

                        incr_usc = usc_new - usc
                        incr_vsc = vsc_new - vsc
                        objval_new = self.objective(usc_new, vsc_new, I, J)
                else:
                    warnings.warn("Maximum number of line-search iterations reached.")
            certificate = linalg.norm(np.hstack([usc - usc_new, vsc - vsc_new]) / step_size)
            usc = usc_new
            vsc = vsc_new

            if trace:
                trace_usc.append(usc.copy())
                trace_vsc.append(vsc.copy())
                trace_obj.append(self.objective(usc, vsc, I, J))
                trace_time.append((datetime.now() - start_time).total_seconds())

            if verbose:
                print("iteration %s, step size: %s" % (k, step_size))

            if certificate < tol:
                print("Achieved relative tolerance at iteration %s" % k)
                success = True
                break

        if k >= max_iter:
            warnings.warn("Projected Gradient did not reach the desired tolerance level.", RuntimeWarning)

        usc_sol = np.full(n, self.epsilon)
        vsc_sol = np.full(m, self.epsilon)
        usc_sol[I] = usc
        vsc_sol[J] = vsc

        I_usc = np.where(usc_sol > self.epsilon)[0].tolist()
        J_vsc = np.where(vsc_sol > self.epsilon)[0].tolist()

        return optimize.OptimizeResult(usc=usc_sol, vsc=vsc_sol,
                                       I_usc=I_usc, J_vsc=J_vsc,
                                       success=success, certificate=certificate, nb_iter=k,
                                       trace_usc=np.array(trace_usc), trace_vsc=np.array(trace_vsc),
                                       trace_obj=np.array(trace_obj), trace_time=trace_time)

    def bloc_proj_grad(self, usc0, vsc0, I, J, max_iter=100, tol=1e-10,
                             step_size=1e-3,
                             backtracking=True, backtracking_factor=0.5, max_iter_backtracking=20,
                             trace=False, verbose=False) -> optimize.OptimizeResult:

        (n, m) = self.M.shape

        if usc0 is None:
            usc = np.zeros(len(I))
        else:
            usc = np.array(usc0[I], copy=True)

        if vsc0 is None:
            vsc = np.zeros(len(J))
        else:
            vsc = np.array(vsc0[J], copy=True)

        usc_new = np.array(usc, copy=True)
        vsc_new = np.array(vsc, copy=True)

        success = False
        trace_usc = []
        trace_vsc = []
        trace_obj = []
        trace_time = []
        start_time = datetime.now()

        for k in range(max_iter):
        # for k in tqdm(range(max_iter)):
            # Compute gradient and step size.

            current_step_size_u = step_size
            grad_usc, _ = self.grad_objective(usc, vsc, I, J)

            usc_new = self._projection(usc - current_step_size_u * grad_usc)

            incr_usc = usc_new - usc

            if backtracking:
                if not max_iter_backtracking > 0:
                    raise ValueError('Backtracking iterations need to be greater than 0')
                objval = self.objective(usc, vsc, I, J)
                objval_new = self.objective(usc_new, vsc_new, I, J)
                for _ in range(max_iter_backtracking):
                    if objval_new <= objval + grad_usc @ incr_usc + incr_usc.dot(incr_usc) / (
                            2.0 * current_step_size_u):
                        # Step size found.
                        break
                    else:
                        # Backtracking, reduce step size.
                        current_step_size_u *= backtracking_factor
                        usc_new = self._projection(usc - current_step_size_u * grad_usc)
                        incr_usc = usc_new - usc
                        objval_new = self.objective(usc_new, vsc_new, I, J)
                else:
                    warnings.warn("Maximum number of line-search iterations reached")

            usc = usc_new

            current_step_size_v = step_size
            _, grad_vsc = self.grad_objective(usc, vsc, I, J)

            vsc_new = self._projection(vsc - current_step_size_v * grad_vsc)

            incr_vsc = vsc_new - vsc

            if backtracking:
                objval = self.objective(usc, vsc, I, J)
                objval_new = self.objective(usc_new, vsc_new, I, J)
                for _ in range(max_iter_backtracking):
                    if objval_new <= objval + grad_vsc @ incr_vsc + incr_vsc.dot(incr_vsc) / (
                            2.0 * current_step_size_v):
                        # Step size found.
                        break
                    else:
                        # Backtracking, reduce step size.
                        current_step_size_v *= backtracking_factor
                        vsc_new = self._projection(vsc - current_step_size_v * grad_vsc)
                        incr_vsc = vsc_new - vsc
                        objval_new = self.objective(usc_new, vsc_new, I, J)
                else:
                    warnings.warn("Maximum number of line-search iterations reached")

            vsc = vsc_new
            certificate = linalg.norm(np.hstack([usc, vsc]) - np.hstack([usc_new, vsc_new]) / step_size)

            if trace:
                trace_usc.append(usc.copy())
                trace_vsc.append(vsc.copy())
                trace_obj.append(self.objective(usc, vsc, I, J))
                trace_time.append((datetime.now() - start_time).total_seconds())

            if verbose:
                if (k % 20 == 0):
                    print('iteration= {},\t objective= {:3f}'.format(k, objval))

            if certificate < tol:
                print("Achieved relative tolerance at iteration %s" % k)
                success = True
                break

        if k >= max_iter:
            warnings.warn("projected_gradient did not reach the desired tolerance level", RuntimeWarning)

        usc_sol = np.full(n, self.epsilon)
        vsc_sol = np.full(m, self.epsilon)
        usc_sol[I] = usc
        vsc_sol[J] = vsc

        I_usc = np.where(usc_sol > self.epsilon)[0].tolist()
        J_vsc = np.where(vsc_sol > self.epsilon)[0].tolist()

        return optimize.OptimizeResult(usc=usc_sol, vsc=vsc_sol,
                                       I_usc=I_usc, J_vsc=J_vsc,
                                       success=success, certificate=certificate, nb_iter=k,
                                       trace_usc=np.array(trace_usc), trace_vsc=np.array(trace_vsc),
                                       trace_obj=np.array(trace_obj), trace_time=trace_time)

    def acc_proj_grad(self, usc0, vsc0, I, J, max_iter=100, tol=1e-10, step_size=1e-3,
                           backtracking=True, backtracking_factor=0.5, max_iter_backtracking=20,
                           trace=False, verbose=False) -> optimize.OptimizeResult:

        (n, m) = self.M.shape
        if not max_iter_backtracking > 0:
            raise ValueError('Backtracking iterations need to be greater than 0')

        if step_size is None:
            step_size = 1.

        usc = usc0[I]
        vsc = vsc0[J]

        usc_aux = usc.copy()
        vsc_aux = vsc.copy()

        usc_prev = usc.copy()
        vsc_prev = vsc.copy()

        success = False
        certificate = np.inf
        trace_usc = []
        trace_vsc = []
        trace_obj = []
        trace_time = []
        start_time = datetime.now()

        t_k = 1.

        for k in range(max_iter):
        # for k in tqdm(range(max_iter)):
            # Compute gradient and step size.

            current_step_size = step_size

            grad_usc, grad_vsc = self.grad_objective(usc_aux, vsc_aux, I, J)

            usc = self._projection(usc_aux - current_step_size * grad_usc)
            vsc = self._projection(vsc_aux - current_step_size * grad_vsc)

            if backtracking:
                for _ in range(max_iter_backtracking):

                    incr_usc = usc - usc_aux
                    incr_vsc = vsc - vsc_aux

                    objval = self.objective(usc, vsc, I, J)
                    objval_aux = self.objective(usc_aux, vsc_aux, I, J)
                    if objval <= objval_aux + np.hstack([grad_usc, grad_vsc]).dot(np.hstack([incr_usc, incr_vsc])) \
                            + np.hstack([incr_usc, incr_vsc]).dot((np.hstack([incr_usc, incr_vsc]))) / (
                            2.0 * current_step_size):
                        # Step size found.
                        break
                    else:
                        # Backtracking, reduce step size.
                        current_step_size *= backtracking_factor
                        usc = self._projection(usc_aux[I] - current_step_size * grad_usc)
                        vsc = self._projection(vsc_aux[J] - current_step_size * grad_vsc)
                else:
                    warnings.warn("Maxium number of line-search iterations reached")

            t_new = (1 + np.sqrt(1 + 4 * t_k ** 2)) / 2
            usc_aux = usc + ((t_k - 1.) / t_new) * (usc - usc_prev)
            vsc_aux = vsc + ((t_k - 1.) / t_new) * (vsc - vsc_prev)

            certificate = np.linalg.norm(np.hstack([usc - usc_prev, vsc - vsc_prev]) / step_size)
            t_k = t_new
            usc_prev = usc.copy()
            vsc_prev = vsc.copy()

            if trace:
                trace_usc.append(usc.copy())
                trace_vsc.append(vsc.copy())
                trace_obj.append(self.objective(usc_aux, vsc_aux, I, J))
                trace_time.append((datetime.now() - start_time).total_seconds())

            if verbose:
                print("iteration %s, step size: %s" % (k, step_size))

            if certificate < tol:
                print("Achieved relative tolerance at iteration %s" % k)
                success = True
                break

        if k >= max_iter:
            warnings.warn("projected_gradient did not reach the desired tolerance level", RuntimeWarning)

        usc_sol = np.full(n, self.epsilon)
        vsc_sol = np.full(m, self.epsilon)
        usc_sol[I] = usc_aux
        vsc_sol[J] = vsc_aux

        I_usc = np.where(usc_sol > self.epsilon)[0].tolist()
        J_vsc = np.where(vsc_sol > self.epsilon)[0].tolist()

        return optimize.OptimizeResult(usc=usc_sol, vsc=vsc_sol,
                                       I_usc=I_usc, J_vsc=J_vsc,
                                       success=success, certificate=certificate, nb_iter=k,
                                       trace_usc=np.array(trace_usc), trace_vsc=np.array(trace_vsc),
                                       trace_obj=np.array(trace_obj), trace_time=trace_time)

    def bloc_acc_proj_grad(self, usc_0, vsc_0, I, J, max_iter=1000, tol=1e-9, step_size=1e-3,
                                 backtracking=True, backtracking_factor=0.5, max_iter_backtracking=20,
                                 trace=False, verbose=False) -> optimize.OptimizeResult:

        (n, m) = self.M.shape

        if not max_iter_backtracking > 0:
            raise ValueError('Backtracking iterations need to be greater than 0')

        if step_size is None:
            step_size = 10.

        usc = usc_0[I]
        vsc = vsc_0[J]

        usc_aux = usc.copy()
        vsc_aux = vsc.copy()

        usc_prev = usc.copy()
        vsc_prev = vsc.copy()

        success = False
        certificate_u = np.inf
        certificate_v = np.inf
        trace_usc = []
        trace_vsc = []
        trace_obj = []
        trace_time = []
        start_time = datetime.now()

        t_k_u = 1.
        t_k_v = 1.

        step_size_u = step_size
        step_size_v = step_size

        for k in range(max_iter):
        # for k in tqdm(range(max_iter)):

            # Compute gradient and step size.
            current_step_size_u = step_size_u
            grad_usc_aux, _ = self.grad_objective(usc_aux, vsc_aux, I, J)

            usc = self._projection(usc_aux - current_step_size_u * grad_usc_aux)

            if backtracking:
                for _ in range(max_iter_backtracking):

                    incr_usc = usc - usc_aux

                    objval = self.objective(usc, vsc, I, J)
                    objval_aux = self.objective(usc_aux, vsc_aux, I, J)
                    if objval <= objval_aux + grad_usc_aux @ incr_usc  \
                            + np.hstack([incr_usc]).dot((np.hstack([incr_usc]))) / (2.0 * current_step_size_u):
                        # Step size found.
                        break
                    else:
                        # Backtracking, reduce step size.
                        current_step_size_u *= backtracking_factor
                        usc = self._projection(usc_aux - current_step_size_u * grad_usc_aux)
                else:
                    warnings.warn("Maximum number of line-search iterations reached")

            t_new_u = (1 + np.sqrt(1 + 4 * t_k_u ** 2)) / 2
            usc_aux = usc + ((t_k_u - 1.) / t_new_u) * (usc - usc_prev)
            certificate_u = np.linalg.norm((usc - usc_prev) / step_size_u)
            t_k_u = t_new_u
            usc_prev = usc.copy()

            current_step_size_v = step_size_v
            _, grad_vsc_aux = self.grad_objective(usc_aux, vsc_aux, I, J)

            vsc = self._projection(vsc_aux - current_step_size_v * grad_vsc_aux)

            if backtracking:
                for _ in range(max_iter_backtracking):
                    incr_vsc = vsc - vsc_aux

                    objval = self.objective(usc, vsc, I, J)
                    objval_aux = self.objective(usc_aux, vsc_aux, I, J)
                    if objval <= objval_aux + grad_vsc_aux @ incr_vsc + incr_vsc.dot(incr_vsc) / (
                            2.0 * current_step_size_v):
                        # Step size found.
                        break
                    else:
                        # Backtracking, reduce step size.
                        current_step_size_v *= backtracking_factor

                        vsc = self._projection(vsc_aux - current_step_size_v * grad_vsc_aux)
                else:
                    warnings.warn("Maximum number of line-search iterations reached")

            t_new_v = (1 + np.sqrt(1 + 4 * t_k_v ** 2)) / 2
            vsc_aux = vsc + ((t_k_v - 1.) / t_new_v) * (vsc - vsc_prev)
            certificate_v = np.linalg.norm((vsc - vsc_prev) / step_size_v)
            t_k_v = t_new_v
            vsc_prev = vsc.copy()

            if trace:
                trace_usc.append(usc.copy())
                trace_vsc.append(vsc.copy())
                trace_obj.append(self.objective(usc_aux, vsc_aux, I, J))
                trace_time.append((datetime.now() - start_time).total_seconds())

            if verbose:
                print("iteration %s, certificate_u: %s, certificate_v: %s, step size_u: %s, step size_v: %s"\
                      % (k, certificate_u, certificate_v, step_size_u, step_size_v))

            if max(certificate_u, certificate_v) < tol:
                print("Achieved relative tolerance at iteration %s" % k)
                success = True
                break

        if k >= max_iter:
            warnings.warn("projected_gradient did not reach the desired tolerance level", RuntimeWarning)

        usc_sol = np.full(n, self.epsilon)
        vsc_sol = np.full(m, self.epsilon)
        usc_sol[I] = usc_aux
        vsc_sol[J] = vsc_aux

        I_usc = np.where(usc_sol > self.epsilon)[0].tolist()
        J_vsc = np.where(vsc_sol > self.epsilon)[0].tolist()


        return optimize.OptimizeResult(usc=usc_sol, vsc=vsc_sol,
                                       I_usc=I_usc, J_vsc=J_vsc,
                                       success=success, certificate=(certificate_u, certificate_v), nb_iter=k,
                                       trace_usc=np.array(trace_usc), trace_vsc=np.array(trace_vsc),
                                       trace_obj=np.array(trace_obj), trace_time=trace_time)

    def restricted_sinkhorn(self, usc, vsc, I, J, K, max_iter=1000, tol=1e-4) -> optimize.OptimizeResult:

        (n, m) = self.M.shape

        # K = np.empty_like(self.M)
        # np.divide(self.M, - self.reg, out=K)
        # np.exp(K, out=K)

        Ic = self._complementary(I, n)
        Jc = self._complementary(J, m)

        # submarginals mu_{I}, nu_{J}, mu_{I^c}, nu_{J^c}
        a_I = self._subvector(I, self.a)
        b_J = self._subvector(J, self.b)

        # submatrices K_{IJ}, K_{IJ^c}, K_{I^cJ}, K_{I^cJ^c}
        K_IJ = self._submatrix(K, I, J)
        K_IJc = self._submatrix(K, I, Jc)
        K_IcJ = self._submatrix(K, Ic, J)
        K_IJ_p = (1 / a_I).reshape(-1, 1) * K_IJ

        cst_u = np.divide(self.epsilon * K_IJc.sum(axis=1), a_I)
        cst_v = self.epsilon * K_IcJ.sum(axis=0)

        tmp = np.empty(b_J.shape, dtype=self.M.dtype)
        certificate = np.inf
        cpt = 1

        # usc = usc0[I].copy()
        # vsc = vsc0[J].copy()

        while (cpt < max_iter):

            K_IJ_transpose_u = K_IJ.T @ usc + cst_v
            vsc = np.divide(b_J, K_IJ_transpose_u)
            KIJ_v = K_IJ_p @ vsc + cst_u
            usc = np.divide(1., KIJ_v)

            usc_prev = usc
            vsc_prev = vsc
            if (np.any(K_IJ_transpose_u == 0.) or \
                    np.any(np.isnan(usc)) or np.any(np.isnan(vsc)) or \
                    np.any(np.isinf(usc)) or np.any(np.isinf(vsc))):
                # we have reached the machine precision
                # come back to previous solution and quit loop
                print('Warning: numerical errors at iteration', cpt)
                usc = usc_prev
                vsc = vsc_prev
                break

            if cpt % 10 == 0:
                np.einsum('i,ij,j->j', usc, K_IJ, vsc, out=tmp)
                certificate = np.linalg.norm(tmp - b_J)**2  # violation of marginal
                # print('certificate: %s' %certificate)

            if certificate < tol:
                print("Certificate is achieved")
                break

            cpt += 1

        usc = self._projection(usc)
        vsc = self._projection(vsc)
        # usc_full = np.full(n, self.epsilon)
        # vsc_full = np.full(m, self.epsilon)
        # usc_full[I] = usc
        # vsc_full[J] = vsc
        # Psc = usc_full.reshape((-1, 1)) * K * vsc_full.reshape((1, -1))

        return optimize.OptimizeResult(usc=usc, vsc=vsc, Psc=None)

    def restricted_greenkhorn(self, usc0, vsc0, I, J, max_iter=1000, tol=1e-6) -> optimize.OptimizeResult:

        (n, m) = self.M.shape

        K = np.empty_like(self.M)
        np.divide(self.M, - self.reg, out=K)
        np.exp(K, out=K)

        Ic = self._complementary(I, n)
        Jc = self._complementary(J, m)

        # submarginals mu_{I}, nu_{J}, mu_{I^c}, nu_{J^c}
        a_I = self._subvector(I, self.a)
        b_J = self._subvector(J, self.b)

        # submatrices K_{IJ}, K_{IJ^c}, K_{I^cJ}, K_{I^cJ^c}
        K_IJ = self._submatrix(K, I, J)
        K_IJc = self._submatrix(K, I, Jc)
        K_IcJ = self._submatrix(K, Ic, J)

        cst_u = np.divide(self.epsilon * K_IJc.sum(axis=1), a_I)
        cst_v = self.epsilon * K_IcJ.sum(axis=0)
        usc = usc0[I].copy()
        vsc = vsc0[J].copy()

        G_IJ = usc[:, np.newaxis] * K_IJ * vsc[np.newaxis, :]

        viol_I = G_IJ.sum(1) - a_I
        viol_J = G_IJ.sum(0) - b_J
        stopThr_val = 1

        for i in range(max_iter):
            i_1 = np.argmax(np.abs(viol_I))
            i_2 = np.argmax(np.abs(viol_J))
            m_viol_I = np.abs(viol_I[i_1])
            m_viol_J = np.abs(viol_J[i_2])
            stopThr_val = np.maximum(m_viol_I, m_viol_J)

            if m_viol_I > m_viol_J:
                old_usc = usc[i_1]
                usc[i_1] = a_I[i_1] / (K_IJ[i_1, :].dot(vsc) + cst_u[i_1])
                G_IJ[i_1, :] = usc[i_1] * K_IJ[i_1, :] * vsc

                viol_I[i_1] = usc[i_1] * K_IJ[i_1, :].dot(vsc) - a_I[i_1]
                viol_J += (K_IJ[i_1, :].T * (usc[i_1] - old_usc) * vsc)

            else:
                old_vsc = vsc[i_2]
                vsc[i_2] = b_J[i_2] / (K_IJ[:, i_2].T.dot(usc) + cst_v[i_2])
                G_IJ[:, i_2] = usc * K_IJ[:, i_2] * vsc[i_2]
                viol_I += (-old_vsc + vsc[i_2]) * K_IJ[:, i_2] * usc
                viol_J[i_2] = vsc[i_2] * K_IJ[:, i_2].dot(usc) - b_J[i_2]

            if stopThr_val <= tol:
                break
        else:
            print('Warning: Algorithm did not converge')

        usc = self._projection(usc)
        vsc = self._projection(vsc)
        usc_full = np.full(n, self.epsilon)
        vsc_full = np.full(m, self.epsilon)
        usc_full[I] = usc
        vsc_full[J] = vsc

        # G = usc_full[:, np.newaxis] * K * vsc_full[np.newaxis, :]

        return optimize.OptimizeResult(usc=usc_full, vsc=vsc_full, Psc=None)


    def _bfgspsot(self, theta, I, J, K):
        u = theta[:len(I)]
        v = theta[len(I):]
        # objective value
        f = self.objective(u, v, I, J, K)
        # gradient
        g_u, g_v = self.grad_objective(u, v, I, J, K)
        g = np.hstack([g_u, g_v])
        return f, g

    def lbfgsb(self, u0_I, v0_J, I, J, K):

        (n, m) = self.M.shape
        Ic = self._complementary(I, n)
        Jc = self._complementary(J, m)
        # K = np.empty_like(self.M)
        # np.divide(self.M, - self.reg, out=K)
        # np.exp(K, out=K)

        a_I = self._subvector(I, self.a)
        b_J = self._subvector(J, self.b)
        # M_IJ = self._submatrix(self.M, I, J)

        # u0_I = self._subvector(I, u0)
        # v0_J = self._subvector(J, v0)

        # --------------------------------- initial point for L-BFGS-B
        # ---------------------------------

        time_start = time()
        res_sink = self.restricted_sinkhorn(u0_I, v0_J, I, J, K, max_iter=10, tol=1e-05)
        u0 = res_sink['usc']
        v0 = res_sink['vsc']
        time_end = time() - time_start
        print("Time spending during the restricted Skinkhorn is %s" % time_end)

        # The following initilaization takes much time than the restricted sinkhorn (20 x)
        # time_start = time()
        # P_sink = sinkhorn(a_I, b_J, M_IJ, reg=self.reg, log=True)
        # outputs_dict = P_sink[1]
        # u0 = outputs_dict['u']
        # v0 = outputs_dict['v']
        # theta0 = np.hstack([u0, v0])
        # time_end = time() - time_start
        # print("Time spending during restricted Skinkhorn is %s" % time_end)

        # The following initilaization takes much time than the restricted sinkhorn (35 x)
        # u_I = self._subvector(I, u0)
        # v_J = self._subvector(J, v0)
        # theta0 = np.hstack([u0_I, v0_J])

        # params of bfgs
        theta0 = np.hstack([u0, v0])
        maxiter = 1000  # max number of iterations
        maxfun = 1000  # max  number of function evaluations
        pgtol = 1e-09  # final objective function accuracy
        param_m = 2  # stored gradients
        factr = 1e5  # tolerance of constraints function

        func = lambda theta: self._bfgspsot(theta, I, J, K)


        bounds_u = [(a_I.min() / (self.epsilon * len(Jc) + len(J) * (b_J.max() / (self.epsilon * n * K.min()))),\
                     a_I.max() /(self.epsilon * m * K.min()))] * len(I)

        bounds_v = [(b_J.min() / (self.epsilon * len(Ic) + len(I) * (a_I.max() / (self.epsilon * m * K.min()))), \
                     b_J.max() / (self.epsilon * n * K.min()))] * len(J)

        # bounds_u = [(self.epsilon, max(self.a / (self.epsilon * K.sum(axis=1))))] * len(I)
        # bounds_v = [(self.epsilon, max(self.b / (self.epsilon * K.T.sum(axis=1))))] * len(J)
        bounds = bounds_u + bounds_v

        theta, obj, d = fmin_l_bfgs_b(func=func,
                                      x0=theta0,
                                      fprime=None,
                                      args=(),
                                      approx_grad=False,
                                      bounds=bounds,
                                      m=param_m,
                                      factr=factr,
                                      pgtol=pgtol,
                                      epsilon=1e-08,
                                      iprint=-1,
                                      maxfun=maxfun,
                                      maxiter=maxiter,
                                      disp=None,
                                      callback=None
                                      )

        usc = theta[:len(I)]
        vsc = theta[len(I):]

        usc_full = np.full(n, self.epsilon)
        vsc_full = np.full(m, self.epsilon)
        usc_full[I] = usc
        vsc_full[J] = vsc

        return usc_full, vsc_full, obj, d