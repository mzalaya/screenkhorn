# -*- coding: utf-8 -*-
"""
    screenkhorn: solver for screening Sinkhorn via dual projection
"""
__author__ = 'Mokhtar Z. Alaya'

import numpy as np
from scipy import optimize, linalg
from ot import sinkhorn
import cvxpy as cvx
from numba import jit
import time
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
        self.reg = reg# astype(np.float64)
        self.epsilon = epsilon# .astype(np.float64)

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

    @staticmethod
    def _init_history():
        """ Initialises a history data structure for tracking values"""
        history = {}
        history["objval"] = list()
        return history

    # @jit(nopython=True)
    def objective(self, u_param, v_param, I, J):
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
        K = np.empty_like(self.M)
        np.divide(self.M, - self.reg, out=K)
        np.exp(K, out=K)
        # K = np.exp(- self.M / self.reg)
        
        Ic = self._complementary(I, n)
        Jc = self._complementary(J, m)

        # subparameters u_I, v_J
        u_param_I = self._subvector(I, u_param)
        v_param_J = self._subvector(J, v_param)

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

        part_IJ = np.exp(u_param_I).T @ K_IJ @ np.exp(v_param_J) - u_param_I.T @ a_I - v_param_J.T @ b_J
        part_IJc = self.epsilon * np.exp(u_param_I).T @ K_IJc @ np.ones(len(Jc))
        part_IcJ = self.epsilon * np.ones(len(Ic)).T @ K_IcJ @ np.exp(v_param_J)
        part_IcJc = self.epsilon**2 * K_IcJc.sum() - np.log(self.epsilon)*(a_Ic.sum() + b_Jc.sum())

        psi_epsilon = (part_IJ + part_IJc + part_IcJ + part_IcJc).sum()
        return psi_epsilon

    # @jit(nopython=True)
    def grad_objective(self, u_param, v_param, I, J):

        # Next 3 lines equivalent to K= np.exp(-M/reg), but faster to compute
        K = np.empty_like(self.M)
        np.divide(self.M, - self.reg, out=K)
        np.exp(K, out=K)
        # K = np.exp(- self.M / self.reg)

        (n, m) = self.M.shape

        Ic = self._complementary(I, n)
        Jc = self._complementary(J, m)

        # subparameters u_I, v_J
        u_param_I = self._subvector(I, u_param)
        v_param_J = self._subvector(J, v_param)

        # submarginals mu_{I}, nu_{J}
        a_I = self._subvector(I, self.a)
        b_J = self._subvector(J, self.b)

        # submatrices K_{IJ}, K_{IJ^c}, K_{I^cJ}
        K_IJ = self._submatrix(K, I, J)
        K_IJc = self._submatrix(K, I, Jc)
        K_IcJ = self._submatrix(K, Ic, J)

        # gradients of Psi_epsilon w. r. t. u and v
        grad_u = np.exp(u_param_I) * (K_IJ @ np.exp(v_param_J) + self.epsilon * K_IJc @ np.ones(len(Jc))) - a_I
        grad_v = np.exp(v_param_J) * (K_IJ.T @ np.exp(u_param_I) + self.epsilon * K_IcJ.T @ np.ones(len(Ic))) - b_J
        return grad_u, grad_v

    def _projection(self, u):
        u_proj = u.copy()
        u_proj[np.where(u_proj < np.log(self.epsilon))] = np.log(self.epsilon)
        return u_proj

    def _projection_cvx(self, u):
        '''
        Returns the point in the convex set
        C_epsilon = {u in R^n : exp(u) > epsilon}
        that is closest to y (according to Euclidian distance)
        '''
        d = len(u)
        u_proj = u.copy()
        # Setup the objective and constraints and solve the problem.
        x = cvx.Variable(shape=d)
        obj = cvx.Minimize(cvx.sum_squares(x - u_proj))
        constraints = [x >= cvx.log(self.epsilon)]
        prob = cvx.Problem(obj, constraints)
        prob.solve(solver=cvx.SCS, verbose=False)
        return np.array(x.value).squeeze()

    # @jit(nopython=True)
    def projected_grad(self, usc0=None, vsc0=None, I=None, J=None, max_iter=1000, tol=1e-9, step_size=None,
                       backtracking=False, backtracking_factor=0.5, max_iter_backtracking=10,
                       trace=True, verbose=False) -> optimize.OptimizeResult:

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
        Ic = self._complementary(I, n)
        Jc = self._complementary(J, m)

        if usc0 is None:
            usc = np.zeros(n)
        else:
            usc = np.array(usc0, copy=True)

        if vsc0 is None:
            vsc = np.zeros(m)
        else:
            vsc = np.array(vsc0, copy=True)

        if not max_iter_backtracking > 0:
            raise ValueError('Backtracking iterations need to be greater than 0')

        if step_size is None:
            # Sample to estimate Lipschitz constant.
            step_size_n_sample = 5
            L = []
            for _ in range(step_size_n_sample):
                usc_tmp = np.random.randn(n)
                vsc_tmp = np.random.randn(m)
                usc_tmp /= linalg.norm(usc_tmp)
                vsc_tmp /= linalg.norm(vsc_tmp)
                obj = self.objective(usc, vsc, I, J)
                obj_tmp = self.objective(usc_tmp, vsc_tmp, I, J)
                L.append(linalg.norm(obj - obj_tmp))
            # Give it a generous upper bound.
            step_size = 2. / (np.mean(L))
            print("Step_size in the begining is %s" % step_size)

        usc[Ic] = np.log(self.epsilon) * np.ones(len(Ic))
        vsc[Jc] = np.log(self.epsilon) * np.ones(len(Jc))

        success = False
        trace_usc = []
        trace_vsc = []
        trace_obj = []
        trace_time = []
        start_time = datetime.now()

        usc_new = np.array(usc, copy=True)
        vsc_new = np.array(vsc, copy=True)

        for k in range(max_iter):
        # for k in tqdm(range(max_iter)):
            # Compute gradient and step size.

            current_step_size = step_size
            grad_usc, grad_vsc = self.grad_objective(usc, vsc, I, J)

            usc_new[I] = self._projection(usc[I] - current_step_size * grad_usc)
            vsc_new[J] = self._projection(vsc[J] - current_step_size * grad_vsc)

            incr_usc = usc_new[I] - usc[I]
            incr_vsc = vsc_new[J] - vsc[J]

            if backtracking:
                objval = self.objective(usc, vsc, I, J)
                objval_new = self.objective(usc_new, vsc_new, I, J)
                for _ in range(max_iter_backtracking):
                    if objval_new <= objval + np.hstack([grad_usc, grad_vsc])@ np.hstack([incr_usc, incr_vsc])\
                       + np.hstack([incr_usc, incr_vsc]).dot((np.hstack([incr_usc, incr_vsc]))) / (2.0 * current_step_size):
                        # Step size found.
                        break
                    else:
                        # Backtracking, reduce step size.
                        current_step_size *= backtracking_factor
                        usc_new[I] = self._projection(usc[I] - current_step_size * grad_usc)
                        vsc_new[J] = self._projection(vsc[J] - current_step_size * grad_vsc)

                        incr_usc = usc_new[I] - usc[I]
                        incr_vsc = vsc_new[J] - vsc[J]
                        objval_new = self.objective(usc_new, vsc_new, I, J)
                else:
                    warnings.warn("Maxium number of line-search iterations reached.")
            certificate = linalg.norm(np.hstack([usc - usc_new, vsc - vsc_new]) / step_size)
            # certificate = linalg.norm(np.hstack([grad_usc, grad_vsc]))
            # certificate = (objval_new - objval) / objval
            usc[I] = usc_new[I]
            vsc[J] = vsc_new[J]

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

        # Checking stability of the active sets
        I_usc = np.where(usc > np.log(self.epsilon))[0].tolist()
        J_vsc = np.where(vsc > np.log(self.epsilon))[0].tolist()

        return optimize.OptimizeResult(usc=usc, vsc=vsc,
                                       I_usc=I_usc, J_vsc=J_vsc,
                                       success=success, certificate=certificate, nb_iter=k,
                                       trace_usc=np.array(trace_usc), trace_vsc=np.array(trace_vsc),
                                       trace_obj=np.array(trace_obj), trace_time=trace_time)

    # @jit(nopython=True)
    def block_projected_grad(self, usc0=None, vsc0=None, I=None, J=None, max_iter=100, tol=1e-10,
                                      step_size=None,
                                      backtracking=True, backtracking_factor=0.5, max_iter_backtracking=10,
                                      trace=True, verbose=False) -> optimize.OptimizeResult:
        (n, m) = self.M.shape
        Ic = self._complementary(I, n)
        Jc = self._complementary(J, m)

        if usc0 is None:
            usc = np.zeros(n)
        else:
            usc = np.array(usc0, copy=True)

        if vsc0 is None:
            vsc = np.zeros(m)
        else:
            vsc = np.array(vsc0, copy=True)

        if not max_iter_backtracking > 0:
            raise ValueError('Backtracking iterations need to be greater than 0.')

        if step_size is None:
            # Sample to estimate Lipschitz constant.
            step_size_n_sample = 5
            L = []
            for _ in range(step_size_n_sample):
                usc_tmp = np.random.randn(n)
                vsc_tmp = np.random.randn(m)
                usc_tmp /= linalg.norm(usc_tmp)
                vsc_tmp /= linalg.norm(vsc_tmp)
                obj = self.objective(usc, vsc, I, J)
                obj_tmp = self.objective(usc_tmp, vsc_tmp, I, J)
                L.append(linalg.norm(obj - obj_tmp))
            # Give it a generous upper bound.
            step_size = 1. / (np.mean(L))
            print("Step_size in the begining is %s" % step_size)

        usc[Ic] = np.log(self.epsilon) * np.ones(len(Ic))
        vsc[Jc] = np.log(self.epsilon) * np.ones(len(Jc))

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

            usc_new[I] = self._projection(usc[I] - current_step_size_u * grad_usc)

            incr_usc = usc_new[I] - usc[I]

            if backtracking:
                objval = self.objective(usc, vsc, I, J)
                objval_new = self.objective(usc_new, vsc_new, I, J)
                for _ in range(max_iter_backtracking):
                    if objval_new <= objval + grad_usc @ incr_usc + incr_usc.dot(incr_usc) / (2.0 * current_step_size_u):
                        # Step size found.
                        break
                    else:
                        # Backtracking, reduce step size.
                        current_step_size_u *= backtracking_factor
                        usc_new[I] = self._projection(usc[I] - current_step_size_u * grad_usc)
                        incr_usc = usc_new[I] - usc[I]
                        objval_new = self.objective(usc_new, vsc_new, I, J)
                else:
                    warnings.warn("Maxium number of line-search iterations reached")

            usc = usc_new

            current_step_size_v = step_size
            _, grad_vsc = self.grad_objective(usc, vsc, I, J)

            vsc_new[J] = self._projection(vsc[J] - current_step_size_v * grad_vsc)

            incr_vsc = vsc_new[J] - vsc[J]

            if backtracking:
                objval = self.objective(usc, vsc, I, J)
                objval_new = self.objective(usc_new, vsc_new, I, J)
                for _ in range(max_iter_backtracking):
                    if objval_new <= objval + grad_vsc @ incr_vsc + incr_vsc.dot(incr_vsc) / (2.0 * current_step_size_v):
                        # Step size found.
                        break
                    else:
                        # Backtracking, reduce step size.
                        current_step_size_v *= backtracking_factor
                        vsc_new[J] = self._projection(vsc[J] - current_step_size_v * grad_vsc)
                        incr_vsc = vsc_new[J] - vsc[J]
                        objval_new = self.objective(usc_new, vsc_new, I, J)
                else:
                    warnings.warn("Maxium number of line-search iterations reached")

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

        I_usc = np.where(usc > np.log(self.epsilon))[0].tolist()
        J_vsc = np.where(vsc > np.log(self.epsilon))[0].tolist()

        return optimize.OptimizeResult(usc=usc, vsc=vsc,
                                       I_usc=I_usc, J_vsc=J_vsc,
                                       success=success, certificate=certificate, nb_iter=k,
                                       trace_usc=np.array(trace_usc), trace_vsc=np.array(trace_vsc),
                                       trace_obj=np.array(trace_obj), trace_time=trace_time)

    #@jit(nopython=True)
    def acc_projected_grad(self, usc0, vsc0, I=None, J=None, max_iter=100, tol=1e-10, step_size=None,
                                  backtracking=True, backtracking_factor=0.5, max_iter_backtracking=10,
                                  trace=True, verbose=False) -> optimize.OptimizeResult:

        (n, m) = self.M.shape
        Ic = self._complementary(I, n)
        Jc = self._complementary(J, m)

        if not max_iter_backtracking > 0:
            raise ValueError('Backtracking iterations need to be greater than 0')

        if step_size is None:
            step_size = 1.

        usc = usc0
        vsc = vsc0

        if step_size is None:
            # Sample to estimate Lipschitz constant.
            step_size_n_sample = 5
            L = []
            for _ in range(step_size_n_sample):
                usc_tmp = np.random.randn(n)
                vsc_tmp = np.random.randn(m)
                usc_tmp /= linalg.norm(usc_tmp)
                vsc_tmp /= linalg.norm(vsc_tmp)
                obj = self.objective(usc, vsc, I, J)
                obj_tmp = self.objective(usc_tmp, vsc_tmp, I, J)
                L.append(linalg.norm(obj - obj_tmp))
            # Give it a generous upper bound.
            step_size = 2. / (np.mean(L))
            print("Step_size in the begining is %s" % step_size)

        usc[Ic] = np.full(len(Ic), np.log(self.epsilon))
        vsc[Jc] = np.full(len(Jc), np.log(self.epsilon))

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

            usc[I] = self._projection(usc_aux[I] - current_step_size * grad_usc)
            vsc[J] = self._projection(vsc_aux[J] - current_step_size * grad_vsc)

            if backtracking:
                for _ in range(max_iter_backtracking):

                    incr_usc = usc[I] - usc_aux[I]
                    incr_vsc = vsc[J] - vsc_aux[J]

                    objval = self.objective(usc, vsc, I, J)
                    objval_aux = self.objective(usc_aux, vsc_aux, I, J)
                    if objval <= objval_aux + np.hstack([grad_usc, grad_vsc]).dot(np.hstack([incr_usc, incr_vsc]))\
                       + np.hstack([incr_usc, incr_vsc]).dot((np.hstack([incr_usc, incr_vsc]))) / (2.0 * current_step_size):
                        # Step size found.
                        break
                    else:
                        # Backtracking, reduce step size.
                        current_step_size *= backtracking_factor

                        usc[I] = self._projection(usc_aux[I] - current_step_size * grad_usc)
                        vsc[J] = self._projection(vsc_aux[J] - current_step_size * grad_vsc)
                else:
                    warnings.warn("Maxium number of line-search iterations reached")

            t_new = (1 + np.sqrt(1 + 4 * t_k**2)) / 2
            usc_aux[I] = usc[I] + ((t_k - 1.) / t_new) * (usc[I] - usc_prev[I])
            vsc_aux[J] = vsc[J] + ((t_k - 1.) / t_new) * (vsc[J] - vsc_prev[J])

            certificate = np.linalg.norm(np.hstack([usc - usc_prev, vsc - vsc_prev]) / step_size)
            # delta_fval = (f_val - old_fval) / abs(f_val)
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

        if  k >= max_iter:
            warnings.warn("projected_gradient did not reach the desired tolerance level", RuntimeWarning)

        I_usc = np.where(usc > np.log(self.epsilon))[0].tolist()
        J_vsc = np.where(vsc > np.log(self.epsilon))[0].tolist()

        return optimize.OptimizeResult(usc=usc_aux, vsc=vsc_aux,
                                       I_usc=I_usc, J_vsc=J_vsc,
                                       success=success, certificate=certificate, nb_iter=k,
                                       trace_usc=np.array(trace_usc), trace_vsc=np.array(trace_vsc),
                                       trace_obj=np.array(trace_obj), trace_time=trace_time)

    # @jit(nopython=True)
    def block_acc_projected_grad(self, usc_0, vsc_0, I=None, J=None, max_iter=1000, tol=1e-19, step_size=10.,
                                  backtracking=True, backtracking_factor=0.5, max_iter_backtracking=30,
                                  trace=True, verbose=False) -> optimize.OptimizeResult:

        (n, m) = self.M.shape
        Ic = self._complementary(I, n)
        Jc = self._complementary(J, m)

        if not max_iter_backtracking > 0:
            raise ValueError('Backtracking iterations need to be greater than 0')

        if step_size is None:
            step_size = 1.

        usc = usc_0
        vsc = vsc_0

        usc[Ic] = np.log(self.epsilon) * np.ones(len(Ic))
        vsc[Jc] = np.log(self.epsilon) * np.ones(len(Jc))

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

            current_step_size_u = step_size
            grad_usc, _ = self.grad_objective(usc_aux, vsc_aux, I, J)

            usc[I] = self._projection(usc_aux[I] - current_step_size_u * grad_usc)

            if backtracking:
                for _ in range(max_iter_backtracking):

                    incr_usc = usc[I] - usc_aux[I]

                    objval = self.objective(usc, vsc, I, J)
                    objval_aux = self.objective(usc_aux, vsc_aux, I, J)
                    if objval <= objval_aux + np.hstack([grad_usc]).dot(np.hstack([incr_usc]))\
                       + np.hstack([incr_usc]).dot((np.hstack([incr_usc]))) / (2.0 * current_step_size_u):
                        # Step size found.
                        break
                    else:
                        # Backtracking, reduce step size.
                        current_step_size_u *= backtracking_factor

                        usc[I] = self._projection(usc_aux[I] - current_step_size_u * grad_usc)
                else:
                    warnings.warn("Maxium number of line-search iterations reached")

            t_new = (1 + np.sqrt(1 + 4 * t_k ** 2)) / 2
            usc_aux[I] = usc[I] + ((t_k - 1.) / t_new) * (usc[I] - usc_prev[I])
            usc_prev = usc.copy()

            current_step_size_v = step_size
            _, grad_vsc = self.grad_objective(usc_aux, vsc_aux, I, J)

            vsc[J] = self._projection(vsc_aux[J] - current_step_size_v * grad_vsc)

            if backtracking:
                for _ in range(max_iter_backtracking):
                    incr_vsc = vsc[J] - vsc_aux[J]

                    objval = self.objective(usc, vsc, I, J)
                    objval_aux = self.objective(usc_aux, vsc_aux, I, J)
                    if objval <= objval_aux + grad_vsc.dot(incr_vsc) + incr_vsc.dot(incr_vsc) / (2.0 * current_step_size_v):
                        # Step size found.
                        break
                    else:
                        # Backtracking, reduce step size.
                        current_step_size_v *= backtracking_factor

                        vsc[J] = self._projection(vsc_aux[J] - current_step_size_v * grad_vsc)


                else:
                    warnings.warn("Maxium number of line-search iterations reached")

            vsc_aux[J] = vsc[J] + ((t_k - 1.) / t_new) * (vsc[J] - vsc_prev[J])

            certificate = np.linalg.norm(np.hstack([usc - usc_prev, vsc - vsc_prev]) / step_size)

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

        if  k >= max_iter:
            warnings.warn("projected_gradient did not reach the desired tolerance level", RuntimeWarning)

        I_usc = np.where(usc > np.log(self.epsilon))[0].tolist()
        J_vsc = np.where(vsc > np.log(self.epsilon))[0].tolist()

        return optimize.OptimizeResult(usc=usc_aux, vsc=vsc_aux,
                                       I_usc=I_usc, J_vsc=J_vsc,
                                       success=success, certificate=certificate, nb_iter=k,
                                       trace_usc=np.array(trace_usc), trace_vsc=np.array(trace_vsc),
                                       trace_obj=np.array(trace_obj), trace_time=trace_time)

    # @jit(nopython=True)
    def _randomization(self, u, v, solver=None, max_iter=1, tol_marg=1e-10, **kwargs) -> optimize.OptimizeResult:
        """
        Randomization
        :param u:
        :param v:
        :return:
        """
        (n, m) = self.M.shape

        list_n = np.arange(n)
        list_n = list_n.tolist()

        list_m = np.arange(m)
        list_m = list_m.tolist()

        if solver is None:
            solver = self.projected_grad

        I = np.where(u > np.log(self.epsilon))[0].tolist()
        J = np.where(v > np.log(self.epsilon))[0].tolist()
        it = 1

        K = np.empty_like(self.M)
        np.divide(self.M, - self.reg, out=K)
        np.exp(K, out=K)

        # for k in tqdm(range(max_iter)):
        for k in range(max_iter):

            # Parameters of the solver.
            step_size_solver = kwargs.get('step_size_solver', 10.)
            max_iter_backtracking = kwargs.get('max_iter_backtracking', 30)
            max_iter_solver = kwargs.get('max_iter_solver', 1000)


            optim_res = solver(u, v, I, J,
                               step_size=step_size_solver,
                               max_iter_backtracking=max_iter_backtracking,
                               max_iter=max_iter_solver)
            u_new = optim_res["usc"]
            v_new = optim_res["vsc"]
            I_cst = optim_res["I_usc"]
            J_cst = optim_res["J_vsc"]

            P_sc = np.diag(np.exp(u_new)) @ K @ np.diag(np.exp(v_new))
            a_sc = P_sc @ np.ones(m)
            b_sc = P_sc.T @ np.ones(n)

            if abs(a_sc - self.a).all() < tol_marg and abs(b_sc - self.b).all() < tol_marg:
                print("Marginals are satisfied")
                break

            # Dual variables

            Ic = self._complementary(I, n)
            Jc = self._complementary(J, m)

            # submatrices K_{IJ}, K_{IJ^c}, K_{I^cJ}, K_{I^cJ^c}
            K_IJ = self._submatrix(K, I, J)
            K_IJc = self._submatrix(K, I, Jc)
            K_IcJ = self._submatrix(K, Ic, J)

            lambda_u = K @ np.exp(v_new) - self.a / np.exp(u_new)
            lambda_v = K.T @ np.exp(u_new) - self.b / np.exp(v_new)

            I_viol = [np.argmax(abs(lambda_u))]
            J_viol = [np.argmax(abs(lambda_v))]
            u = u_new
            v = v_new
            I = sorted(I_cst + I_viol)
            J = sorted(J_cst + J_viol)


            # if it == max_iter:
                # break
            it += 1

        return optimize.OptimizeResult(it=it, usc=u_new, vsc=v_new,
                                       I=I_cst, J=J_cst, lambda_u=lambda_u, lambda_v=lambda_v)

    def restricted_sinkhorn(self, usc0, vsc0, I, J, max_iter=1000, tol=1e-9) -> optimize.OptimizeResult:

        (n,m) = self.M.shape

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

        K_IJ_p = (1 / a_I).reshape(-1, 1) * K_IJ

        tmp = np.empty(b_J.shape, dtype=self.M.dtype)

        certificate = np.inf
        cpt = 1

        usc = usc0[I].copy()
        vsc = vsc0[J].copy()

        #while (certificate > tol and cpt < max_iter):
        for k in range(max_iter):

            usc_prev = usc
            vsc_prev = vsc

            K_IJ_transpose_u = K_IJ.T @ usc + cst_v
            vsc = np.divide(b_J, K_IJ_transpose_u)
            KIJ_v = K_IJ_p @ vsc + cst_u
            usc = np.divide(1., KIJ_v)

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
                certificate = np.linalg.norm(tmp - b_J) ** 2  # violation of marginal

        usc = np.exp(self._projection(np.log(usc)))
        vsc = np.exp(self._projection(np.log(vsc)))

        usc_full = np.full(n, self.epsilon)
        vsc_full = np.full(m, self.epsilon)
        usc_full[I] = usc
        vsc_full[J] = vsc
        Psc = usc_full.reshape((-1, 1)) * K * vsc_full.reshape((1, -1))

        return optimize.OptimizeResult(usc=usc_full, vsc=vsc_full, Psc=Psc)

    def restricted_greenkhorn(self, usc0, vsc0, I, J, max_iter=1000, tol=1e-9) -> optimize.OptimizeResult:

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

        tic = time.time()
        cst_u = np.divide(self.epsilon * K_IJc.sum(axis=1), a_I)
        cst_v = self.epsilon * K_IcJ.sum(axis=0)
        # toc = time.time() - tic
        # print("Time to calculate the constant vectors c_u and c_v is %s" %toc)
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

        usc_full = np.full(n, self.epsilon)
        vsc_full = np.full(m, self.epsilon)
        usc_full[I] = usc
        vsc_full[J] = vsc

        # G = usc_full[:, np.newaxis] * K * vsc_full[np.newaxis, :]

        return optimize.OptimizeResult(usc=usc_full, vsc=vsc_full) #, Psc=G)

