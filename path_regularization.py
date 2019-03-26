import numpy as np
from scipy import optimize, linalg
import pandas

import warnings
import logging
import time

from numba import jit

from tqdm import tqdm

import ot
from ot import sinkhorn, emd

from time import time
import sys
path_files = '/Users/mzalaya/PycharmProjects/OATMIL/oatmilrouen/'
sys.path.insert(0, path_files)
from screenkhorn.screenkhorn import Screenkhorn

__author__ = 'Mokhtar Z. Alaya'

# @jit(nopython=True)
def path_regularization_with_restricted_Sinkhorn(a, b, M, reg, u, v, epsilons, solver='acc_proj_grad', **kwargs):

    (n, m) = M.shape
    K = np.empty_like(M)
    np.divide(M, - reg, out=K)
    np.exp(K, out=K)
    trace_obj = []

    # Parameters of the solver.
    step_size_solver = kwargs.get('step_size_solver', 10.)
    backtracking_solver = kwargs.get('backtracking_solver', True)
    max_iter_backtracking_solver = kwargs.get('max_iter_backtracking', 30)
    max_iter_solver = kwargs.get('max_iter_solver', 1000)
    tol_solver = kwargs.get('tol_solver', 1e-9)

    if solver == 'acc_proj_grad':
        for epsilon in epsilons:
            screenkhorn = Screenkhorn(a, b, M, reg, epsilon)

            I_eps = np.where(a >= epsilon ** 2 * K.sum(axis=1))[0].tolist()
            J_eps = np.where(b >= epsilon ** 2 * K.T.sum(axis=1))[0].tolist()

            Ic_eps = screenkhorn._complementary(I_eps, n)
            Jc_eps = screenkhorn._complementary(J_eps, m)

            time_start = time()

            P_sink = sinkhorn(a[I_eps], b[J_eps], M[np.ix_(I_eps, J_eps)], reg=reg, log=True)
            outputs_dict = P_sink[1]
            exp_u_star = outputs_dict['u']
            exp_v_star = outputs_dict['v']

            time_end = time()- time_start
            print("Time spending during restricted Skinkhorn is %s" %time_end)

            u = np.zeros(n)
            v = np.zeros(m)
            u[I_eps] = np.log(exp_u_star)
            v[J_eps] = np.log(exp_v_star)
            u[Ic_eps] = np.log(epsilon)
            v[Jc_eps] = np.log(epsilon)

            sol_eps = screenkhorn.acc_projected_grad(u, v, I_eps, J_eps,
                                                     backtracking=backtracking_solver,
                                                     max_iter_backtracking=max_iter_backtracking_solver,
                                                     step_size=step_size_solver,
                                                     max_iter=max_iter_solver,
                                                     tol=tol_solver,
                                                     verbose=False)

            u_new = sol_eps["usc"]
            v_new = sol_eps["vsc"]
            # trace_obj.append(screenkhorn.objective(u_new, v_new, I_eps, J_eps))
            u = u_new
            v = v_new

    elif solver == "proj_grad":
        for epsilon in epsilons:
            screenkhorn = Screenkhorn(a, b, M, reg, epsilon)

            I_eps = np.where(a >= epsilon ** 2 * K.sum(axis=1))[0].tolist()
            J_eps = np.where(b >= epsilon ** 2 * K.T.sum(axis=1))[0].tolist()

            Ic_eps = screenkhorn._complementary(I_eps, n)
            Jc_eps = screenkhorn._complementary(J_eps, m)

            P_sink = sinkhorn(a[I_eps], b[J_eps], M[np.ix_(I_eps, J_eps)], reg=reg, log=True)
            outputs_dict = P_sink[1]
            exp_u_star = outputs_dict['u']
            exp_v_star = outputs_dict['v']

            u = np.zeros(n)
            v = np.zeros(m)
            u[I_eps] = np.log(exp_u_star)
            v[J_eps] = np.log(exp_v_star)
            u[Ic_eps] = np.log(epsilon)
            v[Jc_eps] = np.log(epsilon)

            sol_eps = screenkhorn.projected_grad(u, v, I_eps, J_eps,
                                                 backtracking=backtracking_solver,
                                                 max_iter_backtracking=max_iter_backtracking_solver,
                                                 step_size=step_size_solver,
                                                 max_iter=max_iter_solver,
                                                 tol=tol_solver,
                                                 verbose=False)

            u_new = sol_eps["usc"]
            v_new = sol_eps["vsc"]
            # trace_obj.append(screenkhorn.objective(u_new, v_new, I_eps, J_eps))
            u = u_new
            v = v_new

    elif solver == 'block_acc_proj_grad':
        for epsilon in epsilons:
            screenkhorn = Screenkhorn(a, b, M, reg, epsilon)

            I_eps = np.where(a >= epsilon ** 2 * K.sum(axis=1))[0].tolist()
            J_eps = np.where(b >= epsilon ** 2 * K.T.sum(axis=1))[0].tolist()

            Ic_eps = screenkhorn._complementary(I_eps, n)
            Jc_eps = screenkhorn._complementary(J_eps, m)

            P_sink = sinkhorn(a[I_eps], b[J_eps], M[np.ix_(I_eps, J_eps)], reg=reg, log=True)
            outputs_dict = P_sink[1]
            exp_u_star = outputs_dict['u']
            exp_v_star = outputs_dict['v']

            u = np.zeros(n)
            v = np.zeros(m)
            u[I_eps] = np.log(exp_u_star)
            v[J_eps] = np.log(exp_v_star)
            u[Ic_eps] = np.log(epsilon)
            v[Jc_eps] = np.log(epsilon)

            sol_eps = screenkhorn.block_acc_projected_grad(u, v, I_eps, J_eps,
                                                           backtracking=backtracking_solver,
                                                           max_iter_backtracking=max_iter_backtracking_solver,
                                                           step_size=step_size_solver,
                                                           max_iter=max_iter_solver,
                                                           tol=tol_solver,
                                                           verbose=False)

            u_new = sol_eps["usc"]
            v_new = sol_eps["vsc"]
            # trace_obj.append(screenkhorn.objective(u_new, v_new, I_eps, J_eps))
            u = u_new
            v = v_new

    else:
        for epsilon in epsilons:
            screenkhorn = Screenkhorn(a, b, M, reg, epsilon)

            I_eps = np.where(a >= epsilon ** 2 * K.sum(axis=1))[0].tolist()
            J_eps = np.where(b >= epsilon ** 2 * K.T.sum(axis=1))[0].tolist()

            Ic_eps = screenkhorn._complementary(I_eps, n)
            Jc_eps = screenkhorn._complementary(J_eps, m)

            P_sink = sinkhorn(a[I_eps], b[J_eps], M[np.ix_(I_eps, J_eps)], reg=reg, log=True)
            outputs_dict = P_sink[1]
            exp_u_star = outputs_dict['u']
            exp_v_star = outputs_dict['v']

            u = np.zeros(n)
            v = np.zeros(m)
            u[I_eps] = np.log(exp_u_star)
            v[J_eps] = np.log(exp_v_star)
            u[Ic_eps] = np.log(epsilon)
            v[Jc_eps] = np.log(epsilon)

            sol_eps = screenkhorn.block_projected_grad(u, v, I_eps, J_eps,
                                                       backtracking=backtracking_solver,
                                                       max_iter_backtracking=max_iter_backtracking_solver,
                                                       step_size=step_size_solver,
                                                       max_iter=max_iter_solver,
                                                       tol=tol_solver,
                                                       verbose=False)

            u_new = sol_eps["usc"]
            v_new = sol_eps["vsc"]
            # trace_obj.append(screenkhorn.objective(u_new, v_new, I_eps, J_eps))
            u = u_new
            v = v_new


    return u, v

def path_regularization(a, b, M, reg, u, v, epsilons, solver='acc_proj_grad', **kwargs):

    (n, m) = M.shape
    K = np.empty_like(M)
    np.divide(M, - reg, out=K)
    np.exp(K, out=K)
    trace_obj = []

    # Parameters of the solver.
    step_size_solver = kwargs.get('step_size_solver', 10.)
    backtracking_solver = kwargs.get('backtracking_solver', True)
    max_iter_backtracking_solver = kwargs.get('max_iter_backtracking', 30)
    max_iter_solver = kwargs.get('max_iter_solver', 1000)
    tol_solver = kwargs.get('tol_solver', 1e-9)

    if solver == 'acc_proj_grad':
        for epsilon in epsilons:
            screenkhorn = Screenkhorn(a, b, M, reg, epsilon)

            I_eps = np.where(a >= epsilon ** 2 * K.sum(axis=1))[0].tolist()
            J_eps = np.where(b >= epsilon ** 2 * K.T.sum(axis=1))[0].tolist()

            # Ic_eps = screenkhorn._complementary(I_eps, n)
            # Jc_eps = screenkhorn._complementary(J_eps, m)
            #
            # time_start = time()
            #
            # P_sink = sinkhorn(a[I_eps], b[J_eps], M[np.ix_(I_eps, J_eps)], reg=reg, log=True)
            # outputs_dict = P_sink[1]
            # exp_u_star = outputs_dict['u']
            # exp_v_star = outputs_dict['v']
            #
            # time_end = time()- time_start
            # print("Time spending duting restricted Skinkhorn is %s" %time_end)
            #
            # u = np.zeros(n)
            # v = np.zeros(m)
            # u[I_eps] = np.log(exp_u_star)
            # v[J_eps] = np.log(exp_v_star)
            # u[Ic_eps] = np.log(epsilon)
            # v[Jc_eps] = np.log(epsilon)

            sol_eps = screenkhorn.acc_projected_grad(u, v, I_eps, J_eps,
                                                     backtracking=backtracking_solver,
                                                     max_iter_backtracking=max_iter_backtracking_solver,
                                                     step_size=step_size_solver,
                                                     max_iter=max_iter_solver,
                                                     tol=tol_solver,
                                                     verbose=False)

            u_new = sol_eps["usc"]
            v_new = sol_eps["vsc"]
            # trace_obj.append(screenkhorn.objective(u_new, v_new, I_eps, J_eps))
            u = u_new
            v = v_new

    elif solver == "proj_grad":
        for epsilon in epsilons:
            screenkhorn = Screenkhorn(a, b, M, reg, epsilon)

            I_eps = np.where(a >= epsilon ** 2 * K.sum(axis=1))[0].tolist()
            J_eps = np.where(b >= epsilon ** 2 * K.T.sum(axis=1))[0].tolist()

            # Ic_eps = screenkhorn._complementary(I_eps, n)
            # Jc_eps = screenkhorn._complementary(J_eps, m)
            #
            # time_start = time()
            #
            # P_sink = sinkhorn(a[I_eps], b[J_eps], M[np.ix_(I_eps, J_eps)], reg=reg, log=True)
            # outputs_dict = P_sink[1]
            # exp_u_star = outputs_dict['u']
            # exp_v_star = outputs_dict['v']
            #
            # time_end = time()- time_start
            # print("Time spending duting restricted Skinkhorn is %s" %time_end)
            #
            # u = np.zeros(n)
            # v = np.zeros(m)
            # u[I_eps] = np.log(exp_u_star)
            # v[J_eps] = np.log(exp_v_star)
            # u[Ic_eps] = np.log(epsilon)
            # v[Jc_eps] = np.log(epsilon)

            sol_eps = screenkhorn.projected_grad(u, v, I_eps, J_eps,
                                                 backtracking=backtracking_solver,
                                                 max_iter_backtracking=max_iter_backtracking_solver,
                                                 step_size=step_size_solver,
                                                 max_iter=max_iter_solver,
                                                 tol=tol_solver,
                                                 verbose=False)

            u_new = sol_eps["usc"]
            v_new = sol_eps["vsc"]
            # trace_obj.append(screenkhorn.objective(u_new, v_new, I_eps, J_eps))
            u = u_new
            v = v_new

    elif solver == 'block_acc_proj_grad':
        for epsilon in epsilons:
            screenkhorn = Screenkhorn(a, b, M, reg, epsilon)

            I_eps = np.where(a >= epsilon ** 2 * K.sum(axis=1))[0].tolist()
            J_eps = np.where(b >= epsilon ** 2 * K.T.sum(axis=1))[0].tolist()

            # Ic_eps = screenkhorn._complementary(I_eps, n)
            # Jc_eps = screenkhorn._complementary(J_eps, m)
            #
            # time_start = time()
            #
            # P_sink = sinkhorn(a[I_eps], b[J_eps], M[np.ix_(I_eps, J_eps)], reg=reg, log=True)
            # outputs_dict = P_sink[1]
            # exp_u_star = outputs_dict['u']
            # exp_v_star = outputs_dict['v']
            #
            # time_end = time()- time_start
            # print("Time spending duting restricted Skinkhorn is %s" %time_end)
            #
            # u = np.zeros(n)
            # v = np.zeros(m)
            # u[I_eps] = np.log(exp_u_star)
            # v[J_eps] = np.log(exp_v_star)
            # u[Ic_eps] = np.log(epsilon)
            # v[Jc_eps] = np.log(epsilon)

            sol_eps = screenkhorn.block_acc_projected_grad(u, v, I_eps, J_eps,
                                                           backtracking=backtracking_solver,
                                                           max_iter_backtracking=max_iter_backtracking_solver,
                                                           step_size=step_size_solver,
                                                           max_iter=max_iter_solver,
                                                           tol=tol_solver,
                                                           verbose=False)

            u_new = sol_eps["usc"]
            v_new = sol_eps["vsc"]
            # trace_obj.append(screenkhorn.objective(u_new, v_new, I_eps, J_eps))
            u = u_new
            v = v_new

    else:
        for epsilon in epsilons:
            screenkhorn = Screenkhorn(a, b, M, reg, epsilon)

            I_eps = np.where(a >= epsilon ** 2 * K.sum(axis=1))[0].tolist()
            J_eps = np.where(b >= epsilon ** 2 * K.T.sum(axis=1))[0].tolist()

            # Ic_eps = screenkhorn._complementary(I_eps, n)
            # Jc_eps = screenkhorn._complementary(J_eps, m)
            #
            # time_start = time()
            #
            # P_sink = sinkhorn(a[I_eps], b[J_eps], M[np.ix_(I_eps, J_eps)], reg=reg, log=True)
            # outputs_dict = P_sink[1]
            # exp_u_star = outputs_dict['u']
            # exp_v_star = outputs_dict['v']
            #
            # time_end = time()- time_start
            # print("Time spending duting restricted Skinkhorn is %s" %time_end)
            #
            # u = np.zeros(n)
            # v = np.zeros(m)
            # u[I_eps] = np.log(exp_u_star)
            # v[J_eps] = np.log(exp_v_star)
            # u[Ic_eps] = np.log(epsilon)
            # v[Jc_eps] = np.log(epsilon)

            sol_eps = screenkhorn.block_projected_grad(u, v, I_eps, J_eps,
                                                       backtracking=backtracking_solver,
                                                       max_iter_backtracking=max_iter_backtracking_solver,
                                                       step_size=step_size_solver,
                                                       max_iter=max_iter_solver,
                                                       tol=tol_solver,
                                                       verbose=False)

            u_new = sol_eps["usc"]
            v_new = sol_eps["vsc"]
            # trace_obj.append(screenkhorn.objective(u_new, v_new, I_eps, J_eps))
            u = u_new
            v = v_new


    return u, v


