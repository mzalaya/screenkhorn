import numpy as np
import cvxpy as cvx
import time
from tqdm import tqdm

__author__ = 'Mokhtar Z. Alaya'

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

    """
    def __init__(self, a, b, M, reg, epsilon):
        self.a = a
        self.b = b
        self.M = M
        self.reg = reg
        self.epsilon = epsilon

    @staticmethod
    def _complementary(I, r):
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
        K = np.exp(-self.M / self.reg)
        
        card_I, card_J = len(I), len(J)
        Ic = self._complementary(I, n)
        Jc = self._complementary(J, m)
        card_Ic, card_Jc  = len(Ic), len(Jc)

        # subparameters u_I, v_J
        u_param_I, v_param_J = self._subvector(I, u_param), self._subvector(J, v_param)
        # u_param_Ic, v_param_Jc = self._subvector(Ic, u_param), self._subvector(Jc, v_param)
        u_param_Ic = np.array([np.log(self.epsilon)] * card_Ic)
        v_param_Jc =  np.array([np.log(self.epsilon)] * card_Jc)
    
        # submarginals mu_{I}, nu_{J}, mu_{I^c}, nu_{J^c}
        a_I, b_J = self._subvector(I, self.a), self._subvector(J, self.b)
        a_Ic = self._subvector(Ic, self.a)
        b_Jc = self._subvector(Jc, self.b)
    
        # submatrices K_{IJ}, K_{IJ^c}, K_{I^cJ}, K_{I^cJ^c}
        K_IJ, K_IJc = self._submatrix(K, I, J), self._submatrix(K, I, Jc)
        K_IcJ, K_IcJc = self._submatrix(K, Ic, J), self._submatrix(K, Ic, Jc)
    
        part_IJ = np.ones(card_I).T @ np.diag(np.exp(u_param_I)) \
                  @ K_IJ @ np.diag(np.exp(v_param_J)) @ np.ones(card_J) - u_param_I.T @ a_I - v_param_J @ b_J
        part_IJc = self.epsilon * np.ones(card_I).T @ np.diag(np.exp(u_param_I)) @ K_IJc @ np.ones(card_Jc)
        part_IcJ = self.epsilon * np.ones(card_Ic).T @ K_IcJ @ np.diag(np.exp(v_param_J)) @ np.ones(card_J)
        part_IcJc = self.epsilon ** 2 * np.ones(card_Ic).T @ K_IcJc @ np.ones(card_Jc)\
                    - u_param_Ic.T @ a_Ic - v_param_Jc @ b_Jc
    
        psi_epsilon = part_IJ + part_IJc + part_IcJ + part_IcJc
        return psi_epsilon

    def grad_objective(self, u_param, v_param, I, J):

        K = np.exp(- self.M / self.reg)
        (n, m) = self.M.shape

        card_I, card_J = len(I), len(J)
        Ic, Jc = self._complementary(I, n), self._complementary(J, m)
        card_Ic, card_Jc = len(Ic), len(Jc)

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

        # gradient of Psi_epsilon w. r. t. u and v
        grad_u = np.exp(u_param_I) * (K_IJ @ np.exp(v_param_J)) + self.epsilon * (K_IJc @ np.ones(card_Jc)) - a_I
        grad_v = np.exp(v_param_J) * (np.exp(u_param_I).T @ K_IJ) + self.epsilon * (np.ones(card_Ic).T @ K_IcJ) - b_J
        return grad_u, grad_v


    def _lipschitz_constants(self, u_param, v_param, I, J):

        K = np.exp(- self.M / self.reg)
        (n, m) = self.M.shape

        Ic, Jc = self._complementary(I, n), self._complementary(J, m)
        card_Ic, card_Jc = len(Ic), len(Jc)

        # subparameters u_I, v_J
        u_param_I = self._subvector(I, u_param)
        v_param_J = self._subvector(J, v_param)

        # submatrices K_{IJ}, K_{IJ^c}, K_{I^cJ}
        K_IJ = self._submatrix(K, I, J)
        K_IJc = self._submatrix(K, I, Jc)
        K_IcJ = self._submatrix(K, Ic, J)

        # gradient of Psi_epsilon w. r. t. u and v
        lip_u = (K_IJ @ np.exp(v_param_J)) + self.epsilon * (K_IJc @ np.ones(card_Jc))
        lip_v = (np.exp(u_param_I).T @ K_IJ) + self.epsilon * (np.ones(card_Ic).T @ K_IcJ)

        # lip_u = np.linalg.norm(lip_u)
        # lip_v = np.linalg.norm(lip_v)
        lip_u = np.max(lip_u)
        lip_v = np.max(lip_v)
        return lip_u, lip_v

    def _projection_cvx(self, u):
        '''
        Returns the point in the convex set
        C_epsilon = {u in R^n : exp(u) > epsilon}
        that is closest to y (according to Euclidian distance)
        '''
        d = len(u)
        # setup the objective and constraints and solve the problem
        x = cvx.Variable(shape=d)
        obj = cvx.Minimize(cvx.sum_squares(x - u))
        constraints = [x >= cvx.log(self.epsilon)]
        prob = cvx.Problem(obj, constraints)
        prob.solve(solver=cvx.SCS, verbose=False)
        return np.array(x.value).squeeze()

    def _projection(self, u):
        if u.all() >= self.epsilon:
            return u
        else:
            return np.array([np.log(self.epsilon)] * len(u))

    def projected_grad(self, I, J, max_iter=100, tol=1e-10, verbose=False):
        """
        Projected gradient descent
        Parameters
        ----------
        initial : `np.ndarray`, shape=(card(I),) shape(card(J),)
                   initial point
        step_size : `float`, default=None
                    Initial step size used for learning.

        max_iter: `int`, default=100
                   Maximum number of iterations of the solver.

        tol: `float`, default=1e-10
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
        Ic, Jc = self._complementary(I, n), self._complementary(J, m)


        history = self._init_history()

        # PGD initializations

        u_sc = self._projection(np.zeros(n))
        v_sc = self._projection(np.zeros(m))

        grad_u, grad_v = self.grad_objective(u_sc, v_sc, I, J)

        step_u = 1. / 2 #(np.linalg.norm(grad_u))
        step_v = 1. / 2 # (np.linalg.norm(grad_v))

        zu_sc = self._subvector(I, u_sc)
        zv_sc = self._subvector(J, v_sc)

        z_u = zu_sc - step_u * grad_u
        z_v = zv_sc - step_v * grad_v

        z_uc = np.array([np.log(self.epsilon)] * len(Ic))
        z_vc = np.array([np.log(self.epsilon)] * len(Jc))

        u_sc[Ic] = z_uc
        v_sc[Jc] = z_vc

        t_start = time.clock()
        cp = 0
        for k in tqdm(range(max_iter)):

            step_k = (k+1)/(k+3)

            u_sc[I] = z_u
            v_sc[J] = z_v

            objval = self.objective(u_sc, v_sc, I, J)

            grad_u, grad_v = self.grad_objective(u_sc, v_sc, I, J)

            # step_u = 1. / (np.linalg.norm(grad_u))
            # step_v = 1. / (np.linalg.norm(grad_v))

            ##################################
            # z_u = z_u - step_u * grad_u
            # z_v = z_v - step_v * grad_v

            z_u = z_u - step_k * grad_u
            z_v = z_v - step_k * grad_v

            z_u_proj = self._projection(z_u)
            z_v_proj = self._projection(z_v)
            
            u_sc[I] = z_u_proj
            v_sc[J] = z_v_proj

            objval_new = self.objective(u_sc, v_sc, I, J)
            while True:
                if objval_new <= objval - (step_k/2)*np.linalg.norm(np.hstack([grad_u, grad_v]))**2:
                    break
                else:
                    cp = cp + 1
                    step_k *= 0.5

            z_u = z_u_proj
            z_v = z_v_proj

            dif = objval - objval_new
            if verbose:
                print("iter: %d obj: % e, dif: %e" %(k, objval_new, dif))
            if abs(objval - objval_new) < tol:
                break

            history["objval"].append(objval_new)

        total_time = (time.clock() - t_start)
        print("Total time taken: ", total_time)
        
        return u_sc, v_sc, history

    def fista(self, I, J, max_iter=1000, tol=1e-9, verbose=False):
        """
        Accelearated projected gradient descent
        Parameters
        ----------
        initial : `np.ndarray`, shape=(card(I),) shape(card(J),)
                   initial point
        step_size : `float`, default=None
                    Initial step size used for learning.

        max_iter: `int`, default=100
                   Maximum number of iterations of the solver.

        tol: `float`, default=1e-10
              The tolerance of the solver (iterations stop when the stopping criterion is below it).
              If not reached it does ``max_iter`` iterations

        verbose: `bool`, default=True
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
        Ic, Jc = self._complementary(I, n), self._complementary(J, m)


        history = self._init_history()

        # PGD initializations

        u_sc = self._projection(np.zeros(n))
        v_sc = self._projection(np.zeros(m))

        # I^c and J^c
        z_uc = np.array([np.log(self.epsilon)] * len(Ic))
        z_vc = np.array([np.log(self.epsilon)] * len(Jc))
        u_sc[Ic] = z_uc
        v_sc[Jc] = z_vc

        grad_u, grad_v = self.grad_objective(u_sc, v_sc, I, J)
        step_u = 1. / np.linalg.norm(grad_u)
        step_v = 1. / np.linalg.norm(grad_v)

        zu_sc = self._subvector(I, u_sc)
        zv_sc = self._subvector(J, v_sc)

        z_u = zu_sc - step_u * grad_u
        z_v = zv_sc - step_v * grad_v

        z_u_proj = self._projection(z_u)
        z_v_proj = self._projection(z_v)

        t_1 = 1

        # for k in tqdm(range(max_iter)):
        t_start = time.clock()
        for k in range(max_iter):

            t_k = (1 + np.sqrt(1 +4*t_1**2)) / 2
            step_k = (t_1 - 1) / t_k
            t_1 = t_k

            u_sc[I] = z_u_proj
            v_sc[J] = z_v_proj

            objval = self.objective(u_sc, v_sc, I, J)

            grad_u, grad_v = self.grad_objective(u_sc, v_sc, I, J)

            step_u = 1. / np.linalg.norm(grad_u)
            step_v = 1. / np.linalg.norm(grad_v)
            # print(step_u, step_v)
            # print('\n')

            z_u_k = z_u_proj - step_u * grad_u
            z_v_k = z_v_proj - step_v * grad_v

            z_u_proj_k = self._projection(z_u_k)
            z_v_proj_k = self._projection(z_v_k)

            z_u_proj_k = z_u_proj_k - step_k * (z_u_proj_k - z_u_k)
            z_v_proj_k = z_v_proj_k - step_k * (z_v_proj_k - z_v_k)

            u_sc[I] = z_u_proj_k
            v_sc[J] = z_v_proj_k

            objval_new = self.objective(u_sc, v_sc, I, J)

            # z_u = z_u_proj
            # z_v = z_v_proj

            z_u_proj = z_u_proj_k
            z_v_proj = z_v_proj_k

            dif = objval - objval_new

            print("iter: %d obj: % e, dif: %e" %(k, objval_new, dif))
            if abs(objval - objval_new) < tol:
                break

         #   print("Warning: maxIterations reached before convergence")

            history["objval"].append(objval_new)

            # if not verbose:

        # if not verbose:
        total_time = (time.clock() - t_start)
        print("Total time taken: ", total_time)

        return u_sc, v_sc, history
