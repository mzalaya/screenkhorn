import numpy as np
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
                  @ K_IJ @ np.diag(np.exp(v_param_J)) @ np.ones(card_J) - u_param_I.T @ a_I - v_param_J.T @ b_J
        part_IJc = self.epsilon * np.ones(card_I).T @ np.diag(np.exp(u_param_I)) @ K_IJc @ np.ones(card_Jc)
        part_IcJ = self.epsilon * np.ones(card_Ic).T @ K_IcJ @ np.diag(np.exp(v_param_J)) @ np.ones(card_J)
        part_IcJc = self.epsilon** 2 * np.ones(card_Ic).T @ K_IcJc @ np.ones(card_Jc)\
                    - u_param_Ic.T @ a_Ic - v_param_Jc.T @ b_Jc
    
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

        # gradients of Psi_epsilon w. r. t. u and v
        grad_u = np.exp(u_param_I) * (K_IJ @ np.exp(v_param_J)) + self.epsilon * (K_IJc @ np.ones(card_Jc)) - a_I
        # grad_v = np.exp(v_param_J) * (np.exp(u_param_I).T @ K_IJ) + self.epsilon * (np.ones(card_Ic).T @ K_IcJ) - b_J
        grad_v = np.exp(v_param_J) * (K_IJ.T @ np.exp(u_param_I)) + self.epsilon * (K_IcJ.T @ np.ones(card_Ic)) - b_J
        return grad_u, grad_v

    def _projection(self, u):
        if u.all() >= self.epsilon:
            return u
        else:
            return np.array([np.log(self.epsilon)] * len(u))

    def projected_grad(self, I, J, max_iter=100, tol=1e-10, verbose=False):
        """
        Projected Gradient Descent
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

        cp = 0

        # PGD initializations
        u_sc = self._projection(np.zeros(n))
        v_sc = self._projection(np.zeros(m))

        u_sc[Ic] = np.array([np.log(self.epsilon)] * len(Ic))
        v_sc[Jc] = np.array([np.log(self.epsilon)] * len(Jc))

        objval = self.objective(u_sc, v_sc, I, J)
        t_start = time.clock()
        for k in tqdm(range(max_iter)):

            step_k = 100.

            grad_u, grad_v = self.grad_objective(u_sc, v_sc, I, J)

            u_sc_new = u_sc.copy()
            v_sc_new = v_sc.copy()

            zu_sc = self._subvector(I, u_sc)
            zv_sc = self._subvector(J, v_sc)

            u_sc_new[I] = zu_sc - step_k * grad_u
            v_sc_new[J] = zv_sc - step_k * grad_v

            u_sc_new[I] = self._projection(u_sc_new[I])
            v_sc_new[J] = self._projection(v_sc_new[J])

            objval_new = self.objective(u_sc_new, v_sc_new, I, J)

            for j in range(10):
                if objval_new <= objval - (step_k/10)*np.linalg.norm(np.hstack([grad_u, grad_v]))**2:
                    objval = objval_new
                    break
                else:
                    cp = cp + 1
                    step_k *= 0.5
                    u_sc_new[I] = zu_sc - step_k * grad_u
                    v_sc_new[J] = zv_sc - step_k * grad_v

                    u_sc_new[I] = self._projection(u_sc_new[I])
                    v_sc_new[J] = self._projection(v_sc_new[J])

                    objval_new = self.objective(u_sc_new, v_sc_new, I, J)

            u_sc = u_sc_new
            v_sc = v_sc_new

            dif = objval - objval_new
            if verbose:
                print("iter: %d obj: % e, dif: %e" %(k, objval_new, dif))

            if np.linalg.norm(np.hstack([grad_u, grad_v])) < tol:
               break

            history["objval"].append(objval_new)

        total_time = (time.clock() - t_start)
        print("Total time taken: ", total_time)
        print("counting cp in the bachtrack loop: ", cp)
        
        return u_sc, v_sc, history