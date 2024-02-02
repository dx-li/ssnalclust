import numpy as np
import pylops
import scipy
import scipy.sparse
import scipy.sparse.linalg as sla
import math
from numpy.typing import ArrayLike
from typing import Tuple

# Preliminaries

#TO DO: Remove the pylops dependency and use scipy.sparse.linalg.LinearOperator instead?
class B(pylops.LinearOperator):
    r"""
    B operator using node-arc incidence matrices as defined in the paper.
    """

    def __init__(self, map: ArrayLike, dtype=None):
        r"""
        Parameters
        ----------
        map : array-like
            Node-arc incidence matrix.
        dtype : data-type, optional
            The desired data-type for the array. If not given, then the type will be determined as the minimum type required to hold the objects in the sequence.
        """
        self.map = map
        super().__init__(dtype=np.dtype(dtype), shape=(map.shape[0], map.shape[1]))

    def _matmat(self, X: ArrayLike) -> ArrayLike:
        r"""
        Matrix-matrix multiplication.
        """
        return (self.map.T @ X.T).T

    def _adjoint(self, Z):
        r"""
        Adjoint operator.
        """
        return (self.map @ Z.T).T


def fnorm(X: ArrayLike) -> float:
    r"""
    Returns the Frobenius norm of a matrix, accounting for if it is sparse or not.
    """
    if scipy.sparse.issparse(X):
        return sla.norm(X, "fro")
    return np.linalg.norm(X, "fro")


def fnormsq(X: ArrayLike) -> float:
    r"""
    Returns the square of the Frobenius norm of a matrix.
    """
    return fnorm(X) ** 2


def matinner(X: ArrayLike, Y: ArrayLike) -> float:
    r"""
    Returns the inner product of two matrices, assumes they are the same shape.
    """
    return np.einsum("ij,ij->", X, Y)


def column_norms(X) -> ArrayLike:
    r"""
    Returns the column-wise norms of a matrix.
    """
    return np.sqrt(np.einsum("ij,ij->j", X, X))


# SSNAL


class SSNAL:
    r"""
    SSNAL algorithm for solving the convex clustering problem. Please see the paper for more details: https://proceedings.mlr.press/v80/yuan18a.html
    """

    def __init__(self, A, weights_matrix, gamma):
        r"""
        Parameters
        ----------
        A : array-like
            Data matrix. Each column is a data point.
        weights_matrix : array-like
            Weights matrix. Entry (i, j) is the weight between columns i and j.
        gamma : float
            Regularization parameter.
        """
        self.A = A
        self.d, N = A.shape
        self.n = weights_matrix.shape[0]
        self.nz_r, self.nz_c = np.nonzero(np.triu(weights_matrix))
        # saves the non-regularized weights so we can adjust gamma later
        self.pre_weights = weights_matrix[self.nz_r, self.nz_c]
        self.gamma = gamma
        self.E = len(self.nz_r)
        cJ = scipy.sparse.lil_array((self.n, self.E))
        cJ[self.nz_r, np.arange(self.E)] = 1
        cJ[self.nz_c, np.arange(self.E)] = -1
        cJ = cJ.tocsr()
        self.Bop = B(cJ)

    @property
    def gamma(self) -> float:
        r"""
        Regularization parameter.
        """
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        r"""
        Set the regularization parameter.
        """
        if value < 0:
            raise ValueError("gamma must be non-negative")
        self.weights = value * self.pre_weights
        self._gamma = value

    def pU(self, U: ArrayLike) -> float:
        r"""
        Weighted sum of the column norms of U.
        """
        return np.dot(self.weights, column_norms(U))

    def prox_pU(self, U: ArrayLike, tau=1) -> ArrayLike:
        r"""
        Proximal operator of pU.
        Parameters
        ----------
        U : array-like
            Matrix.
        tau : float, optional
            Proximal parameter.
        """
        weights = self.weights
        upper = tau * weights
        norms = column_norms(U)
        norms = np.maximum(norms, upper)
        return U * (1 - upper / norms)

    def proxdual_pU(self, U: ArrayLike, tau=None) -> ArrayLike:
        r"""
        Proximal operator of the dual of pU.
        Parameters
        ----------
        U : array-like
            Matrix.
        tau : float, optional
            Proximal parameter.
        """
        weights = self.weights
        upper = weights
        norms = column_norms(U)
        norms = np.maximum(norms, upper)
        return U * (upper / norms)

    def primal(self, X, U):
        r"""
        Primal objective function.
        """
        return 0.5 * fnormsq(X - self.A) + self.pU(U)

    def dual(self, X, U, Z):
        r"""
        Dual objective function.
        """
        adZ = self.Bop._adjoint(Z)
        return -0.5 * fnormsq(adZ) + matinner(self.A, adZ)

    def kkt_1(self, V, X):
        r"""
        First KKT condition.
        """
        return V + X - self.A

    def kkt_2(self, U, Z):
        r"""
        Second KKT condition.
        """
        return U - self.prox_pU(U + Z)

    def kkt_3(self, X, U):
        r"""
        Third KKT condition.
        """
        return self.Bop._matmat(X) - U

    def kkt_4(self, Z, V):
        r"""
        Fourth KKT condition.
        """
        return self.Bop._adjoint(Z) - V

    def kkt_residual(self, X, U, Z):
        r"""
        KKT residual for stopping criterion.
        """
        etp = (fnorm(self.kkt_3(X, U))) / (1 + fnorm(U))

        etd = fnorm(Z - self.proxdual_pU(Z)) / (1 + fnorm(Z))

        et = fnorm(self.kkt_4(Z, -X + self.A)) / (1 + fnorm(X)) + fnorm(
            self.kkt_2(U, Z)
        ) / (1 + fnorm(U))

        return etp, etd, et

    def lagrangian(self, X, U, Z):
        r"""
        Lagrangian function of the convex clustering problem.
        """
        return 0.5 * fnormsq(X - self.A) + self.pU(U) + matinner(Z, self.kkt_3(X, U))

    def augmented_lagr(self, X, U, Z, sigma):
        r"""
        Augmented Lagrangian function of the convex clustering problem.
        """
        return self.lagrangian(X, U, Z) + sigma / 2 * (fnormsq(self.kkt_3(X, U)))

    def phi(self, X, Z, sigma):
        r"""
        Phi function used in the SSNCG subproblem.
        """
        return (
            0.5 * fnormsq(X - self.A)
            + self.pU(self.prox_pU(self.Bop._matmat(X) + Z / sigma, 1 / sigma))
            + 0.5
            * fnormsq(self.proxdual_pU(sigma * self.Bop._matmat(X) + Z, sigma))
            / sigma
            - 0.5 * fnormsq(Z) / sigma
        )

    def grad_phi(self, X, Z, sigma):
        r"""
        Gradient of the Phi function used in the SSNCG subproblem.
        """
        return (
            X
            - self.A
            + self.Bop._adjoint(self.proxdual_pU(sigma * self.Bop._matmat(X) + Z))
        )

    @staticmethod
    def sigma_update(itera):
        r"""
        Helper to update value of sigma.
        """
        sigma_update_iter = 2
        if itera < 10:
            sigma_update_iter = 2
        elif itera < 20:
            sigma_update_iter = 3
        elif itera < 500:
            sigma_update_iter = 10
        else:
            sigma_update_iter = 10
        return sigma_update_iter

    def fit(
        self,
        max_iter: int,
        sigma0: float = 1,
        eps: float = 1e-6,
        X0: ArrayLike = None,
        U0: ArrayLike = None,
        Z0: ArrayLike = None,
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike, int]:
        r"""
        Fit the convex clustering problem using SSNAL.
        Parameters
        ----------
        max_iter : int
            Maximum number of iterations.
        sigma0 : float, optional
            Initial value of sigma.
        eps : float, optional
            Tolerance for stopping criterion.
        X0 : array-like, optional
            Initial value of X.
        U0 : array-like, optional
            Initial value of U.
        Z0 : array-like, optional
            Initial value of Z.
        Returns
        -------
        X : array-like
            Solution to the convex clustering problem.
        U : array-like
            Solution to the primal problem.
        Z : array-like
            Solution to the dual problem.
        termination : int
            Termination status. 0 if successful, 1 if not.
        """
        if X0 is None:
            Xi = self.A.copy()
        else:
            Xi = X0
        if U0 is None:
            Ui = self.Bop._matmat(Xi)
        else:
            Ui = U0
        if Z0 is None:
            Zi = np.zeros_like(Ui)
        else:
            Zi = Z0
        sigmai = sigma0

        termination = 1
        prim_win = 0
        dual_win = 0
        # TO DO: REMOVE ONCE DONE TESTING
        ncg_iter = 60
        ncg_tol = 1e-6

        for itera in range(max_iter):
            # Step 1 - solve subproblem for best X and U

            Xi = self.ssncg(Xi, Ui, Zi, sigmai, max_iter=ncg_iter, tolerance=ncg_tol)
            Ui = self.prox_pU(self.Bop._matmat(Xi) + Zi / sigmai, 1 / sigmai)

            # Step 2 - update Z

            Zi = Zi + sigmai * self.kkt_3(Xi, Ui)

            # Check stopping criterion

            kkt_residual = self.kkt_residual(Xi, Ui, Zi)
            prim = self.primal(Xi, Ui)
            dual = self.dual(Xi, Ui, Zi)
            prim_dual_gap = np.abs(prim - dual) / (1 + np.abs(prim) + np.abs(dual))
            # TO DO: Option to terminate on kkt vs prim_dual_gap
            if prim_dual_gap < eps:
                termination = 0
                break
            if np.max(kkt_residual) < eps:
                termination = 0
                break

            # Step 3 - update sigma
            if kkt_residual[0] < kkt_residual[1]:
                prim_win += 1
            else:
                dual_win += 1
            sigma_ui = self.sigma_update(itera)
            sigmascale = 5
            sigmamax = 135
            # TO DO: break this into a separate function
            if itera % sigma_ui == 0:
                if itera % sigma_ui == 0:
                    sigmamin = 1e-4
                    if prim_win > max(1, 1.2 * dual_win):
                        prim_win = 0
                        sigmai = max(sigmamin, sigmai / sigmascale)
                    elif dual_win > max(1, 1.2 * prim_win):
                        dual_win = 0
                        sigmai = min(sigmamax, sigmai * sigmascale)
            if kkt_residual[0] < 1e-5:
                ncg_iter = max(ncg_iter, 30)
            elif kkt_residual[0] < 1e-3:
                ncg_iter = max(ncg_iter, 30)
            elif kkt_residual[0] < 1e-1:
                ncg_iter = max(ncg_iter, 20)

        return Xi, Ui, Zi, termination

    # SSNCG
    def ssncg(
        self,
        X0,
        U,
        Z,
        sigma,
        mu=1e-4,
        tau=0.5,
        eta=1e-2,
        delta=0.5,
        max_iter=20,
        tolerance=1e-6,
        cg_max_iter=300,
    ):
        r"""
        SSNCG algorithm for solving the subproblem in SSNAL.
        Parameters
        ----------
        X0 : array-like
            Initial value of X.
        U : array-like
            Value of U.
        Z : array-like
            Value of Z.
        sigma : float
            Value of sigma.
        mu : float, optional
            Parameter for line search.
        tau : float, optional
            Parameter for conjugate-gradient stopping criterion.
        eta : float, optional
            Parameter for conjugate-gradient stopping criterion.
        delta : float, optional
            Parameter for line search.
        max_iter : int, optional
            Maximum number of iterations.
        tolerance : float, optional
            Tolerance for stopping criterion.
        cg_max_iter : int, optional
            Maximum number of iterations for the conjugate-gradient method.
        Returns
        -------
        Xj : array-like
            Solution to the subproblem.
        """

        # Initialization

        Xj = X0
        sigmainv = 1 / sigma

        normborg = 1 + fnorm(self.A)
        Rd = np.maximum(column_norms(Z) - self.weights, 0)
        normRd = np.sum(Rd)
        normU = fnorm(U)
        Rp = self.kkt_3(X0, U)
        normRp = fnorm(Rp)
        # tolerance = fnorm(self.grad_phi(Xj, Z, sigma)) * .1
        score = 0
        norm_prev_best = np.inf

        for itera in range(max_iter):
            # print('ssncg iter: ', iter)

            # Check stopping criterion
            # TO DO: remove testing print statements and comments

            grad_phi_Xj = self.grad_phi(Xj, Z, sigma)
            norm_grad_phi_Xj = fnorm(grad_phi_Xj)

            priminf_sub = norm_grad_phi_Xj
            dualinf_sub = normRd / (1 + normborg)
            if max(priminf_sub, dualinf_sub) < tolerance:
                tolsubconst = 0.9
            else:
                tolsubconst = 0.5
            tolsub = np.maximum(
                np.minimum(1, 0.5 * normRp / (1 + normU)), tolsubconst * tolerance
            )
            # print(np.nanmax(norm_grad_phi_Xj), np.nanmax(tolsub), tolerance)
            # print(tolerance)
            if itera > 0:
                if norm_grad_phi_Xj > norm_prev_best:
                    score += 1
                    if score > 5:
                        print("here")
                        # Xj = Xj_best
                        break
                else:
                    norm_prev_best = norm_grad_phi_Xj
                    Xj_best = Xj
            if (norm_grad_phi_Xj < tolsub) and itera > 0:
                print("subproblem converged")
                break
            elif max(priminf_sub, dualinf_sub) < 0.5 * tolerance:
                print("here2")
                break

            # Step 1 - use conjugate gradient to find the direction of descent

            D = self.Bop._matmat(Xj) + Z / sigma
            upper = self.weights
            norms_D = column_norms(D)
            zet = upper / norms_D
            zet_less_one = zet < 1
            # zet_less_one_explicit = np.arange(self.E)[zet_less_one]
            norms_D_hat = norms_D[zet_less_one]
            D_sel = D[:, zet_less_one] / norms_D_hat

            alpha = self.weights[zet_less_one] / (sigma * norms_D_hat)

            def V(X):
                map_X = self.Bop._matmat(X)
                map_X_hat = map_X[:, zet_less_one]
                rho = np.einsum("j, ij,ij->j", alpha, map_X_hat, D_sel)
                map_X[:, zet_less_one] = map_X_hat * alpha - D_sel * rho
                return X + sigma * self.Bop._adjoint(map_X)

            cg_tolerance = norm_grad_phi_Xj ** (1 + tau)
            # print('cg tolerance: ', cg_tolerance)
            dj = np.zeros_like(Xj)
            residual = grad_phi_Xj
            sd = -residual
            beta = 0
            for j in range(cg_max_iter):
                # print('cg iter: ', j)
                if j != 0:
                    beta = np.sum(np.square(residual)) / ssres  # new over old
                    sd = -residual + beta * sd
                Vsd = V(sd)
                sdVsd = np.sum(sd * Vsd)
                ssres = np.sum(np.square(residual))
                dj = dj + ssres / sdVsd * sd
                residual = residual + ssres / sdVsd * Vsd
                stopping = fnorm(grad_phi_Xj + V(dj))
                # print('stopping: ', stopping)
                if stopping <= min(eta, cg_tolerance):
                    break

            # Step 2 - line search for best step size
            # print('step 2')

            alpj = self.line_search(Xj, dj, Z, sigma, mu)
            # alpj = 10
            # phiXj = self.phi(Xj, Z, sigma)
            # innergpXjdj = matinner(grad_phi_Xj, dj)
            # while self.phi(Xj + alpj * dj, Z, sigma) > phiXj + mu * alpj * innergpXjdj:
            #     alpj *= delta
            print("alpj: ", alpj)

            # Step 3 - update Xj

            Xj = Xj + alpj * dj

            if alpj < 1e-10:
                break

        print("subiter", itera)
        return Xj

    def line_search(self, Xj, dj, Z, sigma, mu=1e-4, c2=0.9, alp_max=1e3):
        r"""
        Line search for the SSNCG subproblem.
        """
        alp_prev = 0
        alpi = 1
        f = lambda alp: self.phi(Xj + alp * dj, Z, sigma)
        f0 = f(0)
        f_1 = lambda alp: matinner(self.grad_phi(Xj + alp * dj, Z, sigma), dj)
        f_1_0 = f_1(0)
        max_iter = 10
        fprev = f0
        for i in range(max_iter):
            curr = f(alpi)
            if curr > f0 + mu * alpi * f_1_0 or (curr > fprev and i > 0):
                alpi = self.zoom(alp_prev, alpi, mu, c2, f, f_1, f0, f_1_0)
                return alpi
            grad = f_1(alpi)
            if abs(grad) <= c2 * abs(f_1_0):
                return alpi
            if grad >= 0:
                alpi = self.zoom(alpi, alp_prev, mu, c2, f, f_1, f0, f_1_0)
                return alpi
            alp_prev = alpi
            fprev = curr
            alpi = min(2 * alpi, alp_max)
        print("line search did not converge")
        return alpi

    def zoom(self, alp_low, alp_high, mu, c2, f, f_1, f0, f_1_0):
        r"""
        Zoom function for the line search.
        """
        while True:
            alp_j = (alp_low + alp_high) / 2
            if f(alp_j) > f0 + mu * alp_j * f_1_0:
                alp_high = alp_j
            else:
                grad = f_1(alp_j)
                if abs(grad) <= c2 * abs(f_1_0):
                    return alp_j
                if grad * (alp_high - alp_low) >= 0:
                    alp_high = alp_low
                alp_low = alp_j
