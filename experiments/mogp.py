import numpy as np
import pymc as pm

from pymc.gp.cov import Constant, Covariance
from pymc.gp.mean import Zero
from pymc.gp.util import (
    JITTER_DEFAULT,
    cholesky,
    conditioned_vars,
    replace_with_values,
    solve_lower,
    solve_upper,
    stabilize,
)

import aesara.tensor as at
from pymc.gp.gp import Base, Latent, Marginal


class MultiOutputMarginal2(Marginal):

    def __init__(self, means, kernels, input_dim, active_dims, num_outputs, W=None, B=None):
        self.means = means
        self.kernels = kernels
        self.cov_funcs = kernels
        self.cov_func = self._get_lcm(input_dim, active_dims, num_outputs, kernels, W, B)
        super().__init__(cov_func = self.cov_func)


    def _get_icm(self, input_dim, kernel, W=None, kappa=None, B=None, active_dims=None, name='ICM'):
        """
        Builds a kernel for an Intrinsic Coregionalization Model (ICM)
        :input_dim: Input dimensionality (include the dimension of indices)
        :num_outputs: Number of outputs
        :kernel: kernel that will be multiplied by the coregionalize kernel (matrix B).
        :W: the W matrix
        :B: the convariance matrix for tasks
        :name: The name of Intrinsic Coregionalization Model
        """

        # coreg = pm.gp.cov.Coregion(input_dim=input_dim, W=W, kappa=kappa, B=B, active_dims=active_dims)
        # return coreg * kernel
        #B = at.dot(W, W.T) + at.diag(kappa)
        coreg = CoregionMatrix(W=W, B=B)
        cov_func = pm.gp.cov.Kron([coreg, kernel])
        return cov_func

    def _get_lcm(self, input_dim, active_dims, num_outputs, kernels, W=None, B=None, name='ICM'):
        if B is None:
            kappa = pm.Gamma(f"{name}_kappa", alpha=5, beta=1, shape=num_outputs)
            if W is None:
                W = pm.Normal(f"{name}_W", mu=0, sigma=5, shape=(num_outputs, 1), 
                                                            initval=np.random.randn(num_outputs, 1))
        else:
            kappa = None
            
        cov_func = 0
        for idx, kernel in enumerate(kernels):            
            icm = self._get_icm(input_dim, kernel, W, kappa, B, active_dims, f'{name}_{idx}')
            cov_func += icm
        return cov_func

    # def _build_marginal_likelihood(self, X, noise, jitter):
    #     mu = self.mean_func(X)
    #     Kxx = self.cov_func(X)
    #     Knx = noise(X)
    #     cov = Kxx + Knx
    #     return mu, stabilize(cov, jitter)

    def _build_marginal_likelihood(self, Xs):
        self.X = cartesian(*Xs)
        mu = self.mean_func(self.X)
        covs = [f(X) for f, X in zip(self.cov_funcs, Xs)]
        return mu, covs


    def _check_inputs(self, Xs, y):
        N = int(np.prod([len(X) for X in Xs]))
        if len(Xs) != len(self.cov_funcs):
            raise ValueError("Must provide a covariance function for each X")
        if N != len(y):
            raise ValueError(
                f"Length of y ({len(y)}) must match length of cartesian product of Xs ({N})"
            )

    def marginal_likelihood(self, name, Xs, y, sigma, is_observed=True, **kwargs):
        """
        Returns the marginal likelihood distribution, given the input
        locations `cartesian(*Xs)` and the data `y`.
        """
        self._check_inputs(Xs, y)
        mu, covs = self._build_marginal_likelihood(Xs)
        self.Xs = Xs
        self.y = y
        self.sigma = sigma
        if is_observed:
            return pm.KroneckerNormal(name, mu=mu, covs=covs, sigma=sigma, observed=y, **kwargs)
        else:
            warnings.warn(
                "The 'is_observed' argument has been deprecated.  If the GP is "
                "unobserved use gp.LatentKron instead.",
                FutureWarning,
            )
            size = int(np.prod([len(X) for X in Xs]))
            return pm.KroneckerNormal(name, mu=mu, covs=covs, sigma=sigma, size=size, **kwargs)

    # def marginal_likelihood(self, name, X, y, noise, jitter=0.0, is_observed=True, **kwargs):
    #     if not isinstance(noise, Covariance):
    #         noise = pm.gp.cov.WhiteNoise(noise)
    #     mu, cov = self._build_marginal_likelihood(X, noise, jitter)
    #     self.X = X
    #     self.y = y
    #     self.noise = noise
    #     if is_observed:
    #         return pm.MvNormal(name, mu=mu, cov=cov, observed=y, **kwargs)
    #     else:
    #         warnings.warn(
    #             "The 'is_observed' argument has been deprecated.  If the GP is "
    #             "unobserved use gp.Latent instead.",
    #             FutureWarning,
    #         )
    #         return pm.MvNormal(name, mu=mu, cov=cov, **kwargs)


class MultiOutputMarginal(Marginal):

    def __init__(self, means, kernels, input_dim, active_dims, num_outputs, W=None, B=None):
        self.means = means
        self.kernels = kernels
        self.cov_func = self._get_lcm(input_dim, active_dims, num_outputs, kernels, W, B)
        super().__init__(cov_func = self.cov_func)


    def _get_icm(self, input_dim, kernel, W=None, kappa=None, B=None, active_dims=None, name='ICM'):
        """
        Builds a kernel for an Intrinsic Coregionalization Model (ICM)
        :input_dim: Input dimensionality (include the dimension of indices)
        :num_outputs: Number of outputs
        :kernel: kernel that will be multiplied by the coregionalize kernel (matrix B).
        :W: the W matrix
        :B: the convariance matrix for tasks
        :name: The name of Intrinsic Coregionalization Model
        """

        coreg = pm.gp.cov.Coregion(input_dim=input_dim, W=W, kappa=kappa, B=B, active_dims=active_dims)
        return coreg * kernel
        # B = at.dot(W, W.T) + at.diag(kappa)
        # coreg = Coregionalize(input_dim=input_dim, active_dims=active_dims, kappa=kappa, W=W)
        # cov_func = pm.gp.cov.Kron([coreg, kernel])
        # return cov_func        
        

    def _get_lcm(self, input_dim, active_dims, num_outputs, kernels, W=None, B=None, name='ICM'):
        if B is None:
            kappa = pm.Gamma(f"{name}_kappa", alpha=5, beta=1, shape=num_outputs)
            if W is None:
                W = pm.Normal(f"{name}_W", mu=0, sigma=5, shape=(num_outputs, 1), 
                                                            initval=np.random.randn(num_outputs, 1))
        else:
            kappa = None
            W = None
        
        cov_func = 0
        for idx, kernel in enumerate(kernels):            
            print(B)        
            icm = self._get_icm(input_dim, kernel, W, kappa, B, active_dims, f'{name}_{idx}')
            cov_func += icm
        return cov_func

    def _build_marginal_likelihood(self, X, noise, jitter):
        mu = self.mean_func(X)
        Kxx = self.cov_func(X)
        Knx = noise(X)
        cov = Kxx + Knx
        return mu, stabilize(cov, jitter)

    def marginal_likelihood(self, name, X, y, noise, jitter=0.0, is_observed=True, **kwargs):
        if not isinstance(noise, Covariance):
            noise = pm.gp.cov.WhiteNoise(noise)
        mu, cov = self._build_marginal_likelihood(X, noise, jitter)
        self.X = X
        self.y = y
        self.noise = noise
        if is_observed:
            return pm.MvNormal(name, mu=mu, cov=cov, observed=y, **kwargs)
        else:
            warnings.warn(
                "The 'is_observed' argument has been deprecated.  If the GP is "
                "unobserved use gp.Latent instead.",
                FutureWarning,
            )
            return pm.MvNormal(name, mu=mu, cov=cov, **kwargs)


class MultiOutputLatent(Latent):

    def __init__(self, means, kernels, input_dim, active_dims, num_outputs, B=None, W=None):

        self.means = means
        self.kernels = kernels
        self.cov_func = self._get_lcm(input_dim=input_dim, active_dims=active_dims, num_outputs=num_outputs, kernels=kernels)
        super().__init__(cov_func = self.cov_func)

