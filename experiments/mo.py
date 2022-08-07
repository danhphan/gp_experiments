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

class Coregionalize(Covariance):
    def __init__(self, input_dim, W=None, kappa=None, B=None, active_dims=None):
        super().__init__(input_dim, active_dims)
        if len(self.active_dims) != 1:
            raise ValueError("Coregion requires exactly one dimension to be active")
        make_B = W is not None or kappa is not None
        if make_B and B is not None:
            raise ValueError("Exactly one of (W, kappa) and B must be provided to Coregion")
        if make_B:
            self.W = at.as_tensor_variable(W)
            self.kappa = at.as_tensor_variable(kappa)
            self.B = at.dot(self.W, self.W.T) + at.diag(self.kappa)
        elif B is not None:
            self.B = at.as_tensor_variable(B)
        else:
            raise ValueError("Exactly one of (W, kappa) and B must be provided to Coregion")

    def full(self, X, Xs=None):
        # X, Xs = self._slice(X, Xs)
        # index = at.cast(X, "int32")
        # if Xs is None:
        #     index2 = index.T
        # else:
        #     index2 = at.cast(Xs, "int32").T
        # return self.B[index, index2]
        return self.B

    def diag(self, X):
        # X, _ = self._slice(X, None)
        # index = at.cast(X, "int32")
        # return at.diag(self.B)[index.ravel()]
        return at.diag(self.B)

class MultiOutputMarginal(Marginal):

    def __init__(self, means, kernels, input_dim, active_dims, num_outputs, B=None, W=None):

        self.means = means
        self.kernels = kernels
        self.cov_func = self._get_lcm(input_dim=input_dim, active_dims=active_dims, num_outputs=num_outputs, kernels=kernels)
        super().__init__(cov_func = self.cov_func)


    def _get_icm(self, input_dim, active_dims, num_outputs, kernel, W_rank=1, W=None, kappa=None, name='ICM'):
        """
        Builds a kernel for an Intrinsic Coregionalization Model (ICM)
        :input_dim: Input dimensionality (include the dimension of indices)
        :num_outputs: Number of outputs
        :kernel: kernel that will be multiplied by the coregionalize kernel (matrix B).
        :W_rank: number tuples of the corregionalization parameters 'W'
        :W: the W matrix
        :kappa:
        :name: The name of Intrinsic Coregionalization Model
        """
        if W is None:
            W = pm.Normal(f"{name}_W", mu=0, sigma=5, shape=(num_outputs,W_rank), 
                                                            initval=np.random.randn(num_outputs,W_rank))
        if kappa is None:
            kappa = pm.Gamma(f"{name}_kappa", alpha=5, beta=1, shape=num_outputs)        
        coreg = pm.gp.cov.Coregion(input_dim=input_dim, active_dims=active_dims, kappa=kappa, W=W)
        
        self.B = coreg.B
        # B = at.dot(W, W.T) + at.diag(kappa)
        # coreg = Coregionalize(input_dim=input_dim, active_dims=active_dims, kappa=kappa, W=W)
        # cov_func = pm.gp.cov.Kron([coreg, kernel])
        # return cov_func
        return coreg * kernel
        
        

    def _get_lcm(self, input_dim, active_dims, num_outputs, kernels, W_rank=1, W=None, kappa=None, name='ICM'):
        cov_func = 0
        for idx, kernel in enumerate(kernels):            
            icm = self._get_icm(input_dim, active_dims, num_outputs, kernel, W_rank, W, kappa, f'{name}_{idx}')
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

        