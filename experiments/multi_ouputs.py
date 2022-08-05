import numpy as np
import pymc as pm
from pymc.gp.cov import Covariance


def build_XY(input_list,output_list=None,index=None):
    num_outputs = len(input_list)
    if output_list is not None:
        assert num_outputs == len(output_list)
        Y = np.vstack(output_list)
    else:
        Y = None

    if index is not None:
        assert len(index) == num_outputs
        I = np.hstack( [np.repeat(j,_x.shape[0]) for _x,j in zip(input_list,index)] )
    else:
        I = np.hstack( [np.repeat(j,_x.shape[0]) for _x,j in zip(input_list,range(num_outputs))] )

    X = np.vstack(input_list)
    X = np.hstack([X,I[:,None]])

    return X,Y,I[:,None] #slices


class ICM(Covariance):

    def __init__(self, input_dim, active_dims, num_outputs, kernel, W_rank=1, W=None, kappa=None, name='ICM'):
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
        super().__init__(input_dim, active_dims)
        if W is None:
            W = pm.Normal(f"{name}_W", mu=0, sigma=5, shape=(num_outputs,W_rank), 
                                                            initval=np.random.randn(num_outputs,W_rank))
        if kappa is None:
            kappa = pm.Gamma(f"{name}_kappa", alpha=5, beta=1, shape=num_outputs)        
        coreg = pm.gp.cov.Coregion(input_dim=input_dim, active_dims=active_dims, kappa=kappa, W=W)
        
        self.cov_func = coreg * kernel

    def __call__(self):
        return self.cov_func
    
   
class LCM(Covariance):

    def __init__(self, input_dim, active_dims, num_outputs, kernel_list, W_rank=1, W=None, kappa=None, name='ICM'):
        super().__init__(input_dim, active_dims)
        self.cov_func = 0
        for idx, kernel in enumerate(kernel_list):            
            icm = ICM(input_dim, active_dims, num_outputs, kernel, W_rank, W, kappa, f'{name}_{idx}')
            self.cov_func += icm()

    def __call__(self):
        return self.cov_func


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

from pymc.gp.gp import Marginal

class MultiMarginal(Marginal):

    def __init__(self, mean_list, cov_list, type='lcm'):

        self.mean_list = mean_list
        self.cov_list = cov_list
        self.cov_func = self._get_lcm(input_dim=2, active_dims=[1], num_outputs=3, kernel_list=cov_list)
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
        
        return coreg * kernel
        

    def _get_lcm(self, input_dim, active_dims, num_outputs, kernel_list, W_rank=1, W=None, kappa=None, name='ICM'):
        cov_func = 0
        for idx, kernel in enumerate(kernel_list):            
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

