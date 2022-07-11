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


class LCM2(Covariance):

    def __init__(self, input_dim, active_dims, kernel_list, n_outputs):
        super().__init__(input_dim, active_dims)
        self.cov_func = 0
        for idx, kernel in enumerate(kernel_list):
            kappa = pm.Gamma(f"kappa_{idx}", alpha=1.5, beta=1, shape=n_outputs)
            W = pm.Normal(f"W_{idx}", mu=0, sigma=3, shape=(n_outputs,len(cov_list)), 
                                                            initval=np.random.randn(n_outputs,len(cov_list)))
            coreg = pm.gp.cov.Coregion(input_dim=input_dim, active_dims=active_dims, kappa=kappa, W=W)
            cov_func_ = coreg * kernel
            self.cov_func = pm.gp.cov.Add([self.cov_func, cov_func_])

    def __call__(self):
        return self.cov_func