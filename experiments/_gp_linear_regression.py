# https://discourse.pymc.io/t/gp-and-bayesian-linear-regression-in-pymc/1678/3

import numpy as np
import pandas as pd
import pymc3 as pm

data = pd.read_csv("../data/sample_data.csv")

print(data.shape)

# X = data[['x1', 'x2']].T.values
# X_gp = data[['gp0', 'gp1']].values
# y = data.y.values

X = data[['x1', 'x2', 'gp1', 'gp2']]
y = data.y.values


with pm.Model() as model_mlr:
    ## Bayesian linear regression
    α_blr = pm.Normal('α_blr', mu=0, sd=10)
    β_blr = pm.Normal('β_blr', mu=0, sd=1, shape=2)
    σ = pm.HalfCauchy('σ', 5)
    μ_blr = α_blr + pm.math.dot([β_blr[0],β_blr[1], 0, 0] , X)


    ## The spatial GP
    η_spatial_trend = pm.Uniform('η_spatial_trend', lower=0, upper=100)
    ℓ_spatial_trend = pm.Uniform('ℓ_spatial_trend', lower=0, upper=100)
    cov_spatial_trend = (
        η_spatial_trend**2 * pm.gp.cov.ExpQuad(input_dim=4, ls=ℓ_spatial_trend, active_dims=[2,3])
    )

    cov_noise = pm.gp.cov.WhiteNoise(σ)

    gp = pm.gp.Margin(mean_func=μ_blr, cov_func=cov_spatial_trend)
    y_ = gp.marginal_likelihood('y', X=X, y=y, noise=cov_noise)
    
    
    trace_mlr = pm.sample()