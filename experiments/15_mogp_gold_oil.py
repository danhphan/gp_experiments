import numpy as np
import pandas as pd
import pymc as pm

oil     = pd.read_csv('../data/gonu/brent-daily.csv')
oil.set_index("Date", inplace=True)

gold    = pd.read_csv('../data/gonu/lmba-gold-usd-am-daily.csv')
gold = gold.replace(".", np.nan)
gold.Price = gold.Price.astype(float)
gold.set_index("Date", inplace=True)

nasdaq  = pd.read_csv('../data/gonu/nasdaq.csv')
nasdaq  = nasdaq.rename(columns={"Adj Close":"Price"}) 
nasdaq = nasdaq[["Date", "Price"]]
nasdaq.set_index("Date", inplace=True)

usd     = pd.read_csv('../data/gonu/TWEXB.csv')
usd.set_index("Date", inplace=True)

print(oil.shape, gold.shape, nasdaq.shape, usd.shape)

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

    return X,Y,I[:,None]#slices


df_list = []
for df in [oil, gold, nasdaq, usd]:
    df = df.loc['2015-01-01':'2018-12-30']
    df = df.dropna()
    df_list.append(df)

X, Y, I = build_XY([df.reset_index().index.values[:, None] for df in df_list], 
                   [df.Price.values[:, None] for df in df_list])


with pm.Model() as model:
    ell = pm.Gamma("ell", alpha=2, beta=0.5)
    eta = pm.Gamma("eta", alpha=2, beta=0.5)
    cov = eta**2 * pm.gp.cov.ExpQuad(1, ls=ell, active_dims=[0])
    
    W = pm.Normal("W", mu=0, sigma=3, shape=(4,2), initval=np.random.randn(4,2))
    kappa = pm.Gamma("kappa", alpha=1.5, beta=1, shape=4)
    coreg = pm.gp.cov.Coregion(input_dim=2, active_dims=[1], kappa=kappa, W=W)
    cov_func = coreg * cov
    
    sigma = pm.HalfNormal("sigma", sigma=3)
    gp = pm.gp.Marginal(cov_func=cov_func)
    y_ = gp.marginal_likelihood("f", X, Y, noise=sigma)

from pymc.sampling_jax import sample_numpyro_nuts
with model:
    jtrace = sample_numpyro_nuts(500, chains=1, target_accept=0.9)