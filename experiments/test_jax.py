import numpy as np
import pymc as pm
import aesara as ae
import aesara.tensor as at
from pymc.sampling_jax import sample_numpyro_nuts
import os
# os.environ["JAX_PLATFORM_NAME"] = "cpu"
import jax

coords = {
    "obs_id": [0, 1, 2, 3, 4],
}
with pm.Model(coords=coords) as rugby_model:
    item_idx = pm.Data("item_idx", [0, 1, 2, 3, 4], dims="obs_id", mutable=False)
    b = ae.shared(0.1)
    obs = np.random.normal(10, 2, size=100)
    c = ae.shared(obs, borrow=True, name="obs")
    a = pm.Normal("a", 0.0, sigma=10.0, shape=5)

    theta = a[item_idx]
    sigma = pm.HalfCauchy("error", 0.5)

    y = pm.Normal("y", theta, sigma=sigma, observed=[3, 2, 6, 8, 4])

    idata = pm.sample()
    idata_jax = sample_numpyro_nuts(tune=1000, chains=2, target_accept=0.9)
    print(jax.devices())
