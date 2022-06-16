import numpy as np
import numpy.testing as npt
import pymc as pm
from pymc.math import cartesian, kronecker

Xs = [
        np.linspace(0, 1, 7)[:, None],
        np.linspace(0, 1, 5)[:, None],
        np.linspace(0, 1, 6)[:, None],
    ]

X = cartesian(*Xs)
N = np.prod([len(X) for X in Xs])
y = np.random.randn(N)

Xnews = (np.random.randn(5, 1), np.random.randn(5, 1), np.random.randn(5, 1))
Xnew = np.concatenate(Xnews, axis=1)
pnew = np.random.randn(len(Xnew))

ls = 0.2

with pm.Model() as latent_model:
    cov_funcs = (
        pm.gp.cov.ExpQuad(1, ls),
        pm.gp.cov.ExpQuad(1, ls),
        pm.gp.cov.ExpQuad(1, ls),
    )
    cov_func = pm.gp.cov.Kron(cov_funcs)
    mean = pm.gp.mean.Constant(0.5)
    gp = pm.gp.Latent(mean_func=mean, cov_func=cov_func)
    f = gp.prior("f", X)
    p = gp.conditional("p", Xnew)

chol = np.linalg.cholesky(cov_func(X).eval())
y_rotated = np.linalg.solve(chol, y - 0.5)
logp = latent_model.compile_logp()({"f_rotated_": y_rotated, "p": pnew})


with pm.Model() as kron_model:
    kron_gp = pm.gp.LatentKron(mean_func=mean, cov_funcs=cov_funcs)
    f = kron_gp.prior("f", Xs)
    p = kron_gp.conditional("p", Xnew)
assert tuple(f.shape.eval()) == (X.shape[0],)
assert tuple(p.shape.eval()) == (Xnew.shape[0],)
kronlatent_logp = kron_model.compile_logp()({"f_rotated_": y_rotated, "p": pnew})
npt.assert_allclose(kronlatent_logp, logp, atol=0, rtol=1e-3)



# class TestLatentKron:
#     """
#     Compare gp.LatentKron to gp.Latent, both with Gaussian noise.
#     """

#     def setup_method(self):
#         self.Xs = [
#             np.linspace(0, 1, 7)[:, None],
#             np.linspace(0, 1, 5)[:, None],
#             np.linspace(0, 1, 6)[:, None],
#         ]
#         self.X = cartesian(*self.Xs)
#         self.N = np.prod([len(X) for X in self.Xs])
#         self.y = np.random.randn(self.N) * 0.1
#         self.Xnews = (np.random.randn(5, 1), np.random.randn(5, 1), np.random.randn(5, 1))
#         self.Xnew = np.concatenate(self.Xnews, axis=1)
#         self.pnew = np.random.randn(len(self.Xnew))
#         ls = 0.2
#         with pm.Model() as latent_model:
#             self.cov_funcs = (
#                 pm.gp.cov.ExpQuad(1, ls),
#                 pm.gp.cov.ExpQuad(1, ls),
#                 pm.gp.cov.ExpQuad(1, ls),
#             )
#             cov_func = pm.gp.cov.Kron(self.cov_funcs)
#             self.mean = pm.gp.mean.Constant(0.5)
#             gp = pm.gp.Latent(mean_func=self.mean, cov_func=cov_func)
#             f = gp.prior("f", self.X)
#             p = gp.conditional("p", self.Xnew)
#         chol = np.linalg.cholesky(cov_func(self.X).eval())
#         self.y_rotated = np.linalg.solve(chol, self.y - 0.5)
#         self.logp = latent_model.compile_logp()({"f_rotated_": self.y_rotated, "p": self.pnew})

#     def testLatentKronvsLatent(self):
#         with pm.Model() as kron_model:
#             kron_gp = pm.gp.LatentKron(mean_func=self.mean, cov_funcs=self.cov_funcs)
#             f = kron_gp.prior("f", self.Xs)
#             p = kron_gp.conditional("p", self.Xnew)
#         assert tuple(f.shape.eval()) == (self.X.shape[0],)
#         assert tuple(p.shape.eval()) == (self.Xnew.shape[0],)
#         kronlatent_logp = kron_model.compile_logp()({"f_rotated_": self.y_rotated, "p": self.pnew})
#         npt.assert_allclose(kronlatent_logp, self.logp, atol=0, rtol=1e-3)

#     def testLatentKronRaisesAdditive(self):
#         with pm.Model() as kron_model:
#             gp1 = pm.gp.LatentKron(mean_func=self.mean, cov_funcs=self.cov_funcs)
#             gp2 = pm.gp.LatentKron(mean_func=self.mean, cov_funcs=self.cov_funcs)
#         with pytest.raises(TypeError):
#             gp1 + gp2

#     def testLatentKronRaisesSizes(self):
#         with pm.Model() as kron_model:
#             gp = pm.gp.LatentKron(mean_func=self.mean, cov_funcs=self.cov_funcs)
#         with pytest.raises(ValueError):
#             gp.prior("f", Xs=[np.linspace(0, 1, 7)[:, None], np.linspace(0, 1, 5)[:, None]])
