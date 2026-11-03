"""The JAX stack the image carries (the `jax` extra): double precision, a
gradient, and a short run of each sampler on a model with a known posterior.
Runs on whatever device JAX finds (the CPU in CI, a GPU under --nv)."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
numpyro = pytest.importorskip("numpyro")
blackjax = pytest.importorskip("blackjax")

import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)


def test_double_precision_and_gradient():
    x = jnp.arange(3.0)
    assert x.dtype == jnp.float64
    g = jax.grad(lambda v: jnp.sum(jnp.log1p(v ** 2)))(x)
    np.testing.assert_allclose(np.asarray(g), 2 * np.arange(3.0) / (1 + np.arange(3.0) ** 2), rtol=1e-14)


def test_float64_linear_algebra_matches_numpy():
    a = np.random.default_rng(0).normal(size=(40, 40))
    s = a @ a.T + 40 * np.eye(40)
    np.testing.assert_allclose(np.asarray(jnp.linalg.cholesky(s)), np.linalg.cholesky(s), rtol=1e-12, atol=1e-12)


def test_numpyro_nuts_recovers_a_normal_mean():
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS

    y = np.random.default_rng(1).normal(2.0, 1.0, size=200)

    def model(y):
        mu = numpyro.sample("mu", dist.Normal(0.0, 10.0))
        numpyro.sample("y", dist.Normal(mu, 1.0), obs=y)

    mcmc = MCMC(NUTS(model), num_warmup=300, num_samples=500, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(0), y=jnp.asarray(y))
    mu = np.asarray(mcmc.get_samples()["mu"])
    # posterior N(mean(y) * 200 / 200.01, 1 / sqrt(200.01))
    assert abs(mu.mean() - y.mean()) < 4 * (1 / np.sqrt(200)) / np.sqrt(100)
    assert 0.05 < mu.std() < 0.09


def test_blackjax_nuts_runs_with_window_adaptation():
    logdensity = lambda x: -0.5 * jnp.sum(x ** 2)          # standard normal in 2-d
    warmup = blackjax.window_adaptation(blackjax.nuts, logdensity)
    (state, params), _ = warmup.run(jax.random.PRNGKey(0), jnp.ones(2), num_steps=300)
    kernel = blackjax.nuts(logdensity, **params).step

    def step(state, key):
        state, _ = kernel(key, state)
        return state, state.position

    _, xs = jax.lax.scan(step, state, jax.random.split(jax.random.PRNGKey(1), 1000))
    xs = np.asarray(xs)
    assert xs.shape == (1000, 2)
    assert np.all(np.abs(xs.mean(0)) < 0.25) and np.all(np.abs(xs.std(0) - 1) < 0.15)
