import numpy as np

from s2r.nodes.esn_engine import EchoStateNetwork


def test_esn_fit_and_infer():
    rng = np.random.default_rng(0)
    T, d = 400, 4
    # Target is a delayed/smoothed transform of input
    u = rng.normal(size=(T, d))
    y = np.zeros_like(u)
    for t in range(1, T):
        y[t] = 0.8 * y[t - 1] + 0.2 * u[t]
    esn = EchoStateNetwork(n_inputs=d, n_outputs=d, reservoir_size=120, seed=0)
    mse = esn.fit_ridge(u, y, washout=30, reg=1e-5)
    assert mse < 0.5
    out = esn.update(u[-1])
    assert out.shape == (d,)


def test_esn_save_load(tmp_path):
    esn = EchoStateNetwork(n_inputs=3, n_outputs=3, reservoir_size=40, seed=1)
    path = tmp_path / "esn.npz"
    esn.save(path)
    loaded = EchoStateNetwork.load(path)
    u = np.array([0.1, -0.2, 0.3])
    a = esn.update(u)
    b = loaded.update(u)
    assert np.allclose(a, b)
