import numpy as np


def simulate_multi_asset_gbm(S0, drift, vol, corr_matrix, T=1.0, steps=252, seed=None):
    """
    Simulate multi-asset GBM paths under correlation.
    Returns array of shape (steps+1, num_assets).
    """
    if seed is not None:
        np.random.seed(seed)

    num_assets = len(S0)

    # Covariance from vol & correlation
    cov_matrix = np.diag(vol) @ corr_matrix @ np.diag(vol)

    dt = T / steps
    chol = np.linalg.cholesky(cov_matrix)

    paths = np.zeros((steps + 1, num_assets))
    paths[0] = S0

    for t in range(1, steps + 1):
        z = np.random.randn(num_assets)
        dz = chol @ z
        drift_term = (drift - 0.5 * vol ** 2) * dt
        diffusion_term = dz * np.sqrt(dt)

        paths[t] = paths[t - 1] * np.exp(drift_term + diffusion_term)

    return paths
