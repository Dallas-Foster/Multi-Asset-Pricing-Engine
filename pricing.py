import numpy as np


def price_vanilla_call_mc(S0, strike, drift, vol, corr_matrix,
                          T=1.0, steps=252, n_sims=10000,
                          r=0.0, seed=None):
    """
    Price a multi-asset vanilla call option via Monte Carlo.
    Payoff = max(avg(S_final) - strike, 0).
    Discounted by e^{-rT}.
    """
    if seed is not None:
        np.random.seed(seed)

    num_assets = len(S0)
    cov_matrix = np.diag(vol) @ corr_matrix @ np.diag(vol)
    dt = T / steps
    chol = np.linalg.cholesky(cov_matrix)

    payoffs = []
    for _ in range(n_sims):
        S = S0.copy()
        for _ in range(steps):
            z = np.random.randn(num_assets)
            dz = chol @ z
            drift_term = (drift - 0.5 * vol ** 2) * dt
            diffusion_term = dz * np.sqrt(dt)
            S = S * np.exp(drift_term + diffusion_term)

        payoff = max(np.mean(S) - strike, 0.0)
        payoffs.append(payoff)

    discounted = np.exp(-r * T) * np.mean(payoffs)
    return discounted


def price_knock_out_barrier_call_mc(S0, strike, barrier, drift, vol, corr_matrix,
                                    T=1.0, steps=252, n_sims=10000,
                                    r=0.0, seed=None):
    """
    Price a multi-asset knock-out barrier call option.
    Option knocks out if avg(S) goes below 'barrier' at any time.
    Payoff = max(avg(S_final) - strike, 0) if not knocked out.
    """
    if seed is not None:
        np.random.seed(seed)

    num_assets = len(S0)
    cov_matrix = np.diag(vol) @ corr_matrix @ np.diag(vol)
    dt = T / steps
    chol = np.linalg.cholesky(cov_matrix)

    payoffs = []
    for _ in range(n_sims):
        S = S0.copy()
        knocked_out = False
        for _ in range(steps):
            z = np.random.randn(num_assets)
            dz = chol @ z
            drift_term = (drift - 0.5 * vol ** 2) * dt
            diffusion_term = dz * np.sqrt(dt)
            S = S * np.exp(drift_term + diffusion_term)

            if np.mean(S) <= barrier:
                knocked_out = True
                break

        if knocked_out:
            payoff = 0.0
        else:
            payoff = max(np.mean(S) - strike, 0.0)

        payoffs.append(payoff)

    discounted = np.exp(-r * T) * np.mean(payoffs)
    return discounted
