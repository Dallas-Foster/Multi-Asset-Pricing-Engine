import numpy as np
from pricing import price_vanilla_call_mc, price_knock_out_barrier_call_mc


def compute_greeks(price_func, S0, strike, barrier, drift, vol, corr_matrix,
                   T=1.0, steps=252, n_sims=5000, r=0.0, epsilon=1e-2):
    """
    Compute Delta, Gamma, Vega for an option via finite differences.
    price_func must accept the same signature as in 'pricing.py'.

    We'll do:
      - Delta w.r.t. the first asset's spot
      - Gamma w.r.t. the first asset's spot
      - Vega w.r.t. the first asset's vol
    """
    # Baseline price
    base_price = price_func(S0, strike, barrier, drift, vol, corr_matrix,
                            T=T, steps=steps, n_sims=n_sims, r=r, seed=1234)

    # Delta
    S0_up = S0.copy()
    S0_down = S0.copy()
    S0_up[0] += epsilon
    S0_down[0] -= epsilon

    price_up = price_func(S0_up, strike, barrier, drift, vol, corr_matrix,
                          T=T, steps=steps, n_sims=n_sims, r=r, seed=1234)
    price_down = price_func(S0_down, strike, barrier, drift, vol, corr_matrix,
                            T=T, steps=steps, n_sims=n_sims, r=r, seed=1234)

    delta = (price_up - price_down) / (2 * epsilon)
    gamma = (price_up - 2 * base_price + price_down) / (epsilon ** 2)

    # Vega (adjust first asset's volatility)
    vol_up = vol.copy()
    vol_down = vol.copy()
    vol_up[0] += epsilon * 0.01
    vol_down[0] -= epsilon * 0.01

    price_vol_up = price_func(S0, strike, barrier, drift, vol_up, corr_matrix,
                              T=T, steps=steps, n_sims=n_sims, r=r, seed=1234)
    price_vol_down = price_func(S0, strike, barrier, drift, vol_down, corr_matrix,
                                T=T, steps=steps, n_sims=n_sims, r=r, seed=1234)

    vega = (price_vol_up - price_vol_down) / (2 * (epsilon * 0.01))

    return delta, gamma, vega
