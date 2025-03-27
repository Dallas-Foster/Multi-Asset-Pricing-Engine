import numpy as np
import pandas as pd


def scenario_analysis(price_func, S0, strike, barrier, drift, vol, corr_matrix,
                      scenarios, T=1.0, steps=252, n_sims=5000, r=0.0):
    """
    Run scenario analysis, returning a DataFrame of scenario results.
    Each scenario can specify spot_shift (%), vol_shift (absolute), drift_shift (absolute).
    Example input:
      scenarios = [
        {"name": "SpotDown10", "spot_shift": -10},
        {"name": "VolUp5pts",  "vol_shift": 0.05},
      ]
    """
    results = []
    for sc in scenarios:
        S0_sc = S0.copy()
        drift_sc = drift.copy()
        vol_sc = vol.copy()

        # Spot shift (percentage)
        if "spot_shift" in sc:
            shift_pct = sc["spot_shift"]
            S0_sc = S0_sc * (1.0 + shift_pct / 100.0)

        # Vol shift (absolute)
        if "vol_shift" in sc:
            vol_sc = vol_sc + sc["vol_shift"]
            vol_sc = np.clip(vol_sc, 1e-6, None)  # avoid negative or zero vol

        # Drift shift (absolute)
        if "drift_shift" in sc:
            drift_sc = drift_sc + sc["drift_shift"]

        price_scenario = price_func(S0_sc, strike, barrier, drift_sc, vol_sc, corr_matrix,
                                    T=T, steps=steps, n_sims=n_sims, r=r, seed=1234)

        results.append({
            "Scenario": sc["name"],
            "Price": price_scenario
        })

    return pd.DataFrame(results)
