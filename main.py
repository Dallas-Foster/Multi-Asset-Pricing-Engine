import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_generation import generate_synthetic_data
from calibration import calibrate_parameters_from_data
from pricing import price_vanilla_call_mc, price_knock_out_barrier_call_mc
from greeks import compute_greeks
from scenario_analysis import scenario_analysis


def main():
    # 1. Generate synthetic data
    df_prices = generate_synthetic_data(num_assets=2, num_days=252, seed=92)

    # 2. Calibrate parameters
    annual_drift, annual_vol, corr_matrix = calibrate_parameters_from_data(df_prices)

    # Current spot is last row of synthetic prices
    S0 = df_prices.iloc[-1].values

    print("=== Synthetic Price Overview ===")
    print(df_prices.tail())
    print("\nCalibrated Annual Drift:", annual_drift)
    print("Calibrated Annual Volatility:", annual_vol)
    print("Calibrated Correlation Matrix:\n", corr_matrix)
    print("Current Spot Prices S0:", S0, "\n")

    # 3. Price a multi-asset vanilla call and a knock-out barrier call
    strike = 100.0
    barrier = 90.0
    r = 0.01
    T = 1.0
    steps = 100
    n_sims = 3000

    vanilla_price = price_vanilla_call_mc(S0, strike, annual_drift, annual_vol, corr_matrix,
                                          T=T, steps=steps, n_sims=n_sims, r=r, seed=1234)

    barrier_price = price_knock_out_barrier_call_mc(S0, strike, barrier, annual_drift, annual_vol,
                                                    corr_matrix, T=T, steps=steps, n_sims=n_sims,
                                                    r=r, seed=1234)

    print(f"Vanilla Call Price (avg payoff): {vanilla_price:.4f}")
    print(f"Knock-Out Barrier Call Price:    {barrier_price:.4f}\n")

    # 4. Compute Greeks for the barrier option
    delta, gamma, vega = compute_greeks(price_knock_out_barrier_call_mc,
                                        S0, strike, barrier,
                                        annual_drift, annual_vol, corr_matrix,
                                        T=T, steps=steps, n_sims=1000, r=r)

    print("=== Greeks for the Barrier Call (w.r.t. first asset) ===")
    print(f"Delta: {delta:.4f}")
    print(f"Gamma: {gamma:.4f}")
    print(f"Vega:  {vega:.4f}\n")

    # 5. Scenario Analysis
    scenarios = [
        {"name": "SpotDown10", "spot_shift": -10},
        {"name": "SpotUp10", "spot_shift": 10},
        {"name": "VolUp5pts", "vol_shift": 0.05},
        {"name": "VolDown5pts", "vol_shift": -0.05},
        {"name": "DriftUp2", "drift_shift": 0.02},
    ]

    scenario_df = scenario_analysis(price_knock_out_barrier_call_mc,
                                    S0, strike, barrier,
                                    annual_drift, annual_vol, corr_matrix,
                                    scenarios, T=T, steps=steps, n_sims=1000, r=r)

    print("=== Scenario Analysis ===")
    print(scenario_df)

    # 6. Plot the synthetic price history
    plt.figure()
    df_prices.plot(title="Synthetic Price Histories (2 Assets)")
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.show()


if __name__ == "__main__":
    main()
