import numpy as np
import pandas as pd


def generate_synthetic_data(num_assets=2, num_days=252, seed=42):
    """
    Generates synthetic daily returns for `num_assets` assets over `num_days`.
    Uses a random covariance matrix with a known correlation structure.
    Returns a DataFrame with columns as assets and rows as daily prices.
    """
    np.random.seed(seed)

    # Random correlation-ish matrix construction
    rand_matrix = np.random.randn(num_assets, num_assets)
    cov_matrix = np.dot(rand_matrix, rand_matrix.T)

    # Scale covariance to a typical daily returns level
    cov_matrix = cov_matrix / np.max(np.abs(cov_matrix)) * 0.0004

    # Cholesky factor for correlation
    chol = np.linalg.cholesky(cov_matrix)

    # We'll assume a small positive drift
    drift = 0.0002  # 2 bps per day

    # Generate correlated returns
    random_norms = np.random.randn(num_days, num_assets)
    correlated_returns = random_norms @ chol.T + drift

    # Convert returns to price levels (start at 100)
    start_price = 100
    prices = start_price * np.exp(np.cumsum(correlated_returns, axis=0))

    # Create DataFrame
    columns = [f"Asset_{i + 1}" for i in range(num_assets)]
    df_prices = pd.DataFrame(prices, columns=columns)

    return df_prices
