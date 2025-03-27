# Multi-Asset-Pricing-Engine
fully functional, modular Python project for simulating and pricing multi-asset derivatives with advanced risk analytics.

# Key Features
Synthetic Data Generation
Automatically creates correlated price paths for multiple assets using a controlled covariance structure.

Parameter Calibration
Calibrates drift, volatility, and correlation from historical (or synthetic) price data.

Correlated GBM Simulations
Implements Monte Carlo simulations for multiple underlyings, capturing realistic correlations via a Cholesky-based approach.

# Pricing Engine

Vanilla Basket Call: Pays on the average of multiple underlying assets.

Knock-Out Barrier: Demonstrates exotic payoff logic, knocking out if the average asset price breaches a predefined barrier.

Risk Analytics

Greeks: Delta, Gamma, and Vega via finite-difference approximations.

Scenario Analysis: Evaluate the option price under user-defined shifts in spot, volatility, or drift.

Modular Architecture
The project is split into multiple Python files (data_generation.py, calibration.py, pricing.py, etc.) for clarity and maintainability.
