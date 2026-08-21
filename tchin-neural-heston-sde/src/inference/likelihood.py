import numpy as np

PARAMETER_NAMES = [
    "kappa",
    "theta",
    "sigma",
    "rho",
    "v0"
]

def prior(params=None):
    if params is None:
        return {
            "kappa": {"distribution": "lognormal", "mean": 1.0, "std": 1.0},
            "theta": {"distribution": "lognormal", "mean": 0.04, "std": 0.02},
            "sigma": {"distribution": "lognormal", "mean": 0.5, "std": 0.25},
            "rho": {"distribution": "uniform", "low": -1.0, "high": 1.0},
            "v0": {"distribution": "lognormal", "mean": 0.04, "std": 0.02},
        }
    kappa, theta, sigma, rho, v0 = np.asarray(params)

    if kappa <= 0 or theta <= 0 or sigma <= 0 or v0 <= 0:
        return -np.inf

    if not -1.0 < rho < 1.0:
        return -np.inf

    def lognormal_logpdf(x, mean, std):
        variance = np.log(1.0 + (std / mean) ** 2)
        mu = np.log(mean) - 0.5 * variance

        return (
            -np.log(x)
            - 0.5 * np.log(2.0 * np.pi * variance)
            - (np.log(x) - mu) ** 2 / (2.0 * variance)
        )

    log_p = 0.0

    log_p += lognormal_logpdf(kappa, 1.0, 1.0)
    log_p += lognormal_logpdf(theta, 0.04, 0.02)
    log_p += lognormal_logpdf(sigma, 0.5, 0.25)
    log_p += lognormal_logpdf(v0, 0.04, 0.02)

    # Uniform prior for correlation.
    log_p += -np.log(2.0)

    return log_p

def log_likelihood(params, returns, dt, risk_free_rate=0.0):
    kappa, theta, sigma, rho, v0 = np.asarray(params)

    if kappa <= 0 or theta <= 0 or sigma <= 0 or v0 <= 0:
        return -np.inf

    if not -1.0 < rho < 1.0:
        return -np.inf

    returns = np.asarray(returns)

    # Initial variance.
    variance = np.empty(len(returns))
    variance[0] = v0

    for t in range(1, len(returns)):
        variance[t] = (
            variance[t - 1]
            + kappa * (theta - variance[t - 1]) * dt
        )

        variance[t] = max(variance[t], 1e-10)

    mean = (
        risk_free_rate
        - 0.5 * variance
    ) * dt

    variance_return = variance * dt

    residual = returns - mean

    log_likelihood = -0.5 * np.sum(
        np.log(2.0 * np.pi * variance_return)
        + residual**2 / variance_return
    )

    return log_likelihood
    