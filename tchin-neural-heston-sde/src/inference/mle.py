import numpy as np
from scipy.optimize import minimize
from likelihood import log_likelihood, PARAMETER_NAMES


def negative_log_likelihood(params, returns, dt, risk_free_rate=0.0):
    """Negative log-likelihood for numerical minimization."""
    value = log_likelihood(params, returns, dt, risk_free_rate)

    if not np.isfinite(value):
        return 1e100

    return -value


def maximum_likelihood(
    returns,
    dt,
    initial_params=None,
    risk_free_rate=0.0,
):
    """Estimate Heston parameters by maximum likelihood."""
    if initial_params is None:
        initial_params = np.array([1.0, 0.04, 0.5, -0.7, 0.04])

    initial_params = np.asarray(initial_params, dtype=float)

    bounds = [
        (1e-8, None),   # kappa
        (1e-8, None),   # theta
        (1e-8, None),   # sigma
        (-0.999999, 0.999999),  # rho
        (1e-8, None),   # v0
    ]

    result = minimize(
        negative_log_likelihood,
        initial_params,
        args=(returns, dt, risk_free_rate),
        method="L-BFGS-B",
        bounds=bounds,
    )

    estimates = dict(zip(PARAMETER_NAMES, result.x))

    return {
        "parameters": estimates,
        "log_likelihood": -result.fun,
        "success": result.success,
        "message": result.message,
        "optimizer_result": result,
    }


if __name__ == "__main__":
    np.random.seed(42)

    returns = np.random.normal(0.0, 0.02, size=1000)
    dt = 1.0 / 252.0

    result = maximum_likelihood(returns, dt)

    print("MLE estimates")
    for name, value in result["parameters"].items():
        print(f"{name:>6}: {value:.6f}")

    print(f"\nLog-likelihood: {result['log_likelihood']:.6f}")
    print(f"Success: {result['success']}")
