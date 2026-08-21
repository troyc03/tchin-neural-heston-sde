import numpy as np

from likelihood import log_likelihood, prior, PARAMETER_NAMES


def log_posterior(params, returns, dt, risk_free_rate=0.0):
    """Unnormalized log posterior: log p(theta | D)."""
    log_prior = prior(params)

    if not np.isfinite(log_prior):
        return -np.inf

    return log_prior + log_likelihood(
        params,
        returns,
        dt,
        risk_free_rate,
    )


def metropolis_hastings(
    returns,
    dt,
    initial_params=None,
    n_samples=10000,
    proposal_scale=None,
    risk_free_rate=0.0,
    burn_in=2000,
    random_seed=None,
):
    """Random-walk Metropolis-Hastings sampler for Heston parameters."""
    rng = np.random.default_rng(random_seed)

    if initial_params is None:
        initial_params = np.array([1.0, 0.04, 0.5, -0.7, 0.04])

    current = np.asarray(initial_params, dtype=float).copy()

    if proposal_scale is None:
        proposal_scale = np.array([0.10, 0.005, 0.05, 0.05, 0.005])

    proposal_scale = np.asarray(proposal_scale, dtype=float)
    current_log_posterior = log_posterior(
        current, returns, dt, risk_free_rate
    )

    if not np.isfinite(current_log_posterior):
        raise ValueError("Initial parameters have zero posterior density.")

    samples = np.empty((n_samples, len(PARAMETER_NAMES)))
    accepted = 0

    for i in range(n_samples):
        proposal = current + rng.normal(0.0, proposal_scale)
        proposal_log_posterior = log_posterior(
            proposal, returns, dt, risk_free_rate
        )

        if np.log(rng.uniform()) < proposal_log_posterior - current_log_posterior:
            current = proposal
            current_log_posterior = proposal_log_posterior
            accepted += 1

        samples[i] = current

    posterior_samples = samples[burn_in:]

    return {
        "samples": posterior_samples,
        "acceptance_rate": accepted / n_samples,
        "posterior_mean": dict(
            zip(PARAMETER_NAMES, posterior_samples.mean(axis=0))
        ),
        "posterior_std": dict(
            zip(PARAMETER_NAMES, posterior_samples.std(axis=0))
        ),
    }


def bayesian_inference(
    returns,
    dt,
    initial_params=None,
    n_samples=10000,
    burn_in=2000,
    risk_free_rate=0.0,
    random_seed=None,
):
    """Run Bayesian Heston inference using Metropolis-Hastings."""
    return metropolis_hastings(
        returns=returns,
        dt=dt,
        initial_params=initial_params,
        n_samples=n_samples,
        burn_in=burn_in,
        risk_free_rate=risk_free_rate,
        random_seed=random_seed,
    )


if __name__ == "__main__":
    returns = np.random.default_rng(42).normal(0.0, 0.02, size=1000)
    dt = 1.0 / 252.0

    result = bayesian_inference(
        returns,
        dt,
        n_samples=5000,
        burn_in=1000,
        random_seed=42,
    )

    print("Bayesian posterior means")
    for name, value in result["posterior_mean"].items():
        print(f"{name:>6}: {value:.6f}")

    print(f"\nAcceptance rate: {result['acceptance_rate']:.3f}")
