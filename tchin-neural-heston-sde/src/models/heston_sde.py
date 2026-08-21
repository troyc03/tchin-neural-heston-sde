import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from enum import Enum

# Define the missing OptionType Enum
class OptionType(Enum):
    CALL = 1
    PUT = 2

# Exact sample for the variance process (CIR)
def CIR_Sample(NoOfPaths, kappa, gamma, vbar, s, t, v_s):
    delta = 4.0 * kappa * vbar / (gamma ** 2)
    c = (gamma ** 2) / (4.0 * kappa) * (1.0 - np.exp(-kappa * (t - s)))
    kappaBar = 4.0 * kappa * v_s * np.exp(-kappa * (t - s)) / ((gamma ** 2) * (1.0 - np.exp(-kappa * (t - s))))
    
    # Avoid division by zero if kappaBar is practically zero
    kappaBar = np.maximum(kappaBar, 1e-10)
    sample = c * np.random.noncentral_chisquare(delta, kappaBar, NoOfPaths)
    return sample

# Standard Euler-Maruyama Discretization for Heston
def GeneratePathsHestonEuler(NoOfPaths, NoOfSteps, T, r, S_0, kappa, gamma, rho, vbar, v0):
    dt = T / float(NoOfSteps)
    X = np.zeros([NoOfPaths, NoOfSteps + 1])
    V = np.zeros([NoOfPaths, NoOfSteps + 1])
    X[:, 0] = np.log(S_0)
    V[:, 0] = v0
    
    time = np.linspace(0, T, NoOfSteps + 1)
    
    for i in range(NoOfSteps):
        # Generate correlated Wieneer processes
        Z1 = np.random.normal(0.0, 1.0, NoOfPaths)
        Z2 = np.random.normal(0.0, 1.0, NoOfPaths)
        
        # Standardize for better convergence properties
        Z1 = (Z1 - np.mean(Z1)) / np.std(Z1)
        Z2 = (Z2 - np.mean(Z2)) / np.std(Z2)
        
        # Correlated Brownian motion components
        ZW = rho * Z1 + np.sqrt(1.0 - rho**2) * Z2
        
        # Truncation scheme to prevent negative variance
        V[:, i+1] = V[:, i] + kappa * (vbar - V[:, i]) * dt + gamma * np.sqrt(np.maximum(V[:, i], 0.0)) * np.sqrt(dt) * Z1
        V[:, i+1] = np.maximum(V[:, i+1], 0.0)
        
        X[:, i+1] = X[:, i] + (r - 0.5 * V[:, i]) * dt + np.sqrt(V[:, i]) * np.sqrt(dt) * ZW
        
    return {"time": time, "S": np.exp(X)}

# Almost Exact Simulation (AES) Scheme for Heston
def GeneratePathsHestonAES(NoOfPaths, NoOfSteps, T, r, S_0, kappa, gamma, rho, vbar, v0):
    dt = T / float(NoOfSteps)
    X = np.zeros([NoOfPaths, NoOfSteps + 1])
    V = np.zeros([NoOfPaths, NoOfSteps + 1])
    X[:, 0] = np.log(S_0)
    V[:, 0] = v0
    
    time = np.linspace(0, T, NoOfSteps + 1)
    
    # Coefficients for the integrated variance approximation
    k0 = (r - rho / gamma * kappa * vbar) * dt
    k1 = (rho * kappa / gamma - 0.5) * dt - rho / gamma
    k2 = rho / gamma
    
    for i in range(NoOfSteps):
        Z1 = np.random.normal(0.0, 1.0, NoOfPaths)
        Z1 = (Z1 - np.mean(Z1)) / np.std(Z1)
        
        # Exact sample of V(t_{i+1}) given V(t_i)
        V[:, i+1] = CIR_Sample(NoOfPaths, kappa, gamma, vbar, 0, dt, V[:, i])
        
        # Variance proxy for the log-asset diffusion term
        # Standard Heston approximation uses the trapezoidal rule for the integral of V
        V_avg = 0.5 * (V[:, i] + V[:, i+1])
        
        X[:, i+1] = X[:, i] + k0 + k1 * V[:, i] + k2 * V[:, i+1] + np.sqrt((1.0 - rho**2) * V_avg) * np.sqrt(dt) * Z1
        
    return {"time": time, "S": np.exp(X)}

# Monte Carlo Pricing Evaluator
def EUOptionPriceFromMCPathsGeneralized(CP, S_T, K, T, r):
    discount_factor = np.exp(-r * T)
    prices = np.zeros_like(K)
    
    for idx, k in enumerate(K):
        if CP == OptionType.CALL:
            payoff = np.maximum(S_T - k, 0.0)
        elif CP == OptionType.PUT:
            payoff = np.maximum(k - S_T, 0.0)
        prices[idx] = discount_factor * np.mean(payoff)
        
    return prices

# Black-Scholes Formula (Used here as a benchmark reference since COS method isn't native)
def bs_call_price(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    # Changed st.norm.inverse_cdf to st.norm.cdf
    return S * st.norm.cdf(d1) - K * np.exp(-r * T) * st.norm.cdf(d2) 

def mainCalculation():
    NoOfPaths = 2000
    NoOfSteps = 200
    
    gamma = 0.3
    kappa = 1.5
    vbar = 0.04
    rho = -0.6
    v0 = 0.04
    T = 1.0
    S_0 = 100.0
    r = 0.05
    CP = OptionType.CALL
    
    K = np.linspace(50.0, S_0 * 1.5, 15)
    
    # Benchmarking using long-term average volatility for Black-Scholes reference mapping
    optValueExact = np.array([bs_call_price(S_0, k, T, r, np.sqrt(vbar)) for k in K])
    
    # Run Schemes
    pathsEULER = GeneratePathsHestonEuler(NoOfPaths, NoOfSteps, T, r, S_0, kappa, gamma, rho, vbar, v0)
    S_Euler = pathsEULER["S"]
    
    pathsAES = GeneratePathsHestonAES(NoOfPaths, NoOfSteps, T, r, S_0, kappa, gamma, rho, vbar, v0)
    S_AES = pathsAES["S"]
    
    OptPrice_EULER = EUOptionPriceFromMCPathsGeneralized(CP, S_Euler[:, -1], K, T, r)
    OptPrice_AES = EUOptionPriceFromMCPathsGeneralized(CP, S_AES[:, -1], K, T, r)
    
    # Plotting code mapping
    plt.style.use('dark_background')
    plt.figure(figsize=(10, 5))
    plt.plot(K, optValueExact, '-r', label='Benchmark Reference')
    plt.plot(K, OptPrice_EULER, '--k', label='Euler Scheme')
    plt.plot(K, OptPrice_AES, '.b', label='AES Scheme')
    plt.legend()
    plt.grid(True)
    plt.xlabel('Strike, K')
    plt.ylabel('Option Price')
    plt.title('Heston Model Discretization Comparison')
    plt.show()

if __name__ == '__main__':
    mainCalculation()
