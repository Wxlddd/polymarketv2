import time
import numpy as np
import scipy.stats as stats
import logging
from config.settings import SystemConfig
from src.core.market_context import MarketContext
from src.strategies.merton_strategy import (
    MertonStrategy,
    HawkesMertonPricer,
    HawkesMertonCharacteristicFunction,
    MicrostructuralState
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BenchmarkPricing")

def legacy_calculate_probability(
    cf, S_t: float, K: float, tau_seconds: float,
    mu: float, sigma: float,
    lambda_plus: float, lambda_minus: float,
    mu_j_plus: float, sigma_j_plus: float,
    mu_j_minus: float, sigma_j_minus: float,
    limit: int = 150
) -> float:
    import scipy.integrate as integrate
    tau = tau_seconds / (365.25 * 24 * 3600)
    if tau_seconds <= 1.0:
        return 1.0 if S_t > K else (0.0 if S_t < K else 0.5)
        
    try:
        d2 = (np.log(S_t / K) + (mu - 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
        p_bs_analytical = float(stats.norm.cdf(d2))
    except Exception:
        p_bs_analytical = 1.0 if S_t > K else (0.0 if S_t < K else 0.5)
        
    if tau_seconds < 15.0:
        return p_bs_analytical
        
    x0 = np.log(S_t)
    
    def integrand(u: float) -> float:
        if u == 0.0:
            return 0.0
        
        # Evaluate characteristic function
        phi_merton = cf.phi(
            u=np.array([u]), S_t=S_t, K=K, tau=tau, mu=mu, sigma=sigma,
            lambda_plus=lambda_plus, lambda_minus=lambda_minus,
            mu_j_plus=mu_j_plus, sigma_j_plus=sigma_j_plus,
            mu_j_minus=mu_j_minus, sigma_j_minus=sigma_j_minus
        )[0]
        
        b_bs = mu - 0.5 * sigma**2
        diffusion_bs = 1j * u * x0 + 1j * u * b_bs * tau - 0.5 * (sigma**2) * (u**2) * tau
        phi_bs = np.exp(diffusion_bs)
        
        z = np.exp(-1j * u * np.log(K)) * (phi_merton - phi_bs)
        return np.imag(z) / u
        
    eps = 1e-8
    upper_bound = 15000.0 / (sigma * np.sqrt(max(tau, 1e-9)))
    upper_bound = max(100.0, min(upper_bound, 50000.0))
    
    try:
        result, _ = integrate.quad(integrand, eps, upper_bound, limit=limit, epsabs=1e-8, epsrel=1e-8)
        prob = p_bs_analytical + (1.0 / np.pi) * result
        return max(0.0, min(1.0, prob))
    except Exception as e:
        logger.error(f"Legacy quad integration failed: {e}")
        return p_bs_analytical

def main():
    print("======================================================================")
    print("    HIGH-FREQUENCY MERTON PRICER & HAWKES BENCHMARK AND VALIDATION   ")
    print("======================================================================")
    
    # 1. VERIFY BRANCHING RATIO SAFETY GUARD
    print("\n--- 1. Testing Branching Ratio Safety Guard ---")
    state = MicrostructuralState(lambda_0=4000.0)
    
    # Explosive Hawkes parameters: n = (8.0 + 4.0) / 5.0 = 2.4 >= 1.0 (highly explosive!)
    kappa_self_exp = 8.0
    kappa_cross_exp = 4.0
    beta_exp = 5.0
    
    print(f"Feeding explosive parameters: self={kappa_self_exp}, cross={kappa_cross_exp}, beta={beta_exp}")
    print("Expected: Spectral radius of 2.4 should be dynamically scaled to 0.8.")
    
    state.update_state(
        timestamp=100.0,
        ofi=5.0,
        kappa_self=kappa_self_exp,
        kappa_cross=kappa_cross_exp,
        beta=beta_exp,
        ofi_multiplier=-1e-6,
        use_local_drift=False
    )
    
    scaled_self = state._cached_kappa_self
    scaled_cross = state._cached_kappa_cross
    scaled_spectral_radius = (scaled_self + scaled_cross) / beta_exp
    
    print(f"Resulting parameters: self={scaled_self:.4f}, cross={scaled_cross:.4f}")
    print(f"Resulting Spectral Radius: {scaled_spectral_radius:.4f}")
    
    if abs(scaled_spectral_radius - 0.8) < 1e-7:
        print("[OK] Safety guard correctly scaled the explosive parameters to spectral radius 0.8.")
    else:
        print(f"[FAIL] Safety guard failed! Spectral radius is {scaled_spectral_radius:.4f} (expected 0.8)")

    # 2. VERIFY MATHEMATICAL CONVERGENCE
    print("\n--- 2. Testing Mathematical Accuracy (Legendre vs. Legacy Quad) ---")
    cf = HawkesMertonCharacteristicFunction()
    pricer = HawkesMertonPricer(n_nodes=64)
    
    # Sample market params
    S_t = 67520.0
    K = 67500.0
    tau_seconds = 180.0
    mu = -0.05
    sigma = 0.35
    lambda_plus = 5500.0
    lambda_minus = 4200.0
    mu_j_plus = 0.0001
    sigma_j_plus = 0.0015
    mu_j_minus = -0.0001
    sigma_j_minus = 0.0015
    
    # Calculate using legacy quad
    p_legacy = legacy_calculate_probability(
        cf, S_t, K, tau_seconds, mu, sigma,
        lambda_plus, lambda_minus, mu_j_plus, sigma_j_plus, mu_j_minus, sigma_j_minus
    )
    
    # Calculate using new vectorized Gauss-Legendre
    p_vectorized = pricer.calculate_probability(
        S_t, K, tau_seconds, mu, sigma,
        lambda_plus, lambda_minus, mu_j_plus, sigma_j_plus, mu_j_minus, sigma_j_minus
    )
    
    diff = abs(p_vectorized - p_legacy)
    print(f"Legacy Quad Probability   : {p_legacy:.8f}")
    print(f"Vectorized GL Probability : {p_vectorized:.8f}")
    print(f"Absolute Difference       : {diff:.8e}")
    
    if diff < 1e-6:
        print("[OK] Vectorized Gauss-Legendre matches legacy quad within tolerance (< 1e-6).")
    else:
        print("[FAIL] Mathematical discrepancy is too high!")

    # 3. COMPUTE PRICING LATENCY SPEEDUP
    print("\n--- 3. Pricing Latency Benchmark (1,000 runs) ---")
    runs = 1000
    
    # Run legacy benchmark
    t_start_legacy = time.perf_counter()
    for _ in range(runs):
        _ = legacy_calculate_probability(
            cf, S_t, K, tau_seconds, mu, sigma,
            lambda_plus, lambda_minus, mu_j_plus, sigma_j_plus, mu_j_minus, sigma_j_minus
        )
    t_end_legacy = time.perf_counter()
    legacy_total_ms = (t_end_legacy - t_start_legacy) * 1000.0
    legacy_avg_us = (legacy_total_ms / runs) * 1000.0
    
    # Run vectorized Gauss-Legendre benchmark
    t_start_vec = time.perf_counter()
    for _ in range(runs):
        _ = pricer.calculate_probability(
            S_t, K, tau_seconds, mu, sigma,
            lambda_plus, lambda_minus, mu_j_plus, sigma_j_plus, mu_j_minus, sigma_j_minus
        )
    t_end_vec = time.perf_counter()
    vec_total_ms = (t_end_vec - t_start_vec) * 1000.0
    vec_avg_us = (vec_total_ms / runs) * 1000.0
    
    speedup = legacy_avg_us / vec_avg_us
    print(f"Legacy quad average latency        : {legacy_avg_us:.2f} microseconds")
    print(f"Vectorized Gauss-Legendre latency  : {vec_avg_us:.2f} microseconds")
    print(f"Speedup Factor                     : {speedup:.1f}x")
    
    if speedup > 50.0:
        print(f"[OK] Computational speedup is outstanding (> 50x). Latency is safely sub-millisecond.")
    else:
        print(f"[WARNING] Speedup is {speedup:.1f}x. Check vectorization or CPU performance.")
        
    print("\n======================================================================")

if __name__ == "__main__":
    main()
