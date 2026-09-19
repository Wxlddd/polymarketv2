import numpy as np
import scipy.integrate as integrate
import scipy.stats as stats
import logging
from typing import Dict, Any, Tuple, Optional
from src.core.base_strategy import BaseStrategy
from src.core.market_context import MarketContext

logger = logging.getLogger("MertonStrategy")

class HighFrequencyVolatilityCalibrator:
    """
    Tracks and calibrates continuous realized volatility using rolling price ticks.
    """
    
    def __init__(self, window_size: int = 300, min_ticks: int = 10):
        self.window_size = window_size
        self.min_ticks = min_ticks
        self.prices: list[float] = []
        self.timestamps: list[float] = []
        
    def add_tick(self, price: float, timestamp: float) -> None:
        """
        Appends a new price tick at controlled >= 5-second intervals.
        Filters bid-ask bounce and applies an outlier log-return filter.
        Resets buffer if a massive price jump is detected to prevent volatility spikes.
        """
        if not self.timestamps or (timestamp - self.timestamps[-1] >= 5.0):
            if self.prices:
                last_price = self.prices[-1]
                log_ret = np.log(price / last_price)
                
                # If a massive price jump (> 1.5% in 5 seconds) is detected, reset buffer.
                if abs(log_ret) > 0.015:
                    logger.info(
                        f"Volatility Calibrator: detected price jump of {log_ret:.2%}. "
                        f"Resetting buffer to prevent volatility spike."
                    )
                    self.prices = [price]
                    self.timestamps = [timestamp]
                    return
                
                # Cap log-return to +/- 0.5% (0.005) to filter out outliers
                log_ret_capped = np.clip(log_ret, -0.005, 0.005)
                filtered_price = last_price * np.exp(log_ret_capped)
            else:
                filtered_price = price
                
            self.prices.append(filtered_price)
            self.timestamps.append(timestamp)
            
            # Evict ticks older than window_size
            cutoff = timestamp - self.window_size
            while self.timestamps and self.timestamps[0] < cutoff:
                self.timestamps.pop(0)
                self.prices.pop(0)
            
    def calculate_volatility(self, default_annual_vol: float) -> float:
        """
        Calculates annualized realized volatility from log-returns.
        """
        n = len(self.prices)
        if n < self.min_ticks:
            return default_annual_vol
            
        prices_arr = np.array(self.prices)
        log_returns = np.diff(np.log(prices_arr))
        
        # Standard deviation of log-returns
        std_dev = np.std(log_returns)
        
        # Duration of window in seconds
        duration = self.timestamps[-1] - self.timestamps[0]
        if duration <= 1.0:
            return default_annual_vol
            
        # Annualized scaling: std_dev * sqrt(average_ticks_per_year)
        ticks_per_second = n / duration
        seconds_in_year = 365.25 * 24 * 3600
        annualization_factor = np.sqrt(ticks_per_second * seconds_in_year)
        
        sigma_est = std_dev * annualization_factor
        
        # Apply sanity boundaries [15%, 300%] to prevent spike pathologies
        return float(np.clip(sigma_est, 0.15, 3.0))


def update_bivariate_hawkes_intensities(
    prev_lambda_plus: float,
    prev_lambda_minus: float,
    lambda_0: float,
    kappa_self: float,
    kappa_cross: float,
    beta: float,
    dt: float,
    ofi_shock_plus: float,
    ofi_shock_minus: float
) -> Tuple[float, float]:
    """
    Computes recursive updates for a symmetric bivariate Hawkes intensity process
    with cross-excitation. Designed for high performance and potential C++ export.
    """
    decay_factor = np.exp(-beta * dt)
    excess_plus = (prev_lambda_plus - lambda_0) * decay_factor
    excess_minus = (prev_lambda_minus - lambda_0) * decay_factor
    
    next_lambda_plus = lambda_0 + excess_plus + kappa_self * ofi_shock_plus + kappa_cross * ofi_shock_minus
    next_lambda_minus = lambda_0 + excess_minus + kappa_self * ofi_shock_minus + kappa_cross * ofi_shock_plus
    
    return next_lambda_plus, next_lambda_minus


class MicrostructuralState:
    """
    Manages and updates the microstructural state vector [lambda_plus, lambda_minus, mu]
    tick-by-tick based on Order Flow Imbalance (OFI) and time elapsed.
    Uses a coupled symmetric bivariate Hawkes process for cross-excitation.
    """
    
    def __init__(self, lambda_0: float):
        self.lambda_0 = lambda_0
        self.last_timestamp: float = 0.0
        self.lambda_plus: float = lambda_0
        self.lambda_minus: float = lambda_0
        self.mu: float = 0.0
        self._stationarity_checked: bool = False
        self._cached_kappa_self: float = 0.0
        self._cached_kappa_cross: float = 0.0
        
    def reset(self) -> None:
        """Resets intensities and drift back to baseline values."""
        self.last_timestamp = 0.0
        self.lambda_plus = self.lambda_0
        self.lambda_minus = self.lambda_0
        self.mu = 0.0
        
    def update_state(
        self,
        timestamp: float,
        ofi: float,
        kappa_self: float,
        kappa_cross: float,
        beta: float,
        ofi_multiplier: float,
        use_local_drift: bool,
        r: float = 0.0
    ) -> None:
        """
        Updates the stochastic intensities and drift on each price/CLOB tick using Bivariate Hawkes.
        """
        # Validate Hawkes stationarity on the first tick (avoiding high-frequency tick log spam)
        if not self._stationarity_checked:
            total_kappa = kappa_self + kappa_cross
            if total_kappa >= beta:
                target_total = 0.8 * beta
                scale_factor = target_total / total_kappa
                scaled_self = kappa_self * scale_factor
                scaled_cross = kappa_cross * scale_factor
                logger.warning(
                    f"Hawkes Branching Ratio is explosive (spectral radius = {total_kappa / beta:.2f} >= 1.0)! "
                    f"Applying mathematical safety guard: automatically scaled "
                    f"KAPPA_SELF to {scaled_self:.4f} and KAPPA_CROSS to {scaled_cross:.4f} "
                    f"to force stability (spectral radius = 0.8)."
                )
                kappa_self = scaled_self
                kappa_cross = scaled_cross
            self._stationarity_checked = True
            self._cached_kappa_self = kappa_self
            self._cached_kappa_cross = kappa_cross
        else:
            # Use cached (potentially scaled) values
            kappa_self = self._cached_kappa_self
            kappa_cross = self._cached_kappa_cross

        if self.last_timestamp == 0.0:
            # Initialization tick
            self.last_timestamp = timestamp
            self.lambda_plus = self.lambda_0
            self.lambda_minus = self.lambda_0
            self.mu = r
            return
            
        dt = timestamp - self.last_timestamp
        
        # If time went backwards (e.g. due to backtesting clock jitter or duplicate ticks), cap dt at 0.0
        if dt < 0.0:
            dt = 0.0
            
        # Split OFI pressure into positive and negative shocks
        ofi_shock_plus = max(ofi, 0.0)
        ofi_shock_minus = max(-ofi, 0.0)
        
        # Coupled Bivariate Hawkes updates
        self.lambda_plus, self.lambda_minus = update_bivariate_hawkes_intensities(
            prev_lambda_plus=self.lambda_plus,
            prev_lambda_minus=self.lambda_minus,
            lambda_0=self.lambda_0,
            kappa_self=kappa_self,
            kappa_cross=kappa_cross,
            beta=beta,
            dt=dt,
            ofi_shock_plus=ofi_shock_plus,
            ofi_shock_minus=ofi_shock_minus
        )
        
        # Enforce mathematical lower bounds on intensities to prevent zero/negative values
        self.lambda_plus = max(1e-9, self.lambda_plus)
        self.lambda_minus = max(1e-9, self.lambda_minus)
        
        # Update diffusive drift component
        if use_local_drift:
            self.mu = r + ofi_multiplier * ofi
        else:
            self.mu = r
            
        # Update timestamp for next tick
        self.last_timestamp = timestamp


SECONDS_PER_YEAR = 365.25 * 24 * 3600.0


def integrated_expected_intensity(
    lambda_current: float, lambda_0: float, beta: float, tau_seconds: float
) -> float:
    r"""
    Time-integrates the expected future path of a mean-reverting Hawkes intensity
    over the remaining horizon, instead of naively multiplying the instantaneous
    (possibly transient) intensity by the entire remaining time.

    Absent further shocks, the intensity relaxes exponentially back to baseline:
    $E[\lambda(t+s)] = \lambda_0 + (\lambda(t) - \lambda_0)e^{-\beta s}$. Integrating
    over the remaining horizon $\tau$ gives the expected jump count:
    $\int_0^\tau E[\lambda(t+s)]\,ds = \lambda_0 \tau + (\lambda(t)-\lambda_0)\frac{1-e^{-\beta\tau}}{\beta}$

    This bounds the contribution of a fleeting OFI-driven excitation (which decays
    with half-life ~ln(2)/beta, typically well under a second) to at most
    (lambda_current - lambda_0) / beta "extra" expected jumps, regardless of how
    far away expiry is — a single microstructure tick can no longer be amplified
    by the full remaining time-to-expiry as it was in the naive lambda*tau form.

    Returns the expected jump count already converted to "years" so it can be used
    as a drop-in replacement for lambda*tau in the characteristic function
    (lambda is expressed in jumps/year, beta and tau_seconds in real seconds).
    """
    if beta <= 1e-12:
        # No mean reversion: degenerates to the naive instantaneous*tau form.
        return lambda_current * (tau_seconds / SECONDS_PER_YEAR)
    excess = lambda_current - lambda_0
    integral_seconds_domain = lambda_0 * tau_seconds + excess * (1.0 - np.exp(-beta * tau_seconds)) / beta
    return integral_seconds_domain / SECONDS_PER_YEAR


class HawkesMertonCharacteristicFunction:
    """
    Computes the characteristic function for the Hawkes-Driven Merton Jump-Diffusion model
    with split positive and negative jump processes.
    """

    def phi(
        self, u: np.ndarray, S_t: float, K: float, tau: float, mu: float, sigma: float,
        lambda_plus: float, lambda_minus: float,
        mu_j_plus: float, sigma_j_plus: float,
        mu_j_minus: float, sigma_j_minus: float,
        tau_seconds: float = 0.0, beta: float = 0.0, lambda_0: float = 0.0
    ) -> np.ndarray:
        x0 = np.log(S_t)

        # MATHEMATICAL NOTE ON MARTINGALE COMPENSATOR (P-MEASURE FIX):
        # Under standard Gil-Pelaez option pricing, we use a Q-measure Risk-Neutral world
        # where the compensator (lambda * kappa) offsets jumps to maintain a martingale.
        # However, for prediction markets we predict real-world probabilities (P-measure).
        # We WANT the asymmetric jumps to shift the asset drift directionally.
        # Using a Q-measure compensator caused an "Inversion Bug" where positive jumps
        # dragged continuous drift negatively, making p_hat drop instead of rise.
        # We remove the jump compensators to properly model the real-world drift.
        b = mu - 0.5 * sigma**2
        # Continuous diffusion part
        diffusion = 1j * u * x0 + 1j * u * b * tau - 0.5 * (sigma**2) * (u**2) * tau

        # TIME-INTEGRATED JUMP INTENSITY (see integrated_expected_intensity docstring):
        # lambda_plus/lambda_minus are the INSTANTANEOUS Hawkes intensities, which can
        # spike sharply on a single OFI tick and decay back to lambda_0 within ~1/beta
        # seconds. Multiplying that spike directly by the full remaining tau (as in the
        # naive Merton formula) would let a fleeting microstructure signal dominate the
        # probability of a settlement that may be minutes or hours away. We instead use
        # the expected integral of the mean-reverting intensity path over [0, tau].
        if beta > 0.0:
            jump_plus_tau = integrated_expected_intensity(lambda_plus, lambda_0, beta, tau_seconds)
            jump_minus_tau = integrated_expected_intensity(lambda_minus, lambda_0, beta, tau_seconds)
        else:
            jump_plus_tau = lambda_plus * tau
            jump_minus_tau = lambda_minus * tau

        # Positive jump process characteristic part
        jump_plus = jump_plus_tau * (np.exp(1j * u * mu_j_plus - 0.5 * (sigma_j_plus**2) * (u**2)) - 1.0)

        # Negative jump process characteristic part
        jump_minus = jump_minus_tau * (np.exp(1j * u * mu_j_minus - 0.5 * (sigma_j_minus**2) * (u**2)) - 1.0)

        return np.exp(diffusion + jump_plus + jump_minus)


class HawkesMertonPricer:
    """
    Solves the option exercise probability P(S_T > K) using Gil-Pelaez Fourier inversion
    under the Hawkes-Driven Merton model, using Black-Scholes as a control variate.
    Uses ultra-fast vectorized Gauss-Legendre quadrature.
    """
    
    def __init__(self, n_nodes: int = 64):
        self.cf = HawkesMertonCharacteristicFunction()
        # Pre-cache standard Gauss-Legendre nodes and weights on [-1, 1]
        self.gl_nodes_std, self.gl_weights_std = np.polynomial.legendre.leggauss(n_nodes)
        
    def calculate_probability(
        self, S_t: float, K: float, tau_seconds: float,
        mu: float, sigma: float,
        lambda_plus: float, lambda_minus: float,
        mu_j_plus: float, sigma_j_plus: float,
        mu_j_minus: float, sigma_j_minus: float,
        beta: float = 0.0, lambda_0: float = 0.0,
        limit: int = 150
    ) -> float:
        # Convert tau to annualized terms (seconds to years)
        tau = tau_seconds / (365.25 * 24 * 3600)
        
        # Microstructure boundary case: extremely close to expiration
        if tau_seconds <= 1.0:
            if S_t > K:
                return 1.0
            elif S_t < K:
                return 0.0
            else:
                return 0.5
                
        # Calculate standard analytical Black-Scholes/GBM exercise probability (N(d2))
        try:
            d2 = (np.log(S_t / K) + (mu - 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
            p_bs_analytical = float(stats.norm.cdf(d2))
        except Exception as e:
            logger.debug(f"Analytical BS calculation failed: {e}. Falling back to step function.")
            p_bs_analytical = 1.0 if S_t > K else (0.0 if S_t < K else 0.5)
 
        # Late-cycle bypass: within 15 seconds of expiry, jump probability is virtually 0.
        # Bypassing the double Fourier integration prevents high-frequency numerical oscillations
        # near the boundary from slowing down the simulation.
        if tau_seconds < 15.0:
            return p_bs_analytical
            
        x0 = np.log(S_t)
        eps = 1e-8
        upper_bound = 15000.0 / (sigma * np.sqrt(max(tau, 1e-9)))
        upper_bound = max(100.0, min(upper_bound, 50000.0))
        
        try:
            # Map Legendre nodes from [-1, 1] to [eps, upper_bound]
            nodes = 0.5 * (upper_bound - eps) * self.gl_nodes_std + 0.5 * (upper_bound + eps)
            weights = 0.5 * (upper_bound - eps) * self.gl_weights_std
            
            # Vectorized evaluation of the integrand
            phi_merton = self.cf.phi(
                u=nodes, S_t=S_t, K=K, tau=tau, mu=mu, sigma=sigma,
                lambda_plus=lambda_plus, lambda_minus=lambda_minus,
                mu_j_plus=mu_j_plus, sigma_j_plus=sigma_j_plus,
                mu_j_minus=mu_j_minus, sigma_j_minus=sigma_j_minus,
                tau_seconds=tau_seconds, beta=beta, lambda_0=lambda_0
            )
            
            # Black-Scholes CF (same continuous drift and volatility)
            b_bs = mu - 0.5 * sigma**2
            diffusion_bs = 1j * nodes * x0 + 1j * nodes * b_bs * tau - 0.5 * (sigma**2) * (nodes**2) * tau
            phi_bs = np.exp(diffusion_bs)
            
            # Difference term
            z = np.exp(-1j * nodes * np.log(K)) * (phi_merton - phi_bs)
            integrand_vals = np.imag(z) / nodes
            
            # Gauss-Legendre Quadrature sum (vectorized dot product)
            result = float(np.dot(weights, integrand_vals))
            
            prob = p_bs_analytical + (1.0 / np.pi) * result
            # Bound probability to strict [0, 1] range to avoid floating-point noise
            return max(0.0, min(1.0, prob))
            
        except Exception as e:
            logger.error(f"Vectorized Gauss-Legendre integration failed for Hawkes-Merton: {e}. Falling back to default GBM.")
            return p_bs_analytical


class MertonStrategy(BaseStrategy):
    """
    Hawkes-Driven Merton Jump-Diffusion Strategy.
    
    Integrates the Order Flow Imbalance (OFI) directly into the underlying asset's SDE.
    The jump intensities lambda+ and lambda- are stochastic and driven by positive/negative
    OFI pressure via a coupled bivariate Hawkes cross-excitation kernel.
    Optionally, the diffusive drift mu_t can also be modulated by the OFI.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.vol_calibrator = HighFrequencyVolatilityCalibrator(
            window_size=config.merton.VOL_ROLLING_WINDOW_SEC
        )
        self.pricer = HawkesMertonPricer()
        
        # Initialize Hawkes baseline intensity and state
        self.lambda_0 = getattr(config.merton, "HAWKES_LAMBDA_0", config.merton.DEFAULT_LAMBDA)
        self.state = MicrostructuralState(lambda_0=self.lambda_0)
        
    def reset(self) -> None:
        """Resets intensities and drift back to baseline values on cycle rollover."""
        self.state.reset()
        
    def get_probability(self, context: MarketContext) -> Optional[float]:
        if context.strike_price is None:
            return None
            
        # 1. Update Volatility
        self.vol_calibrator.add_tick(context.spot_price, context.timestamp)
        sigma = self.vol_calibrator.calculate_volatility(self.config.merton.DEFAULT_SIGMA)
        
        # 2. Update Hawkes/Microstructural state tick-by-tick using Bivariate parameters
        beta = self.config.merton.HAWKES_BETA
        kappa_self = self.config.merton.HAWKES_KAPPA_SELF
        kappa_cross = self.config.merton.HAWKES_KAPPA_CROSS
        ofi_multiplier = self.config.merton.OFI_DRIFT_MULTIPLIER
        use_local_drift = self.config.merton.USE_LOCAL_INFORMED_DRIFT
        
        # Risk-free rate (assumed 0 in short-term predictions, but can be customized)
        r = 0.0
        
        self.state.update_state(
            timestamp=context.timestamp,
            ofi=context.ofi,
            kappa_self=kappa_self,
            kappa_cross=kappa_cross,
            beta=beta,
            ofi_multiplier=ofi_multiplier,
            use_local_drift=use_local_drift,
            r=r
        )
        
        # 3. Compute probability using HawkesMertonPricer
        # Jump size distribution parameters
        mu_j = self.config.merton.DEFAULT_MU_J
        sigma_j = self.config.merton.DEFAULT_SIGMA_J
        
        # We split jump parameters: positive jumps have +mu_j, negative jumps have -mu_j
        mu_j_plus = abs(mu_j)
        sigma_j_plus = sigma_j
        mu_j_minus = -abs(mu_j)
        sigma_j_minus = sigma_j
        
        p_merton = self.pricer.calculate_probability(
            S_t=context.spot_price,
            K=context.strike_price,
            tau_seconds=context.tau_seconds,
            mu=self.state.mu,
            sigma=sigma,
            lambda_plus=self.state.lambda_plus,
            lambda_minus=self.state.lambda_minus,
            mu_j_plus=mu_j_plus,
            sigma_j_plus=sigma_j_plus,
            mu_j_minus=mu_j_minus,
            sigma_j_minus=sigma_j_minus,
            beta=beta,
            lambda_0=self.lambda_0
        )
        
        # Bound probability to avoid tail instabilities in Kelly sizing
        return float(np.clip(p_merton, 0.01, 0.99))
