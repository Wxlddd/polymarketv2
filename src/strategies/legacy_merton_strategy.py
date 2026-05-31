import numpy as np
import scipy.integrate as integrate
import scipy.stats as stats
import logging
from typing import Dict, Any, Tuple, Optional
from src.core.base_strategy import BaseStrategy
from src.core.market_context import MarketContext

logger = logging.getLogger("LegacyMertonStrategy")

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


class MertonCharacteristicFunction:
    """
    Computes the characteristic function for the Merton Jump-Diffusion (MJD) model.
    """
    
    def phi(
        self, u: float, S_t: float, K: float, tau: float, mu: float, sigma: float,
        lambda_: float, mu_j: float, sigma_j: float
    ) -> complex:
        x0 = np.log(S_t)
        
        # Jump drift corrector: kappa = E[e^Y] - 1
        kappa = np.exp(mu_j + 0.5 * sigma_j**2) - 1.0
        
        # Drift component in log-price: b = mu - lambda*kappa - 0.5*sigma^2
        b = mu - lambda_ * kappa - 0.5 * sigma**2
        
        # Continuous diffusion part
        diffusion = 1j * u * x0 + 1j * u * b * tau - 0.5 * (sigma**2) * (u**2) * tau
        
        # Jump process characteristic part
        jump = lambda_ * tau * (np.exp(1j * u * mu_j - 0.5 * (sigma_j**2) * (u**2)) - 1.0)
        
        return np.exp(diffusion + jump)


class GilPelaezIntegrator:
    """
    Solves the option exercise probability P(S_T > K) using Gil-Pelaez Fourier inversion
    with Black-Scholes as a control variate to ensure high-frequency stability.
    """
    
    def __init__(self):
        self.cf = MertonCharacteristicFunction()
        
    def calculate_probability(
        self, S_t: float, K: float, tau_seconds: float, 
        mu: float, sigma: float, lambda_: float, 
        mu_j: float, sigma_j: float, limit: int = 150
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
            
        x0 = np.log(S_t)
        
        def integrand(u: float) -> float:
            if u == 0.0:
                return 0.0
            
            # Merton CF
            kappa = np.exp(mu_j + 0.5 * sigma_j**2) - 1.0
            b_merton = mu - lambda_ * kappa - 0.5 * sigma**2
            diffusion_merton = 1j * u * x0 + 1j * u * b_merton * tau - 0.5 * (sigma**2) * (u**2) * tau
            jump_merton = lambda_ * tau * (np.exp(1j * u * mu_j - 0.5 * (sigma_j**2) * (u**2)) - 1.0)
            phi_merton = np.exp(diffusion_merton + jump_merton)
            
            # Black-Scholes CF (same continuous drift and volatility)
            b_bs = mu - 0.5 * sigma**2
            diffusion_bs = 1j * u * x0 + 1j * u * b_bs * tau - 0.5 * (sigma**2) * (u**2) * tau
            phi_bs = np.exp(diffusion_bs)
            
            # Difference term
            z = np.exp(-1j * u * np.log(K)) * (phi_merton - phi_bs)
            return np.imag(z) / u
            
        try:
            # Integrate the difference term. The integrand decays to zero extremely rapidly
            eps = 1e-8
            upper_bound = 15000.0 / (sigma * np.sqrt(max(tau, 1e-9)))
            upper_bound = max(100.0, min(upper_bound, 50000.0))
            
            result, _ = integrate.quad(integrand, eps, upper_bound, limit=limit, epsabs=1e-8, epsrel=1e-8)
            
            prob = p_bs_analytical + (1.0 / np.pi) * result
            # Bound probability to strict [0, 1] range to avoid floating-point noise
            return max(0.0, min(1.0, prob))
            
        except Exception as e:
            logger.error(f"Control Variate Gil-Pelaez integration failed: {e}. Falling back to default GBM.")
            return p_bs_analytical


class OFILogitShifter:
    """
    Applies smoothed OFI as a bounded logit-space adjustment on top of the Merton
    base probability, keeping order-flow signal strictly decoupled from the
    diffusion/jump model.

    Formula:
        p_final = sigmoid( logit(p_merton) + β × ofi_z )

    where ofi_z = smoothed_ofi / sqrt(EMA(ofi²)) is the dimensionless z-score.

    Properties:
    - p_final is always in (0, 1) by construction — OFI can never collapse it to 0/1.
    - β calibrates sensitivity independently of market depth or contract size.
      At p=0.5: β=0.5 and ofi_z=+1σ → p_final ≈ 0.62 (+12pp).
      beta=1.0 → 1σ OFI shifts ATM probability by ~23pp.
    - z-score is clipped to ±3σ so max logit shift is ±3β, preventing tail spikes.
    - Running variance is warm-started at the first observed ofi² to avoid a cold
      zero-std phase that would divide by near-zero.
    """

    def __init__(self, beta: float, ema_alpha: float = 0.1):
        self.beta = beta
        self.ema_alpha = ema_alpha
        self._ema_sq: float = 0.0
        self._initialized: bool = False

    def shift(self, p_merton: float, smoothed_ofi: float) -> float:
        alpha = self.ema_alpha
        ofi_sq = smoothed_ofi ** 2

        if not self._initialized:
            self._ema_sq = ofi_sq if ofi_sq > 1e-12 else 1.0
            self._initialized = True
        else:
            self._ema_sq = alpha * ofi_sq + (1.0 - alpha) * self._ema_sq

        ofi_std = np.sqrt(max(self._ema_sq, 1e-12))
        ofi_z = float(np.clip(smoothed_ofi / ofi_std, -3.0, 3.0))

        if p_merton <= 0.0 or p_merton >= 1.0:
            return p_merton

        logit_p = np.log(p_merton / (1.0 - p_merton))
        return float(1.0 / (1.0 + np.exp(-(logit_p + self.beta * ofi_z))))


class MertonStrategy(BaseStrategy):
    """
    Fourier Merton Jump-Diffusion Strategy.

    Merton + Gil-Pelaez prices the pure diffusion/jump component (μ=0).
    OFI enters separately as a logit-space shift via OFILogitShifter, keeping
    the two signals orthogonal and preventing annualization artifacts.
    """

    def __init__(self, config):
        super().__init__(config)
        self.integrator = GilPelaezIntegrator()
        self.vol_calibrator = HighFrequencyVolatilityCalibrator(
            window_size=config.merton.VOL_ROLLING_WINDOW_SEC
        )
        self.ofi_shifter = OFILogitShifter(
            beta=config.merton.OFI_LOGIT_BETA,
            ema_alpha=config.merton.OFI_NORM_EMA_ALPHA
        )

    def reset(self) -> None:
        """Resets OFI variance normalization state on cycle rollover."""
        self.ofi_shifter._ema_sq = 0.0
        self.ofi_shifter._initialized = False

    def get_probability(self, context: MarketContext) -> Optional[float]:
        if context.strike_price is None:
            return None

        self.vol_calibrator.add_tick(context.spot_price, context.timestamp)
        sigma = self.vol_calibrator.calculate_volatility(self.config.merton.DEFAULT_SIGMA)

        # Merton runs with μ=0: drift contribution is purely from the logit shift below.
        p_merton = self.integrator.calculate_probability(
            S_t=context.spot_price,
            K=context.strike_price,
            tau_seconds=context.tau_seconds,
            mu=0.0,
            sigma=sigma,
            lambda_=self.config.merton.DEFAULT_LAMBDA,
            mu_j=self.config.merton.DEFAULT_MU_J,
            sigma_j=self.config.merton.DEFAULT_SIGMA_J
        )

        p_final = self.ofi_shifter.shift(p_merton, context.ofi)
        return float(np.clip(p_final, 0.01, 0.99))
