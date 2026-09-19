import os
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

# Load .env file automatically
load_dotenv()

@dataclass(frozen=True)
class PolymarketConfig:
    """Polymarket CLOB API, Spot feed and Contract Configurations."""
    # WebSocket Spot Feed settings (relay of Chainlink Oracle CEX median)
    INTERNAL_WSS_URL: str = field(default_factory=lambda: os.getenv("POLYMARKET_WSS_URL", "wss://ws-live-data.polymarket.com/"))
    INTERNAL_SUBSCRIBE_PAYLOAD: str = field(default_factory=lambda: os.getenv("POLYMARKET_WS_PAYLOAD", '{"type": "subscribe", "topic": "crypto_prices_chainlink"}'))

    # Polymarket CLOB Order Book WS & REST URL
    WS_URL: str = field(default_factory=lambda: os.getenv("CLOB_WS_URL", "wss://ws-subscriptions-clob.polymarket.com/ws/market"))
    REST_URL: str = field(default_factory=lambda: os.getenv("CLOB_REST_URL", "https://clob.polymarket.com"))

    # Active token IDs
    YES_TOKEN_ID: str = field(default_factory=lambda: os.getenv("YES_TOKEN_ID", "0xYourYesTokenId"))
    NO_TOKEN_ID: str = field(default_factory=lambda: os.getenv("NO_TOKEN_ID", "0xYourNoTokenId"))

    # Market cycle settings — Polymarket runs several parallel Up/Down cycle
    # lengths per ticker (e.g. "5m" every 300s, "4h" every 14400s). Both the
    # rollover clock and the deterministic event slug depend on these two
    # values, so they must be changed together when switching timeframes.
    # CYCLE_DURATION_SEC must be a divisor of 86400 (aligned to Unix epoch).
    CYCLE_DURATION_SEC: int = field(default_factory=lambda: int(os.getenv("CYCLE_DURATION_SEC", "300")))
    MARKET_SLUG_TYPE: str = field(default_factory=lambda: os.getenv("MARKET_SLUG_TYPE", "5m"))

@dataclass(frozen=True)
class MertonJumpDiffusionConfig:
    """Merton Jump-Diffusion hyper-parameters for pricing calibration."""
    DEFAULT_LAMBDA: float = field(default_factory=lambda: float(os.getenv("DEFAULT_LAMBDA", "4000.0")))
    DEFAULT_MU_J: float = field(default_factory=lambda: float(os.getenv("DEFAULT_MU_J", "0.0001")))
    DEFAULT_SIGMA_J: float = field(default_factory=lambda: float(os.getenv("DEFAULT_SIGMA_J", "0.0015")))
    DEFAULT_SIGMA: float = field(default_factory=lambda: float(os.getenv("DEFAULT_SIGMA", "0.25")))
    VOL_ROLLING_WINDOW_SEC: int = field(default_factory=lambda: int(os.getenv("VOL_ROLLING_WINDOW_SEC", "300")))
    # OFI logit-shift parameters.
    # OFI enters the model as: p_final = sigmoid(logit(p_merton) + beta * ofi_z)
    # where ofi_z = smoothed_ofi / rolling_std(ofi) is the z-score of the smoothed OFI.
    # OFI_LOGIT_BETA: sensitivity — at p=0.5, a 1σ OFI event shifts p by ~beta/4 (small beta).
    #   beta=0.5 → 1σ OFI shifts ATM probability by ~12pp.
    #   beta=1.0 → 1σ OFI shifts ATM probability by ~23pp.
    # OFI_NORM_EMA_ALPHA: EMA decay for the rolling variance used to normalize OFI.
    OFI_LOGIT_BETA: float = field(default_factory=lambda: float(os.getenv("OFI_LOGIT_BETA", "0.5")))
    OFI_NORM_EMA_ALPHA: float = field(default_factory=lambda: float(os.getenv("OFI_NORM_EMA_ALPHA", "0.1")))
    # Half-life (seconds) for the time-based EMA applied to raw p_yes before the engine.
    # Time-based EMA ensures consistent smoothing regardless of CLOB tick rate.
    EMA_HALFLIFE_SEC: float = field(default_factory=lambda: float(os.getenv("EMA_HALFLIFE_SEC", "3.0")))
    
    # Hawkes stochastic intensity parameters
    HAWKES_LAMBDA_0: float = field(default_factory=lambda: float(os.getenv("HAWKES_LAMBDA_0", "4000.0")))
    HAWKES_KAPPA_SELF: float = field(default_factory=lambda: float(os.getenv("HAWKES_KAPPA_SELF", "3.0")))
    HAWKES_KAPPA_CROSS: float = field(default_factory=lambda: float(os.getenv("HAWKES_KAPPA_CROSS", "1.0")))
    HAWKES_BETA: float = field(default_factory=lambda: float(os.getenv("HAWKES_BETA", "5.0")))
    OFI_DRIFT_MULTIPLIER: float = field(default_factory=lambda: float(os.getenv("OFI_DRIFT_MULTIPLIER", "-1e-6")))
    USE_LOCAL_INFORMED_DRIFT: bool = field(default_factory=lambda: os.getenv("USE_LOCAL_INFORMED_DRIFT", "False").lower() == "true")

@dataclass(frozen=True)
class ArbitrageConfig:
    """Trading and capital sizing configurations."""
    INITIAL_CAPITAL: float = field(default_factory=lambda: float(os.getenv("INITIAL_CAPITAL", "10000.0")))
    KELLY_FRACTION: float = field(default_factory=lambda: float(os.getenv("KELLY_FRACTION", "0.05")))
    GAS_FEE_USD: float = field(default_factory=lambda: float(os.getenv("GAS_FEE_USD", "0.03")))
    TAKER_FEE_MULTIPLIER: float = field(default_factory=lambda: float(os.getenv("TAKER_FEE_MULTIPLIER", "0.072")))
    MIN_ORDER_USD: float = field(default_factory=lambda: float(os.getenv("MIN_ORDER_USD", "1.0")))
    # Maker-only by default: on recorded data the model's information is real but not
    # executable — a signal traded one tick after it appears loses money (see README).
    TAKER_ENABLED: bool = field(default_factory=lambda: os.getenv("TAKER_ENABLED", "False").lower() == "true")
    PANIC_CONCESSION: float = field(default_factory=lambda: float(os.getenv("PANIC_CONCESSION", "0.15")))

@dataclass(frozen=True)
class RiskConfig:
    """Parameters of the divergence velocity filter (the only risk filter in the engine)."""
    DIVERGENCE_WINDOW_SECONDS: float = field(default_factory=lambda: float(os.getenv("DIVERGENCE_WINDOW_SECONDS", "60.0")))
    VELOCITY_LOOKBACK_SECONDS: float = field(default_factory=lambda: float(os.getenv("VELOCITY_LOOKBACK_SECONDS", "10.0")))
    V_MAX: float = field(default_factory=lambda: float(os.getenv("V_MAX", "0.005")))
    GAMMA: float = field(default_factory=lambda: float(os.getenv("GAMMA", "2.0")))

@dataclass(frozen=True)
class MarketMakerConfig:
    """Statistical market making and quoting configurations."""
    RISK_AVERSION: float = field(default_factory=lambda: float(os.getenv("MM_RISK_AVERSION", "2.5")))
    MIN_FEE_BUFFER: float = field(default_factory=lambda: float(os.getenv("MM_MIN_FEE_BUFFER", "0.005")))
    TOXICITY_BUFFER: float = field(default_factory=lambda: float(os.getenv("MM_TOXICITY_BUFFER", "0.005")))
    MAKER_SIZE: float = field(default_factory=lambda: float(os.getenv("MM_MAKER_SIZE", "100.0")))
    MAX_INVENTORY: float = field(default_factory=lambda: float(os.getenv("MM_MAX_INVENTORY", "500.0")))
    UNWIND_THRESHOLD: float = field(default_factory=lambda: float(os.getenv("MM_UNWIND_THRESHOLD", "0.01")))
    TAKER_EDGE_EPSILON: float = field(default_factory=lambda: float(os.getenv("MM_TAKER_EDGE_EPSILON", "0.015")))
    FIXED_HORIZON_SEC: float = field(default_factory=lambda: float(os.getenv("MM_FIXED_HORIZON_SEC", "300.0")))
    TICK_SIZE: float = field(default_factory=lambda: float(os.getenv("MM_TICK_SIZE", "0.01")))
    REQUOTE_THRESHOLD: float = field(default_factory=lambda: float(os.getenv("MM_REQUOTE_THRESHOLD", "0.01")))

@dataclass(frozen=True)
class WebServerConfig:
    """Integrated HTTP and WebSocket server settings for Bloomberg Terminal UI."""
    ENABLED: bool = field(default_factory=lambda: os.getenv("WEB_SERVER_ENABLED", "True").lower() == "true")
    HOST: str = field(default_factory=lambda: os.getenv("WEB_SERVER_HOST", "localhost"))
    PORT: int = field(default_factory=lambda: int(os.getenv("WEB_SERVER_PORT", "8080")))
    UI_BROADCAST_THROTTLE_HZ: float = field(default_factory=lambda: float(os.getenv("UI_BROADCAST_THROTTLE_HZ", "4.0")))

@dataclass(frozen=True)
class SystemConfig:
    """Root configuration holding all sub-modules."""
    TICKER: str = field(default_factory=lambda: os.getenv("TICKER", "BTC").upper())
    LOG_DIR: str = field(default_factory=lambda: os.getenv("LOG_DIR", "logs"))
    STRATEGY_NAME: str = field(default_factory=lambda: os.getenv("STRATEGY_NAME", "merton"))
    
    polymarket: PolymarketConfig = field(default_factory=PolymarketConfig)
    merton: MertonJumpDiffusionConfig = field(default_factory=MertonJumpDiffusionConfig)
    arbitrage: ArbitrageConfig = field(default_factory=ArbitrageConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    maker: MarketMakerConfig = field(default_factory=MarketMakerConfig)
    web_server: WebServerConfig = field(default_factory=WebServerConfig)

