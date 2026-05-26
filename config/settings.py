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

    # Strike details
    PRESUMED_STRIKE_PRICE: float = field(default_factory=lambda: float(os.getenv("PRESUMED_STRIKE_PRICE", "67500.0")))
    EXPIRATION_TIMESTAMP: float = field(default_factory=lambda: float(os.getenv("EXPIRATION_TIMESTAMP", "0.0")))

@dataclass(frozen=True)
class MertonJumpDiffusionConfig:
    """Merton Jump-Diffusion hyper-parameters for pricing calibration."""
    DEFAULT_LAMBDA: float = field(default_factory=lambda: float(os.getenv("DEFAULT_LAMBDA", "4000.0")))
    DEFAULT_MU_J: float = field(default_factory=lambda: float(os.getenv("DEFAULT_MU_J", "0.0001")))
    DEFAULT_SIGMA_J: float = field(default_factory=lambda: float(os.getenv("DEFAULT_SIGMA_J", "0.0015")))
    DEFAULT_SIGMA: float = field(default_factory=lambda: float(os.getenv("DEFAULT_SIGMA", "0.25")))
    VOL_ROLLING_WINDOW_SEC: int = field(default_factory=lambda: int(os.getenv("VOL_ROLLING_WINDOW_SEC", "300")))
    OFI_DRIFT_MULTIPLIER: float = field(default_factory=lambda: float(os.getenv("OFI_DRIFT_MULTIPLIER", "-1e-6")))
    OFI_HORIZON_SEC: float = field(default_factory=lambda: float(os.getenv("OFI_HORIZON_SEC", "5.0")))

@dataclass(frozen=True)
class ArbitrageConfig:
    """Trading and capital sizing configurations."""
    INITIAL_CAPITAL: float = field(default_factory=lambda: float(os.getenv("INITIAL_CAPITAL", "10000.0")))
    KELLY_FRACTION: float = field(default_factory=lambda: float(os.getenv("KELLY_FRACTION", "0.15")))
    GAS_FEE_USD: float = field(default_factory=lambda: float(os.getenv("GAS_FEE_USD", "0.03")))
    TAKER_FEE_MULTIPLIER: float = field(default_factory=lambda: float(os.getenv("TAKER_FEE_MULTIPLIER", "0.072")))
    MIN_EXPECTED_VALUE: float = field(default_factory=lambda: float(os.getenv("MIN_EXPECTED_VALUE", "0.005")))
    MIN_ACCEPTABLE_MARGIN_BPS: float = field(default_factory=lambda: float(os.getenv("MIN_ACCEPTABLE_MARGIN_BPS", "5.0")))
    ABSOLUTE_MAX_SLIPPAGE_BPS: float = field(default_factory=lambda: float(os.getenv("ABSOLUTE_MAX_SLIPPAGE_BPS", "150.0")))
    MAX_POSITION_SIZE_USD: float = field(default_factory=lambda: float(os.getenv("MAX_POSITION_SIZE_USD", "5000.0")))

@dataclass(frozen=True)
class RiskConfig:
    """Risk management parameters."""
    DESYNC_Z_SCORE: float = field(default_factory=lambda: float(os.getenv("DESYNC_Z_SCORE", "2.0")))
    POF_LATENCY_TAU: float = field(default_factory=lambda: float(os.getenv("POF_LATENCY_TAU", "15.0")))
    POF_DECAY_K: float = field(default_factory=lambda: float(os.getenv("POF_DECAY_K", "0.25")))
    ORACLE_NOISE_BPS: float = field(default_factory=lambda: float(os.getenv("ORACLE_NOISE_BPS", "1.5")))
    PIN_RISK_SECONDS: float = field(default_factory=lambda: float(os.getenv("PIN_RISK_SECONDS", "5.0")))
    COOLDOWN_PERIOD_SEC: int = field(default_factory=lambda: int(os.getenv("COOLDOWN_PERIOD_SEC", "10")))

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
    
    polymarket: PolymarketConfig = field(default_factory=PolymarketConfig)
    merton: MertonJumpDiffusionConfig = field(default_factory=MertonJumpDiffusionConfig)
    arbitrage: ArbitrageConfig = field(default_factory=ArbitrageConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    web_server: WebServerConfig = field(default_factory=WebServerConfig)
