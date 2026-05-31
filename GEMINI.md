# GEMINI.md

This file provides guidance to Gemini CLI when working with code in this repository.

## Commands

**Package manager**: `uv` (not pip). All commands use `uv run python`.

```bash
# Run live orchestrator — Rich terminal dashboard
uv run python main.py

# Run live orchestrator — Bloomberg web dashboard at http://localhost:8080
uv run python main.py --no-term

# Run backtest with time window
uv run python run_backtest.py --start "2026-05-23 20:51:00" --end "2026-05-23 21:51:00"

# Run backtest with single file
uv run python run_backtest.py --file "path/to/ticks.parquet"

# Run individual validation scripts (no test runner — each is standalone)
uv run python tests/verify_live_data_and_pricing.py
uv run python tests/verify_phase3.py   # Merton pricing + vol calibration
uv run python tests/verify_phase4.py   # Shadow book + order execution
uv run python tests/verify_web_server.py
```

Configuration lives in `.env` (copy from `.env.example`). All env vars are loaded by `config/settings.py` into frozen dataclasses at startup.

## Architecture

### Data flow (per CLOB tick)

```
ChainlinkSpotFeed (WS)     ClobOrderBookFeed (WS, YES-only)
        │                            │
        └──────────┬─────────────────┘
                   ▼
         LiveOrchestrator._clob_callback()
                   │
         ShadowOrderBook.update_book()  → OFI
                   │
         MertonStrategy.get_probability()  → p_yes (Gil-Pelaez)
                   │
         EMA smoothing (α=0.05)
                   │
         ExecutionEngine.evaluate_and_trade()  → decision
                   │
         MockExecutionClient.execute_trade()  ← await (prevents double signals)
                   │
         ShadowOrderBook.paper_execute()  ← depletes shadow book only
                   │
         DataRecorder.record_tick/signal/trade()
```

### Module map

| Path | Purpose |
|---|---|
| `main.py` | `LiveOrchestrator` — async event loop, coordinates all subsystems |
| `config/settings.py` | Frozen dataclasses: `SystemConfig` → `PolymarketConfig`, `MertonJumpDiffusionConfig`, `ArbitrageConfig`, `RiskConfig`, `WebServerConfig` |
| `src/core/interfaces.py` | Abstract interfaces: `ISpotFeed`, `IOrderBook`, `IExecutionClient`, `IDataRecorder` |
| `src/core/market_context.py` | `MarketContext` frozen dataclass — the data contract between all layers |
| `src/core/strike_manager.py` | Resolves and locks strike price K |
| `src/ingestion/live_feeds.py` | `ChainlinkSpotFeed` (Chainlink oracle WS), `ClobOrderBookFeed` (CLOB WS) |
| `src/ingestion/market_manager.py` | Gamma API discovery, deterministic slug generation, 5-min rollover handling |
| `src/strategies/merton_strategy.py` | `MertonStrategy`: `HighFrequencyVolatilityCalibrator` + `GilPelaezIntegrator` |
| `src/execution/shadow_book.py` | `ShadowOrderBook`: dual `q_real`/`q_shadow` books |
| `src/execution/engine.py` | `ExecutionEngine`: fractional Kelly sizing, L2 book walk, Pin/Desync/PoF risk filters |
| `src/execution/clients.py` | `MockExecutionClient`: paper-trading, position tracking, settlement |
| `src/logging/recorder.py` | `DataRecorder`: Parquet (ticks, buffered) + CSV (signals/trades, immediate) |
| `src/ui/dashboard.py` | Rich CLI terminal dashboard |
| `src/ui/web_server.py` | Integrated aiohttp HTTP + WebSocket server |
| `src/ui/dashboard.html` | Bloomberg-style web terminal UI |
| `src/backtest/` | `BacktestRunner` for historical replay |

### Critical invariants

**YES-only CLOB subscription**: `ClobOrderBookFeed` subscribes only to the YES token. NO prices are derived as `1.0 - YES_price`. Subscribing to both tokens would mix NO bids (~0.84) into the shadow book top-of-book, making `p_mkt` read ~0.5 regardless of the real market.

**Dual-book separation**: `ShadowOrderBook` maintains two independent books. `q_real` is only ever updated by live WebSocket ticks and is never depleted. `q_shadow` is updated by deltas from live ticks AND depleted by `paper_execute`. `get_market_top_of_book()` reads `q_real`; `get_sorted_bids/asks()` reads `q_shadow`. This ensures `p_mkt` in the UI always reflects true market liquidity.

**Awaited execution**: `_clob_callback` is `async` and `await`s `client.execute_trade()` before returning. This guarantees the shadow book is depleted before the next CLOB tick arrives, preventing duplicate trade signals on the same opportunity.

**Strike resolution guard**: `strike_price` in `MarketContext` is `None` until resolved. Both `MertonStrategy.get_probability()` and `ExecutionEngine.evaluate_and_trade()` return `None`/`HOLD` immediately if strike is `None`. Strike is set either from the Gamma API at rollover or from the first Chainlink tick after `cycle_start = expiry - 300s`.

**Cycle timing**: `MarketManager.get_next_expiry()` preempts the rollover by 15 seconds so the bot subscribes to the new cycle early. Expirations are always multiples of 300 seconds (Unix epoch).

### Pricing pipeline detail

1. **Volatility**: `HighFrequencyVolatilityCalibrator` samples spot every ≥5s, filters >1.5% jumps (resets buffer), caps returns at ±0.5%, computes annualized realized vol over a 300s rolling window.
2. **Drift**: `MertonStrategy.estimate_drift()` scales smoothed OFI (EMA α=0.1) via `OFI_DRIFT_MULTIPLIER` (default `-1e-7`, sign validated empirically) into an annualized drift, then dilutes it over `tau` using `OFI_HORIZON_SEC`.
3. **Probability**: `GilPelaezIntegrator` computes P(S_T > K) via Gil-Pelaez Fourier inversion of the Merton characteristic function, using Black-Scholes as a control variate for numerical stability near expiry.
4. **EMA smoothing**: Raw `p_yes` from the integrator is smoothed with α=0.05 (~30s memory at 1 tick/s) in the orchestrator before passing to the engine.

### Risk filters in ExecutionEngine

- **Pin Risk**: blocks trades when `|spot - strike| < oracle_noise` in the final `PIN_RISK_SECONDS` before expiry.
- **Desync/Staleness**: blocks if top-of-book quotes are >100ms stale and spot has moved more than the diffusion-implied threshold (z-score × σ × √dt).
- **PoF (Fill Probability)**: scales EV by a logistic decay `1 / (1 + exp(-k*(tau - tau_lim)))` to penalize late-cycle trades that risk not being filled before expiry.

### Logging output structure

```
logs/
└── YYYY-MM-DD/
    └── merton/
        └── live_<unix_ts>/
            ├── ticks.parquet    # buffered (every 100 ticks), zstd compressed
            ├── signals.csv      # immediate append on every signal
            └── trades.csv       # immediate append on every execution
```

### Windows-specific

The orchestrator runs `_keep_awake_loop` (every 30s) to call `SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED)` via `ctypes`, preventing Windows sleep during live sessions. This is OS-gated (`os.name == 'nt'`).
