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
uv run python run_backtest.py --start "2026-05-28 19:15:00" --end "2026-05-28 19:20:00"

# Run backtest with single file
uv run python run_backtest.py --file "path/to/ticks.parquet"

# Run individual validation scripts (no test runner — each is standalone)
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
         ShadowOrderBook.update_book()  → OFI + ConsumptionTracker reconcile
                   │
         MertonStrategy.get_probability()  → p_yes (Bivariate Hawkes + Gil-Pelaez GL64)
                   │
         EMA smoothing (time-based halflife, adaptive near expiry)
                   │
         ExecutionEngine.evaluate_and_trade()  → decision
                   │
         MockExecutionClient.execute_trade()  ← await (prevents double signals)
                   │
         ShadowOrderBook.paper_execute()  ← registers fills in ConsumptionTracker
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
| `src/strategies/factory.py` | `StrategyFactory` — dynamic resolution of active pricing strategy |
| `src/strategies/merton_strategy.py` | `MertonStrategy`: Bivariate Hawkes + `HawkesMertonPricer` (vectorized Gauss-Legendre GL64) |
| `src/strategies/legacy_merton_strategy.py` | Legacy Merton strategy (homogeneous Poisson, scalar `scipy.integrate.quad`) |
| `src/execution/shadow_book.py` | `ShadowOrderBook`: V_hist (q_real) + `ConsumptionTracker` (V_cons) with exponential decay |
| `src/execution/engine.py` | `ExecutionEngine`: fractional Kelly sizing, L2 book walk, Pin/Desync/PoF risk filters |
| `src/execution/clients.py` | `MockExecutionClient`: paper-trading, position tracking, settlement, realized_trades |
| `src/backtest/runner.py` | `BacktestRunner`: event-driven historical replay with summary.json output |
| `src/logging/recorder.py` | `DataRecorder`: Parquet (ticks, buffered) + CSV (signals/trades, immediate) |
| `src/ui/dashboard.py` | Rich CLI terminal dashboard |
| `src/ui/web_server.py` | Integrated aiohttp HTTP + WebSocket server |
| `src/ui/dashboard.html` | Bloomberg-style web terminal UI with Live Monitor + Backtester tabs |

### Critical invariants

**YES-only CLOB subscription**: `ClobOrderBookFeed` subscribes only to the YES token. NO prices are derived as `1.0 - YES_price`. Subscribing to both tokens would mix NO bids (~0.84) into the shadow book top-of-book, making `p_mkt` read ~0.5 regardless of the real market.

**ConsumptionTracker separation**: `ShadowOrderBook` maintains V_hist (feed state, never mutated by bot) and a parallel `ConsumptionTracker` V_cons (bot fills only, with exponential decay). `get_market_top_of_book()` reads V_hist; `get_sorted_bids/asks()` returns V_eff = V_hist - V_cons. Both are cached per tick and invalidated on update_book() or paper_execute().

**Awaited execution**: `_clob_callback` is `async` and `await`s `client.execute_trade()` before returning. This guarantees fills are registered in the consumption tracker before the next CLOB tick arrives, preventing duplicate trade signals on the same opportunity.

**Strike resolution guard**: `strike_price` in `MarketContext` is `None` until resolved. Both `MertonStrategy.get_probability()` and `ExecutionEngine.evaluate_and_trade()` return `None`/`HOLD` immediately if strike is `None`. Strike is set either from the Gamma API at rollover or from the first Chainlink tick after `cycle_start = expiry - 300s`.

**Hawkes stationarity guard**: On first tick, `MicrostructuralState` validates the branching ratio (kappa_self + kappa_cross) / beta < 1.0. If explosive, parameters are automatically scaled to force spectral radius = 0.8.

**Cycle timing**: `MarketManager.get_next_expiry()` preempts the rollover by 15 seconds so the bot subscribes to the new cycle early. Expirations are always multiples of 300 seconds (Unix epoch).

### Pricing pipeline detail

1. **Volatility**: `HighFrequencyVolatilityCalibrator` samples spot every ≥5s, filters >1.5% jumps (resets buffer), caps returns at ±0.5%, computes annualized realized vol over a 300s rolling window.
2. **Microstructural State**: `MicrostructuralState.update_state()` updates bivariate Hawkes intensities λ+ and λ- tick-by-tick from directional OFI shocks, with optional OFI-modulated drift.
3. **Probability**: `HawkesMertonPricer.calculate_probability()` computes P(S_T > K) via Gil-Pelaez Fourier inversion of the Hawkes-Merton characteristic function, using Black-Scholes as a control variate, with vectorized Gauss-Legendre quadrature (64 nodes).
4. **EMA smoothing**: Raw `p_yes` from the pricer is smoothed with a time-based EMA (halflife adaptive, compressed near expiry) in the orchestrator before passing to the engine.

### Risk filters in ExecutionEngine

- **Pin Risk**: blocks trades when `|spot - strike| < oracle_noise` in the final `PIN_RISK_SECONDS` before expiry. Uses bisect for O(log N) spot history lookups.
- **Desync/Staleness**: blocks if top-of-book quotes are >100ms stale and spot has moved more than the diffusion-implied threshold (z-score × σ × √dt). Uses bisect binary search for closest-spot lookup.
- **PoF (Fill Probability)**: scales EV by a logistic decay `1 / (1 + exp(-k*(tau - tau_lim)))` to penalize late-cycle trades that risk not being filled before expiry.

### Logging output structure

```
logs/
└── YYYY-MM-DD/
    └── merton/
        ├── live_<unix_ts>/
        │   ├── ticks.parquet    # buffered (every 100 ticks), zstd compressed
        │   ├── signals.csv      # immediate append on every signal
        │   └── trades.csv       # immediate append on every execution
        └── backtest_YYYYMMDD_HHMMSS/
            ├── signals.csv      # backtest signals
            ├── trades.csv       # backtest executions
            └── summary.json     # full performance metrics report
```

### Windows-specific

The orchestrator runs `_keep_awake_loop` (every 30s) to call `SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED)` via `ctypes`, preventing Windows sleep during live sessions. This is OS-gated (`os.name == 'nt'`).
