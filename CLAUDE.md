# Agent guidance for polymarketv2

This file is read by coding agents (Claude Code, Gemini CLI — `GEMINI.md` is an identical copy). It describes how the repository actually works; the user-facing documentation is `README.md`.

## Commands

Package manager is `uv`. Python 3.13 (`.python-version`).

```bash
uv sync                                          # create .venv from uv.lock
uv run python main.py                            # live paper trading, web dashboard on :8080
uv run python main.py --term                     # Rich terminal dashboard
uv run python main.py --strategy merton          # skip the interactive strategy prompt

uv run python run_backtest.py --file path/to/ticks.parquet
uv run python run_backtest.py --start "2026-09-17 10:00:00" --end "2026-09-17 12:00:00"

# Validation scripts are standalone (no test runner) and need the repo root on the path:
PYTHONPATH=. uv run python tests/verify_phase3.py           # pricer + vol calibrator
PYTHONPATH=. uv run python tests/verify_phase4.py           # router -> paper client -> settlement
PYTHONPATH=. uv run python tests/verify_maker_execution.py  # unittest for ExecutionRouter regimes
```

`tests/verify_live_data_and_pricing.py` needs the live WebSocket feeds. `tests/verify_dashboard.py` drives the terminal UI and is not meaningful headless.

Configuration is `.env` (template `.env.example`), loaded once into frozen dataclasses by `config/settings.py`. `main.py` mutates a few fields through `__dict__` at runtime (token IDs, strategy name) — that is the only sanctioned way to change a frozen config.

## Per-tick data flow

```
ChainlinkSpotFeed (WS)      ClobOrderBookFeed (WS, YES token only)
        └──────────┬──────────────┘
     LiveOrchestrator._clob_callback()                         main.py
     ShadowOrderBook.update_book()          -> OFI, V_hist/V_cons reconcile
     MertonStrategy.get_probability()       -> raw p_yes
     time-based EMA (half-life min(EMA_HALFLIFE_SEC, tau/10))
     DivergenceVelocityFilter.get_scale()   -> sizing scale in [0,1]
     MockExecutionClient.process_market_data()  -> resting maker fills
     ExecutionEngine.evaluate_and_route()   -> List[OrderInstruction]
     MockExecutionClient.process_instruction()  (awaited per instruction)
     DataRecorder.record_tick / record_signal / record_trade
```

`src/backtest/runner.py` runs exactly this pipeline over a recorded `ticks.parquet` (taker instructions delayed 150–300 ms in event time via a pending queue). If you change the orchestration in `main.py`, mirror it in the runner.

## Module map

| Path | Purpose |
|---|---|
| `main.py` | `LiveOrchestrator`: feeds, rollover loop, strike resolution loop, settlement task, UI state |
| `config/settings.py` | `SystemConfig` → `PolymarketConfig`, `MertonJumpDiffusionConfig`, `ArbitrageConfig`, `RiskConfig`, `MarketMakerConfig`, `WebServerConfig` |
| `src/core/interfaces.py` | `ISpotFeed`, `IOrderBook`, `IExecutionClient`, `IDataRecorder`, `OrderInstruction` |
| `src/core/market_context.py` | Frozen `MarketContext` (timestamp, spot, strike, tau, vol, ofi, L2 books) |
| `src/core/strike_manager.py` | Holds the presumed strike and locks it at expiry |
| `src/ingestion/market_manager.py` | Next-expiry clock, slug `<ticker>-updown-<MARKET_SLUG_TYPE>-<expiry - CYCLE_DURATION_SEC>`, Gamma API lookup |
| `src/ingestion/live_feeds.py` | `ChainlinkSpotFeed`, `ClobOrderBookFeed` |
| `src/strategies/merton_strategy.py` | `HighFrequencyVolatilityCalibrator`, `MicrostructuralState` (bivariate Hawkes), `HawkesMertonPricer` (GL64 Gil-Pelaez), `MertonStrategy` |
| `src/strategies/legacy_merton_strategy.py` | Homogeneous Poisson Merton + posterior OFI logit shift |
| `src/execution/engine.py` | `ExecutionRouter.evaluate_regimes()` state machine; `ExecutionEngine` wraps strategy + client + router |
| `src/execution/divergence_filter.py` | `DivergenceVelocityFilter` |
| `src/execution/shadow_book.py` | `ShadowOrderBook` + `ConsumptionTracker` |
| `src/execution/clients.py` | `MockExecutionClient`: IOC book walk with fees and stochastic rejection, resting maker queue model, position merge, settlement |
| `src/backtest/runner.py` | `BacktestRunner`, writes `summary.json` |
| `src/logging/recorder.py` | Parquet ticks (buffered), CSV signals/trades (immediate), daily directory rollover |
| `src/ui/web_server.py`, `dashboard.html`, `dashboard.py` | Web dashboard (HTTP + WS) and terminal dashboard |

## Router regimes (`ExecutionRouter.evaluate_regimes`)

Evaluated in this order every tick: **PANIC** (tau ≤ 15 s, or tau ≤ 45 s with spread > 0.10: cancel all, IOC-sweep the whole net position at `best_bid − PANIC_CONCESSION` / `best_ask + PANIC_CONCESSION`, re-issued every tick until flat, sets `locked`) → **locked** (cancel all, no new positions until rollover) → **REDUCE** (tau ≤ 45 s, reduce-only quoting) → **C** unwind (|q| > MAX_INVENTORY or model realigned with mid) → **B** taker (target quote crosses the book by more than `MM_TAKER_EDGE_EPSILON`; fractional Kelly size) → **A** maker (post-only quotes at `P_res ± delta`).

`p_hat` is the EMA-smoothed probability computed once per tick in `main.py` (and the runner) and passed to `ExecutionEngine.evaluate_and_route(context, p_hat=...)`. The engine only calls `strategy.get_probability` itself when no `p_hat` is supplied (unit tests).

## Invariants — do not break

- **YES-only CLOB subscription.** NO prices are `1 − p_YES`. Subscribing to both tokens mixes NO bids into the top of book.
- **V_hist vs V_cons.** `get_market_top_of_book()` reads the untouched feed state; `get_sorted_bids/asks()` return `V_hist − V_cons`. Paper fills go through `paper_execute()` only.
- **Awaited execution.** `_clob_callback` awaits every `process_instruction` so fills are in V_cons before the next tick.
- **Strike guard.** `MarketContext.strike_price is None` ⇒ strategy returns `None` and the router cancels everything (`SAFE`).
- **`strategy.get_probability` runs exactly once per tick.** `MicrostructuralState.update_state` adds the tick's OFI shock on every call; a second call at the same timestamp (dt = 0) double-counts the Hawkes excitation. Pass the smoothed `p_hat` into the engine instead of letting it re-price. Recorded sessions before this rule had 1,020 taker round trips closed in a median 0.34 s on model swings of 0.55 with spot unchanged.
- **Cycle timing is config-driven.** Every boundary computation uses `CYCLE_DURATION_SEC` (`MarketManager.cycle_duration_sec` in `main.py`/`dashboard.py`, `config.polymarket.CYCLE_DURATION_SEC` in the runner). Never hard-code 300.
- **Hawkes intensities are time-integrated in the CF.** `integrated_expected_intensity()` replaces `lambda * tau`; passing raw `lambda * tau` re-introduces the horizon-amplification bug.
- **`ExecutionEngine` uses `__slots__`.** Add a slot before assigning a new attribute (this is how `divergence_filter` is attached).
- **Panic sweeps are priced off the book, not off `p_hat`**, and are re-issued until `q == 0`.

## History worth knowing

Commit `2324e84` consolidated the maker/taker engines into `ExecutionRouter`. That refactor left `main.py`, `run_backtest.py` and several `tests/verify_*.py` pointing at removed APIs (`ExecutionEngine(config)`, `evaluate_and_trade`, `client.execute_trade`, `src/execution/maker_execution`), so neither the live orchestrator nor the backtester could start. Those were repaired in the same change that introduced configurable cycle length, the time-integrated Hawkes intensity and the book-anchored, retried panic sweep. Old `README` text describing a "Kelly + Pin/Desync/PoF" engine referred to code that no longer exists.
