# polymarketv2

**English** | [Italiano](README_IT.md)

Paper-trading bot and backtester for Polymarket crypto **Up/Down** binary markets (`btc-updown-5m-…`, `btc-updown-4h-…`).
It prices the YES contract with a Hawkes–Merton jump-diffusion driven by CLOB order-flow imbalance and quotes it with an
Avellaneda–Stoikov style maker/taker router. It connects to the real Chainlink and CLOB WebSocket feeds but **never
sends a real order**: execution is a simulated exchange, and every tick, signal and fill is logged for backtesting.

## Status (2026-09-17)

**The strategy has not made money.** What the recorded data says:

| Evidence | Result |
|---|---|
| 21 live paper sessions, 26 May – 2 Jun 2026 (code before PR #3) | 177 settlements: 86 won / 91 lost. Coin flip. |
| Settlement size vs. trade size in those sessions | Median settlement 800–27 000 contracts vs. 100–300 per trade: inventory was ridden into expiry every cycle. |
| Round-trip (non-settlement) P&L in the 12 sessions with sane capital | Negative in 8 of 12. |
| Taker fills, 388k-tick session | Mid moves **against** the fill by 0.07 at 5 s and 0.03 at 30 s (adverse selection). |
| 3 sessions ending at $125k–$2.9M | Old paper client had no cash cap; settlements of 1–7 M contracts on $10k. Not real. |
| Backtest of 2 Jun showing +222 % | Old runner bug (infinite liquidity, 972 winning exits). Ignore. |
| Current `main`, real 128k-tick file (6 cycles) | −3.6 %, max drawdown 9.7 %, inventory capped at 500, panic sweep flattens before expiry. |

PR #3 fixed the mechanical bugs (live start crash, broken backtester, Hawkes horizon amplification, panic sweep never
re-issued, no cash cap, hard-coded 5-minute cycle).

### Why the bot is maker-only (2026-09-19)

Replaying all 69 complete recorded cycles through the pricer, sampling model and book every 5 s:

| Test | Result |
|---|---|
| Brier score, model vs. book mid | 0.182 vs. 0.220 — the model is better calibrated than the market |
| Slope of (outcome − mid) on (model − mid) | 0.94, 95% CI [0.73, 1.10] — the divergence predicts the market's error |
| Buy-and-hold taker, 100 contracts, executed **on the signal tick** | +11.5 per entry, CI [+7.8, +15.8] |
| Same, executed **one tick later** | **−6.8 per entry**, CI [−12.0, −1.5] |
| Same, ignoring levels not refreshed in 10 s | unchanged (+11.5) — not stale liquidity |

The model reacts to the Chainlink tick a fraction of a second before the book does, and the book catches up within one
tick. The information is real; it is not executable at a 150–300 ms order round trip. That is the same adverse selection
the live taker fills showed (mid moving 0.07 against the fill within 5 s).

So directional crossing is off (`TAKER_ENABLED=False`) and the only taker orders left are the PANIC sweeps that flatten
inventory before settlement. A fast model that sees the move first is still worth having as a **defensive** signal —
pulling or repricing a quote before it gets picked off — which is what the remaining work is about.

The open question is now the **fill model**: every maker P&L in this repo comes from a simulator that assumes 40% of any
depth reduction at your level was a trade, plus a 2%/tick queue decay. Nothing has been calibrated against real
executions. `prints.parquet` (exchange trade prints, recorded since this branch) is what makes that calibration possible,
which is why collecting data matters more right now than tuning parameters.

## Quickstart

Python 3.13+ and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Wxlddd/polymarketv2 && cd polymarketv2
cp .env.example .env
uv sync
uv run python main.py --strategy merton        # web dashboard at http://localhost:8080
uv run python main.py --strategy merton --term # Rich terminal dashboard instead
```

Trading starts at the **next** cycle boundary, never mid-cycle. Every value in `.env` has a default in
`config/settings.py`; an old `.env` still works.

## Running overnight / collecting data

The process is a single asyncio loop with no interactive dependency once `--strategy` is given.

- **Windows desktop**: just run it. The orchestrator calls `SetThreadExecutionState` every 30 s so the machine does not sleep. Close the lid = no; screen off = fine.
- **Linux VPS** (1 vCPU / 1 GB is enough):

```ini
# /etc/systemd/system/polymarketv2.service
[Service]
WorkingDirectory=/opt/polymarketv2
ExecStart=/usr/local/bin/uv run python main.py --strategy merton
Restart=always
Environment=PYTHONUNBUFFERED=1
[Install]
WantedBy=multi-user.target
```

- **Docker**: `docker build -t polymarketv2 . && docker run -d --restart unless-stopped --env-file .env -v $(pwd)/logs:/app/logs polymarketv2`

Set `WEB_SERVER_HOST=0.0.0.0` to reach the dashboard remotely (no auth: keep it behind SSH).

Output per session:

```
logs/YYYY-MM-DD/merton/live_<unix_ts>/
├── ticks.parquet   # every CLOB update (snapshot/delta + is_snapshot flag) with spot, OFI, vol, reconciled top of book
├── prints.parquet  # every YES trade print (last_trade_price): price, size, aggressor side, exchange timestamp
├── signals.csv     # model vs market probability at each taker signal / maker fill
└── trades.csv      # every paper fill and settlement with P&L and capital
```

`logs/` is git-ignored. `git pull` never touches it.

## Backtesting

Replays `ticks.parquet` through the **same** router, paper client, shadow book and divergence filter used live.

```bash
uv run python run_backtest.py --file logs/2026-05-28/merton/live_1779986166/ticks.parquet
uv run python run_backtest.py --start "2026-05-27 10:00" --end "2026-05-27 12:00"   # scans logs/ and data/raw
BACKTEST_SEED=42 uv run python run_backtest.py --file ...                              # deterministic
```

Writes `signals.csv`, `trades.csv`, `summary.json` to `logs/<today>/merton/backtest_<stamp>/`.
Throughput is roughly 700 ticks/s (128k ticks ≈ 3 min).

**Parameter sweep** (parallel, one process per config × file, env-var overrides, results in `scratch/sweep_results.csv`):

```bash
uv run python scratch/sweep_backtest.py --set train --workers 6
uv run python scratch/sweep_backtest.py --set test --configs baseline,maker_only
```

Edit `CONFIGS` / `FILES` in the script. Keep a held-out `test` set: anything tuned on `train` is biased by construction.

**Log analysis**: `scratch/analyze_trades.py` breaks a `trades.csv` down by side, settlement win rate and unaccounted quantity.

## How it trades

```
Chainlink spot WS ─┐
CLOB L2 WS (YES) ──┴─► ShadowOrderBook (feed state + own paper consumption)
                          │
                          ├─► MertonStrategy.get_probability()  p_yes  (Hawkes–Merton, Gil-Pelaez GL64)
                          ├─► time EMA (EMA_HALFLIFE_SEC, shrinks to τ/10 near expiry)
                          ├─► DivergenceVelocityFilter → size scale 0..1 (0 = pull all quotes)
                          ├─► MockExecutionClient.process_market_data()  → resting maker fills
                          └─► ExecutionRouter.evaluate_regimes() → NEW / REPLACE / CANCEL
```

| Regime | When | Does |
|---|---|---|
| A maker | default | rest bid/ask at `P_res ± δ`, `P_res = p̂ − γ·q/Q_max·σ²`, `δ = fee + toxicity + ½γσ²τ` |
| B taker | target bid > best ask + ε (or mirror) | cancel quotes, IOC cross with fractional Kelly, floored at `MM_MAKER_SIZE`, capped by cash and `MM_MAX_INVENTORY` |
| C unwind | \|q\| > Q_max or \|p̂ − mid\| < `MM_UNWIND_THRESHOLD` | quote only the side that reduces \|q\| |
| REDUCE | τ ≤ 45 s | reduce-only |
| PANIC | τ ≤ 15 s, or τ ≤ 45 s with spread > 0.10 | IOC the whole position at `best bid − PANIC_CONCESSION` (or ask +), re-issued every tick until flat, lock until rollover |

Settlement pays 1.0 per winning contract at the last Chainlink tick before expiry vs. the strike locked at cycle start.

## Parameters that actually matter

All in `.env`, read by `config/settings.py`.

| Parameter | Default | Effect |
|---|---|---|
| `CYCLE_DURATION_SEC` / `MARKET_SLUG_TYPE` | `300` / `5m` | Series. `14400` / `4h` for the 4-hour market. Change both. |
| `INITIAL_CAPITAL` | `10000` | Paper cash. |
| `TAKER_ENABLED` | `False` | Directional crossing. Off by default (see Status); PANIC still crosses to flatten. |
| `MM_TAKER_EDGE_EPSILON` | `0.015` | Edge over the spread before crossing. |
| `KELLY_FRACTION` | `0.05` | Taker sizing. |
| `MM_MAX_INVENTORY` | `500` | Hard cap on \|YES − NO\|. |
| `MM_RISK_AVERSION` | `2.5` | γ: inventory skew and spread. |
| `MM_MIN_FEE_BUFFER`, `MM_TOXICITY_BUFFER` | `0.005`, `0.005` | Half-spread floor. |
| `MM_UNWIND_THRESHOLD` | `0.01` | Take profit when model and market re-align. |
| `PANIC_CONCESSION` | `0.15` | How far through the book the final sweep goes. |
| `EMA_HALFLIFE_SEC` | `3` | Probability smoothing. Tuned for 5m; too fast for 4h. |
| `VOL_ROLLING_WINDOW_SEC` | `300` | Realised vol window. Tuned for 5m. |
| `HAWKES_KAPPA_SELF`, `HAWKES_KAPPA_CROSS`, `HAWKES_BETA` | `3`, `1`, `5` | OFI → jump intensity. Set kappas to `0` for pure diffusion. |
| `STRATEGY_NAME` | `merton` | or `legacy_merton` (Poisson jumps + OFI logit shift). |
| `V_MAX`, `GAMMA` | `0.005`, `2` | Divergence filter. `scratch/calibrate_divergence.py` prints percentiles from a log. |

Every key above is read by the code. Settings that no longer did anything (`MIN_EXPECTED_VALUE`,
`PIN_RISK_SECONDS`, `DESYNC_Z_SCORE`, `POF_*`, `COOLDOWN_PERIOD_SEC`, `MAX_POSITION_SIZE_USD`,
`MIN_ACCEPTABLE_MARGIN_BPS`, `ABSOLUTE_MAX_SLIPPAGE_BPS`, `ORACLE_NOISE_BPS`, `MIN_KELLY_THRESHOLD`,
`MM_ENABLED`, `HAWKES_KAPPA`, `PRESUMED_STRIKE_PRICE`, `EXPIRATION_TIMESTAMP`) were removed — they
described the pre-refactor engine. Leaving them in a `.env` is harmless; they are simply ignored.

## Validation scripts

No test runner; each script is standalone.

```bash
PYTHONPATH=. uv run python tests/verify_phase4.py           # router → paper client → settlement
PYTHONPATH=. uv run python tests/verify_maker_execution.py  # 8 unit tests on the router
PYTHONPATH=. uv run python tests/verify_phase3.py           # vol calibration + pricer
```

Also `verify_phase1/2.py`, `verify_factory.py`, `verify_web_server.py`, `benchmark_pricing.py`.

## Layout

```
main.py                  LiveOrchestrator (feeds, pricing, routing, settlement, UI)
run_backtest.py          replay CLI
config/settings.py       frozen config dataclasses
src/strategies/          merton_strategy.py (Hawkes–Merton), legacy_merton_strategy.py, factory.py
src/execution/           engine.py (router), clients.py (paper exchange), shadow_book.py, divergence_filter.py
src/ingestion/           live_feeds.py (WS), market_manager.py (slug, rollover, Gamma API)
src/backtest/runner.py   BacktestRunner
src/logging/recorder.py  parquet + csv recorder
src/ui/                  web_server.py + dashboard.html, dashboard.py (Rich)
scratch/                 analyze_trades.py, sweep_backtest.py, calibrate_divergence.py
tests/                   verify_*.py
```

## Known gaps

- No evidence of edge. The directional signal is seconds-scale order flow; its value at 5 min is unproven, at 4 h doubtful.
- Fill model is assumed, not measured: taker fee `0.072·p(1−p)`, logistic rejection, maker queue at 40 % of depth reduction, 0.2 s consumption half-life.
- Taker IOC at `best_ask` after the simulated 150–300 ms latency often finds no liquidity (`IOC fill too small` warnings).
- Strike comes from the first Chainlink tick after cycle start; Polymarket's official "price to beat" is only logged.
- The divergence filter's scale is not logged, so its effect cannot be audited from `signals.csv`.
- One asset and one cycle length per process.
