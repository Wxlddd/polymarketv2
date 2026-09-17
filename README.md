# Polymarket V2 — Hawkes–Merton Pricing & Market-Making Engine for Crypto Up/Down Markets

**English** | [Versione Italiana](README_IT.md)

Polymarket V2 is an event-driven research and paper-trading system for Polymarket's fixed-length crypto **Up/Down** binary markets (`btc-updown-5m-…`, `btc-updown-4h-…`, …). It prices the YES contract with a Merton jump-diffusion model whose jump intensities are driven by a bivariate Hawkes process fed by L2 order-flow imbalance, and trades against the CLOB with an Avellaneda–Stoikov-style market-making router that can also cross the spread when the model edge is large and flattens inventory before settlement.

Everything runs against the real Polymarket feeds (Chainlink spot relay and CLOB L2 WebSocket) but executes on a simulated exchange, so it can be left running unattended to collect tick data and paper P&L. The same execution pipeline is used by the historical backtester.

> **Status.** Paper trading only — there is no live order submission. The strategy has not been shown to be profitable; see [Known limitations](#known-limitations-and-open-questions). Treat this repository as research infrastructure, not as a signal.

---

## Contents

1. [Architecture](#architecture)
2. [Pricing model](#pricing-model)
3. [Execution engine](#execution-engine)
4. [Shadow order book](#shadow-order-book)
5. [Market cycles: 5-minute, 4-hour and beyond](#market-cycles-5-minute-4-hour-and-beyond)
6. [Backtester](#backtester)
7. [Data logging](#data-logging)
8. [Installation and usage](#installation-and-usage)
9. [Running unattended and collecting data](#running-unattended-and-collecting-data)
10. [Configuration reference](#configuration-reference)
11. [Validation scripts](#validation-scripts)
12. [Repository layout](#repository-layout)
13. [Known limitations and open questions](#known-limitations-and-open-questions)

---

## Architecture

```
ChainlinkSpotFeed (WS)          ClobOrderBookFeed (WS, YES token only)
        │                                   │
        └──────────────┬────────────────────┘
                       ▼
        LiveOrchestrator._clob_callback()            main.py
                       │
        ShadowOrderBook.update_book()  → OFI, V_hist / V_cons reconciliation
                       │
        MertonStrategy.get_probability() → raw p_yes (Hawkes–Merton, Gil-Pelaez GL64)
                       │
        time-based EMA (half-life compressed near expiry)
                       │
        DivergenceVelocityFilter.get_scale() → toxic-flow sizing scale
                       │
        MockExecutionClient.process_market_data() → resting maker fills
                       │
        ExecutionEngine.evaluate_and_route()  → OrderInstruction[] (A / B / C / REDUCE / PANIC)
                       │
        MockExecutionClient.process_instruction()  (awaited: fills land before the next tick)
                       │
        DataRecorder  (ticks.parquet, signals.csv, trades.csv)
```

Market discovery and rollover run in a separate loop (`MarketManager`): the next cycle's expiry is derived from the wall clock, the event slug is generated deterministically, the YES/NO token IDs are fetched from the Gamma API, the CLOB feed is re-subscribed, and the previous cycle is settled at its true expiry using the last Chainlink tick before expiry.

| Path | Responsibility |
|---|---|
| `main.py` | `LiveOrchestrator` — async event loop coordinating feeds, pricing, routing, settlement, logging and UI |
| `config/settings.py` | Frozen config dataclasses loaded from `.env` |
| `src/ingestion/live_feeds.py` | `ChainlinkSpotFeed` (spot relay WS), `ClobOrderBookFeed` (CLOB L2 WS) |
| `src/ingestion/market_manager.py` | Deterministic slug generation, Gamma API discovery, rollover clock |
| `src/core/strike_manager.py` | Strike `K` resolution and locking per cycle |
| `src/core/market_context.py` | `MarketContext` — the immutable per-tick data contract |
| `src/strategies/merton_strategy.py` | Hawkes–Merton pricer (vectorised Gauss–Legendre), realised-vol calibrator |
| `src/strategies/legacy_merton_strategy.py` | Homogeneous-Poisson Merton with posterior OFI logit shift (`--strategy legacy_merton`) |
| `src/execution/engine.py` | `ExecutionRouter` (regimes) + `ExecutionEngine` wrapper |
| `src/execution/divergence_filter.py` | Divergence velocity / acceleration guard |
| `src/execution/shadow_book.py` | `ShadowOrderBook` with `ConsumptionTracker` |
| `src/execution/clients.py` | `MockExecutionClient` — paper exchange: IOC book-walk, resting maker queue, settlement |
| `src/backtest/runner.py` | Event-driven replay through the identical pipeline |
| `src/logging/recorder.py` | Parquet / CSV recorder with daily directory rollover |
| `src/ui/web_server.py`, `src/ui/dashboard.html` | aiohttp HTTP + WebSocket dashboard (Live Monitor + Backtester tabs) |
| `src/ui/dashboard.py` | Rich terminal dashboard (`--term`) |

---

## Pricing model

### Underlying dynamics

The log-price follows a jump-diffusion with separate positive and negative jump streams:

$$dS_t = \mu_t S_t\,dt + \sigma S_t\,dW_t + S_t\,d\Big(\sum_{i=1}^{N_t^+}(V_i^+-1)\Big) + S_t\,d\Big(\sum_{j=1}^{N_t^-}(V_j^--1)\Big)$$

with $\ln V^{\pm} \sim \mathcal N(\pm|\mu_j|, \sigma_j^2)$ and counting processes $N^{\pm}$ whose intensities $\lambda^{\pm}(t)$ are stochastic.

### Bivariate Hawkes intensities

Intensities are updated tick-by-tick from the directional order-flow imbalance $\text{OFI}^{+}=\max(\text{OFI},0)$, $\text{OFI}^{-}=\max(-\text{OFI},0)$ with self- and cross-excitation and exponential decay:

$$\lambda^{\pm}(t_k) = \lambda_0 + \big(\lambda^{\pm}(t_{k-1})-\lambda_0\big)e^{-\beta\Delta t} + \kappa_{\text{self}}\,\text{OFI}^{\pm} + \kappa_{\text{cross}}\,\text{OFI}^{\mp}$$

A stationarity guard rescales $(\kappa_{\text{self}},\kappa_{\text{cross}})$ on the first tick if $\kappa_{\text{self}}+\kappa_{\text{cross}} \ge \beta$ so that the branching ratio is $0.8$.

### Time-integrated jump intensity

$\lambda^{\pm}(t)$ is an *instantaneous* rate that can spike on a single tick and relax back to $\lambda_0$ within $\sim 1/\beta$ seconds. The characteristic function therefore does **not** use $\lambda^{\pm}(t)\,\tau$; it uses the expected number of jumps over the remaining horizon under the mean-reverting dynamics:

$$\Lambda^{\pm}(\tau) = \int_0^{\tau}\mathbb E[\lambda^{\pm}(t+s)]\,ds = \lambda_0\,\tau + \big(\lambda^{\pm}(t)-\lambda_0\big)\frac{1-e^{-\beta\tau}}{\beta}$$

This bounds the contribution of any transient excitation to at most $(\lambda-\lambda_0)/\beta$ extra expected jumps regardless of how far away expiry is. Without it, a one-tick OFI spike would be multiplied by the full time-to-expiry — a distortion that grows linearly with cycle length and made the model unusable on anything longer than a few minutes.

### Characteristic function and inversion

$$\phi(u) = \exp\!\Big( iu\,x_t + iu\,b\,\tau - \tfrac12\sigma^2u^2\tau + \Lambda^{+}(\tau)\big(e^{iu\mu_j^+-\frac12\sigma_j^2u^2}-1\big) + \Lambda^{-}(\tau)\big(e^{iu\mu_j^--\frac12\sigma_j^2u^2}-1\big)\Big), \qquad b = \mu_t - \tfrac12\sigma^2$$

The martingale jump compensator is deliberately omitted: the target is the real-world (P-measure) probability, and a Q-measure compensator would push the continuous drift *down* when bullish flow raises $\lambda^+$.

$P(S_T > K)$ is obtained by Gil-Pelaez inversion with Black–Scholes as control variate,

$$P(S_T>K) = N(d_2) + \frac1\pi\int_0^\infty \operatorname{Im}\!\left[\frac{e^{-iu\ln K}\big(\phi(u)-\phi_{\text{BS}}(u)\big)}{u}\right]du,$$

evaluated with a vectorised 64-node Gauss–Legendre quadrature. Within 15 s of expiry the integral is skipped and $N(d_2)$ is used; within 1 s the payoff step function is returned.

### Inputs

* **Volatility** — `HighFrequencyVolatilityCalibrator`: samples spot at ≥5 s intervals, drops the buffer on a >1.5 % jump, caps single returns at ±0.5 %, annualises the realised standard deviation over a rolling window (`VOL_ROLLING_WINDOW_SEC`), clipped to [15 %, 300 %].
* **Drift** — zero by default; `USE_LOCAL_INFORMED_DRIFT=True` sets $\mu_t = \text{OFI}\cdot\text{OFI\_DRIFT\_MULTIPLIER}$.
* **Smoothing** — the raw probability is passed through a time-based EMA with half-life `EMA_HALFLIFE_SEC`, compressed to $\tau/10$ near expiry so the estimate can collapse to 0/1 as the market does.

---

## Execution engine

`ExecutionRouter.evaluate_regimes()` is a tick-level state machine over the model price $\hat p$, the top of book, net inventory $q = \text{YES} - \text{NO}$ and time-to-expiry $\tau$. It emits `OrderInstruction`s (`NEW` / `REPLACE` / `CANCEL`) that the client executes.

**Reservation price and quotes.** With $q_{\text{norm}} = \operatorname{clamp}(q/Q_{\max},-1,1)$ and the EWMA variance $\sigma^2$ of the *contract* mid-price (10 s sampling):

$$P_{\text{res}} = \hat p - \gamma\,q_{\text{norm}}\,\sigma^2, \qquad \delta = \text{fee buffer} + \tfrac12\gamma\sigma^2\tau + \text{toxicity buffer}$$

Target quotes are $P_{\text{res}}\pm\delta$ rounded to the tick and kept post-only (at least one tick inside the spread).

| Regime | Trigger | Behaviour |
|---|---|---|
| **A — Maker** | default | Rest a bid and an ask at the target quotes. Bid size is the remaining inventory capacity, ask size is the YES held; both scaled by the divergence filter. Re-quote only when price moves ≥ `MM_REQUOTE_THRESHOLD`. |
| **B — Taker** | target bid $>$ best ask $+\epsilon$ (or target ask $<$ best bid $-\epsilon$) | Cancel quotes and cross the spread with fractional-Kelly size $f^* = \text{KELLY\_FRACTION}\cdot\frac{\hat p - p_{\text{ask}}}{1-p_{\text{ask}}}$ (capped at 50 % of wealth, floored at `MM_MAKER_SIZE`, capped by cash and inventory). Buys NO when short edge and no YES is held. |
| **C — Unwind** | $\lvert q\rvert > Q_{\max}$, or $\lvert\hat p - p_{\text{mid}}\rvert <$ `MM_UNWIND_THRESHOLD` with $q\ne0$ | Quote only on the side that reduces $\lvert q\rvert$. |
| **REDUCE** | $\tau \le 45$ s | Reduce-only: cancel the side that would add inventory; if flat, cancel everything. |
| **PANIC** | $\tau \le 15$ s, or $\tau \le 45$ s with spread $> 0.10$ | Cancel all quotes and send an IOC sweep for the entire net position, priced through the book (`best bid − PANIC_CONCESSION` when selling YES, `best ask + PANIC_CONCESSION` when covering NO). Re-issued on every tick until flat; sets a lock that blocks new positions until rollover. |

The **divergence velocity filter** tracks $d_t = \hat p_{\text{EMA}} - p_{\text{mkt}}$, its velocity over `VELOCITY_LOOKBACK_SECONDS` and its acceleration, and returns a sizing scale $\max\big(0, 1-(|v|/V_{\max})^{\gamma}\big)$ that goes to zero when divergence is accelerating away from the market — all quotes are pulled in that state.

**Paper exchange (`MockExecutionClient`).** Taker orders walk the effective L2 book as IOC up to the limit price, pay `TAKER_FEE_MULTIPLIER · p(1−p)` per contract plus `GAS_FEE_USD`, and are subject to a logistic stochastic rejection in size and volatility. Maker orders rest with an estimated queue position that is advanced from observed depth changes and fill at the limit price with no taker fee. YES/NO pairs are merged to cash automatically. Settlement pays 1.0 per winning contract using the last Chainlink tick before expiry against the locked strike.

---

## Shadow order book

`ShadowOrderBook` keeps two ledgers so paper fills do not corrupt the view of the real market:

| Ledger | Content |
|---|---|
| **V_hist** | The feed's L2 state (top 20 levels per side). Never touched by the bot. |
| **V_cons** (`ConsumptionTracker`) | Liquidity consumed by the bot's own paper fills, decaying exponentially (half-life 0.2 s) to model re-quoting. Reconciled on every feed delta: replenishment reduces it, removals cap it. |
| **V_eff** | $\max(0, V_{\text{hist}} - V_{\text{cons}})$ — what the router and the IOC walker see. |

`get_market_top_of_book()` reads V_hist and is used for the market-implied probability, mark-to-market and the divergence filter; `get_sorted_bids()/asks()` return V_eff. Both are cached per tick.

Only the YES token is subscribed; NO prices are $1-p_{\text{YES}}$. Subscribing to both would mix NO bids into the top of book and pin the implied probability near 0.5.

The CLOB callback `await`s execution before returning, so a fill is registered in V_cons before the next tick can trigger the same opportunity again. Snapshot bursts within 1 s (reconnect storms) are downgraded to deltas.

---

## Market cycles: 5-minute, 4-hour and beyond

Polymarket runs several Up/Down series per asset with different cycle lengths. All cycle boundaries are multiples of the cycle length on the Unix epoch and the event slug encodes the cycle **start** time:

```
btc-updown-5m-1789675200     (300 s cycles)
btc-updown-4h-1789675200     (14 400 s cycles)
```

Two settings select the series, and must be changed together:

```ini
CYCLE_DURATION_SEC=14400
MARKET_SLUG_TYPE=4h
```

`MarketManager` derives the next expiry as the next multiple of `CYCLE_DURATION_SEC` (pre-empted by 15 s so the new token IDs are subscribed before the boundary), the strike is locked from the first Chainlink tick after `expiry − CYCLE_DURATION_SEC`, and the backtester uses the same alignment when replaying a file.

**What does and does not scale with cycle length.** The pricing model is horizon-aware: $\tau$ enters the diffusion term and the time-integrated jump intensity above, so no re-parameterisation is needed for the option maths itself. The remaining parameters are absolute-time quantities that were tuned for 5-minute cycles and should be revisited for longer ones:

* `EMA_HALFLIFE_SEC` (3 s) and `VOL_ROLLING_WINDOW_SEC` (300 s) — a 4-hour contract does not need sub-second responsiveness; longer smoothing and a longer vol window reduce noise trading.
* The Hawkes/OFI channel (`HAWKES_KAPPA_*`, `HAWKES_BETA`) encodes seconds-scale microstructure. Its information content about a settlement hours away is small; consider lowering the kappas (or the OFI weight) for long cycles.
* The REDUCE / PANIC windows (45 s / 15 s) are about settlement-tick noise, not cycle length, and are intentionally left absolute.
* `MM_MAX_INVENTORY`, `KELLY_FRACTION` and the fee buffers bound exposure per cycle; with fewer cycles per day the per-cycle exposure is a larger share of daily risk.

---

## Backtester

`run_backtest.py` replays recorded `ticks.parquet` files through the *same* `ExecutionEngine`, `MockExecutionClient`, `ShadowOrderBook` and `DivergenceVelocityFilter` used live — there is no separate simulation model. Differences from live operation:

* Taker instructions are delayed by a uniform 150–300 ms in **event time** (the replay itself is instantaneous).
* The first tick of each cycle is applied as a full book snapshot; later ticks are deltas, so paper consumption persists.
* Pricing is re-evaluated only when spot, the top of book, the model probability or inventory changed (or 5 s elapsed), which removes most redundant Fourier inversions.
* Each cycle's strike is the first spot tick of the cycle and positions are settled at each rollover and at the end of the file.

```bash
uv run python run_backtest.py --file logs/2026-09-17/merton/live_1789675200/ticks.parquet
uv run python run_backtest.py --start "2026-09-17 10:00:00" --end "2026-09-17 12:00:00"   # scans LOG_DIR and data/raw
```

Output goes to `logs/<date>/merton/backtest_<stamp>/` as `signals.csv`, `trades.csv` and `summary.json` (net P&L, return, max drawdown, win rate, profit factor, average win/loss, capital curve and every realised trade). The web dashboard's Backtester tab runs the same code.

---

## Data logging

```
logs/
└── YYYY-MM-DD/                      # rolls over at midnight
    └── merton/
        ├── live_<unix_ts>/
        │   ├── ticks.parquet        # every CLOB tick: spot, OFI, vol, top of book, full L2 (zstd)
        │   ├── signals.csv          # model vs market probability at each fill / taker signal
        │   └── trades.csv           # every paper execution and settlement with P&L and capital
        └── backtest_YYYYMMDD_HHMMSS/
            ├── signals.csv
            ├── trades.csv
            └── summary.json
```

Ticks are buffered (1 000 rows) before being appended to Parquet; signals and trades are written immediately. `ticks.parquet` is the input format the backtester expects.

---

## Installation and usage

Requirements: Python 3.13+ and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Wxlddd/polymarketv2 && cd polymarketv2
cp .env.example .env            # edit TICKER, CYCLE_DURATION_SEC / MARKET_SLUG_TYPE, capital, …
uv sync                         # creates .venv from uv.lock
```

Run the live paper-trading orchestrator:

```bash
uv run python main.py                       # web dashboard at http://localhost:8080
uv run python main.py --term                # Rich terminal dashboard instead
uv run python main.py --strategy merton     # skip the interactive strategy prompt (merton | legacy_merton)
```

On start the orchestrator waits for the **next** cycle boundary before trading, so the first cycle is never entered mid-way with an unknown strike. Logs go to `system_run.log` and to `LOG_DIR`.

---

## Running unattended and collecting data

The process is a single long-running `asyncio` loop with no interactive dependencies (the strategy prompt is skipped when stdin is not a TTY), so it can be run as a service on any always-on Linux host — a small VPS is enough. Everything it observes is written to `LOG_DIR`; leaving it running for days is the intended way to build a backtest dataset. It never submits real orders.

**systemd**

```ini
# /etc/systemd/system/polymarketv2.service
[Unit]
Description=Polymarket V2 paper trader / data collector
After=network-online.target
Wants=network-online.target

[Service]
User=polymarket
WorkingDirectory=/opt/polymarketv2
ExecStart=/usr/local/bin/uv run python main.py --strategy merton
Restart=always
RestartSec=10
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now polymarketv2
journalctl -u polymarketv2 -f
```

**Docker**

```bash
docker build -t polymarketv2 .
docker run -d --name polymarketv2 --restart unless-stopped \
  --env-file .env -p 8080:8080 -v $(pwd)/logs:/app/logs polymarketv2
```

Set `WEB_SERVER_HOST=0.0.0.0` in `.env` to reach the dashboard from outside the container or host; the dashboard has no authentication, so keep it behind a firewall or SSH tunnel.

**Windows** — the orchestrator refreshes `SetThreadExecutionState` every 30 s to keep the machine awake while running; `runbot.ps1` is a convenience launcher.

Collected data lives under `logs/<date>/merton/live_<ts>/`; point `run_backtest.py --file` at a `ticks.parquet` (or the whole run directory) to replay it.

---

## Configuration reference

All values are read from `.env` by `config/settings.py`. Only parameters that are actually consumed by the code are listed.

| Parameter | Default | Used by |
|---|---|---|
| `TICKER` | `BTC` | Slug generation |
| `CYCLE_DURATION_SEC` / `MARKET_SLUG_TYPE` | `300` / `5m` | Rollover clock, slug, strike timing, backtester |
| `STRATEGY_NAME` | `merton` | `StrategyFactory` (`merton`, `legacy_merton`) |
| `INITIAL_CAPITAL` | `10000` | Paper cash |
| `KELLY_FRACTION` | `0.05` | Regime B sizing |
| `GAS_FEE_USD`, `TAKER_FEE_MULTIPLIER` | `0.03`, `0.072` | Paper fees |
| `MIN_ORDER_USD` | `1.0` | Minimum maker quote / IOC fill |
| `PANIC_CONCESSION` | `0.15` | Panic sweep limit offset from the book |
| `DEFAULT_SIGMA` | `0.25` | Vol fallback before calibration |
| `VOL_ROLLING_WINDOW_SEC` | `300` | Realised-vol window |
| `EMA_HALFLIFE_SEC` | `3.0` | Probability smoothing |
| `DEFAULT_MU_J`, `DEFAULT_SIGMA_J` | `1e-4`, `1.5e-3` | Jump size distribution |
| `HAWKES_LAMBDA_0` | `4000` | Baseline jump intensity (per year) |
| `HAWKES_KAPPA_SELF`, `HAWKES_KAPPA_CROSS`, `HAWKES_BETA` | `3.0`, `1.0`, `5.0` | Hawkes excitation / decay (per second) |
| `OFI_DRIFT_MULTIPLIER`, `USE_LOCAL_INFORMED_DRIFT` | `-1e-6`, `False` | Optional OFI drift |
| `OFI_LOGIT_BETA`, `OFI_NORM_EMA_ALPHA` | `0.5`, `0.1` | `legacy_merton` only |
| `MM_RISK_AVERSION` (γ) | `2.5` | Reservation price skew and spread |
| `MM_MIN_FEE_BUFFER`, `MM_TOXICITY_BUFFER` | `0.005`, `0.005` | Half-spread floor |
| `MM_MAKER_SIZE` | `100` | Minimum taker clip |
| `MM_MAX_INVENTORY` | `500` | $Q_{\max}$ (contracts) |
| `MM_UNWIND_THRESHOLD`, `MM_TAKER_EDGE_EPSILON` | `0.01`, `0.015` | Regime C / B triggers |
| `MM_TICK_SIZE`, `MM_REQUOTE_THRESHOLD` | `0.01`, `0.01` | Quote rounding / re-quote hysteresis |
| `DIVERGENCE_WINDOW_SECONDS`, `VELOCITY_LOOKBACK_SECONDS`, `V_MAX`, `GAMMA` | `60`, `10`, `0.005`, `2.0` | Divergence velocity filter |
| `WEB_SERVER_ENABLED`, `WEB_SERVER_HOST`, `WEB_SERVER_PORT`, `UI_BROADCAST_THROTTLE_HZ` | `True`, `localhost`, `8080`, `4` | Dashboard |
| `LOG_DIR` | `logs` | Recorder root |

`scratch/calibrate_divergence.py --file <ticks.parquet | signals.csv>` prints velocity percentiles from recorded data to help choose `V_MAX`.

---

## Validation scripts

There is no test runner; each script is standalone and needs the repository root on `PYTHONPATH`:

```bash
PYTHONPATH=. uv run python tests/verify_phase1.py            # feeds / market manager plumbing
PYTHONPATH=. uv run python tests/verify_phase2.py            # shadow book reconciliation
PYTHONPATH=. uv run python tests/verify_phase3.py            # vol calibration + Gil-Pelaez pricer
PYTHONPATH=. uv run python tests/verify_phase4.py            # router → paper client → settlement, end to end
PYTHONPATH=. uv run python tests/verify_maker_execution.py   # unit tests for the regime router
PYTHONPATH=. uv run python tests/verify_factory.py
PYTHONPATH=. uv run python tests/verify_web_server.py
PYTHONPATH=. uv run python tests/benchmark_pricing.py
PYTHONPATH=. uv run python tests/verify_live_data_and_pricing.py   # requires the live WebSocket feeds
```

---

## Repository layout

```
polymarketv2/
├── config/settings.py
├── main.py                      # live / paper orchestrator
├── run_backtest.py              # historical replay CLI
├── src/
│   ├── core/                    # interfaces, MarketContext, StrikeManager, base strategy
│   ├── ingestion/               # WebSocket feeds, MarketManager
│   ├── strategies/              # Hawkes–Merton, legacy Merton, factory
│   ├── execution/               # ExecutionRouter/Engine, divergence filter, shadow book, paper client
│   ├── backtest/                # BacktestRunner
│   ├── logging/                 # DataRecorder
│   └── ui/                      # web server, dashboard.html, Rich terminal UI
├── tests/                       # standalone validation scripts
├── scratch/                     # analysis helpers (trade breakdown, V_MAX calibration)
├── Dockerfile
├── pyproject.toml / uv.lock
└── .env.example
```

---

## Known limitations and open questions

* **No evidence of edge.** Paper results to date have been negative. The model's directional information comes almost entirely from seconds-scale order flow; whether that has any predictive value at the 5-minute horizon — let alone 4 hours — has not been established. The backtester now exercises the real pipeline, so this can be tested on recorded data rather than guessed.
* **Fee and fill realism.** The taker fee `p(1−p)·0.072`, the stochastic rejection model, the maker queue estimate (40 % of depth reduction assumed traded) and the 0.2 s consumption half-life are assumptions, not measurements against Polymarket's actual matching.
* **Settlement gas** is charged in the reported settlement P&L but not deducted from paper cash.
* **Strike source.** The strike is taken from the first Chainlink tick after cycle start; Polymarket's official "price to beat" from the Gamma API is fetched when available but only used for logging.
* **Single asset, single series.** One ticker and one cycle length per process; run several processes for several series.
* **No live execution client.** `IExecutionClient` is the seam for one; nothing signs or submits orders today.
