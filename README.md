# Polymarket V2: High-Frequency Options Pricing & Execution Engine

**English** | [Versione Italiana](README_IT.md)

Polymarket V2 is a high-performance, ultra-modular quantitative architecture designed for real-time options pricing and automated execution on Polymarket's 5-minute (300-second) binary options markets (Up/Down). The system implements the Merton Jump-Diffusion (MJD) model with stochastic jump intensities driven by a coupled bivariate Hawkes process, solved in the frequency domain via Fourier inversion (Gil-Pelaez) with vectorized Gauss-Legendre quadrature (64 nodes) and a Black-Scholes control variate. It integrates a fractional Kelly sizing engine and strict risk controls tailored for High-Frequency Trading (HFT) environments.

---

## Mathematical Model and Pricing

### 1. Underlying Dynamics: Hawkes-Driven Merton Jump-Diffusion
The underlying asset price $S_t$ follows a diffusion process with asymmetric jumps governed by the following Stochastic Differential Equation (SDE):

$$dS_t = \mu_t S_t dt + \sigma S_t dW_t + S_t d\left( \sum_{i=1}^{N_t^+} (V_i^+ - 1) \right) + S_t d\left( \sum_{j=1}^{N_t^-} (V_j^- - 1) \right)$$

Where:
- $\mu_t$ represents the continuous instantaneous drift rate, optionally modulated by the L2 Order Flow Imbalance (OFI).
- $\sigma$ represents the continuous diffusion coefficient (high-frequency realized volatility).
- $W_t$ represents a standard Brownian motion on a filtered probability space.
- $N_t^+, N_t^-$ represent counting processes with stochastic intensities $\lambda^+(t), \lambda^-(t)$.
- $V_i^+$ and $V_j^-$ represent positive and negative jump amplitudes respectively, with $Y^+ = \ln(V^+) \sim \mathcal{N}(\mu_j^+, (\sigma_j^+)^2)$ and $Y^- = \ln(V^-) \sim \mathcal{N}(\mu_j^-, (\sigma_j^-)^2)$.

### 2. Coupled Bivariate Hawkes Process
The jump intensities $\lambda^+(t)$ and $\lambda^-(t)$ evolve according to a symmetric bivariate Hawkes process with cross-excitation:

$$\lambda^+(t_k) = \lambda_0 + (\lambda^+(t_{k-1}) - \lambda_0) e^{-\beta \Delta t} + \kappa_{\text{self}} \cdot \text{OFI}^+ + \kappa_{\text{cross}} \cdot \text{OFI}^-$$

$$\lambda^-(t_k) = \lambda_0 + (\lambda^-(t_{k-1}) - \lambda_0) e^{-\beta \Delta t} + \kappa_{\text{self}} \cdot \text{OFI}^- + \kappa_{\text{cross}} \cdot \text{OFI}^+$$

Where:
- $\lambda_0$ is the baseline jump intensity.
- $\kappa_{\text{self}}$ is the self-excitation coefficient.
- $\kappa_{\text{cross}}$ is the cross-excitation coefficient.
- $\beta$ is the exponential decay parameter.
- $\text{OFI}^+ = \max(\text{OFI}, 0)$ and $\text{OFI}^- = \max(-\text{OFI}, 0)$ are the directional components of the OFI.

**Stationarity Guard**: The system automatically verifies that the spectral radius of the branching matrix is $< 1$ (i.e., $\kappa_{\text{self}} + \kappa_{\text{cross}} < \beta$). If not, parameters are automatically scaled to force stability ($\rho = 0.8$).

### 3. Asset Characteristic Function
The characteristic function $\phi(u)$ of $x_T = \ln(S_T)$ at time to expiry $\tau = T - t$ is defined in closed form as:

$$\phi(u) = \exp\left( i u x_t + i u b \tau - \frac{1}{2}\sigma^2 u^2 \tau + \lambda^+ \tau \left( e^{i u \mu_j^+ - \frac{1}{2}(\sigma_j^+)^2 u^2} - 1 \right) + \lambda^- \tau \left( e^{i u \mu_j^- - \frac{1}{2}(\sigma_j^-)^2 u^2} - 1 \right) \right)$$

Where the overall martingale-corrected drift is defined by:

$$b = \mu_t - \lambda^+ \kappa^+ - \lambda^- \kappa^- - \frac{1}{2}\sigma^2$$

### 4. Gil-Pelaez Fourier Inversion with Vectorized Gauss-Legendre Quadrature
The theoretical probability $P(S_T > K)$ that the YES option expires in-the-money is solved via the Gil-Pelaez Fourier inversion of the characteristic function with a Black-Scholes control variate:

$$P(S_T > K) = P_{\text{BS}}(S_T > K) + \frac{1}{\pi} \int_0^\infty \text{Im}\left[ \frac{e^{-i u \ln K} \left( \phi(u) - \phi_{\text{BS}}(u) \right)}{u} \right] du$$

The integral is solved using a 64-node Gauss-Legendre quadrature, completely vectorized on NumPy, eliminating the scalar loop of `scipy.integrate.quad` and achieving an approximate 10x speedup compared to the legacy implementation.

---

## Microstructure and Execution Module

### 1. Instantaneous Drift Estimation via L2 OFI
The short-term drift $\mu_t$ is estimated in real time from the Order Flow Imbalance (OFI) extracted from L2 order book quotes:

$$\text{OFI}_t = \Delta \text{Bid}_t - \Delta \text{Ask}_t$$

Liquidity variations at the best book levels are formalized as:

$$\Delta \text{Bid}_t = \begin{cases} I(P^{\text{bid}}_t > P^{\text{bid}}_{t-1}) \cdot Q^{\text{bid}}_t \\ I(P^{\text{bid}}_t = P^{\text{bid}}_{t-1}) \cdot (Q^{\text{bid}}_t - Q^{\text{bid}}_{t-1}) \\ 0 \end{cases}$$

$$\Delta \text{Ask}_t = \begin{cases} I(P^{\text{ask}}_t < P^{\text{ask}}_{t-1}) \cdot Q^{\text{ask}}_t \\ I(P^{\text{ask}}_t = P^{\text{ask}}_{t-1}) \cdot (Q^{\text{ask}}_t - Q^{\text{ask}}_{t-1}) \\ 0 \end{cases}$$

The annualized instantaneous drift is obtained by scaling the smoothed OFI using a multiplier $\gamma$:

$$\mu_t = r + \text{OFI}_{\text{smoothed}} \cdot \gamma \cdot (365.25 \times 24 \times 3600)$$

The resulting Merton probability is stabilized with a time-based EMA filter with an adaptive half-life (compressed near expiry).

### 5. P-Measure vs Q-Measure Pricing
Unlike traditional derivatives pricing which relies on Risk-Neutral (Q-measure) valuation using a martingale compensator, Polymarket binary options require predicting the real-world (P-measure) probability. If a standard martingale compensator is used, an increase in positive jumps (due to bullish OFI) mathematically forces the continuous drift downwards to keep the expected value constant, paradoxically resulting in a bearish probability drop. Polymarket V2 explicitly strips out the martingale jump compensator from the characteristic function to allow the microstructural jump intensities to correctly alter the real-world directional drift.

### 2. Fractional Kelly Sizing
The optimal allocation percentage of portfolio capital on the YES/NO book is calibrated using the fractional Kelly formula with a regularization buffer for taker fees:

$$f^*_{\text{YES}} = \gamma \cdot \frac{p_{\text{yes}} - p_{\text{ask}}}{1 - p_{\text{ask}}}$$

The effective target is regularized with a buffer $\delta = \text{taker fee multiplier} \times \gamma$ to prevent churning on marginal edges.

### 3. Pre-Settlement Unwind State Machine (HFT Liquidation)
To eliminate the terminal option variance typical of 5-minute 0-DTE options held until settlement, the engine implements a dynamic state machine based on the Time-To-Expiry (TTE) and the current spread:
- **Phase 1: Soft Unwind (Reduce-Only)** (TTE $\le 45.0$ seconds):
  Enters reduce-only mode. All active maker orders that would increase the absolute inventory $|q|$ are canceled. The bot is only allowed to quote or execute orders that reduce $|q|$ toward zero (if $q > 0$, it quotes asks to liquidate YES; if $q < 0$, it quotes bids to cover NO; if $q == 0$, trading activity is halted). In the taker filters, all trades except those aimed at flattening exposure are blocked.
- **Phase 2: Hard Liquidation Sweep (Panic Sweep)** (TTE $\le 15.0$ seconds OR (TTE $\le 45.0$ and spread > 0.10 USD)):
  Instantly cancels all pending maker quotes. Completely bypasses Kelly sizing and fires an aggressive Taker Market Order crossing the spread to flatten inventory to zero immediately (selling YES if $q > 0$, or buying YES if $q < 0$). Activates a lock state (`locked`) that inhibits any new position openings until the rollover of the next cycle.

### 4. Centralized Minimum Sizing Parameter
The rigid retail $50 minimum sizing constraints have been removed and centralized inside the `MIN_ORDER_USD` parameter (default `1.0` USD), allowing the bot to execute micro-hedges and micro-orders of 5 USD or 10 USD to finely tune the optimal inventory.

---

## Shadow Order Book Architecture

The `ShadowOrderBook` implements a double-book architecture with a parallel consumption ledger based on exponential decay:

| Component | Description |
|---|---|
| **V_hist** (`q_real_bids/asks`) | Historical state of the L2 book — updated exclusively by WebSocket feed ticks. Never mutated by bot fills. |
| **V_cons** (`ConsumptionTracker`) | Parallel consumption ledger — records ONLY the liquidity consumed by the bot. Decays exponentially with a configurable half-life (default 0.2s) to model market maker re-quoting latency. |
| **V_eff** | Effective volume available: $V_{\text{eff}}(p, t) = \max(0, V_{\text{hist}}(p) - V_{\text{cons}}(p, t))$ |

### Event-Driven Reconciliation
At each feed tick, the tracker performs event-driven reconciliation:
- **Replenishment** ($\Delta V > 0$): $V_{\text{cons}} \leftarrow \max(0, V_{\text{cons}} - \Delta V)$
- **Drop/Removal** ($\Delta V \leq 0$): $V_{\text{cons}} \leftarrow \min(V_{\text{cons}}, V_{\text{new}})$

### Pruning and Caching
- Deep book L2 levels are pruned (top 20 per side) to eliminate CPU overhead on irrelevant levels.
- `get_top_of_book()` and `get_market_top_of_book()` queries are cached and invalidated only on updates or paper fills.

### Double Trade Prevention
The CLOB callback (`_clob_callback`) is an `async` coroutine that `await`s `client.execute_trade(...)` before returning control. This guarantees that `paper_execute` has registered the fills in the consumption tracker before the next tick arrives, avoiding double signals on the same market event.

### Keep-Awake Windows
The system runs an asynchronous loop `_keep_awake_loop` that updates the Windows `SetThreadExecutionState` every 30 seconds with `ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED` flags, preventing sleep during prolonged live trading sessions.

---

## Event-Driven Historical Backtester

The backtest engine (`BacktestRunner`) replicates the live execution pipeline on high-frequency historical data with the following optimizations:

- **Smart Bypass**: Engine evaluations are skipped when probabilities, top-of-book, and positions are unchanged (yielding an 83% reduction in execution calls).
- **Binary Search $O(\log N)$**: Staleness and pin risk queries utilize `bisect` instead of linear scans.
- **Lazy L2 Sorting**: The L2 book is only sorted when a trade needs to be evaluated, not on every tick.
- **Simulated Latency**: Orders are queued with a uniform stochastic delay $[150, 300]$ ms to simulate network latency.

### Output
Each backtest run produces the following in the `logs/YYYY-MM-DD/merton/backtest_YYYYMMDD_HHMMSS/` directory:
- `signals.csv` — generated trading signals
- `trades.csv` — simulated executions
- `summary.json` — complete performance metrics report:
  - Net P&L, Return %, Max Drawdown (% and USD)
  - Win Rate, Profit Factor, Win/Loss Ratio
  - Gross Profit/Loss, Average Win/Loss
  - Full list of realized trades with entry/exit prices and P&L

---

## Repository Structure

```
polymarketv2/
├── config/
│   └── settings.py               # Configuration management via environment variables (.env)
├── src/
│   ├── core/
│   │   ├── base_strategy.py      # Base class for quantitative strategy abstraction
│   │   ├── events.py             # Decoupled HFT logging events and signals
│   │   ├── interfaces.py         # Abstract interfaces for data feeds, execution, and logging
│   │   ├── market_context.py     # Unified MarketContext data model
│   │   └── strike_manager.py     # Strike Price K resolver and manager
│   ├── ingestion/
│   │   ├── live_feeds.py         # WebSocket Chainlink Spot Feed & CLOB L2 Orderbook Feed (YES-only)
│   │   └── market_manager.py     # Dynamic discovery of Gamma API markets and rollover handling
│   ├── strategies/
│   │   ├── factory.py            # Strategy Factory — dynamic resolution of the active strategy
│   │   ├── merton_strategy.py    # Bivariate Hawkes-Merton, vectorized Gil-Pelaez, Vol/OFI calibration
│   │   └── legacy_merton_strategy.py # Legacy Merton strategy (homogeneous Poisson, scipy.integrate.quad)
│   ├── execution/
│   │   ├── shadow_book.py        # Shadow Book with ConsumptionTracker (V_hist + V_cons + V_eff)
│   │   ├── engine.py             # L2 book walk, Kelly sizing, and risk filters (Pin, Desync, PoF)
│   │   └── clients.py            # Mock/Simulated Execution Client for paper trading and backtests
│   ├── backtest/
│   │   └── runner.py             # Event-Driven Backtester with automatic summary.json reporting
│   ├── logging/
│   │   └── recorder.py           # High-performance Parquet (ticks) and CSV (signals/trades) recorder
│   └── ui/
│       ├── dashboard.py          # Rich CLI terminal dashboard (local)
│       ├── web_server.py         # Integrated HTTP/WS server (Bloomberg Web Dashboard)
│       └── dashboard.html        # Bloomberg Stark Terminal UI with Live Monitor + Backtest tabs
├── tests/
│   ├── verify_live_data_and_pricing.py  # Validation of the real-time pricing pipeline
│   ├── verify_phase3.py                 # Validation of Merton pricing and vol calibration
│   ├── verify_phase4.py                 # Validation of shadow book and order execution
│   ├── verify_web_server.py             # Validation of the web server HTTP/WS endpoints
│   ├── verify_factory.py               # Validation of the Strategy Factory
│   └── benchmark_pricing.py            # Performance benchmark of the pricer
├── main.py                  # Entrypoint of the Live/Paper Trading Orchestrator
├── run_backtest.py          # CLI script to run the Historical Backtester
├── pyproject.toml           # Python project configuration and dependency manager (uv)
└── uv.lock                  # Environment lockfile for reproducibility
```

---

## Key Parameters (.env)

| Parameter | Current Value | Description |
|---|---|---|
| `OFI_DRIFT_MULTIPLIER` | `-1e-6` | Scale OFI → annualized drift |
| `DEFAULT_LAMBDA` | `4000` | Baseline jump intensity (jumps/year) |
| `DEFAULT_MU_J` | `0.0001` | Log-normal jump mean |
| `DEFAULT_SIGMA_J` | `0.0015` | Log-normal jump standard deviation |
| `DEFAULT_SIGMA` | `0.25` | Default implied volatility |
| `HAWKES_BETA` | `5.0` | Hawkes exponential decay parameter |
| `HAWKES_KAPPA_SELF` | `3.0` | Hawkes self-excitation coefficient |
| `HAWKES_KAPPA_CROSS` | `1.0` | Hawkes cross-excitation coefficient |
| `KELLY_FRACTION` | `0.05` | Fractional Kelly factor (reduced to 0.05 for HFT) |
| `MIN_EXPECTED_VALUE` | `0.005` | Minimum EV threshold to execute a trade |
| `EMA_HALFLIFE_SEC` | `3.0` | Time-based EMA half-life on Merton probability |
| `MIN_ORDER_USD` | `1.0` | Minimum order size in USD for micro-hedges |
| `MAX_POSITION_SIZE_USD` | `250.0` | Maximum directional taker exposure in USD |
| `MM_MAX_INVENTORY` | `500.0` | Maximum inventory accumulation in maker tokens |
| `MM_RISK_AVERSION` | `2.5` | Risk aversion coefficient gamma for market making (Avellaneda) |

---

## Installation and Usage

The system utilizes the `uv` package manager for ultra-fast virtual environment and dependency resolution.

1. **Environment Setup**:
   Generate the local configuration file from the template:
   ```bash
   cp .env.example .env
   ```

2. **Running the Live Orchestrator**:
   - **Rich CLI Dashboard (Console)**:
     ```bash
     uv run python main.py
     ```
   - **Bloomberg Web Dashboard (Web Terminal)**:
     ```bash
     uv run python main.py --no-term
     ```
     The interactive web terminal will be accessible at **`http://localhost:8080`**.

3. **Starting the Backtest Replayer**:
   - **With a Time Window**:
     ```bash
     uv run python run_backtest.py --start "2026-05-28 19:15:00" --end "2026-05-28 19:20:00"
     ```
   - **With a Single File**:
     ```bash
     uv run python run_backtest.py --file "path/to/ticks.parquet"
     ```

4. **Internal Validation Suite**:
   ```bash
   uv run python tests/verify_phase3.py   # Merton pricing + vol calibration
   uv run python tests/verify_phase4.py   # Shadow book + order execution
   uv run python tests/verify_web_server.py
   ```
