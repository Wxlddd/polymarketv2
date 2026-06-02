# Polymarket V2: High-Frequency Options Pricing & Execution Engine

Polymarket V2 è un'architettura ultra-modulare ad alte prestazioni progettata per il pricing quantitativo e l'esecuzione automatizzata sui mercati opzionali binari a 5 minuti di Polymarket (Up/Down). Il sistema implementa il modello a salti di Merton (MJD) con intensità di salto stocastiche guidate da un processo di Hawkes bivariato accoppiato, risolto nello spazio delle frequenze tramite inversione di Fourier (Gil-Pelaez) con quadratura vettorizzata di Gauss-Legendre e variata di controllo Black-Scholes, integrando un motore di sizing Kelly frazionario e controlli di rischio per contesti High-Frequency Trading (HFT).

---

## Modello Matematico e Pricing

### 1. Dinamica del Sottostante: Hawkes-Driven Merton Jump-Diffusion
Il prezzo del sottostante $S_t$ segue un processo di diffusione con salti asimmetrici governato dalla seguente equazione differenziale stocastica (SDE):

$$dS_t = \mu_t S_t dt + \sigma S_t dW_t + S_t d\left( \sum_{i=1}^{N_t^+} (V_i^+ - 1) \right) + S_t d\left( \sum_{j=1}^{N_t^-} (V_j^- - 1) \right)$$

Dove:
- $\mu_t$ rappresenta il tasso di drift istantaneo continuo, modulato opzionalmente dall'OFI.
- $\sigma$ rappresenta il coefficiente di diffusione continuo (volatilità realizzata HF).
- $W_t$ rappresenta un moto browniano standard su uno spazio di probabilità filtrato.
- $N_t^+, N_t^-$ rappresentano processi di conteggio con intensità stocastiche $\lambda^+(t), \lambda^-(t)$.
- $V_i^+$ e $V_j^-$ rappresentano le ampiezze di salto positive e negative rispettivamente, con $Y^+ = \ln(V^+) \sim \mathcal{N}(\mu_j^+, (\sigma_j^+)^2)$ e $Y^- = \ln(V^-) \sim \mathcal{N}(\mu_j^-, (\sigma_j^-)^2)$.

### 2. Processo di Hawkes Bivariato Accoppiato
Le intensità di salto $\lambda^+(t)$ e $\lambda^-(t)$ evolvono secondo un processo di Hawkes bivariato simmetrico con eccitazione incrociata:

$$\lambda^+(t_k) = \lambda_0 + (\lambda^+(t_{k-1}) - \lambda_0) e^{-\beta \Delta t} + \kappa_{\text{self}} \cdot \text{OFI}^+ + \kappa_{\text{cross}} \cdot \text{OFI}^-$$

$$\lambda^-(t_k) = \lambda_0 + (\lambda^-(t_{k-1}) - \lambda_0) e^{-\beta \Delta t} + \kappa_{\text{self}} \cdot \text{OFI}^- + \kappa_{\text{cross}} \cdot \text{OFI}^+$$

Dove:
- $\lambda_0$ è l'intensità di base (baseline intensity).
- $\kappa_{\text{self}}$ è il coefficiente di auto-eccitazione.
- $\kappa_{\text{cross}}$ è il coefficiente di eccitazione incrociata.
- $\beta$ è il parametro di decadimento esponenziale.
- $\text{OFI}^+ = \max(\text{OFI}, 0)$ e $\text{OFI}^- = \max(-\text{OFI}, 0)$ sono le componenti direzionali dell'OFI.

**Vincolo di stazionarietà**: Il sistema verifica automaticamente che il raggio spettrale della matrice di branching sia $< 1$ (cioè $\kappa_{\text{self}} + \kappa_{\text{cross}} < \beta$). In caso contrario, i parametri vengono riscalati automaticamente per garantire la stabilità ($\rho = 0.8$).

### 3. Funzione Caratteristica dell'Asset
La funzione caratteristica $\phi(u)$ di $x_T = \ln(S_T)$ al tempo di scadenza $\tau = T - t$ è definita in forma chiusa come:

$$\phi(u) = \exp\left( i u x_t + i u b \tau - \frac{1}{2}\sigma^2 u^2 \tau + \lambda^+ \tau \left( e^{i u \mu_j^+ - \frac{1}{2}(\sigma_j^+)^2 u^2} - 1 \right) + \lambda^- \tau \left( e^{i u \mu_j^- - \frac{1}{2}(\sigma_j^-)^2 u^2} - 1 \right) \right)$$

Dove la deriva complessiva corretta per la martingala è definita da:

$$b = \mu_t - \lambda^+ \kappa^+ - \lambda^- \kappa^- - \frac{1}{2}\sigma^2$$

### 4. Soluzione di Gil-Pelaez con Gauss-Legendre Vettorizzato
La probabilità teorica $P(S_T > K)$ che l'opzione YES scada in-the-money viene espressa tramite l'inversione di Fourier di Gil-Pelaez con variata di controllo Black-Scholes:

$$P(S_T > K) = P_{\text{BS}}(S_T > K) + \frac{1}{\pi} \int_0^\infty \text{Im}\left[ \frac{e^{-i u \ln K} \left( \phi(u) - \phi_{\text{BS}}(u) \right)}{u} \right] du$$

L'integrale viene risolto con quadratura di Gauss-Legendre a 64 nodi, completamente vettorizzata su NumPy, eliminando il loop scalare di `scipy.integrate.quad` e ottenendo un speedup di circa 10x rispetto all'implementazione precedente.

---

## Modulo di Microstruttura ed Esecuzione

### 1. Stima del Drift Istantaneo tramite OFI
Il drift di breve termine $\mu_t$ viene stimato in tempo reale a partire dall'Order Flow Imbalance (OFI) estratto dal book L2 delle quotazioni:

$$\text{OFI}_t = \Delta \text{Bid}_t - \Delta \text{Ask}_t$$

Le variazioni di liquidità ai migliori livelli del book sono formalizzate come:

$$\Delta \text{Bid}_t = \begin{cases} I(P^{\text{bid}}_t > P^{\text{bid}}_{t-1}) \cdot Q^{\text{bid}}_t \\ I(P^{\text{bid}}_t = P^{\text{bid}}_{t-1}) \cdot (Q^{\text{bid}}_t - Q^{\text{bid}}_{t-1}) \\ 0 \end{cases}$$

$$\Delta \text{Ask}_t = \begin{cases} I(P^{\text{ask}}_t < P^{\text{ask}}_{t-1}) \cdot Q^{\text{ask}}_t \\ I(P^{\text{ask}}_t = P^{\text{ask}}_{t-1}) \cdot (Q^{\text{ask}}_t - Q^{\text{ask}}_{t-1}) \\ 0 \end{cases}$$

Il drift istantaneo annualizzato è ottenuto riscalando l'OFI livellato tramite un moltiplicatore $\gamma$:

$$\mu_t = r + \text{OFI}_{\text{smoothed}} \cdot \gamma \cdot (365.25 \times 24 \times 3600)$$

La probabilità Merton risultante è stabilizzata con un filtro EMA time-based con halflife adattivo (compresso verso la scadenza).

### 2. Sizing Frazionario di Kelly
L'esposizione ottimale in percentuale del capitale di portafoglio sul book YES/NO viene calibrata applicando la formula di Kelly frazionaria con buffer di regolarizzazione per le commissioni taker:

$$f^*_{\text{YES}} = \gamma \cdot \frac{p_{\text{yes}} - p_{\text{ask}}}{1 - p_{\text{ask}}}$$

Il target effettivo è regolarizzato con un buffer $\delta = \text{taker fee multiplier} \times \gamma$ per evitare churning su edge marginali.

### 3. Macchina a Stati del Pre-Settlement Unwind (Liquidazione HFT)
Per eliminare la varianza terminale tipica delle opzioni 0-DTE a 5 minuti detenute fino alla scadenza, il motore implementa una macchina a stati dinamica basata sul tempo rimanente alla scadenza (TTE, Time-To-Expiry) e sullo spread corrente:
- **Fase 1: Soft Unwind (Reduce-Only)** (TTE $\le 45.0$ secondi):
  Entra in modalità reduce-only. Vengono cancellati tutti gli ordini maker attivi che aumenterebbero l'inventario assoluto $|q|$. È consentito quotare o eseguire solo operazioni che riducono $|q|$ verso lo zero (se $q > 0$ si quota solo ASK per liquidare YES; se $q < 0$ si quota solo BID per coprire NO; se $q == 0$ si azzera l'attività di trading). Nei filtri taker, vengono bloccati tutti i trade tranne quelli diretti ad appiattire l'esposizione.
- **Fase 2: Hard Liquidation Sweep (Panic Sweep)** (TTE $\le 15.0$ secondi OR (TTE $\le 45.0$ e spread > 0.10 USD)):
  Cancella istantaneamente qualsiasi quotazione maker pendente. Ignora completamente il sizing di Kelly e spara un ordine Taker Market aggressivo che incrocia il book per appiattire l'inventario istantaneamente a zero (vendendo YES se $q > 0$, o comprando YES se $q < 0$). Attiva uno stato di blocco (`locked`) che inibisce nuove aperture fino al rollover del ciclo successivo.

### 4. Parametro di Sizing Minimo Centralizzato
Le soglie rigide di dimensione minima al dettaglio di 50 USD sono state rimosse e centralizzate nel parametro `MIN_ORDER_USD` (default `1.0` USD), consentendo al bot di eseguire micro-operazioni e micro-coperture da 5 USD o 10 USD per sintonizzare finemente l'inventario ottimale.

---

## Architettura del Shadow Order Book

Il `ShadowOrderBook` implementa un'architettura a doppio libro con un ledger di consumo parallelo basato su decadimento esponenziale:

| Componente | Descrizione |
|---|---|
| **V_hist** (`q_real_bids/asks`) | Stato storico del book L2 — aggiornato esclusivamente dai tick del feed WebSocket. Mai mutato da paper fills. |
| **V_cons** (`ConsumptionTracker`) | Ledger di consumo parallelo — registra SOLO la liquidità consumata dal bot. Decade esponenzialmente con half-life configurabile (default 0.2s) per modellare la latenza di re-quoting dei Market Maker. |
| **V_eff** | Volume effettivo disponibile: $V_{\text{eff}}(p, t) = \max(0, V_{\text{hist}}(p) - V_{\text{cons}}(p, t))$ |

### Riconciliazione Event-Driven
Ad ogni tick del feed, il tracker esegue una riconciliazione event-driven:
- **Replenishment** ($\Delta V > 0$): $V_{\text{cons}} \leftarrow \max(0, V_{\text{cons}} - \Delta V)$
- **Drop/Removal** ($\Delta V \leq 0$): $V_{\text{cons}} \leftarrow \min(V_{\text{cons}}, V_{\text{new}})$

### Pruning e Caching
- I livelli profondi vengono potati (top 20 per lato) per eliminare overhead CPU su livelli irrilevanti.
- Le query `get_top_of_book()` e `get_market_top_of_book()` sono cachate e invalidate solo su aggiornamenti o paper fills.

### Prevenzione del Double Trade
Il callback CLOB (`_clob_callback`) è una coroutine `async` che esegue `await client.execute_trade(...)` prima di restituire il controllo al loop. Questo garantisce che `paper_execute` abbia già registrato i fill nel consumption tracker prima dell'arrivo del tick successivo.

### Keep-Awake Windows
Il sistema esegue un loop asincrono `_keep_awake_loop` che aggiorna ogni 30 secondi il `SetThreadExecutionState` di Windows con i flag `ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED`, prevenendo lo standby durante le sessioni di trading prolungate.

---

## Backtester Storico Event-Driven

Il motore di backtest (`BacktestRunner`) replica fedelmente la pipeline di esecuzione live su dati storici ad alta frequenza, con le seguenti ottimizzazioni:

- **Bypass intelligente**: Le valutazioni dell'engine vengono saltate quando probabilità, top-of-book e posizioni non sono cambiati (riduzione del 83% delle chiamate).
- **Ricerca binaria $O(\log N)$**: Le query di staleness e pin risk usano `bisect` invece di scansioni lineari.
- **L2 sorting lazy**: Il book L2 viene ordinato solo quando serve valutare un trade, non su ogni tick.
- **Latenza simulata**: Gli ordini vengono messi in coda con un delay stocastico uniforme $[150, 300]$ ms per simulare la latenza di rete.

### Output
Ogni run di backtest produce nella cartella `logs/YYYY-MM-DD/merton/backtest_YYYYMMDD_HHMMSS/`:
- `signals.csv` — segnali di trading generati
- `trades.csv` — esecuzioni simulate
- `summary.json` — report completo con tutte le metriche di performance:
  - Net P&L, Return %, Max Drawdown (% e USD)
  - Win Rate, Profit Factor, Win/Loss Ratio
  - Gross Profit/Loss, Average Win/Loss
  - Lista completa dei trade realizzati con entry/exit price e P&L

---

## Struttura del Repository

```
polymarketv2/
├── config/
│   └── settings.py               # Gestione configurazioni da variabili d'ambiente (.env)
├── src/
│   ├── core/
│   │   ├── base_strategy.py      # Classe base per l'astrazione delle strategie quantitative
│   │   ├── events.py             # Eventi di log e segnali HFT disaccoppiati
│   │   ├── interfaces.py         # Interfacce astratte per feed dati, esecuzione e log
│   │   ├── market_context.py     # Modello dati unificato MarketContext
│   │   └── strike_manager.py     # Gestore dello Strike Price K
│   ├── ingestion/
│   │   ├── live_feeds.py         # WebSocket Chainlink Spot Feed & CLOB L2 Orderbook Feed (YES-only)
│   │   └── market_manager.py     # Dynamic discovery dei mercati Gamma API & Rollover
│   ├── strategies/
│   │   ├── factory.py            # Strategy Factory — risoluzione dinamica della strategia attiva
│   │   ├── merton_strategy.py    # Hawkes-Merton bivariato, Gil-Pelaez vettorizzato, calibrazione Vol/OFI
│   │   └── legacy_merton_strategy.py # Legacy Merton strategy (Poisson omogeneo, scipy.integrate.quad)
│   ├── execution/
│   │   ├── shadow_book.py        # Shadow Book con ConsumptionTracker (V_hist + V_cons + V_eff)
│   │   ├── engine.py             # Walk del book L2, Kelly sizing e filtri di rischio (Pin, Desync, PoF)
│   │   └── clients.py            # Mock/Simulated Execution Client per paper trading e backtest
│   ├── backtest/
│   │   └── runner.py             # Event-Driven Backtester con summary.json automatico
│   ├── logging/
│   │   └── recorder.py           # Scrittura Parquet (ticks) e CSV (segnali/esecuzioni) ad alte prestazioni
│   └── ui/
│       ├── dashboard.py          # Terminal Dashboard Rich CLI (locale)
│       ├── web_server.py         # Integrated HTTP/WS Server (Bloomberg Web Dashboard)
│       └── dashboard.html        # Bloomberg Stark Terminal UI con tab Live Monitor + Backtester
├── tests/
│   ├── verify_live_data_and_pricing.py  # Convalida della pipeline dei prezzi in tempo reale
│   ├── verify_phase3.py                 # Convalida pricing Merton e calibrazione vol
│   ├── verify_phase4.py                 # Convalida shadow book ed esecuzione ordini
│   ├── verify_web_server.py             # Convalida degli endpoint HTTP/WS del web server
│   ├── verify_factory.py               # Convalida della Strategy Factory
│   └── benchmark_pricing.py            # Benchmark performance del pricer
├── main.py                  # Entrypoint dell'Orchestratore Live/Paper Trading
├── run_backtest.py          # Script CLI per avviare il Backtester Storico Temporale
├── pyproject.toml           # Gestione dipendenze e configurazione del progetto Python (uv)
└── uv.lock                  # Lockfile di riproducibilità ambientale
```

---

## Parametri Chiave (.env)

| Parametro | Valore corrente | Descrizione |
|---|---|---|
| `OFI_DRIFT_MULTIPLIER` | `-1e-6` | Scala OFI → drift annualizzato |
| `DEFAULT_LAMBDA` | `4000` | Intensità di salto baseline (salti/anno) |
| `DEFAULT_MU_J` | `0.0001` | Media log-normale del salto |
| `DEFAULT_SIGMA_J` | `0.0015` | Deviazione standard del salto |
| `DEFAULT_SIGMA` | `0.25` | Volatilità implicita di default |
| `HAWKES_BETA` | `5.0` | Parametro di decadimento esponenziale Hawkes |
| `HAWKES_KAPPA_SELF` | `3.0` | Coefficiente di auto-eccitazione Hawkes |
| `HAWKES_KAPPA_CROSS` | `1.0` | Coefficiente di eccitazione incrociata Hawkes |
| `KELLY_FRACTION` | `0.05` | Fattore frazionario di Kelly (ridotto a 0.05 per HFT) |
| `MIN_EXPECTED_VALUE` | `0.005` | Soglia minima di EV per eseguire un trade |
| `EMA_HALFLIFE_SEC` | `3.0` | Halflife EMA time-based sulla probabilità Merton |
| `MIN_ORDER_USD` | `1.0` | Dimensione minima dell'ordine in USD per micro-coperture |
| `MAX_POSITION_SIZE_USD` | `250.0` | Massima esposizione direzionale taker in USD |
| `MM_MAX_INVENTORY` | `500.0` | Massimo accumulo di inventario in token maker |
| `MM_RISK_AVERSION` | `2.5` | Coefficiente gamma di avversione al rischio maker (Avellaneda) |

---

## Installazione e Utilizzo

Il sistema utilizza lo strumento `uv` per la gestione rapida dell'ambiente virtuale e delle librerie.

1. **Predisposizione dell'ambiente**:
   Generare il file di configurazione locale a partire dal template:
   ```bash
   cp .env.example .env
   ```

2. **Esecuzione dell'Orchestratore Live**:
   - **Rich CLI Dashboard (Console)**:
     ```bash
     uv run python main.py
     ```
   - **Bloomberg Web Dashboard (Interfaccia Web)**:
     ```bash
     uv run python main.py --no-term
     ```
     La dashboard interattiva sarà accessibile all'indirizzo **`http://localhost:8080`**.

3. **Avvio del Backtest Replayer**:
   - **Con Finestra Temporale**:
     ```bash
     uv run python run_backtest.py --start "2026-05-28 19:15:00" --end "2026-05-28 19:20:00"
     ```
   - **Con File Singolo**:
     ```bash
     uv run python run_backtest.py --file "path/to/ticks.parquet"
     ```

4. **Suite di Validazione Interna**:
   ```bash
   uv run python tests/verify_phase3.py   # Pricing Merton + vol calibration
   uv run python tests/verify_phase4.py   # Shadow book + order execution
   uv run python tests/verify_web_server.py
   ```
