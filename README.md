# Polymarket V2: High-Frequency Options Pricing & Execution Engine

Polymarket V2 è un'architettura ultra-modulare ad alte prestazioni progettata per il pricing quantitativo e l'esecuzione automatizzata sui mercati opzionali binari a 5 minuti di Polymarket (Up/Down). Il sistema implementa il modello a salti di Merton (MJD) risolto nello spazio delle frequenze tramite inversione di Fourier (Gil-Pelaez) con tecniche di riduzione della varianza, integrando un motore di sizing Kelly frazionario e controlli di rischio per contesti High-Frequency Trading (HFT).

---

## Modello Matematico e Pricing

### 1. Dinamica del Sottostante: Merton Jump-Diffusion
Il prezzo del sottostante $S_t$ segue un processo di diffusione con salti governato dalla seguente equazione differenziale stocastica (SDE):

$$dS_t = \mu S_t dt + \sigma S_t dW_t + S_t d\left( \sum_{i=1}^{N_t} (V_i - 1) \right)$$

Dove:
- $\mu$ rappresenta il tasso di drift istantaneo continuo.
- $\sigma$ rappresenta il coefficiente di diffusione continuo (volatilità realizzata).
- $W_t$ rappresenta un moto browniano standard su uno spazio di probabilità filtrato.
- $N_t$ rappresenta un processo di Poisson omogeneo con intensità di salto $\lambda$, indipendente da $W_t$.
- $V_i$ rappresenta l'ampiezza del salto stocastico $i$-esimo, con $Y_i = \ln(V_i) \sim \mathcal{N}(\mu_j, \sigma_j^2)$.

Il log-prezzo $x_t = \ln(S_t)$ segue la dinamica differenziale stocastica:

$$dx_t = \left( \mu - \frac{1}{2}\sigma^2 - \lambda \kappa \right)dt + \sigma dW_t + \sum_{i=1}^{dN_t} Y_i$$

Con il correttore di deriva definito da:

$$\kappa = \mathbb{E}[e^{Y_i}] - 1 = \exp\left(\mu_j + \frac{1}{2}\sigma_j^2\right) - 1$$

### 2. Funzione Caratteristica dell'Asset
La funzione caratteristica $\phi(u)$ di $x_T = \ln(S_T)$ al tempo di scadenza $\tau = T - t$ è definita in forma chiusa come:

$$\phi(u) = \exp\left( i u x_t + i u b \tau - \frac{1}{2}\sigma^2 u^2 \tau + \lambda \tau \left( e^{i u \mu_j - \frac{1}{2}\sigma_j^2 u^2} - 1 \right) \right)$$

Dove la deriva complessiva corretta per la martingala è definita da:

$$b = \mu - \lambda\kappa - \frac{1}{2}\sigma^2$$

### 3. Soluzione di Gil-Pelaez con Riduzione della Varianza
La probabilità teorica $P(S_T > K)$ che l'opzione YES scada in-the-money (cioè che lo spot alla scadenza superi il prezzo strike $K$) viene espressa tramite l'inversione di Fourier di Gil-Pelaez:

$$P(S_T > K) = \frac{1}{2} + \frac{1}{\\pi} \int_0^\infty \text{Im}\left[ \frac{e^{-i u \ln K} \phi(u)}{u} \right] du$$

Per eliminare le instabilità numeriche ad alta frequenza in prossimità della scadenza ($\tau \to 0$), il sistema applica una tecnica di riduzione della varianza basata su Black-Scholes come variata di controllo (Control Variate):

$$P(S_T > K) = P_{\text{BS}}(S_T > K) + \frac{1}{\pi} \int_0^\infty \text{Im}\left[ \frac{e^{-i u \ln K} \left( \phi(u) - \phi_{\text{BS}}(u) \right)}{u} \right] du$$

Dove:
- $P_{\text{BS}}(S_T > K) = \Phi(d_2)$ indica la probabilità analitica del modello geometrico browniano continuo.
- $\phi_{\text{BS}}(u)$ rappresenta la funzione caratteristica di Black-Scholes con i medesimi parametri continui di deriva e volatilità.

---

## Modulo di Microstruttura ed Esecuzione

### 1. Stima del Drift Istantaneo tramite OFI
Il drift di breve termine $\mu$ viene stimato in tempo reale a partire dall'Order Flow Imbalance (OFI) estratto dal book L2 delle quotazioni:

$$\text{OFI}_t = \Delta \text{Bid}_t - \Delta \text{Ask}_t$$

Le variazioni di liquidità ai migliori livelli del book sono formalizzate come:

$$\Delta \text{Bid}_t = \begin{cases} I(P^{\text{bid}}_t > P^{\text{bid}}_{t-1}) \cdot Q^{\text{bid}}_t \\ I(P^{\text{bid}}_t = P^{\text{bid}}_{t-1}) \cdot (Q^{\text{bid}}_t - Q^{\text{bid}}_{t-1}) \\ 0 \end{cases}$$

$$\Delta \text{Ask}_t = \begin{cases} I(P^{\text{ask}}_t < P^{\text{ask}}_{t-1}) \cdot Q^{\text{ask}}_t \\ I(P^{\text{ask}}_t = P^{\text{ask}}_{t-1}) \cdot (Q^{\text{ask}}_t - Q^{\text{ask}}_{t-1}) \\ 0 \end{cases}$$

Il drift istantaneo annualizzato è ottenuto riscalando l'OFI livellato tramite un moltiplicatore $\gamma$ (valore corrente: `-1e-7`, segno negativo validato empiricamente):

$$\mu_t = \mu_{\text{default}} + \text{OFI}_{\text{smoothed}} \cdot \gamma \cdot (365.25 \times 24 \times 3600)$$

La probabilità Merton risultante è stabilizzata con un filtro EMA a $\alpha = 0.05$ (~30s di memoria a 1 tick/s).

### 2. Sizing Frazionario di Kelly
L'esposizione ottimale in percentuale del capitale di portafoglio sul book YES/NO viene calibrata applicando la formula di Kelly frazionaria:

$$f^* = \frac{P \cdot (b + 1) - 1}{b} \cdot f_{\text{Kelly}}$$

Dove:
- $P$ è la probabilità corretta stimata dal modello teorico di Merton.
- $b$ rappresenta le quote del mercato (odds), calcolate come $b = \frac{1 - P_{\text{market}}}{P_{\text{market}}}$.
- $f_{\text{Kelly}}$ indica il fattore frazionario di Kelly per limitare l'over-betting in contesti con rischi di modello o latenza.

---

## Architettura del Shadow Order Book

Il `ShadowOrderBook` mantiene due rappresentazioni distinte e indipendenti del book L2:

| Book | Struttura dati | Scopo |
|---|---|---|
| **Real Book** (`q_real_bids/asks`) | Aggiornato da ogni tick CLOB | Calcolo di `p_mkt` (probabilità implicita di mercato) |
| **Shadow Book** (`q_shadow_bids/asks`) | Stesso del Real Book, ma depleto da `paper_execute` | Calcolo VWAP, sizing Kelly, filtri di rischio |

Il `paper_execute` depleta solo il Shadow Book, mai il Real Book. Questo garantisce che la probabilità di mercato visualizzata nella dashboard rifletta sempre la vera liquidità disponibile, indipendentemente dalle fill simulate.

### Risoluzione dello Strike in Tempo Reale
Se la Gamma API non fornisce lo strike price $K$ all'avvio, il sistema cattura il **primo tick Chainlink** generato immediatamente dopo l'inizio del ciclo (`cycle_start = expiry - 300s`) e lo utilizza come strike di riferimento dell'opzione attiva.

### Prevenzione del Double Trade
Il callback CLOB (`_clob_callback`) è una coroutine `async` che esegue `await client.execute_trade(...)` prima di restituire il controllo al loop. Questo garantisce che `paper_execute` abbia già depleto il Shadow Book prima dell'arrivo del tick successivo, prevenendo segnali duplicati sulla stessa opportunità.

### Mitigazione del "Ghost Token" (Bug Fix)
In precedenza, se la query delle API di Gamma per i nuovi token ID falliva durante il rollover, il sistema manteneva i token del ciclo precedente. Questo portava ad arbitraggi falsi su contratti morti (scambiati a $0.01) con payoff simulati errati.
- **Risoluzione Robustezza Gamma API**: È stato implementato un ciclo di retry con backoff fino a 5 tentativi se le chiamate API falliscono.
- **Gestione Token Stale**: Se i token falliscono del tutto la risoluzione, `YES_TOKEN_ID` e `NO_TOKEN_ID` vengono impostati a `None` e il feed CLOB viene disattivato, prevenendo trading su contratti scaduti.
- **Mid-Cycle Auto-Recovery**: Il loop di discovery riprova periodicamente a risolvere i token mancanti ogni 10 secondi durante il ciclo, riavviando il feed e ripristinando il trading appena l'API ritorna disponibile.

### Keep-Awake Windows
Il sistema esegue un loop asincrono `_keep_awake_loop` che aggiorna ogni 30 secondi il `SetThreadExecutionState` di Windows con i flag `ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED`, prevenendo lo standby durante le sessioni di trading prolungate.

---

## Struttura del Repository

```
polymarketv2/
├── config/
│   └── settings.py          # Gestione configurazioni da variabili d'ambiente (.env)
├── src/
│   ├── core/
│   │   ├── base_strategy.py # Classe base per l'astrazione delle strategie quantitative
│   │   ├── events.py        # Eventi di log e segnali HFT disaccoppiati
│   │   ├── interfaces.py    # Interfacce astratte per feed dati, esecuzione e log
│   │   ├── market_context.py# Modello dati unificato MarketContext
│   │   └── strike_manager.py# Gestore dello Strike Price K
│   ├── ingestion/
│   │   ├── live_feeds.py    # WebSocket Chainlink Spot Feed & CLOB L2 Orderbook Feed (YES-only)
│   │   └── market_manager.py# Dynamic discovery dei mercati Gamma API & Rollover
│   ├── strategies/
│   │   └── merton_strategy.py# Caratteristica Merton, Gil-Pelaez e calibrazione Vol/OFI
│   ├── execution/
│   │   ├── shadow_book.py   # Dual-book L2 (Real + Shadow) con get_market_top_of_book()
│   │   ├── engine.py        # Walk del book L2, Kelly sizing e filtri di rischio (Pin, Desync, PoF)
│   │   └── clients.py       # Mock/Simulated Execution Client per paper trading e backtest
│   ├── logging/
│   │   └── recorder.py      # Scrittura Parquet (ticks) e CSV (segnali/esecuzioni) ad alte prestazioni
│   └── ui/
│       ├── dashboard.py     # Terminal Dashboard Rich CLI (locale)
│       ├── web_server.py    # Integrated HTTP/WS Server (Bloomberg Web Dashboard)
│       └── dashboard.html   # Bloomberg Stark Terminal UI (Flat black, no shadows, 2D canvases)
├── tests/
│   ├── verify_dashboard.py            # Convalida terminal CLI Rich Dashboard
│   ├── verify_live_data_and_pricing.py# Convalida della pipeline dei prezzi in tempo reale
│   ├── verify_phase1.py               # Convalida dei log Parquet e modulo dati
│   ├── verify_phase2.py               # Convalida slug e risoluzione strike
│   ├── verify_phase3.py               # Convalida pricing Merton e calibrazione vol
│   ├── verify_phase4.py               # Convalida shadow book ed esecuzione ordini
│   └── verify_web_server.py           # Convalida degli endpoint HTTP/WS del web server
├── main.py                  # Entrypoint dell'Orchestratore Live/Paper Trading
├── run_backtest.py          # Script CLI per avviare il Backtester Storico Temporale
├── pyproject.toml           # Gestione dipendenze e configurazione del progetto Python (uv)
└── uv.lock                  # Lockfile di riproducibilità ambientale
```

---

## Parametri Chiave (.env)

| Parametro | Valore corrente | Descrizione |
|---|---|---|
| `OFI_DRIFT_MULTIPLIER` | `-1e-7` | Scala OFI → drift annualizzato (segno negativo validato empiricamente) |
| `DEFAULT_LAMBDA` | `4000` | Intensità di salto Poisson (salti/anno) |
| `DEFAULT_MU_J` | `0.0001` | Media log-normale del salto |
| `DEFAULT_SIGMA_J` | `0.0015` | Deviazione standard del salto |
| `DEFAULT_SIGMA` | `0.25` | Volatilità implicita di default |
| `KELLY_FRACTION` | `0.1` | Fattore frazionario di Kelly |
| `MIN_EXPECTED_VALUE` | `0.015` | Soglia minima di EV per eseguire un trade |
| `EMA_ALPHA` (Merton) | `0.05` | Smoothing EMA sulla probabilità Merton (~30s memoria) |
| `EMA_ALPHA` (OFI) | `0.1` | Smoothing EMA sull'Order Flow Imbalance |

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
     uv run python run_backtest.py --start "2026-05-23 20:51:00" --end "2026-05-23 21:51:00"
     ```
   - **Con File Singolo**:
     ```bash
     uv run python run_backtest.py --file "c:/percorso/del/tuo/file.parquet"
     ```

4. **Suite di Validazione Interna**:
   ```bash
   uv run python tests/verify_live_data_and_pricing.py
   uv run python tests/verify_phase1.py
   uv run python tests/verify_phase2.py
   uv run python tests/verify_phase3.py
   uv run python tests/verify_phase4.py
   uv run python tests/verify_web_server.py
   ```
