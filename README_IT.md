# Polymarket V2 — Motore di pricing Hawkes–Merton e market making per i mercati Up/Down crypto

[English](README.md) | **Italiano**

Polymarket V2 è un sistema event-driven di ricerca e paper trading per i mercati binari crypto **Up/Down** a durata fissa di Polymarket (`btc-updown-5m-…`, `btc-updown-4h-…`, …). Prezza il contratto YES con un modello jump-diffusion di Merton le cui intensità di salto sono guidate da un processo di Hawkes bivariato alimentato dall'order-flow imbalance L2, e opera sul CLOB con un router di market making in stile Avellaneda–Stoikov che può anche incrociare lo spread quando l'edge del modello è elevato e che appiattisce l'inventario prima del settlement.

Tutto gira sui feed reali di Polymarket (relay spot Chainlink e WebSocket L2 del CLOB) ma esegue su un exchange simulato: può quindi restare in esecuzione non presidiato per raccogliere tick e P&L su carta. La stessa pipeline di esecuzione è usata dal backtester storico.

> **Stato.** Solo paper trading — non esiste invio di ordini reali. La strategia non è stata dimostrata profittevole; vedi [Limiti noti](#limiti-noti-e-questioni-aperte). Questo repository va considerato infrastruttura di ricerca, non un segnale.

---

## Indice

1. [Architettura](#architettura)
2. [Modello di pricing](#modello-di-pricing)
3. [Motore di esecuzione](#motore-di-esecuzione)
4. [Shadow order book](#shadow-order-book)
5. [Cicli di mercato: 5 minuti, 4 ore e oltre](#cicli-di-mercato-5-minuti-4-ore-e-oltre)
6. [Backtester](#backtester)
7. [Logging dei dati](#logging-dei-dati)
8. [Installazione e utilizzo](#installazione-e-utilizzo)
9. [Esecuzione non presidiata e raccolta dati](#esecuzione-non-presidiata-e-raccolta-dati)
10. [Riferimento configurazione](#riferimento-configurazione)
11. [Script di validazione](#script-di-validazione)
12. [Struttura del repository](#struttura-del-repository)
13. [Limiti noti e questioni aperte](#limiti-noti-e-questioni-aperte)

---

## Architettura

```
ChainlinkSpotFeed (WS)          ClobOrderBookFeed (WS, solo token YES)
        │                                   │
        └──────────────┬────────────────────┘
                       ▼
        LiveOrchestrator._clob_callback()            main.py
                       │
        ShadowOrderBook.update_book()  → OFI, riconciliazione V_hist / V_cons
                       │
        MertonStrategy.get_probability() → p_yes grezza (Hawkes–Merton, Gil-Pelaez GL64)
                       │
        EMA time-based (half-life compressa verso la scadenza)
                       │
        DivergenceVelocityFilter.get_scale() → scala di sizing anti flusso tossico
                       │
        MockExecutionClient.process_market_data() → fill degli ordini maker in attesa
                       │
        ExecutionEngine.evaluate_and_route()  → OrderInstruction[] (A / B / C / REDUCE / PANIC)
                       │
        MockExecutionClient.process_instruction()  (awaited: i fill sono registrati prima del tick successivo)
                       │
        DataRecorder  (ticks.parquet, signals.csv, trades.csv)
```

La discovery dei mercati e il rollover girano in un loop separato (`MarketManager`): la scadenza del prossimo ciclo è derivata dall'orologio di sistema, lo slug dell'evento è generato in modo deterministico, gli ID dei token YES/NO sono letti dalla Gamma API, il feed CLOB viene ri-sottoscritto e il ciclo precedente è liquidato alla sua scadenza reale usando l'ultimo tick Chainlink prima della scadenza.

| Percorso | Responsabilità |
|---|---|
| `main.py` | `LiveOrchestrator` — event loop asincrono che coordina feed, pricing, routing, settlement, logging e UI |
| `config/settings.py` | Dataclass di configurazione immutabili caricate da `.env` |
| `src/ingestion/live_feeds.py` | `ChainlinkSpotFeed` (WS relay spot), `ClobOrderBookFeed` (WS L2 CLOB) |
| `src/ingestion/market_manager.py` | Generazione deterministica degli slug, discovery Gamma API, orologio dei rollover |
| `src/core/strike_manager.py` | Risoluzione e blocco dello strike `K` per ciclo |
| `src/core/market_context.py` | `MarketContext` — il contratto dati immutabile per tick |
| `src/strategies/merton_strategy.py` | Pricer Hawkes–Merton (Gauss–Legendre vettorizzato), calibratore di volatilità realizzata |
| `src/strategies/legacy_merton_strategy.py` | Merton a Poisson omogeneo con shift logit posteriore dell'OFI (`--strategy legacy_merton`) |
| `src/execution/engine.py` | `ExecutionRouter` (regimi) + wrapper `ExecutionEngine` |
| `src/execution/divergence_filter.py` | Guardia su velocità / accelerazione della divergenza |
| `src/execution/shadow_book.py` | `ShadowOrderBook` con `ConsumptionTracker` |
| `src/execution/clients.py` | `MockExecutionClient` — exchange simulato: walk IOC del book, coda maker, settlement |
| `src/backtest/runner.py` | Replay event-driven attraverso la pipeline identica |
| `src/logging/recorder.py` | Recorder Parquet / CSV con rollover giornaliero delle directory |
| `src/ui/web_server.py`, `src/ui/dashboard.html` | Dashboard aiohttp HTTP + WebSocket (tab Live Monitor + Backtester) |
| `src/ui/dashboard.py` | Dashboard terminale Rich (`--term`) |

---

## Modello di pricing

### Dinamica del sottostante

Il log-prezzo segue una jump-diffusion con flussi di salto positivi e negativi separati:

$$dS_t = \mu_t S_t\,dt + \sigma S_t\,dW_t + S_t\,d\Big(\sum_{i=1}^{N_t^+}(V_i^+-1)\Big) + S_t\,d\Big(\sum_{j=1}^{N_t^-}(V_j^--1)\Big)$$

con $\ln V^{\pm} \sim \mathcal N(\pm|\mu_j|, \sigma_j^2)$ e processi di conteggio $N^{\pm}$ con intensità stocastiche $\lambda^{\pm}(t)$.

### Intensità di Hawkes bivariate

Le intensità sono aggiornate tick per tick dall'order-flow imbalance direzionale $\text{OFI}^{+}=\max(\text{OFI},0)$, $\text{OFI}^{-}=\max(-\text{OFI},0)$ con auto- ed etero-eccitazione e decadimento esponenziale:

$$\lambda^{\pm}(t_k) = \lambda_0 + \big(\lambda^{\pm}(t_{k-1})-\lambda_0\big)e^{-\beta\Delta t} + \kappa_{\text{self}}\,\text{OFI}^{\pm} + \kappa_{\text{cross}}\,\text{OFI}^{\mp}$$

Una guardia di stazionarietà riscala $(\kappa_{\text{self}},\kappa_{\text{cross}})$ al primo tick se $\kappa_{\text{self}}+\kappa_{\text{cross}} \ge \beta$, imponendo un branching ratio di $0.8$.

### Intensità di salto integrata nel tempo

$\lambda^{\pm}(t)$ è un tasso *istantaneo* che può impennarsi su un singolo tick e rilassarsi verso $\lambda_0$ in $\sim 1/\beta$ secondi. La funzione caratteristica quindi **non** usa $\lambda^{\pm}(t)\,\tau$, ma il numero atteso di salti sull'orizzonte residuo sotto la dinamica mean-reverting:

$$\Lambda^{\pm}(\tau) = \int_0^{\tau}\mathbb E[\lambda^{\pm}(t+s)]\,ds = \lambda_0\,\tau + \big(\lambda^{\pm}(t)-\lambda_0\big)\frac{1-e^{-\beta\tau}}{\beta}$$

Questo limita il contributo di qualsiasi eccitazione transitoria ad al più $(\lambda-\lambda_0)/\beta$ salti attesi aggiuntivi, indipendentemente dalla distanza dalla scadenza. Senza di esso uno spike di OFI di un solo tick verrebbe moltiplicato per l'intero tempo alla scadenza — una distorsione che cresce linearmente con la lunghezza del ciclo e che rendeva il modello inutilizzabile oltre pochi minuti.

### Funzione caratteristica e inversione

$$\phi(u) = \exp\!\Big( iu\,x_t + iu\,b\,\tau - \tfrac12\sigma^2u^2\tau + \Lambda^{+}(\tau)\big(e^{iu\mu_j^+-\frac12\sigma_j^2u^2}-1\big) + \Lambda^{-}(\tau)\big(e^{iu\mu_j^--\frac12\sigma_j^2u^2}-1\big)\Big), \qquad b = \mu_t - \tfrac12\sigma^2$$

Il compensatore martingala dei salti è volutamente omesso: l'obiettivo è la probabilità nel mondo reale (misura P), e un compensatore in misura Q spingerebbe il drift continuo *verso il basso* quando un flusso rialzista alza $\lambda^+$.

$P(S_T > K)$ si ottiene per inversione di Gil-Pelaez con Black–Scholes come variata di controllo,

$$P(S_T>K) = N(d_2) + \frac1\pi\int_0^\infty \operatorname{Im}\!\left[\frac{e^{-iu\ln K}\big(\phi(u)-\phi_{\text{BS}}(u)\big)}{u}\right]du,$$

valutata con una quadratura di Gauss–Legendre vettorizzata a 64 nodi. Entro 15 s dalla scadenza l'integrale è saltato e si usa $N(d_2)$; entro 1 s si restituisce la funzione gradino del payoff.

### Input

* **Volatilità** — `HighFrequencyVolatilityCalibrator`: campiona lo spot a intervalli ≥5 s, azzera il buffer su un salto >1,5 %, tronca i singoli rendimenti a ±0,5 %, annualizza la deviazione standard realizzata su una finestra mobile (`VOL_ROLLING_WINDOW_SEC`), con clip a [15 %, 300 %].
* **Drift** — zero per default; `USE_LOCAL_INFORMED_DRIFT=True` imposta $\mu_t = \text{OFI}\cdot\text{OFI\_DRIFT\_MULTIPLIER}$.
* **Smoothing** — la probabilità grezza passa in una EMA time-based con half-life `EMA_HALFLIFE_SEC`, compressa a $\tau/10$ vicino alla scadenza così che la stima possa collassare a 0/1 come fa il mercato.

---

## Motore di esecuzione

`ExecutionRouter.evaluate_regimes()` è una macchina a stati valutata a ogni tick sul prezzo modello $\hat p$, il top of book, l'inventario netto $q = \text{YES} - \text{NO}$ e il tempo alla scadenza $\tau$. Emette `OrderInstruction` (`NEW` / `REPLACE` / `CANCEL`) che il client esegue.

**Prezzo di riserva e quote.** Con $q_{\text{norm}} = \operatorname{clamp}(q/Q_{\max},-1,1)$ e la varianza EWMA $\sigma^2$ del mid-price del *contratto* (campionamento a 10 s):

$$P_{\text{res}} = \hat p - \gamma\,q_{\text{norm}}\,\sigma^2, \qquad \delta = \text{fee buffer} + \tfrac12\gamma\sigma^2\tau + \text{toxicity buffer}$$

Le quote target sono $P_{\text{res}}\pm\delta$ arrotondate al tick e mantenute post-only (almeno un tick dentro lo spread).

| Regime | Trigger | Comportamento |
|---|---|---|
| **A — Maker** | default | Espone un bid e un ask alle quote target. La size del bid è la capacità di inventario residua, quella dell'ask è lo YES detenuto; entrambe scalate dal filtro di divergenza. Ri-quota solo se il prezzo si muove di ≥ `MM_REQUOTE_THRESHOLD`. |
| **B — Taker** | bid target $>$ best ask $+\epsilon$ (o ask target $<$ best bid $-\epsilon$) | Cancella le quote e incrocia lo spread con size Kelly frazionaria $f^* = \text{KELLY\_FRACTION}\cdot\frac{\hat p - p_{\text{ask}}}{1-p_{\text{ask}}}$ (cap al 50 % della ricchezza, minimo `MM_MAKER_SIZE`, limitata da cash e inventario). Compra NO quando l'edge è short e non si detiene YES. |
| **C — Unwind** | $\lvert q\rvert > Q_{\max}$, oppure $\lvert\hat p - p_{\text{mid}}\rvert <$ `MM_UNWIND_THRESHOLD` con $q\ne0$ | Quota solo sul lato che riduce $\lvert q\rvert$. |
| **REDUCE** | $\tau \le 45$ s | Reduce-only: cancella il lato che aggiungerebbe inventario; se piatto, cancella tutto. |
| **PANIC** | $\tau \le 15$ s, oppure $\tau \le 45$ s con spread $> 0.10$ | Cancella tutte le quote e invia uno sweep IOC per l'intera posizione netta, prezzato attraverso il book (`best bid − PANIC_CONCESSION` vendendo YES, `best ask + PANIC_CONCESSION` coprendo NO). Ri-emesso a ogni tick fino a posizione piatta; attiva un lock che blocca nuove posizioni fino al rollover. |

Il **filtro di velocità della divergenza** traccia $d_t = \hat p_{\text{EMA}} - p_{\text{mkt}}$, la sua velocità su `VELOCITY_LOOKBACK_SECONDS` e la sua accelerazione, e restituisce una scala di sizing $\max\big(0, 1-(|v|/V_{\max})^{\gamma}\big)$ che va a zero quando la divergenza accelera allontanandosi dal mercato — in quello stato tutte le quote vengono ritirate.

**Exchange simulato (`MockExecutionClient`).** Gli ordini taker percorrono il book L2 effettivo come IOC fino al prezzo limite, pagano `TAKER_FEE_MULTIPLIER · p(1−p)` per contratto più `GAS_FEE_USD` e sono soggetti a un rifiuto stocastico logistico in size e volatilità. Gli ordini maker restano in attesa con una posizione stimata in coda, aggiornata dalle variazioni di profondità osservate, e vengono eseguiti al prezzo limite senza fee taker. Le coppie YES/NO sono convertite automaticamente in cash. Il settlement paga 1,0 per contratto vincente usando l'ultimo tick Chainlink prima della scadenza rispetto allo strike bloccato.

---

## Shadow order book

`ShadowOrderBook` mantiene due registri così che i fill su carta non corrompano la vista del mercato reale:

| Registro | Contenuto |
|---|---|
| **V_hist** | Lo stato L2 del feed (top 20 livelli per lato). Mai toccato dal bot. |
| **V_cons** (`ConsumptionTracker`) | Liquidità consumata dai fill su carta del bot, con decadimento esponenziale (half-life 0,2 s) per modellare la ri-quotazione. Riconciliato a ogni delta del feed: i rifornimenti lo riducono, le rimozioni lo limitano. |
| **V_eff** | $\max(0, V_{\text{hist}} - V_{\text{cons}})$ — ciò che vedono il router e il walker IOC. |

`get_market_top_of_book()` legge V_hist ed è usato per la probabilità implicita di mercato, il mark-to-market e il filtro di divergenza; `get_sorted_bids()/asks()` restituiscono V_eff. Entrambi sono cachati per tick.

È sottoscritto solo il token YES; i prezzi NO sono $1-p_{\text{YES}}$. Sottoscrivere entrambi mischierebbe i bid NO nel top of book, inchiodando la probabilità implicita vicino a 0,5.

Il callback CLOB fa `await` dell'esecuzione prima di ritornare, quindi un fill è registrato in V_cons prima che il tick successivo possa riattivare la stessa opportunità. Raffiche di snapshot entro 1 s (tempeste di riconnessione) vengono declassate a delta.

---

## Cicli di mercato: 5 minuti, 4 ore e oltre

Polymarket gestisce più serie Up/Down per asset con lunghezze di ciclo diverse. Tutti i confini di ciclo sono multipli della lunghezza del ciclo sull'epoca Unix e lo slug dell'evento codifica l'istante di **inizio** del ciclo:

```
btc-updown-5m-1789675200     (cicli da 300 s)
btc-updown-4h-1789675200     (cicli da 14 400 s)
```

Due impostazioni selezionano la serie e vanno cambiate insieme:

```ini
CYCLE_DURATION_SEC=14400
MARKET_SLUG_TYPE=4h
```

`MarketManager` deriva la prossima scadenza come il prossimo multiplo di `CYCLE_DURATION_SEC` (anticipato di 15 s così che i nuovi token siano sottoscritti prima del confine), lo strike è bloccato dal primo tick Chainlink dopo `scadenza − CYCLE_DURATION_SEC`, e il backtester usa lo stesso allineamento nel replay di un file.

**Cosa scala e cosa non scala con la lunghezza del ciclo.** Il modello di pricing è consapevole dell'orizzonte: $\tau$ entra nel termine diffusivo e nell'intensità di salto integrata nel tempo, quindi la matematica dell'opzione non richiede riparametrizzazioni. I parametri restanti sono grandezze in tempo assoluto, tarate per cicli da 5 minuti, e vanno riviste per cicli più lunghi:

* `EMA_HALFLIFE_SEC` (3 s) e `VOL_ROLLING_WINDOW_SEC` (300 s) — un contratto a 4 ore non ha bisogno di reattività sub-secondo; smoothing più lungo e finestra di vol più ampia riducono il trading sul rumore.
* Il canale Hawkes/OFI (`HAWKES_KAPPA_*`, `HAWKES_BETA`) codifica microstruttura su scala di secondi. Il suo contenuto informativo su un settlement a ore di distanza è piccolo; per cicli lunghi conviene abbassare i kappa (o il peso dell'OFI).
* Le finestre REDUCE / PANIC (45 s / 15 s) riguardano il rumore del tick di settlement, non la lunghezza del ciclo, e sono volutamente lasciate assolute.
* `MM_MAX_INVENTORY`, `KELLY_FRACTION` e i buffer di fee limitano l'esposizione per ciclo; con meno cicli al giorno l'esposizione per ciclo è una quota maggiore del rischio giornaliero.

---

## Backtester

`run_backtest.py` riproduce i file `ticks.parquet` registrati attraverso gli *stessi* `ExecutionEngine`, `MockExecutionClient`, `ShadowOrderBook` e `DivergenceVelocityFilter` usati live — non esiste un modello di simulazione separato. Differenze rispetto al live:

* Le istruzioni taker sono ritardate di 150–300 ms uniformi in **tempo evento** (il replay in sé è istantaneo).
* Il primo tick di ogni ciclo è applicato come snapshot completo del book; i tick successivi sono delta, così il consumo su carta persiste.
* Il pricing è rivalutato solo quando cambiano spot, top of book, probabilità modello o inventario (o sono passati 5 s), il che elimina la maggior parte delle inversioni di Fourier ridondanti.
* Lo strike di ogni ciclo è il primo tick spot del ciclo e le posizioni sono liquidate a ogni rollover e alla fine del file.

```bash
uv run python run_backtest.py --file logs/2026-09-17/merton/live_1789675200/ticks.parquet
uv run python run_backtest.py --start "2026-09-17 10:00:00" --end "2026-09-17 12:00:00"   # cerca in LOG_DIR e data/raw
```

L'output va in `logs/<data>/merton/backtest_<stamp>/` come `signals.csv`, `trades.csv` e `summary.json` (P&L netto, rendimento, max drawdown, win rate, profit factor, vincita/perdita media, curva del capitale e ogni trade realizzato). Il tab Backtester della dashboard web esegue lo stesso codice.

---

## Logging dei dati

```
logs/
└── YYYY-MM-DD/                      # rollover a mezzanotte
    └── merton/
        ├── live_<unix_ts>/
        │   ├── ticks.parquet        # ogni tick CLOB: spot, OFI, vol, top of book, L2 completo (zstd)
        │   ├── signals.csv          # probabilità modello vs mercato a ogni fill / segnale taker
        │   └── trades.csv           # ogni esecuzione su carta e settlement con P&L e capitale
        └── backtest_YYYYMMDD_HHMMSS/
            ├── signals.csv
            ├── trades.csv
            └── summary.json
```

I tick sono bufferizzati (1 000 righe) prima dell'append su Parquet; segnali e trade sono scritti immediatamente. `ticks.parquet` è il formato di input atteso dal backtester.

---

## Installazione e utilizzo

Requisiti: Python 3.13+ e [`uv`](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Wxlddd/polymarketv2 && cd polymarketv2
cp .env.example .env            # modifica TICKER, CYCLE_DURATION_SEC / MARKET_SLUG_TYPE, capitale, …
uv sync                         # crea .venv da uv.lock
```

Avvio dell'orchestratore di paper trading live:

```bash
uv run python main.py                       # dashboard web su http://localhost:8080
uv run python main.py --term                # dashboard terminale Rich
uv run python main.py --strategy merton     # salta il prompt interattivo (merton | legacy_merton)
```

All'avvio l'orchestratore attende il **prossimo** confine di ciclo prima di operare, così il primo ciclo non viene mai preso a metà con strike sconosciuto. I log vanno in `system_run.log` e in `LOG_DIR`.

---

## Esecuzione non presidiata e raccolta dati

Il processo è un singolo loop `asyncio` di lunga durata senza dipendenze interattive (il prompt della strategia è saltato quando stdin non è un TTY), quindi può girare come servizio su qualsiasi host Linux sempre acceso — basta un piccolo VPS. Tutto ciò che osserva viene scritto in `LOG_DIR`; lasciarlo girare per giorni è il modo previsto per costruire un dataset di backtest. Non invia mai ordini reali.

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

Imposta `WEB_SERVER_HOST=0.0.0.0` in `.env` per raggiungere la dashboard dall'esterno del container o dell'host; la dashboard non ha autenticazione, quindi tienila dietro un firewall o un tunnel SSH.

**Windows** — l'orchestratore rinfresca `SetThreadExecutionState` ogni 30 s per tenere sveglia la macchina; `runbot.ps1` è un launcher di comodo.

I dati raccolti stanno in `logs/<data>/merton/live_<ts>/`; passa un `ticks.parquet` (o l'intera directory del run) a `run_backtest.py --file` per riprodurli.

---

## Riferimento configurazione

Tutti i valori sono letti da `.env` da `config/settings.py`. Sono elencati solo i parametri effettivamente usati dal codice.

| Parametro | Default | Usato da |
|---|---|---|
| `TICKER` | `BTC` | Generazione slug |
| `CYCLE_DURATION_SEC` / `MARKET_SLUG_TYPE` | `300` / `5m` | Orologio rollover, slug, timing strike, backtester |
| `STRATEGY_NAME` | `merton` | `StrategyFactory` (`merton`, `legacy_merton`) |
| `INITIAL_CAPITAL` | `10000` | Cash su carta |
| `KELLY_FRACTION` | `0.05` | Sizing regime B |
| `GAS_FEE_USD`, `TAKER_FEE_MULTIPLIER` | `0.03`, `0.072` | Fee simulate |
| `MIN_ORDER_USD` | `1.0` | Quota maker / fill IOC minimi |
| `PANIC_CONCESSION` | `0.15` | Offset del limite dello sweep panic rispetto al book |
| `DEFAULT_SIGMA` | `0.25` | Vol di fallback prima della calibrazione |
| `VOL_ROLLING_WINDOW_SEC` | `300` | Finestra vol realizzata |
| `EMA_HALFLIFE_SEC` | `3.0` | Smoothing della probabilità |
| `DEFAULT_MU_J`, `DEFAULT_SIGMA_J` | `1e-4`, `1.5e-3` | Distribuzione della size dei salti |
| `HAWKES_LAMBDA_0` | `4000` | Intensità di salto baseline (annua) |
| `HAWKES_KAPPA_SELF`, `HAWKES_KAPPA_CROSS`, `HAWKES_BETA` | `3.0`, `1.0`, `5.0` | Eccitazione / decadimento Hawkes (al secondo) |
| `OFI_DRIFT_MULTIPLIER`, `USE_LOCAL_INFORMED_DRIFT` | `-1e-6`, `False` | Drift OFI opzionale |
| `OFI_LOGIT_BETA`, `OFI_NORM_EMA_ALPHA` | `0.5`, `0.1` | Solo `legacy_merton` |
| `MM_RISK_AVERSION` (γ) | `2.5` | Skew del prezzo di riserva e spread |
| `MM_MIN_FEE_BUFFER`, `MM_TOXICITY_BUFFER` | `0.005`, `0.005` | Floor del semi-spread |
| `MM_MAKER_SIZE` | `100` | Clip taker minima |
| `MM_MAX_INVENTORY` | `500` | $Q_{\max}$ (contratti) |
| `MM_UNWIND_THRESHOLD`, `MM_TAKER_EDGE_EPSILON` | `0.01`, `0.015` | Trigger regimi C / B |
| `MM_TICK_SIZE`, `MM_REQUOTE_THRESHOLD` | `0.01`, `0.01` | Arrotondamento quote / isteresi di ri-quotazione |
| `DIVERGENCE_WINDOW_SECONDS`, `VELOCITY_LOOKBACK_SECONDS`, `V_MAX`, `GAMMA` | `60`, `10`, `0.005`, `2.0` | Filtro velocità divergenza |
| `WEB_SERVER_ENABLED`, `WEB_SERVER_HOST`, `WEB_SERVER_PORT`, `UI_BROADCAST_THROTTLE_HZ` | `True`, `localhost`, `8080`, `4` | Dashboard |
| `LOG_DIR` | `logs` | Radice del recorder |

`scratch/calibrate_divergence.py --file <ticks.parquet | signals.csv>` stampa i percentili di velocità dai dati registrati per aiutare a scegliere `V_MAX`.

---

## Script di validazione

Non c'è un test runner; ogni script è autonomo e richiede la radice del repository in `PYTHONPATH`:

```bash
PYTHONPATH=. uv run python tests/verify_phase1.py            # plumbing feed / market manager
PYTHONPATH=. uv run python tests/verify_phase2.py            # riconciliazione shadow book
PYTHONPATH=. uv run python tests/verify_phase3.py            # calibrazione vol + pricer Gil-Pelaez
PYTHONPATH=. uv run python tests/verify_phase4.py            # router → client simulato → settlement, end to end
PYTHONPATH=. uv run python tests/verify_maker_execution.py   # unit test del router dei regimi
PYTHONPATH=. uv run python tests/verify_factory.py
PYTHONPATH=. uv run python tests/verify_web_server.py
PYTHONPATH=. uv run python tests/benchmark_pricing.py
PYTHONPATH=. uv run python tests/verify_live_data_and_pricing.py   # richiede i feed WebSocket live
```

---

## Struttura del repository

```
polymarketv2/
├── config/settings.py
├── main.py                      # orchestratore live / paper
├── run_backtest.py              # CLI di replay storico
├── src/
│   ├── core/                    # interfacce, MarketContext, StrikeManager, strategia base
│   ├── ingestion/               # feed WebSocket, MarketManager
│   ├── strategies/              # Hawkes–Merton, Merton legacy, factory
│   ├── execution/               # ExecutionRouter/Engine, filtro divergenza, shadow book, client simulato
│   ├── backtest/                # BacktestRunner
│   ├── logging/                 # DataRecorder
│   └── ui/                      # web server, dashboard.html, UI terminale Rich
├── tests/                       # script di validazione autonomi
├── scratch/                     # helper di analisi (breakdown trade, calibrazione V_MAX)
├── Dockerfile
├── pyproject.toml / uv.lock
└── .env.example
```

---

## Limiti noti e questioni aperte

* **Nessuna evidenza di edge.** I risultati su carta finora sono stati negativi. L'informazione direzionale del modello viene quasi interamente dall'order flow su scala di secondi; se questo abbia valore predittivo sull'orizzonte a 5 minuti — figuriamoci a 4 ore — non è stato stabilito. Il backtester ora esercita la pipeline reale, quindi lo si può verificare su dati registrati invece di supporlo.
* **Realismo di fee e fill.** La fee taker `p(1−p)·0.072`, il modello di rifiuto stocastico, la stima della coda maker (40 % della riduzione di profondità assunto eseguito) e la half-life di consumo di 0,2 s sono assunzioni, non misure contro il matching reale di Polymarket.
* **Gas del settlement** addebitato nel P&L di settlement riportato ma non dedotto dal cash su carta.
* **Fonte dello strike.** Lo strike è preso dal primo tick Chainlink dopo l'inizio del ciclo; il "price to beat" ufficiale della Gamma API è letto quando disponibile ma usato solo per il logging.
* **Un asset, una serie.** Un ticker e una lunghezza di ciclo per processo; per più serie, più processi.
* **Nessun client di esecuzione live.** `IExecutionClient` è il punto di innesto per uno; oggi nulla firma o invia ordini.
