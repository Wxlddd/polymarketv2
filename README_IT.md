# polymarketv2

[English](README.md) | **Italiano**

Bot di paper trading e backtester per i mercati binari crypto **Up/Down** di Polymarket (`btc-updown-5m-…`, `btc-updown-4h-…`).
Prezza il contratto YES con un jump-diffusion Hawkes–Merton guidato dall'order-flow imbalance del CLOB e lo quota con un
router maker/taker stile Avellaneda–Stoikov. Si collega ai feed WebSocket reali (Chainlink e CLOB) ma **non invia mai un
ordine reale**: l'esecuzione è un exchange simulato e ogni tick, segnale e fill viene loggato per il backtest.

## Stato (17-09-2026)

**La strategia non ha fatto soldi.** Cosa dicono i dati registrati:

| Evidenza | Risultato |
|---|---|
| 21 sessioni live paper, 26 mag – 2 giu 2026 (codice pre PR #3) | 177 settlement: 86 vinti / 91 persi. Coin flip. |
| Qty di settlement vs qty per trade | Mediana 800–27 000 contratti a settlement contro 100–300 per trade: l'inventario andava in scadenza ogni ciclo. |
| P&L round-trip (escluso settlement) nelle 12 sessioni con capitale sano | Negativo in 8 su 12. |
| Fill taker, sessione da 388k tick | Il mid si muove **contro** il fill di 0.07 a 5 s e 0.03 a 30 s (adverse selection). |
| 3 sessioni finite a $125k–$2.9M | Il vecchio paper client non aveva cap di cassa; settlement da 1–7 M contratti su $10k. Non reali. |
| Backtest del 2 giu a +222 % | Bug del vecchio runner (liquidità infinita, 972 exit vincenti). Da ignorare. |
| `main` attuale, file reale da 128k tick (6 cicli) | −3.6 %, max drawdown 9.7 %, inventario cappato a 500, il panic sweep appiattisce prima della scadenza. |

La PR #3 ha sistemato i bug meccanici (crash all'avvio live, backtester rotto, amplificazione dell'orizzonte Hawkes, panic
sweep mai riemesso, nessun cap di cassa, ciclo 5 minuti hardcoded).

### Perché il bot è solo maker (19-09-2026)

Replay di tutti i 69 cicli completi registrati attraverso il pricer, campionando modello e book ogni 5 s:

| Test | Risultato |
|---|---|
| Brier score, modello vs mid di mercato | 0.182 vs 0.220 — il modello è meglio calibrato del mercato |
| Pendenza di (esito − mid) su (modello − mid) | 0.94, IC 95% [0.73, 1.10] — la divergenza predice l'errore del mercato |
| Taker buy-and-hold, 100 contratti, eseguito **sul tick del segnale** | +11.5 per entrata, IC [+7.8, +15.8] |
| Idem, eseguito **un tick dopo** | **−6.8 per entrata**, IC [−12.0, −1.5] |
| Idem, scartando livelli non aggiornati da 10 s | invariato (+11.5) — non è liquidità stale |

Il modello reagisce al tick Chainlink una frazione di secondo prima del book, e il book recupera entro un tick.
L'informazione è reale; non è eseguibile con un round trip d'ordine di 150–300 ms. È la stessa adverse selection che
mostravano i fill taker live (il mid si muove di 0.07 contro il fill entro 5 s).

Quindi il crossing direzionale è spento (`TAKER_ENABLED=False`) e gli unici ordini taker rimasti sono gli sweep PANIC che
appiattiscono l'inventario prima del settlement. Un modello veloce che vede il movimento per primo resta utile come
segnale **difensivo** — ritirare o riprezzare una quota prima che venga raccolta — ed è di questo che tratta il lavoro
rimanente.

La domanda aperta ora è il **modello di fill**: tutto il P&L maker di questo repo viene da un simulatore che assume che il
40% di ogni riduzione di profondità al tuo livello sia stato eseguito, più un decadimento della coda del 2% per tick.
Niente è mai stato calibrato su esecuzioni reali. `prints.parquet` (i trade print dell'exchange, registrati da questo
branch) è ciò che rende possibile quella calibrazione: per questo adesso raccogliere dati conta più che tarare parametri.

## Avvio rapido

Python 3.13+ e [`uv`](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Wxlddd/polymarketv2 && cd polymarketv2
cp .env.example .env
uv sync
uv run python main.py --strategy merton        # dashboard web su http://localhost:8080
uv run python main.py --strategy merton --term # dashboard Rich nel terminale
```

Il trading parte al **prossimo** confine di ciclo, mai a metà. Ogni valore in `.env` ha un default in
`config/settings.py`; un `.env` vecchio funziona lo stesso.

## Girare una notte / raccogliere dati

Il processo è un singolo loop asyncio senza dipendenze interattive una volta passato `--strategy`.

- **Fisso Windows**: lancialo e basta. L'orchestrator chiama `SetThreadExecutionState` ogni 30 s, la macchina non va in sleep. Schermo spento va bene.
- **VPS Linux** (1 vCPU / 1 GB bastano):

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

`WEB_SERVER_HOST=0.0.0.0` per raggiungere la dashboard da remoto (nessuna auth: tienila dietro SSH).

Output per sessione:

```
logs/YYYY-MM-DD/merton/live_<unix_ts>/
├── ticks.parquet   # ogni update CLOB (snapshot/delta + flag is_snapshot) con spot, OFI, vol, top of book riconciliato
├── prints.parquet  # ogni trade print YES (last_trade_price): prezzo, size, lato aggressore, timestamp exchange
├── signals.csv     # probabilità modello vs mercato a ogni segnale taker / fill maker
└── trades.csv      # ogni fill paper e settlement con P&L e capitale
```

`logs/` è in `.gitignore`. `git pull` non lo tocca mai.

## Backtest

Riproduce `ticks.parquet` attraverso lo **stesso** router, paper client, shadow book e divergence filter usati live.

```bash
uv run python run_backtest.py --file logs/2026-05-28/merton/live_1779986166/ticks.parquet
uv run python run_backtest.py --start "2026-05-27 10:00" --end "2026-05-27 12:00"   # cerca in logs/ e data/raw
BACKTEST_SEED=42 uv run python run_backtest.py --file ...                              # deterministico
```

Scrive `signals.csv`, `trades.csv`, `summary.json` in `logs/<oggi>/merton/backtest_<stamp>/`.
Circa 700 tick/s (128k tick ≈ 3 min).

**Sweep di parametri** (parallelo, un processo per config × file, override via env var, risultati in `scratch/sweep_results.csv`):

```bash
uv run python scratch/sweep_backtest.py --set train --workers 6
uv run python scratch/sweep_backtest.py --set test --configs baseline,maker_only
```

Modifica `CONFIGS` / `FILES` nello script. Tieni un set `test` separato: tutto ciò che viene scelto su `train` è biased per costruzione.

**Analisi log**: `scratch/analyze_trades.py` spacca un `trades.csv` per side, win rate dei settlement e quantità non riconciliata.

## Come trada

```
Chainlink spot WS ─┐
CLOB L2 WS (YES) ──┴─► ShadowOrderBook (stato feed + consumo paper proprio)
                          │
                          ├─► MertonStrategy.get_probability()  p_yes  (Hawkes–Merton, Gil-Pelaez GL64)
                          ├─► EMA temporale (EMA_HALFLIFE_SEC, scende a τ/10 vicino a scadenza)
                          ├─► DivergenceVelocityFilter → scala size 0..1 (0 = ritira tutte le quote)
                          ├─► MockExecutionClient.process_market_data()  → fill delle quote maker
                          └─► ExecutionRouter.evaluate_regimes() → NEW / REPLACE / CANCEL
```

| Regime | Quando | Fa |
|---|---|---|
| A maker | default | bid/ask a `P_res ± δ`, `P_res = p̂ − γ·q/Q_max·σ²`, `δ = fee + toxicity + ½γσ²τ` |
| B taker | bid target > best ask + ε (o speculare) | cancella le quote, IOC con Kelly frazionario, minimo `MM_MAKER_SIZE`, cap su cassa e `MM_MAX_INVENTORY` |
| C unwind | \|q\| > Q_max oppure \|p̂ − mid\| < `MM_UNWIND_THRESHOLD` | quota solo il lato che riduce \|q\| |
| REDUCE | τ ≤ 45 s | solo riduzione |
| PANIC | τ ≤ 15 s, o τ ≤ 45 s con spread > 0.10 | IOC dell'intera posizione a `best bid − PANIC_CONCESSION` (o ask +), riemesso ogni tick fino a flat, lock fino al rollover |

Il settlement paga 1.0 per contratto vincente all'ultimo tick Chainlink prima della scadenza contro lo strike bloccato a inizio ciclo.

## Parametri che contano davvero

Tutti in `.env`, letti da `config/settings.py`.

| Parametro | Default | Effetto |
|---|---|---|
| `CYCLE_DURATION_SEC` / `MARKET_SLUG_TYPE` | `300` / `5m` | Serie. `14400` / `4h` per il mercato a 4 ore. Cambiali insieme. |
| `INITIAL_CAPITAL` | `10000` | Cassa paper. |
| `TAKER_ENABLED` | `False` | Crossing direzionale. Spento di default (vedi Stato); il PANIC incrocia comunque per appiattire. |
| `MM_TAKER_EDGE_EPSILON` | `0.015` | Edge oltre lo spread prima di attraversare. |
| `KELLY_FRACTION` | `0.05` | Sizing taker. |
| `MM_MAX_INVENTORY` | `500` | Cap duro su \|YES − NO\|. |
| `MM_RISK_AVERSION` | `2.5` | γ: skew dell'inventario e spread. |
| `MM_MIN_FEE_BUFFER`, `MM_TOXICITY_BUFFER` | `0.005`, `0.005` | Floor del mezzo spread. |
| `MM_UNWIND_THRESHOLD` | `0.01` | Take profit quando modello e mercato si riallineano. |
| `PANIC_CONCESSION` | `0.15` | Quanto in profondità nel book va lo sweep finale. |
| `EMA_HALFLIFE_SEC` | `3` | Smoothing della probabilità. Tarato per 5m; troppo veloce per 4h. |
| `VOL_ROLLING_WINDOW_SEC` | `300` | Finestra vol realizzata. Tarata per 5m. |
| `HAWKES_KAPPA_SELF`, `HAWKES_KAPPA_CROSS`, `HAWKES_BETA` | `3`, `1`, `5` | OFI → intensità dei salti. Kappa a `0` = diffusione pura. |
| `STRATEGY_NAME` | `merton` | oppure `legacy_merton` (salti Poisson + shift logit OFI). |
| `V_MAX`, `GAMMA` | `0.005`, `2` | Divergence filter. `scratch/calibrate_divergence.py` stampa i percentili da un log. |

Definiti ma non usati da nessun codice: `MIN_EXPECTED_VALUE`, `PIN_RISK_SECONDS`, `DESYNC_Z_SCORE`, `POF_*`, `COOLDOWN_PERIOD_SEC`, `MAX_POSITION_SIZE_USD`.

## Script di verifica

Nessun test runner; ogni script è standalone.

```bash
PYTHONPATH=. uv run python tests/verify_phase4.py           # router → paper client → settlement
PYTHONPATH=. uv run python tests/verify_maker_execution.py  # 8 unit test sul router
PYTHONPATH=. uv run python tests/verify_phase3.py           # calibrazione vol + pricer
```

Anche `verify_phase1/2.py`, `verify_factory.py`, `verify_web_server.py`, `benchmark_pricing.py`.

## Struttura

```
main.py                  LiveOrchestrator (feed, pricing, routing, settlement, UI)
run_backtest.py          CLI di replay
config/settings.py       dataclass di config frozen
src/strategies/          merton_strategy.py (Hawkes–Merton), legacy_merton_strategy.py, factory.py
src/execution/           engine.py (router), clients.py (exchange paper), shadow_book.py, divergence_filter.py
src/ingestion/           live_feeds.py (WS), market_manager.py (slug, rollover, Gamma API)
src/backtest/runner.py   BacktestRunner
src/logging/recorder.py  recorder parquet + csv
src/ui/                  web_server.py + dashboard.html, dashboard.py (Rich)
scratch/                 analyze_trades.py, sweep_backtest.py, calibrate_divergence.py
tests/                   verify_*.py
```

## Buchi noti

- Nessuna evidenza di edge. Il segnale direzionale è order flow su scala di secondi; il suo valore a 5 min non è provato, a 4 h è dubbio.
- Il modello di fill è assunto, non misurato: fee taker `0.072·p(1−p)`, rifiuto logistico, coda maker al 40 % della riduzione di depth, half-life del consumo 0.2 s.
- L'IOC taker a `best_ask` dopo i 150–300 ms di latenza simulata spesso non trova liquidità (warning `IOC fill too small`).
- Lo strike viene dal primo tick Chainlink dopo l'inizio ciclo; il "price to beat" ufficiale di Polymarket è solo loggato.
- La scala del divergence filter non è loggata, quindi il suo effetto non si può verificare da `signals.csv`.
- Un asset e una durata di ciclo per processo.
