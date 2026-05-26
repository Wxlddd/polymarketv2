# Polymarket V2: High-Frequency Options Pricing & Execution Engine

Polymarket V2 è un'architettura ultra-modulare ad alte prestazioni (HFT) progettata per il pricing quantitativo e l'esecuzione automatizzata sui mercati binari a 5 minuti di Polymarket (Up/Down). Il sistema implementa il modello a salti di Merton (MJD) risolto tramite inversione di Fourier (Gil-Pelaez) ed esegue ordini simulati tramite un motore di sizing frazionario Kelly con rigorosi filtri di rischio.

---

## Indice delle Funzionalità

- **Data Ingestion Infallibile**: WebSocket nativo per il feed Spot medianizzato (oracoli Chainlink di Polymarket) e feed L2 CLOB per il book YES/NO con riconnessione automatica.
- **Scoperta Dinamica dei Mercati (MarketManager)**: Risoluzione deterministica degli slug del ciclo a 5 minuti e recupero asincrono di Token ID, Condition ID e Strike Price tramiteGamma API.
- **Pricing Merton con Variate di Controllo**: Calcolo della probabilità di esercizio dell'opzione binaria integrando la funzione caratteristica Merton tramite Gil-Pelaez e utilizzando Black-Scholes come variata di controllo per la massima stabilità ad alta frequenza.
- **Calibrazione Volatilità e Drift**: Monitoraggio continuo della volatilità realizzata (con filtri di rendimento ed eliminazione degli spike da salti) e stima del drift istantaneo tramite OFI (Order Flow Imbalance) lisciato.
- **Motore di Esecuzione e Shadow Book**: Tracciamento dello stato di profondità reale vs. virtuale (post-trade shadow depth) con arrotondamento dei prezzi a 6 decimali.
- **Bloomberg Terminal Dashboard (Web Server)**: Server web `aiohttp` integrato con WebSocket a trasmissione parzializzata (cap a 4Hz) per azzerare il sovraccarico GPU del browser, mantenendo l'engine HFT sottostante a frequenza illimitata.
- **Backtester Storico Temporale**: Simulazione event-driven che riproduce la pipeline di trading reale analizzando più file Parquet/CSV concatenati e filtrati per una precisa finestra temporale.

---

## Struttura del Progetto

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
│   │   └── strike_manager.py# Gestore dello Strike Price K (con fallback su primo spot tick)
│   ├── ingestion/
│   │   ├── live_feeds.py    # WebSocket Chainlink Spot Feed & CLOB L2 Orderbook Feed
│   │   └── market_manager.py# Dynamic discovery dei mercati Gamma API & Rollover
│   ├── strategies/
│   │   └── merton_strategy.py# Caratteristica Merton, Gil-Pelaez e calibrazione Vol/OFI
│   ├── execution/
│   │   ├── shadow_book.py   # Riconciliazione L2 e proxy virtuale del book post-trade
│   │   ├── engine.py        # walks del book L2, Kelly sizing e filtri di rischio (Pin, Desync, PoF)
│   │   └── clients.py       # Mock/Simulated Execution Client per paper trading e backtest
│   ├── logging/
│   │   └── recorder.py      # Scrittura Parquet (ticks) e CSV (segnali/esecuzioni) ad alte prestazioni
│   └── ui/
│       ├── dashboard.py     # Terminal Dashboard Rich CLI (locale)
│       ├── web_server.py    # Integrated HTTP/WS Server (Bloomberg Web Dashboard)
│       └── dashboard.html   # Bloomberg Stark Terminal UI (Flat black, no shadows, 2D canvases)
├── main.py                  # Entrypoint dell'Orchestratore Live/Paper Trading
├── run_backtest.py          # Script CLI per avviare il Backtester Storico Temporale
├── pyproject.toml           # Gestione dipendenze e configurazione del progetto Python (uv)
└── verify_*.py              # Suite di test e script di convalida delle fasi di sviluppo
```

---

## Installazione e Avvio

Il progetto utilizza `uv` per la gestione ultra-rapida dei pacchetti Python. Assicurati che `uv` sia installato sul sistema.

1. **Configurazione ambiente**:
   Crea il file `.env` a partire dal template ed inserisci le variabili necessarie (es. URL WebSocket, ID dei token di default, capitale iniziale):
   ```bash
   cp .env.example .env
   ```

2. **Avviare il Live Trading / Paper Trading**:
   È possibile avviare il sistema in due modalità:
   
   - **Modalità CLI Terminale (Rich)**:
     ```bash
     uv run python main.py
     ```
     Mostra una dashboard interattiva Rich CLI direttamente sul terminale per monitorare il book locale e il portafoglio.
     
   - **Modalità Web Server (Bloomberg UI)**:
     ```bash
     uv run python main.py --no-term
     ```
     Disabilita la CLI a schermo intero e stampa a terminale log puliti ad intervalli di 2 secondi, avviando contemporaneamente la dashboard Bloomberg all'indirizzo **`http://localhost:8080`**.
     
     La dashboard web include:
     * Ticker correntemente attivo, tempo rimanente alla scadenza del ciclo e status dell'oracolo.
     * KPI finanziari (Equity Totale, Saldo Cash, Valore Posizioni, P&L Cumulativo, Trade Totali).
     * Tabella con metriche core e spread bid-ask reali del token YES.
     * Grafici canvas 2D ultra-leggeri per l'andamento Spot vs. Strike e Merton P_YES vs. Market Implied P_YES.
     * Pannello **Historical Backtester** con finestre temporali d'avvio per simulazioni storiche.
     * Finestra di log interattiva `REAL-TIME EXECUTION LOGS` alimentata dal server.

---

## Eseguire un Backtest Storico

Lo script `run_backtest.py` permette di simulare la strategia Merton su dati storici registrati. È possibile eseguirlo in due modi:

1. **Finestra Temporale (Consigliato)**:
   Specifica un intervallo temporale utilizzando le date nel formato `YYYY-MM-DD HH:MM:SS` (o timestamp numerici). Lo script scansionerà la directory `data/raw`, caricherà tutti i file Parquet/CSV sovrapposti, li allineerà allo schema V2 ed eseguirà la simulazione:
   ```bash
   uv run python run_backtest.py --start "2026-05-23 20:51:00" --end "2026-05-23 21:51:00"
   ```

2. **File Singolo**:
   Esegui la simulazione puntando ad un singolo file Parquet di tick storici:
   ```bash
   uv run python run_backtest.py --file "c:/percorso/del/tuo/file.parquet"
   ```

A completamento, verranno stampati a terminale i KPI della simulazione:
* Patrimonio iniziale vs finale.
* Profitto/Perdita Netto in USD.
* Tasso di Ritorno (Return Rate %).
* Drawdown Massimo (Max Drawdown %).
* Numero totale di trade effettuati.
* Percorso dei log storici salvati in formato Parquet/CSV.

---

## Verifica e Test di Qualità

Per garantire la massima correttezza del codice ad ogni modifica, puoi eseguire gli script di verifica dedicati:

- Convalida Modulo Dati e Logging: `uv run python verify_phase1.py`
- Convalida Slug e Strike Resolution: `uv run python verify_phase2.py`
- Convalida Modello Matematico Merton: `uv run python verify_phase3.py`
- Convalida Esecuzioni e Shadow Book: `uv run python verify_phase4.py`
- Convalida Connessioni e Endpoints Web Server: `uv run python verify_web_server.py`
- Convalida Calcolo delle Probabilità in Tempo Reale: `uv run python verify_live_data_and_pricing.py`
