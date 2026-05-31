import asyncio
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from config.settings import SystemConfig
from src.ingestion.market_manager import MarketManager
from src.core.interfaces import ISpotFeed, IExecutionClient
from src.execution.shadow_book import ShadowOrderBook
from src.core.base_strategy import BaseStrategy
from src.execution.engine import ExecutionEngine

try:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.text import Text
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

def _fmt_price(v: Optional[float], decimals: int = 2) -> str:
    if v is None:
        return "—"
    return f"${v:,.{decimals}f}"

def _fmt_pct(v: Optional[float]) -> str:
    if v is None:
        return "—"
    return f"{v * 100:.2f}%"

def _fmt_pnl(v: Optional[float]) -> str:
    if v is None:
        return "—"
    sign = "+" if v >= 0.0 else ""
    color = "green" if v >= 0.0 else "red"
    return f"[{color}]{sign}${v:,.2f}[/]"

def _fmt_side(side: str) -> str:
    if "YES" in side:
        return f"[bold green]{side}[/]"
    if "NO" in side:
        return f"[bold red]{side}[/]"
    if "HOLD" in side:
        return f"[bold dim]{side}[/]"
    return side

def build_dashboard(
    config: SystemConfig,
    market_manager: MarketManager,
    spot_feed: ISpotFeed,
    shadow_book: ShadowOrderBook,
    strategy: BaseStrategy,
    client: IExecutionClient,
    engine: ExecutionEngine,
    latest_decision: Dict[str, Any]
) -> Layout:
    """Builds the Rich terminal layout combining all system modules."""
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=1)
    )
    
    layout["body"].split_row(
        Layout(name="left", ratio=3),
        Layout(name="right", ratio=2)
    )
    
    layout["left"].split_column(
        Layout(name="portfolio", size=7),
        Layout(name="market_status")
    )
    
    layout["right"].split_column(
        Layout(name="orderbook"),
        Layout(name="execution_engine", size=10)
    )

    # 1. Header Layout
    t_now = time.time()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Calculate cycle countdown remaining time (tau)
    expiry = market_manager.current_expiry or int(t_now - (t_now % 300) + 300)
    tau_sec = max(0.0, expiry - t_now)
    mins, secs = int(tau_sec // 60), int(tau_sec % 60)
    countdown_color = "red" if tau_sec < 45.0 else "yellow" if tau_sec < 120.0 else "cyan"
    
    header_text = Text()
    header_text.append("⚡ POLYMARKET V2 HFT ENGINE ", style="bold cyan")
    header_text.append(f"  |  Time: {now_str}", style="dim")
    header_text.append(f"  |  Cycle Expiry: {expiry}", style="dim")
    header_text.append("  |  Time remaining: ", style="dim")
    header_text.append(f"{mins:02d}:{secs:02d}", style=f"bold {countdown_color}")
    
    layout["header"].update(Panel(header_text, box=box.HORIZONTALS, border_style="bright_black"))

    # 2. Portfolio Panel
    cash = client.cash_balance
    qty_yes = client.get_position_size("YES")
    qty_no = client.get_position_size("NO")
    
    # Get reference price from the real book (never depleted by paper fills)
    top_bid, top_ask = shadow_book.get_market_top_of_book()
    bid_price_yes = top_bid[0] if top_bid else 0.5
    
    # Evaluate position portfolio value (mock client)
    mtm_val = 0.0
    if hasattr(client, "get_portfolio_value"):
        mtm_val = client.get_portfolio_value(bid_price_yes)
    else:
        mtm_val = qty_yes * bid_price_yes + qty_no * (1.0 - bid_price_yes)
        
    equity = cash + mtm_val
    initial_cap = config.arbitrage.INITIAL_CAPITAL
    cum_pnl = equity - initial_cap
    
    pf_table = Table(box=None, show_header=False, padding=(0, 2))
    pf_table.add_column("Col1", style="dim", width=15)
    pf_table.add_column("Val1", width=20)
    pf_table.add_column("Col2", style="dim", width=15)
    pf_table.add_column("Val2", width=20)
    
    pf_table.add_row(
        "Net Equity:", f"[bold white]{_fmt_price(equity)}[/]",
        "Cash Balance:", f"[cyan]{_fmt_price(cash)}[/]"
    )
    pf_table.add_row(
        "Open Positions Value:", f"[white]{_fmt_price(mtm_val)}[/]",
        "Initial Capital:", f"[dim]{_fmt_price(initial_cap)}[/]"
    )
    pf_table.add_row(
        "Cumulative P&L:", _fmt_pnl(cum_pnl),
        "Initial Ticker:", f"[bold yellow]{config.TICKER.upper()}[/]"
    )
    layout["portfolio"].update(Panel(pf_table, title="[bold]Portfolio Summary[/]", border_style="bright_black"))

    # 3. Market Status Panel
    spot = spot_feed.price
    resolved_strike = market_manager.strike_price
    
    # Check what strike is active: resolved one, or wait
    strike_str = _fmt_price(resolved_strike) if resolved_strike is not None else "[yellow]waiting rollover spot...[/]"
    
    # Model option probability
    vol = strategy.vol_calibrator.calculate_volatility(config.merton.DEFAULT_SIGMA)
    
    # Dummy mock MarketContext to run get_probability (zero-assumptions safe)
    dummy_context = MarketContext(
        timestamp=t_now,
        spot_price=spot or 0.0,
        strike_price=resolved_strike if resolved_strike is not None else 0.0,
        tau_seconds=tau_sec,
        volatility=vol,
        ofi=shadow_book.smoothed_ofi,
        bids_l2=[],
        asks_l2=[]
    )
    
    p_yes = strategy.get_probability(dummy_context) if resolved_strike is not None and spot is not None else None
    p_mkt = 0.5 * (top_bid[0] + top_ask[0]) if top_bid and top_ask else None
    
    edge = None
    if p_yes is not None and p_mkt is not None:
        edge = p_yes - p_mkt
        
    mkt_table = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold dim", border_style="bright_black")
    mkt_table.add_column("Active Ticker", justify="left")
    mkt_table.add_column("Spot Price", justify="right", style="gold1")
    mkt_table.add_column("Strike Price (K)", justify="right")
    mkt_table.add_column("YES", justify="right", style="bold blue")
    mkt_table.add_column("Market YES", justify="right", style="magenta")
    mkt_table.add_column("Expected Edge", justify="right")
    mkt_table.add_column("Smooth OFI", justify="right", style="dim")
    
    edge_str = f"{edge * 100:+.2f}%" if edge is not None else "—"
    edge_style = "green" if (edge is not None and edge > 0.005) else "red" if (edge is not None and edge < -0.005) else "white"
    
    mkt_table.add_row(
        config.TICKER.upper(),
        _fmt_price(spot, decimals=2) if spot else "—",
        strike_str,
        _fmt_pct(p_yes) if p_yes is not None else "—",
        _fmt_pct(p_mkt) if p_mkt is not None else "—",
        f"[{edge_style}]{edge_str}[/]",
        f"{shadow_book.smoothed_ofi:+.1f}"
    )
    
    # Active contract details description
    status_text = Text()
    status_text.append(f"  Active Slug: {market_manager.current_slug or '—'}\n", style="dim")
    status_text.append(f"  Active Condition: {market_manager.condition_id or '—'}\n", style="dim")
    status_text.append(f"  Calibrated Volatility: {vol:.2%}  |  Smoothed OFI: {shadow_book.smoothed_ofi:+.1f}\n", style="dim")
    
    body_layout = Layout()
    body_layout.split_column(
        Layout(mkt_table, size=4),
        Layout(Panel(status_text, box=box.MINIMAL, border_style="dim"))
    )
    
    layout["market_status"].update(Panel(body_layout, title="[bold]Market Discovery & Options Pricing[/]", border_style="bright_black"))

    # 4. Right: Order Book L2 Panel
    ob_lines = Text()
    ob_lines.append(f"  Shadow Order Book (YES contract)\n", style="bold dim")
    
    asks_l2 = shadow_book.get_sorted_asks()[:5]
    bids_l2 = shadow_book.get_sorted_bids()[:5]
    
    # Format Asks (red)
    for p, q in reversed(asks_l2):
        ob_lines.append(f"    [red]{p:.3f}[/red]   {q:>8.1f}\n")
        
    # Spread line
    spread = asks_l2[0][0] - bids_l2[0][0] if bids_l2 and asks_l2 else 0.0
    ob_lines.append(f"    ────── spread {spread:.4f} ──────\n", style="dim")
    
    # Format Bids (green)
    for p, q in bids_l2:
        ob_lines.append(f"    [green]{p:.3f}[/green]   {q:>8.1f}\n")
        
    layout["orderbook"].update(Panel(ob_lines, title="[bold]Order Book YES[/]", border_style="bright_black"))

    # 5. Right: Sizing & Execution Panel
    ex_table = Table(box=None, show_header=False, padding=(0, 2))
    ex_table.add_column("Col1", style="dim", width=18)
    ex_table.add_column("Val1", width=18)
    
    dec_side = latest_decision.get("side", "HOLD")
    dec_size = latest_decision.get("size", 0.0)
    dec_vwap = latest_decision.get("vwap", 0.0)
    dec_reason = latest_decision.get("reason", "NO_EV_OR_SIZE_OPPORTUNITY")
    
    ex_table.add_row("Latest Decision:", _fmt_side(dec_side))
    ex_table.add_row("Execution Quantity:", f"{dec_size:.2f} contracts")
    ex_table.add_row("Target VWAP price:", f"{dec_vwap:.4f}")
    ex_table.add_row("Execution Msg/Reason:", f"[dim]{dec_reason}[/]")
    ex_table.add_row("YES Position Quantity:", f"[green]{qty_yes:.2f}[/]")
    ex_table.add_row("NO Position Quantity:", f"[red]{qty_no:.2f}[/]")
    
    layout["execution_engine"].update(Panel(ex_table, title="[bold]Risk & Sizing Engine[/]", border_style="bright_black"))

    # 6. Footer Layout
    layout["footer"].update(Text("   [q] Exit Dashboard  |  HFT Update Loop: 500ms  |  Modularity level: ultra", style="dim"))

    return layout

async def run_terminal_dashboard(
    config: SystemConfig,
    market_manager: MarketManager,
    spot_feed: ISpotFeed,
    shadow_book: ShadowOrderBook,
    strategy: BaseStrategy,
    client: IExecutionClient,
    engine: ExecutionEngine,
    latest_decision_ref: List[Dict[str, Any]],  # mutable list reference to share engine updates
    stop_event: asyncio.Event
) -> None:
    """Async loop printing and rendering rich dashboard layout."""
    if not RICH_AVAILABLE:
        print("[Dashboard] Rich library not installed. Terminal dashboard disabled.")
        return
        
    console = Console()
    
    # Hide cursor and clean screen
    console.clear()
    
    with Live(console=console, refresh_per_second=2, screen=True, vertical_overflow="visible") as live:
        while not stop_event.is_set():
            try:
                latest_decision = latest_decision_ref[0] if latest_decision_ref else {"side": "HOLD", "size": 0.0}
                
                layout = build_dashboard(
                    config=config,
                    market_manager=market_manager,
                    spot_feed=spot_feed,
                    shadow_book=shadow_book,
                    strategy=strategy,
                    client=client,
                    engine=engine,
                    latest_decision=latest_decision
                )
                live.update(layout)
            except Exception as e:
                # Do not crash HFT execution thread
                pass
            await asyncio.sleep(0.5)
