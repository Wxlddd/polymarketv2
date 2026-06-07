import polars as pl

def main():
    file_path = "logs/2026-06-04/merton/live_1780593895/trades.csv"
    print(f"Reading: {file_path}")
    df = pl.read_csv(file_path)
    
    print("\n=== TOTAL TRADE ROW COUNT ===")
    print("Total rows:", len(df))
    
    print("\n=== TRADE SIDE BREAKDOWN ===")
    for row in df["side"].value_counts().iter_rows(named=True):
        print(f"{row['side']}: {row['count']}")
    
    # Filter for settlements
    settle_df = df.filter(pl.col("side").str.starts_with("SETTLE"))
    print("\n=== SETTLEMENT BREAKDOWN ===")
    print(f"Total Settlements: {len(settle_df)}")
    if len(settle_df) > 0:
        for row in settle_df["side"].value_counts().iter_rows(named=True):
            print(f"{row['side']}: {row['count']}")
        total_settle_pnl = settle_df["pnl"].sum()
        print(f"Total Settlement PnL: ${total_settle_pnl:.2f}")
        
        won = settle_df.filter(pl.col("pnl") > 0)
        lost = settle_df.filter(pl.col("pnl") < 0)
        print(f"Won: {len(won)} | Lost: {len(lost)} | Win Rate: {len(won)/len(settle_df)*100:.2f}%")
        
        print("\n=== SETTLEMENT QUANTITY STATS ===")
        print(f"Mean Qty: {settle_df['qty'].mean():.2f}")
        print(f"Max Qty: {settle_df['qty'].max():.2f}")
        print(f"Min Qty: {settle_df['qty'].min():.2f}")
        
        # Look at the last few settlements
        print("\n=== LAST 10 SETTLEMENT DETAILS ===")
        last_settles = settle_df.tail(10)
        for row in last_settles.iter_rows(named=True):
            print(f"Time: {row['timestamp']} | Side: {row['side']} | Qty: {row['qty']:.2f} | VWAP: {row['vwap']:.4f} | PnL: ${row['pnl']:.2f} | Capital: ${row['capital']:.2f}")
            
    # Non-settlement trades
    non_settle = df.filter(~pl.col("side").str.starts_with("SETTLE"))
    print("\n=== ACTIVE TRADES BREAKDOWN (EXCLUDING SETTLEMENT) ===")
    for row in non_settle["side"].value_counts().iter_rows(named=True):
        print(f"{row['side']}: {row['count']}")
    print(f"Total non-settle PnL: ${non_settle['pnl'].sum():.2f}")
    
    # Check if there are BUY_YES and BUY_NO without matching SELL trades
    print("\n=== NET QUANTITIES BOUGHT VS SOLD ===")
    qty_buy_yes = df.filter(pl.col("side") == "BUY_YES")["qty"].sum()
    qty_sell_yes = df.filter(pl.col("side") == "SELL_YES")["qty"].sum()
    qty_settle_yes = df.filter(pl.col("side") == "SETTLE_YES")["qty"].sum()
    
    qty_buy_no = df.filter(pl.col("side") == "BUY_NO")["qty"].sum()
    qty_sell_no = df.filter(pl.col("side") == "SELL_NO")["qty"].sum()
    qty_settle_no = df.filter(pl.col("side") == "SETTLE_NO")["qty"].sum()
    
    print(f"YES: Bought {qty_buy_yes:.2f} | Sold {qty_sell_yes:.2f} | Settled {qty_settle_yes:.2f} | Unaccounted: {qty_buy_yes - qty_sell_yes - qty_settle_yes:.2f}")
    print(f"NO:  Bought {qty_buy_no:.2f} | Sold {qty_sell_no:.2f} | Settled {qty_settle_no:.2f} | Unaccounted: {qty_buy_no - qty_sell_no - qty_settle_no:.2f}")

if __name__ == "__main__":
    main()
