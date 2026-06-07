        # Evaluate Trade Sizing and Execution
        
        # 1. Process market data through client to detect limit order fills
        context_state = {"timestamp": t_now, "strike_price": active_strike}
        fills = await self.client.process_market_data(context.bids_l2, context.asks_l2, context_state)
        
        for result in fills:
            if result.get("success"):
                instr = result.get("instruction")
                if instr:
                    self.total_trades += 1
                    self.hft_metrics["volume_executed"] += result["qty"]
                    
                    self.hft_metrics["pending_mtm_trades"].append({
                        "timestamp": t_now,
                        "side": result["side"],
                        "exec_price": result["price"],
                        "mtm_1s": None,
                        "mtm_5s": None,
                        "mtm_15s": None
                    })
                    
                    p_mkt_fill = self._get_market_implied_price(p_yes)
                    status_str = "MAKER_FILL_BUY" if "BUY" in result["side"] else "MAKER_FILL_SELL"
                    self.recorder.record_signal(
                        timestamp=t_now,
                        spot_price=spot,
                        strike=active_strike,
                        model_prob=p_yes,
                        implied_prob=p_mkt_fill if p_mkt_fill is not None else p_yes,
                        kelly_size=result["qty"],
                        status=status_str
                    )
                    
                    self.log_message("info", f"[MAKER FILL] {result['side']} filled at {result['price']:.4f} | Size: {result['qty']:.2f}")
                    if self.web_server:
                        self.web_server.broadcast_message({
                            "type": "trade_signal",
                            "signal": {"side": result["side"], "size": result["qty"], "vwap": result["price"], "ev": 0.0}
                        })
        
        # 2. Generate and process new quoting instructions
        instructions = self.engine.evaluate_and_route(context)
        
        # Format diagnostic decision for the dashboard
        bid_p = self.engine.execution_router.active_bid_price
        ask_p = self.engine.execution_router.active_ask_price
        decision = {
            "side": "QUOTING",
            "size": self.engine.execution_router.active_bid_qty,
            "vwap": bid_p if bid_p > 0.0 else ask_p,
            "reason": f"Bid: {bid_p:.2f} Ask: {ask_p:.2f}",
            "expected_slippage_bps": 0.0,
            "ev": 0.0
        }
        self.latest_decision_ref[0] = decision
        
        for instr in instructions:
            if instr.action == "NEW":
                self.hft_metrics["total_orders_sent"] += 1
            elif instr.action == "CANCEL":
                self.hft_metrics["total_orders_cancelled"] += 1
            elif instr.action == "REPLACE":
                self.hft_metrics["total_orders_sent"] += 1
                self.hft_metrics["total_orders_cancelled"] += 1
                
            res = await self.client.process_instruction(instr, context_state)
            
            # Taker executions are returned synchronously from process_instruction
            if instr.action == "NEW" and instr.regime in ("B", "PANIC") and res.get("success"):
                self.total_trades += 1
                p_mkt_exec = self._get_market_implied_price(p_yes)
                status_str = f"TAKER_{instr.side}" if instr.regime == "B" else f"PANIC_{instr.side}"
                self.recorder.record_signal(
                    timestamp=t_now,
                    spot_price=spot,
                    strike=active_strike,
                    model_prob=p_yes,
                    implied_prob=p_mkt_exec if p_mkt_exec is not None else p_yes,
                    kelly_size=res["qty"],
                    status=status_str
                )
                type_str = 'TAKER' if instr.regime == 'B' else 'PANIC'
                self.log_message("info", f"[{type_str} ORDER] Executed {instr.side} | Size: {res['qty']:.2f} | Price: {res['price']:.4f}")
                
                self.hft_metrics["total_taker_edge"] += abs(p_yes - res["price"] if p_yes is not None else 0.0)
                self.hft_metrics["total_taker_trades"] += 1
                
                self.hft_metrics["pending_mtm_trades"].append({
                    "timestamp": t_now,
                    "side": res["side"],
                    "exec_price": res["price"],
                    "mtm_1s": None,
                    "mtm_5s": None,
                    "mtm_15s": None
                })
