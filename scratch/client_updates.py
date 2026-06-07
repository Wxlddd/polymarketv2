    async def process_instruction(self, instruction: OrderInstruction, context_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a routing instruction (NEW, CANCEL, REPLACE).
        For Taker (Regime B/PANIC), executes immediately (IOC).
        For Maker (Regime A/C), places a limit order resting in the mock client book.
        """
        side_key = "bid" if "BUY" in instruction.side else "ask"
        
        if instruction.action == "CANCEL":
            if self.active_maker_orders[side_key] and self.active_maker_orders[side_key]["instruction"].order_id == instruction.order_id:
                self.active_maker_orders[side_key] = None
            return {"success": True, "action": "CANCEL", "order_id": instruction.order_id}

        # For NEW or REPLACE
        if instruction.regime in ("B", "PANIC"):
            # Taker execution (IOC)
            ev = 0.0 # Taker ev should ideally be passed in context_state or calculated
            return await self._execute_trade_internal(
                side=instruction.side,
                qty=instruction.qty,
                price=instruction.price,
                ev=ev,
                expected_slippage_bps=0.0,
                context_state=context_state,
                is_maker=False
            )
        else:
            # Maker execution (RESTING)
            cur_depth = 0.0
            if side_key == "bid":
                for p, q in self.shadow_book.get_sorted_bids():
                    if p >= instruction.price:
                        cur_depth += q
                    else:
                        break
            else:
                for p, q in self.shadow_book.get_sorted_asks():
                    if p <= instruction.price:
                        cur_depth += q
                    else:
                        break

            self.active_maker_orders[side_key] = {
                "instruction": instruction,
                "queue_ahead": cur_depth,
                "prev_depth": cur_depth,
                "touched": False,
                "recorded_touch": False
            }
            return {"success": True, "action": instruction.action, "order_id": instruction.order_id}

    async def process_market_data(self, bids_l2: List[Tuple[float, float]], asks_l2: List[Tuple[float, float]], context_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Updates the L2 book and evaluates resting Maker limit orders for queue depth and simulated fills.
        Returns a list of fill results (if any).
        """
        fills = []
        best_bid_p = bids_l2[0][0] if bids_l2 else None
        best_ask_p = asks_l2[0][0] if asks_l2 else None

        # 1. Update Queue Depth for BID
        if self.active_maker_orders["bid"]:
            order = self.active_maker_orders["bid"]
            instr = order["instruction"]
            active_bid_p = instr.price
            
            cur_depth = 0.0
            for p, q in bids_l2:
                if p >= active_bid_p:
                    cur_depth += q
                else:
                    break
                    
            if best_ask_p is not None:
                if best_ask_p < active_bid_p:
                    order["queue_ahead"] = 0.0
                    order["touched"] = True
                elif best_ask_p == active_bid_p:
                    order["touched"] = True
                    delta_depth = order["prev_depth"] - cur_depth
                    if delta_depth > 0:
                        alpha = 0.40
                        v_traded = alpha * delta_depth
                        c_cancels = (1.0 - alpha) * delta_depth
                        ratio = order["queue_ahead"] / order["prev_depth"] if order["prev_depth"] > 0 else 0.0
                        v_cancelled_ahead = c_cancels * ratio
                        order["queue_ahead"] = max(0.0, order["queue_ahead"] - v_traded - v_cancelled_ahead)
                    
                    stoch_decay = 0.02 * order["queue_ahead"]
                    order["queue_ahead"] = max(0.0, order["queue_ahead"] - stoch_decay)
                    order["prev_depth"] = cur_depth
                    
        # 2. Update Queue Depth for ASK
        if self.active_maker_orders["ask"]:
            order = self.active_maker_orders["ask"]
            instr = order["instruction"]
            active_ask_p = instr.price
            
            cur_depth = 0.0
            for p, q in asks_l2:
                if p <= active_ask_p:
                    cur_depth += q
                else:
                    break
                    
            if best_bid_p is not None:
                if best_bid_p > active_ask_p:
                    order["queue_ahead"] = 0.0
                    order["touched"] = True
                elif best_bid_p == active_ask_p:
                    order["touched"] = True
                    delta_depth = order["prev_depth"] - cur_depth
                    if delta_depth > 0:
                        alpha = 0.40
                        v_traded = alpha * delta_depth
                        c_cancels = (1.0 - alpha) * delta_depth
                        ratio = order["queue_ahead"] / order["prev_depth"] if order["prev_depth"] > 0 else 0.0
                        v_cancelled_ahead = c_cancels * ratio
                        order["queue_ahead"] = max(0.0, order["queue_ahead"] - v_traded - v_cancelled_ahead)
                    
                    stoch_decay = 0.02 * order["queue_ahead"]
                    order["queue_ahead"] = max(0.0, order["queue_ahead"] - stoch_decay)
                    order["prev_depth"] = cur_depth

        # 3. Process Fills
        # Buy Fill
        if self.active_maker_orders["bid"]:
            order = self.active_maker_orders["bid"]
            instr = order["instruction"]
            if best_ask_p is not None:
                is_crossed = best_ask_p < instr.price
                is_touched_and_front = best_ask_p == instr.price and order["queue_ahead"] <= 0.0
                if is_crossed or is_touched_and_front:
                    # Execute fill
                    fill_result = await self._execute_trade_internal(
                        side=instr.side,
                        qty=instr.qty,
                        price=instr.price,
                        ev=0.0,
                        expected_slippage_bps=0.0,
                        context_state=context_state,
                        is_maker=True
                    )
                    fill_result["instruction"] = instr
                    fills.append(fill_result)
                    self.active_maker_orders["bid"] = None

        # Sell Fill
        if self.active_maker_orders["ask"]:
            order = self.active_maker_orders["ask"]
            instr = order["instruction"]
            if best_bid_p is not None:
                is_crossed = best_bid_p > instr.price
                is_touched_and_front = best_bid_p == instr.price and order["queue_ahead"] <= 0.0
                if is_crossed or is_touched_and_front:
                    # We might need to split into SELL_YES and BUY_NO if yes_shares are insufficient.
                    # This logic should be here.
                    yes_shares = self.get_position_size("YES")
                    exec_qty = instr.qty
                    exec_price = instr.price
                    
                    if yes_shares >= exec_qty:
                        fill_result = await self._execute_trade_internal(
                            side="SELL_YES",
                            qty=exec_qty,
                            price=exec_price,
                            ev=0.0,
                            expected_slippage_bps=0.0,
                            context_state=context_state,
                            is_maker=True
                        )
                        fill_result["instruction"] = instr
                        fills.append(fill_result)
                    else:
                        if yes_shares > 0.0:
                            fill1 = await self._execute_trade_internal(
                                side="SELL_YES",
                                qty=yes_shares,
                                price=exec_price,
                                ev=0.0,
                                expected_slippage_bps=0.0,
                                context_state=context_state,
                                is_maker=True
                            )
                            fill1["instruction"] = instr
                            fills.append(fill1)
                            
                        rem_q = exec_qty - yes_shares
                        no_price = 1.0 - exec_price
                        fill2 = await self._execute_trade_internal(
                            side="BUY_NO",
                            qty=rem_q,
                            price=no_price,
                            ev=0.0,
                            expected_slippage_bps=0.0,
                            context_state=context_state,
                            is_maker=True
                        )
                        fill2["instruction"] = instr
                        fill2["is_split"] = True
                        fills.append(fill2)
                        
                    self.active_maker_orders["ask"] = None

        return fills
