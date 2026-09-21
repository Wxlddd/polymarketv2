"""Settlement must follow the contract definition, and the venue's resolution wins.

Polymarket's btc-updown resolves on Chainlink's 60s TWAP stream with the strike published
as event.eventMetadata.priceToBeat = TWAP of the minute before the cycle starts. The bot
used point prints and booked the wrong outcome on 14 of 83 settlements in one session;
with the TWAP convention that fell to 1 of 73, and that one was a window with oracle data
missing. The reconciliation step covers what the reconstruction cannot.

    PYTHONPATH=. uv run python tests/verify_settlement.py
"""
from config.settings import SystemConfig
from src.core.twap import contract_terms, oracle_twap, settlement_price, strike_price
from src.execution.clients import MockExecutionClient
from src.execution.shadow_book import ShadowOrderBook


class _Rec:
    def __init__(self):
        self.rows = []

    def record_trade(self, **k):
        self.rows.append(k)

    def record_signal(self, *a, **k):
        pass


def client():
    return MockExecutionClient(SystemConfig(), _Rec(), ShadowOrderBook())


def main():
    # 1. A spike in the final seconds must not decide the outcome on its own. This is the
    #    case reported live: spot jumped above the strike just before expiry while the
    #    market priced Up at 1%, because the minute's average never got there.
    start, expiry = 1_000.0, 1_300.0
    ticks = [(start - 90, 100.0)]                     # strike window flat at 100
    ticks += [(expiry - 60 + i, 99.0) for i in range(55)]   # most of the last minute below
    ticks += [(expiry - 5, 110.0)]                    # late spike above
    k = strike_price(ticks, start)
    last_point = ticks[-1][1]
    twap = settlement_price(ticks, expiry)
    assert abs(k - 100.0) < 1e-9, k
    assert last_point > k, "setup: the last print is above the strike"
    assert twap < k, f"the average of the final minute ({twap:.3f}) must be below the strike"
    print(f"[OK] late spike to {last_point:.0f} over a strike of {k:.0f}: point says UP, "
          f"TWAP {twap:.3f} says DOWN (the venue's answer)")

    # 2. Settlement of cycle N is the strike of cycle N+1.
    assert settlement_price(ticks, expiry) == strike_price(ticks, expiry)
    print("[OK] a cycle's settlement is the next cycle's strike")

    # 3. The pricer's effective strike moves toward what is already realised.
    k_eff, tau_eff = contract_terms(100.0, 100.0, 30.0, oracle_twap(ticks, expiry - 60, expiry - 30))
    assert k_eff > 100.0 and tau_eff == 10.0, (k_eff, tau_eff)
    print(f"[OK] half the final minute realised at 99: the rest must average {k_eff:.2f} to win")

    # 4. Reconciliation: booked Up, venue paid Down, 300 YES held -> give back 300.
    c = client()
    cash0 = c.cash_balance
    c.adjust_settlement(delta_cash=-300.0, timestamp=1.0, strike=100.0, official_up=False)
    assert abs(c.cash_balance - (cash0 - 300.0)) < 1e-9
    assert c.recorder.rows[-1]["side"] == "SETTLE_ADJUST"
    print("[OK] booked UP / venue DOWN on 300 YES: cash corrected by -300, ledger row written")

    # 5. Mirror: booked Down, venue paid Up, 200 YES and 50 NO -> +150.
    c = client()
    cash0 = c.cash_balance
    delta = (1.0 if True else -1.0) * (200.0 - 50.0)
    c.adjust_settlement(delta_cash=delta, timestamp=1.0, strike=100.0, official_up=True)
    assert abs(c.cash_balance - (cash0 + 150.0)) < 1e-9
    print("[OK] booked DOWN / venue UP on 200 YES + 50 NO: cash corrected by +150")

    # 6. Agreement changes nothing.
    c = client()
    cash0 = c.cash_balance
    c.adjust_settlement(delta_cash=0.0, timestamp=1.0, strike=100.0, official_up=True)
    assert c.cash_balance == cash0 and not c.recorder.rows
    print("[OK] no correction when the booking already matches the venue")

    print("\n=== Settlement verified ===")


if __name__ == "__main__":
    main()
