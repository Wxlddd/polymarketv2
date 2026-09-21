"""How Polymarket's btc-updown contracts are actually defined.

The resolution source is Chainlink's `btc-usd-twap-60s` stream, and the published strike
(event.eventMetadata.priceToBeat) is the time-weighted average of the oracle over the 60s
BEFORE the cycle starts. Checked against 50 official strikes: median error 0.28 USD for
this formula, versus 11.87 USD for the first print after cycle start that the bot used.
A cycle's settlement price is the next cycle's strike, i.e. the same average over the 60s
before expiry.

The contract therefore pays "average of the last minute > average of the minute before the
start", not "spot at expiry > spot at start". Booking against point prices put the wrong
outcome on 14 of 83 settlements in one session.
"""
from bisect import bisect_right
from typing import Optional, Sequence, Tuple

WINDOW_SEC = 60.0


def oracle_twap(ticks: Sequence[Tuple[float, float]], t0: float, t1: float,
                times: Optional[Sequence[float]] = None) -> Optional[float]:
    """Time-weighted average of the oracle step function over [t0, t1).

    `ticks` is a time-sorted sequence of (timestamp, price). The price in force at t0 is the
    last print at or before t0, so a print is needed before the window starts; without one
    the value is unknown and None is returned rather than a guess. Pass `times` (the
    timestamps alone) when calling repeatedly on a long series, so they are not rebuilt.
    """
    if t1 <= t0 or not ticks:
        return None
    if times is None:
        times = [t for t, _ in ticks]
    i = bisect_right(times, t0) - 1
    if i < 0:
        return None
    acc = 0.0
    t = t0
    while t < t1:
        nxt = times[i + 1] if i + 1 < len(times) and times[i + 1] < t1 else t1
        acc += ticks[i][1] * (nxt - t)
        t = nxt
        i += 1
        if i >= len(times):
            acc += ticks[-1][1] * (t1 - t) if t < t1 else 0.0
            t = t1
    return acc / (t1 - t0)


def strike_price(ticks, cycle_start: float, times=None) -> Optional[float]:
    """The official priceToBeat: the oracle TWAP over the minute before the cycle starts."""
    return oracle_twap(ticks, cycle_start - WINDOW_SEC, cycle_start, times)


def settlement_price(ticks, expiry: float, times=None) -> Optional[float]:
    """The price the cycle resolves on: the oracle TWAP over the minute before expiry."""
    return oracle_twap(ticks, expiry - WINDOW_SEC, expiry, times)


def contract_terms(strike: float, spot: float, tau_sec: float,
                   realized_avg: Optional[float]) -> Tuple[float, float]:
    """Map the TWAP contract onto a plain "spot at T > K" question a pricer can answer.

    Returns (effective_strike, effective_tau). With the settlement averaging the final
    minute, what matters is the variance of that average, not of the spot at expiry:

      tau >= 60s  nothing of the final average is fixed yet; its variance is that of the
                  spot out to T-60 plus a third of the last minute:  tau_eff = tau - 40
      tau <  60s  a fraction w = (60 - tau)/60 of the average is already realised at A, the
                  remainder is the mean of the path over tau seconds (variance tau/3), so
                  TWAP > K  <=>  M > (K - w*A)/(1 - w)  with  tau_eff = tau/3

    The two branches agree at tau = 60 (tau_eff = 20).
    """
    if tau_sec >= WINDOW_SEC or realized_avg is None:
        return strike, max(tau_sec - 40.0, 0.0) if tau_sec >= WINDOW_SEC else tau_sec / 3.0
    w = (WINDOW_SEC - tau_sec) / WINDOW_SEC
    if w >= 1.0 - 1e-9:
        return strike, 0.0
    k_eff = (strike - w * realized_avg) / (1.0 - w)
    return k_eff, tau_sec / 3.0


if __name__ == "__main__":
    # Constant price: every average is that price.
    flat = [(0.0, 100.0)]
    assert abs(oracle_twap(flat, 10.0, 70.0) - 100.0) < 1e-9

    # Step at t=30 from 100 to 110 over [0,60): half the time at each.
    step = [(0.0, 100.0), (30.0, 110.0)]
    assert abs(oracle_twap(step, 0.0, 60.0) - 105.0) < 1e-9

    # Needs a print at or before the window start.
    assert oracle_twap([(50.0, 1.0)], 0.0, 60.0) is None

    # Strike and settlement are the same window convention.
    assert strike_price(step, 60.0) == oracle_twap(step, 0.0, 60.0)
    assert settlement_price(step, 60.0) == oracle_twap(step, 0.0, 60.0)

    # Continuity of the effective horizon at the start of the final minute.
    _, a = contract_terms(100.0, 100.0, 60.0, None)
    _, b = contract_terms(100.0, 100.0, 59.999, 100.0)
    assert abs(a - b) < 1e-2, (a, b)

    # Half the final minute realised above the strike lowers the bar for the rest.
    k_eff, _ = contract_terms(100.0, 100.0, 30.0, 104.0)
    assert k_eff < 100.0, k_eff        # (100 - 0.5*104)/0.5 = 96
    assert abs(k_eff - 96.0) < 1e-9
    print("[OK] twap, strike/settlement convention, and contract terms")
