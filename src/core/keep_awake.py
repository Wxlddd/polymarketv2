"""Keep Windows awake while a long run is in progress, without keeping the screen on.

Used by the live orchestrator and by the backtester: an overnight collection run or a
multi-hour sweep is worth nothing if the machine suspends halfway through.

ES_DISPLAY_REQUIRED is deliberately NOT set. Neither the bot nor a replay needs the panel
lit, and on a laptop the display is the single largest draw.
"""
import logging
import os
import threading
import time

logger = logging.getLogger("KeepAwake")

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001

_started = False


def keep_awake(interval_sec: float = 30.0) -> bool:
    """Starts a daemon thread holding the no-sleep state. No-op off Windows, or if already
    running. Returns True if the state is being held.

    The flag is per-thread and Windows drops it when that thread exits, so one long-lived
    thread owns it and refreshes it rather than every caller setting it once.
    """
    global _started
    if _started:
        return True
    if os.name != "nt":
        return False

    try:
        import ctypes
    except Exception:
        return False

    def _hold() -> None:
        while True:
            try:
                ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
            except Exception as e:
                logger.warning(f"[KeepAwake] Failed to refresh execution state: {e}")
                return
            time.sleep(interval_sec)

    threading.Thread(target=_hold, daemon=True, name="keep-awake").start()
    _started = True
    logger.info("[KeepAwake] System sleep suppressed (display left alone).")
    return True


if __name__ == "__main__":
    import ctypes

    assert keep_awake(interval_sec=0.1) == (os.name == "nt")
    assert keep_awake() is (True if os.name == "nt" else False), "second call must not start a second thread"
    threads = [t for t in threading.enumerate() if t.name == "keep-awake"]
    assert len(threads) <= 1, f"{len(threads)} keep-awake threads running"

    if os.name == "nt":
        time.sleep(0.3)
        # Querying with ES_CONTINUOUS alone returns the previous state without changing it.
        prev = ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
        assert prev != 0, "Windows rejected the execution state call"
        assert not (prev & 0x00000002), "display requirement set: the screen would stay on"
        print(f"[OK] sleep suppressed, display flag clear (state=0x{prev:08x})")
    else:
        print("[OK] no-op off Windows")
