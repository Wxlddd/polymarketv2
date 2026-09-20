"""Keep the machine running a long session, without keeping the screen on.

Two different Windows mechanisms, and only the second one works on modern laptops:

- SetThreadExecutionState(ES_SYSTEM_REQUIRED) blocks classic S3 sleep. On a Modern
  Standby (S0 low-power idle) machine it does NOT prevent the system from entering
  standby once the screen turns off. Measured: a collection run went to sleep 18 minutes
  in and lost 2 of its first 2.5 hours, with the wake-ups visible as
  "exiting Modern Standby" in the Kernel-Power log and as getaddrinfo failures in ours.
- PowerSetRequest(PowerRequestExecutionRequired) is the API that keeps a process running
  through Modern Standby. This is the one that matters.

Both are set, because a machine can be either kind. ES_DISPLAY_REQUIRED is deliberately
NOT used: the panel is the largest draw on a laptop and nothing here needs it lit.
"""
import ctypes
import logging
import os
import threading
import time
from ctypes import wintypes

logger = logging.getLogger("KeepAwake")

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001

# POWER_REQUEST_TYPE
PowerRequestSystemRequired = 0
PowerRequestExecutionRequired = 3

_started = False
_request_handle = None


class _Detailed(ctypes.Structure):
    _fields_ = [("LocalizedReasonModule", wintypes.HMODULE),
                ("LocalizedReasonId", wintypes.ULONG),
                ("ReasonStringCount", wintypes.ULONG),
                ("ReasonStrings", ctypes.POINTER(wintypes.LPWSTR))]


class _Reason(ctypes.Union):
    _fields_ = [("Detailed", _Detailed),
                ("SimpleReasonString", wintypes.LPWSTR)]


class _ReasonContext(ctypes.Structure):
    """REASON_CONTEXT carrying a plain string reason, which is what shows up in
    `powercfg /requests`."""
    _fields_ = [("Version", wintypes.ULONG),
                ("Flags", wintypes.DWORD),
                ("Reason", _Reason)]

POWER_REQUEST_CONTEXT_VERSION = 0
POWER_REQUEST_CONTEXT_SIMPLE_STRING = 0x1


def _hold_execution(reason: str) -> bool:
    """Ask Windows to keep this process executing, Modern Standby included."""
    global _request_handle
    k32 = ctypes.windll.kernel32
    ctx = _ReasonContext()
    ctx.Version = POWER_REQUEST_CONTEXT_VERSION
    ctx.Flags = POWER_REQUEST_CONTEXT_SIMPLE_STRING
    ctx.Reason.SimpleReasonString = reason

    k32.PowerCreateRequest.restype = wintypes.HANDLE
    handle = k32.PowerCreateRequest(ctypes.byref(ctx))
    if not handle or handle == wintypes.HANDLE(-1).value:
        logger.warning("[KeepAwake] PowerCreateRequest failed; Modern Standby may still suspend us.")
        return False

    ok = True
    for req in (PowerRequestExecutionRequired, PowerRequestSystemRequired):
        if not k32.PowerSetRequest(handle, req):
            # ExecutionRequired needs Windows 8+; SystemRequired always exists.
            if req == PowerRequestExecutionRequired:
                logger.info("[KeepAwake] ExecutionRequired unavailable, falling back to SystemRequired.")
            else:
                ok = False
    _request_handle = handle
    return ok


def keep_awake(interval_sec: float = 30.0, reason: str = "polymarketv2 data collection") -> bool:
    """Suppresses sleep for the life of the process. No-op off Windows or if already held."""
    global _started
    if _started:
        return True
    if os.name != "nt":
        return False

    held = _hold_execution(reason)

    def _refresh() -> None:
        while True:
            try:
                ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
            except Exception as e:
                logger.warning(f"[KeepAwake] Failed to refresh execution state: {e}")
                return
            time.sleep(interval_sec)

    threading.Thread(target=_refresh, daemon=True, name="keep-awake").start()
    _started = True
    logger.info(f"[KeepAwake] Sleep suppressed (execution request {'held' if held else 'unavailable'}, "
                f"display left alone).")
    return True


if __name__ == "__main__":
    assert keep_awake(interval_sec=0.1) == (os.name == "nt")
    assert keep_awake() is (os.name == "nt"), "second call must not start a second thread"
    threads = [t for t in threading.enumerate() if t.name == "keep-awake"]
    assert len(threads) <= 1, f"{len(threads)} keep-awake threads running"

    if os.name == "nt":
        time.sleep(0.3)
        assert _request_handle, "no power request handle: Modern Standby would still suspend us"
        prev = ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
        assert prev != 0, "Windows rejected the execution state call"
        assert not (prev & 0x00000002), "display requirement set: the screen would stay on"
        print(f"[OK] execution request held, sleep suppressed, display flag clear (state=0x{prev & 0xFFFFFFFF:08x})")
        print("     verify with an elevated:  powercfg /requests")
    else:
        print("[OK] no-op off Windows")
