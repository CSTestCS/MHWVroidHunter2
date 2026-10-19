"""PyInstaller entry point for VRoidHunter.exe."""

import os
import sys
import traceback


def _log_path():
    base = os.path.dirname(os.path.abspath(sys.executable if getattr(sys, "frozen", False) else __file__))
    return os.path.join(base, "VRoidHunter.log")


if __name__ == "__main__":
    # A windowed exe has no console: route output to a log file before anything can print.
    if sys.stdout is None or sys.stderr is None:
        try:
            stream = open(_log_path(), "w", encoding="utf-8", buffering=1)
        except OSError:
            stream = open(os.devnull, "w")
        sys.stdout = sys.stdout or stream
        sys.stderr = sys.stderr or stream
    try:
        from vroidhunter.__main__ import main
        main()
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        try:
            import ctypes
            if getattr(sys, "frozen", False) and "--help" not in sys.argv:
                ctypes.windll.user32.MessageBoxW(None, traceback.format_exc()[-1500:], "VRoid Hunter error", 0x10)
        except Exception:
            pass
        sys.exit(1)
