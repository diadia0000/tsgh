"""Windows compatibility shims for tuspyserver (4.2.12, the latest release as of
2026-07 -- both bugs below are still present upstream).

Two POSIX-only spots make resumable CZI upload impossible on Windows, and the
first one takes the whole backend down with it:

1. `tuspyserver/lock.py` does a top-level `import fcntl`, so merely importing
   the package raises ModuleNotFoundError. Since `backend/main.py` imports the
   alignment router, the entire API -- tiles, jobs, alignment -- fails to start.
2. `tuspyserver/info.py:serialize()` finalises the upload's `.info` sidecar with
   `os.rename(tmp, path)`. On Windows rename fails with WinError 183 when the
   destination exists, and the sidecar is rewritten on every PATCH, so every
   chunk after the first returns 500 and no upload can ever complete.

Import this module BEFORE `tuspyserver` (see backend/api/alignment.py); the
fcntl stub has to be in sys.modules by the time tuspyserver's own imports run.
On non-Windows platforms this module does nothing at all.
"""
import os
import sys
import types


def _install_fcntl_stub() -> None:
    """Advisory flock as a no-op.

    tuspyserver takes the lock to serialise concurrent PATCHes against one
    upload. We run a single uvicorn worker for a single user, and tus's own
    Upload-Offset check already rejects a mis-ordered chunk, so a no-op lock
    costs us nothing here. Windows has no flock equivalent worth emulating for
    this (msvcrt.locking is byte-range and mandatory, not advisory).
    """
    stub = types.ModuleType("fcntl")
    stub.LOCK_SH, stub.LOCK_EX, stub.LOCK_NB, stub.LOCK_UN = 1, 2, 4, 8
    stub.flock = lambda fd, operation: None
    stub.lockf = lambda fd, operation, *args: None
    sys.modules["fcntl"] = stub


class _RenameIsReplace:
    """Stand-in for the `os` module that makes `os.rename` overwrite.

    Swapped into tuspyserver.info only. `os.replace` is what that code wants --
    atomic, overwrites the destination -- and it behaves identically to
    `os.rename` on POSIX, so this narrows a Windows-only failure without
    changing behaviour anywhere else.
    """

    rename = staticmethod(os.replace)

    def __getattr__(self, name):
        return getattr(os, name)


def _patch_info_rename() -> None:
    from tuspyserver import info

    info.os = _RenameIsReplace()


if sys.platform == "win32":
    if "fcntl" not in sys.modules:
        _install_fcntl_stub()
    _patch_info_rename()
