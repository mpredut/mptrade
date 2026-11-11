"""Cross-process and cross-thread file mutex (`FileLock`)."""
from botcore import FileLock, _HAVE_FCNTL

__all__ = ["FileLock", "_HAVE_FCNTL"]


