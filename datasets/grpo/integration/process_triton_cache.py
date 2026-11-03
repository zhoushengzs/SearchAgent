"""Isolate concurrent workers and tolerate delayed visibility on shared storage."""
import os
import socket
import time
from triton.runtime.cache import FileCacheManager


def _install_loader_retry():
    from triton.compiler.compiler import CompiledKernel
    if getattr(CompiledKernel, "_shared_storage_retry", False):
        return
    original = CompiledKernel.__init__
    def initialize(self, *args, **kwargs):
        deadline = time.monotonic() + 30
        while True:
            try:
                return original(self, *args, **kwargs)
            except FileNotFoundError as error:
                root = os.environ.get("TRITON_CACHE_DIR", "")
                filename = str(error.filename or "")
                if not root or not filename.startswith(root.rstrip("/") + "/") or time.monotonic() >= deadline:
                    raise
                time.sleep(0.5)
    CompiledKernel.__init__ = initialize
    CompiledKernel._shared_storage_retry = True


class ProcessFileCacheManager(FileCacheManager):
    def __init__(self, key, override=False, dump=False):
        scoped_key = f"{socket.gethostname()}_{os.getpid()}_{key}"
        super().__init__(scoped_key, override=override, dump=dump)
        _install_loader_retry()
