"""Bounded cached public downloads and range access to large image archives."""
from __future__ import annotations

import io
import os
import re
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import requests

from core.provenance import fingerprint


def public_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Expected an unauthenticated HTTP(S) URL")
    # Dataset downloads use fixed author URLs. Agent tools additionally check DNS.
    if parsed.hostname.lower() in {"localhost", "127.0.0.1", "::1", "0.0.0.0"}:
        raise ValueError("Local URL is not a public dataset asset")
    return url


def cached_download(url: str, directory: Path, *, max_bytes: int = 64 * 1024 * 1024,
                    timeout: int = 45) -> Path:
    """Cache by full pinned URL; never accept an incomplete response as a file."""
    public_url(url)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / fingerprint(url)
    if target.is_file():
        if target.stat().st_size > max_bytes:
            raise ValueError("Cached download exceeds configured size limit")
        return target
    fd, temporary = tempfile.mkstemp(prefix="download_", dir=directory)
    try:
        with os.fdopen(fd, "wb") as output, requests.get(url, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            length = response.headers.get("Content-Length")
            if length and int(length) > max_bytes:
                raise ValueError("Download exceeds configured size limit; provide the archive locally")
            received = 0
            for chunk in response.iter_content(128 * 1024):
                received += len(chunk)
                if received > max_bytes:
                    raise ValueError("Download exceeded configured size limit")
                output.write(chunk)
            if not received:
                raise ValueError("Empty asset response")
            # requests decodes compressed HTTP bodies; lengths then differ legitimately.
            if length and not response.headers.get("Content-Encoding") and received != int(length):
                raise ValueError("Incomplete asset response")
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return target


class RangeFile(io.RawIOBase):
    'Seekable HTTP file for zipfile, downloading only requested byte blocks.'
    def __init__(self, url: str, size: int, cache_dir: Path, block_size: int = 256 * 1024):
        super().__init__()
        self.url, self.size, self.position = public_url(url), size, 0
        self.cache_dir = cache_dir / fingerprint({"url": url, "size": size})
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.block_size = block_size

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        position = offset + (0 if whence == io.SEEK_SET else self.position if whence == io.SEEK_CUR else self.size)
        if position < 0 or whence not in (io.SEEK_SET, io.SEEK_CUR, io.SEEK_END):
            raise ValueError("Invalid archive seek")
        self.position = position
        return position

    def _block(self, start: int) -> bytes:
        end = min(start + self.block_size, self.size) - 1
        path = self.cache_dir / str(start)
        expected = end - start + 1
        if path.is_file() and path.stat().st_size == expected:
            return path.read_bytes()
        with requests.get(self.url, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
                          stream=True, timeout=60) as response:
            response.raise_for_status()
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
            if response.status_code != 206 or not match or tuple(map(int, match.groups())) != (start, end, self.size):
                raise ValueError("Archive server did not honor byte ranges; provide images.zip locally")
            data = response.raw.read(expected + 1)
        if len(data) != expected:
            raise ValueError("Archive byte-range response is incomplete")
        path.write_bytes(data)
        return data

    def read(self, size=-1):
        remaining = max(0, self.size - self.position) if size < 0 else min(size, max(0, self.size - self.position))
        if remaining > 32 * 1024 * 1024:
            raise ValueError("Archive member exceeds bounded read limit")
        parts = []
        while remaining:
            start = self.position // self.block_size * self.block_size
            block = self._block(start)
            offset = self.position - start
            piece = block[offset:offset + remaining]
            if not piece:
                raise ValueError("Archive range made no progress")
            parts.append(piece)
            self.position += len(piece)
            remaining -= len(piece)
        return b"".join(parts)
