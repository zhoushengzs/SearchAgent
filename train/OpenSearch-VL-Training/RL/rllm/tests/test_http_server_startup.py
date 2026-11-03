"""Verify that the real startup helper leaves a reachable HTTP listener."""
import ast
import asyncio
import contextlib
import ipaddress
import logging
from pathlib import Path
import socket
from types import SimpleNamespace
import unittest

import aiohttp
from fastapi import FastAPI
import uvicorn


SOURCE = Path(__file__).resolve().parents[1] / "verl/verl/workers/rollout/utils.py"


def load_helper():
    tree = ast.parse(SOURCE.read_text())
    names = {"is_valid_ipv6_address", "get_free_port", "run_unvicorn"}
    module = ast.fix_missing_locations(ast.Module(body=[n for n in tree.body if getattr(n, "name", None) in names], type_ignores=[]))
    namespace = {
        "asyncio": asyncio, "ipaddress": ipaddress, "socket": socket,
        "uvicorn": uvicorn, "FastAPI": FastAPI,
        "logger": logging.getLogger(__name__), "os": __import__("os"),
    }
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace["run_unvicorn"]


class StartupTests(unittest.IsolatedAsyncioTestCase):
    async def test_returned_server_stays_reachable(self):
        app = FastAPI()

        @app.get("/health")
        async def health():
            return {"ready": True}

        port, task = await load_helper()(app, SimpleNamespace(), "127.0.0.1")
        try:
            async with aiohttp.ClientSession() as session:
                for _ in range(2):
                    async with session.get(f"http://127.0.0.1:{port}/health", timeout=aiohttp.ClientTimeout(total=2)) as response:
                        self.assertEqual(response.status, 200)
                        self.assertEqual(await response.json(), {"ready": True})
                    await asyncio.sleep(0.1)
            self.assertFalse(task.done())
        finally:
            # Request normal shutdown of the real Uvicorn server.
            server = task.get_coro().cr_frame.f_locals.get("self") if task.get_coro().cr_frame else None
            if server is not None:
                server.should_exit = True
                await asyncio.wait_for(task, timeout=5)
            else:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task


if __name__ == "__main__":
    unittest.main()
