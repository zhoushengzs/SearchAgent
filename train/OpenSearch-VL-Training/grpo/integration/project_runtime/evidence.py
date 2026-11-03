"""Portable copy of the SFT evidence tools; no image-upload service required."""
from __future__ import annotations
import ast
import hashlib
import ipaddress
import json
import math
import socket
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse
import requests
from .identity import file_sha, fingerprint
from .views import materialize_view
from .serper_keys import configured_keys, serper_post

TOOL_NAMES = {"vs_web_search", "vs_image_search", "vs_visit", "vs_inspect_image", "vs_crop", "vs_calculate"}
TRAJECTORY_CATALOG = json.loads((Path(__file__).parent / "blocked_sources.json").read_text())

def public_url(url):
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Expected an unauthenticated HTTP(S) URL")
    return url

def blocked_benchmark_url(url: str) -> bool:
    parsed = urlparse(url)
    path = parsed.path.lower()
    for entry in TRAJECTORY_CATALOG:
        hf = entry.get("hf_id")
        if hf and parsed.hostname == "huggingface.co" and path.startswith("/datasets/" + hf.lower()):
            return True
        official = urlparse(entry["official_url"])
        if official.hostname == "github.com" and parsed.hostname in {"github.com", "raw.githubusercontent.com"}:
            if path.startswith(official.path.lower() + "/") or path == official.path.lower():
                return True
    return False

def check_public_url(url: str):
    public_url(url)
    if blocked_benchmark_url(url):
        raise ValueError("Benchmark reference repositories are excluded from agent retrieval")
    host = urlparse(url).hostname
    addresses = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    if not addresses or any(not ipaddress.ip_address(item[4][0]).is_global for item in addresses):
        raise ValueError("Agent retrieval requires a public network address")

def fetch_public(url: str, *, max_bytes: int = 8 * 1024 * 1024, timeout=30):
    """Validate each redirect and stop oversized bodies before materialization."""
    for _ in range(6):
        check_public_url(url)
        with requests.get(url, timeout=timeout, stream=True, allow_redirects=False,
                          headers={"User-Agent": "VisualEvidenceAgent/1.0"}) as response:
            if response.status_code in (301, 302, 303, 307, 308):
                url = urljoin(url, response.headers["Location"])
                continue
            response.raise_for_status()
            body = bytearray()
            for chunk in response.iter_content(64 * 1024):
                body.extend(chunk)
                if len(body) > max_bytes:
                    raise ValueError("Tool response exceeds byte budget")
            return bytes(body), url, response.headers.get("Content-Type", ""), response.status_code
    raise ValueError("Too many redirects")

class PageParser(HTMLParser):
    def __init__(self, url):
        super().__init__(convert_charrefs=True)
        self.url, self.text, self.links, self.images = url, [], [], []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag in {"script", "style", "noscript", "head"}:
            self.hidden += 1
        if self.hidden:
            return
        if tag == "a" and values.get("href"):
            self.links.append(urljoin(self.url, values["href"]))
        if tag == "img" and (values.get("src") or values.get("data-src")):
            self.images.append({"url": urljoin(self.url, values.get("src") or values["data-src"]),
                                "alt": values.get("alt", "")})

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript", "head"} and self.hidden:
            self.hidden -= 1

    def handle_data(self, data):
        if not self.hidden and data.strip():
            self.text.append(data.strip())

def calculate(expression: str) -> float | int:
    if not isinstance(expression, str) or len(expression) > 200:
        raise ValueError("Invalid arithmetic expression")
    tree = ast.parse(expression, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 50:
        raise ValueError("Arithmetic expression too complex")
    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = node.value
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
            a, b = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Pow) and abs(b) > 20:
                raise ValueError("Exponent outside budget")
            value = (a + b if isinstance(node.op, ast.Add) else a - b if isinstance(node.op, ast.Sub)
                     else a * b if isinstance(node.op, ast.Mult) else a / b if isinstance(node.op, ast.Div) else a ** b)
        else:
            raise ValueError("Only numeric arithmetic is allowed")
        if not isinstance(value, (float, int)) or not math.isfinite(value) or abs(value) > 1e15:
            raise ValueError("Arithmetic result outside budget")
        return value
    return visit(tree.body)

class EvidenceTools:
    def __init__(self, directory: Path, *, image_side=1536, page_chars=8000):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.serper_key = bool(configured_keys())
        self.image_side, self.page_chars = image_side, page_chars
        self.assets, self.urls, self.image_urls = {}, set(), set()
        self.kb = None
        self.kb_sha = None

    def configuration(self):
        return {"backend": "serper", "available_tools": sorted(TOOL_NAMES),
                "image_side": self.image_side, "page_chars": self.page_chars,
                "version": "sft-evidence-tools-v1"}

    def close(self):
        pass

    def initialize(self, images: list[dict]) -> list[dict]:
        previews = []
        for image in images:
            if file_sha(image["path"]) != image["sha256"]:
                raise ValueError("Initial image bytes changed")
            key = f"image_{image['index']}"
            self.assets[key] = {"source_path": image["path"], "source_sha256": image["sha256"], "origin": "initial_input"}
            view = materialize_view(image["path"], self.directory / (key + "_preview.png"), None, max_side=512)
            previews.append({"asset_id": key, **view})
        return previews

    def inventory(self):
        return [{"asset_id": key, "origin": value["origin"]} for key, value in self.assets.items()]

    def _search(self, query: str, images: bool):
        if not isinstance(query, str) or not query.strip() or len(query) > 1000:
            raise ValueError("Search query must be nonempty and bounded")
        if not images and self.kb:
            result = self.kb.search(query)
            self.urls.update(row["url"] for row in result)
            return {"query": query, "results": result, "backend": "local_kb", "kb_sha256": self.kb_sha}
        if not self.serper_key:
            raise ValueError("Set SERPER_API_KEY for real web/image search or supply a full local KB")
        endpoint = "images" if images else "search"
        response = serper_post("https://google.serper.dev/" + endpoint,
                               json={"q": query, "num": 5}, timeout=(10, 45))
        try:
            response.raise_for_status()
            data = response.json()
        finally:
            response.close()
        if not isinstance(data, dict) or data.get("error"):
            raise ValueError("Search provider returned an error")
        rows = data.get("images" if images else "organic", [])
        if not isinstance(rows, list):
            raise ValueError("Search provider returned no result array")
        results = []
        for row in rows[:5]:
            url = row.get("link")
            if not isinstance(url, str) or blocked_benchmark_url(url):
                continue
            result = {"url": url, "title": str(row.get("title", "")), "snippet": str(row.get("snippet", ""))}
            self.urls.add(url)
            if images and isinstance(row.get("imageUrl"), str) and not blocked_benchmark_url(row["imageUrl"]):
                result["image_url"] = row["imageUrl"]
                self.image_urls.add(row["imageUrl"])
            results.append(result)
        return {"query": query, "results": results, "backend": "serper"}

    def _visit(self, url: str, offset: int):
        if type(offset) is not int or offset < 0:
            raise ValueError("Page offset must be a nonnegative character offset")
        if url not in self.urls:
            raise ValueError("Visit only URLs acquired from search, links or the original question")
        if url.startswith("kb:") and self.kb:
            doc = self.kb.document(url)
            text = doc["text"]
            path = self.directory / ("kb_page_" + fingerprint(doc) + ".json")
            path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
            return {"url": url, "source_url": doc["url"], "title": doc["title"], "text": text[offset:offset + self.page_chars],
                    "offset": offset, "total_chars": len(text), "kb_sha256": self.kb_sha,
                    "raw_artifact": {"path": str(path), "sha256": file_sha(path)}}
        raw, final_url, content_type, status = fetch_public(url)
        path = self.directory / ("page_" + hashlib.sha256(raw).hexdigest() + ".raw")
        path.write_bytes(raw)
        if "html" in content_type:
            parser = PageParser(final_url)
            parser.feed(raw.decode("utf-8", errors="replace"))
            text = "\n".join(parser.text)
            links = list(dict.fromkeys(u for u in parser.links if u.startswith(("https://", "http://"))
                                       and not blocked_benchmark_url(u)))[:60]
            images = [im for im in parser.images if im["url"].startswith(("http://", "https://"))
                      and not blocked_benchmark_url(im["url"])][:30]
            self.urls.update(links)
            self.image_urls.update(im["url"] for im in images)
        elif content_type.startswith("text/"):
            text, links, images = raw.decode("utf-8", errors="replace"), [], []
        else:
            raise ValueError("visit supports text/HTML; acquire image URLs with inspect_image")
        return {"url": url, "final_url": final_url, "http_status": status,
                "text": text[offset:offset + self.page_chars], "offset": offset, "total_chars": len(text),
                "links": links, "images": images, "raw_artifact": {"path": str(path), "sha256": file_sha(path)}}

    def execute(self, tool: str, args: dict, step: int) -> dict:
        """Always log a result, including real tool errors. No generated fallback data."""
        try:
            if tool in {"web_search", "image_search"}:
                data = self._search(args["query"], tool == "image_search")
            elif tool == "visit":
                data = self._visit(args["url"], args.get("offset", 0))
            elif tool in {"inspect_image", "crop"}:
                if "image_url" in args:
                    url = args["image_url"]
                    if url not in self.image_urls:
                        raise ValueError("Acquire the image URL from search or a visited page first")
                    raw, final_url, _, _ = fetch_public(url, max_bytes=16 * 1024 * 1024)
                    path = self.directory / ("acquired_" + hashlib.sha256(raw).hexdigest() + ".image")
                    path.write_bytes(raw)
                    key = "retrieved_" + fingerprint({"url": url, "sha": file_sha(path)})[:12]
                    self.assets[key] = {"source_path": str(path), "source_sha256": file_sha(path),
                                        "origin": final_url}
                else:
                    key = args["asset_id"]
                asset = self.assets[key]
                view = materialize_view(asset["source_path"], self.directory / f"step_{step}_{key}.png",
                    args.get("bbox"), max_side=self.image_side)
                data = {"asset_id": key, "origin": asset["origin"], "source_sha256": asset["source_sha256"],
                        "source_path": asset["source_path"], "view": view}
            elif tool == "calculate":
                data = {"expression": args["expression"], "value": calculate(args["expression"])}
            else:
                raise ValueError("Not an executable evidence tool")
            result = {"tool": tool, "status": "ok", "data": data}
        except Exception as exc:
            # Do not serialize provider request objects, headers or API-key URLs.
            result = {"tool": tool, "status": "error", "error_type": type(exc).__name__}
            if isinstance(exc, requests.HTTPError) and exc.response is not None:
                result["http_status"] = exc.response.status_code
        result["observed_at"] = datetime.now(timezone.utc).isoformat()
        path = self.directory / f"step_{step}_result.json"
        path.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        return {**result, "artifact": {"path": str(path), "sha256": file_sha(path)}}
