from __future__ import annotations

import json
import os
import re
import time
import uuid
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from email.parser import BytesParser
from email.policy import default
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread
from typing import Any
from urllib.parse import quote, unquote

import fitz
import requests


ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = ROOT / "uploads"
OUTPUT_DIR = ROOT / "outputs"
STATIC_DIR = ROOT / "static"
HOST = "127.0.0.1"
PORT = int(os.environ.get("PDF_TRANSLATOR_PORT", "8765"))
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_FORM_BYTES = MAX_UPLOAD_BYTES + 1024 * 1024
JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = Lock()
TRANSLATION_CACHE: dict[str, str] = {}
CACHE_LOCK = Lock()

DEFAULT_FONT_CANDIDATES = [
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Songti.ttc",
]
FONT_OPTIONS: dict[str, tuple[str, list[str]]] = {
    "academic": (
        "论文模板",
        DEFAULT_FONT_CANDIDATES,
    ),
    "auto": (
        "自动",
        DEFAULT_FONT_CANDIDATES,
    ),
    "songti": (
        "宋体",
        [
            "/System/Library/Fonts/Supplemental/Songti.ttc",
            "/Library/Fonts/Songti.ttc",
        ],
    ),
    "heiti": (
        "黑体",
        [
            "/System/Library/Fonts/STHeiti Medium.ttc",
            "/System/Library/Fonts/STHeiti Light.ttc",
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
        ],
    ),
    "arial_unicode": (
        "Arial Unicode",
        [
            "/Library/Fonts/Arial Unicode.ttf",
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        ],
    ),
    "helvetica": (
        "内置 Helvetica",
        [],
    ),
}
ROLE_FONT_CHOICES = {
    "title": "heiti",
    "heading": "heiti",
    "body": "songti",
    "caption": "songti",
    "reference": "songti",
    "running": "songti",
}

SYSTEM_PROMPT = (
    "You are a professional academic paper translator. Translate the given text "
    "faithfully into {target}. Preserve numbers, citations, equations, symbols, "
    "abbreviations, and line meaning. Return only the translation."
)

DEEPSEEK_CHAT_URL = "https://api.deepseek.com/chat/completions"
GEMINI_CHAT_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
DEEPSEEK_MODELS = [
    "deepseek-v4-flash",
    "deepseek-v4-pro",
    "deepseek-chat",
    "deepseek-reasoner",
]
GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]
OPENAI_COMPATIBLE_PROVIDERS = {"openai", "deepseek", "gemini"}


@dataclass
class TextLine:
    page_index: int
    rect: fitz.Rect
    text: str
    font_size: float
    color: tuple[float, float, float]
    role: str = "body"


def ensure_dirs() -> None:
    UPLOAD_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    STATIC_DIR.mkdir(exist_ok=True)


def pick_font(font_choice: str = "auto") -> tuple[str | None, str]:
    label, candidates = FONT_OPTIONS.get(font_choice, FONT_OPTIONS["auto"])
    if font_choice == "helvetica":
        return None, label
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate, label
    for candidate in DEFAULT_FONT_CANDIDATES:
        if Path(candidate).exists():
            return candidate, "自动"
    return None, "内置 Helvetica"


def pick_font_path(font_choice: str) -> str | None:
    return pick_font(font_choice)[0]


def pick_font_bundle(font_choice: str = "auto") -> tuple[dict[str, str | None], str]:
    if font_choice != "academic":
        font_file, font_label = pick_font(font_choice)
        return {"default": font_file}, font_label
    fonts = {"default": pick_font_path("songti")}
    for role, role_choice in ROLE_FONT_CHOICES.items():
        fonts[role] = pick_font_path(role_choice) or fonts["default"]
    return fonts, "论文模板"


def parse_multipart(headers: dict[str, str], body: bytes) -> dict[str, Any]:
    content_type = headers.get("content-type", "")
    message = BytesParser(policy=default).parsebytes(
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode()
        + body
    )
    result: dict[str, Any] = {}
    for part in message.iter_parts():
        disposition = part.get("content-disposition", "")
        name_match = re.search(r'name="([^"]+)"', disposition)
        if not name_match:
            continue
        name = name_match.group(1)
        filename_match = re.search(r'filename="([^"]*)"', disposition)
        payload = part.get_payload(decode=True) or b""
        if filename_match and filename_match.group(1):
            result[name] = {
                "filename": Path(filename_match.group(1)).name,
                "content_type": part.get_content_type(),
                "content": payload,
            }
        else:
            result[name] = payload.decode(part.get_content_charset() or "utf-8").strip()
    return result


def rgb_from_int(color: int) -> tuple[float, float, float]:
    r = ((color >> 16) & 255) / 255
    g = ((color >> 8) & 255) / 255
    b = (color & 255) / 255
    return (r, g, b)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def classify_text_unit(text: str, rect: fitz.Rect, font_size: float, page_rect: fitz.Rect) -> str:
    stripped = text.strip()
    lower = stripped.lower()
    top_ratio = rect.y0 / max(page_rect.height, 1)
    bottom_ratio = rect.y1 / max(page_rect.height, 1)
    if top_ratio < 0.07 and (
        "remote sensing of environment" in lower
        or re.search(r"\b\d{4}\)\s*\d{5,}", stripped)
        or re.match(r"^[A-Z]\.\s*[A-Za-z]+ et al\.?$", stripped)
    ):
        return "running"
    if bottom_ratio > 0.95 and re.fullmatch(r"\d+", stripped):
        return "running"
    if re.match(r"^(table|fig\.?|figure|表|图)\s*[\d一二三四五六七八九十]", stripped, re.IGNORECASE):
        return "caption"
    if lower.startswith(("references", "reference ")) or stripped.startswith(("参考文献", "致谢")):
        return "reference"
    if re.match(r"^\d+(\.\d+){0,3}\s+\S+", stripped) and len(stripped) < 120:
        return "heading"
    if font_size >= 16 and rect.y0 < page_rect.height * 0.35 and len(stripped) < 180:
        return "title"

    number_count = len(re.findall(r"[-+]?\d+(?:\.\d+)?", stripped))
    letters = len(re.findall(r"[A-Za-z\u4e00-\u9fff]", stripped))
    chars = max(len(stripped), 1)
    numeric_ratio = sum(ch.isdigit() for ch in stripped) / chars
    separators = stripped.count(" ") + stripped.count("\t")
    math_marks = len(re.findall(r"[=∑Σ√≤≥±×÷/%]", stripped))
    if number_count >= 8 and numeric_ratio > 0.18 and separators >= 8:
        return "table"
    if number_count >= 5 and letters < number_count * 3:
        return "table"
    if math_marks >= 3 and number_count >= 2 and len(stripped) < 160:
        return "formula"
    return "body"


def should_translate_unit(line: TextLine) -> bool:
    if line.role in {"table", "formula", "caption", "running"}:
        return False
    return True


def extract_lines(doc: fitz.Document, page_filter: set[int] | None = None) -> list[TextLine]:
    lines: list[TextLine] = []
    for page_index, page in enumerate(doc):
        if page_filter is not None and page_index not in page_filter:
            continue
        data = page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT)
        for block in data.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                spans = [span for span in line.get("spans", []) if normalize_text(span.get("text", ""))]
                if not spans:
                    continue
                text = normalize_text("".join(span.get("text", "") for span in spans))
                if len(text) < 2:
                    continue
                rect = fitz.Rect(line["bbox"])
                font_size = max(6.0, min(float(span.get("size", 10)) for span in spans))
                color = rgb_from_int(int(spans[0].get("color", 0)))
                role = classify_text_unit(text, rect, font_size, page.rect)
                lines.append(TextLine(page_index, rect, text, font_size, color, role))
    return lines


def extract_blocks(doc: fitz.Document, page_filter: set[int] | None = None) -> list[TextLine]:
    blocks_out: list[TextLine] = []
    for page_index, page in enumerate(doc):
        if page_filter is not None and page_index not in page_filter:
            continue
        data = page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT)
        for block in data.get("blocks", []):
            if block.get("type") != 0:
                continue
            line_texts: list[str] = []
            span_sizes: list[float] = []
            first_color: tuple[float, float, float] | None = None
            for line in block.get("lines", []):
                spans = [span for span in line.get("spans", []) if normalize_text(span.get("text", ""))]
                if not spans:
                    continue
                text = normalize_text("".join(span.get("text", "") for span in spans))
                if text:
                    line_texts.append(text)
                for span in spans:
                    span_sizes.append(float(span.get("size", 10)))
                    if first_color is None:
                        first_color = rgb_from_int(int(span.get("color", 0)))
            text = normalize_text(" ".join(line_texts))
            if len(text) < 2:
                continue
            rect = fitz.Rect(block["bbox"])
            font_size = max(5.5, min(span_sizes or [10]))
            role = classify_text_unit(text, rect, font_size, page.rect)
            blocks_out.append(TextLine(page_index, rect, text, font_size, first_color or (0, 0, 0), role))
    return blocks_out


def is_probably_non_language(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    letters = re.findall(r"[A-Za-z\u4e00-\u9fff]", stripped)
    return len(letters) < 2


def chunk_lines(lines: list[TextLine], max_chars: int) -> list[list[TextLine]]:
    chunks: list[list[TextLine]] = []
    current: list[TextLine] = []
    size = 0
    for line in lines:
        line_size = len(line.text) + 1
        if current and size + line_size > max_chars:
            chunks.append(current)
            current = []
            size = 0
        current.append(line)
        size += line_size
    if current:
        chunks.append(current)
    return chunks


def parse_int(value: str | None, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(str(value or "").strip())
    except ValueError:
        parsed = default
    return max(minimum, min(maximum, parsed))


def parse_page_range(page_range: str, page_count: int) -> set[int] | None:
    value = (page_range or "").strip()
    if not value or value.lower() in {"all", "全部", "全部页面"}:
        return None
    first_match = re.fullmatch(r"(?:前|first:?)\s*(\d+)\s*(?:页|pages?)?", value, flags=re.IGNORECASE)
    if first_match:
        first_count = max(1, min(page_count, int(first_match.group(1))))
        return set(range(first_count))
    value = value.replace("~", "-").replace("～", "-").replace("—", "-").replace("–", "-")
    pages: set[int] = set()
    for part in re.split(r"[,，\s]+", value):
        if not part:
            continue
        try:
            if "-" in part:
                start_text, end_text = part.split("-", 1)
                if not start_text.strip() or not end_text.strip():
                    raise ValueError
                start = int(start_text.strip())
                end = int(end_text.strip())
                if start > end:
                    start, end = end, start
                pages.update(range(start, end + 1))
            else:
                pages.add(int(part.strip()))
        except ValueError as exc:
            raise ValueError("页码范围格式不正确，请使用类似 1-5,8,10-12 的格式。") from exc
    valid_pages = {page - 1 for page in pages if 1 <= page <= page_count}
    if not valid_pages:
        raise ValueError(f"页码范围没有匹配到 PDF 页面。当前 PDF 共 {page_count} 页。")
    return valid_pages


def format_page_range(page_filter: set[int] | None, page_count: int) -> str:
    if page_filter is None:
        return f"全部 {page_count} 页"
    ordered = sorted(page + 1 for page in page_filter)
    if len(ordered) <= 8:
        return "第 " + ",".join(str(page) for page in ordered) + " 页"
    return f"已选择 {len(ordered)}/{page_count} 页"


def parse_api_pool(config: dict[str, str]) -> list[dict[str, str]]:
    raw = (config.get("api_pool") or "").strip()
    if not raw:
        return [config]

    pool: list[dict[str, str]] = []
    base = {
        "source": config.get("source", "auto"),
        "target": config.get("target", "中文"),
        "prompt": config.get("prompt", ""),
        "timeout": config.get("timeout", "90"),
    }
    for line_number, line in enumerate(raw.splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) > 4:
            raise ValueError(f"API 池第 {line_number} 行格式不正确，请使用 类型|URL|Key|模型。")
        provider, api_url, api_key, model = (parts + ["", "", "", ""])[:4]
        provider = provider or config.get("provider", "deepseek")
        if provider not in OPENAI_COMPATIBLE_PROVIDERS | {"deepl", "libre", "custom", "mock"}:
            raise ValueError(f"API 池第 {line_number} 行的类型不支持：{provider}")
        item = dict(base)
        item.update(
            {
                "provider": provider,
                "api_url": api_url,
                "api_key": api_key,
                "model": model,
            }
        )
        pool.append(item)
    if not pool:
        raise ValueError("API 池为空，请至少填写一行 API 配置，或清空 API 池使用上方单个 API。")
    return pool


PROTECT_PATTERNS = [
    re.compile(r"https?://[^\s,;]+", re.IGNORECASE),
    re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE),
    re.compile(r"\[[0-9,\-\s;]+\]"),
    re.compile(r"\$[^$]+\$"),
    re.compile(r"\\[A-Za-z]+(?:\{[^{}]*\})?"),
    re.compile(r"\b(?:Fig|Figure|Table|Eq|Equation|Sec|Section)\.?\s*\d+(?:\.\d+)*\b", re.IGNORECASE),
    re.compile(r"\([A-Z][A-Za-z\-]+(?:\s+et\s+al\.)?,?\s*(?:19|20)\d{2}[a-z]?\)"),
]


def protect_fragments(text: str) -> tuple[str, dict[str, str]]:
    replacements: dict[str, str] = {}
    protected = text

    def replace(match: re.Match[str]) -> str:
        original = match.group(0)
        for key, value in replacements.items():
            if value == original:
                return key
        placeholder = f"@@PDFTR_{len(replacements)}@@"
        replacements[placeholder] = original
        return placeholder

    for pattern in PROTECT_PATTERNS:
        protected = pattern.sub(replace, protected)
    return protected, replacements


def restore_fragments(text: str, replacements: dict[str, str]) -> str:
    restored = text
    for placeholder, original in replacements.items():
        restored = restored.replace(placeholder, original)
    return restored


def cache_enabled(config: dict[str, str]) -> bool:
    return str(config.get("cache_enabled", "on")).lower() in {"1", "true", "yes", "on"}


def cache_key(translator: "Translator", text: str, mode: str) -> str:
    payload = {
        "provider": translator.provider,
        "api_url": translator.chat_url() if translator.provider in OPENAI_COMPATIBLE_PROVIDERS else translator.api_url,
        "model": translator.model or translator.default_model(),
        "source": translator.source,
        "target": translator.target,
        "prompt": translator.prompt,
        "mode": mode,
        "text": text,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class Translator:
    def __init__(self, config: dict[str, str]):
        self.provider = config.get("provider", "mock")
        self.api_url = config.get("api_url", "").strip()
        self.api_key = config.get("api_key", "").strip()
        self.model = config.get("model", "").strip()
        self.source = config.get("source", "auto").strip() or "auto"
        self.target = config.get("target", "中文").strip() or "中文"
        self.prompt = config.get("prompt", "").strip()
        self.timeout = float(config.get("timeout", "90") or 90)

    def translate_batch(self, texts: list[str]) -> list[str]:
        if self.provider == "mock":
            return [f"[{self.target}] {text}" for text in texts]
        if self.provider in OPENAI_COMPATIBLE_PROVIDERS:
            return self._translate_openai(texts)
        if self.provider == "deepl":
            return self._translate_deepl(texts)
        if self.provider == "libre":
            return self._translate_libre(texts)
        if self.provider == "custom":
            return [self._translate_custom(text) for text in texts]
        raise ValueError(f"不支持的 API 类型：{self.provider}")

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def request_timeout(self) -> tuple[float, float]:
        return (15.0, self.timeout)

    def _translate_openai(self, texts: list[str]) -> list[str]:
        url = self.chat_url()
        model = self.model or self.default_model()
        delimiter = "\n---PDF_TRANSLATOR_LINE---\n"
        joined = delimiter.join(texts)
        prompt = self.prompt or SYSTEM_PROMPT.format(target=self.target)
        payload = {
            "model": model,
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        f"Translate each segment from {self.source} to {self.target}. "
                        "Keep the same segment count and separate outputs with this exact delimiter: "
                        f"{delimiter!r} Preserve placeholders like @@PDFTR_0@@ exactly.\n\n{joined}"
                    ),
                },
            ],
        }
        response = requests.post(url, headers=self._headers(), json=payload, timeout=self.request_timeout())
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].strip()
        parts = [part.strip() for part in content.split(delimiter)]
        if len(parts) != len(texts):
            return [self._translate_openai([text])[0] for text in texts]
        return parts

    def chat_url(self) -> str:
        if self.provider == "deepseek":
            if self.api_url:
                return normalize_chat_url(self.api_url, self.provider)
            return DEEPSEEK_CHAT_URL
        if self.provider == "gemini":
            if self.api_url:
                return normalize_chat_url(self.api_url, self.provider)
            return GEMINI_CHAT_URL
        if self.api_url:
            return normalize_chat_url(self.api_url, self.provider)
        return "https://api.openai.com/v1/chat/completions"

    def default_model(self) -> str:
        if self.provider == "deepseek":
            return "deepseek-v4-flash"
        if self.provider == "gemini":
            return "gemini-2.5-flash"
        return "gpt-4.1-mini"

    def _translate_deepl(self, texts: list[str]) -> list[str]:
        url = self.api_url or "https://api-free.deepl.com/v2/translate"
        headers = {"Authorization": f"DeepL-Auth-Key {self.api_key}"}
        data: list[tuple[str, str]] = [("target_lang", self.target.upper())]
        if self.source.lower() != "auto":
            data.append(("source_lang", self.source.upper()))
        data.extend(("text", text) for text in texts)
        response = requests.post(url, headers=headers, data=data, timeout=self.request_timeout())
        response.raise_for_status()
        return [item["text"] for item in response.json()["translations"]]

    def _translate_libre(self, texts: list[str]) -> list[str]:
        url = self.api_url.rstrip("/") or "https://libretranslate.com/translate"
        results = []
        for text in texts:
            payload = {
                "q": text,
                "source": self.source if self.source.lower() != "auto" else "auto",
                "target": self.target,
                "format": "text",
            }
            if self.api_key:
                payload["api_key"] = self.api_key
            response = requests.post(url, json=payload, timeout=self.request_timeout())
            response.raise_for_status()
            results.append(response.json()["translatedText"])
        return results

    def _translate_custom(self, text: str) -> str:
        if not self.api_url:
            raise ValueError("Custom JSON 模式需要填写 API URL。")
        payload = {"text": text, "source": self.source, "target": self.target}
        if self.model:
            payload["model"] = self.model
        response = requests.post(self.api_url, headers=self._headers(), json=payload, timeout=self.request_timeout())
        response.raise_for_status()
        data = response.json()
        for key in ("translation", "translatedText", "translated_text", "text", "result", "output"):
            value = data.get(key)
            if isinstance(value, str):
                return value.strip()
        if isinstance(data.get("data"), dict):
            for key in ("translation", "translatedText", "text", "result"):
                value = data["data"].get(key)
                if isinstance(value, str):
                    return value.strip()
        raise ValueError("Custom JSON 返回中没有找到 translation / translatedText / text 等字段。")


def normalize_chat_url(api_url: str, provider: str) -> str:
    url = (api_url or "").strip().rstrip("/")
    if not url:
        if provider == "deepseek":
            return DEEPSEEK_CHAT_URL
        if provider == "gemini":
            return GEMINI_CHAT_URL
        return "https://api.openai.com/v1/chat/completions"
    if url.endswith("/chat/completions"):
        return url
    if provider == "deepseek":
        if url.endswith("/v1"):
            return url + "/chat/completions"
        if url == "https://api.deepseek.com":
            return DEEPSEEK_CHAT_URL
        return url + "/chat/completions"
    if provider == "gemini":
        if url.endswith("/v1beta/openai"):
            return url + "/chat/completions"
        if url == "https://generativelanguage.googleapis.com":
            return GEMINI_CHAT_URL
        return url + "/chat/completions"
    if url.endswith("/v1"):
        return url + "/chat/completions"
    return url


def openai_models_url(api_url: str) -> str:
    url = (api_url or "https://api.openai.com/v1/chat/completions").strip().split("?", 1)[0]
    url = url.rstrip("/")
    if "/v1/" in url:
        return url.split("/v1/", 1)[0] + "/v1/models"
    if url.endswith("/v1"):
        return url + "/models"
    if url.endswith("/chat/completions"):
        return url.removesuffix("/chat/completions") + "/models"
    return url + "/models"


def provider_label(provider: str) -> str:
    return {
        "openai": "OpenAI-compatible 模式",
        "deepseek": "DeepSeek 官方 API",
        "gemini": "Gemini 官方 API",
    }.get(provider, provider)


def request_error_message(exc: Exception) -> str:
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        detail = exc.response.text[:500]
        return f"HTTP {exc.response.status_code}: {detail}"
    if isinstance(exc, requests.Timeout):
        return "请求超时，请检查 API URL、网络或调大超时秒数。"
    if isinstance(exc, requests.ConnectionError):
        return "连接失败，请检查 API URL 是否可访问。"
    return str(exc)


def fetch_models(config: dict[str, str]) -> list[str]:
    translator = Translator(config)
    if translator.provider == "mock":
        return ["mock-layout-test"]
    if translator.provider in OPENAI_COMPATIBLE_PROVIDERS:
        if not translator.api_key:
            label = provider_label(translator.provider)
            raise ValueError(f"{label} 需要填写 API Key，页面里的 sk-... 只是占位符。")
        try:
            response = requests.get(
                openai_models_url(translator.chat_url()),
                headers=translator._headers(),
                timeout=translator.request_timeout(),
            )
            response.raise_for_status()
            data = response.json()
            models = data.get("data", data if isinstance(data, list) else [])
            ids = []
            for item in models:
                if isinstance(item, dict) and isinstance(item.get("id"), str):
                    ids.append(item["id"])
                elif isinstance(item, str):
                    ids.append(item)
            if ids:
                return sorted(set(ids))
        except requests.RequestException:
            if translator.provider not in {"deepseek", "gemini"}:
                raise
        if translator.provider == "deepseek":
            return DEEPSEEK_MODELS
        if translator.provider == "gemini":
            return GEMINI_MODELS
        return []
    if translator.provider == "deepl":
        return ["deepl-default"]
    if translator.provider == "libre":
        return ["libretranslate-default"]
    if translator.provider == "custom":
        return [translator.model] if translator.model else []
    return []


def check_connection(config: dict[str, str]) -> dict[str, Any]:
    translator = Translator(config)
    if translator.provider == "mock":
        return {"ok": True, "message": "Mock 模式可用，不会连接外部 API。"}

    if translator.provider in OPENAI_COMPATIBLE_PROVIDERS:
        if not translator.api_key:
            label = provider_label(translator.provider)
            raise ValueError(f"{label} 需要填写 API Key，页面里的 sk-... 只是占位符。")
        if translator.model:
            payload = {
                "model": translator.model,
                "temperature": 0,
                "max_tokens": 3,
                "messages": [
                    {"role": "system", "content": "Reply with OK only."},
                    {"role": "user", "content": "ping"},
                ],
            }
            response = requests.post(
                translator.chat_url(),
                headers=translator._headers(),
                json=payload,
                timeout=translator.request_timeout(),
            )
            response.raise_for_status()
            return {"ok": True, "message": f"模型 {translator.model} 连接成功。"}
        models = fetch_models(config)
        return {"ok": True, "message": f"API 可连接，已读取 {len(models)} 个模型。"}

    if translator.provider == "deepl":
        usage_url = (translator.api_url or "https://api-free.deepl.com/v2/translate").replace(
            "/translate", "/usage"
        )
        response = requests.get(
            usage_url,
            headers={"Authorization": f"DeepL-Auth-Key {translator.api_key}"},
            timeout=translator.request_timeout(),
        )
        response.raise_for_status()
        return {"ok": True, "message": "DeepL 连接成功。"}

    if translator.provider == "libre":
        url = (translator.api_url.rstrip("/") if translator.api_url else "https://libretranslate.com") + "/languages"
        response = requests.get(url, timeout=translator.request_timeout())
        response.raise_for_status()
        return {"ok": True, "message": "LibreTranslate 连接成功。"}

    if translator.provider == "custom":
        translator._translate_custom("ping")
        return {"ok": True, "message": "Custom JSON 接口连接成功。"}

    raise ValueError(f"不支持的 API 类型：{translator.provider}")


def fit_font_size(text: str, rect: fitz.Rect, original_size: float) -> float:
    if not text:
        return original_size
    width = max(rect.width, 1)
    height = max(rect.height, 1)
    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", text))
    latin_count = max(len(text) - cjk_count, 0)
    weighted_chars = cjk_count + latin_count * 0.55
    for size in [original_size, original_size * 0.92, original_size * 0.84, original_size * 0.76, original_size * 0.68, original_size * 0.6, original_size * 0.52]:
        size = max(4.2, size)
        chars_per_line = max(1, width / max(size * 0.56, 1))
        estimated_lines = max(1, weighted_chars / chars_per_line)
        if estimated_lines * size * 1.25 <= height:
            return size
    return max(4.2, min(original_size * 0.5, height * 0.45))


def role_font_size(line: TextLine, translated: str, rect: fitz.Rect) -> float:
    base = line.font_size
    if line.role == "title":
        base = max(base, 16)
    elif line.role == "heading":
        base = max(base, 12)
    elif line.role in {"caption", "reference"}:
        base = min(max(base, 7.2), 8.8)
    else:
        base = min(max(base, 9), 10.5)
    return fit_font_size(translated, rect, base)


def role_text_rect(line: TextLine, page: fitz.Page, layout_mode: str) -> fitz.Rect:
    if line.role == "caption":
        rect = line.rect + (0, -0.8, 0, 10.0)
        return rect & page.rect
    if line.role == "heading":
        rect = line.rect + (0, -1.2, 0, 3.0)
        return rect & page.rect
    rect = line.rect + (0, -1.0, 0, 4.0 if layout_mode == "block" else 1.5)
    return rect & page.rect


def role_redact_rect(line: TextLine) -> fitz.Rect:
    if line.role == "caption":
        return line.rect + (-0.5, -0.5, 0.5, 1.0)
    return line.rect + (-0.5, -0.5, 0.5, 0.5)


def write_translations(
    doc: fitz.Document,
    lines: list[TextLine],
    translations: dict[int, str],
    font_bundle: dict[str, str | None],
    layout_mode: str = "block",
) -> None:
    by_page: dict[int, list[tuple[int, TextLine]]] = {}
    for index, line in enumerate(lines):
        if index in translations:
            by_page.setdefault(line.page_index, []).append((index, line))

    for page_index, page_lines in by_page.items():
        page = doc[page_index]
        for _, line in page_lines:
            page.add_redact_annot(role_redact_rect(line), fill=(1, 1, 1))
        page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)

        for index, line in page_lines:
            translated = translations[index].strip()
            if not translated:
                continue
            rect = role_text_rect(line, page, layout_mode)
            font_size = role_font_size(line, translated, rect)
            font_file = font_bundle.get(line.role) or font_bundle.get("default")
            align = fitz.TEXT_ALIGN_CENTER if line.role == "title" else fitz.TEXT_ALIGN_LEFT
            page.insert_textbox(
                rect,
                translated,
                fontsize=font_size,
                fontname="PDFTranslatorFont" if font_file else "helv",
                fontfile=font_file,
                color=line.color,
                align=align,
            )


def update_job(job_id: str | None, **changes: Any) -> None:
    if not job_id:
        return
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        job.update(changes)
        job["updated_at"] = time.time()


def get_job(job_id: str) -> dict[str, Any] | None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        return dict(job) if job else None


def translate_texts_optimized(
    translator: Translator,
    texts: list[str],
    config: dict[str, str],
) -> tuple[list[str], int, int]:
    mode = config.get("translation_mode", "fast")
    use_cache = cache_enabled(config)
    results: list[str | None] = [None] * len(texts)
    cache_hits = 0
    protected_payloads: list[tuple[int, str, dict[str, str], str]] = []

    for index, text in enumerate(texts):
        key = cache_key(translator, text, mode)
        if use_cache:
            with CACHE_LOCK:
                cached = TRANSLATION_CACHE.get(key)
            if cached is not None:
                results[index] = cached
                cache_hits += 1
                continue

        if mode == "precise":
            protected, replacements = protect_fragments(text)
        else:
            protected, replacements = text, {}
        protected_payloads.append((index, protected, replacements, key))

    if protected_payloads:
        translated = translator.translate_batch([item[1] for item in protected_payloads])
        if len(translated) != len(protected_payloads):
            raise ValueError("翻译 API 返回的条数和请求条数不一致。")
        for (index, _, replacements, key), item in zip(protected_payloads, translated):
            restored = restore_fragments(item, replacements) if mode == "precise" else item
            results[index] = restored
            if use_cache:
                with CACHE_LOCK:
                    TRANSLATION_CACHE[key] = restored

    return [item or "" for item in results], cache_hits, len(protected_payloads)


def translate_chunk_with_retry(
    translators: list[Translator],
    texts: list[str],
    chunk_index: int,
    chunk_count: int,
    retries: int,
    job_id: str | None,
    config: dict[str, str],
) -> tuple[list[str], int]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 2):
        translator = translators[(chunk_index - 1 + attempt - 1) % len(translators)]
        api_label = translator.provider
        if translator.model:
            api_label += f"/{translator.model}"
        try:
            retry_note = "" if attempt == 1 else f"，第 {attempt}/{retries + 1} 次尝试"
            update_job(job_id, stage=f"正在翻译第 {chunk_index}/{chunk_count} 批，使用 {api_label}{retry_note}")
            translated, cache_hits, requested = translate_texts_optimized(translator, texts, config)
            if cache_hits:
                update_job(job_id, stage=f"第 {chunk_index}/{chunk_count} 批命中缓存 {cache_hits} 行，请求 {requested} 行")
            if len(translated) != len(texts):
                raise ValueError("翻译 API 返回的条数和请求条数不一致。")
            return translated, cache_hits
        except Exception as exc:
            last_error = exc
            if attempt > retries:
                break
            update_job(job_id, stage=f"第 {chunk_index}/{chunk_count} 批在 {api_label} 失败，准备重试 {attempt}/{retries}")
            time.sleep(min(2 * attempt, 8))
    assert last_error is not None
    raise last_error


def translate_pdf(
    input_path: Path,
    output_path: Path,
    config: dict[str, str],
    job_id: str | None = None,
) -> dict[str, Any]:
    translator_configs = parse_api_pool(config)
    translators = [Translator(item) for item in translator_configs]
    font_choice = config.get("font_choice", "academic")
    if font_choice not in FONT_OPTIONS:
        font_choice = "academic"
    font_bundle, font_label = pick_font_bundle(font_choice)
    update_job(job_id, progress=5, stage="正在打开 PDF")
    doc = fitz.open(input_path)
    try:
        page_filter = parse_page_range(config.get("page_range", ""), doc.page_count)
        selected_pages = format_page_range(page_filter, doc.page_count)
        concurrency = parse_int(config.get("concurrency"), default=2, minimum=1, maximum=8)
        retries = parse_int(config.get("retries"), default=2, minimum=0, maximum=5)
        chunk_chars = parse_int(config.get("chunk_chars"), default=2500, minimum=200, maximum=8000)
        translation_mode = config.get("translation_mode", "fast")
        if translation_mode not in {"fast", "precise"}:
            translation_mode = "fast"
        layout_mode = config.get("layout_mode", "block")
        if layout_mode not in {"block", "line"}:
            layout_mode = "block"
        unit_label = "文本块" if layout_mode == "block" else "文字行"
        use_cache = cache_enabled(config)
        update_job(job_id, progress=10, stage="正在解析 PDF 文本")
        extracted_units = extract_blocks(doc, page_filter) if layout_mode == "block" else extract_lines(doc, page_filter)
        candidate_lines = [
            line
            for line in extracted_units
            if not is_probably_non_language(line.text)
        ]
        skipped_units = [
            line
            for line in candidate_lines
            if not should_translate_unit(line)
        ]
        lines = [
            line
            for line in candidate_lines
            if should_translate_unit(line)
        ]
        if not lines:
            if skipped_units:
                update_job(
                    job_id,
                    progress=96,
                    stage=f"所选页面只有图表/公式/页眉内容，已保护 {len(skipped_units)} 个文本块",
                    line_count=0,
                    translated_count=0,
                    chunk_count=0,
                    translated_chunks=0,
                    concurrency=concurrency,
                    retries=retries,
                    page_range=selected_pages,
                    api_count=len(translators),
                    translation_mode=translation_mode,
                    cache_enabled=use_cache,
                    layout_mode=layout_mode,
                    unit_label=unit_label,
                    font_choice=font_choice,
                    font_label=font_label,
                    preserve_color=True,
                    skipped_units=len(skipped_units),
                )
                doc.save(output_path, garbage=4, deflate=True, deflate_fonts=True, use_objstms=True)
                return {
                    "line_count": 0,
                    "translated_count": 0,
                    "font": font_label,
                    "font_file": font_bundle.get("default") or "built-in Helvetica",
                    "font_choice": font_choice,
                    "preserve_color": True,
                    "skipped_units": len(skipped_units),
                    "page_range": selected_pages,
                    "chunk_count": 0,
                    "concurrency": concurrency,
                    "retries": retries,
                    "api_count": len(translators),
                    "translation_mode": translation_mode,
                    "cache_enabled": use_cache,
                    "cache_hits": 0,
                    "layout_mode": layout_mode,
                    "unit_label": unit_label,
                }
            raise ValueError("没有提取到可翻译文字。这个 PDF 可能是扫描版，需要先 OCR。")

        translations: dict[int, str] = {}
        chunks = chunk_lines(lines, chunk_chars)
        indexed_chunks: list[tuple[int, int, list[TextLine]]] = []
        start = 0
        for chunk_index, chunk in enumerate(chunks, start=1):
            indexed_chunks.append((chunk_index, start, chunk))
            start += len(chunk)
        update_job(
            job_id,
            progress=15,
            stage=f"已解析 {selected_pages}、{len(lines)} 个{unit_label}，准备使用 {len(translators)} 组 API 并发翻译",
            line_count=len(lines),
            translated_count=0,
            chunk_count=len(chunks),
            translated_chunks=0,
            concurrency=concurrency,
            retries=retries,
            page_range=selected_pages,
            api_count=len(translators),
            translation_mode=translation_mode,
            cache_enabled=use_cache,
            layout_mode=layout_mode,
            unit_label=unit_label,
            font_choice=font_choice,
            font_label=font_label,
            preserve_color=True,
            skipped_units=len(skipped_units),
        )
        processed_lines = 0
        completed_chunks = 0
        total_cache_hits = 0
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(
                    translate_chunk_with_retry,
                    translators,
                    [line.text for line in chunk],
                    chunk_index,
                    len(chunks),
                    retries,
                    job_id,
                    config,
                ): (chunk_index, start_index, chunk)
                for chunk_index, start_index, chunk in indexed_chunks
            }
            for future in as_completed(futures):
                chunk_index, start_index, chunk = futures[future]
                translated, cache_hits = future.result()
                total_cache_hits += cache_hits
                for offset, item in enumerate(translated):
                    translations[start_index + offset] = item
                processed_lines += len(chunk)
                completed_chunks += 1
                progress = 15 + int(70 * completed_chunks / max(len(chunks), 1))
                update_job(
                    job_id,
                    progress=progress,
                    translated_count=processed_lines,
                    translated_chunks=completed_chunks,
                    cache_hits=total_cache_hits,
                    stage=f"已完成 {completed_chunks}/{len(chunks)} 批，并发数 {concurrency}，API {len(translators)} 组",
                )

        update_job(job_id, progress=90, stage="正在写回译文并保持版式")
        write_translations(doc, lines, translations, font_bundle, layout_mode)
        update_job(job_id, progress=96, stage="正在保存译文 PDF")
        doc.save(output_path, garbage=4, deflate=True, deflate_fonts=True, use_objstms=True)
        return {
            "line_count": len(lines),
            "translated_count": len(translations),
            "font": font_label,
            "font_file": font_bundle.get("default") or "built-in Helvetica",
            "font_choice": font_choice,
            "preserve_color": True,
            "skipped_units": len(skipped_units),
            "page_range": selected_pages,
            "chunk_count": len(chunks),
            "concurrency": concurrency,
            "retries": retries,
            "api_count": len(translators),
            "translation_mode": translation_mode,
            "cache_enabled": use_cache,
            "cache_hits": total_cache_hits,
            "layout_mode": layout_mode,
            "unit_label": unit_label,
        }
    finally:
        doc.close()


def run_translation_job(job_id: str, input_path: Path, output_path: Path, config: dict[str, str]) -> None:
    started = time.time()
    try:
        stats = translate_pdf(input_path, output_path, config, job_id)
        stats["seconds"] = round(time.time() - started, 2)
        update_job(
            job_id,
            ok=True,
            status="done",
            progress=100,
            stage="翻译完成",
            download_url=f"/download/{quote(output_path.name)}",
            preview_url=f"/preview/{quote(output_path.name)}",
            filename=output_path.name,
            output_path=str(output_path),
            stats=stats,
        )
    except Exception as exc:
        update_job(job_id, ok=False, status="error", error=request_error_message(exc), stage="翻译失败")


class Handler(BaseHTTPRequestHandler):
    server_version = "PaperPDFTranslator/1.0"

    def do_HEAD(self) -> None:
        if self.path in ("/", "/index.html"):
            self.send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8", send_body=False)
            return
        if self.path == "/static/style.css":
            self.send_file(STATIC_DIR / "style.css", "text/css; charset=utf-8", send_body=False)
            return
        if self.path.startswith("/download/"):
            filename = Path(unquote(self.path.removeprefix("/download/"))).name
            target = OUTPUT_DIR / filename
            if not target.exists():
                self.send_json({"error": "文件不存在或已被清理。"}, HTTPStatus.NOT_FOUND)
                return
            self.send_file(target, "application/pdf", attachment_name=filename, send_body=False)
            return
        if self.path.startswith("/preview/"):
            filename = Path(unquote(self.path.removeprefix("/preview/"))).name
            target = OUTPUT_DIR / filename
            if not target.exists():
                self.send_json({"error": "文件不存在或已被清理。"}, HTTPStatus.NOT_FOUND)
                return
            self.send_file(target, "application/pdf", inline_name=filename, send_body=False)
            return
        self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self.send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
            return
        if self.path == "/static/style.css":
            self.send_file(STATIC_DIR / "style.css", "text/css; charset=utf-8")
            return
        if self.path.startswith("/download/"):
            filename = Path(unquote(self.path.removeprefix("/download/"))).name
            target = OUTPUT_DIR / filename
            if not target.exists():
                self.send_json({"error": "文件不存在或已被清理。"}, HTTPStatus.NOT_FOUND)
                return
            self.send_file(target, "application/pdf", attachment_name=filename)
            return
        if self.path.startswith("/preview/"):
            filename = Path(unquote(self.path.removeprefix("/preview/"))).name
            target = OUTPUT_DIR / filename
            if not target.exists():
                self.send_json({"error": "文件不存在或已被清理。"}, HTTPStatus.NOT_FOUND)
                return
            self.send_file(target, "application/pdf", inline_name=filename)
            return
        if self.path.startswith("/api/jobs/"):
            job_id = Path(self.path.removeprefix("/api/jobs/")).name
            job = get_job(job_id)
            if not job:
                self.send_json({"ok": False, "error": "任务不存在或已被清理。"}, HTTPStatus.NOT_FOUND)
                return
            self.send_json(job)
            return
        self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if self.path == "/api/models":
            self.handle_models()
            return
        if self.path == "/api/check":
            self.handle_check()
            return
        if self.path != "/api/translate":
            self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("content-length", "0"))
            if length > MAX_FORM_BYTES:
                raise ValueError("PDF 文件最大支持 50MB。")
            form = parse_multipart({k.lower(): v for k, v in self.headers.items()}, self.rfile.read(length))
            upload = form.get("pdf")
            if not upload or not upload["content"]:
                raise ValueError("请上传 PDF 文件。")
            if len(upload["content"]) > MAX_UPLOAD_BYTES:
                raise ValueError("PDF 文件最大支持 50MB。")
            if upload["content_type"] != "application/pdf" and not upload["filename"].lower().endswith(".pdf"):
                raise ValueError("只支持 PDF 文件。")

            job_id = uuid.uuid4().hex
            input_path = UPLOAD_DIR / f"{job_id}.pdf"
            output_path = OUTPUT_DIR / f"{Path(upload['filename']).stem}-translated-{job_id[:8]}.pdf"
            input_path.write_bytes(upload["content"])

            config = {key: str(value) for key, value in form.items() if key != "pdf"}
            with JOBS_LOCK:
                JOBS[job_id] = {
                    "ok": True,
                    "job_id": job_id,
                    "status": "queued",
                    "progress": 1,
                    "stage": "任务已创建，等待开始翻译",
                    "created_at": time.time(),
                    "updated_at": time.time(),
                }
            worker = Thread(target=run_translation_job, args=(job_id, input_path, output_path, config), daemon=True)
            worker.start()
            self.send_json({"ok": True, "job_id": job_id, "status_url": f"/api/jobs/{job_id}"}, HTTPStatus.ACCEPTED)
        except Exception as exc:
            self.send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def handle_models(self) -> None:
        try:
            config = self.read_json_body()
            models = fetch_models(config)
            self.send_json({"ok": True, "models": models, "count": len(models)})
        except Exception as exc:
            self.send_json({"ok": False, "error": request_error_message(exc)}, HTTPStatus.BAD_REQUEST)

    def handle_check(self) -> None:
        try:
            config = self.read_json_body()
            result = check_connection(config)
            self.send_json(result)
        except Exception as exc:
            self.send_json(
                {"ok": False, "message": "连接失败。", "error": request_error_message(exc)},
                HTTPStatus.BAD_REQUEST,
            )

    def read_json_body(self) -> dict[str, str]:
        length = int(self.headers.get("content-length", "0"))
        if length > 1024 * 1024:
            raise ValueError("请求体过大。")
        raw = self.rfile.read(length)
        if not raw:
            return {}
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("JSON 请求体必须是对象。")
        return {str(key): str(value) for key, value in data.items() if value is not None}

    def send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_file(
        self,
        path: Path,
        content_type: str,
        attachment_name: str | None = None,
        inline_name: str | None = None,
        send_body: bool = True,
    ) -> None:
        file_size = path.stat().st_size
        start = 0
        end = file_size - 1
        status = HTTPStatus.OK
        range_header = self.headers.get("Range")
        if range_header:
            match = re.match(r"bytes=(\d*)-(\d*)", range_header)
            if match:
                if match.group(1):
                    start = int(match.group(1))
                if match.group(2):
                    end = int(match.group(2))
                if not match.group(1) and match.group(2):
                    suffix = int(match.group(2))
                    start = max(file_size - suffix, 0)
                    end = file_size - 1
                if start >= file_size or end < start:
                    self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    self.send_header("Content-Range", f"bytes */{file_size}")
                    self.end_headers()
                    return
                end = min(end, file_size - 1)
                status = HTTPStatus.PARTIAL_CONTENT

        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        if attachment_name:
            encoded = quote(attachment_name)
            self.send_header("Content-Disposition", f"attachment; filename*=UTF-8''{encoded}")
        if inline_name:
            encoded = quote(inline_name)
            self.send_header("Content-Disposition", f"inline; filename*=UTF-8''{encoded}")
        self.end_headers()
        if not send_body:
            return
        try:
            with path.open("rb") as file:
                file.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = file.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            return

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}")


def write_static_files() -> None:
    index = STATIC_DIR / "index.html"
    css = STATIC_DIR / "style.css"
    if not index.exists():
        index.write_text(INDEX_HTML, encoding="utf-8")
    if not css.exists():
        css.write_text(STYLE_CSS, encoding="utf-8")


def cleanup_old_files() -> None:
    cutoff = time.time() - 24 * 60 * 60
    for folder in (UPLOAD_DIR, OUTPUT_DIR):
        for path in folder.glob("*"):
            if path.is_file() and path.stat().st_mtime < cutoff:
                path.unlink(missing_ok=True)


INDEX_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>论文 PDF 翻译器</title>
  <link rel="stylesheet" href="/static/style.css">
</head>
<body>
  <main class="shell">
    <section class="workspace">
      <header class="topbar">
        <div>
          <h1>论文 PDF 翻译器</h1>
          <p>上传 PDF，调用官方或第三方 API，原位替换文字并保留页面结构。</p>
        </div>
        <a class="ghost" href="https://platform.openai.com/docs" target="_blank" rel="noreferrer">API 文档</a>
      </header>

      <form id="translatorForm" class="panel">
        <label class="dropzone" id="dropzone">
          <input id="pdf" name="pdf" type="file" accept="application/pdf" required>
          <span class="drop-title">选择或拖入 PDF</span>
          <span class="drop-meta" id="fileMeta">未选择文件</span>
        </label>

        <div class="grid">
          <label>
            <span>API 类型</span>
            <select name="provider" id="provider">
              <option value="openai">OpenAI-compatible</option>
              <option value="gemini">Gemini 官方 API</option>
              <option value="deepl">DeepL</option>
              <option value="libre">LibreTranslate</option>
              <option value="custom">Custom JSON</option>
              <option value="mock">Mock 版式测试</option>
            </select>
          </label>
          <label>
            <span>API URL</span>
            <input name="api_url" id="apiUrl" placeholder="https://api.openai.com/v1/chat/completions">
          </label>
          <label>
            <span>API Key</span>
            <input name="api_key" type="password" autocomplete="off" placeholder="sk-...">
          </label>
          <label>
            <span>模型 / 引擎</span>
            <input name="model" id="model" placeholder="gpt-4.1-mini">
          </label>
          <label>
            <span>源语言</span>
            <input name="source" value="auto">
          </label>
          <label>
            <span>目标语言</span>
            <input name="target" value="中文">
          </label>
          <label>
            <span>每批字符数</span>
            <input name="chunk_chars" type="number" min="200" max="8000" value="2500">
          </label>
          <label>
            <span>超时秒数</span>
            <input name="timeout" type="number" min="10" max="300" value="90">
          </label>
        </div>

        <label class="prompt">
          <span>自定义翻译提示词</span>
          <textarea name="prompt" rows="4" placeholder="可选。OpenAI-compatible 模式会使用它作为 system prompt。"></textarea>
        </label>

        <div class="actions">
          <button type="submit" id="submitBtn">
            <span class="icon">↻</span>
            开始翻译
          </button>
          <a id="downloadBtn" class="download hidden" href="#">下载译文 PDF</a>
        </div>
      </form>

      <section class="status" id="status">
        <div class="status-dot"></div>
        <p>准备就绪。第一次建议用 Mock 模式验证 PDF 是否能正确提取文字。</p>
      </section>
    </section>
  </main>

  <script>
    const form = document.getElementById('translatorForm');
    const pdfInput = document.getElementById('pdf');
    const fileMeta = document.getElementById('fileMeta');
    const statusBox = document.getElementById('status');
    const submitBtn = document.getElementById('submitBtn');
    const downloadBtn = document.getElementById('downloadBtn');
    const provider = document.getElementById('provider');
    const apiUrl = document.getElementById('apiUrl');
    const model = document.getElementById('model');

    const defaults = {
      openai: ['https://api.openai.com/v1/chat/completions', 'gpt-4.1-mini'],
      gemini: ['https://generativelanguage.googleapis.com/v1beta/openai/chat/completions', 'gemini-2.5-flash'],
      deepl: ['https://api-free.deepl.com/v2/translate', ''],
      libre: ['https://libretranslate.com/translate', ''],
      custom: ['https://example.com/translate', ''],
      mock: ['', '']
    };

    provider.addEventListener('change', () => {
      const [url, modelName] = defaults[provider.value];
      apiUrl.placeholder = url;
      if (!apiUrl.value || Object.values(defaults).some(([item]) => item === apiUrl.value)) apiUrl.value = url;
      model.placeholder = modelName;
    });

    pdfInput.addEventListener('change', () => {
      const file = pdfInput.files[0];
      fileMeta.textContent = file ? `${file.name} · ${(file.size / 1024 / 1024).toFixed(2)} MB` : '未选择文件';
    });

    function setStatus(text, tone = 'idle') {
      statusBox.className = `status ${tone}`;
      statusBox.querySelector('p').textContent = text;
    }

    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      downloadBtn.classList.add('hidden');
      submitBtn.disabled = true;
      setStatus('正在翻译并重写 PDF。大文件或外部 API 较慢时需要等一会儿。', 'working');

      try {
        const response = await fetch('/api/translate', {
          method: 'POST',
          body: new FormData(form)
        });
        const data = await response.json();
        if (!response.ok || !data.ok) throw new Error(data.error || '翻译失败');
        downloadBtn.href = data.download_url;
        downloadBtn.download = data.filename;
        downloadBtn.classList.remove('hidden');
        const stats = data.stats;
        setStatus(`完成：翻译 ${stats.translated_count}/${stats.line_count} 行，用时 ${stats.seconds}s，字体 ${stats.font}。`, 'done');
      } catch (error) {
        setStatus(error.message, 'error');
      } finally {
        submitBtn.disabled = false;
      }
    });
  </script>
</body>
</html>
"""


STYLE_CSS = """* {
  box-sizing: border-box;
}

body {
  margin: 0;
  min-height: 100vh;
  font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: #16201c;
  background: #f5f7f3;
}

.shell {
  width: min(1120px, calc(100vw - 32px));
  margin: 0 auto;
  padding: 32px 0;
}

.workspace {
  display: grid;
  gap: 18px;
}

.topbar {
  display: flex;
  align-items: end;
  justify-content: space-between;
  gap: 16px;
  padding: 10px 0 4px;
}

h1 {
  margin: 0;
  font-size: clamp(28px, 4vw, 44px);
  line-height: 1.05;
}

p {
  margin: 8px 0 0;
  color: #52615a;
}

.ghost,
.download,
button {
  min-height: 44px;
  border-radius: 8px;
  border: 1px solid #1d4d3a;
  padding: 0 16px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  font-weight: 700;
  text-decoration: none;
  white-space: nowrap;
}

.ghost {
  color: #1d4d3a;
  background: transparent;
}

.panel {
  display: grid;
  gap: 18px;
  padding: 20px;
  border: 1px solid #d9e0d8;
  border-radius: 8px;
  background: #ffffff;
  box-shadow: 0 16px 42px rgba(32, 45, 39, 0.08);
}

.dropzone {
  min-height: 138px;
  border: 1px dashed #87a091;
  border-radius: 8px;
  display: grid;
  place-items: center;
  gap: 8px;
  padding: 24px;
  background: #fbfcfa;
  cursor: pointer;
}

.dropzone input {
  position: absolute;
  opacity: 0;
  pointer-events: none;
}

.drop-title {
  font-size: 22px;
  font-weight: 800;
}

.drop-meta {
  color: #69776f;
}

.grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 14px;
}

label {
  display: grid;
  gap: 7px;
  font-weight: 700;
  color: #2a3731;
}

label span {
  font-size: 13px;
}

input,
select,
textarea {
  width: 100%;
  border: 1px solid #cbd5ce;
  border-radius: 8px;
  padding: 11px 12px;
  font: inherit;
  color: #17231d;
  background: #fff;
}

textarea {
  resize: vertical;
  min-height: 104px;
}

.prompt {
  font-weight: 700;
}

.actions {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}

button {
  background: #1d4d3a;
  color: #ffffff;
  cursor: pointer;
}

button:disabled {
  opacity: 0.55;
  cursor: wait;
}

.download {
  background: #e1f2dd;
  color: #173c2d;
}

.hidden {
  display: none;
}

.status {
  display: flex;
  align-items: flex-start;
  gap: 10px;
  padding: 14px 16px;
  border-radius: 8px;
  border: 1px solid #d9e0d8;
  background: #ffffff;
}

.status p {
  margin: 0;
}

.status-dot {
  width: 10px;
  height: 10px;
  margin-top: 5px;
  border-radius: 999px;
  background: #6f8077;
  flex: 0 0 auto;
}

.status.working .status-dot {
  background: #bd7d20;
}

.status.done .status-dot {
  background: #237a43;
}

.status.error .status-dot {
  background: #c73838;
}

@media (max-width: 860px) {
  .topbar {
    align-items: flex-start;
    flex-direction: column;
  }

  .grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 560px) {
  .shell {
    width: min(100vw - 20px, 1120px);
    padding: 16px 0;
  }

  .panel {
    padding: 14px;
  }

  .grid {
    grid-template-columns: 1fr;
  }

  .ghost,
  .download,
  button {
    width: 100%;
  }
}
"""


def main() -> None:
    ensure_dirs()
    cleanup_old_files()
    write_static_files()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"PDF translator is running at http://{HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
