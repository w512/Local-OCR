"""Service layer for Local OCR: validation, PDF rendering, Ollama, saving.

This module must stay free of Tk imports so every function can be tested
headlessly and no worker can accidentally touch the GUI.
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

import ollama
import openai
import pymupdf

import config

LogCallback = Callable[[str], None]
ProgressCallback = Callable[[str, int, int], None]  # phase, current, total
EventCallback = Callable[[str, dict], None]  # (kind, payload)


class OCRServiceError(Exception):
    """A service operation (Ollama, PDF rendering, saving) failed."""


@dataclass(frozen=True)
class OCRRequest:
    """Immutable snapshot of everything an OCR worker needs."""

    input_path: Path
    output_path: Path
    provider: config.Provider
    server_url: str
    model: str
    dpi: int


def normalize_ollama_url(value: str) -> str:
    """Validate a user-entered Ollama base URL and return it normalized.

    Keeps any path prefix so reverse-proxy URLs work; never appends /api
    because the official client handles API paths itself.
    """
    url = value.strip().rstrip("/")
    if not url:
        raise ValueError("Ollama server URL is empty.")
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            "Ollama server URL must start with http:// or https:// "
            f"(got: {value.strip()!r})."
        )
    if not parsed.netloc:
        raise ValueError(f"Ollama server URL has no host: {value.strip()!r}.")
    return url


def normalize_server_url(value: str) -> str:
    """Validate a user-entered server base URL and return it normalized.

    Works for both Ollama and LM Studio URLs. Keeps any path prefix so
    reverse-proxy URLs work; never appends /api because the respective
    clients handle API paths themselves.
    """
    url = value.strip().rstrip("/")
    if not url:
        raise ValueError("Server URL is empty.")
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            "Server URL must start with http:// or https:// "
            f"(got: {value.strip()!r})."
        )
    if not parsed.netloc:
        raise ValueError(f"Server URL has no host: {value.strip()!r}.")
    return url


def normalize_ollama_url(value: str) -> str:
    """Validate a user-entered Ollama base URL and return it normalized.

    Keeps any path prefix so reverse-proxy URLs work; never appends /api
    because the official client handles API paths itself.
    """
    return normalize_server_url(value)


def validate_input_path(path: Path) -> None:
    """Raise ValueError unless path is a readable, supported document."""
    if not path.exists():
        raise ValueError(f"File does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Not a regular file: {path}")
    if path.suffix.lower() not in config.SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(config.SUPPORTED_EXTENSIONS))
        raise ValueError(
            f"Unsupported file type {path.suffix!r}. Supported: {supported}"
        )
    if not os.access(path, os.R_OK):
        raise ValueError(f"File is not readable: {path}")


def build_output_path(input_path: Path) -> Path:
    """Return /dir/document_extracted.md for /dir/document.<ext>."""
    return input_path.with_name(f"{input_path.stem}_extracted.md")


def list_models(url: str) -> list[str]:
    """Fetch model tags from an Ollama server, deduplicated and sorted."""
    try:
        client = ollama.Client(host=url, timeout=config.MODEL_LIST_TIMEOUT)
        response = client.list()
        tags = {
            (getattr(item, "model", None) or "").strip()
            for item in response.models
        }
    except Exception as exc:
        raise OCRServiceError(f"Could not fetch models from {url}: {exc}") from exc
    tags.discard("")
    return sorted(tags, key=str.lower)


def encode_image_as_base64(image_path: Path) -> str:
    """Read an image file and return its contents as a base64 data URL.

    Tk-free so it can be unit-tested headlessly.
    """
    import base64

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def list_openai_compatible_models(url: str, provider_name: str = "server") -> list[str]:
    """Fetch model IDs from an OpenAI-compatible server (LM Studio, vLLM).

    Both LM Studio and vLLM expose an OpenAI-compatible ``/v1/models``
    endpoint. The ``openai`` library is used with a custom base_url so the
    request goes to the local server rather than OpenAI's API.
    """
    try:
        client = openai.OpenAI(
            base_url=url.rstrip("/") + "/v1",
            api_key="lm-studio",  # local servers ignore the key; a non-empty value is required
            timeout=config.MODEL_LIST_TIMEOUT,
        )
        response = client.models.list()
        tags = {
            (getattr(item, "id", None) or "").strip()
            for item in response.data
        }
    except Exception as exc:
        raise OCRServiceError(
            f"Could not fetch models from {provider_name} at {url}: {exc}"
        ) from exc
    tags.discard("")
    return sorted(tags, key=str.lower)


def list_lm_studio_models(url: str) -> list[str]:
    """Fetch model IDs from an LM Studio server."""
    return list_openai_compatible_models(url, "LM Studio")


def list_vllm_models(url: str) -> list[str]:
    """Fetch model IDs from a vLLM server."""
    return list_openai_compatible_models(url, "vLLM")


def render_pdf(
    pdf_path: Path,
    dpi: int,
    temp_dir: Path,
    log_callback: LogCallback,
    progress_callback: ProgressCallback | None = None,
) -> list[Path]:
    """Render every PDF page to a PNG in temp_dir, in original page order."""
    try:
        document = pymupdf.open(pdf_path)
    except Exception as exc:
        raise OCRServiceError(f"Could not open PDF: {exc}") from exc
    with document:
        if document.needs_pass:
            raise OCRServiceError(
                "PDF is password-protected; encrypted documents are not supported."
            )
        page_count = document.page_count
        if page_count == 0:
            raise OCRServiceError("PDF contains no pages.")
        image_paths: list[Path] = []
        for index in range(page_count):
            page_number = index + 1
            if progress_callback is not None:
                progress_callback("render", page_number, page_count)
            log_callback(f"Rendering page {page_number}/{page_count}...")
            try:
                page = document.load_page(index)
                pixmap = page.get_pixmap(
                    dpi=dpi, colorspace=pymupdf.csRGB, alpha=False
                )
                image_path = temp_dir / f"page_{page_number:04d}.png"
                pixmap.save(str(image_path))
            except Exception as exc:
                raise OCRServiceError(
                    f"Failed to render page {page_number}/{page_count}: {exc}"
                ) from exc
            image_paths.append(image_path)
    return image_paths


def make_thumbnail_png(image_path: Path, max_side: int) -> bytes:
    """Return a downscaled PNG of an image file, longest side <= max_side.

    Tk-free (PIL only) so it can be unit-tested headlessly. Accepts any
    format PIL can read — the input images (PNG/JPEG/WebP) and the PNGs
    rendered from PDF pages. Smaller images are never upscaled.
    """
    from PIL import Image  # local import keeps module load light and headless

    with Image.open(image_path) as img:
        thumbnail = img.convert("RGB")
        thumbnail.thumbnail((max_side, max_side))
        buffer = io.BytesIO()
        thumbnail.save(buffer, format="PNG")
        return buffer.getvalue()


def recognize_images(
    client: "ollama.Client",
    model: str,
    image_paths: list[Path],
    log_callback: LogCallback,
    progress_callback: ProgressCallback | None = None,
    event_callback: EventCallback | None = None,
) -> list[str]:
    """Send one independent chat request per image; return texts in order.

    Each page is requested with ``stream=True`` so that the recognized text
    appears in the UI as the model generates it.  Every chunk delta is
    surfaced via ``("stream_chunk", {"page": n, "text": delta})`` events;
    after the full page text is assembled and validated a final
    ``("page_text", {"page": n, "total": N, "text": text})`` event is
    emitted.  The non-streaming result is identical — streaming only adds
    the live deltas.
    """
    total = len(image_paths)
    results: list[str] = []
    for number, image_path in enumerate(image_paths, start=1):
        if progress_callback is not None:
            progress_callback("ocr", number, total)
        if event_callback is not None:
            # A failed preview must never abort OCR: log it and move on.
            try:
                png = make_thumbnail_png(image_path, config.THUMBNAIL_MAX_SIDE)
            except Exception as exc:
                log_callback(
                    f"Could not build preview for page {number}/{total}: {exc}"
                )
            else:
                event_callback(
                    "page_image",
                    {"page": number, "total": total, "png": png},
                )
        log_callback(f"Sending page {number}/{total} to Ollama...")
        chunks: list[str] = []
        try:
            stream = client.chat(
                model=model,
                messages=[
                    {"role": "system", "content": config.SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": config.USER_PROMPT,
                        "images": [str(image_path)],
                    },
                ],
                stream=True,
            )
            for chunk in stream:
                delta = (getattr(getattr(chunk, "message", None), "content", None)
                         or "")
                if delta:
                    chunks.append(delta)
                    if event_callback is not None:
                        event_callback(
                            "stream_chunk",
                            {"page": number, "text": delta},
                        )
        except Exception as exc:
            raise OCRServiceError(
                f"Ollama request failed on page {number}/{total} "
                f"(model {model!r}): {exc}"
            ) from exc
        content = "".join(chunks).strip()
        if not content:
            raise OCRServiceError(
                f"Ollama returned no text for page {number}/{total} "
                f"(model {model!r})."
            )
        results.append(content)
        if event_callback is not None:
            event_callback(
                "page_text",
                {"page": number, "total": total, "text": content},
            )
    return results


def recognize_images_openai(
    client: "openai.OpenAI",
    model: str,
    image_paths: list[Path],
    log_callback: LogCallback,
    progress_callback: ProgressCallback | None = None,
    event_callback: EventCallback | None = None,
    provider_name: str = "LM Studio",
) -> list[str]:
    """Send one independent chat request per image to an OpenAI-compatible server.

    Both LM Studio and vLLM expose an OpenAI-compatible chat completions
    endpoint. Images are base64-encoded and sent as ``image_url`` content
    parts. Streaming works the same way as the Ollama path: each delta is
    emitted as a ``stream_chunk`` event, and a final ``page_text`` event
    follows.
    """
    total = len(image_paths)
    results: list[str] = []
    for number, image_path in enumerate(image_paths, start=1):
        if progress_callback is not None:
            progress_callback("ocr", number, total)
        if event_callback is not None:
            try:
                png = make_thumbnail_png(image_path, config.THUMBNAIL_MAX_SIDE)
            except Exception as exc:
                log_callback(
                    f"Could not build preview for page {number}/{total}: {exc}"
                )
            else:
                event_callback(
                    "page_image",
                    {"page": number, "total": total, "png": png},
                )
        log_callback(f"Sending page {number}/{total} to {provider_name}...")
        chunks: list[str] = []
        try:
            data_url = encode_image_as_base64(image_path)
            stream = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": config.SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": config.USER_PROMPT},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    },
                ],
                stream=True,
            )
            for chunk in stream:
                delta = (
                    getattr(getattr(chunk, "choices", [None])[0] if chunk.choices else None, "delta", None)
                    if chunk.choices
                    else None
                )
                if delta is not None:
                    content = getattr(delta, "content", None) or ""
                    if content:
                        chunks.append(content)
                        if event_callback is not None:
                            event_callback(
                                "stream_chunk",
                                {"page": number, "text": content},
                            )
        except Exception as exc:
            raise OCRServiceError(
                f"{provider_name} request failed on page {number}/{total} "
                f"(model {model!r}): {exc}"
            ) from exc
        text = "".join(chunks).strip()
        if not text:
            raise OCRServiceError(
                f"{provider_name} returned no text for page {number}/{total} "
                f"(model {model!r})."
            )
        results.append(text)
        if event_callback is not None:
            event_callback(
                "page_text",
                {"page": number, "total": total, "text": text},
            )
    return results


def save_markdown_atomic(output_path: Path, content: str) -> None:
    """Write content as UTF-8 with \\n newlines, then publish atomically."""
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            delete=False,
            dir=output_path.parent,
            prefix=f".{output_path.stem}_",
            suffix=".tmp",
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)
            tmp_file.write(normalized)
            tmp_file.flush()
        os.replace(tmp_path, output_path)
    except Exception as exc:
        if tmp_path is not None:
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise OCRServiceError(f"Could not save output file: {exc}") from exc


def open_in_default_app(path: Path) -> None:
    """Open a file with the OS default application. Tk-free.

    macOS uses ``open``, Windows ``os.startfile``, other platforms
    ``xdg-open``. Any failure is wrapped in OCRServiceError.
    """
    try:
        if sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=True)
        elif sys.platform.startswith("win"):
            os.startfile(str(path))  # type: ignore[attr-defined]  # Windows only
        else:
            subprocess.run(["xdg-open", str(path)], check=True)
    except Exception as exc:
        raise OCRServiceError(f"Could not open {path}: {exc}") from exc


def reveal_in_file_manager(path: Path) -> None:
    """Reveal a file in the OS file manager (Finder/Explorer). Tk-free.

    macOS selects the file with ``open -R``; Windows with
    ``explorer /select,``; other platforms open the containing directory
    (no portable "select" flag exists). Any failure is wrapped.
    """
    try:
        if sys.platform == "darwin":
            subprocess.run(["open", "-R", str(path)], check=True)
        elif sys.platform.startswith("win"):
            # explorer returns exit code 1 even on success, so no check=True.
            subprocess.run(["explorer", f"/select,{path}"])
        else:
            subprocess.run(["xdg-open", str(path.parent)], check=True)
    except Exception as exc:
        raise OCRServiceError(f"Could not reveal {path}: {exc}") from exc


def process_ocr(request: OCRRequest, event_queue) -> Path:
    """Run the full OCR pipeline; emit ('log', message) events; return output.

    Raises on any failure. The temporary render directory is always removed
    in the one outer finally, on success and on every failure path. The
    caller (worker wrapper) enqueues the single terminal success/error event
    after this function has returned or raised, so cleanup always precedes
    the terminal event.
    """

    def log(message: str) -> None:
        event_queue.put(("log", message))

    def progress(phase: str, current: int, total: int) -> None:
        event_queue.put(("progress", {"phase": phase, "current": current, "total": total}))

    def emit_event(kind: str, payload: dict) -> None:
        event_queue.put((kind, payload))

    temp_dir: Path | None = None
    try:
        if request.input_path.suffix.lower() in config.PDF_EXTENSIONS:
            log("[1/3] Preparing document...")
            try:
                temp_dir = Path(tempfile.mkdtemp(prefix="local_ocr_"))
            except Exception as exc:
                raise OCRServiceError(
                    f"Could not create temporary render directory: {exc}"
                ) from exc
            image_paths = render_pdf(
                request.input_path,
                request.dpi,
                temp_dir,
                lambda message: log(f"[1/3] {message}"),
                progress,
            )
        else:
            log("[1/3] Preparing image...")
            image_paths = [request.input_path]

        try:
            if request.provider in (config.Provider.LM_STUDIO, config.Provider.VLLM):
                provider_name = (
                    "LM Studio" if request.provider == config.Provider.LM_STUDIO else "vLLM"
                )
                client = openai.OpenAI(
                    base_url=request.server_url.rstrip("/") + "/v1",
                    api_key="lm-studio",
                    timeout=config.OCR_STREAM_IDLE_TIMEOUT,
                )
                page_texts = recognize_images_openai(
                    client,
                    request.model,
                    image_paths,
                    lambda message: log(f"[2/3] {message}"),
                    progress,
                    emit_event,
                    provider_name=provider_name,
                )
            else:
                client = ollama.Client(
                    host=request.server_url,
                    timeout=config.OCR_STREAM_IDLE_TIMEOUT,
                )
                page_texts = recognize_images(
                    client,
                    request.model,
                    image_paths,
                    lambda message: log(f"[2/3] {message}"),
                    progress,
                    emit_event,
                )
        except Exception as exc:
            raise OCRServiceError(
                f"Could not create client for {request.server_url}: {exc}"
            ) from exc

        log("[3/3] Saving Markdown...")
        save_markdown_atomic(request.output_path, "\n\n".join(page_texts))
        return request.output_path
    finally:
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)
