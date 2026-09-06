"""Configuration constants for the Local OCR application."""

from enum import Enum


class Provider(str, Enum):
    """Supported LLM backend providers."""

    OLLAMA = "ollama"
    LM_STUDIO = "lm_studio"
    VLLM = "vllm"


# Default server URLs per provider.
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_LM_STUDIO_URL = "http://localhost:1234"
DEFAULT_VLLM_URL = "http://localhost:8000"

# Suggestions only. These tags are never assumed to exist on the server and
# are never pulled automatically.
EXAMPLE_MODELS = ["glm-ocr",]

DPI_OPTIONS = [100, 150, 200, 300]
DEFAULT_DPI = 150

PDF_EXTENSIONS = frozenset({".pdf"})
IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp"})
SUPPORTED_EXTENSIONS = PDF_EXTENSIONS | IMAGE_EXTENSIONS

# Seconds. Model listing should fail fast.
MODEL_LIST_TIMEOUT = 10
# Seconds of "silence" between stream chunks before a page request is
# considered stuck. OCR runs with stream=True, so the httpx timeout applies
# to the pauses between chunks rather than to the whole response — better
# than a single overall timeout (a long page won't hit it).
OCR_STREAM_IDLE_TIMEOUT = 120

# Milliseconds — throttling of live stream UI updates so Tk doesn't choke
# on a flood of tiny inserts.
STREAM_UI_FLUSH_MS = 100

# Pixels — longest side of the page preview thumbnail. Large enough to stay
# crisp if a bigger side-by-side view reuses the same bytes later; the live
# preview just displays it scaled down.
THUMBNAIL_MAX_SIDE = 900

# Milliseconds between main-thread drains of the worker event queue.
UI_POLL_INTERVAL_MS = 50

SYSTEM_PROMPT = (
    "Convert this image into Markdown text format. Your task is to perform "
    "high-accuracy Optical Character Recognition (OCR). Preserve the "
    "document's structure as accurately as possible: headers, lists, and "
    "tables. Do not add any greetings, explanations, or "
    "introductory/concluding remarks. Output only the raw recognized text."
)

USER_PROMPT = "Recognize this document page."
