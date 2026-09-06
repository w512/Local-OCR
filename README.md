# Local OCR

A local, privacy-focused desktop application that converts PDFs and images to
structured Markdown using Vision-Language models. Nothing leaves your machine
or network: the app talks only to the server URL you configure.

## Supported Backends

The app supports three local LLM backends:

- **Ollama** — the default. Uses the official Ollama API.
- **LM Studio** — uses the OpenAI-compatible API exposed by LM Studio's
  local server.
- **vLLM** — uses the OpenAI-compatible API exposed by vLLM's server.

Select your backend from the **Provider** dropdown in the app's settings
panel. The server URL field and default port update automatically.

## Requirements

- Python 3.10 or newer with Tkinter support (macOS, Linux, or Windows).
  Both Tk 8.6 and Tk 9.0 work with the pinned customtkinter version
  (customtkinter 6.x; older 5.2.x renders blank windows under Tk 9.0 on
  macOS).
- A running Ollama or LM Studio server — locally or reachable on your
  network. The app never starts, installs, or pulls anything itself.
- A vision-capable model installed on that server.

## Installation

### Using uv (recommended)

```bash
uv venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
uv sync
```

### Using pip

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running

```bash
python main.py

# using uv
uv run main.py
```

## Setting up Ollama

The app requires an Ollama server that is already running. To see which
models a server has installed:

```bash
ollama list
```

To install a vision-capable model:

```bash
ollama pull <model-tag>
```

The model suggestions shown in the app (`gemma4:12b`, `qwen3.6:27b`) are
examples only — they are not guaranteed to exist on your server and are never
pulled automatically. Use `Refresh Models` to list what your server actually
has, or type any model tag manually.

### Server URL

- Local Ollama: keep the default `http://localhost:11434`.
- Ollama on another machine: use its address, e.g.
  `http://192.168.1.50:11434`.

**Network safety:** exposing Ollama beyond localhost makes it reachable by
anyone who can connect to that port. Only bind it to a trusted network and
protect it with your firewall; Ollama has no built-in authentication.

## Setting up LM Studio

LM Studio exposes an OpenAI-compatible API. To use it:

1. Install and launch [LM Studio](https://lmstudio.ai/).
2. Open the **Local Server** tab and start the server (default port `1234`).
3. Load a vision-capable model in LM Studio (e.g. `llama-3.2-vision`,
   `gemma-3-vision`, or `qwen-vl`).
4. In the app, select **LM Studio** from the Provider dropdown. The server
   URL defaults to `http://localhost:1234`.
5. Use `Refresh Models` to list what your LM Studio server has loaded, or
   type a model ID manually.

**Note:** LM Studio's API key is ignored by the local server; the app sends
a placeholder key (`lm-studio`) as required by the OpenAI client library.

## Setting up vLLM

[vLLM](https://github.com/vllm-project/vllm) exposes an OpenAI-compatible API.
To use it:

1. Install vLLM and start the server with a vision-capable model:

   ```bash
   pip install vllm
   vllm serve <model-id> --port 8000
   ```

2. In the app, select **vLLM** from the Provider dropdown. The server URL
   defaults to `http://localhost:8000`.
3. Use `Refresh Models` to list what your vLLM server has loaded, or type a
   model ID manually.

**Note:** Like LM Studio, vLLM's local server ignores the API key; the app
sends a placeholder key (`lm-studio`) as required by the OpenAI client
library.

## Usage

1. `Select File` — choose one PDF or image (`.pdf`, `.png`, `.jpg`, `.jpeg`,
   `.webp`).
2. Select a **Provider** (Ollama, LM Studio, or vLLM), confirm the server
   URL, pick or type a model tag, and choose a PDF DPI (100/150/200/300;
   higher is sharper but slower — DPI only affects PDFs).
3. `Start OCR`. Each PDF page is rendered and sent to the model in order.

While a job runs you can follow it in several places:

- A **progress bar** with a page counter (`Page 3 / 12`) shows real progress —
  a short render phase followed by recognition.
- A **preview panel** displays a thumbnail of the page currently being read.
- The **Result** tab fills with the recognized Markdown live, token by token,
  as the model streams it. A `Copy` button copies the full text to the
  clipboard.
- The **Log** tab keeps the status messages.
- The **Review** tab pairs each finished page's image with its text
  side by side, with `◀` / `▶` navigation for spot-checking quality.

On success the app switches to the Result tab and shows a dialog with `Open`
(open the `.md` in your default app), `Show in Finder` (or `Open Folder` off
macOS), and `OK`. On error it switches to the Log tab.

### Output

The result is saved as UTF-8 Markdown **next to the input file** with the
`_extracted.md` suffix — `/docs/report.pdf` becomes
`/docs/report_extracted.md`. If that file already exists you are asked before
it is overwritten; declining leaves it untouched. The file is written
atomically, so a failed run never leaves a partial result.

## Troubleshooting

| Symptom | Likely cause and fix |
| --- | --- |
| `connection refused` | The server is not running, or the URL/port is wrong. Start Ollama (`ollama serve`) or LM Studio's local server, and verify the URL. |
| Timeout | Server unreachable (wrong LAN address, firewall) or the model is too slow for the page. Try a smaller model or lower DPI. |
| `model not found` | The tag is not loaded on that server. Check the server's model list and load a model. Models are never pulled automatically. |
| Empty or garbage output / "returned no text" | The selected model has no vision support. Choose a vision-capable model. |
