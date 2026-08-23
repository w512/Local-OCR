"""LocalOCRApp: all widgets, state, and main-thread UI behavior.

Threading contract: workers never touch Tk. They only put plain-value
events on self.event_queue; drain_ui_events() runs on the Tk main thread
via .after() and performs every GUI update.
"""

from __future__ import annotations

import bisect
import io
import queue
import sys
import threading
from collections import OrderedDict
from enum import Enum, auto
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk
from PIL import Image

import config
import ocr_service
from ocr_service import OCRRequest


class OperationState(Enum):
    IDLE = auto()
    REFRESHING_MODELS = auto()
    PROCESSING_OCR = auto()


FILE_DIALOG_FILTERS = [
    ("Supported documents", "*.pdf *.png *.jpg *.jpeg *.webp"),
    ("PDF files", "*.pdf"),
    ("Images", "*.png *.jpg *.jpeg *.webp"),
    ("All files", "*.*"),
]

PADX = 12
PADY = 8
PREVIEW_WIDTH = 240
REVIEW_IMAGE_WIDTH = 400  # px, target width of the Review page image
REVIEW_IMAGE_MAX_H = 560  # px, cap the height of tall pages
REVIEW_IMAGE_CACHE_SIZE = 5  # keep at most this many decoded CTkImages


class LocalOCRApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Local OCR")
        self.geometry("780x680")
        self.minsize(620, 520)

        self.operation_state = OperationState.IDLE
        self.closing = False
        self.selected_path: Path | None = None
        self.event_queue: queue.Queue = queue.Queue()
        self._render_phase_seen = False

        # Stream-chunk throttling: chunks arrive frequently; buffer them and
        # flush to the textbox no more often than STREAM_UI_FLUSH_MS.
        self._stream_buffer = ""
        self._stream_flush_scheduled = False
        # Last page number written to the Result panel (0 = none yet). Used to
        # insert the inter-page separator exactly once and to avoid
        # re-appending a page's text that streaming already wrote live.
        self._result_page = 0

        # Review model: per-page image bytes + text, filled as pages complete.
        self.review_pages: dict[int, dict] = {}
        self._review_order: list[int] = []  # page numbers ready for review
        self._review_index = 0  # position within _review_order
        self._review_total = 0  # document page count (for the "X / N" label)
        # LRU of decoded CTkImages so hundreds of pages don't stay in memory.
        self._review_image_cache: "OrderedDict[int, ctk.CTkImage]" = OrderedDict()

        self._build_layout()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(config.UI_POLL_INTERVAL_MS, self.drain_ui_events)

    # ------------------------------------------------------------- layout

    def _build_layout(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(4, weight=1)  # the log absorbs resize space

        # File section
        file_frame = ctk.CTkFrame(self)
        file_frame.grid(row=0, column=0, sticky="ew", padx=PADX, pady=(PADY, 4))
        file_frame.grid_columnconfigure(1, weight=1)
        self.select_button = ctk.CTkButton(
            file_frame, text="Select File", command=self.select_file
        )
        self.select_button.grid(row=0, column=0, padx=PADX, pady=PADY)
        self.file_label = ctk.CTkLabel(file_frame, text="No file selected", anchor="w")
        self.file_label.grid(row=0, column=1, sticky="ew", padx=(0, PADX), pady=PADY)

        # Settings section
        settings = ctk.CTkFrame(self)
        settings.grid(row=1, column=0, sticky="ew", padx=PADX, pady=4)
        settings.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(settings, text="Provider").grid(
            row=0, column=0, sticky="w", padx=PADX, pady=(PADY, 4)
        )
        self.provider_combobox = ctk.CTkComboBox(
            settings,
            values=[p.value for p in config.Provider],
            state="readonly",
            width=130,
        )
        self.provider_combobox.set(config.Provider.OLLAMA.value)
        self.provider_combobox.grid(
            row=0, column=1, sticky="w", padx=(0, 8), pady=(PADY, 4)
        )
        self.provider_combobox.bind("<<ComboboxSelected>>", self._on_provider_changed)

        self.url_label = ctk.CTkLabel(settings, text="Server URL")
        self.url_label.grid(
            row=1, column=0, sticky="w", padx=PADX, pady=(PADY, 4)
        )
        self.url_entry = ctk.CTkEntry(settings)
        self.url_entry.insert(0, config.DEFAULT_OLLAMA_URL)
        self.url_entry.grid(row=1, column=1, sticky="ew", padx=(0, 8), pady=(PADY, 4))
        self.refresh_button = ctk.CTkButton(
            settings, text="Refresh Models", width=130, command=self.refresh_models
        )
        self.refresh_button.grid(row=1, column=2, padx=(0, PADX), pady=(PADY, 4))

        ctk.CTkLabel(settings, text="Model").grid(
            row=2, column=0, sticky="w", padx=PADX, pady=4
        )
        self.model_combobox = ctk.CTkComboBox(
            settings, values=list(config.EXAMPLE_MODELS)
        )
        self.model_combobox.set("")  # suggestions are not installed models
        self.model_combobox.grid(
            row=2, column=1, columnspan=2, sticky="ew", padx=(0, PADX), pady=4
        )

        ctk.CTkLabel(settings, text="PDF DPI").grid(
            row=3, column=0, sticky="w", padx=PADX, pady=(4, PADY)
        )
        self.dpi_combobox = ctk.CTkComboBox(
            settings,
            values=[str(dpi) for dpi in config.DPI_OPTIONS],
            state="readonly",
            width=120,
        )
        self.dpi_combobox.set(str(config.DEFAULT_DPI))
        self.dpi_combobox.grid(row=3, column=1, sticky="w", pady=(4, PADY))

        # Action + feedback section
        self.start_button = ctk.CTkButton(
            self,
            text="Start OCR",
            height=44,
            font=ctk.CTkFont(size=16, weight="bold"),
            command=self.start_ocr,
        )
        self.start_button.grid(row=2, column=0, sticky="ew", padx=PADX, pady=4)

        # Status section: progress bar + page counter label
        self.status_frame = ctk.CTkFrame(self)
        self.status_frame.grid(row=3, column=0, sticky="ew", padx=PADX, pady=4)
        self.status_frame.grid_columnconfigure(0, weight=1)

        self.progress = ctk.CTkProgressBar(self.status_frame, mode="indeterminate")
        self.progress.grid(row=0, column=0, sticky="ew", padx=(PADX, 8), pady=PADY)
        self.status_label = ctk.CTkLabel(self.status_frame, text="", width=160)
        self.status_label.grid(row=0, column=1, padx=(0, PADX), pady=PADY)
        self.progress.set(0)

        # Bottom section: preview panel (left, fixed) + tabbed Log/Result panel.
        self.bottom_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.bottom_frame.grid(
            row=4, column=0, sticky="nsew", padx=PADX, pady=(4, PADY)
        )
        self.bottom_frame.grid_columnconfigure(0, weight=0)  # preview: fixed
        self.bottom_frame.grid_columnconfigure(1, weight=1)  # tabs: grow
        self.bottom_frame.grid_rowconfigure(0, weight=1)

        # Preview panel: thumbnail of the page being recognized + a caption.
        self.preview_panel = ctk.CTkFrame(self.bottom_frame, width=PREVIEW_WIDTH)
        self.preview_panel.grid(row=0, column=0, sticky="ns", padx=(0, 8))
        self.preview_panel.grid_propagate(False)  # keep the fixed width
        self.preview_panel.grid_columnconfigure(0, weight=1)
        self.preview_panel.grid_rowconfigure(0, weight=1)
        self.preview_image_label = ctk.CTkLabel(self.preview_panel, text="")
        self.preview_image_label.grid(
            row=0, column=0, sticky="nsew", padx=6, pady=(6, 2)
        )
        self.preview_caption = ctk.CTkLabel(self.preview_panel, text="")
        self.preview_caption.grid(row=1, column=0, sticky="ew", padx=6, pady=(0, 6))
        self._preview_image = None  # hold the CTkImage ref so GC keeps it alive

        self.tabview = ctk.CTkTabview(self.bottom_frame)
        self.tabview.add("Log")
        self.tabview.add("Result")
        self.tabview.add("Review")
        self.tabview.set("Log")
        self.tabview.grid(row=0, column=1, sticky="nsew")

        mono_font = ctk.CTkFont(family="Courier New", size=12)

        self.log_box = ctk.CTkTextbox(
            self.tabview.tab("Log"),
            font=mono_font,
            state="disabled",
            wrap="word",
        )
        self.log_box.pack(fill="both", expand=True)

        # Result tab: a Copy button in the top-right corner + read-only text.
        self.result_frame = self.tabview.tab("Result")
        self.copy_button = ctk.CTkButton(
            self.result_frame, text="Copy", width=80, command=self.copy_result
        )
        self.copy_button.pack(anchor="e", padx=(0, 4), pady=(4, 2))
        self.result_box = ctk.CTkTextbox(
            self.result_frame,
            font=mono_font,
            state="disabled",
            wrap="word",
        )
        self.result_box.pack(fill="both", expand=True)

        # Review tab: page image (left) ↔ its text (right), with navigation.
        review = self.tabview.tab("Review")
        review.grid_columnconfigure(0, weight=0)  # image: fixed
        review.grid_columnconfigure(1, weight=1)  # text: grows
        review.grid_rowconfigure(1, weight=1)

        nav = ctk.CTkFrame(review, fg_color="transparent")
        nav.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(4, 6))
        nav.grid_columnconfigure(1, weight=1)
        self.review_prev_button = ctk.CTkButton(
            nav, text="◀", width=44, state="disabled", command=self.review_prev
        )
        self.review_prev_button.grid(row=0, column=0, padx=(4, 8))
        self.review_nav_label = ctk.CTkLabel(nav, text="No pages yet")
        self.review_nav_label.grid(row=0, column=1)
        self.review_next_button = ctk.CTkButton(
            nav, text="▶", width=44, state="disabled", command=self.review_next
        )
        self.review_next_button.grid(row=0, column=2, padx=(8, 4))

        self.review_image_label = ctk.CTkLabel(
            review, text="", width=REVIEW_IMAGE_WIDTH
        )
        self.review_image_label.grid(
            row=1, column=0, sticky="nsew", padx=(4, 8), pady=4
        )
        self.review_text = ctk.CTkTextbox(
            review, font=mono_font, state="disabled", wrap="word"
        )
        self.review_text.grid(row=1, column=1, sticky="nsew", pady=4)

    # ---------------------------------------------------- event plumbing

    def drain_ui_events(self) -> None:
        if self.closing:
            return  # discard queued UI work during shutdown
        try:
            while True:
                try:
                    kind, payload = self.event_queue.get_nowait()
                except queue.Empty:
                    break
                self.handle_event(kind, payload)
        finally:
            # Reschedule even if a handler raised; a broken .after() chain
            # would silently stop all event processing and leave the UI
            # stuck in its busy state.
            self.after(config.UI_POLL_INTERVAL_MS, self.drain_ui_events)

    def handle_event(self, kind: str, payload) -> None:
        if kind == "log":
            self.append_log(payload)
        elif kind == "models_loaded":
            self.on_models_loaded(payload)
        elif kind == "refresh_error":
            self.on_refresh_error(payload)
        elif kind == "ocr_success":
            self.on_ocr_success(payload)
        elif kind == "progress":
            self.on_progress(payload)
        elif kind == "ocr_error":
            self.on_ocr_error(payload)
        elif kind == "page_text":
            self.on_page_text(payload)
        elif kind == "page_image":
            self.on_page_image(payload)
        elif kind == "stream_chunk":
            self.on_stream_chunk(payload)
        else:
            # Surface protocol mismatches (e.g. a new worker event kind
            # without a handler) instead of dropping them silently.
            self.append_log(f"[Warn] Unhandled event kind: {kind!r}")

    def append_log(self, message: str) -> None:
        self.log_box.configure(state="normal")
        self.log_box.insert("end", message + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def append_result(self, text: str) -> None:
        """Append text to the read-only Result panel and autoscroll."""
        self.result_box.configure(state="normal")
        self.result_box.insert("end", text)
        self.result_box.see("end")
        self.result_box.configure(state="disabled")

    def _clear_result_panel(self) -> None:
        self.result_box.configure(state="normal")
        self.result_box.delete("1.0", "end")
        self.result_box.configure(state="disabled")
        self._stream_buffer = ""
        self._stream_flush_scheduled = False
        self._result_page = 0

    def on_page_image(self, payload: dict) -> None:
        """Show the thumbnail of the page currently being recognized and keep
        its bytes for the Review tab."""
        page = payload["page"]
        self._review_total = payload["total"]
        self.review_pages.setdefault(page, {"png": None, "text": None})["png"] = (
            payload["png"]
        )

        image = Image.open(io.BytesIO(payload["png"]))
        width, height = image.size
        max_width = PREVIEW_WIDTH - 20  # leave room for the panel padding
        scale = min(1.0, max_width / width)
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        self._preview_image = ctk.CTkImage(
            light_image=image, dark_image=image, size=size
        )
        self.preview_image_label.configure(image=self._preview_image)
        self.preview_caption.configure(text=f"Page {page} / {payload['total']}")

    def _clear_preview(self) -> None:
        self.preview_image_label.configure(image=None)
        self._preview_image = None
        self.preview_caption.configure(text="")

    # -------------------------------------------------------- review tab

    def _clear_review(self) -> None:
        self.review_pages = {}
        self._review_order = []
        self._review_index = 0
        self._review_total = 0
        self._review_image_cache.clear()
        self.review_image_label.configure(image=None)
        self.review_text.configure(state="normal")
        self.review_text.delete("1.0", "end")
        self.review_text.configure(state="disabled")
        self.review_nav_label.configure(text="No pages yet")
        self.review_prev_button.configure(state="disabled")
        self.review_next_button.configure(state="disabled")

    def _register_review_page(self, page: int) -> None:
        """Make a page navigable once its text is ready; keep the user's
        current position, only showing the first page automatically."""
        if page in self._review_order:
            self._update_review_nav()
            return
        bisect.insort(self._review_order, page)
        if len(self._review_order) == 1:
            self.show_review_page(0)
        else:
            self._update_review_nav()

    def _review_image_for(self, page: int, png: bytes | None):
        if png is None:
            return None
        cached = self._review_image_cache.get(page)
        if cached is not None:
            self._review_image_cache.move_to_end(page)
            return cached
        image = Image.open(io.BytesIO(png))
        width, height = image.size
        scale = min(
            REVIEW_IMAGE_WIDTH / width, REVIEW_IMAGE_MAX_H / height, 1.0
        )
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        ctk_image = ctk.CTkImage(light_image=image, dark_image=image, size=size)
        self._review_image_cache[page] = ctk_image
        while len(self._review_image_cache) > REVIEW_IMAGE_CACHE_SIZE:
            self._review_image_cache.popitem(last=False)  # evict least-recent
        return ctk_image

    def show_review_page(self, index: int) -> None:
        if not self._review_order:
            return
        index = max(0, min(index, len(self._review_order) - 1))
        self._review_index = index
        page = self._review_order[index]
        entry = self.review_pages.get(page, {})
        self.review_image_label.configure(
            image=self._review_image_for(page, entry.get("png"))
        )
        self.review_text.configure(state="normal")
        self.review_text.delete("1.0", "end")
        self.review_text.insert("1.0", entry.get("text") or "")
        self.review_text.configure(state="disabled")
        self._update_review_nav()

    def _update_review_nav(self) -> None:
        ready = len(self._review_order)
        if ready == 0:
            self.review_nav_label.configure(text="No pages yet")
            self.review_prev_button.configure(state="disabled")
            self.review_next_button.configure(state="disabled")
            return
        page = self._review_order[self._review_index]
        document_total = self._review_total or ready
        self.review_nav_label.configure(text=f"Page {page} / {document_total}")
        self.review_prev_button.configure(
            state="normal" if self._review_index > 0 else "disabled"
        )
        self.review_next_button.configure(
            state="normal" if self._review_index < ready - 1 else "disabled"
        )

    def review_prev(self) -> None:
        self.show_review_page(self._review_index - 1)

    def review_next(self) -> None:
        self.show_review_page(self._review_index + 1)

    # ---------------------------------------------------- control states

    def _apply_refresh_busy_state(self) -> None:
        self.provider_combobox.configure(state="disabled")
        self.url_entry.configure(state="disabled")
        self.refresh_button.configure(state="disabled")
        self.start_button.configure(state="disabled")

    def _apply_ocr_busy_state(self) -> None:
        self.select_button.configure(state="disabled")
        self.provider_combobox.configure(state="disabled")
        self.url_entry.configure(state="disabled")
        self.refresh_button.configure(state="disabled")
        self.model_combobox.configure(state="disabled")
        self.dpi_combobox.configure(state="disabled")
        self.start_button.configure(
            state="disabled", text="Processing, please wait..."
        )
        self._render_phase_seen = False
        self._clear_result_panel()
        self._clear_preview()
        self._clear_review()
        self.tabview.set("Log")
        self.status_label.configure(text="")
        self.progress.configure(mode="indeterminate")
        self.progress.start()

    def _restore_idle(self) -> None:
        self.select_button.configure(state="normal")
        self.provider_combobox.configure(state="readonly")
        self.url_entry.configure(state="normal")
        self.refresh_button.configure(state="normal")
        self.model_combobox.configure(state="normal")
        self.dpi_combobox.configure(state="readonly")
        self.start_button.configure(state="normal", text="Start OCR")
        self.progress.stop()
        self.progress.configure(mode="indeterminate")
        self.progress.set(0)
        self.status_label.configure(text="")
        self._render_phase_seen = False
        self.operation_state = OperationState.IDLE

    # ----------------------------------------------------- file selection

    def select_file(self) -> None:
        if self.operation_state is not OperationState.IDLE:
            return
        filename = filedialog.askopenfilename(
            title="Select a PDF or image",
            filetypes=FILE_DIALOG_FILTERS,
            parent=self,
        )
        if not filename:
            return
        path = Path(filename)
        try:
            ocr_service.validate_input_path(path)
        except ValueError as exc:
            # Keep the previous valid selection.
            messagebox.showwarning("Unsupported file", str(exc), parent=self)
            return
        self.selected_path = path
        self.file_label.configure(text=path.name)

    # ------------------------------------------------------ model refresh

    def _current_provider(self) -> config.Provider:
        """Return the currently selected provider from the combobox."""
        value = self.provider_combobox.get().strip()
        try:
            return config.Provider(value)
        except ValueError:
            return config.Provider.OLLAMA

    def _on_provider_changed(self, event=None) -> None:
        """Update the URL entry default when the provider changes."""
        provider = self._current_provider()
        if provider == config.Provider.LM_STUDIO:
            default_url = config.DEFAULT_LM_STUDIO_URL
        elif provider == config.Provider.VLLM:
            default_url = config.DEFAULT_VLLM_URL
        else:
            default_url = config.DEFAULT_OLLAMA_URL
        current = self.url_entry.get().strip()
        # Only replace the URL if it's empty or matches a known default.
        if not current or current in (
            config.DEFAULT_OLLAMA_URL,
            config.DEFAULT_LM_STUDIO_URL,
            config.DEFAULT_VLLM_URL,
        ):
            self.url_entry.delete(0, "end")
            self.url_entry.insert(0, default_url)

    def refresh_models(self) -> None:
        if self.operation_state is not OperationState.IDLE:
            return
        try:
            url = ocr_service.normalize_server_url(self.url_entry.get())
        except ValueError as exc:
            messagebox.showerror("Invalid URL", str(exc), parent=self)
            return
        provider = self._current_provider()
        self.operation_state = OperationState.REFRESHING_MODELS
        self._apply_refresh_busy_state()
        self.append_log(f"Refreshing model list from {url}...")
        threading.Thread(
            target=self._refresh_worker, args=(url, provider), daemon=True
        ).start()

    def _refresh_worker(self, url: str, provider: config.Provider) -> None:
        try:
            if provider == config.Provider.LM_STUDIO:
                models = ocr_service.list_lm_studio_models(url)
            elif provider == config.Provider.VLLM:
                models = ocr_service.list_vllm_models(url)
            else:
                models = ocr_service.list_models(url)
        except Exception as exc:
            self.event_queue.put(("refresh_error", str(exc)))
        else:
            self.event_queue.put(("models_loaded", models))

    def on_progress(self, payload: dict) -> None:
        phase = payload["phase"]
        current = payload["current"]
        total = payload["total"]

        if phase == "render":
            self._render_phase_seen = True
            fraction = 0.2 * current / total
        else:  # "ocr"
            if self._render_phase_seen:
                fraction = 0.2 + 0.8 * current / total
            else:
                fraction = current / total

        self.progress.stop()
        self.progress.configure(mode="determinate")
        self.progress.set(fraction)

        phase_label = "OCR" if phase == "ocr" else "Render"
        self.status_label.configure(text=f"Page {current} / {total} ({phase_label})")

    def on_page_text(self, payload: dict) -> None:
        """Finalize a page in the Result panel.

        With streaming on, the page's text is already in the panel from
        on_stream_chunk, so it must not be appended again. This only appends
        as a fallback for a page that produced no stream chunks (which cannot
        happen for a non-empty page, but keeps the panel correct should
        streaming ever be bypassed)."""
        self._flush_stream_buffer()
        page = payload["page"]
        text = payload["text"]

        # Feed the Review tab: a page becomes navigable once its text is ready.
        self._review_total = payload.get("total", self._review_total)
        self.review_pages.setdefault(page, {"png": None, "text": None})["text"] = text
        self._register_review_page(page)

        if page == self._result_page:
            return  # already streamed live — do not duplicate the text
        if self._result_page:
            self.append_result("\n\n")
        self._result_page = page
        self.append_result(text)

    def on_stream_chunk(self, payload: dict) -> None:
        """Buffer a stream delta; flush to the textbox at most every
        STREAM_UI_FLUSH_MS to avoid choking Tk with a flood of inserts.

        The inter-page separator is embedded into the buffer at the page
        boundary so a live-streamed run reads exactly like the saved file."""
        page = payload["page"]
        if self._result_page and page != self._result_page:
            self._stream_buffer += "\n\n"
        self._result_page = page
        self._stream_buffer += payload["text"]
        if not self._stream_flush_scheduled:
            self._stream_flush_scheduled = True
            self.after(config.STREAM_UI_FLUSH_MS, self._flush_stream_buffer)

    def _flush_stream_buffer(self) -> None:
        if self._stream_buffer:
            self.append_result(self._stream_buffer)
            self._stream_buffer = ""
        self._stream_flush_scheduled = False

    def copy_result(self) -> None:
        """Copy the Result panel text to the system clipboard."""
        self.clipboard_clear()
        self.clipboard_append(self.result_box.get("1.0", "end-1c"))

    def on_models_loaded(self, models: list[str]) -> None:
        self._restore_idle()
        if not models:
            self.append_log(
                "No models found on the server; enter a model tag manually."
            )
            return
        typed = self.model_combobox.get().strip()
        self.model_combobox.configure(values=models)
        self.model_combobox.set(typed if typed else models[0])
        self.append_log(f"Found {len(models)} model(s).")

    def on_refresh_error(self, message: str) -> None:
        self._restore_idle()
        self.append_log(f"[Error] {message}")
        messagebox.showerror("Model refresh failed", message, parent=self)

    # -------------------------------------------------------------- OCR

    def start_ocr(self) -> None:
        if self.operation_state is not OperationState.IDLE:
            return
        # Snapshot and validate every input on the main thread.
        if self.selected_path is None:
            messagebox.showerror(
                "No file", "Select a PDF or image file first.", parent=self
            )
            return
        input_path = self.selected_path
        try:
            ocr_service.validate_input_path(input_path)
        except ValueError as exc:
            messagebox.showerror("Invalid file", str(exc), parent=self)
            return
        try:
            url = ocr_service.normalize_server_url(self.url_entry.get())
        except ValueError as exc:
            messagebox.showerror("Invalid URL", str(exc), parent=self)
            return
        provider = self._current_provider()
        model = self.model_combobox.get().strip()
        if not model:
            messagebox.showerror(
                "No model", "Enter or select a model tag.", parent=self
            )
            return
        try:
            dpi = int(self.dpi_combobox.get())
        except ValueError:
            dpi = -1
        if dpi not in config.DPI_OPTIONS:
            options = ", ".join(str(d) for d in config.DPI_OPTIONS)
            messagebox.showerror(
                "Invalid DPI", f"DPI must be one of: {options}", parent=self
            )
            return

        output_path = ocr_service.build_output_path(input_path)
        if output_path.exists():
            overwrite = messagebox.askyesno(
                "Overwrite existing file?",
                f"{output_path.name} already exists in the same folder.\n"
                "Overwrite it?",
                parent=self,
            )
            if not overwrite:
                return

        request = OCRRequest(
            input_path=input_path,
            output_path=output_path,
            provider=provider,
            server_url=url,
            model=model,
            dpi=dpi,
        )
        self.operation_state = OperationState.PROCESSING_OCR
        self._apply_ocr_busy_state()
        self.append_log(f"[Start] Input: {input_path}")
        self.append_log(f"[Start] Provider: {provider.value} | Server: {url} | Model: {model}")
        threading.Thread(
            target=self._ocr_worker, args=(request,), daemon=True
        ).start()

    def _ocr_worker(self, request: OCRRequest) -> None:
        saved_path = None
        error: Exception | None = None
        try:
            saved_path = ocr_service.process_ocr(request, self.event_queue)
        except Exception as exc:
            error = exc
        # process_ocr's finally has already cleaned up temporary files;
        # now enqueue exactly one terminal event.
        if error is not None:
            self.event_queue.put(("ocr_error", str(error)))
        else:
            self.event_queue.put(("ocr_success", str(saved_path)))

    def on_ocr_success(self, saved_path: str) -> None:
        self._flush_stream_buffer()
        self._restore_idle()
        self.append_log(f"[Success] File saved: {saved_path}")
        self.tabview.set("Result")
        self._show_completion_dialog(saved_path)

    @staticmethod
    def _reveal_button_text() -> str:
        if sys.platform == "darwin":
            return "Show in Finder"
        if sys.platform.startswith("win"):
            return "Show in Explorer"
        return "Open Folder"

    def _show_completion_dialog(self, saved_path: str) -> None:
        """Completion popup with Open / Show in Finder / OK actions.

        A native messagebox can't carry custom buttons, so this is a small
        CTkToplevel. It is non-blocking: the user dismisses it with OK."""
        path = Path(saved_path)
        dialog = ctk.CTkToplevel(self)
        dialog.title("OCR complete")
        dialog.resizable(False, False)
        dialog.transient(self)

        ctk.CTkLabel(
            dialog,
            text=f"Markdown saved to:\n{saved_path}",
            justify="left",
            wraplength=420,
        ).grid(row=0, column=0, columnspan=3, padx=PADX, pady=(PADX, 8), sticky="w")

        ctk.CTkButton(
            dialog, text="Open", width=110,
            command=lambda: self._run_file_action(
                ocr_service.open_in_default_app, path
            ),
        ).grid(row=1, column=0, padx=(PADX, 4), pady=(0, PADX))
        ctk.CTkButton(
            dialog, text=self._reveal_button_text(), width=140,
            command=lambda: self._run_file_action(
                ocr_service.reveal_in_file_manager, path
            ),
        ).grid(row=1, column=1, padx=4, pady=(0, PADX))
        ok_button = ctk.CTkButton(
            dialog, text="OK", width=80, command=dialog.destroy
        )
        ok_button.grid(row=1, column=2, padx=(4, PADX), pady=(0, PADX))

        # Bring to front and focus OK once the window is realized.
        dialog.after(50, dialog.lift)
        ok_button.focus_set()

    def _run_file_action(self, action, path: Path) -> None:
        """Invoke a Tk-free file action, surfacing failures to the user."""
        try:
            action(path)
        except ocr_service.OCRServiceError as exc:
            self.append_log(f"[Error] {exc}")
            messagebox.showerror("Action failed", str(exc), parent=self)

    def on_ocr_error(self, message: str) -> None:
        self._flush_stream_buffer()
        self._restore_idle()
        self.append_log(f"[Error] {message}")
        self.tabview.set("Log")
        messagebox.showerror("OCR failed", message, parent=self)

    # ---------------------------------------------------------- shutdown

    def on_close(self) -> None:
        if self.closing or self.operation_state is OperationState.IDLE:
            self.closing = True
            self.destroy()
            return
        if messagebox.askyesno(
            "Quit", "An operation is still running. Close anyway?", parent=self
        ):
            self.closing = True
            self.destroy()
