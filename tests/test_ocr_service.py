"""Tests for ocr_service. No running Ollama server or real model required."""

import os
import queue
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import config
import ocr_service
from ocr_service import OCRRequest, OCRServiceError


def chat_response(content):
    """Shape of ollama.Client.chat() responses: response.message.content."""
    return SimpleNamespace(message=SimpleNamespace(content=content))


def stream_response(*deltas):
    """An iterator of stream chunks, the shape of chat(..., stream=True).

    Each delta becomes one chunk with ``message.content``; ``None``/empty
    deltas are kept to mimic real server output (the service skips them).
    """
    return iter([SimpleNamespace(message=SimpleNamespace(content=d)) for d in deltas])


def model_entry(tag):
    """Shape of ollama.Client.list() entries: item.model."""
    return SimpleNamespace(model=tag)


def make_fake_document(page_count, needs_pass=False):
    """A PyMuPDF document mock usable as a context manager."""
    document = mock.MagicMock()
    document.needs_pass = needs_pass
    document.page_count = page_count
    document.__enter__.return_value = document
    document.__exit__.return_value = False
    pages = []
    for _ in range(page_count):
        page = mock.MagicMock()
        page.get_pixmap.return_value = mock.MagicMock()
        pages.append(page)
    document.load_page.side_effect = lambda index: pages[index]
    document.fake_pages = pages
    return document


def drain(event_queue):
    events = []
    while True:
        try:
            events.append(event_queue.get_nowait())
        except queue.Empty:
            return events


class TestNormalizeOllamaUrl(unittest.TestCase):
    def test_default_url_unchanged(self):
        self.assertEqual(
            ocr_service.normalize_ollama_url("http://localhost:11434"),
            "http://localhost:11434",
        )

    def test_whitespace_and_trailing_slash_trimmed(self):
        self.assertEqual(
            ocr_service.normalize_ollama_url("  http://192.168.1.20:11434/  "),
            "http://192.168.1.20:11434",
        )

    def test_multiple_trailing_slashes_trimmed(self):
        self.assertEqual(
            ocr_service.normalize_ollama_url("http://ollama.local:11434//"),
            "http://ollama.local:11434",
        )

    def test_https_accepted(self):
        self.assertEqual(
            ocr_service.normalize_ollama_url("https://ollama.example.com"),
            "https://ollama.example.com",
        )

    def test_reverse_proxy_path_prefix_preserved(self):
        self.assertEqual(
            ocr_service.normalize_ollama_url("https://server.lan/ollama/"),
            "https://server.lan/ollama",
        )

    def test_empty_rejected(self):
        for value in ("", "   ", "///"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    ocr_service.normalize_ollama_url(value)

    def test_missing_host_rejected(self):
        for value in ("http://", "http:///path"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    ocr_service.normalize_ollama_url(value)

    def test_non_http_scheme_rejected(self):
        for value in ("ftp://host:11434", "file:///tmp/x", "localhost:11434"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    ocr_service.normalize_ollama_url(value)


class TestNormalizeServerUrl(unittest.TestCase):
    """normalize_server_url is the shared validator for both providers."""

    def test_default_urls_unchanged(self):
        self.assertEqual(
            ocr_service.normalize_server_url("http://localhost:11434"),
            "http://localhost:11434",
        )
        self.assertEqual(
            ocr_service.normalize_server_url("http://localhost:1234"),
            "http://localhost:1234",
        )

    def test_whitespace_and_trailing_slash_trimmed(self):
        self.assertEqual(
            ocr_service.normalize_server_url("  http://192.168.1.20:1234/  "),
            "http://192.168.1.20:1234",
        )

    def test_https_accepted(self):
        self.assertEqual(
            ocr_service.normalize_server_url("https://lmstudio.example.com"),
            "https://lmstudio.example.com",
        )

    def test_reverse_proxy_path_prefix_preserved(self):
        self.assertEqual(
            ocr_service.normalize_server_url("https://server.lan/ollama/"),
            "https://server.lan/ollama",
        )

    def test_empty_rejected(self):
        for value in ("", "   ", "///"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    ocr_service.normalize_server_url(value)

    def test_missing_host_rejected(self):
        for value in ("http://", "http:///path"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    ocr_service.normalize_server_url(value)

    def test_non_http_scheme_rejected(self):
        for value in ("ftp://host:1234", "file:///tmp/x", "localhost:1234"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    ocr_service.normalize_server_url(value)

    def test_normalize_ollama_url_delegates(self):
        """normalize_ollama_url should produce the same result as normalize_server_url."""
        url = "  http://ollama.local:11434/  "
        self.assertEqual(
            ocr_service.normalize_ollama_url(url),
            ocr_service.normalize_server_url(url),
        )


class TestValidateInputPath(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def make_file(self, name):
        path = self.dir / name
        path.write_bytes(b"data")
        return path

    def test_missing_file_rejected(self):
        with self.assertRaisesRegex(ValueError, "exist"):
            ocr_service.validate_input_path(self.dir / "missing.pdf")

    def test_directory_rejected(self):
        subdir = self.dir / "folder.pdf"
        subdir.mkdir()
        with self.assertRaises(ValueError):
            ocr_service.validate_input_path(subdir)

    def test_unsupported_extension_rejected(self):
        path = self.make_file("notes.txt")
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            ocr_service.validate_input_path(path)

    def test_supported_extensions_accepted(self):
        for name in ("a.pdf", "b.png", "c.jpg", "d.jpeg", "e.webp"):
            with self.subTest(name=name):
                ocr_service.validate_input_path(self.make_file(name))

    def test_uppercase_extensions_accepted(self):
        for name in ("UPPER.PDF", "SHOUT.PNG", "MIXED.JpEg"):
            with self.subTest(name=name):
                ocr_service.validate_input_path(self.make_file(name))

    @unittest.skipIf(
        os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
        "chmod-based unreadability is not enforced for root or on Windows",
    )
    def test_unreadable_file_rejected(self):
        path = self.make_file("locked.pdf")
        path.chmod(0)
        self.addCleanup(path.chmod, 0o600)
        with self.assertRaisesRegex(ValueError, "readable"):
            ocr_service.validate_input_path(path)


class TestBuildOutputPath(unittest.TestCase):
    def test_pdf(self):
        self.assertEqual(
            ocr_service.build_output_path(Path("/docs/report.pdf")),
            Path("/docs/report_extracted.md"),
        )

    def test_image(self):
        self.assertEqual(
            ocr_service.build_output_path(Path("/pics/scan.png")),
            Path("/pics/scan_extracted.md"),
        )

    def test_dotted_stem(self):
        self.assertEqual(
            ocr_service.build_output_path(Path("/docs/report.v2.pdf")),
            Path("/docs/report.v2_extracted.md"),
        )


class TestListModels(unittest.TestCase):
    URL = "http://server:11434"

    def test_extraction_dedup_and_case_insensitive_sort(self):
        response = SimpleNamespace(
            models=[
                model_entry("zeta:7b"),
                model_entry("Alpha:12b"),
                model_entry("  "),
                model_entry("zeta:7b"),
                model_entry(None),
                model_entry("beta:2b "),
            ]
        )
        with mock.patch.object(ocr_service.ollama, "Client") as client_cls:
            client_cls.return_value.list.return_value = response
            result = ocr_service.list_models(self.URL)
        self.assertEqual(result, ["Alpha:12b", "beta:2b", "zeta:7b"])
        client_cls.assert_called_once_with(
            host=self.URL, timeout=config.MODEL_LIST_TIMEOUT
        )

    def test_empty_server_list(self):
        with mock.patch.object(ocr_service.ollama, "Client") as client_cls:
            client_cls.return_value.list.return_value = SimpleNamespace(models=[])
            self.assertEqual(ocr_service.list_models(self.URL), [])

    def test_client_construction_error_propagates_with_context(self):
        with mock.patch.object(ocr_service.ollama, "Client") as client_cls:
            client_cls.side_effect = ConnectionError("connection refused")
            with self.assertRaises(OCRServiceError) as ctx:
                ocr_service.list_models(self.URL)
        self.assertIn(self.URL, str(ctx.exception))
        self.assertIn("connection refused", str(ctx.exception))

    def test_list_call_error_propagates_with_context(self):
        with mock.patch.object(ocr_service.ollama, "Client") as client_cls:
            client_cls.return_value.list.side_effect = TimeoutError("timed out")
            with self.assertRaises(OCRServiceError) as ctx:
                ocr_service.list_models(self.URL)
        self.assertIn("timed out", str(ctx.exception))

    def test_unexpected_response_shape_wrapped_with_context(self):
        # e.g. a proxy or an incompatible client version returning a plain
        # dict instead of an object with a .models attribute.
        with mock.patch.object(ocr_service.ollama, "Client") as client_cls:
            client_cls.return_value.list.return_value = {"models": []}
            with self.assertRaises(OCRServiceError) as ctx:
                ocr_service.list_models(self.URL)
        self.assertIn(self.URL, str(ctx.exception))


class TestListLmStudioModels(unittest.TestCase):
    URL = "http://server:1234"

    def test_extraction_dedup_and_case_insensitive_sort(self):
        response = SimpleNamespace(
            data=[
                SimpleNamespace(id="zeta-7b"),
                SimpleNamespace(id="Alpha-12b"),
                SimpleNamespace(id="  "),
                SimpleNamespace(id="zeta-7b"),
                SimpleNamespace(id=None),
                SimpleNamespace(id="beta-2b "),
            ]
        )
        with mock.patch.object(ocr_service.openai, "OpenAI") as client_cls:
            client_cls.return_value.models.list.return_value = response
            result = ocr_service.list_lm_studio_models(self.URL)
        self.assertEqual(result, ["Alpha-12b", "beta-2b", "zeta-7b"])
        client_cls.assert_called_once()
        call_kwargs = client_cls.call_args.kwargs
        self.assertEqual(call_kwargs["base_url"], self.URL + "/v1")
        self.assertEqual(call_kwargs["api_key"], "lm-studio")

    def test_empty_server_list(self):
        with mock.patch.object(ocr_service.openai, "OpenAI") as client_cls:
            client_cls.return_value.models.list.return_value = SimpleNamespace(data=[])
            self.assertEqual(ocr_service.list_lm_studio_models(self.URL), [])

    def test_client_construction_error_propagates_with_context(self):
        with mock.patch.object(ocr_service.openai, "OpenAI") as client_cls:
            client_cls.side_effect = ConnectionError("connection refused")
            with self.assertRaises(OCRServiceError) as ctx:
                ocr_service.list_lm_studio_models(self.URL)
        self.assertIn(self.URL, str(ctx.exception))
        self.assertIn("connection refused", str(ctx.exception))

    def test_list_call_error_propagates_with_context(self):
        with mock.patch.object(ocr_service.openai, "OpenAI") as client_cls:
            client_cls.return_value.models.list.side_effect = TimeoutError("timed out")
            with self.assertRaises(OCRServiceError) as ctx:
                ocr_service.list_lm_studio_models(self.URL)
        self.assertIn("timed out", str(ctx.exception))


class TestOpenInDefaultApp(unittest.TestCase):
    PATH = Path("/docs/out_extracted.md")

    def test_macos_uses_open(self):
        with mock.patch.object(ocr_service.sys, "platform", "darwin"), \
                mock.patch.object(ocr_service.subprocess, "run") as run:
            ocr_service.open_in_default_app(self.PATH)
        run.assert_called_once_with(["open", str(self.PATH)], check=True)

    def test_linux_uses_xdg_open(self):
        with mock.patch.object(ocr_service.sys, "platform", "linux"), \
                mock.patch.object(ocr_service.subprocess, "run") as run:
            ocr_service.open_in_default_app(self.PATH)
        run.assert_called_once_with(["xdg-open", str(self.PATH)], check=True)

    def test_windows_uses_startfile(self):
        with mock.patch.object(ocr_service.sys, "platform", "win32"), \
                mock.patch.object(ocr_service.os, "startfile", create=True) as startfile:
            ocr_service.open_in_default_app(self.PATH)
        startfile.assert_called_once_with(str(self.PATH))

    def test_failure_wrapped(self):
        with mock.patch.object(ocr_service.sys, "platform", "darwin"), \
                mock.patch.object(
                    ocr_service.subprocess, "run",
                    side_effect=OSError("no such tool"),
                ):
            with self.assertRaises(OCRServiceError) as ctx:
                ocr_service.open_in_default_app(self.PATH)
        self.assertIn(str(self.PATH), str(ctx.exception))


class TestRevealInFileManager(unittest.TestCase):
    PATH = Path("/docs/out_extracted.md")

    def test_macos_selects_with_open_r(self):
        with mock.patch.object(ocr_service.sys, "platform", "darwin"), \
                mock.patch.object(ocr_service.subprocess, "run") as run:
            ocr_service.reveal_in_file_manager(self.PATH)
        run.assert_called_once_with(["open", "-R", str(self.PATH)], check=True)

    def test_windows_selects_with_explorer(self):
        with mock.patch.object(ocr_service.sys, "platform", "win32"), \
                mock.patch.object(ocr_service.subprocess, "run") as run:
            ocr_service.reveal_in_file_manager(self.PATH)
        run.assert_called_once_with(["explorer", f"/select,{self.PATH}"])

    def test_linux_opens_parent_directory(self):
        with mock.patch.object(ocr_service.sys, "platform", "linux"), \
                mock.patch.object(ocr_service.subprocess, "run") as run:
            ocr_service.reveal_in_file_manager(self.PATH)
        run.assert_called_once_with(["xdg-open", str(self.PATH.parent)], check=True)

    def test_failure_wrapped(self):
        with mock.patch.object(ocr_service.sys, "platform", "darwin"), \
                mock.patch.object(
                    ocr_service.subprocess, "run",
                    side_effect=OSError("boom"),
                ):
            with self.assertRaises(OCRServiceError) as ctx:
                ocr_service.reveal_in_file_manager(self.PATH)
        self.assertIn(str(self.PATH), str(ctx.exception))


class TestSaveMarkdownAtomic(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.output = self.dir / "doc_extracted.md"

    def test_writes_utf8(self):
        content = "# Überschrift\n\nТекст — naïve café ✓"
        ocr_service.save_markdown_atomic(self.output, content)
        self.assertEqual(self.output.read_text(encoding="utf-8"), content)

    def test_normalizes_newlines(self):
        ocr_service.save_markdown_atomic(self.output, "a\r\nb\rc\n")
        self.assertEqual(self.output.read_bytes(), b"a\nb\nc\n")

    def test_replaces_existing_file(self):
        self.output.write_text("old", encoding="utf-8")
        ocr_service.save_markdown_atomic(self.output, "new")
        self.assertEqual(self.output.read_text(encoding="utf-8"), "new")

    def test_no_leftover_temp_file_on_success(self):
        ocr_service.save_markdown_atomic(self.output, "content")
        self.assertEqual(list(self.dir.iterdir()), [self.output])

    def test_replace_failure_removes_temp_and_keeps_existing(self):
        self.output.write_text("old", encoding="utf-8")
        with mock.patch.object(ocr_service.os, "replace") as replace:
            replace.side_effect = OSError("disk full")
            with self.assertRaises(OCRServiceError) as ctx:
                ocr_service.save_markdown_atomic(self.output, "new")
        self.assertIn("disk full", str(ctx.exception))
        self.assertEqual(self.output.read_text(encoding="utf-8"), "old")
        self.assertEqual(list(self.dir.iterdir()), [self.output])

    def test_temp_creation_failure_reports_error(self):
        with mock.patch.object(
            ocr_service.tempfile, "NamedTemporaryFile"
        ) as ntf:
            ntf.side_effect = OSError("permission denied")
            with self.assertRaises(OCRServiceError):
                ocr_service.save_markdown_atomic(self.output, "content")
        self.assertEqual(list(self.dir.iterdir()), [])


class TestRecognizeImages(unittest.TestCase):
    MODEL = "vision-model:latest"

    def test_one_independent_request_per_image_with_exact_prompt(self):
        client = mock.MagicMock()
        client.chat.side_effect = [stream_response(" one "), stream_response("two\n")]
        paths = [Path("/imgs/page_0001.png"), Path("/imgs/page_0002.png")]
        results = ocr_service.recognize_images(
            client, self.MODEL, paths, lambda _msg: None
        )
        self.assertEqual(results, ["one", "two"])
        self.assertEqual(client.chat.call_count, 2)
        for call, path in zip(client.chat.call_args_list, paths):
            self.assertTrue(call.kwargs["stream"])
            self.assertEqual(call.kwargs["model"], self.MODEL)
            self.assertEqual(
                call.kwargs["messages"],
                [
                    {"role": "system", "content": config.SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": "Recognize this document page.",
                        "images": [str(path)],
                    },
                ],
            )
        first, second = (c.kwargs["messages"] for c in client.chat.call_args_list)
        self.assertIsNot(first, second)

    def test_progress_messages_in_order(self):
        client = mock.MagicMock()
        client.chat.side_effect = [stream_response(f"p{i}") for i in range(3)]
        logs = []
        ocr_service.recognize_images(
            client, self.MODEL, [Path(f"/i/{i}.png") for i in range(3)], logs.append
        )
        self.assertEqual(
            logs,
            [
                "Sending page 1/3 to Ollama...",
                "Sending page 2/3 to Ollama...",
                "Sending page 3/3 to Ollama...",
            ],
        )

    def test_progress_callback_emits_ocr_before_each_page(self):
        client = mock.MagicMock()
        client.chat.side_effect = [stream_response(f"p{i}") for i in range(3)]
        events = []
        ocr_service.recognize_images(
            client,
            self.MODEL,
            [Path(f"/i/{i}.png") for i in range(3)],
            lambda _msg: None,
            lambda phase, cur, tot: events.append((phase, cur, tot)),
        )
        self.assertEqual(
            events,
            [
                ("ocr", 1, 3),
                ("ocr", 2, 3),
                ("ocr", 3, 3),
            ],
        )
        # Each progress event fires before the corresponding chat call.
        self.assertEqual(client.chat.call_count, 3)

    def test_empty_content_fails_identifying_page(self):
        for empty in (None, "", "   \n\t"):
            with self.subTest(content=repr(empty)):
                client = mock.MagicMock()
                client.chat.side_effect = [
                    stream_response("fine"),
                    stream_response(empty),
                ]
                with self.assertRaises(OCRServiceError) as ctx:
                    ocr_service.recognize_images(
                        client,
                        self.MODEL,
                        [Path("/i/1.png"), Path("/i/2.png")],
                        lambda _msg: None,
                    )
                self.assertIn("page 2/2", str(ctx.exception))

    def test_empty_stream_fails_identifying_page(self):
        """A stream that yields zero chunks triggers 'returned no text'."""
        client = mock.MagicMock()
        client.chat.side_effect = [stream_response("fine"), stream_response()]
        with self.assertRaises(OCRServiceError) as ctx:
            ocr_service.recognize_images(
                client,
                self.MODEL,
                [Path("/i/1.png"), Path("/i/2.png")],
                lambda _msg: None,
            )
        self.assertIn("page 2/2", str(ctx.exception))

    def test_chat_failure_wrapped_with_page_and_model_context(self):
        client = mock.MagicMock()
        client.chat.side_effect = [
            stream_response("ok"),
            RuntimeError("model not found"),
        ]
        with self.assertRaises(OCRServiceError) as ctx:
            ocr_service.recognize_images(
                client,
                self.MODEL,
                [Path("/i/1.png"), Path("/i/2.png")],
                lambda _msg: None,
            )
        message = str(ctx.exception)
        self.assertIn("page 2/2", message)
        self.assertIn(self.MODEL, message)
        self.assertIn("model not found", message)

    def test_generator_exception_mid_stream_wrapped_with_context(self):
        """An exception raised while iterating the stream is wrapped."""

        def exploding_stream():
            yield SimpleNamespace(message=SimpleNamespace(content="part"))
            raise RuntimeError("connection dropped")

        client = mock.MagicMock()
        client.chat.side_effect = [stream_response("ok"), exploding_stream()]
        with self.assertRaises(OCRServiceError) as ctx:
            ocr_service.recognize_images(
                client,
                self.MODEL,
                [Path("/i/1.png"), Path("/i/2.png")],
                lambda _msg: None,
            )
        message = str(ctx.exception)
        self.assertIn("page 2/2", message)
        self.assertIn("connection dropped", message)

    def test_stream_chunk_events_emitted_in_order(self):
        """stream_chunk events carry each delta, in order, with page number."""
        client = mock.MagicMock()
        client.chat.side_effect = [
            stream_response("Hel", "lo", " world"),
            stream_response("foo"),
        ]
        events = []
        ocr_service.recognize_images(
            client,
            self.MODEL,
            [Path("/i/1.png"), Path("/i/2.png")],
            lambda _msg: None,
            event_callback=lambda kind, payload: events.append((kind, payload)),
        )
        stream_events = [(k, p) for k, p in events if k == "stream_chunk"]
        self.assertEqual(
            stream_events,
            [
                ("stream_chunk", {"page": 1, "text": "Hel"}),
                ("stream_chunk", {"page": 1, "text": "lo"}),
                ("stream_chunk", {"page": 1, "text": " world"}),
                ("stream_chunk", {"page": 2, "text": "foo"}),
            ],
        )

    def test_page_text_events_emitted_after_each_page(self):
        """page_text events carry the assembled (stripped) text per page."""
        client = mock.MagicMock()
        client.chat.side_effect = [
            stream_response("  Hello ", "world  "),
            stream_response("second"),
        ]
        events = []
        results = ocr_service.recognize_images(
            client,
            self.MODEL,
            [Path("/i/1.png"), Path("/i/2.png")],
            lambda _msg: None,
            event_callback=lambda kind, payload: events.append((kind, payload)),
        )
        page_events = [(k, p) for k, p in events if k == "page_text"]
        self.assertEqual(
            page_events,
            [
                ("page_text", {"page": 1, "total": 2, "text": "Hello world"}),
                ("page_text", {"page": 2, "total": 2, "text": "second"}),
            ],
        )
        # The returned texts match the page_text payloads (stripped).
        self.assertEqual(results, ["Hello world", "second"])

    def test_page_image_event_before_send_with_valid_png(self):
        """A page_image event with valid PNG bytes precedes the send log."""
        import io as _io

        from PIL import Image

        with tempfile.TemporaryDirectory() as d:
            img_path = Path(d) / "page.png"
            Image.new("RGB", (400, 300), (10, 20, 30)).save(img_path)
            client = mock.MagicMock()
            client.chat.side_effect = [stream_response("text")]
            timeline = []
            ocr_service.recognize_images(
                client,
                self.MODEL,
                [img_path],
                lambda msg: timeline.append(("log", msg)),
                event_callback=lambda kind, payload: timeline.append((kind, payload)),
            )
        kinds = [entry[0] for entry in timeline]
        self.assertIn("page_image", kinds)
        image_index = kinds.index("page_image")
        send_index = next(
            i for i, entry in enumerate(timeline)
            if entry[0] == "log" and "Sending page" in entry[1]
        )
        self.assertLess(image_index, send_index)
        payload = timeline[image_index][1]
        self.assertEqual(payload["page"], 1)
        self.assertEqual(payload["total"], 1)
        Image.open(_io.BytesIO(payload["png"])).verify()

    def test_thumbnail_failure_does_not_abort_ocr(self):
        """A missing image file skips the preview but still recognizes."""
        client = mock.MagicMock()
        client.chat.side_effect = [stream_response("recognized")]
        events = []
        results = ocr_service.recognize_images(
            client,
            self.MODEL,
            [Path("/does/not/exist.png")],
            lambda _msg: None,
            event_callback=lambda kind, payload: events.append((kind, payload)),
        )
        self.assertEqual(results, ["recognized"])
        self.assertNotIn("page_image", [kind for kind, _ in events])


class TestMakeThumbnailPng(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def _write_image(self, name, size, color=(120, 60, 200)):
        from PIL import Image

        path = self.dir / name
        Image.new("RGB", size, color).save(path)
        return path

    def _open(self, data):
        import io as _io

        from PIL import Image

        return Image.open(_io.BytesIO(data))

    def test_downscales_png_preserving_aspect_and_returns_png(self):
        path = self._write_image("big.png", (2000, 1000))
        data = ocr_service.make_thumbnail_png(path, 900)
        self.assertIsInstance(data, bytes)
        out = self._open(data)
        self.assertEqual(out.format, "PNG")
        self.assertEqual(out.size, (900, 450))  # 2:1 aspect preserved

    def test_jpeg_input_supported(self):
        path = self._write_image("photo.jpg", (1200, 800))
        out = self._open(ocr_service.make_thumbnail_png(path, 600))
        self.assertEqual(out.format, "PNG")
        self.assertLessEqual(max(out.size), 600)

    def test_small_image_not_upscaled(self):
        path = self._write_image("small.png", (100, 50))
        out = self._open(ocr_service.make_thumbnail_png(path, 900))
        self.assertEqual(out.size, (100, 50))

    def test_real_pdf_rendered_page(self):
        """The pipeline feeds PNGs rendered from PDF pages — thumbnail one."""
        document = ocr_service.pymupdf.open()
        document.new_page(width=1200, height=1600)
        pixmap = document.load_page(0).get_pixmap(dpi=150)
        path = self.dir / "page_0001.png"
        pixmap.save(str(path))
        document.close()
        out = self._open(ocr_service.make_thumbnail_png(path, 900))
        self.assertEqual(out.format, "PNG")
        self.assertLessEqual(max(out.size), 900)


class TestRenderPdf(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.temp_dir = Path(self.tmp.name)
        self.pdf_path = Path("/docs/input.pdf")

    def test_ordered_render_with_dpi_rgb_no_alpha(self):
        document = make_fake_document(3)
        logs = []
        with mock.patch.object(ocr_service, "pymupdf") as fake_pymupdf:
            fake_pymupdf.open.return_value = document
            paths = ocr_service.render_pdf(
                self.pdf_path, 200, self.temp_dir, logs.append
            )
        fake_pymupdf.open.assert_called_once_with(self.pdf_path)
        self.assertEqual(
            document.load_page.call_args_list,
            [mock.call(0), mock.call(1), mock.call(2)],
        )
        for page in document.fake_pages:
            page.get_pixmap.assert_called_once_with(
                dpi=200, colorspace=fake_pymupdf.csRGB, alpha=False
            )
        self.assertEqual(
            [p.name for p in paths],
            ["page_0001.png", "page_0002.png", "page_0003.png"],
        )
        self.assertTrue(all(p.parent == self.temp_dir for p in paths))
        for page, path in zip(document.fake_pages, paths):
            page.get_pixmap.return_value.save.assert_called_once_with(str(path))
        self.assertEqual(
            logs,
            [
                "Rendering page 1/3...",
                "Rendering page 2/3...",
                "Rendering page 3/3...",
            ],
        )
        self.assertTrue(document.__exit__.called)

    def test_zero_padding_keeps_numeric_order_past_page_nine(self):
        document = make_fake_document(12)
        with mock.patch.object(ocr_service, "pymupdf") as fake_pymupdf:
            fake_pymupdf.open.return_value = document
            paths = ocr_service.render_pdf(
                self.pdf_path, 150, self.temp_dir, lambda _msg: None
            )
        names = [p.name for p in paths]
        self.assertEqual(names[9], "page_0010.png")
        self.assertEqual(names, sorted(names))

    def test_password_protected_fails_before_rendering(self):
        document = make_fake_document(5, needs_pass=True)
        with mock.patch.object(ocr_service, "pymupdf") as fake_pymupdf:
            fake_pymupdf.open.return_value = document
            with self.assertRaisesRegex(OCRServiceError, "password"):
                ocr_service.render_pdf(
                    self.pdf_path, 150, self.temp_dir, lambda _msg: None
                )
        document.load_page.assert_not_called()
        self.assertTrue(document.__exit__.called)

    def test_zero_page_pdf_fails(self):
        document = make_fake_document(0)
        with mock.patch.object(ocr_service, "pymupdf") as fake_pymupdf:
            fake_pymupdf.open.return_value = document
            with self.assertRaisesRegex(OCRServiceError, "no pages"):
                ocr_service.render_pdf(
                    self.pdf_path, 150, self.temp_dir, lambda _msg: None
                )
        document.load_page.assert_not_called()
        self.assertTrue(document.__exit__.called)

    def test_open_failure_wrapped(self):
        with mock.patch.object(ocr_service, "pymupdf") as fake_pymupdf:
            fake_pymupdf.open.side_effect = RuntimeError("broken xref")
            with self.assertRaises(OCRServiceError) as ctx:
                ocr_service.render_pdf(
                    self.pdf_path, 150, self.temp_dir, lambda _msg: None
                )
        self.assertIn("broken xref", str(ctx.exception))

    def test_page_render_failure_identifies_page_and_closes_document(self):
        document = make_fake_document(3)
        document.fake_pages[1].get_pixmap.return_value.save.side_effect = OSError(
            "write failed"
        )
        with mock.patch.object(ocr_service, "pymupdf") as fake_pymupdf:
            fake_pymupdf.open.return_value = document
            with self.assertRaises(OCRServiceError) as ctx:
                ocr_service.render_pdf(
                    self.pdf_path, 150, self.temp_dir, lambda _msg: None
                )
        self.assertIn("page 2/3", str(ctx.exception))
        self.assertTrue(document.__exit__.called)

    def test_progress_callback_emits_render_before_each_page(self):
        document = make_fake_document(3)
        events = []
        with mock.patch.object(ocr_service, "pymupdf") as fake_pymupdf:
            fake_pymupdf.open.return_value = document
            ocr_service.render_pdf(
                self.pdf_path, 150, self.temp_dir, lambda _msg: None,
                lambda phase, cur, tot: events.append((phase, cur, tot)),
            )
        self.assertEqual(
            events,
            [
                ("render", 1, 3),
                ("render", 2, 3),
                ("render", 3, 3),
            ],
        )
        # Each progress event fires before the corresponding page render.
        self.assertEqual(len(document.fake_pages), 3)


class TestProcessOcr(unittest.TestCase):
    URL = "http://server:11434"
    MODEL = "vision:7b"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def make_request(self, name, dpi=150):
        input_path = self.dir / name
        input_path.write_bytes(b"fake bytes")
        return OCRRequest(
            input_path=input_path,
            output_path=ocr_service.build_output_path(input_path),
            provider=config.Provider.OLLAMA,
            server_url=self.URL,
            model=self.MODEL,
            dpi=dpi,
        )

    def run_pdf_pipeline(self, request, document, chat_side_effect,
                         replace_error=None):
        """Run process_ocr for a PDF with mocks; return (result, error, events,
        created_temp_dirs)."""
        events = queue.Queue()
        created = []
        real_mkdtemp = tempfile.mkdtemp

        def spy_mkdtemp(*args, **kwargs):
            path = real_mkdtemp(*args, **kwargs)
            created.append(path)
            return path

        patches = [
            mock.patch.object(ocr_service, "pymupdf"),
            mock.patch.object(ocr_service.ollama, "Client"),
            mock.patch.object(ocr_service.tempfile, "mkdtemp", side_effect=spy_mkdtemp),
        ]
        result = error = None
        with patches[0] as fake_pymupdf, patches[1] as client_cls, patches[2]:
            fake_pymupdf.open.return_value = document
            client_cls.return_value.chat.side_effect = chat_side_effect
            self.client_cls = client_cls
            try:
                if replace_error is not None:
                    with mock.patch.object(
                        ocr_service.os, "replace", side_effect=replace_error
                    ):
                        result = ocr_service.process_ocr(request, events)
                else:
                    result = ocr_service.process_ocr(request, events)
            except Exception as exc:
                error = exc
        return result, error, drain(events), created

    def test_pdf_progress_events_render_then_ocr(self):
        """A 3-page PDF emits 3 render events then 3 ocr events, in order."""
        request = self.make_request("doc.pdf")
        document = make_fake_document(3)
        _result, _error, events, _created = self.run_pdf_pipeline(
            request,
            document,
            [stream_response("p1"), stream_response("p2"), stream_response("p3")],
        )
        progress_events = [
            payload for kind, payload in events if kind == "progress"
        ]
        self.assertEqual(len(progress_events), 6)
        # First three are render phase.
        for i in range(3):
            self.assertEqual(progress_events[i]["phase"], "render")
            self.assertEqual(progress_events[i]["current"], i + 1)
            self.assertEqual(progress_events[i]["total"], 3)
        # Last three are ocr phase.
        for i in range(3):
            self.assertEqual(progress_events[3 + i]["phase"], "ocr")
            self.assertEqual(progress_events[3 + i]["current"], i + 1)
            self.assertEqual(progress_events[3 + i]["total"], 3)

    def test_image_progress_events_only_ocr(self):
        """An image input emits exactly one ocr progress event, no render."""
        request = self.make_request("photo.png")
        events = queue.Queue()
        with mock.patch.object(ocr_service.ollama, "Client") as client_cls:
            client_cls.return_value.chat.return_value = stream_response("recognized")
            ocr_service.process_ocr(request, events)
        progress_events = [
            payload for kind, payload in drain(events) if kind == "progress"
        ]
        self.assertEqual(len(progress_events), 1)
        self.assertEqual(progress_events[0]["phase"], "ocr")
        self.assertEqual(progress_events[0]["current"], 1)
        self.assertEqual(progress_events[0]["total"], 1)

    def test_image_input_passed_directly_no_render_dir(self):
        request = self.make_request("photo.png")
        events = queue.Queue()
        with mock.patch.object(ocr_service.ollama, "Client") as client_cls, \
                mock.patch.object(ocr_service.tempfile, "mkdtemp") as mkdtemp:
            client_cls.return_value.chat.return_value = stream_response("recognized")
            result = ocr_service.process_ocr(request, events)
        mkdtemp.assert_not_called()
        self.assertEqual(result, request.output_path)
        self.assertEqual(
            request.output_path.read_text(encoding="utf-8"), "recognized"
        )
        client_cls.assert_called_once_with(
            host=self.URL, timeout=config.OCR_STREAM_IDLE_TIMEOUT
        )
        self.assertTrue(client_cls.return_value.chat.call_args.kwargs["stream"])
        images = client_cls.return_value.chat.call_args.kwargs["messages"][1]["images"]
        self.assertEqual(images, [str(request.input_path)])
        logs = [payload for kind, payload in drain(events) if kind == "log"]
        self.assertEqual(logs[0], "[1/3] Preparing image...")
        self.assertIn("[2/3] Sending page 1/1 to Ollama...", logs)
        self.assertIn("[3/3] Saving Markdown...", logs)

    def test_pdf_pipeline_order_join_and_cleanup(self):
        request = self.make_request("doc.pdf", dpi=300)
        document = make_fake_document(3)
        result, error, events, created = self.run_pdf_pipeline(
            request,
            document,
            [stream_response("p1"), stream_response("p2"), stream_response("p3")],
        )
        self.assertIsNone(error)
        self.assertEqual(result, request.output_path)
        self.assertEqual(
            request.output_path.read_text(encoding="utf-8"), "p1\n\np2\n\np3"
        )
        # Temp dir was created with the required prefix and removed afterwards.
        self.assertEqual(len(created), 1)
        self.assertTrue(Path(created[0]).name.startswith("local_ocr_"))
        self.assertFalse(Path(created[0]).exists())
        # Rendering finished before recognition started.
        logs = [payload for kind, payload in events if kind == "log"]
        self.assertEqual(logs[0], "[1/3] Preparing document...")
        last_render = max(i for i, m in enumerate(logs) if m.startswith("[1/3]"))
        first_send = min(i for i, m in enumerate(logs) if m.startswith("[2/3]"))
        self.assertLess(last_render, first_send)
        # Pages were sent in numeric order from the render directory.
        calls = self.client_cls.return_value.chat.call_args_list
        sent = [c.kwargs["messages"][1]["images"][0] for c in calls]
        expected = [
            str(Path(created[0]) / f"page_{n:04d}.png") for n in (1, 2, 3)
        ]
        self.assertEqual(sent, expected)

    def test_render_failure_cleans_temp_dir_and_no_output(self):
        request = self.make_request("doc.pdf")
        document = make_fake_document(2)
        document.fake_pages[0].get_pixmap.side_effect = RuntimeError("render boom")
        result, error, _events, created = self.run_pdf_pipeline(
            request, document, []
        )
        self.assertIsNone(result)
        self.assertIsInstance(error, OCRServiceError)
        self.assertFalse(Path(created[0]).exists())
        self.assertFalse(request.output_path.exists())

    def test_client_construction_failure_cleans_temp_dir(self):
        request = self.make_request("doc.pdf")
        document = make_fake_document(1)
        events = queue.Queue()
        created = []
        real_mkdtemp = tempfile.mkdtemp

        def spy_mkdtemp(*args, **kwargs):
            path = real_mkdtemp(*args, **kwargs)
            created.append(path)
            return path

        with mock.patch.object(ocr_service, "pymupdf") as fake_pymupdf, \
                mock.patch.object(ocr_service.ollama, "Client") as client_cls, \
                mock.patch.object(
                    ocr_service.tempfile, "mkdtemp", side_effect=spy_mkdtemp
                ):
            fake_pymupdf.open.return_value = document
            client_cls.side_effect = ConnectionError("no route to host")
            with self.assertRaises(OCRServiceError) as ctx:
                ocr_service.process_ocr(request, events)
        self.assertIn("no route to host", str(ctx.exception))
        self.assertFalse(Path(created[0]).exists())
        self.assertFalse(request.output_path.exists())

    def test_mkdtemp_failure_wrapped_and_no_output(self):
        request = self.make_request("doc.pdf")
        events = queue.Queue()
        with mock.patch.object(
            ocr_service.tempfile, "mkdtemp", side_effect=OSError("no space left")
        ):
            with self.assertRaises(OCRServiceError) as ctx:
                ocr_service.process_ocr(request, events)
        self.assertIn("no space left", str(ctx.exception))
        self.assertFalse(request.output_path.exists())

    def test_late_page_failure_leaves_no_new_output(self):
        request = self.make_request("doc.pdf")
        document = make_fake_document(2)
        result, error, _events, created = self.run_pdf_pipeline(
            request,
            document,
            [stream_response("p1"), RuntimeError("model exploded")],
        )
        self.assertIsNone(result)
        self.assertIsInstance(error, OCRServiceError)
        self.assertFalse(request.output_path.exists())
        self.assertFalse(Path(created[0]).exists())

    def test_late_page_failure_preserves_existing_output(self):
        request = self.make_request("doc.pdf")
        request.output_path.write_text("previous run", encoding="utf-8")
        document = make_fake_document(2)
        result, error, _events, created = self.run_pdf_pipeline(
            request,
            document,
            [stream_response("p1"), RuntimeError("model exploded")],
        )
        self.assertIsNone(result)
        self.assertIsNotNone(error)
        self.assertEqual(
            request.output_path.read_text(encoding="utf-8"), "previous run"
        )
        self.assertFalse(Path(created[0]).exists())

    def test_save_failure_cleans_temp_dir_and_preserves_existing_output(self):
        request = self.make_request("doc.pdf")
        request.output_path.write_text("previous run", encoding="utf-8")
        document = make_fake_document(1)
        result, error, _events, created = self.run_pdf_pipeline(
            request,
            document,
            [stream_response("p1")],
            replace_error=OSError("disk full"),
        )
        self.assertIsNone(result)
        self.assertIsInstance(error, OCRServiceError)
        self.assertEqual(
            request.output_path.read_text(encoding="utf-8"), "previous run"
        )
        self.assertFalse(Path(created[0]).exists())
        # No stray temp output file remains next to the output either.
        leftovers = [p for p in self.dir.iterdir() if p.suffix == ".tmp"]
        self.assertEqual(leftovers, [])


    def test_page_text_events_through_pipeline(self):
        """process_ocr emits page_text events with assembled text per page."""
        request = self.make_request("doc.pdf")
        document = make_fake_document(2)
        _result, _error, events, _created = self.run_pdf_pipeline(
            request,
            document,
            [stream_response("  page one ", "text"), stream_response("two")],
        )
        page_events = [
            (kind, payload) for kind, payload in events if kind == "page_text"
        ]
        self.assertEqual(
            page_events,
            [
                ("page_text", {"page": 1, "total": 2, "text": "page one text"}),
                ("page_text", {"page": 2, "total": 2, "text": "two"}),
            ],
        )

    def test_stream_chunk_events_through_pipeline(self):
        """process_ocr emits stream_chunk events for each delta."""
        request = self.make_request("photo.png")
        events = queue.Queue()
        with mock.patch.object(ocr_service.ollama, "Client") as client_cls:
            client_cls.return_value.chat.return_value = stream_response("A", "B", "C")
            ocr_service.process_ocr(request, events)
        chunk_events = [
            (kind, payload) for kind, payload in drain(events) if kind == "stream_chunk"
        ]
        self.assertEqual(
            chunk_events,
            [
                ("stream_chunk", {"page": 1, "text": "A"}),
                ("stream_chunk", {"page": 1, "text": "B"}),
                ("stream_chunk", {"page": 1, "text": "C"}),
            ],
        )

    def test_mid_stream_generator_exception_cleans_temp_dir(self):
        """An exception while iterating a stream cleans the temp dir."""

        def exploding_stream():
            yield SimpleNamespace(message=SimpleNamespace(content="partial"))
            raise RuntimeError("connection dropped")

        request = self.make_request("doc.pdf")
        document = make_fake_document(2)
        result, error, _events, created = self.run_pdf_pipeline(
            request,
            document,
            [exploding_stream(), stream_response("p2")],
        )
        self.assertIsNone(result)
        self.assertIsInstance(error, OCRServiceError)
        self.assertIn("page 1/2", str(error))
        self.assertIn("connection dropped", str(error))
        self.assertFalse(Path(created[0]).exists())
        self.assertFalse(request.output_path.exists())

    def test_empty_stream_through_pipeline_fails(self):
        """A stream with zero chunks triggers 'returned no text'."""
        request = self.make_request("photo.png")
        events = queue.Queue()
        with mock.patch.object(ocr_service.ollama, "Client") as client_cls:
            client_cls.return_value.chat.return_value = stream_response()
            error = None
            try:
                ocr_service.process_ocr(request, events)
            except Exception as exc:
                error = exc
        self.assertIsInstance(error, OCRServiceError)
        self.assertIn("returned no text", str(error))
        self.assertIn("page 1/1", str(error))
        self.assertFalse(request.output_path.exists())


class TestRecognizeImagesOpenAI(unittest.TestCase):
    """Tests for the OpenAI-compatible recognition path (LM Studio, vLLM)."""

    MODEL = "llama-3.2-vision"

    def _openai_stream_chunk(self, content):
        """Shape of openai streaming chunks: choices[0].delta.content."""
        delta = SimpleNamespace(content=content)
        choice = SimpleNamespace(delta=delta)
        return SimpleNamespace(choices=[choice])

    def _patch_encode(self):
        """Patch encode_image_as_base64 so tests don't need real image files."""
        return mock.patch.object(
            ocr_service, "encode_image_as_base64",
            return_value="data:image/png;base64,fake",
        )

    def test_one_request_per_image_with_base64_and_prompt(self):
        client = mock.MagicMock()
        client.chat.completions.create.side_effect = [
            iter([self._openai_stream_chunk(" one ")]),
            iter([self._openai_stream_chunk("two\n")]),
        ]
        paths = [Path("/imgs/page_0001.png"), Path("/imgs/page_0002.png")]
        with self._patch_encode():
            results = ocr_service.recognize_images_openai(
                client, self.MODEL, paths, lambda _msg: None
            )
        self.assertEqual(results, ["one", "two"])
        self.assertEqual(client.chat.completions.create.call_count, 2)
        for call, path in zip(client.chat.completions.create.call_args_list, paths):
            self.assertTrue(call.kwargs["stream"])
            self.assertEqual(call.kwargs["model"], self.MODEL)
            messages = call.kwargs["messages"]
            self.assertEqual(messages[0]["role"], "system")
            self.assertEqual(messages[0]["content"], config.SYSTEM_PROMPT)
            user_msg = messages[1]
            self.assertEqual(user_msg["role"], "user")
            content_parts = user_msg["content"]
            self.assertEqual(content_parts[0]["type"], "text")
            self.assertEqual(content_parts[0]["text"], config.USER_PROMPT)
            self.assertEqual(content_parts[1]["type"], "image_url")
            self.assertEqual(
                content_parts[1]["image_url"]["url"],
                "data:image/png;base64,fake",
            )

    def test_progress_callback_emits_ocr_before_each_page(self):
        client = mock.MagicMock()
        client.chat.completions.create.side_effect = [
            iter([self._openai_stream_chunk(f"p{i}")]) for i in range(3)
        ]
        events = []
        with self._patch_encode():
            ocr_service.recognize_images_openai(
                client,
                self.MODEL,
                [Path(f"/i/{i}.png") for i in range(3)],
                lambda _msg: None,
                lambda phase, cur, tot: events.append((phase, cur, tot)),
            )
        self.assertEqual(
            events,
            [("ocr", 1, 3), ("ocr", 2, 3), ("ocr", 3, 3)],
        )

    def test_empty_content_fails_identifying_page(self):
        for empty in (None, "", "   \n\t"):
            with self.subTest(content=repr(empty)):
                client = mock.MagicMock()
                client.chat.completions.create.side_effect = [
                    iter([self._openai_stream_chunk("fine")]),
                    iter([self._openai_stream_chunk(empty)]),
                ]
                with self._patch_encode():
                    with self.assertRaises(OCRServiceError) as ctx:
                        ocr_service.recognize_images_openai(
                            client,
                            self.MODEL,
                            [Path("/i/1.png"), Path("/i/2.png")],
                            lambda _msg: None,
                        )
                self.assertIn("page 2/2", str(ctx.exception))

    def test_chat_failure_wrapped_with_page_and_model_context(self):
        client = mock.MagicMock()
        client.chat.completions.create.side_effect = [
            iter([self._openai_stream_chunk("ok")]),
            RuntimeError("model not found"),
        ]
        with self._patch_encode():
            with self.assertRaises(OCRServiceError) as ctx:
                ocr_service.recognize_images_openai(
                    client,
                    self.MODEL,
                    [Path("/i/1.png"), Path("/i/2.png")],
                    lambda _msg: None,
                )
        message = str(ctx.exception)
        self.assertIn("page 2/2", message)
        self.assertIn(self.MODEL, message)
        self.assertIn("model not found", message)

    def test_stream_chunk_events_emitted_in_order(self):
        client = mock.MagicMock()
        client.chat.completions.create.side_effect = [
            iter([
                self._openai_stream_chunk("Hel"),
                self._openai_stream_chunk("lo"),
                self._openai_stream_chunk(" world"),
            ]),
            iter([self._openai_stream_chunk("foo")]),
        ]
        events = []
        with self._patch_encode():
            ocr_service.recognize_images_openai(
                client,
                self.MODEL,
                [Path("/i/1.png"), Path("/i/2.png")],
                lambda _msg: None,
                event_callback=lambda kind, payload: events.append((kind, payload)),
            )
        stream_events = [(k, p) for k, p in events if k == "stream_chunk"]
        self.assertEqual(
            stream_events,
            [
                ("stream_chunk", {"page": 1, "text": "Hel"}),
                ("stream_chunk", {"page": 1, "text": "lo"}),
                ("stream_chunk", {"page": 1, "text": " world"}),
                ("stream_chunk", {"page": 2, "text": "foo"}),
            ],
        )

    def test_page_text_events_emitted_after_each_page(self):
        client = mock.MagicMock()
        client.chat.completions.create.side_effect = [
            iter([
                self._openai_stream_chunk("  Hello "),
                self._openai_stream_chunk("world  "),
            ]),
            iter([self._openai_stream_chunk("second")]),
        ]
        events = []
        with self._patch_encode():
            results = ocr_service.recognize_images_openai(
                client,
                self.MODEL,
                [Path("/i/1.png"), Path("/i/2.png")],
                lambda _msg: None,
                event_callback=lambda kind, payload: events.append((kind, payload)),
            )
        page_events = [(k, p) for k, p in events if k == "page_text"]
        self.assertEqual(
            page_events,
            [
                ("page_text", {"page": 1, "total": 2, "text": "Hello world"}),
                ("page_text", {"page": 2, "total": 2, "text": "second"}),
            ],
        )
        self.assertEqual(results, ["Hello world", "second"])

    def test_provider_name_in_error_messages(self):
        """The provider_name parameter appears in error messages."""
        client = mock.MagicMock()
        client.chat.completions.create.side_effect = [
            iter([self._openai_stream_chunk("ok")]),
            RuntimeError("model not found"),
        ]
        with self._patch_encode():
            with self.assertRaises(OCRServiceError) as ctx:
                ocr_service.recognize_images_openai(
                    client,
                    self.MODEL,
                    [Path("/i/1.png"), Path("/i/2.png")],
                    lambda _msg: None,
                    provider_name="vLLM",
                )
        self.assertIn("vLLM request failed", str(ctx.exception))


class TestProcessOcrOpenAI(unittest.TestCase):
    """Integration-style tests for process_ocr with OpenAI-compatible providers."""

    URL = "http://server:1234"
    MODEL = "llama-3.2-vision"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def make_request(self, name, dpi=150, provider=config.Provider.LM_STUDIO):
        input_path = self.dir / name
        input_path.write_bytes(b"fake bytes")
        return OCRRequest(
            input_path=input_path,
            output_path=ocr_service.build_output_path(input_path),
            provider=provider,
            server_url=self.URL,
            model=self.MODEL,
            dpi=dpi,
        )

    def _openai_stream_chunk(self, content):
        delta = SimpleNamespace(content=content)
        choice = SimpleNamespace(delta=delta)
        return SimpleNamespace(choices=[choice])

    def test_image_input_uses_openai_client(self):
        request = self.make_request("photo.png")
        events = queue.Queue()
        with mock.patch.object(ocr_service.openai, "OpenAI") as client_cls, \
                mock.patch.object(
                    ocr_service, "encode_image_as_base64",
                    return_value="data:image/png;base64,fake",
                ):
            client_cls.return_value.chat.completions.create.return_value = iter([
                self._openai_stream_chunk("recognized")
            ])
            result = ocr_service.process_ocr(request, events)
        self.assertEqual(result, request.output_path)
        self.assertEqual(
            request.output_path.read_text(encoding="utf-8"), "recognized"
        )
        client_cls.assert_called_once()
        call_kwargs = client_cls.call_args.kwargs
        self.assertEqual(call_kwargs["base_url"], self.URL + "/v1")
        self.assertEqual(call_kwargs["api_key"], "lm-studio")
        self.assertTrue(
            client_cls.return_value.chat.completions.create.call_args.kwargs["stream"]
        )
        logs = [payload for kind, payload in drain(events) if kind == "log"]
        self.assertEqual(logs[0], "[1/3] Preparing image...")
        self.assertIn("[2/3] Sending page 1/1 to LM Studio...", logs)
        self.assertIn("[3/3] Saving Markdown...", logs)

    def test_pdf_pipeline_uses_openai_client(self):
        request = self.make_request("doc.pdf")
        document = make_fake_document(2)
        events = queue.Queue()
        created = []
        real_mkdtemp = tempfile.mkdtemp

        def spy_mkdtemp(*args, **kwargs):
            path = real_mkdtemp(*args, **kwargs)
            created.append(path)
            return path

        with mock.patch.object(ocr_service, "pymupdf") as fake_pymupdf, \
                mock.patch.object(ocr_service.openai, "OpenAI") as client_cls, \
                mock.patch.object(
                    ocr_service.tempfile, "mkdtemp", side_effect=spy_mkdtemp
                ), \
                mock.patch.object(
                    ocr_service, "encode_image_as_base64",
                    return_value="data:image/png;base64,fake",
                ):
            fake_pymupdf.open.return_value = document
            client_cls.return_value.chat.completions.create.side_effect = [
                iter([self._openai_stream_chunk("p1")]),
                iter([self._openai_stream_chunk("p2")]),
            ]
            result = ocr_service.process_ocr(request, events)
        self.assertEqual(result, request.output_path)
        self.assertEqual(
            request.output_path.read_text(encoding="utf-8"), "p1\n\np2"
        )
        self.assertEqual(len(created), 1)
        self.assertFalse(Path(created[0]).exists())

    def test_vllm_provider_uses_openai_client(self):
        """vLLM provider should use the same OpenAI-compatible path."""
        request = self.make_request("photo.png", provider=config.Provider.VLLM)
        events = queue.Queue()
        with mock.patch.object(ocr_service.openai, "OpenAI") as client_cls, \
                mock.patch.object(
                    ocr_service, "encode_image_as_base64",
                    return_value="data:image/png;base64,fake",
                ):
            client_cls.return_value.chat.completions.create.return_value = iter([
                self._openai_stream_chunk("recognized")
            ])
            result = ocr_service.process_ocr(request, events)
        self.assertEqual(result, request.output_path)
        self.assertEqual(
            request.output_path.read_text(encoding="utf-8"), "recognized"
        )
        client_cls.assert_called_once()
        call_kwargs = client_cls.call_args.kwargs
        self.assertEqual(call_kwargs["base_url"], self.URL + "/v1")
        logs = [payload for kind, payload in drain(events) if kind == "log"]
        self.assertIn("[2/3] Sending page 1/1 to vLLM...", logs)


class TestListVllmModels(unittest.TestCase):
    URL = "http://server:8000"

    def test_extraction_dedup_and_case_insensitive_sort(self):
        response = SimpleNamespace(
            data=[
                SimpleNamespace(id="zeta-7b"),
                SimpleNamespace(id="Alpha-12b"),
                SimpleNamespace(id="  "),
                SimpleNamespace(id="zeta-7b"),
                SimpleNamespace(id=None),
                SimpleNamespace(id="beta-2b "),
            ]
        )
        with mock.patch.object(ocr_service.openai, "OpenAI") as client_cls:
            client_cls.return_value.models.list.return_value = response
            result = ocr_service.list_vllm_models(self.URL)
        self.assertEqual(result, ["Alpha-12b", "beta-2b", "zeta-7b"])
        client_cls.assert_called_once()
        call_kwargs = client_cls.call_args.kwargs
        self.assertEqual(call_kwargs["base_url"], self.URL + "/v1")
        self.assertEqual(call_kwargs["api_key"], "lm-studio")

    def test_empty_server_list(self):
        with mock.patch.object(ocr_service.openai, "OpenAI") as client_cls:
            client_cls.return_value.models.list.return_value = SimpleNamespace(data=[])
            self.assertEqual(ocr_service.list_vllm_models(self.URL), [])

    def test_client_construction_error_propagates_with_context(self):
        with mock.patch.object(ocr_service.openai, "OpenAI") as client_cls:
            client_cls.side_effect = ConnectionError("connection refused")
            with self.assertRaises(OCRServiceError) as ctx:
                ocr_service.list_vllm_models(self.URL)
        self.assertIn(self.URL, str(ctx.exception))
        self.assertIn("connection refused", str(ctx.exception))

    def test_list_call_error_propagates_with_context(self):
        with mock.patch.object(ocr_service.openai, "OpenAI") as client_cls:
            client_cls.return_value.models.list.side_effect = TimeoutError("timed out")
            with self.assertRaises(OCRServiceError) as ctx:
                ocr_service.list_vllm_models(self.URL)
        self.assertIn("timed out", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
