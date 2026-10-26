import asyncio
import contextlib
import importlib.util
import io
import json
import logging
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from app.privacy import RequestAuditMiddleware, SafeFormatter

spec = importlib.util.spec_from_file_location(
    "download_model", Path(__file__).resolve().parents[1] / "downloader/download_model.py"
)
downloader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(downloader)


class PrivacyTests(unittest.TestCase):
    def test_untrusted_message_arguments_and_traceback_are_never_formatted(self):
        record = logging.LogRecord("vllm", logging.ERROR, "", 0, "PROMPT %s", ("ANSWER",), None)
        try:
            raise ValueError("SECRET_TOKEN PROMPT ANSWER")
        except ValueError:
            import sys
            record.exc_info = sys.exc_info()
        result = SafeFormatter().format(record)
        self.assertEqual(json.loads(result)["event"], "runtime_log")
        for private in ["PROMPT", "ANSWER", "SECRET_TOKEN", "ValueError"]:
            self.assertNotIn(private, result)

    def test_middleware_logs_metadata_for_stream_and_error_without_reading_body(self):
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(SafeFormatter())
        logger = logging.getLogger("privacy.audit")
        with patch.object(logger, "handlers", [handler]), patch.object(logger, "level", logging.INFO):
            async def receive():
                raise AssertionError("Middleware must not read request body")

            async def send(message):
                pass

            async def application(scope, receive, send):
                await send({"type": "http.response.start", "status": 200})
                await send({"type": "http.response.body", "body": b"PRIVATE_ANSWER", "more_body": False})

            scope = {"type": "http", "path": "/v1/completions", "method": "POST",
                     "query_string": b"PRIVATE_PROMPT", "headers": [(b"authorization", b"Bearer SECRET")]}
            asyncio.run(RequestAuditMiddleware(application, api_key="SECRET")(scope, receive, send))

            async def broken(scope, receive, send):
                raise ValueError("PRIVATE_PROMPT")

            with self.assertRaises(ValueError):
                asyncio.run(RequestAuditMiddleware(broken, api_key="SECRET")(scope, receive, send))
        items = [json.loads(line) for line in stream.getvalue().splitlines()]
        self.assertEqual([i["status"] for i in items], [200, 500])
        self.assertTrue(all(i["duration_ms"] >= 0 for i in items))
        self.assertNotIn("PRIVATE", stream.getvalue())
        self.assertNotIn("SECRET", stream.getvalue())


class CacheTests(unittest.TestCase):
    revision = "a" * 40

    def fake_download(self, **kwargs):
        dest = Path(kwargs["local_dir"])
        dest.mkdir()
        for name in ["config.json", "tokenizer_config.json", "model.safetensors"]:
            (dest / name).write_bytes(b"fake fixture, not model weights")

    def test_cold_download_and_warm_cache_skip_hf_and_detect_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake = types.SimpleNamespace(snapshot_download=self.fake_download)
            with patch.dict("sys.modules", {"huggingface_hub": fake}), contextlib.redirect_stdout(io.StringIO()):
                target = downloader.download(root, "test/model", self.revision)
            # None makes any attempted HF import fail, including offline client creation.
            with patch.dict("sys.modules", {"huggingface_hub": None}), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(downloader.download(root, "test/model", self.revision), target)
            (target / "model.safetensors").write_bytes(b"corrupt")
            self.assertFalse(downloader.valid_snapshot(target, "test/model", self.revision))

    def test_interrupted_download_is_not_accepted_as_ready(self):
        def fail(**kwargs):
            Path(kwargs["local_dir"]).mkdir()
            raise OSError("Download interrupted")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake = types.SimpleNamespace(snapshot_download=fail)
            with patch.dict("sys.modules", {"huggingface_hub": fake}):
                with self.assertRaises(OSError):
                    downloader.download(root, "test/model", self.revision)
            self.assertFalse((root / self.revision / ".ready.json").exists())


if __name__ == "__main__":
    unittest.main()
