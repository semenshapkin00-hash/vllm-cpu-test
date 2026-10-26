"""Allowlisted JSON logging and request metadata without request/response bodies."""
import json
import hmac
import logging
import os
import time
import uuid
from datetime import datetime, timezone


class SafeFormatter(logging.Formatter):
    def format(self, record):
        # Never call getMessage()/formatException(): either may contain user data.
        item = {
            "time": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "event": "runtime_log",
        }
        if record.name == "privacy.audit" and getattr(record, "event", None) == "request_end":
            item.update(
                event="request_end",
                request_id=record.request_id,
                route=record.route,
                method=record.method,
                status=record.status,
                duration_ms=record.duration_ms,
            )
        return json.dumps(item, ensure_ascii=True, separators=(",", ":"))


class RequestAuditMiddleware:
    def __init__(self, app, api_key=None):
        self.app = app
        self.api_key = (api_key if api_key is not None else os.environ["VLLM_API_KEY"]).encode()
        self.logger = logging.getLogger("privacy.audit")

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        # Query strings, headers, bodies and client-provided IDs are not logged.
        path = scope.get("path", "")
        if path in {"/health", "/metrics", "/ping"}:
            return await self.app(scope, receive, send)
        route = path if path in {
            "/v1/completions", "/v1/chat/completions", "/v1/models", "/tokenize"
        } else "other"
        method = scope.get("method", "")
        method = method if method in {"GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"} else "other"
        request_id = uuid.uuid4().hex
        started = time.perf_counter()
        status = 500

        async def audited_send(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = int(message["status"])
            await send(message)

        try:
            authorization = dict(scope.get("headers", [])).get(b"authorization", b"")
            if not hmac.compare_digest(authorization, b"Bearer " + self.api_key):
                await audited_send({
                    "type": "http.response.start", "status": 401,
                    "headers": [(b"content-type", b"application/json"),
                                (b"www-authenticate", b"Bearer")],
                })
                await audited_send({"type": "http.response.body", "body": b'{"error":"Unauthorized"}'})
                return
            await self.app(scope, receive, audited_send)
        finally:
            self.logger.info("", extra={
                "event": "request_end", "request_id": request_id, "route": route,
                "method": method, "status": status,
                "duration_ms": round((time.perf_counter() - started) * 1000, 3),
            })
