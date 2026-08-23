"""Minimal standard-library HTTP client for the PhysiCar simulator."""

from __future__ import annotations

import json
import math
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class SimClientError(RuntimeError):
    pass


class SimClient:
    def __init__(self, base_url: str, timeout_s: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def _request(self, path: str, method: str = "GET", body: dict[str, Any] | None = None) -> dict[str, Any]:
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = Request(
            self.base_url + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        started = time.monotonic()
        try:
            with urlopen(request, timeout=self.timeout_s) as response:
                payload = json.load(response)
        except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise SimClientError(f"{method} {path} failed: {exc}") from exc
        elapsed = time.monotonic() - started
        if elapsed > self.timeout_s:
            raise SimClientError(f"{method} {path} exceeded timeout ({elapsed:.3f}s)")
        if not isinstance(payload, dict):
            raise SimClientError(f"{method} {path} returned non-object JSON")
        return payload

    def openapi(self) -> dict[str, Any]:
        return self._request("/openapi.json")

    def status(self) -> dict[str, Any]:
        return self._request("/sim/api/status")

    def route(self) -> dict[str, Any]:
        return self._request("/sim/api/route")

    def pose(self) -> dict[str, Any]:
        payload = self._request("/sim/api/pose")
        for key in ("x", "y", "yaw"):
            if key not in payload or not math.isfinite(float(payload[key])):
                raise SimClientError(f"pose has invalid {key!r}")
        return payload

    def clock(self) -> dict[str, Any]:
        payload = self._request("/sim/api/clock")
        if "sim_time" not in payload or not math.isfinite(float(payload["sim_time"])):
            raise SimClientError("simulator clock has invalid 'sim_time'")
        return payload

    def bounds(self) -> dict[str, Any]:
        return self._request("/sim/api/bounds")

    def objects(self) -> dict[str, Any]:
        return self._request("/sim/api/objects")

    def reset(self) -> dict[str, Any]:
        return self._request("/sim/api/reset", method="POST")

    def command_steering(self, value: float) -> dict[str, Any]:
        return self._control("/steering", value)

    def command_speed(self, value: float) -> dict[str, Any]:
        return self._control("/speed", value)

    def _control(self, path: str, value: float) -> dict[str, Any]:
        if not math.isfinite(value):
            raise SimClientError(f"refusing non-finite command for {path}")
        response = self._request(path, method="POST", body={"value": float(value)})
        if response.get("success") is not True:
            raise SimClientError(f"control rejected by {path}: {response}")
        return response

    def safe_stop(self) -> list[str]:
        """Best-effort independent zero commands; return error messages."""
        errors: list[str] = []
        for name, command in (("speed", self.command_speed), ("steering", self.command_steering)):
            try:
                command(0.0)
            except Exception as exc:  # both commands must be attempted
                errors.append(f"{name} stop failed: {exc}")
        return errors


def verify_control_schema(schema: dict[str, Any]) -> None:
    """Reject schemas that do not expose the verified JSON control API."""
    paths = schema.get("paths")
    components = schema.get("components", {}).get("schemas", {})
    if not isinstance(paths, dict):
        raise SimClientError("OpenAPI has no paths object")
    for path, expected_schema in (("/speed", "SpeedRequest"), ("/steering", "SteeringRequest")):
        operation = paths.get(path, {}).get("post")
        if not isinstance(operation, dict):
            raise SimClientError(f"OpenAPI does not expose POST {path}")
        request_schema = operation.get("requestBody", {}).get("content", {}).get("application/json", {}).get("schema", {})
        reference = request_schema.get("$ref", "")
        if reference.rsplit("/", 1)[-1] != expected_schema:
            raise SimClientError(f"POST {path} has unexpected request schema: {request_schema}")
        model = components.get(expected_schema, {})
        value = model.get("properties", {}).get("value", {})
        if "value" not in model.get("required", []) or value.get("type") != "number":
            raise SimClientError(f"{expected_schema} does not require numeric 'value'")
