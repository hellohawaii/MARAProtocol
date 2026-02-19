import os
import json
from typing import Any, Dict, Optional
from urllib import request, error


class RemoteEnvWrapper:
    def __init__(self, base_url: Optional[str] = None, timeout_seconds: int = 30):
        self.base_url = (base_url or os.getenv("ENV_API_BASE_URL") or "http://host.docker.internal:8000").rstrip("/")
        self.timeout_seconds = timeout_seconds

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as resp:
                resp_body = resp.read().decode("utf-8")
        except error.HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} calling {url}: {message}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Failed to call {url}: {exc}") from exc

        try:
            return json.loads(resp_body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON response from {url}: {resp_body}") from exc

    def reset(self, env_name: str, data_dir: Optional[str] = None, seed: int = 0) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"env_name": env_name, "seed": seed}
        if data_dir:
            payload["data_dir"] = data_dir
        return self._post("/reset", payload)

    def step(self, action: str) -> Dict[str, Any]:
        return self._post("/step", {"action": action})

    def save_trajectory(self, filename: Optional[str] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if filename:
            payload["filename"] = filename
        return self._post("/save_trajectory", payload)
