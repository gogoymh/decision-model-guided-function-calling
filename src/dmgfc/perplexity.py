"""Opt-in native Perplexity Decider adapter; the historical Jev default is unchanged.

The HTTP service must expose the pinned checkpoint's deployment provenance at
``/provenance``. Generic causal chat serving is not this decision checkpoint.
Only the frozen Jev questions, observable inputs and schema gate are reused.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import jev_fc.jev as frozen_jev

MODEL_ID = "perplexity-ai/pplx-decider-v1.1-27b"
MODEL_REVISION = "14c9d90137962d300f3de95633f7b6535e80c5ce"
CALIBRATION_TEMPERATURE = 1.0087417621345625
MAX_INPUT_TOKENS = 8192
INHERITED_SERVER_MODEL = "autojev-qwen3.8-27b"


def fingerprint(value: object) -> str:
    """Hash finite JSON with deterministic key ordering, without changing source data."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                         allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


class DeciderError(frozen_jev.JevAPIError):
    """Transport/provenance/protocol failures never count as semantic rejections."""

    def __init__(self, message: str, *, code: str = "decider_error", **details):
        super().__init__(message, **details)
        self.code = self.provider_code = code


class PerplexityDecider(frozen_jev.JevVerifier):
    """Use the pinned native checkpoint over a verified SystemOne HTTP endpoint.

    No TypeSafe key, environment route or hosted model is selected implicitly.
    Pass this object explicitly as ``DMGFC(..., jev=decider)``. The service's
    inherited ``jev-latest`` alias is a wire-format identifier, not model identity.
    ``api_calls`` counts POST attempts; provenance GETs are reported separately.
    """

    def __init__(self, *, base_url: str = "http://127.0.0.1:18081", api_key: str | None = None,
                 threshold: float = 0.5, function_threshold: float | None = None,
                 arguments_threshold: float | None = None, timeout: float = 180.0,
                 max_retries: int = 0, client: httpx.Client | None = None,
                 sleep=time.sleep):
        parsed = urlsplit(base_url)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username is not None or parsed.password is not None
                or parsed.query or parsed.fragment or parsed.path not in {"", "/"}):
            raise ValueError("Decider base_url must be an HTTP(S) origin without credentials or path.")
        if parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Use an SSH tunnel for a local HTTP service, or an HTTPS origin.")
        values = [threshold if value is None else value for value in
                  (function_threshold, arguments_threshold)]
        for value in values:
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError("Decider thresholds must be finite numbers between 0 and 1.")
        if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
            raise ValueError("max_retries must be a nonnegative integer.")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be a positive finite number.")
        if api_key is not None and (not isinstance(api_key, str) or not api_key.strip()
                                    or any(char in api_key for char in "\r\n")):
            raise ValueError("api_key must be a nonempty single-line string.")
        self.base_url = base_url.rstrip("/")
        self.gateway = "native_systemone"
        self.model = "jev-latest"
        self.function_threshold, self.arguments_threshold = map(float, values)
        self.timeout, self.max_retries = float(timeout), max_retries
        self._headers = {"Content-Type": "application/json"}
        if api_key is not None:
            self._headers["Authorization"] = "Bearer " + api_key
        self._client = client or httpx.Client(timeout=self.timeout, trust_env=False)
        self._owns_client, self._sleep = client is None, sleep
        self._request = threading.local()

    @staticmethod
    def _validate_provenance(body: object) -> dict:
        if not isinstance(body, dict):
            raise DeciderError("Decider deployment provenance must be an object.", code="provenance_invalid")
        expected = {"model_id": MODEL_ID, "revision": MODEL_REVISION,
                    "native_attention": "noncausal_full_attention", "max_input_tokens": MAX_INPUT_TOKENS}
        if any(body.get(key) != value for key, value in expected.items()):
            raise DeciderError("Decider service does not match the pinned native checkpoint.",
                               code="provenance_mismatch")
        temperature = body.get("temperature")
        if (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
                or not math.isfinite(temperature)
                or not math.isclose(temperature, CALIBRATION_TEMPERATURE, rel_tol=0, abs_tol=1e-12)):
            raise DeciderError("Decider calibration does not match the checkpoint.", code="provenance_mismatch")
        result = {**expected, "temperature": float(temperature)}
        # The optional batch marker describes actual model invocations. The
        # checkpoint creates one input branch per question, batched up to eight.
        batch_size = body.get("inference_batch_size", body.get("native_inference_batch_size"))
        if batch_size == 8 and body.get("one_branch_per_question") is True:
            result.update(inference_batch_size=8, one_branch_per_question=True)
        return result

    def provenance(self) -> dict:
        try:
            response = self._client.get(self.base_url + "/provenance", headers=self._headers,
                                        timeout=self.timeout)
        except httpx.TransportError:
            raise DeciderError("Decider provenance endpoint could not be reached.", code="connection_failed") from None
        if response.status_code != 200:
            raise DeciderError("Decider provenance endpoint is unavailable.", code="provenance_unavailable",
                               status_code=response.status_code)
        try:
            return self._validate_provenance(response.json())
        except ValueError:
            raise DeciderError("Decider provenance response was not valid JSON.", code="provenance_invalid") from None

    def configuration(self) -> dict:
        """A credential-free, immutable experiment identity; sends no requests."""
        files = [Path(__file__), Path(frozen_jev.__file__)]
        source = {file.name: hashlib.sha256(file.read_bytes()).hexdigest() for file in files}
        return {"adapter": "perplexity_decider_native_systemone", "model_id": MODEL_ID,
                "revision": MODEL_REVISION, "native_attention": "noncausal_full_attention",
                "calibration_temperature": CALIBRATION_TEMPERATURE, "max_input_tokens": MAX_INPUT_TOKENS,
                "base_url": self.base_url, "wire_model": self.model,
                "function_threshold": self.function_threshold, "arguments_threshold": self.arguments_threshold,
                "timeout_s": self.timeout, "max_retries": self.max_retries,
                "protocol_source_sha256": source, "questions_sha256": fingerprint({
                    "function_correct": frozen_jev.FUNCTION_QUESTION,
                    "arguments_correct": frozen_jev.ARGUMENT_QUESTION})}

    def _post(self, payload: dict) -> tuple[dict, int, float]:
        started = time.perf_counter()
        self._request.provenance = self.provenance()
        self._request.control_http_time_s = time.perf_counter() - started
        self._request.control_calls = 1
        self._request.question_count = len(payload["questions"])
        waited = 0.0
        for attempt in range(self.max_retries + 1):
            started = time.perf_counter()
            try:
                response = self._client.post(self.base_url + "/v1/systemone", headers=self._headers,
                                             json=payload, timeout=self.timeout)
            except httpx.TransportError:
                self._request.decision_http_time_s += time.perf_counter() - started
                if attempt < self.max_retries:
                    waited += self._wait(min(2.0 ** attempt, 30.0))
                    continue
                raise DeciderError("Decider connection failed after bounded retries.",
                                   code="connection_failed", attempts=attempt + 1) from None
            self._request.decision_http_time_s += time.perf_counter() - started
            if response.status_code in frozen_jev.RETRY_STATUSES and attempt < self.max_retries:
                waited += self._wait(min(frozen_jev._retry_delay(response, attempt), 30.0))
                continue
            if not 200 <= response.status_code < 300:
                code = "http_error"
                if response.status_code == 422:
                    # Inspect only the known native error markers; never expose
                    # server bodies, request content or echoed credentials.
                    try:
                        detail = response.json().get("detail", "")
                        if "8192-token limit" in str(detail):
                            code = "context_limit_exceeded"
                    except (ValueError, AttributeError):
                        pass
                raise DeciderError(f"Decider HTTP {response.status_code}; no semantic verdict recorded.",
                                   code=code, status_code=response.status_code, attempts=attempt + 1,
                                   request_id=frozen_jev._safe_request_id(response, self._headers))
            try:
                body = response.json()
            except ValueError:
                raise DeciderError("Decider response was not valid JSON.", code="response_invalid") from None
            if not isinstance(body, dict):
                raise DeciderError("Decider response must be an object.", code="response_invalid")
            reported = body.get("model")
            if reported not in {MODEL_ID, INHERITED_SERVER_MODEL}:
                raise DeciderError("Decider response has an unexpected model ID.", code="model_mismatch")
            if "revision" in body and body["revision"] != MODEL_REVISION:
                raise DeciderError("Decider response has an unexpected model revision.", code="model_mismatch")
            self._request.reported_model = reported
            # JevVerifier's immutable semantic parser expects a resolved model
            # and a usage object. Missing usage is unknown, never a fabricated 0.
            body = {**body, "model": MODEL_ID}
            if "usage" not in body:
                body["usage"] = {"input_tokens": None, "output_tokens": None}
            return body, attempt + 1, waited
        raise AssertionError("Unreachable retry branch")

    def verify(self, example: dict, calls: object) -> dict:
        self._request.provenance = None
        self._request.control_calls = self._request.question_count = 0
        self._request.reported_model = None
        self._request.control_http_time_s = self._request.decision_http_time_s = 0.0
        started = time.perf_counter()
        try:
            result = super().verify(example, calls)
        except DeciderError:
            raise
        except frozen_jev.JevAPIError as error:
            raise DeciderError(str(error).replace("Jev", "Decider"), code="response_invalid",
                               status_code=error.status_code, attempts=error.attempts,
                               request_id=error.request_id) from None
        result["feedback"]["source"] = "perplexity_decider_noul_and_deterministic_schema_validation"
        provenance = self._request.provenance
        result.update(model_provenance=provenance, reported_server_model=self._request.reported_model,
                      configuration_fingerprint=fingerprint(self.configuration()),
                      control_api_calls=self._request.control_calls,
                      control_http_time_s=self._request.control_http_time_s,
                      decision_http_time_s=self._request.decision_http_time_s,
                      wall_time_s=time.perf_counter() - started,
                      question_evaluations=self._request.question_count)
        batches = (math.ceil(self._request.question_count / 8)
                   if provenance and provenance.get("inference_batch_size") == 8 else None)
        if result["api_calls"] == 0:
            batches = 0
        result["successful_forward_batches"] = batches
        result["forward_calls"] = batches if result["api_calls"] <= 1 else None
        result["forward_calls_complete"] = result["forward_calls"] is not None
        result["successful_response_usage"] = dict(result["usage"])
        if result["api_calls"] > 1:
            result["usage"] = {"input_tokens": None, "output_tokens": None}
        result["usage_complete"] = all(value is not None for value in result["usage"].values())
        result["estimated_usd"] = None  # local electricity/hardware cost was not measured
        return result

    def __repr__(self):
        return (f"PerplexityDecider(model_id={MODEL_ID!r}, revision={MODEL_REVISION!r}, "
                f"base_url={self.base_url!r})")


__all__ = ["CALIBRATION_TEMPERATURE", "DeciderError", "MAX_INPUT_TOKENS", "MODEL_ID",
           "MODEL_REVISION", "PerplexityDecider", "fingerprint"]
