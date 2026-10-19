"""OpenCodifier backend: the deterministic decision runtime over /v1/decide.

OpenCodifier is a local-first, deterministic-first decision runtime — its
default stack runs exact rules, a metadata filter, and lexical (BM25)
scoring with zero learned weights, escalating to a decision model only when
the cheap layers cannot decide. It accepts this benchmark's System One
question schema natively: start `opencodifier serve` and POST to /v1/decide
with the `x-opencodifier-format: jev` header — the Jev codec decodes the
same wire shape bench.cases.question_payload emits.

Two wire adaptations happen here, both forced by the canonical IR:
- choice option keys become candidate ids, and the IR only accepts ids of
  ASCII letters/digits/`_-.` — so options are renamed opt_0..opt_n and the
  answer is translated back to the original key.
- score levels are keyed "0".."n-1" so the sorted-key decode of a JSON
  object keeps the level order the benchmark's list defines (suites cap at
  10 levels, so string sort == numeric sort), and the response's index
  distribution maps straight back onto the benchmark's levels.
"""

import json
import os
import time
import urllib.error
import urllib.request
from typing import Optional

from bench.cases import Choice, Noul, Score
from .base import Backend, Prediction

DEFAULT_BASE_URL = "http://127.0.0.1:8092"
FORMAT_HEADER = "x-opencodifier-format"


class OpencodifierBackend(Backend):
  name = "opencodifier"
  service_label = "OpenCodifier /v1/decide"
  description = "OpenCodifier deterministic engine (identity reported at warmup)"

  def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None):
    self.base_url = (
      base_url or os.environ.get("OPENCODIFIER_URL") or DEFAULT_BASE_URL
    ).rstrip("/")
    self.model = model or "opencodifier-engine"

  def warmup(self) -> float:
    t0 = time.perf_counter()
    with urllib.request.urlopen(f"{self.base_url}/v1/healthz", timeout=10) as response:
      health = json.loads(response.read().decode())
    identity = health.get("identity", {})
    parts = ["OpenCodifier deterministic engine"]
    if identity.get("model_id"):
      parts.append(f"model_id={identity['model_id']}")
    if identity.get("engine_semver"):
      parts.append(f"v{identity['engine_semver']}")
    self.description = " ".join(parts)
    return time.perf_counter() - t0

  def _invoke(self, state: str, payload: dict) -> object:
    body = json.dumps(
      {"state": state, "model": self.model, "questions": {"q": payload}}
    ).encode()
    headers = {"Content-Type": "application/json", FORMAT_HEADER: "jev"}
    request = urllib.request.Request(
      f"{self.base_url}/v1/decide", data=body, headers=headers, method="POST"
    )
    last_error: Optional[Exception] = None
    for attempt in range(3):
      try:
        with urllib.request.urlopen(request, timeout=120) as response:
          data = json.loads(response.read().decode())
        # An absent answer is the runtime abstaining, which is a result
        # (reported as no prediction), not a transport error to retry.
        return data.get("answers", {}).get("q")
      except urllib.error.HTTPError as exc:
        # A 4xx/5xx is deterministic (decode refusal or engine fault);
        # the body names the code, so surface it instead of retrying.
        raise RuntimeError(
          f"{self.service_label} refused the request ({exc.code}): "
          f"{exc.read().decode(errors='replace')[:500]}"
        ) from exc
      except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as exc:
        last_error = exc
    raise RuntimeError(f"{self.service_label} request failed after 3 attempts: {last_error}")

  def predict_choice(self, state: str, question: Choice) -> Prediction:
    # The IR's candidate ids are stricter than the benchmark's option keys,
    # so options ride as opt_0..opt_n with the original text in the
    # description (the primary signal the engine scores against).
    ids = {}
    criteria = {}
    for index, (key, text) in enumerate(question.criteria.items()):
      renamed = f"opt_{index}"
      ids[renamed] = key
      criteria[renamed] = text
    answer = self._invoke(state, {"type": "choice", "instructions": question.instructions, "criteria": criteria})
    if answer is None:
      return Prediction(label=None)
    if answer not in ids:
      raise ValueError(f"OpenCodifier returned an unoffered option: {answer}")
    return Prediction(label=ids[answer])

  def predict_noul(self, state: str, question: Noul) -> Prediction:
    payload = {"type": "noul", "instructions": question.instructions}
    if question.criteria:
      payload["criteria"] = dict(question.criteria)
    probability = self._invoke(state, payload)
    if probability is None:
      return Prediction(label=None)
    probability = float(probability)
    return Prediction(
      label="yes" if probability >= 0.5 else "no",
      probability=probability,
      confidence=abs(probability - 0.5) * 2,
      raw=probability,
    )

  def predict_score(self, state: str, question: Score) -> Prediction:
    # Index keys, not the level labels: the sorted-key decode must not
    # reorder the levels the benchmark's list defines.
    criteria = {str(index): text for index, text in enumerate(question.criteria)}
    answer = self._invoke(state, {"type": "score", "instructions": question.instructions, "criteria": criteria})
    if answer is None:
      return Prediction(label=None)
    probabilities = {str(k): float(v) for k, v in answer.items()}
    if not probabilities:
      raise ValueError(f"OpenCodifier returned an empty score distribution: {answer}")
    label = max(probabilities, key=probabilities.get)
    return Prediction(
      label=str(label),
      probabilities=probabilities,
      confidence=max(probabilities.values()),
    )
