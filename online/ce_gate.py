"""Runtime for a digest-pinned BM25-feature gate on DuReader RRF to CE@40."""

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from contracts.ce import GATE_FEATURE_NAMES


FEATURE_NAMES = GATE_FEATURE_NAMES
_ARTIFACT_FIELDS = {
    "schema_version", "feature_names", "mean", "scale", "coef", "intercept", "binding",
}
_RETRIEVAL_FIELDS = {"index_identity", "vector_identity", "search_config"}


def features(reader, terms, ranked) -> list[float]:
    """Extract the four serving features from the existing BM25 candidate list."""
    unique_terms = set(terms)
    coverage = sum(reader.df(term) > 0 for term in unique_terms) / max(len(unique_terms), 1)
    gap = 0.0
    if len(ranked) >= 2:
        first, second = float(ranked[0][1]), float(ranked[1][1])
        gap = (first - second) / max(abs(first), 1e-12)
    return [float(len(unique_terms)), coverage, gap, float(len(ranked))]


def _finite_number(value, field: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"CE gate {field} must contain finite numbers")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError(f"CE gate {field} must contain finite numbers") from error
    if not math.isfinite(number):
        raise ValueError(f"CE gate {field} must contain finite numbers")
    return number


def _feature_vector(value, field: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != len(FEATURE_NAMES):
        raise ValueError(f"CE gate {field} must contain {len(FEATURE_NAMES)} values")
    return tuple(_finite_number(entry, field) for entry in value)


def _unique_json_object(pairs):
    value = {}
    for key, entry in pairs:
        if key in value:
            raise ValueError(f"duplicate CE gate JSON key: {key}")
        value[key] = entry
    return value


def _identity_json(identity) -> str:
    return json.dumps(identity, sort_keys=True, allow_nan=False)


@dataclass(frozen=True)
class Gate:
    sha256: str
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    coef: tuple[float, ...]
    intercept: float
    _retrieval_binding: str
    _model_binding: str

    def use_ce(self, features) -> bool:
        """Match the binary classifier's positive-margin decision."""
        values = _feature_vector(features, "features")
        logit = self.intercept + sum(
            weight * ((value - mean) / scale)
            for value, mean, scale, weight in zip(
                values, self.mean, self.scale, self.coef, strict=True
            )
        )
        if not math.isfinite(logit):
            raise ValueError("CE gate produced a non-finite decision score")
        return logit > 0.0

    def check_binding(self, retrieval_identity, model_identity) -> None:
        """Require the serving identities recorded by the training export."""
        if _identity_json(retrieval_identity) != self._retrieval_binding:
            raise ValueError("CE gate retrieval binding does not match the serving identity")
        if _identity_json(model_identity) != self._model_binding:
            raise ValueError("CE gate model binding does not match the serving identity")


def load_gate(path, expected_sha256) -> Gate:
    """Verify the local artifact digest, schema, and coefficient dimensions."""
    if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256):
        raise ValueError("CE gate SHA256 must be a 64-digit hexadecimal digest")
    digest = expected_sha256.lower()
    try:
        raw = Path(path).read_bytes()
    except (OSError, TypeError, ValueError) as error:
        raise ValueError("unable to read CE gate artifact") from error
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("CE gate SHA256 does not match the artifact")
    try:
        payload = json.loads(raw, object_pairs_hook=_unique_json_object)
    except ValueError as error:
        raise ValueError("CE gate artifact must contain valid JSON") from error
    if not isinstance(payload, dict) or payload.keys() != _ARTIFACT_FIELDS:
        raise ValueError("CE gate artifact fields do not match schema version 1")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("unsupported CE gate schema version")
    if payload["feature_names"] != list(FEATURE_NAMES):
        raise ValueError("unsupported CE gate feature names or order")

    binding = payload["binding"]
    if not isinstance(binding, dict) or binding.keys() != {"retrieval", "model"}:
        raise ValueError("CE gate binding must specify retrieval and model identities")
    retrieval, model = binding["retrieval"], binding["model"]
    if (not isinstance(retrieval, dict) or retrieval.keys() != _RETRIEVAL_FIELDS
            or any(not isinstance(entry, dict) or not entry for entry in retrieval.values())):
        raise ValueError("CE gate retrieval binding must specify index, vector and search identities")
    if not isinstance(model, dict) or not model:
        raise ValueError("CE gate model binding must be a nonempty object")

    scale = _feature_vector(payload["scale"], "scale")
    if any(value <= 0 for value in scale):
        raise ValueError("CE gate scale must contain positive numbers")
    return Gate(
        sha256=digest,
        mean=_feature_vector(payload["mean"], "mean"),
        scale=scale,
        coef=_feature_vector(payload["coef"], "coef"),
        intercept=_finite_number(payload["intercept"], "intercept"),
        _retrieval_binding=_identity_json(retrieval),
        _model_binding=_identity_json(model),
    )
