from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json_metadata(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def public_model_info(
    *,
    name: str,
    version: str,
    algorithm: str | None,
    loaded: bool,
    feature_count: int | None,
    used_in_production: bool,
    notes: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": name,
        "version": version,
        "algorithm": algorithm,
        "loaded": loaded,
        "featureCount": feature_count,
        "usedInProduction": used_in_production,
    }
    if notes:
        payload["notes"] = notes
    return payload
