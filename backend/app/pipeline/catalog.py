from __future__ import annotations

import json
from pathlib import Path

from app.agents.schemas import CatalogFilter
from app.config import get_settings


def load_catalog() -> list[dict]:
    path = Path(get_settings().catalog_path)
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def filter_catalog(flt: CatalogFilter) -> dict:
    rows = load_catalog()
    matched = []
    for row in rows:
        if flt.drug_class and row.get("drug_class", "").lower() != flt.drug_class.lower():
            continue
        if flt.condition and flt.condition.lower() not in row.get("condition", "").lower():
            continue
        if flt.status and row.get("status", "").lower() != flt.status.lower():
            continue
        if flt.primary_endpoint and flt.primary_endpoint.lower() not in row.get("primary_endpoint", "").lower():
            continue
        matched.append(row)
    return {
        "n": len(matched),
        "filters": flt.model_dump(),
        "rows": matched,
    }
