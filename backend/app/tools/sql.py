"""Safe Text-to-SQL execution against the demo ``clinical_trials`` table.

The LLM proposes a SELECT; :mod:`app.tools.sql_guard` validates it before
execution. ``SCHEMA_HINT`` is injected into the SQL agent prompt so the model
knows the only allowlisted table and column semantics.
"""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.observability.tracing import span
from app.tools.sql_guard import UnsafeSQLError, validate_select

SCHEMA_HINT = """
The only queryable table is clinical_trials with columns:
  id INTEGER
  trial_name TEXT          -- e.g. EMPA-REG OUTCOME, LEADER, SELECT
  condition TEXT           -- type 2 diabetes, heart failure, chronic kidney disease, overweight or obesity
  intervention TEXT        -- generic drug name
  drug_class TEXT          -- SGLT2, GLP-1, GIP/GLP-1, biguanide
  phase TEXT               -- 3
  n_participants INTEGER
  region TEXT              -- global, UK
  status TEXT              -- completed, ongoing
  primary_endpoint TEXT    -- 3-point MACE, kidney composite, etc.
  start_year INTEGER

Return a single PostgreSQL SELECT. Do not use joins onto other tables.
This catalog is a demo subset of well-known outcomes trials, not ClinicalTrials.gov.
"""


async def run_select(session: AsyncSession, sql: str) -> dict:
    """Validate and execute a read-only SELECT; return rows or a structured error."""
    with span("tool.sql") as sp:
        try:
            safe = validate_select(sql)
        except UnsafeSQLError as exc:
            sp.attrs.update({"ok": False, "error": str(exc)})
            return {"ok": False, "error": str(exc), "sql": sql, "rows": []}
        result = await session.execute(text(safe))
        rows = [dict(r) for r in result.mappings().all()]
        sp.attrs.update({"ok": True, "n": len(rows)})
        return {"ok": True, "error": None, "sql": safe, "rows": rows, "n": len(rows)}
