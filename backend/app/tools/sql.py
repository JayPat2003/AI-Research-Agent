from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

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
    try:
        safe = validate_select(sql)
    except UnsafeSQLError as exc:
        return {"ok": False, "error": str(exc), "sql": sql, "rows": []}
    result = await session.execute(text(safe))
    rows = [dict(r) for r in result.mappings().all()]
    return {"ok": True, "error": None, "sql": safe, "rows": rows, "n": len(rows)}
