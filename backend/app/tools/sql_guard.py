from __future__ import annotations

import re

import sqlglot
from sqlglot import exp

ALLOWED_TABLES = {"clinical_trials"}
FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|alter|truncate|grant|revoke|create|copy|call)\b", re.I)


class UnsafeSQLError(ValueError):
    pass


def validate_select(sql: str, allowed_tables: set[str] | None = None) -> str:
    """Allow only a single SELECT against allowlisted tables."""
    allowed = allowed_tables or ALLOWED_TABLES
    stripped = sql.strip().rstrip(";")
    if ";" in stripped:
        raise UnsafeSQLError("Multiple statements are not allowed")
    if FORBIDDEN.search(stripped):
        raise UnsafeSQLError("Only SELECT queries are allowed")
    try:
        parsed = sqlglot.parse_one(stripped, dialect="postgres")
    except sqlglot.errors.ParseError as exc:
        raise UnsafeSQLError(f"Could not parse SQL: {exc}") from exc
    if not isinstance(parsed, exp.Select):
        raise UnsafeSQLError("Query must be a SELECT")
    tables = {t.name.lower() for t in parsed.find_all(exp.Table) if t.name}
    unknown = tables - {name.lower() for name in allowed}
    if unknown:
        raise UnsafeSQLError(f"Tables not allowed: {sorted(unknown)}")
    if not tables:
        raise UnsafeSQLError("Query does not reference an allowed table")
    # Force a limit
    if parsed.args.get("limit") is None:
        parsed = parsed.limit(100)
    return parsed.sql(dialect="postgres")
