"""Build the reviewed SQLite snapshot that backs the audit report tables."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ARTIFACT = ROOT / "artifact.json"
DATABASE = ROOT / "audit_tables.sqlite"


SCHEMAS = {
    "problematic_phrases": [
        ("order_no", "INTEGER", "order"),
        ("fragment", "TEXT", "fragment"),
        ("problem_type", "TEXT", "problem_type"),
        ("explanation", "TEXT", "explanation"),
        ("rewrite", "TEXT", "rewrite"),
    ],
    "terminology_audit": [
        ("order_no", "INTEGER", "order"),
        ("term", "TEXT", "term"),
        ("standard", "TEXT", "standard"),
        ("problem", "TEXT", "problem"),
        ("recommended", "TEXT", "recommended"),
    ],
    "result_validation": [
        ("order_no", "INTEGER", "order"),
        ("result", "TEXT", "result"),
        ("theory", "TEXT", "theory"),
        ("numerics", "TEXT", "numerics"),
        ("equilibrium", "TEXT", "equilibrium"),
        ("verdict", "TEXT", "verdict"),
    ],
}


def main() -> None:
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    datasets = artifact["snapshot"]["datasets"]
    with sqlite3.connect(DATABASE) as connection:
        for table, columns in SCHEMAS.items():
            connection.execute(f'DROP TABLE IF EXISTS "{table}"')
            column_sql = ", ".join(
                f'"{name}" {column_type}' for name, column_type, _ in columns
            )
            connection.execute(f'CREATE TABLE "{table}" ({column_sql})')
            names = ", ".join(f'"{name}"' for name, _, _ in columns)
            placeholders = ", ".join("?" for _ in columns)
            rows = [
                tuple(row[source_key] for _, _, source_key in columns)
                for row in datasets[table]
            ]
            connection.executemany(
                f'INSERT INTO "{table}" ({names}) VALUES ({placeholders})', rows
            )
        connection.commit()


if __name__ == "__main__":
    main()
