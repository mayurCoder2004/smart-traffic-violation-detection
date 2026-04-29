"""
Copy the local Docker PostgreSQL database into the DATABASE_URL target.

Source:
    postgresql://traffic_user:traffic_pass@localhost:5432/traffic_violations

Target:
    DATABASE_URL from .env, intended to be the Neon database.

The script creates the current app schema on the target, then upserts rows
from the Docker database while preserving primary keys and relationships.
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from sqlalchemy import MetaData, Table, create_engine, text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app import app
from backend.extensions import db


ROOT = Path(__file__).parent
SOURCE_URL = "postgresql://traffic_user:traffic_pass@localhost:5432/traffic_violations"
TABLES = [
    "users",
    "violations",
    "scanner_challans",
    "scanner_challan_items",
    "payments",
]


def ensure_user_columns(session) -> None:
    statements = [
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS vehicle VARCHAR(20) DEFAULT 'Unknown'",
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS rc BOOLEAN NOT NULL DEFAULT TRUE",
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS insurance BOOLEAN NOT NULL DEFAULT TRUE",
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS puc BOOLEAN NOT NULL DEFAULT TRUE",
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS dl BOOLEAN NOT NULL DEFAULT TRUE",
    ]
    for stmt in statements:
        session.execute(text(stmt))
    session.commit()


def count_rows(conn, table_name: str) -> Optional[int]:
    try:
        return conn.execute(text(f"SELECT count(*) FROM {table_name}")).scalar()
    except Exception:
        return None


def upsert_table(src_conn, dst_conn, src_table: Table, dst_table: Table) -> int:
    rows = [dict(row._mapping) for row in src_conn.execute(src_table.select()).fetchall()]
    if not rows:
        return 0

    pk_cols = [col.name for col in dst_table.primary_key.columns]
    if not pk_cols:
        raise RuntimeError(f"Table {dst_table.name} has no primary key; cannot upsert safely.")

    inserted = 0
    chunk_size = 500
    for start in range(0, len(rows), chunk_size):
        chunk = rows[start : start + chunk_size]
        stmt = pg_insert(dst_table).values(chunk)
        update_cols = {
            col.name: getattr(stmt.excluded, col.name)
            for col in dst_table.columns
            if col.name not in pk_cols
        }
        if update_cols:
            stmt = stmt.on_conflict_do_update(index_elements=pk_cols, set_=update_cols)
        else:
            stmt = stmt.on_conflict_do_nothing(index_elements=pk_cols)
        dst_conn.execute(stmt)
        inserted += len(chunk)
    return inserted


def main() -> None:
    load_dotenv(ROOT / ".env")
    target_url = os.environ.get("DATABASE_URL")
    if not target_url:
        raise RuntimeError("DATABASE_URL is not set in .env")
    if "localhost" in target_url or "traffic_violations" in target_url:
        raise RuntimeError("DATABASE_URL still looks like the local Docker database, not Neon.")

    source_engine = create_engine(SOURCE_URL)
    target_engine = create_engine(target_url)

    with app.app_context():
        db.create_all()
        ensure_user_columns(db.session)

    src_meta = MetaData()
    dst_meta = MetaData()
    src_meta.reflect(bind=source_engine, only=TABLES)
    dst_meta.reflect(bind=target_engine, only=TABLES)

    with source_engine.connect() as src_conn, target_engine.begin() as dst_conn:
        print("Before migration:")
        for table_name in TABLES:
            print(f"  {table_name}: source={count_rows(src_conn, table_name)} target={count_rows(dst_conn, table_name)}")

        print("\nCopying rows:")
        for table_name in TABLES:
            copied = upsert_table(src_conn, dst_conn, src_meta.tables[table_name], dst_meta.tables[table_name])
            print(f"  {table_name}: {copied} rows upserted")

        print("\nAfter migration:")
        for table_name in TABLES:
            print(f"  {table_name}: target={count_rows(dst_conn, table_name)}")


if __name__ == "__main__":
    main()
