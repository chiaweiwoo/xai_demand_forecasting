"""
Run all pending SQL migrations against the database.

Usage:
    uv run python migrate.py
    uv run python migrate.py --db db/forecasting.db   (default)

Migrations live in migrations/*.sql, applied in filename order.
Applied migrations are tracked in the schema_migrations table.
Safe to re-run — already-applied migrations are skipped.
"""

import argparse
import sqlite3
from pathlib import Path

DB_PATH = 'db/forecasting.db'
MIGRATIONS_DIR = Path('migrations')


def run(db_path: str = DB_PATH) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA synchronous=NORMAL')

    conn.execute('''
        CREATE TABLE IF NOT EXISTS schema_migrations (
            filename TEXT PRIMARY KEY,
            applied_at TEXT DEFAULT (datetime("now"))
        )
    ''')
    conn.commit()

    applied = {r[0] for r in conn.execute('SELECT filename FROM schema_migrations')}
    migrations = sorted(MIGRATIONS_DIR.glob('*.sql'))

    if not migrations:
        print('No migration files found in migrations/')
        conn.close()
        return

    pending = [m for m in migrations if m.name not in applied]
    if not pending:
        print(f'All {len(migrations)} migrations already applied.')
        conn.close()
        return

    for path in pending:
        print(f'  Applying {path.name}...')
        sql = path.read_text()
        conn.executescript(sql)
        conn.execute('INSERT INTO schema_migrations (filename) VALUES (?)', (path.name,))
        conn.commit()
        print(f'  Done.')

    print(f'\n{len(pending)} migration(s) applied to {db_path}')
    conn.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', default=DB_PATH)
    args = parser.parse_args()
    run(args.db)
