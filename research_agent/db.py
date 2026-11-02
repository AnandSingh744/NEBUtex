"""PostgreSQL + pgvector: the connection and the schema. The only module that knows the table layout.

Usage (apply any pending migrations, then show what's there):
    python -m research_agent.db
"""

import atexit
import logging
import re
import threading
import time

import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool, PoolTimeout

from research_agent import config

# Migrations: numbered, ordered changes to the database structure. Each is applied exactly once,
# and schema_migrations remembers which ones a database already has. NEVER edit a migration that
# has been applied somewhere; add a new one instead.
MIGRATIONS: list[tuple[int, str, str]] = [
    (1, "documents and chunks", f"""
        CREATE EXTENSION IF NOT EXISTS vector;

        CREATE TABLE IF NOT EXISTS documents (
            id            BIGSERIAL PRIMARY KEY,           -- 1, 2, 3, ... assigned by the database
            name          TEXT NOT NULL UNIQUE,            -- file name, shown in citations
            sha256        TEXT NOT NULL,                   -- content fingerprint: skip unchanged re-uploads
            num_pages     INTEGER NOT NULL,
            settings      JSONB NOT NULL,                  -- chunking + embedding settings used
            processed_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        );

        CREATE TABLE IF NOT EXISTS chunks (
            id            TEXT PRIMARY KEY,                -- e.g. 'paper.pdf:p3:c1' (deterministic)
            document_id   BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            page_number   INTEGER NOT NULL,
            chunk_index   INTEGER NOT NULL,                -- position in the document, to keep order
            text          TEXT NOT NULL,
            embedding     vector({config.EMBEDDING_DIM}) NOT NULL
        );

        -- Fast "all chunks of document X" (used when replacing or deleting a document).
        CREATE INDEX IF NOT EXISTS chunks_document_id_idx ON chunks (document_id);
        -- Fast approximate nearest-neighbour search on the vectors (same HNSW idea as Chroma).
        CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
    """),
    (2, "conversations and messages", """
        CREATE TABLE conversations (
            id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),  -- random, hard to guess; used in URLs
            title       TEXT NOT NULL,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()            -- for "most recent first"
        );

        CREATE TABLE messages (
            id               BIGSERIAL PRIMARY KEY,                   -- increasing, so it also gives the order
            conversation_id  UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            role             TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
            content          TEXT NOT NULL,
            status           TEXT,                                    -- assistant only: answered, not_found, ...
            citations        JSONB NOT NULL DEFAULT '[]',             -- a COPY of the cited passages
            created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
        );

        CREATE INDEX messages_conversation_idx ON messages (conversation_id, id);
    """),
]

# Any fixed number works; it just has to be the same everywhere. See init_db().
_MIGRATION_LOCK_ID = 727274
CONNECT_ATTEMPTS = 3  # connections can fail briefly (e.g. Neon waking up, a network blip): retry

logger = logging.getLogger(__name__)


def _safe_error(e: Exception) -> str:
    """The error's type and message with any connection string or host removed, for logs."""
    msg = re.sub(r"postgres(ql)?://\S+", "<url>", str(e))
    msg = re.sub(r"[\w.-]+\.neon\.tech", "<host>", msg)
    return f"{type(e).__name__}: {msg.strip()[:300]}"


class DatabaseError(Exception):
    """A database problem, with a message that is safe to show the user (no passwords)."""


def connect() -> psycopg.Connection:
    """Open a connection with pgvector support. Use it in a `with` block so it gets closed."""
    if not config.DATABASE_URL:
        raise DatabaseError("DATABASE_URL is not set. Add it to the .env file (or the app's secrets).")
    try:
        # prepare_threshold=None: no server-side prepared statements, which connection poolers
        # (like Neon's "-pooler" address) can mishandle. They'd save us nothing at this scale anyway.
        conn = psycopg.connect(config.DATABASE_URL, connect_timeout=15, prepare_threshold=None)
    except psycopg.OperationalError as e:
        # Don't include the library's message: it can echo parts of the connection string.
        raise DatabaseError("Could not connect to the database. Check DATABASE_URL and the internet connection.") from e
    register_vector(conn)  # lets us send and receive numpy arrays as `vector` values
    return conn


def init_db() -> list[str]:
    """Bring the database up to date by applying any migrations it doesn't have yet.

    Returns the names of the migrations applied now (empty if it was already up to date).
    Brief connection failures are retried.
    """
    if not config.DATABASE_URL:
        raise DatabaseError("DATABASE_URL is not set. Add it to the .env file (or the app's secrets).")
    for attempt in range(1, CONNECT_ATTEMPTS + 1):
        try:
            return _apply_migrations()
        except psycopg.OperationalError as e:
            logger.warning("Database connection failed (attempt %d/%d): %s", attempt, CONNECT_ATTEMPTS, _safe_error(e))
            if attempt == CONNECT_ATTEMPTS:
                raise DatabaseError("Could not connect to the database. Check DATABASE_URL and the internet connection.") from e
            time.sleep(2 ** attempt)  # 2s, then 4s


def _apply_migrations() -> list[str]:
    applied_now = []
    # A plain connection: register_vector() needs the vector extension, which migration 1 creates.
    with psycopg.connect(config.DATABASE_URL, connect_timeout=15, prepare_threshold=None) as conn:
        with conn.transaction():  # all pending migrations succeed together, or none are applied
            # Only one app instance migrates at a time; others wait here, then find nothing to do.
            conn.execute("SELECT pg_advisory_xact_lock(%s)", (_MIGRATION_LOCK_ID,))
            conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version     INTEGER PRIMARY KEY,
                    name        TEXT NOT NULL,
                    applied_at  TIMESTAMPTZ NOT NULL DEFAULT now()
                )
            """)
            done = {row[0] for row in conn.execute("SELECT version FROM schema_migrations")}
            for version, name, sql in MIGRATIONS:
                if version not in done:
                    conn.execute(sql)
                    conn.execute("INSERT INTO schema_migrations (version, name) VALUES (%s, %s)", (version, name))
                    applied_now.append(f"{version}: {name}")
    return applied_now


_pool: ConnectionPool | None = None
_pool_lock = threading.Lock()


def get_pool() -> ConnectionPool:
    """The shared pool of ready-to-use connections, created on first use.

    Opening a connection is slow (several round trips), so connections are reused. Each one is
    health-checked before it's handed out, because Neon closes idle connections when it scales
    to zero; a dead one is silently replaced instead of crashing the next query.

    Use:  with get_pool().connection() as conn: ...   (commits at the end, rolls back on error)
    """
    global _pool
    with _pool_lock:  # two threads arriving at once must not create two pools
        if _pool is None:
            init_db()  # the vector extension must exist before register_vector() runs on each connection
            pool = ConnectionPool(
                config.DATABASE_URL,
                # min_size=0 + max_idle: close connections idle for 2 minutes. An open connection can
                # keep Neon's compute awake, using up the free plan's monthly compute hours.
                min_size=0,
                max_size=4,
                max_idle=120,
                kwargs={"connect_timeout": 15, "prepare_threshold": None},
                configure=register_vector,               # every new connection understands `vector`
                check=ConnectionPool.check_connection,   # test each connection before lending it out
                open=False,
            )
            try:
                pool.open(wait=True, timeout=30)
            except PoolTimeout as e:
                pool.close()
                raise DatabaseError("Could not connect to the database. Check DATABASE_URL and the internet connection.") from e
            _pool = pool
            atexit.register(pool.close)  # stop the pool's background threads cleanly when the program exits
    return _pool


if __name__ == "__main__":
    applied = init_db()
    print("Applied now: " + (", ".join(applied) if applied else "nothing (already up to date)"))
    with connect() as conn:
        version = conn.execute("SELECT version()").fetchone()[0].split(",")[0]
        vector_version = conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()[0]
        print(f"Connected: {version}, pgvector {vector_version}\n")
        print("Migrations:")
        for v, name, at in conn.execute("SELECT version, name, applied_at FROM schema_migrations ORDER BY version"):
            print(f"  {v}. {name}  (applied {at:%Y-%m-%d %H:%M})")
        print("\nTables:")
        tables = [r[0] for r in conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name")]
        for table in tables:
            count = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            print(f"  {table:18} {count} rows")
