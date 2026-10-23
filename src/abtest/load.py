import sys
import time
from pathlib import Path

import duckdb
import pandas as pd

DATA_FILE = "criteo-research-uplift-v2.1.csv.gz"
DATA_URL = "https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/criteo-research-uplift-v2.1.csv.gz"
DATA_PATH = Path(__file__).resolve().parents[2] / "data" / DATA_FILE

RAW_TABLE = "raw"


def data_path() -> Path:
    return DATA_PATH


def require_data(path: Path | str | None = None) -> Path:
    """Return the dataset path, failing with download instructions if absent."""
    p = Path(path) if path is not None else DATA_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"Dataset not found at {p}\n\n"
            f"Download it (~297 MB, kept gzipped):\n"
            f"  mkdir -p data && curl -L -o data/{DATA_FILE} \\\n"
            f"    {DATA_URL}"
        )
    return p


def connect(path: Path | None = None, materialize: bool = True) -> duckdb.DuckDBPyConnection:
    """Open an in-memory DuckDB connection with the raw dataset as ``raw``.

    The dataset is read from the ``.gz`` once and materialized as an in-memory
    table. Pass ``materialize=False`` for a lightweight view (every query reads 
    the full file).
    """
    p = require_data(path)
    conn = duckdb.connect()
    if not sys.stdout.isatty():
        # Don't render duckdb progress widget
        conn.execute("SET enable_progress_bar=false")
    rel_kind = "TABLE" if materialize else "VIEW"
    t0 = time.time()
    print(f"loading {p.name} (≈1 minute) ...", flush=True)
    conn.execute(
        f"""
        CREATE OR REPLACE {rel_kind} {RAW_TABLE} AS
        SELECT * FROM read_csv_auto('{p.as_posix()}', sample_size=-1)
        """
    )
    if materialize:
        n = row_count(conn)
        print(f"loaded {n:,} rows into memory in {time.time() - t0:.1f}s "
              f"(from {p.name})")
    return conn


def head(conn: duckdb.DuckDBPyConnection, n: int = 5) -> pd.DataFrame:
    return conn.execute(f"SELECT * FROM {RAW_TABLE} LIMIT {n}").df()


def row_count(conn: duckdb.DuckDBPyConnection) -> int:
    return conn.execute(f"SELECT count(*) FROM {RAW_TABLE}").fetchone()[0]
