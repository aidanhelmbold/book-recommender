-- bookmap DuckDB schema.
--
-- Three layers, deliberately kept separate:
--   books / edges_raw  -- what sources said, preserved verbatim with provenance
--   edges_fused        -- the derived undirected graph the algorithms run on
--   aliases            -- identifier resolution across sources
--
-- Keeping edges_raw immutable means fusion parameters can be retuned and the
-- graph rebuilt without re-ingesting multi-gigabyte dumps.

CREATE TABLE IF NOT EXISTS books (
    work_id        VARCHAR PRIMARY KEY,
    title          VARCHAR NOT NULL,
    authors        VARCHAR[],
    year           INTEGER,
    isbn13         VARCHAR,
    asin           VARCHAR,
    goodreads_id   VARCHAR,
    openlibrary_id VARCHAR,
    series         VARCHAR,
    avg_rating     DOUBLE,
    ratings_count  BIGINT,
    subjects       VARCHAR[],
    title_norm     VARCHAR
);

-- title_norm backs fuzzy seed resolution; the identifier indexes back the
-- cross-source joins performed during ingest.
CREATE INDEX IF NOT EXISTS books_title_norm_idx ON books (title_norm);
CREATE INDEX IF NOT EXISTS books_goodreads_idx  ON books (goodreads_id);
CREATE INDEX IF NOT EXISTS books_asin_idx       ON books (asin);
CREATE INDEX IF NOT EXISTS books_isbn13_idx     ON books (isbn13);

-- Directed, typed, rank-preserving. The primary key is what makes ingest
-- idempotent: re-running a dump replaces rows instead of accumulating weight.
CREATE TABLE IF NOT EXISTS edges_raw (
    src     VARCHAR NOT NULL,
    dst     VARCHAR NOT NULL,
    kind    VARCHAR NOT NULL,
    rank    INTEGER NOT NULL DEFAULT 1,
    source  VARCHAR NOT NULL,
    PRIMARY KEY (src, dst, kind, source)
);

CREATE INDEX IF NOT EXISTS edges_raw_src_idx ON edges_raw (src);
CREATE INDEX IF NOT EXISTS edges_raw_dst_idx ON edges_raw (dst);

-- Derived by graph.build.fuse_edges. Stored canonically with src < dst so an
-- undirected pair can never appear twice.
CREATE TABLE IF NOT EXISTS edges_fused (
    src       VARCHAR NOT NULL,
    dst       VARCHAR NOT NULL,
    weight    DOUBLE  NOT NULL,
    dir_asym  DOUBLE  NOT NULL DEFAULT 0.0,
    kinds     VARCHAR[],
    PRIMARY KEY (src, dst)
);

CREATE INDEX IF NOT EXISTS edges_fused_src_idx ON edges_fused (src);
CREATE INDEX IF NOT EXISTS edges_fused_dst_idx ON edges_fused (dst);

-- Maps any upstream identifier onto a canonical work_id.
CREATE TABLE IF NOT EXISTS aliases (
    id_type VARCHAR NOT NULL,   -- 'isbn13' | 'asin' | 'goodreads_id' | 'openlibrary_id'
    id_value VARCHAR NOT NULL,
    work_id VARCHAR NOT NULL,
    PRIMARY KEY (id_type, id_value)
);

-- Community + centrality results, so the map does not recompute them per request.
CREATE TABLE IF NOT EXISTS node_metrics (
    work_id     VARCHAR PRIMARY KEY,
    degree      INTEGER,
    weighted_degree DOUBLE,
    community   INTEGER,
    betweenness DOUBLE
);

CREATE TABLE IF NOT EXISTS communities (
    community INTEGER PRIMARY KEY,
    label     VARCHAR,
    size      INTEGER
);

-- Makes multi-gigabyte ingests resumable and records what has been loaded.
CREATE TABLE IF NOT EXISTS ingest_state (
    source        VARCHAR NOT NULL,
    artifact      VARCHAR NOT NULL,   -- file path or URL that was ingested
    rows_done     BIGINT  NOT NULL DEFAULT 0,
    books_added   BIGINT  NOT NULL DEFAULT 0,
    edges_added   BIGINT  NOT NULL DEFAULT 0,
    completed     BOOLEAN NOT NULL DEFAULT FALSE,
    updated_at    TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (source, artifact)
);
