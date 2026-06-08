"""
VeritasAI Enterprise Logging Engine.
Funnels operational logs to two targets: rotating flat file and structured SQLite database.
All 4 tables are initialized on startup: logs, queries, model_responses, feedback.
"""

import os
import sqlite3
import logging
import json
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import Optional


class VeritasLogger:
    """
    Dual-target logger: RotatingFileHandler for flat logs + SQLite for structured events.
    Never raises — all DB errors are caught and logged to flat file only.
    """

    def __init__(
        self,
        log_dir: str = "logs",
        db_name: str = "admin.db",
        level: int = logging.INFO,
    ) -> None:
        """
        Initialize logger with rotating file handler and SQLite backend.

        Args:
            log_dir: Directory to store log files and database (created if missing)
            db_name: SQLite database filename
            level: Python logging level (default: INFO)
        """
        self.log_dir = log_dir
        self.db_path = os.path.join(log_dir, db_name)
        os.makedirs(log_dir, exist_ok=True)

        # --- Flat file rotating logger ---
        self.logger = logging.getLogger("VeritasAI")
        self.logger.setLevel(level)
        self.logger.handlers = []  # Prevent duplicate handlers on Streamlit reruns

        formatter = logging.Formatter(
            "%(asctime)s | [%(levelname)s] | %(context)s | %(message)s"
        )
        file_handler = RotatingFileHandler(
            os.path.join(log_dir, "veritasai.log"),
            maxBytes=10_485_760,  # 10 MB
            backupCount=5,
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Also add a stream handler for visibility during development
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(logging.WARNING)
        self.logger.addHandler(stream_handler)

        # --- SQLite database initialization ---
        self._initialize_sqlite()

    def _initialize_sqlite(self) -> None:
        """Create all required tables in the SQLite database if they don't exist."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Table 1: Generic event log
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS logs (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts                  TEXT NOT NULL,
                    level               TEXT NOT NULL,
                    session_id          TEXT NOT NULL,
                    event               TEXT NOT NULL,
                    model               TEXT,
                    query_hash          TEXT,
                    latency_ms          INTEGER,
                    trust_score         REAL,
                    consensus_ratio     REAL,
                    error_type          TEXT,
                    error_msg           TEXT,
                    tokens_used         INTEGER,
                    hallucination_flagged INTEGER DEFAULT 0,
                    cache_hit           INTEGER DEFAULT 0,
                    component           TEXT
                )
            """)

            # Table 2: Per-query record
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS queries (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts                  TEXT NOT NULL,
                    session_id          TEXT NOT NULL,
                    query_hash          TEXT NOT NULL,
                    original_query      TEXT NOT NULL,
                    enhanced_query      TEXT NOT NULL,
                    query_type          TEXT,
                    consensus_ratio     REAL,
                    final_answer        TEXT,
                    total_latency_ms    INTEGER,
                    models_used         TEXT,
                    models_trusted      TEXT,
                    models_flagged      TEXT
                )
            """)

            # Table 3: Per-model response record
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_responses (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_hash  TEXT NOT NULL,
                    model       TEXT NOT NULL,
                    response    TEXT,
                    trust_score REAL,
                    peer_score  REAL,
                    is_outlier  INTEGER DEFAULT 0,
                    latency_ms  INTEGER,
                    tokens_used INTEGER,
                    status      TEXT,
                    error_type  TEXT
                )
            """)

            # Table 4: User feedback
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts          TEXT NOT NULL,
                    session_id  TEXT NOT NULL,
                    query_hash  TEXT NOT NULL,
                    vote        TEXT NOT NULL,
                    comment     TEXT
                )
            """)

            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(
                f"Failed to initialize SQLite schema: {e}",
                extra={"context": "SYSTEM_INIT"},
            )

    def log_event(
        self,
        level: str,
        session_id: str,
        event: str,
        component: str,
        model: Optional[str] = None,
        query_hash: Optional[str] = None,
        trust_score: Optional[float] = None,
        latency_ms: Optional[int] = None,
        consensus_ratio: Optional[float] = None,
        error_type: Optional[str] = None,
        error_msg: Optional[str] = None,
        tokens_used: int = 0,
        hallucination_flagged: int = 0,
        cache_hit: int = 0,
        message: str = "",
    ) -> None:
        """
        Write a structured event to both the flat log and SQLite.

        Args:
            level: Log level string — INFO | WARN | ERROR | DEBUG
            session_id: Current Streamlit session UUID
            event: Event name (e.g. 'llm_call_success', 'query_complete')
            component: Pipeline stage (e.g. 'enhancer', 'dispatcher', 'detector')
            model: Model name if applicable
            query_hash: SHA-256 hash of the enhanced query
            trust_score: Trust score computed for this model/result
            latency_ms: Elapsed milliseconds for this operation
            consensus_ratio: Consensus ratio from detector stage
            error_type: Error category string if applicable
            error_msg: Full error message if applicable
            tokens_used: Token count from API response
            hallucination_flagged: 1 if hallucination was detected, 0 otherwise
            cache_hit: 1 if this result was served from cache, 0 otherwise
            message: Human-readable log message
        """
        ts_str = datetime.utcnow().isoformat()
        level_map = {
            "INFO": logging.INFO,
            "WARN": logging.WARNING,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "DEBUG": logging.DEBUG,
        }
        py_level = level_map.get(level.upper(), logging.INFO)

        # Write to flat file
        self.logger.log(
            py_level,
            f"[{event}] {message}",
            extra={"context": f"sess={session_id[:8]} | comp={component} | model={model or '-'}"},
        )

        # Write to SQLite
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO logs (
                    ts, level, session_id, event, model, query_hash, latency_ms,
                    trust_score, consensus_ratio, error_type, error_msg, tokens_used,
                    hallucination_flagged, cache_hit, component
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts_str, level, session_id, event, model, query_hash, latency_ms,
                    trust_score, consensus_ratio, error_type, error_msg, tokens_used,
                    hallucination_flagged, cache_hit, component,
                ),
            )
            conn.commit()
            conn.close()
        except Exception as db_err:
            self.logger.error(
                f"SQLite write failed: {db_err}",
                extra={"context": "SYSTEM_CRITICAL"},
            )

    def log_query(
        self,
        session_id: str,
        query_hash: str,
        original_query: str,
        enhanced_query: str,
        query_type: str,
        consensus_ratio: float,
        final_answer: Optional[str],
        total_latency_ms: int,
        models_used: list,
        models_trusted: list,
        models_flagged: list,
    ) -> None:
        """
        Persist a complete query record to the queries table.

        Args:
            session_id: Current session identifier
            query_hash: SHA-256 of enhanced query
            original_query: Raw user input
            enhanced_query: Rewritten query after enhancement
            query_type: Detected type (factual/analytical/code/etc)
            consensus_ratio: Final consensus ratio
            final_answer: Synthesized answer or None if low consensus
            total_latency_ms: Total pipeline execution time
            models_used: List of all model names queried
            models_trusted: List of model names in consensus cluster
            models_flagged: List of model names flagged as outliers
        """
        ts_str = datetime.utcnow().isoformat()
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO queries (
                    ts, session_id, query_hash, original_query, enhanced_query,
                    query_type, consensus_ratio, final_answer, total_latency_ms,
                    models_used, models_trusted, models_flagged
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts_str, session_id, query_hash, original_query, enhanced_query,
                    query_type, consensus_ratio, final_answer, total_latency_ms,
                    json.dumps(models_used), json.dumps(models_trusted), json.dumps(models_flagged),
                ),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(
                f"Failed to log query record: {e}",
                extra={"context": "SYSTEM_CRITICAL"},
            )

    def log_model_response(
        self,
        query_hash: str,
        model: str,
        response: Optional[str],
        trust_score: float,
        peer_score: float,
        is_outlier: bool,
        latency_ms: int,
        tokens_used: int,
        status: str,
        error_type: Optional[str],
    ) -> None:
        """
        Persist one model's response record for a given query.

        Args:
            query_hash: The query this response belongs to
            model: Model name
            response: Response text or None
            trust_score: Computed trust score (0-1)
            peer_score: Peer ranking score (0-1)
            is_outlier: Whether this model was flagged
            latency_ms: API call duration
            tokens_used: Tokens consumed
            status: 'success' | 'failed' | 'skipped'
            error_type: Error category if applicable
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO model_responses (
                    query_hash, model, response, trust_score, peer_score,
                    is_outlier, latency_ms, tokens_used, status, error_type
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    query_hash, model, response, trust_score, peer_score,
                    1 if is_outlier else 0, latency_ms, tokens_used, status, error_type,
                ),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(
                f"Failed to log model response: {e}",
                extra={"context": "SYSTEM_CRITICAL"},
            )

    def log_feedback(
        self,
        session_id: str,
        query_hash: str,
        vote: str,
        comment: Optional[str] = None,
    ) -> None:
        """
        Record user feedback (thumbs up/down) to the feedback table.

        Args:
            session_id: Current session identifier
            query_hash: The query this feedback refers to
            vote: 'up' or 'down'
            comment: Optional text comment from user
        """
        ts_str = datetime.utcnow().isoformat()
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO feedback (ts, session_id, query_hash, vote, comment) VALUES (?, ?, ?, ?, ?)",
                (ts_str, session_id, query_hash, vote, comment),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(
                f"Failed to log feedback: {e}",
                extra={"context": "SYSTEM_CRITICAL"},
            )