from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any


INSPECTION_COLUMNS = [
    "timestamp",
    "image_path",
    "overlay_path",
    "product_type",
    "model_result",
    "anomaly_score",
    "roi_confidence",
    "edge_damage_score",
    "deformation_score",
    "size_tolerance_score",
    "color_anomaly_score",
    "glass_burn_score",
    "raw_fiber_score",
    "crack_score",
    "local_anomaly_score",
    "operator_label",
    "operator_note",
    "is_model_wrong",
    "is_confirmed",
    "previous_overlay_path",
    "previous_model_result",
    "previous_anomaly_score",
    "last_reprocessed_at",
    "rule_details",
]

# update_model_result çağrılarında kabul edilen skor kolonları (kolon adıyla).
MODEL_SCORE_COLUMNS = [
    "edge_damage_score",
    "deformation_score",
    "size_tolerance_score",
    "color_anomaly_score",
    "glass_burn_score",
    "raw_fiber_score",
    "crack_score",
    "local_anomaly_score",
]


class SQLiteStore:
    """Small SQLite wrapper for inspection records."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def initialize(self) -> None:
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS inspection_records (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        image_path TEXT NOT NULL,
                        overlay_path TEXT NOT NULL,
                        product_type TEXT,
                        model_result TEXT,
                        anomaly_score REAL,
                        roi_confidence REAL,
                        edge_damage_score REAL,
                        deformation_score REAL,
                        size_tolerance_score REAL,
                        color_anomaly_score REAL,
                        glass_burn_score REAL,
                        raw_fiber_score REAL,
                        crack_score REAL,
                        local_anomaly_score REAL,
                        operator_label TEXT,
                        operator_note TEXT,
                        is_model_wrong INTEGER NOT NULL DEFAULT 0,
                        is_confirmed INTEGER NOT NULL DEFAULT 0,
                        previous_overlay_path TEXT,
                        previous_model_result TEXT,
                        previous_anomaly_score REAL,
                        last_reprocessed_at TEXT
                    )
                    """
                )
                self._ensure_column(connection, "local_anomaly_score", "REAL")
                self._ensure_column(connection, "roi_confidence", "REAL")
                self._ensure_column(connection, "deformation_score", "REAL")
                self._ensure_column(connection, "glass_burn_score", "REAL")
                self._ensure_column(connection, "raw_fiber_score", "REAL")
                self._ensure_column(connection, "size_tolerance_score", "REAL")
                self._ensure_column(connection, "rule_details", "TEXT")
                self._ensure_column(connection, "is_model_wrong", "INTEGER NOT NULL DEFAULT 0")
                self._ensure_column(connection, "previous_overlay_path", "TEXT")
                self._ensure_column(connection, "previous_model_result", "TEXT")
                self._ensure_column(connection, "previous_anomaly_score", "REAL")
                self._ensure_column(connection, "last_reprocessed_at", "TEXT")
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS operator_feedback (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        record_id INTEGER NOT NULL,
                        roi_ok INTEGER NOT NULL DEFAULT 1,
                        expected_verdict TEXT NOT NULL,
                        expected_defects TEXT NOT NULL DEFAULT '[]',
                        note TEXT NOT NULL DEFAULT '',
                        created_at TEXT NOT NULL,
                        FOREIGN KEY(record_id) REFERENCES inspection_records(id) ON DELETE CASCADE
                    )
                    """
                )
                self._ensure_column(connection, "roi_ok", "INTEGER NOT NULL DEFAULT 1", table="operator_feedback")

    def insert_inspection_record(self, record: dict[str, Any]) -> int:
        columns = INSPECTION_COLUMNS
        placeholders = ", ".join("?" for _ in columns)
        values = [record.get(column) for column in columns]

        with closing(self._connect()) as connection:
            with connection:
                cursor = connection.execute(
                    f"INSERT INTO inspection_records ({', '.join(columns)}) VALUES ({placeholders})",
                    values,
                )
                return int(cursor.lastrowid)

    def fetch_recent_inspection_records(
        self,
        limit: int = 50,
        model_result: str | None = None,
        operator_label: str | None = None,
    ) -> list[dict[str, Any]]:
        safe_limit = max(1, min(int(limit), 500))
        filters = []
        values: list[Any] = []
        if model_result:
            filters.append("model_result = ?")
            values.append(model_result)
        if operator_label:
            filters.append("operator_label = ?")
            values.append(operator_label)

        where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
        values.append(safe_limit)
        with closing(self._connect()) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                f"""
                SELECT
                    id,
                    timestamp,
                    image_path,
                    overlay_path,
                    product_type,
                    model_result,
                    anomaly_score,
                    roi_confidence,
                    edge_damage_score,
                    deformation_score,
                    size_tolerance_score,
                    color_anomaly_score,
                    glass_burn_score,
                    raw_fiber_score,
                    crack_score,
                    local_anomaly_score,
                    operator_label,
                    operator_note,
                    is_model_wrong,
                    is_confirmed,
                    previous_overlay_path,
                    previous_model_result,
                    previous_anomaly_score,
                    last_reprocessed_at,
                    rule_details
                FROM inspection_records
                {where_clause}
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                values,
            ).fetchall()
        return [dict(row) for row in rows]

    def fetch_inspection_record(self, record_id: int) -> dict[str, Any] | None:
        with closing(self._connect()) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                """
                SELECT
                    id,
                    timestamp,
                    image_path,
                    overlay_path,
                    product_type,
                    model_result,
                    anomaly_score,
                    roi_confidence,
                    edge_damage_score,
                    deformation_score,
                    size_tolerance_score,
                    color_anomaly_score,
                    glass_burn_score,
                    raw_fiber_score,
                    crack_score,
                    local_anomaly_score,
                    operator_label,
                    operator_note,
                    is_model_wrong,
                    is_confirmed,
                    previous_overlay_path,
                    previous_model_result,
                    previous_anomaly_score,
                    last_reprocessed_at,
                    rule_details
                FROM inspection_records
                WHERE id = ?
                """,
                (int(record_id),),
            ).fetchone()
        return dict(row) if row is not None else None

    def delete_inspection_record(self, record_id: int) -> dict[str, Any] | None:
        record = self.fetch_inspection_record(record_id)
        if record is None:
            return None

        with closing(self._connect()) as connection:
            with connection:
                connection.execute("DELETE FROM inspection_records WHERE id = ?", (int(record_id),))
        return record

    def delete_all_inspection_records(self) -> list[dict[str, Any]]:
        records = self.fetch_recent_inspection_records(limit=500)
        with closing(self._connect()) as connection:
            with connection:
                connection.execute("DELETE FROM inspection_records")
        return records

    def update_operator_feedback(
        self,
        record_id: int,
        operator_label: str,
        operator_note: str,
        is_model_wrong: bool = False,
    ) -> bool:
        with closing(self._connect()) as connection:
            with connection:
                cursor = connection.execute(
                    """
                    UPDATE inspection_records
                    SET operator_label = ?, operator_note = ?, is_model_wrong = ?
                    WHERE id = ?
                    """,
                    (operator_label, operator_note, int(is_model_wrong), int(record_id)),
                )
        return cursor.rowcount > 0

    def insert_operator_feedback(
        self,
        record_id: int,
        roi_ok: bool,
        expected_verdict: str,
        expected_defects: str,
        note: str,
        created_at: str,
    ) -> int:
        with closing(self._connect()) as connection:
            with connection:
                cursor = connection.execute(
                    """
                    INSERT INTO operator_feedback (
                        record_id,
                        roi_ok,
                        expected_verdict,
                        expected_defects,
                        note,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (int(record_id), int(roi_ok), expected_verdict, expected_defects, note, created_at),
                )
                return int(cursor.lastrowid)

    def fetch_operator_feedback(self, limit: int = 500) -> list[dict[str, Any]]:
        safe_limit = max(1, min(int(limit), 1000))
        with closing(self._connect()) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT
                    feedback.id,
                    feedback.record_id,
                    feedback.roi_ok,
                    feedback.expected_verdict,
                    feedback.expected_defects,
                    feedback.note,
                    feedback.created_at,
                    records.model_result,
                    records.anomaly_score,
                    records.roi_confidence,
                    records.edge_damage_score,
                    records.deformation_score,
                    records.color_anomaly_score,
                    records.glass_burn_score,
                    records.raw_fiber_score,
                    records.crack_score,
                    records.local_anomaly_score
                FROM operator_feedback AS feedback
                LEFT JOIN inspection_records AS records ON records.id = feedback.record_id
                ORDER BY feedback.created_at DESC, feedback.id DESC
                LIMIT ?
                """,
                (safe_limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def update_model_result(
        self,
        record_id: int,
        model_result: str,
        anomaly_score: float,
        roi_confidence: float,
        scores: dict[str, float],
        *,
        rule_details: str | None = None,
        previous_overlay_path: str | None = None,
        previous_model_result: str | None = None,
        previous_anomaly_score: float | None = None,
        last_reprocessed_at: str | None = None,
    ) -> bool:
        """Yeniden işleme sonrası model çıktılarını günceller.

        ``scores`` kolon adıyla anahtarlanmış skor sözlüğüdür
        (bkz. ``MODEL_SCORE_COLUMNS``); eksik kolonlar 0.0 kabul edilir.
        """
        score_values = [float(scores.get(column, 0.0) or 0.0) for column in MODEL_SCORE_COLUMNS]
        score_assignments = ",\n                        ".join(
            f"{column} = ?" for column in MODEL_SCORE_COLUMNS
        )
        with closing(self._connect()) as connection:
            with connection:
                cursor = connection.execute(
                    f"""
                    UPDATE inspection_records
                    SET
                        model_result = ?,
                        anomaly_score = ?,
                        roi_confidence = ?,
                        {score_assignments},
                        rule_details = ?,
                        previous_overlay_path = ?,
                        previous_model_result = ?,
                        previous_anomaly_score = ?,
                        last_reprocessed_at = ?
                    WHERE id = ?
                    """,
                    (
                        model_result,
                        anomaly_score,
                        roi_confidence,
                        *score_values,
                        rule_details,
                        previous_overlay_path,
                        previous_model_result,
                        previous_anomaly_score,
                        last_reprocessed_at,
                        int(record_id),
                    ),
                )
        return cursor.rowcount > 0

    def fetch_quality_summary(self) -> dict[str, Any]:
        with closing(self._connect()) as connection:
            connection.row_factory = sqlite3.Row
            totals = connection.execute(
                """
                SELECT
                    COUNT(*) AS total_count,
                    AVG(anomaly_score) AS avg_anomaly_score,
                    AVG(edge_damage_score) AS avg_edge_damage_score,
                    AVG(deformation_score) AS avg_deformation_score,
                    AVG(color_anomaly_score) AS avg_color_anomaly_score,
                    AVG(glass_burn_score) AS avg_glass_burn_score,
                    AVG(raw_fiber_score) AS avg_raw_fiber_score,
                    AVG(crack_score) AS avg_crack_score,
                    AVG(local_anomaly_score) AS avg_local_anomaly_score,
                    SUM(is_model_wrong) AS wrong_count
                FROM inspection_records
                """
            ).fetchone()
            label_rows = connection.execute(
                """
                SELECT COALESCE(model_result, 'BILINMIYOR') AS label, COUNT(*) AS count
                FROM inspection_records
                GROUP BY COALESCE(model_result, 'BILINMIYOR')
                ORDER BY count DESC
                """
            ).fetchall()
            operator_rows = connection.execute(
                """
                SELECT COALESCE(operator_label, 'etiketsiz') AS label, COUNT(*) AS count
                FROM inspection_records
                GROUP BY COALESCE(operator_label, 'etiketsiz')
                ORDER BY count DESC
                """
            ).fetchall()

        return {
            "totals": dict(totals) if totals is not None else {},
            "model_result_counts": [dict(row) for row in label_rows],
            "operator_label_counts": [dict(row) for row in operator_rows],
        }

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.database_path)

    def _ensure_column(
        self,
        connection: sqlite3.Connection,
        column: str,
        definition: str,
        *,
        table: str = "inspection_records",
    ) -> None:
        existing_columns = {
            row[1]
            for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column not in existing_columns:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
