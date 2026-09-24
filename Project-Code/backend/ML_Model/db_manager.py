"""
Phase 8: Database Storage Manager (SQLite + PostgreSQL/TimescaleDB Support)
===========================================================================
Supports both PostgreSQL/TimescaleDB (for production deployment) and SQLite 
(for local dev/testing fallback).

Schema:
- telemetry: timestamp, plasma_speed, density, temperature, flow_pressure, Bx, By, Bz, Bmag, raw_json
- predictions: timestamp, cme_probability, risk_level, model_version, inference_latency
- anomalies: timestamp, anomaly_score, anomaly_flag
- alerts: timestamp, alert_type, probability, message, status
"""

import os
import json
import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Optional

try:
    import psycopg2
    import psycopg2.extras
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

DB_DIR = Path(__file__).resolve().parent / "data"
DB_DIR.mkdir(exist_ok=True)
SQLITE_DB_PATH = DB_DIR / "cme_forecasting.db"
DATABASE_URL = os.getenv("DATABASE_URL", "")


class DatabaseManager:
    def __init__(self, db_path: Optional[Path] = None, database_url: Optional[str] = None):
        self.db_path = db_path or SQLITE_DB_PATH
        self.database_url = database_url or DATABASE_URL
        self.use_postgres = bool(self.database_url and PSYCOPG2_AVAILABLE)
        self.init_db()

    def get_connection(self):
        if self.use_postgres:
            conn = psycopg2.connect(self.database_url)
            return conn
        else:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            return conn

    def init_db(self):
        """Initializes tables for telemetry, predictions, anomalies, and alerts."""
        if self.use_postgres:
            self._init_postgres()
        else:
            self._init_sqlite()

    def _init_sqlite(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            # 1. Telemetry Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS telemetry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    plasma_speed REAL,
                    density REAL,
                    temperature REAL,
                    flow_pressure REAL,
                    Bx REAL,
                    By REAL,
                    Bz REAL,
                    Bmag REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # 2. Predictions Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    cme_probability REAL NOT NULL,
                    risk_level TEXT NOT NULL,
                    model_version TEXT DEFAULT 'v1.0.0',
                    inference_latency REAL DEFAULT 0.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Auto-migrate columns if table already existed without them
            cursor.execute("PRAGMA table_info(predictions)")
            cols = [col[1] for col in cursor.fetchall()]
            if "model_version" not in cols:
                cursor.execute("ALTER TABLE predictions ADD COLUMN model_version TEXT DEFAULT 'v1.0.0'")
            if "inference_latency" not in cols:
                cursor.execute("ALTER TABLE predictions ADD COLUMN inference_latency REAL DEFAULT 0.0")

            # 3. Anomalies Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS anomalies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    anomaly_score REAL NOT NULL,
                    anomaly_flag INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # 4. Alerts Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    probability REAL NOT NULL,
                    message TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            conn.commit()
            print(f"[Phase 8 DB] SQLite initialized -> {self.db_path}")

    def _init_postgres(self):
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS telemetry (
                        id SERIAL PRIMARY KEY,
                        timestamp VARCHAR(64) NOT NULL,
                        plasma_speed DOUBLE PRECISION,
                        density DOUBLE PRECISION,
                        temperature DOUBLE PRECISION,
                        flow_pressure DOUBLE PRECISION,
                        Bx DOUBLE PRECISION,
                        By DOUBLE PRECISION,
                        Bz DOUBLE PRECISION,
                        Bmag DOUBLE PRECISION,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    CREATE TABLE IF NOT EXISTS predictions (
                        id SERIAL PRIMARY KEY,
                        timestamp VARCHAR(64) NOT NULL,
                        cme_probability DOUBLE PRECISION NOT NULL,
                        risk_level VARCHAR(32) NOT NULL,
                        model_version VARCHAR(32) NOT NULL,
                        inference_latency DOUBLE PRECISION DEFAULT 0.0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    CREATE TABLE IF NOT EXISTS anomalies (
                        id SERIAL PRIMARY KEY,
                        timestamp VARCHAR(64) NOT NULL,
                        anomaly_score DOUBLE PRECISION NOT NULL,
                        anomaly_flag INTEGER NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    CREATE TABLE IF NOT EXISTS alerts (
                        id SERIAL PRIMARY KEY,
                        timestamp VARCHAR(64) NOT NULL,
                        alert_type VARCHAR(32) NOT NULL,
                        probability DOUBLE PRECISION NOT NULL,
                        message TEXT NOT NULL,
                        status VARCHAR(32) NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                conn.commit()
                print("[Phase 8 DB] PostgreSQL/TimescaleDB tables initialized.")
        finally:
            conn.close()

    def save_record(
        self,
        timestamp: str,
        telemetry_dict: Dict[str, float],
        cme_probability: float,
        risk_level: str,
        model_version: str,
        inference_latency: float,
        anomaly_score: float,
        anomaly_flag: bool,
        alert_info: Optional[Dict[str, Any]] = None
    ):
        """Saves telemetry, prediction, anomaly score, and alert into DB."""
        # Extract base parameters from telemetry_dict
        ps = float(telemetry_dict.get("plasma_speed_kmps", telemetry_dict.get("plasma_speed", 0.0)))
        den = float(telemetry_dict.get("proton_density", telemetry_dict.get("density", 0.0)))
        temp = float(telemetry_dict.get("plasma_temperature_k", telemetry_dict.get("temperature", 0.0)))
        fp = float(telemetry_dict.get("flow_pressure_npa", telemetry_dict.get("flow_pressure", 0.0)))
        bx = float(telemetry_dict.get("bx_gse", telemetry_dict.get("Bx", 0.0)))
        by = float(telemetry_dict.get("by_gse", telemetry_dict.get("By", 0.0)))
        bz = float(telemetry_dict.get("bz_gse", telemetry_dict.get("Bz", 0.0)))
        bmag = float(telemetry_dict.get("B_magnitude", telemetry_dict.get("Bmag", 0.0)))

        if self.use_postgres:
            conn = self.get_connection()
            try:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        INSERT INTO telemetry (timestamp, plasma_speed, density, temperature, flow_pressure, Bx, By, Bz, Bmag)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (timestamp, ps, den, temp, fp, bx, by, bz, bmag)
                    )
                    halo_cme_pred = 1 if risk_level in ["WARNING", "CRITICAL"] else 0
                    cursor.execute(
                        """
                        INSERT INTO predictions (timestamp, cme_probability, halo_cme_predicted, risk_level, is_anomaly, anomaly_score, model_version, inference_latency)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (timestamp, cme_probability, halo_cme_pred, risk_level, int(anomaly_flag), anomaly_score, model_version, inference_latency)
                    )
                    cursor.execute(
                        """
                        INSERT INTO anomalies (timestamp, anomaly_score, anomaly_flag)
                        VALUES (%s, %s, %s)
                        """,
                        (timestamp, anomaly_score, int(anomaly_flag))
                    )
                    if alert_info:
                        cursor.execute(
                            """
                            INSERT INTO alerts (timestamp, alert_type, probability, message, status)
                            VALUES (%s, %s, %s, %s, %s)
                            """,
                            (
                                alert_info.get("timestamp", timestamp),
                                alert_info.get("alert_type", risk_level),
                                alert_info.get("probability", cme_probability),
                                alert_info.get("message", ""),
                                alert_info.get("status", "SENT")
                            )
                        )
                    conn.commit()
            finally:
                conn.close()
        else:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO telemetry (timestamp, plasma_speed, density, temperature, flow_pressure, Bx, By, Bz, Bmag)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (timestamp, ps, den, temp, fp, bx, by, bz, bmag)
                )
                halo_cme_pred = 1 if risk_level in ["WARNING", "CRITICAL"] else 0
                cursor.execute(
                    """
                    INSERT INTO predictions (timestamp, cme_probability, halo_cme_predicted, risk_level, is_anomaly, anomaly_score, model_version, inference_latency)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (timestamp, cme_probability, halo_cme_pred, risk_level, int(anomaly_flag), anomaly_score, model_version, inference_latency)
                )
                cursor.execute(
                    """
                    INSERT INTO anomalies (timestamp, anomaly_score, anomaly_flag)
                    VALUES (?, ?, ?)
                    """,
                    (timestamp, anomaly_score, int(anomaly_flag))
                )
                if alert_info:
                    cursor.execute(
                        """
                        INSERT INTO alerts (timestamp, alert_type, probability, message, status)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            alert_info.get("timestamp", timestamp),
                            alert_info.get("alert_type", risk_level),
                            alert_info.get("probability", cme_probability),
                            alert_info.get("message", ""),
                            alert_info.get("status", "SENT")
                        )
                    )
                conn.commit()


    def get_latest_telemetry(self) -> Optional[Dict[str, Any]]:
        if self.use_postgres:
            conn = self.get_connection()
            try:
                with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                    cursor.execute("SELECT * FROM telemetry ORDER BY id DESC LIMIT 1")
                    row = cursor.fetchone()
                    return dict(row) if row else None
            finally:
                conn.close()
        else:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM telemetry ORDER BY id DESC LIMIT 1")
                row = cursor.fetchone()
                return dict(row) if row else None

    def get_latest_prediction(self) -> Optional[Dict[str, Any]]:
        if self.use_postgres:
            conn = self.get_connection()
            try:
                with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                    cursor.execute("""
                        SELECT p.*, a.anomaly_score, a.anomaly_flag 
                        FROM predictions p 
                        LEFT JOIN anomalies a ON p.timestamp = a.timestamp 
                        ORDER BY p.id DESC LIMIT 1
                    """)
                    row = cursor.fetchone()
                    return dict(row) if row else None
            finally:
                conn.close()
        else:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT p.*, a.anomaly_score, a.anomaly_flag 
                    FROM predictions p 
                    LEFT JOIN anomalies a ON p.timestamp = a.timestamp 
                    ORDER BY p.id DESC LIMIT 1
                """)
                row = cursor.fetchone()
                return dict(row) if row else None

    def get_historical_predictions(self, limit: int = 100) -> List[Dict[str, Any]]:
        if self.use_postgres:
            conn = self.get_connection()
            try:
                with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                    cursor.execute("""
                        SELECT p.id, p.timestamp, p.cme_probability, p.risk_level, p.model_version, p.inference_latency,
                               a.anomaly_score, a.anomaly_flag,
                               t.plasma_speed, t.density, t.temperature, t.flow_pressure, t.Bx, t.By, t.Bz, t.Bmag
                        FROM predictions p
                        LEFT JOIN anomalies a ON p.timestamp = a.timestamp
                        LEFT JOIN telemetry t ON p.timestamp = t.timestamp
                        ORDER BY p.id DESC LIMIT %s
                    """, (limit,))
                    rows = cursor.fetchall()
                    return [dict(r) for r in rows]
            finally:
                conn.close()
        else:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT p.id, p.timestamp, p.cme_probability, p.risk_level, p.model_version, p.inference_latency,
                           a.anomaly_score, a.anomaly_flag,
                           t.plasma_speed, t.density, t.temperature, t.flow_pressure, t.Bx, t.By, t.Bz, t.Bmag
                    FROM predictions p
                    LEFT JOIN anomalies a ON p.timestamp = a.timestamp
                    LEFT JOIN telemetry t ON p.timestamp = t.timestamp
                    ORDER BY p.id DESC LIMIT ?
                """, (limit,))
                rows = cursor.fetchall()
                return [dict(r) for r in rows]
