"""
fan_db.py — Persistent SQLite database layer for the Axial Fan Tool.

Responsibilities
----------------
* Store fan registry entries (name, duct_dia, constants …)
* Store raw test rows per fan (same schema as TA18 / TA24 dicts)
* Track a data-hash per fan so the model store can detect stale models
* Seed the DB with the two built-in fans (TA18 / TA24) on first run
* Provide simple CRUD helpers used by the Streamlit extension pages

The DB file lives at  ./fan_data/fans.db  (created automatically).
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from contextlib import contextmanager
from typing import Optional

import pandas as pd

# ── locate / create the data directory ───────────────────────────────────────
DB_DIR  = os.path.join(os.path.dirname(__file__), "fan_data")
DB_PATH = os.path.join(DB_DIR, "fans.db")
os.makedirs(DB_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────────────────────────────────────

@contextmanager
def _conn():
    """Yield an auto-committing / auto-closing SQLite connection."""
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


# ─────────────────────────────────────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────────────────────────────────────

_DDL = """
CREATE TABLE IF NOT EXISTS fans (
    fan_id       TEXT PRIMARY KEY,          -- e.g. "18in_TA"
    display_name TEXT NOT NULL UNIQUE,      -- e.g. "18\" Tube Axial Fan"
    duct_dia_m   REAL NOT NULL,
    constants    TEXT NOT NULL,             -- JSON blob of the full constants dict
    data_hash    TEXT,                      -- SHA-256 of serialised test rows
    created_at   TEXT DEFAULT (datetime('now')),
    updated_at   TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS test_rows (
    row_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    fan_id    TEXT    NOT NULL REFERENCES fans(fan_id) ON DELETE CASCADE,
    Srno      INTEGER,
    ANGLE     REAL    NOT NULL,
    DEL_P     REAL    NOT NULL,
    SP        REAL    NOT NULL,
    W1        REAL    NOT NULL,
    W2        REAL    NOT NULL,
    Volt      REAL    NOT NULL,
    Amp       REAL    NOT NULL,
    RPM       REAL    NOT NULL
);
"""


def init_db() -> None:
    """Create tables, seed built-in fans, and auto-split any multi-pole fans in DB."""
    with _conn() as con:
        con.executescript(_DDL)

    _seed_builtin_fans()
    _split_multipole_db_fans()


def _split_multipole_db_fans() -> None:
    """Scan all fans in the DB and split any that contain test rows for multiple pole classes."""
    from data import infer_design_rpm
    for fan in list_fans():
        fid = fan["fan_id"]
        name = fan["display_name"]
        try:
            df = get_raw_df(fid)
            angle_poles = {}
            for angle in df["ANGLE"].unique():
                ang_df = df[df["ANGLE"] == angle]
                med_rpm = ang_df["RPM"].median()
                pole = infer_design_rpm([med_rpm])[1]
                angle_poles[angle] = pole
            
            unique_poles = sorted(list(set(angle_poles.values())))
            if len(unique_poles) > 1:
                constants = get_fan_constants(fid)
                for pole in unique_poles:
                    angles_for_pole = [ang for ang, p in angle_poles.items() if p == pole]
                    sub_df = df[df["ANGLE"].isin(angles_for_pole)].reset_index(drop=True)
                    sub_df["Srno"] = list(range(1, len(sub_df) + 1))
                    
                    std_rpm = 950 if pole == 6 else (1460 if pole == 4 else 2850)
                    base_id = fid.rsplit("_", 1)[0] if "_" in fid else fid
                    new_id = f"{base_id}_{pole}P"
                    base_name = name.split(" (")[0]
                    new_name = f"{base_name} ({pole}-Pole)"
                    
                    new_consts = dict(constants)
                    new_consts["poles"] = pole
                    new_consts["design_speed_rpm"] = std_rpm
                    
                    _upsert_fan(new_id, new_name, new_consts, sub_df.to_dict(orient="list"))
                delete_fan(fid)
        except Exception:
            pass



# ─────────────────────────────────────────────────────────────────────────────
# Seeding built-in fans
# ─────────────────────────────────────────────────────────────────────────────

def _seed_builtin_fans() -> None:
    """Import built-in fans from data.py into the DB."""
    from data import (
        TA14, TA16, TA18, TA24_6P, TA24_4P, TA41, TA48_6P, TA48_4P, TA54_6P, TA54_4P,
        DEFAULT_CONSTANTS_14, DEFAULT_CONSTANTS_16, DEFAULT_CONSTANTS,
        DEFAULT_CONSTANTS_24_6P, DEFAULT_CONSTANTS_24_4P,
        DEFAULT_CONSTANTS_41,
        DEFAULT_CONSTANTS_48_6P, DEFAULT_CONSTANTS_48_4P,
        DEFAULT_CONSTANTS_54_6P, DEFAULT_CONSTANTS_54_4P,
    )

    builtin_fans = [
        ("14in_TA_2P", '14" Tube Axial Fan (2-Pole)', DEFAULT_CONSTANTS_14, TA14),
        ("16in_TA_4P", '16" Tube Axial Fan (4-Pole)', DEFAULT_CONSTANTS_16, TA16),
        ("18in_TA_4P", '18" Tube Axial Fan (4-Pole)', DEFAULT_CONSTANTS, TA18),
        ("24in_TA_6P", '24" Tube Axial Fan (6-Pole)', DEFAULT_CONSTANTS_24_6P, TA24_6P),
        ("24in_TA_4P", '24" Tube Axial Fan (4-Pole)', DEFAULT_CONSTANTS_24_4P, TA24_4P),
        ("41in_TA_6P", '41" Tube Axial Fan (6-Pole)', DEFAULT_CONSTANTS_41, TA41),
        ("48in_TA_6P", '48" Tube Axial Fan (6-Pole)', DEFAULT_CONSTANTS_48_6P, TA48_6P),
        ("48in_TA_4P", '48" Tube Axial Fan (4-Pole)', DEFAULT_CONSTANTS_48_4P, TA48_4P),
        ("54in_TA_6P", '54" Tube Axial Fan (6-Pole)', DEFAULT_CONSTANTS_54_6P, TA54_6P),
        ("54in_TA_4P", '54" Tube Axial Fan (4-Pole)', DEFAULT_CONSTANTS_54_4P, TA54_4P),
    ]

    # Delete legacy un-suffixed builtin fans if present to ensure clean pole separation
    legacy_ids = {"14in_TA", "16in_TA", "18in_TA", "24in_TA", "41in_TA", "48in_TA", "54in_TA"}
    existing_ids = {f["fan_id"] for f in list_fans()}
    for leg_id in legacy_ids:
        if leg_id in existing_ids:
            delete_fan(leg_id)

    # Refresh existing_ids after legacy cleanup
    existing_ids = {f["fan_id"] for f in list_fans()}
    for fid, name, consts, raw_dict in builtin_fans:
        if fid not in existing_ids:
            _upsert_fan(
                fan_id=fid,
                display_name=name,
                constants=consts,
                raw_dict=raw_dict,
            )
        else:
            cur_const = get_fan_constants(fid)
            if "poles" not in cur_const or cur_const.get("design_speed_rpm") != consts.get("design_speed_rpm"):
                cur_const.update(consts)
                save_constants(fid, cur_const)



def _upsert_fan(
    fan_id: str,
    display_name: str,
    constants: dict,
    raw_dict: dict,
) -> None:
    """Insert (or replace) a fan and its test rows."""
    df = pd.DataFrame(raw_dict)
    data_hash = _hash_df(df)
    con_json = json.dumps(constants)

    with _conn() as con:
        con.execute(
            """INSERT OR REPLACE INTO fans
               (fan_id, display_name, duct_dia_m, constants, data_hash, updated_at)
               VALUES (?, ?, ?, ?, ?, datetime('now'))""",
            (fan_id, display_name, constants["duct_dia_m"], con_json, data_hash),
        )
        con.execute("DELETE FROM test_rows WHERE fan_id = ?", (fan_id,))
        _insert_rows(con, fan_id, df)


# ─────────────────────────────────────────────────────────────────────────────
# Public CRUD API
# ─────────────────────────────────────────────────────────────────────────────

RAW_COLS = ["Srno", "ANGLE", "DEL_P", "SP", "W1", "W2", "Volt", "Amp", "RPM"]


def list_fans() -> list[dict]:
    """Return all fans as a list of dicts (no test rows)."""
    with _conn() as con:
        rows = con.execute(
            "SELECT fan_id, display_name, duct_dia_m, constants, data_hash, "
            "       updated_at FROM fans ORDER BY display_name"
        ).fetchall()
    return [dict(r) for r in rows]


def get_fan_constants(fan_id: str) -> dict:
    """Return the constants dict for a fan."""
    with _conn() as con:
        row = con.execute(
            "SELECT constants FROM fans WHERE fan_id = ?", (fan_id,)
        ).fetchone()
    if row is None:
        raise KeyError(f"Fan '{fan_id}' not found in DB.")
    return json.loads(row["constants"])


def get_raw_df(fan_id: str) -> pd.DataFrame:
    """Return all test rows for a fan as a DataFrame."""
    with _conn() as con:
        rows = con.execute(
            "SELECT Srno, ANGLE, DEL_P, SP, W1, W2, Volt, Amp, RPM "
            "FROM test_rows WHERE fan_id = ? ORDER BY Srno",
            (fan_id,),
        ).fetchall()
    if not rows:
        raise KeyError(f"No test rows found for fan '{fan_id}'.")
    return pd.DataFrame([dict(r) for r in rows])


def save_raw_df(fan_id: str, df: pd.DataFrame) -> str:
    """
    Overwrite the test rows for an existing fan with *df*.
    Recomputes and persists the data hash.
    Returns the new hash.
    """
    new_hash = _hash_df(df)
    with _conn() as con:
        con.execute("DELETE FROM test_rows WHERE fan_id = ?", (fan_id,))
        _insert_rows(con, fan_id, df)
        con.execute(
            "UPDATE fans SET data_hash = ?, updated_at = datetime('now') "
            "WHERE fan_id = ?",
            (new_hash, fan_id),
        )
    return new_hash


def save_constants(fan_id: str, constants: dict) -> None:
    """Persist updated constants for a fan."""
    with _conn() as con:
        con.execute(
            "UPDATE fans SET constants = ?, updated_at = datetime('now') "
            "WHERE fan_id = ?",
            (json.dumps(constants), fan_id),
        )


def create_fan(
    fan_id: str,
    display_name: str,
    constants: dict,
    raw_df: Optional[pd.DataFrame] = None,
) -> None:
    """
    Add a brand-new fan to the DB.
    *raw_df* may be None — the fan starts with zero test rows.
    """
    df = raw_df if raw_df is not None else pd.DataFrame(columns=RAW_COLS)
    data_hash = _hash_df(df)
    with _conn() as con:
        con.execute(
            """INSERT INTO fans
               (fan_id, display_name, duct_dia_m, constants, data_hash)
               VALUES (?, ?, ?, ?, ?)""",
            (
                fan_id,
                display_name,
                constants["duct_dia_m"],
                json.dumps(constants),
                data_hash,
            ),
        )
        if len(df):
            _insert_rows(con, fan_id, df)


def delete_fan(fan_id: str) -> None:
    """Remove a fan and all its test rows."""
    with _conn() as con:
        con.execute("DELETE FROM fans WHERE fan_id = ?", (fan_id,))
        # CASCADE handles test_rows


def get_data_hash(fan_id: str) -> Optional[str]:
    """Return the stored data hash for a fan (None if not found)."""
    with _conn() as con:
        row = con.execute(
            "SELECT data_hash FROM fans WHERE fan_id = ?", (fan_id,)
        ).fetchone()
    return row["data_hash"] if row else None


def current_data_hash(fan_id: str) -> str:
    """Compute the hash of the *current* live test rows in the DB."""
    df = get_raw_df(fan_id)
    return _hash_df(df)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _insert_rows(con: sqlite3.Connection, fan_id: str, df: pd.DataFrame) -> None:
    """Bulk-insert test rows; silently adds missing optional columns."""
    for col in RAW_COLS:
        if col not in df.columns:
            df[col] = None
    records = [
        (
            fan_id,
            row.get("Srno"),
            row["ANGLE"],
            row["DEL_P"],
            row["SP"],
            row["W1"],
            row["W2"],
            row["Volt"],
            row["Amp"],
            row["RPM"],
        )
        for _, row in df.iterrows()
    ]
    con.executemany(
        "INSERT INTO test_rows (fan_id, Srno, ANGLE, DEL_P, SP, W1, W2, Volt, Amp, RPM) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        records,
    )


def _hash_df(df: pd.DataFrame) -> str:
    """SHA-256 of the sorted, serialised DataFrame (order-independent)."""
    canonical = df.sort_values(list(df.columns)).reset_index(drop=True)
    blob = canonical.to_json(orient="records").encode()
    return hashlib.sha256(blob).hexdigest()
