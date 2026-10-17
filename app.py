from __future__ import annotations
import csv
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import io
import json
import math
import os
from pathlib import Path
import sqlite3
from typing import Dict, Any, Optional, List
import uuid

from flask import Flask, jsonify, render_template, request, send_file, abort
from services.classifier import classify_feedback, classify_batch, normalize_text, get_policy, get_highlight_terms, CRITICAL
from services.first_signal import detect_signals, OWNERS
from services.reports import build_weekly_report, build_multi_region_brief, report_csv, parse_dt
from services.store_service import get_store, get_all_stores

BASE = Path(__file__).resolve().parent
DB_PATH = Path(os.environ.get("R5_DB_PATH", BASE / "data" / "r5.sqlite3"))
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

ISSUE_OWNERS = {
    "food_safety": "Store Operations",
    "allergen_label": "Store Operations",
    "hygiene": "Store Operations",
    "staff_misconduct": "HR",
    "billing": "Digital",
    "payment_failure": "Digital",
    "stockout": "Supply Chain",
    "freezer_failure": "Store Operations",
    "pricing": "Merchandising",
    "other": "Customer Experience",
}

def connect():
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA cache_size = -32000")
    return conn

@contextmanager
def db_session():
    conn = connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def ensure_columns(conn):
    # Ensure all tables exist and any newer columns are added gracefully
    cursor = conn.cursor()
    
    # feedback table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS feedback (
        id TEXT PRIMARY KEY,
        source_id TEXT,
        store_id TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        channel TEXT NOT NULL,
        original_text TEXT NOT NULL,
        cleaned_text TEXT NOT NULL,
        issue_type TEXT NOT NULL,
        severity TEXT NOT NULL,
        actionable INTEGER NOT NULL,
        confidence REAL NOT NULL,
        status TEXT NOT NULL DEFAULT 'New',
        metadata_json TEXT NOT NULL DEFAULT '{}',
        demo_record TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL
    );
    """)
    
    # stores table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS stores (
        store_id TEXT PRIMARY KEY,
        store_name TEXT NOT NULL,
        region TEXT NOT NULL,
        city TEXT NOT NULL,
        address TEXT,
        latitude REAL,
        longitude REAL,
        manager_name TEXT,
        phone TEXT,
        is_demo INTEGER NOT NULL DEFAULT 1
    );
    """)

    # incidents table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS incidents (
        id TEXT PRIMARY KEY,
        store_id TEXT NOT NULL,
        issue_type TEXT NOT NULL,
        summary TEXT NOT NULL,
        description TEXT,
        why_raised TEXT NOT NULL,
        report_count INTEGER NOT NULL,
        first_seen TEXT NOT NULL,
        latest_seen TEXT NOT NULL,
        severity TEXT NOT NULL,
        department TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'Detected — awaiting review',
        assigned_user TEXT,
        priority TEXT NOT NULL,
        due_at TEXT,
        next_action TEXT NOT NULL,
        policy_rule TEXT,
        escalated INTEGER NOT NULL DEFAULT 0,
        escalation_reason TEXT,
        investigation_findings TEXT,
        investigation_verdict TEXT,
        corrective_action TEXT,
        resolution TEXT,
        root_cause TEXT,
        is_demo INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    """)

    # incident_evidence table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS incident_evidence (
        incident_id TEXT NOT NULL REFERENCES incidents(id) ON DELETE CASCADE,
        feedback_id TEXT NOT NULL REFERENCES feedback(id),
        grouping_reason TEXT,
        PRIMARY KEY (incident_id, feedback_id)
    );
    """)

    # audit_log table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS audit_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        incident_id TEXT NOT NULL,
        actor TEXT NOT NULL,
        action TEXT NOT NULL,
        old_value TEXT,
        new_value TEXT,
        note TEXT,
        timestamp TEXT NOT NULL
    );
    """)

    # resolution_memory table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS resolution_memory (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        incident_id TEXT NOT NULL,
        issue_type TEXT NOT NULL,
        store_id TEXT NOT NULL,
        root_cause TEXT,
        actions TEXT NOT NULL,
        outcome TEXT NOT NULL,
        recorded_at TEXT NOT NULL
    );
    """)

    # Check and add columns if upgrading existing db
    def add_col_if_missing(table, col_name, col_type):
        c = conn.execute(f"PRAGMA table_info({table})")
        cols = [r[1] for r in c.fetchall()]
        if col_name not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}")

    add_col_if_missing("feedback", "demo_record", "TEXT NOT NULL DEFAULT ''")
    add_col_if_missing("incidents", "description", "TEXT")
    add_col_if_missing("incidents", "policy_rule", "TEXT")
    add_col_if_missing("incidents", "escalated", "INTEGER NOT NULL DEFAULT 0")
    add_col_if_missing("incidents", "escalation_reason", "TEXT")
    add_col_if_missing("incidents", "investigation_findings", "TEXT")
    add_col_if_missing("incidents", "investigation_verdict", "TEXT")
    add_col_if_missing("incidents", "corrective_action", "TEXT")
    add_col_if_missing("incidents", "is_demo", "INTEGER NOT NULL DEFAULT 0")
    add_col_if_missing("incident_evidence", "grouping_reason", "TEXT")

    # Indices
    conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_store_time ON feedback(store_id, timestamp);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_issue ON feedback(issue_type);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_severity ON feedback(severity);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_status ON feedback(status);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_incidents_store ON incidents(store_id);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_incidents_status ON incidents(status);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_incidents_sev ON incidents(severity);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_incident ON audit_log(incident_id);")

def init_db():
    with db_session() as c:
        ensure_columns(c)
        # Seed stores
        seed_stores_data(c)

def seed_stores_data(conn):
    stores = get_all_stores()
    for s in stores.values():
        conn.execute("""
        INSERT OR REPLACE INTO stores (store_id, store_name, region, city, address, latitude, longitude, manager_name, phone, is_demo)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            s["store_id"], s["store_name"], s["region"], s["city"],
            s.get("address"), s.get("latitude"), s.get("longitude"),
            s.get("manager_name"), s.get("phone"), s.get("is_demo", 1)
        ))

def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def rowdict(row) -> Optional[Dict[str, Any]]:
    return dict(row) if row else None

def parse_timestamp(value: Any) -> Optional[str]:
    if not value:
        return now_iso()
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()
    except ValueError:
        return None

def create_feedback(record: Dict[str, Any]):
    original = str(record.get("text") or record.get("feedback") or record.get("comment") or "").strip()
    store_id = str(record.get("store_id") or "").strip().upper()
    timestamp = parse_timestamp(record.get("timestamp") or record.get("created_at"))
    if not original or not store_id or not timestamp:
        return None, "Missing text/store_id or invalid timestamp"
    
    prediction = classify_feedback(original)
    issue = str(record.get("issue_type") or prediction["issue_type"])
    severity = str(record.get("severity") or prediction["severity"])
    actionable = bool(record.get("actionable", prediction["actionable"]))
    fid = str(record.get("id") or record.get("feedback_id") or f"FB-{uuid.uuid4().hex[:8].upper()}")
    channel = str(record.get("channel") or record.get("source") or "staff_note")
    demo_rec = str(record.get("demo_record") or ("Demo data" if fid.startswith("DEMO-") else ""))
    
    metadata = {k: v for k, v in record.items() if k not in {"text", "feedback", "comment"}}
    
    with db_session() as c:
        c.execute("""
        INSERT OR IGNORE INTO feedback
        (id, source_id, store_id, timestamp, channel, original_text, cleaned_text, issue_type, severity, actionable, confidence, status, metadata_json, demo_record, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            fid, str(record.get("source_id") or ""), store_id, timestamp, channel, original,
            normalize_text(original), issue, severity, int(actionable), float(prediction["confidence"]),
            "New", json.dumps(metadata, ensure_ascii=False), demo_rec, now_iso()
        ))
        inserted = c.execute("SELECT changes()").fetchone()[0] == 1
        
    if not inserted:
        return fid, "Duplicate ID ignored"
    return fid, None

def get_feedback(filters=None):
    filters = filters or {}
    sql = "SELECT f.*, ie.incident_id as linked_incident_id FROM feedback f LEFT JOIN incident_evidence ie ON ie.feedback_id = f.id WHERE 1=1"
    args = []
    
    for key, col in [("store_id", "f.store_id"), ("issue_type", "f.issue_type"), ("severity", "f.severity"), ("channel", "f.channel"), ("status", "f.status")]:
        if filters.get(key):
            sql += f" AND {col}=?"
            args.append(filters[key])
            
    if filters.get("actionable") is not None and filters.get("actionable") != "":
        act_val = 1 if str(filters["actionable"]).lower() in {"1", "true", "yes"} else 0
        sql += " AND f.actionable=?"
        args.append(act_val)

    if filters.get("q"):
        q_val = f"%{filters['q']}%"
        sql += " AND (f.original_text LIKE ? OR f.id LIKE ? OR f.store_id LIKE ?)"
        args.extend([q_val, q_val, q_val])
        
    if filters.get("start"):
        sql += " AND f.timestamp>=?"
        args.append(filters["start"])
        
    if filters.get("end"):
        sql += " AND f.timestamp<?"
        args.append(filters["end"])
        
    sql += " GROUP BY f.id ORDER BY f.timestamp DESC LIMIT ?"
    args.append(min(int(filters.get("limit", 500)), 5000))
    
    with db_session() as c:
        return [rowdict(r) for r in c.execute(sql, args).fetchall()]

def sync_first_signal():
    records = get_feedback({"limit": 5000})
    signals = detect_signals(records)
    
    with db_session() as c:
        for sig in signals:
            existing = c.execute("SELECT * FROM incidents WHERE id=?", (sig["id"],)).fetchone()
            if existing:
                existing = rowdict(existing)
                # Keep active workflow status, assignments, investigations, etc.
                # Update report count, latest timestamp, and next action if awaiting review
                c.execute("""
                UPDATE incidents SET
                    report_count = ?,
                    latest_seen = ?,
                    updated_at = ?
                WHERE id = ?
                """, (sig["report_count"], sig["latest_seen"], now_iso(), sig["id"]))
            else:
                desc = sig.get("description") or f"FIRST SIGNAL early warning: {sig['why_raised']}"
                c.execute("""
                INSERT INTO incidents (
                    id, store_id, issue_type, summary, description, why_raised, report_count,
                    first_seen, latest_seen, severity, department, status, assigned_user, priority,
                    due_at, next_action, policy_rule, escalated, is_demo, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    sig["id"], sig["store_id"], sig["issue_type"], sig["summary"], desc, sig["why_raised"],
                    sig["report_count"], sig["first_seen"], sig["latest_seen"], sig["severity"], sig["department"],
                    sig["status"], sig["assigned_user"], sig["priority"], sig["due_at"], sig["next_action"],
                    sig["policy_rule"], sig["escalated"], sig.get("is_demo", 0), now_iso(), now_iso()
                ))
                # Add initial audit log
                c.execute("""
                INSERT INTO audit_log (incident_id, actor, action, old_value, new_value, note, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    sig["id"], "FIRST SIGNAL Engine", "incident_detected", None, sig["status"],
                    f"FIRST SIGNAL generated lead: {sig['why_raised']}", now_iso()
                ))
                
            # Associate evidence
            for fid in sig["feedback_ids"]:
                c.execute("""
                INSERT OR IGNORE INTO incident_evidence (incident_id, feedback_id, grouping_reason)
                VALUES (?, ?, ?)
                """, (sig["id"], fid, sig.get("grouping_explanation", "Linked by store and issue cluster.")))

def enrich_incident_details(c, item: Dict[str, Any]) -> Dict[str, Any]:
    # Store info
    store = c.execute("SELECT * FROM stores WHERE store_id=?", (item["store_id"],)).fetchone()
    if store:
        item["store"] = rowdict(store)
    else:
        # Fallback store directory lookup or missing info
        fallback = get_store(item["store_id"])
        item["store"] = fallback if fallback else {
            "store_id": item["store_id"],
            "store_name": f"Store {item['store_id']}",
            "region": "Regional Operations",
            "city": "Unknown",
            "address": None,
            "latitude": None,
            "longitude": None,
            "manager_name": None,
            "phone": None,
            "is_demo": 0
        }
        
    # Policy information
    policy = get_policy(item["issue_type"])
    item["policy_details"] = policy
    item["highlight_terms"] = get_highlight_terms(item["issue_type"])
    
    # Evidence
    ev_rows = c.execute("""
    SELECT f.id, f.timestamp, f.channel, f.original_text, f.cleaned_text, f.issue_type, f.severity, f.store_id, ie.grouping_reason
    FROM feedback f
    JOIN incident_evidence ie ON ie.feedback_id = f.id
    WHERE ie.incident_id = ?
    ORDER BY f.timestamp ASC
    """, (item["id"],)).fetchall()
    
    item["evidence"] = [rowdict(x) for x in ev_rows]
    item["evidence_ids"] = [x["id"] for x in ev_rows]
    
    # Audit trail
    audits = c.execute("""
    SELECT * FROM audit_log WHERE incident_id = ? ORDER BY timestamp DESC
    """, (item["id"],)).fetchall()
    item["audit"] = [rowdict(x) for x in audits]
    
    # Workflow progress steps calculation
    statuses = [
        "Detected",
        "Assigned",
        "Investigating",
        "Action in progress",
        "Pending verification",
        "Resolved"
    ]
    current_status = item.get("status", "Detected — awaiting review")
    
    # Normalizing status to matching progress step index
    status_map = {
        "Detected — awaiting review": 0,
        "Detected": 0,
        "Assigned": 1,
        "Under investigation": 2,
        "Investigating": 2,
        "Action in progress": 3,
        "Confirmed": 3,
        "Pending verification": 4,
        "Resolved": 5,
        "Not substantiated": -1,
        "Dismissed": -1
    }
    cur_step = status_map.get(current_status, 0)
    
    steps = []
    for idx, s in enumerate(statuses):
        if cur_step == -1:
            state = "dismissed"
        elif idx < cur_step:
            state = "completed"
        elif idx == cur_step:
            state = "current"
        else:
            state = "upcoming"
        steps.append({"step": s, "state": state, "index": idx + 1})
    item["progress_steps"] = steps
    item["is_dismissed"] = (cur_step == -1)
    
    return item

# --- Web routes ---

@app.get("/")
def index():
    return render_template("index.html")

@app.get("/incidents/<incident_id>")
def incident_page(incident_id):
    # Direct deep-link route to an incident, rendering the same app shell
    return render_template("index.html")

# --- API routes ---

@app.get("/api/health")
def health():
    with db_session() as c:
        fb_count = c.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        inc_count = c.execute("SELECT COUNT(*) FROM incidents").fetchone()[0]
        active_count = c.execute("SELECT COUNT(*) FROM incidents WHERE status NOT IN ('Resolved', 'Not substantiated', 'Dismissed')").fetchone()[0]
    return jsonify({
        "status": "ok",
        "feedback_count": fb_count,
        "incident_count": inc_count,
        "active_incidents": active_count,
        "timestamp": now_iso()
    })

@app.get("/api/overview")
def overview():
    sync_first_signal()
    with db_session() as c:
        total_feedback = c.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        active_incidents = c.execute("SELECT COUNT(*) FROM incidents WHERE status NOT IN ('Resolved', 'Not substantiated', 'Dismissed')").fetchone()[0]
        critical_high = c.execute("SELECT COUNT(*) FROM incidents WHERE severity IN ('Critical', 'High') AND status NOT IN ('Resolved', 'Not substantiated', 'Dismissed')").fetchone()[0]
        awaiting_verif = c.execute("SELECT COUNT(*) FROM incidents WHERE status = 'Pending verification'").fetchone()[0]
        resolved = c.execute("SELECT COUNT(*) FROM incidents WHERE status = 'Resolved'").fetchone()[0]

        # Needs attention: urgent active incidents
        urgent_rows = c.execute("""
        SELECT * FROM incidents
        WHERE status NOT IN ('Resolved', 'Not substantiated', 'Dismissed')
        ORDER BY
            CASE severity
                WHEN 'Critical' THEN 0
                WHEN 'High' THEN 1
                WHEN 'Medium' THEN 2
                ELSE 3
            END,
            latest_seen DESC
        LIMIT 6
        """).fetchall()
        needs_attention = [enrich_incident_details(c, rowdict(r)) for r in urgent_rows]

        # Recent activity from audit log
        audit_rows = c.execute("""
        SELECT a.*, i.summary, i.store_id, i.severity
        FROM audit_log a
        LEFT JOIN incidents i ON i.id = a.incident_id
        ORDER BY a.timestamp DESC
        LIMIT 10
        """).fetchall()
        recent_activity = [rowdict(r) for r in audit_rows]

        # Emerging signals (FIRST SIGNAL newly generated or awaiting review)
        emerging_rows = c.execute("""
        SELECT * FROM incidents
        WHERE status IN ('Detected — awaiting review', 'Assigned')
        ORDER BY
            CASE severity
                WHEN 'Critical' THEN 0
                WHEN 'High' THEN 1
                WHEN 'Medium' THEN 2
                ELSE 3
            END,
            latest_seen DESC
        LIMIT 6
        """).fetchall()
        emerging_issues = [enrich_incident_details(c, rowdict(r)) for r in emerging_rows]

    return jsonify({
        "stats": {
            "total_feedback": total_feedback,
            "active_incidents": active_incidents,
            "critical_high_incidents": critical_high,
            "awaiting_verification": awaiting_verif,
            "resolved_incidents": resolved
        },
        # Backward-compatible fields
        "feedback_count": total_feedback,
        "unresolved_count": active_incidents,
        "high_priority_count": critical_high,
        "awaiting_verification": awaiting_verif,
        "resolved_incidents": resolved,
        "incidents": needs_attention,
        "needs_attention": needs_attention,
        "recent_activity": recent_activity,
        "emerging_issues": emerging_issues
    })

@app.get("/api/notifications")
def notifications_api():
    sync_first_signal()
    with db_session() as c:
        rows = c.execute("""
        SELECT i.*, s.store_name, s.region
        FROM incidents i
        LEFT JOIN stores s ON s.store_id = i.store_id
        ORDER BY
            CASE i.severity
                WHEN 'Critical' THEN 0
                WHEN 'High' THEN 1
                WHEN 'Medium' THEN 2
                ELSE 3
            END,
            i.latest_seen DESC
        LIMIT 20
        """).fetchall()

        now = datetime.now(timezone.utc)
        items = []
        for r in rows:
            inc = rowdict(r)
            ts_str = inc.get("latest_seen") or inc.get("created_at") or inc.get("first_seen")
            dt_obj = parse_dt(ts_str)
            time_ago = "Just now"
            if dt_obj:
                diff_sec = max(int((now - dt_obj).total_seconds()), 0)
                if diff_sec < 60:
                    time_ago = "1 min ago" if diff_sec > 30 else "Just now"
                elif diff_sec < 3600:
                    mins = max(diff_sec // 60, 1)
                    time_ago = f"{mins} min{'s' if mins > 1 else ''} ago"
                elif diff_sec < 86400:
                    hrs = diff_sec // 3600
                    time_ago = f"{hrs} hour{'s' if hrs > 1 else ''} ago"
                else:
                    days = diff_sec // 86400
                    time_ago = f"{days} day{'s' if days > 1 else ''} ago"

            is_new = inc.get("status") in ("Detected — awaiting review", "Detected", "Assigned", "Under investigation", "Investigating")
            items.append({
                "id": f"notif-{inc['id']}",
                "incident_id": inc["id"],
                "header": "New alert received",
                "time_ago": time_ago,
                "timestamp": ts_str,
                "title": inc["summary"],
                "store_id": inc["store_id"],
                "store_name": inc.get("store_name") or f"Store {inc['store_id']}",
                "severity": inc["severity"],
                "status": inc["status"],
                "why_raised": inc.get("why_raised", ""),
                "is_new": is_new
            })

        return jsonify({
            "notifications": items,
            "unread_count": sum(1 for n in items if n["is_new"])
        })

@app.get("/api/incidents")
def incidents_api():
    sync_first_signal()
    with db_session() as c:
        sql = "SELECT * FROM incidents WHERE 1=1"
        args = []

        q = request.args.get("q")
        if q:
            q_val = f"%{q.strip()}%"
            sql += " AND (summary LIKE ? OR id LIKE ? OR store_id LIKE ? OR assigned_user LIKE ? OR department LIKE ?)"
            args.extend([q_val, q_val, q_val, q_val, q_val])

        status = request.args.get("status")
        if status:
            sql += " AND status = ?"
            args.append(status)

        severity = request.args.get("severity")
        if severity:
            sql += " AND severity = ?"
            args.append(severity)

        store_id = request.args.get("store_id")
        if store_id:
            sql += " AND store_id = ?"
            args.append(store_id.strip().upper())

        department = request.args.get("department")
        if department:
            sql += " AND department = ?"
            args.append(department)

        issue_type = request.args.get("issue_type")
        if issue_type:
            sql += " AND issue_type = ?"
            args.append(issue_type)

        sort_by = request.args.get("sort", "severity")
        if sort_by == "newest":
            sql += " ORDER BY latest_seen DESC"
        elif sort_by == "oldest":
            sql += " ORDER BY first_seen ASC"
        elif sort_by == "due_date":
            sql += " ORDER BY due_at ASC"
        elif sort_by == "evidence":
            sql += " ORDER BY report_count DESC"
        else: # severity
            sql += """ ORDER BY
                CASE severity
                    WHEN 'Critical' THEN 0
                    WHEN 'High' THEN 1
                    WHEN 'Medium' THEN 2
                    ELSE 3
                END,
                updated_at DESC"""

        rows = c.execute(sql, args).fetchall()
        result = [enrich_incident_details(c, rowdict(r)) for r in rows]

    return jsonify(result)

@app.get("/api/incidents/<incident_id>")
def get_incident(incident_id):
    sync_first_signal()
    with db_session() as c:
        row = c.execute("SELECT * FROM incidents WHERE id=?", (incident_id,)).fetchone()
        if not row:
            return jsonify({"error": "Incident not found"}), 404
        data = enrich_incident_details(c, rowdict(row))
    return jsonify(data)

@app.post("/api/incidents/<incident_id>/view")
def track_incident_view(incident_id):
    data = request.get_json(silent=True) or {}
    actor = str(data.get("actor") or "Alex Chen (Regional Director)")
    with db_session() as c:
        if not c.execute("SELECT 1 FROM incidents WHERE id=?", (incident_id,)).fetchone():
            return jsonify({"error": "Incident not found"}), 404
        c.execute("""
        INSERT INTO audit_log (incident_id, actor, action, old_value, new_value, note, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            incident_id, actor, "investigation_opened", None, "viewed",
            "Reviewer accessed detailed location, evidence, and severity record via View Location & Severity.",
            now_iso()
        ))
    return jsonify({"message": "Incident view recorded in audit log"})

@app.patch("/api/incidents/<incident_id>")
def update_incident(incident_id):
    data = request.get_json(silent=True) or {}
    allowed = {
        "status", "assigned_user", "department", "severity", "priority",
        "due_at", "next_action", "resolution", "root_cause",
        "investigation_findings", "investigation_verdict", "corrective_action",
        "escalated", "escalation_reason"
    }
    changes = {k: v for k, v in data.items() if k in allowed}
    if not changes:
        return jsonify({"error": "No supported fields supplied"}), 400

    actor = str(data.get("actor") or "Alex Chen (Regional Director)")
    note = str(data.get("note") or "")

    with db_session() as c:
        old = c.execute("SELECT * FROM incidents WHERE id=?", (incident_id,)).fetchone()
        if not old:
            return jsonify({"error": "Incident not found"}), 404
        old = rowdict(old)

        set_clauses = []
        params = []
        for key, value in changes.items():
            set_clauses.append(f"{key}=?")
            params.append(value)
            # Log audit entry
            c.execute("""
            INSERT INTO audit_log (incident_id, actor, action, old_value, new_value, note, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                incident_id, actor, f"updated_{key}", str(old.get(key)), str(value),
                note if note else f"Updated {key} to '{value}'", now_iso()
            ))

        params.extend([now_iso(), incident_id])
        c.execute(f"UPDATE incidents SET {', '.join(set_clauses)}, updated_at=? WHERE id=?", params)

        current = rowdict(c.execute("SELECT * FROM incidents WHERE id=?", (incident_id,)).fetchone())

        # If resolved and has details, store in resolution memory
        if current.get("status") == "Resolved":
            c.execute("""
            INSERT INTO resolution_memory (incident_id, issue_type, store_id, root_cause, actions, outcome, recorded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                incident_id, current["issue_type"], current["store_id"],
                current.get("root_cause") or "Root cause verified during investigation",
                current.get("corrective_action") or current.get("resolution") or "Standard corrective action applied",
                "Resolved successfully", now_iso()
            ))

        enriched = enrich_incident_details(c, current)

    return jsonify({"message": "Incident updated successfully", "incident": enriched})

@app.post("/api/incidents/<incident_id>/escalate")
def escalate_incident(incident_id):
    data = request.get_json(silent=True) or {}
    actor = str(data.get("actor") or "Alex Chen (Regional Director)")
    reason = str(data.get("reason") or "Critical policy threshold breached; escalated to Regional Operations Lead.")
    
    with db_session() as c:
        old = c.execute("SELECT * FROM incidents WHERE id=?", (incident_id,)).fetchone()
        if not old:
            return jsonify({"error": "Incident not found"}), 404
        old = rowdict(old)
        
        c.execute("""
        UPDATE incidents SET
            escalated = 1,
            severity = CASE WHEN severity = 'Critical' THEN 'Critical' ELSE 'High' END,
            priority = 'Critical',
            escalation_reason = ?,
            status = 'Pending verification',
            updated_at = ?
        WHERE id = ?
        """, (reason, now_iso(), incident_id))
        
        c.execute("""
        INSERT INTO audit_log (incident_id, actor, action, old_value, new_value, note, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            incident_id, actor, "incident_escalated", old.get("severity"), "Critical",
            f"Escalation logged: {reason}", now_iso()
        ))
        current = enrich_incident_details(c, rowdict(c.execute("SELECT * FROM incidents WHERE id=?", (incident_id,)).fetchone()))
        
    return jsonify({"message": "Incident escalated according to safety policy", "incident": current})

@app.post("/api/incidents/<incident_id>/investigate")
def record_investigation(incident_id):
    data = request.get_json(silent=True) or {}
    actor = str(data.get("actor") or "Alex Chen (Regional Director)")
    findings = str(data.get("findings") or "")
    verdict = str(data.get("verdict") or "Confirmed")
    corrective_action = str(data.get("corrective_action") or "")
    next_status = str(data.get("status") or ("Action in progress" if verdict == "Confirmed" else ("Not substantiated" if verdict == "Not substantiated" else "Investigating")))

    with db_session() as c:
        old = c.execute("SELECT * FROM incidents WHERE id=?", (incident_id,)).fetchone()
        if not old:
            return jsonify({"error": "Incident not found"}), 404
            
        c.execute("""
        UPDATE incidents SET
            investigation_findings = ?,
            investigation_verdict = ?,
            corrective_action = ?,
            status = ?,
            updated_at = ?
        WHERE id = ?
        """, (findings, verdict, corrective_action, next_status, now_iso(), incident_id))
        
        c.execute("""
        INSERT INTO audit_log (incident_id, actor, action, old_value, new_value, note, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            incident_id, actor, "investigation_recorded", verdict, next_status,
            f"Findings: {findings[:120]} | Action: {corrective_action[:120]}", now_iso()
        ))
        current = enrich_incident_details(c, rowdict(c.execute("SELECT * FROM incidents WHERE id=?", (incident_id,)).fetchone()))
        
    return jsonify({"message": "Investigation details recorded", "incident": current})

@app.post("/api/incidents/<incident_id>/evidence")
def attach_evidence(incident_id):
    data = request.get_json(silent=True) or {}
    feedback_id = str(data.get("feedback_id") or "")
    reason = str(data.get("reason") or "Corroborating report linked by reviewer")
    actor = str(data.get("actor") or "Reviewer")
    
    with db_session() as c:
        if not c.execute("SELECT 1 FROM incidents WHERE id=?", (incident_id,)).fetchone():
            return jsonify({"error": "Incident not found"}), 404
        if not c.execute("SELECT 1 FROM feedback WHERE id=?", (feedback_id,)).fetchone():
            return jsonify({"error": "Feedback not found"}), 404
            
        c.execute("""
        INSERT OR IGNORE INTO incident_evidence (incident_id, feedback_id, grouping_reason)
        VALUES (?, ?, ?)
        """, (incident_id, feedback_id, reason))
        
        # Increment report count
        c.execute("""
        UPDATE incidents SET
            report_count = (SELECT COUNT(*) FROM incident_evidence WHERE incident_id = ?),
            updated_at = ?
        WHERE id = ?
        """, (incident_id, now_iso(), incident_id))
        
        c.execute("""
        INSERT INTO audit_log (incident_id, actor, action, old_value, new_value, note, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (incident_id, actor, "evidence_linked", None, feedback_id, reason, now_iso()))
        
    return jsonify({"message": "Evidence linked; original feedback record preserved"})

@app.get("/api/feedback")
def feedback_api():
    # If page parameter is supplied, return paginated dictionary
    # If not supplied, return list for 100% backward compatibility with test_app.py
    is_paginated = "page" in request.args
    filters = dict(request.args)
    
    if is_paginated:
        try:
            page = max(int(request.args.get("page", 1)), 1)
        except ValueError:
            page = 1
        try:
            limit = min(max(int(request.args.get("limit", 25)), 5), 100)
        except ValueError:
            limit = 25
            
        filters["limit"] = 5000 # get total matching
        all_records = get_feedback(filters)
        total = len(all_records)
        total_pages = max(math.ceil(total / limit), 1)
        offset = (page - 1) * limit
        page_records = all_records[offset:offset + limit]
        
        return jsonify({
            "records": page_records,
            "total": total,
            "page": page,
            "limit": limit,
            "total_pages": total_pages
        })
    else:
        return jsonify(get_feedback(filters))

@app.get("/api/feedback/<feedback_id>")
def get_single_feedback(feedback_id):
    with db_session() as c:
        row = c.execute("""
        SELECT f.*, ie.incident_id, i.summary as incident_summary, i.severity as incident_severity
        FROM feedback f
        LEFT JOIN incident_evidence ie ON ie.feedback_id = f.id
        LEFT JOIN incidents i ON i.id = ie.incident_id
        WHERE f.id = ?
        """, (feedback_id,)).fetchone()
        if not row:
            return jsonify({"error": "Feedback record not found"}), 404
        item = rowdict(row)
        try:
            item["metadata"] = json.loads(item.get("metadata_json") or "{}")
        except Exception:
            item["metadata"] = {}
        item["store"] = get_store(item["store_id"])
    return jsonify(item)

@app.delete("/api/feedback/<feedback_id>")
def delete_single_feedback(feedback_id):
    with db_session() as c:
        row = c.execute("SELECT id FROM feedback WHERE id = ?", (feedback_id,)).fetchone()
        if not row:
            return jsonify({"error": "Feedback record not found"}), 404

        # 1. Identify all incidents referencing this feedback as evidence
        linked_incidents = [
            r[0] for r in c.execute(
                "SELECT DISTINCT incident_id FROM incident_evidence WHERE feedback_id = ?",
                (feedback_id,)
            ).fetchall()
        ]

        # 2. Delete from incident_evidence
        c.execute("DELETE FROM incident_evidence WHERE feedback_id = ?", (feedback_id,))

        # 3. Update report counts / auto-detected incidents
        now_ts = now_iso()
        for inc_id in linked_incidents:
            rem_count = c.execute(
                "SELECT COUNT(*) FROM incident_evidence WHERE incident_id = ?",
                (inc_id,)
            ).fetchone()[0]

            inc_row = c.execute("SELECT * FROM incidents WHERE id = ?", (inc_id,)).fetchone()
            if inc_row:
                inc = rowdict(inc_row)
                has_manual_work = bool(
                    inc.get("investigation_findings")
                    or inc.get("investigation_verdict")
                    or inc.get("corrective_action")
                    or (inc.get("status") not in ("Detected — awaiting review", "Detected")
                        and inc.get("assigned_user") not in ("Unassigned", None, ""))
                )

                if rem_count == 0 and not has_manual_work:
                    c.execute("DELETE FROM incidents WHERE id = ?", (inc_id,))
                else:
                    latest_fb = c.execute("""
                        SELECT MAX(f.timestamp) FROM feedback f
                        JOIN incident_evidence ie ON ie.feedback_id = f.id
                        WHERE ie.incident_id = ?
                    """, (inc_id,)).fetchone()[0]

                    c.execute("""
                        UPDATE incidents SET
                            report_count = ?,
                            latest_seen = COALESCE(?, latest_seen),
                            updated_at = ?
                        WHERE id = ?
                    """, (rem_count, latest_fb, now_ts, inc_id))

                    c.execute("""
                        INSERT INTO audit_log (incident_id, actor, action, old_value, new_value, note, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        inc_id, "System", "evidence_deleted", feedback_id, None,
                        f"Linked feedback {feedback_id} was deleted; remaining report count: {rem_count}.",
                        now_ts
                    ))

        # 4. Delete the feedback record from feedback table
        c.execute("DELETE FROM feedback WHERE id = ?", (feedback_id,))

    return jsonify({
        "message": "Feedback record deleted successfully",
        "id": feedback_id,
        "deleted": True
    }), 200

@app.post("/api/feedback")
def add_feedback_api():
    data = request.get_json(silent=True) or {}
    fid, error = create_feedback(data)
    if error and error != "Duplicate ID ignored":
        return jsonify({"error": error}), 400
    sync_first_signal()
    return jsonify({"id": fid, "message": error or "Feedback ingested and signals evaluated"}), 201 if not error else 200

def import_feedback_batch(records: List[Dict[str, Any]], batch_size: int = 5000) -> Tuple[int, int, List[Dict[str, Any]]]:
    """High-throughput batched ingestion:
    - Batches text vectorization and model predictions in chunks
    - Uses executemany inside a single transaction
    - Handles deduplication and validation errors gracefully
    """
    if not records:
        return 0, 0, []

    valid_items = []
    errors = []
    
    texts_to_classify = []
    indices_to_classify = []
    
    for i, r in enumerate(records, start=1):
        original = str(r.get("text") or r.get("feedback") or r.get("comment") or "").strip()
        store_id = str(r.get("store_id") or "").strip().upper()
        timestamp = parse_timestamp(r.get("timestamp") or r.get("created_at"))
        if not original or not store_id or not timestamp:
            errors.append({"row": i, "error": "Missing text/store_id or invalid timestamp"})
            continue
            
        fid = str(r.get("id") or r.get("feedback_id") or f"FB-{uuid.uuid4().hex[:8].upper()}")
        channel = str(r.get("channel") or r.get("source") or "staff_note")
        demo_rec = str(r.get("demo_record") or ("Demo data" if fid.startswith("DEMO-") else ""))
        metadata = {k: v for k, v in r.items() if k not in {"text", "feedback", "comment"}}
        
        provided_issue = r.get("issue_type")
        provided_sev = r.get("severity")
        provided_act = r.get("actionable")
        
        item = {
            "id": fid,
            "source_id": str(r.get("source_id") or ""),
            "store_id": store_id,
            "timestamp": timestamp,
            "channel": channel,
            "original_text": original,
            "cleaned_text": normalize_text(original),
            "metadata_json": json.dumps(metadata, ensure_ascii=False),
            "demo_record": demo_rec,
            "created_at": now_iso(),
            "provided_issue": provided_issue,
            "provided_sev": provided_sev,
            "provided_act": provided_act,
            "issue_type": None,
            "severity": None,
            "actionable": None,
            "confidence": 0.85
        }
        
        if not provided_issue or not provided_sev:
            texts_to_classify.append(original)
            indices_to_classify.append(len(valid_items))
            
        valid_items.append(item)

    # 2. Batch classification if predictions needed
    if texts_to_classify:
        batch_preds = classify_batch(texts_to_classify)
        for idx_in_valid, pred in zip(indices_to_classify, batch_preds):
            item = valid_items[idx_in_valid]
            item["issue_type"] = str(item["provided_issue"] or pred["issue_type"])
            item["severity"] = str(item["provided_sev"] or pred["severity"])
            if item["provided_act"] is None:
                item["actionable"] = 1 if pred["actionable"] else 0
            else:
                item["actionable"] = 1 if str(item["provided_act"]).lower() in {"1", "true", "yes"} else 0
            item["confidence"] = float(pred["confidence"])

    for item in valid_items:
        if item["issue_type"] is None:
            item["issue_type"] = str(item["provided_issue"] or "other")
        if item["severity"] is None:
            item["severity"] = str(item["provided_sev"] or "Low")
        if item["actionable"] is None:
            item["actionable"] = 1 if str(item["provided_act"]).lower() in {"1", "true", "yes"} else 0

    # 3. Batch DB insert in single session
    inserted = 0
    duplicates = 0
    insert_sql = """
    INSERT OR IGNORE INTO feedback
    (id, source_id, store_id, timestamp, channel, original_text, cleaned_text, issue_type, severity, actionable, confidence, status, metadata_json, demo_record, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    
    with db_session() as c:
        for chunk_start in range(0, len(valid_items), batch_size):
            chunk = valid_items[chunk_start:chunk_start + batch_size]
            params = [
                (
                    it["id"], it["source_id"], it["store_id"], it["timestamp"], it["channel"],
                    it["original_text"], it["cleaned_text"], it["issue_type"], it["severity"],
                    it["actionable"], it["confidence"], "New", it["metadata_json"],
                    it["demo_record"], it["created_at"]
                )
                for it in chunk
            ]
            before_count = c.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
            c.executemany(insert_sql, params)
            after_count = c.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
            chunk_inserted = after_count - before_count
            inserted += chunk_inserted
            duplicates += (len(chunk) - chunk_inserted)

    return inserted, duplicates, errors

@app.post("/api/feedback/import")
def import_feedback():
    if "file" not in request.files:
        return jsonify({"error": "Choose a CSV or JSON file"}), 400
    file = request.files["file"]
    name = file.filename.lower()
    raw = file.read().decode("utf-8-sig", errors="replace")
    
    try:
        if name.endswith(".json"):
            data = json.loads(raw)
            records = data if isinstance(data, list) else data.get("records", [])
        elif name.endswith(".csv"):
            records = list(csv.DictReader(io.StringIO(raw)))
        else:
            return jsonify({"error": "Only CSV and JSON imports are supported"}), 400
    except Exception as exc:
        return jsonify({"error": f"Could not parse file: {exc}"}), 400

    inserted, duplicates, errors = import_feedback_batch(records)
    sync_first_signal()
    return jsonify({
        "inserted": inserted,
        "duplicates": duplicates,
        "invalid": len(errors),
        "errors": errors[:30]
    })

@app.get("/api/reports/weekly")
def weekly_report():
    days = int(request.args.get("days", 7))
    days = min(max(days, 1), 31)
    report = build_weekly_report(get_feedback({"limit": 5000}), days=days, owners=ISSUE_OWNERS)
    return jsonify(report)

@app.get("/api/reports/weekly.csv")
def weekly_csv():
    report = build_weekly_report(get_feedback({"limit": 5000}), days=7, owners=ISSUE_OWNERS)
    out = io.BytesIO(report_csv(report).encode("utf-8-sig"))
    return send_file(
        out,
        mimetype="text/csv",
        as_attachment=True,
        download_name="voice_of_store_weekly_brief.csv"
    )

@app.get("/api/reports/regional-brief")
def regional_brief_api():
    sync_first_signal()
    days = int(request.args.get("days", 7))
    days = min(max(days, 1), 60)
    records = get_feedback({"limit": 5000})
    with db_session() as c:
        inc_rows = c.execute("SELECT * FROM incidents").fetchall()
        incidents = [rowdict(r) for r in inc_rows]
    brief = build_multi_region_brief(records, incidents, days=days)
    return jsonify(brief)

@app.get("/reports/regional-brief/print")
def regional_brief_print_page():
    sync_first_signal()
    days = int(request.args.get("days", 7))
    days = min(max(days, 1), 60)
    records = get_feedback({"limit": 5000})
    with db_session() as c:
        inc_rows = c.execute("SELECT * FROM incidents").fetchall()
        incidents = [rowdict(r) for r in inc_rows]
    brief = build_multi_region_brief(records, incidents, days=days)
    return render_template("regional_brief_print.html", brief=brief)

@app.get("/api/stores")
def get_stores():
    with db_session() as c:
        rows = c.execute("SELECT * FROM stores ORDER BY region, store_name").fetchall()
        return jsonify([rowdict(r) for r in rows])

@app.get("/api/stores/<store_id>")
def store_detail(store_id):
    with db_session() as c:
        row = c.execute("SELECT * FROM stores WHERE store_id = ?", (store_id.strip().upper(),)).fetchone()
        if not row:
            # Check fallback
            fb = get_store(store_id)
            if fb:
                return jsonify(fb)
            return jsonify({"error": "Store not found"}), 404
        return jsonify(rowdict(row))

@app.get("/api/audit")
def get_audit():
    with db_session() as c:
        rows = c.execute("""
        SELECT a.*, i.summary, i.store_id, i.severity
        FROM audit_log a
        LEFT JOIN incidents i ON i.id = a.incident_id
        ORDER BY a.timestamp DESC
        LIMIT 50
        """).fetchall()
        return jsonify([rowdict(r) for r in rows])

@app.get("/api/resolution-memory")
def resolution_memory():
    issue = request.args.get("issue_type")
    with db_session() as c:
        sql = "SELECT * FROM resolution_memory"
        args = []
        if issue:
            sql += " WHERE issue_type=?"
            args.append(issue)
        sql += " ORDER BY recorded_at DESC LIMIT 50"
        return jsonify([rowdict(r) for r in c.execute(sql, args).fetchall()])

@app.post("/api/demo/reset")
def reset_demo():
    with db_session() as c:
        c.execute("DELETE FROM incident_evidence")
        c.execute("DELETE FROM incidents")
        c.execute("DELETE FROM feedback")
        c.execute("DELETE FROM audit_log")
        c.execute("DELETE FROM resolution_memory")
        seed_stores_data(c)
    seeded = seed_demo_data_if_empty()
    return jsonify({"message": f"Demo data reloaded: {seeded} feedback records created.", "count": seeded})

@app.errorhandler(404)
def not_found(_):
    if request.path.startswith("/api/"):
        return jsonify({"error": "API endpoint not found"}), 404
    return render_template("index.html")

def seed_demo_data_if_empty():
    """Load clearly labeled fictional demo records on first local launch only."""
    with db_session() as c:
        existing = c.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
    if existing:
        return 0
    demo_file = BASE / "demo_data.csv"
    if not demo_file.exists():
        return 0
    with demo_file.open("r", encoding="utf-8-sig", newline="") as f:
        records = list(csv.DictReader(f))
    inserted = 0
    for record in records:
        _, error = create_feedback(record)
        if error is None:
            inserted += 1
    if inserted:
        sync_first_signal()
    return inserted

init_db()

if __name__ == "__main__":
    if os.environ.get("R5_LOAD_DEMO", "1").strip().lower() not in {"0", "false", "no"}:
        seeded = seed_demo_data_if_empty()
        if seeded:
            print(f"Loaded {seeded} fictional demo feedback records. Replace/import your real dataset when ready.")
    port = int(os.environ.get("PORT", 5000))
    print(f"Voice of Store Engine running at http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False)
