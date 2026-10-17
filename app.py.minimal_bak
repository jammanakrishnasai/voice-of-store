from __future__ import annotations
import csv, io, json, os, sqlite3, uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from flask import Flask, jsonify, render_template, request, send_file, abort
from services.classifier import classify_feedback, normalize_text
from services.first_signal import detect_signals
from services.reports import build_weekly_report, report_csv

BASE = Path(__file__).resolve().parent
DB_PATH = Path(os.environ.get("R5_DB_PATH", BASE / "data" / "r5.sqlite3"))
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

ISSUE_OWNERS = {
    "food_safety": "Store Ops", "allergen_label": "Store Ops",
    "hygiene": "Store Ops", "staff_misconduct": "HR",
    "billing": "Digital", "payment_failure": "Digital",
    "stockout": "Supply Chain", "freezer_failure": "Store Ops",
    "pricing": "Merchandising", "other": "Unassigned"
}
HIGH_RISK = {"food_safety", "allergen_label", "hygiene", "staff_misconduct"}

def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def init_db():
    with connect() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS feedback (
          id TEXT PRIMARY KEY, source_id TEXT, store_id TEXT NOT NULL,
          timestamp TEXT NOT NULL, channel TEXT NOT NULL, original_text TEXT NOT NULL,
          cleaned_text TEXT NOT NULL, issue_type TEXT NOT NULL, severity TEXT NOT NULL,
          actionable INTEGER NOT NULL, confidence REAL NOT NULL, status TEXT NOT NULL DEFAULT 'New',
          metadata_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_feedback_store_time ON feedback(store_id, timestamp);
        CREATE INDEX IF NOT EXISTS idx_feedback_issue ON feedback(issue_type);
        CREATE TABLE IF NOT EXISTS incidents (
          id TEXT PRIMARY KEY, store_id TEXT NOT NULL, issue_type TEXT NOT NULL,
          summary TEXT NOT NULL, why_raised TEXT NOT NULL, report_count INTEGER NOT NULL,
          first_seen TEXT NOT NULL, latest_seen TEXT NOT NULL, severity TEXT NOT NULL,
          department TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'Detected — awaiting review',
          assigned_user TEXT, priority TEXT NOT NULL, due_at TEXT, next_action TEXT NOT NULL,
          resolution TEXT, root_cause TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS incident_evidence (
          incident_id TEXT NOT NULL REFERENCES incidents(id) ON DELETE CASCADE,
          feedback_id TEXT NOT NULL REFERENCES feedback(id),
          PRIMARY KEY (incident_id, feedback_id)
        );
        CREATE TABLE IF NOT EXISTS audit_log (
          id INTEGER PRIMARY KEY AUTOINCREMENT, incident_id TEXT NOT NULL,
          actor TEXT NOT NULL, action TEXT NOT NULL, old_value TEXT, new_value TEXT,
          note TEXT, timestamp TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS resolution_memory (
          id INTEGER PRIMARY KEY AUTOINCREMENT, incident_id TEXT NOT NULL,
          issue_type TEXT NOT NULL, store_id TEXT NOT NULL, root_cause TEXT,
          actions TEXT NOT NULL, outcome TEXT NOT NULL, recorded_at TEXT NOT NULL
        );
        """)
        c.commit()

def now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def rowdict(row):
    return dict(row) if row else None

def parse_timestamp(value):
    if not value:
        return now_iso()
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()
    except ValueError:
        return None

def create_feedback(record):
    original = str(record.get("text") or record.get("feedback") or record.get("comment") or "").strip()
    store_id = str(record.get("store_id") or "").strip()
    timestamp = parse_timestamp(record.get("timestamp") or record.get("created_at"))
    if not original or not store_id or not timestamp:
        return None, "Missing text/store_id or invalid timestamp"
    prediction = classify_feedback(original)
    issue = str(record.get("issue_type") or prediction["issue_type"])
    severity = str(record.get("severity") or prediction["severity"])
    actionable = bool(record.get("actionable", prediction["actionable"]))
    fid = str(record.get("id") or record.get("feedback_id") or uuid.uuid4().hex[:12])
    channel = str(record.get("channel") or record.get("source") or "unknown")
    metadata = {k: v for k, v in record.items() if k not in {"text","feedback","comment"}}
    with connect() as c:
        c.execute("""INSERT OR IGNORE INTO feedback
        (id,source_id,store_id,timestamp,channel,original_text,cleaned_text,issue_type,severity,actionable,confidence,status,metadata_json,created_at)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (fid, str(record.get("source_id") or ""), store_id, timestamp, channel, original,
         normalize_text(original), issue, severity, int(actionable), float(prediction["confidence"]),
         "New", json.dumps(metadata, ensure_ascii=False), now_iso()))
        inserted = c.execute("SELECT changes()").fetchone()[0] == 1
    if not inserted:
        return fid, "Duplicate ID ignored"
    return fid, None

def get_feedback(filters=None):
    filters = filters or {}
    sql = "SELECT * FROM feedback WHERE 1=1"
    args = []
    for key, col in [("store_id","store_id"),("issue_type","issue_type"),("severity","severity"),("channel","channel"),("status","status")]:
        if filters.get(key):
            sql += f" AND {col}=?"; args.append(filters[key])
    if filters.get("q"):
        sql += " AND (original_text LIKE ? OR id LIKE ?)"
        args += [f"%{filters['q']}%", f"%{filters['q']}%"]
    if filters.get("start"):
        sql += " AND timestamp>=?"; args.append(filters["start"])
    if filters.get("end"):
        sql += " AND timestamp<?"; args.append(filters["end"])
    sql += " ORDER BY timestamp DESC LIMIT ?"
    args.append(min(int(filters.get("limit", 500)), 5000))
    with connect() as c:
        return [rowdict(r) for r in c.execute(sql, args).fetchall()]

def sync_first_signal():
    records = get_feedback({"limit": 5000})
    signals = detect_signals(records)
    with connect() as c:
        for sig in signals:
            if c.execute("SELECT 1 FROM incidents WHERE id=?", (sig["id"],)).fetchone():
                # Keep current workflow status; only refresh counts and timestamps/evidence.
                c.execute("""UPDATE incidents SET report_count=?,first_seen=?,latest_seen=?,updated_at=?
                             WHERE id=? AND status='Detected — awaiting review'""",
                          (sig["report_count"],sig["first_seen"],sig["latest_seen"],now_iso(),sig["id"]))
            else:
                c.execute("""INSERT INTO incidents
                (id,store_id,issue_type,summary,why_raised,report_count,first_seen,latest_seen,severity,department,status,assigned_user,priority,due_at,next_action,created_at,updated_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (sig["id"],sig["store_id"],sig["issue_type"],sig["summary"],sig["why_raised"],
                 sig["report_count"],sig["first_seen"],sig["latest_seen"],sig["severity"],sig["department"],
                 "Detected — awaiting review",None,sig["severity"],sig["due_at"],sig["next_action"],now_iso(),now_iso()))
            for fid in sig["feedback_ids"]:
                c.execute("INSERT OR IGNORE INTO incident_evidence VALUES(?,?)", (sig["id"], fid))
        c.commit()

@app.get("/")
def index():
    return render_template("index.html")

@app.get("/api/health")
def health():
    with connect() as c:
        count = c.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        incidents = c.execute("SELECT COUNT(*) FROM incidents").fetchone()[0]
    return jsonify({"status":"ok","feedback_count":count,"incident_count":incidents})

@app.get("/api/overview")
def overview():
    sync_first_signal()
    with connect() as c:
        total = c.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        unresolved = c.execute("SELECT COUNT(*) FROM incidents WHERE status NOT IN ('Resolved','Not substantiated')").fetchone()[0]
        high = c.execute("SELECT COUNT(*) FROM incidents WHERE severity='High' AND status NOT IN ('Resolved','Not substantiated')").fetchone()[0]
        recent = [rowdict(r) for r in c.execute("SELECT * FROM incidents WHERE status NOT IN ('Resolved','Not substantiated') ORDER BY CASE severity WHEN 'High' THEN 0 WHEN 'Medium' THEN 1 ELSE 2 END, latest_seen DESC LIMIT 6")]
    return jsonify({"feedback_count":total,"unresolved_count":unresolved,"high_priority_count":high,"incidents":recent})

@app.get("/api/feedback")
def feedback_api():
    return jsonify(get_feedback(request.args))

@app.post("/api/feedback")
def add_feedback_api():
    data = request.get_json(silent=True) or {}
    fid, error = create_feedback(data)
    if error and error != "Duplicate ID ignored":
        return jsonify({"error":error}), 400
    sync_first_signal()
    return jsonify({"id":fid,"message":error or "Feedback ingested"}), 201 if not error else 200

@app.post("/api/feedback/import")
def import_feedback():
    if "file" not in request.files:
        return jsonify({"error":"Choose a CSV or JSON file"}), 400
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
            return jsonify({"error":"Only CSV and JSON imports are supported"}), 400
    except Exception as exc:
        return jsonify({"error":f"Could not parse file: {exc}"}), 400
    inserted, skipped, errors = 0, 0, []
    for i, record in enumerate(records, start=1):
        fid, err = create_feedback(record)
        if err == "Duplicate ID ignored": skipped += 1
        elif err:
            errors.append({"row":i,"error":err})
        else: inserted += 1
    sync_first_signal()
    return jsonify({"inserted":inserted,"duplicates":skipped,"invalid":len(errors),"errors":errors[:30]})

@app.get("/api/incidents")
def incidents_api():
    sync_first_signal()
    with connect() as c:
        rows = c.execute("SELECT * FROM incidents ORDER BY CASE severity WHEN 'High' THEN 0 WHEN 'Medium' THEN 1 ELSE 2 END, updated_at DESC").fetchall()
        result = []
        for row in rows:
            item = rowdict(row)
            ev = c.execute("""SELECT f.id,f.timestamp,f.channel,f.original_text,f.issue_type,f.severity
                              FROM feedback f JOIN incident_evidence e ON e.feedback_id=f.id
                              WHERE e.incident_id=? ORDER BY f.timestamp""",(item["id"],)).fetchall()
            item["evidence"] = [rowdict(x) for x in ev]
            item["evidence_ids"] = [x["id"] for x in ev]
            item["audit"] = [rowdict(x) for x in c.execute("SELECT * FROM audit_log WHERE incident_id=? ORDER BY timestamp DESC",(item["id"],))]
            result.append(item)
    status = request.args.get("status")
    if status: result = [r for r in result if r["status"] == status]
    return jsonify(result)

@app.post("/api/incidents/<incident_id>/view")
def track_incident_view(incident_id):
    data = request.get_json(silent=True) or {}
    with connect() as c:
        if not c.execute("SELECT 1 FROM incidents WHERE id=?", (incident_id,)).fetchone():
            return jsonify({"error": "Incident not found"}), 404
        c.execute("INSERT INTO audit_log(incident_id,actor,action,old_value,new_value,note,timestamp) VALUES(?,?,?,?,?,?,?)",
                  (incident_id, str(data.get("actor") or "Reviewer"), "problem_statement_viewed", None, "opened",
                   "User opened the full problem statement from the View option", now_iso()))
        c.commit()
    return jsonify({"message": "Problem statement view tracked"})

@app.patch("/api/incidents/<incident_id>")
def update_incident(incident_id):
    data = request.get_json(silent=True) or {}
    allowed = {"status","assigned_user","department","severity","priority","due_at","next_action","resolution","root_cause"}
    changes = {k:v for k,v in data.items() if k in allowed}
    if not changes:
        return jsonify({"error":"No supported fields supplied"}),400
    with connect() as c:
        old = c.execute("SELECT * FROM incidents WHERE id=?",(incident_id,)).fetchone()
        if not old: return jsonify({"error":"Incident not found"}),404
        old = rowdict(old)
        for key, value in changes.items():
            c.execute(f"UPDATE incidents SET {key}=?,updated_at=? WHERE id=?",(value,now_iso(),incident_id))
            c.execute("INSERT INTO audit_log(incident_id,actor,action,old_value,new_value,note,timestamp) VALUES(?,?,?,?,?,?,?)",
                      (incident_id,str(data.get("actor") or "Reviewer"),key,str(old.get(key)),str(value),str(data.get("note") or ""),now_iso()))
        current = rowdict(c.execute("SELECT * FROM incidents WHERE id=?",(incident_id,)).fetchone())
        if current["status"] == "Resolved" and (current.get("resolution") or current.get("root_cause")):
            c.execute("""INSERT INTO resolution_memory(incident_id,issue_type,store_id,root_cause,actions,outcome,recorded_at)
                         VALUES(?,?,?,?,?,?,?)""",(incident_id,current["issue_type"],current["store_id"],current.get("root_cause"),
                         current.get("resolution") or "No action details recorded",current["status"],now_iso()))
        c.commit()
    return jsonify({"message":"Incident updated","incident":current})

@app.post("/api/incidents/<incident_id>/evidence")
def attach_evidence(incident_id):
    data = request.get_json(silent=True) or {}
    feedback_id = str(data.get("feedback_id") or "")
    with connect() as c:
        if not c.execute("SELECT 1 FROM incidents WHERE id=?",(incident_id,)).fetchone(): return jsonify({"error":"Incident not found"}),404
        if not c.execute("SELECT 1 FROM feedback WHERE id=?",(feedback_id,)).fetchone(): return jsonify({"error":"Feedback not found"}),404
        c.execute("INSERT OR IGNORE INTO incident_evidence VALUES(?,?)",(incident_id,feedback_id))
        c.execute("INSERT INTO audit_log(incident_id,actor,action,new_value,note,timestamp) VALUES(?,?,?,?,?,?)",
                  (incident_id,str(data.get("actor") or "Reviewer"),"evidence_linked",feedback_id,str(data.get("note") or ""),now_iso()))
    return jsonify({"message":"Evidence linked; original record preserved"})

@app.get("/api/reports/weekly")
def weekly_report():
    days = int(request.args.get("days",7))
    days = min(max(days,1),31)
    report = build_weekly_report(get_feedback({"limit":5000}), days=days, owners=ISSUE_OWNERS)
    return jsonify(report)

@app.get("/api/reports/weekly.csv")
def weekly_csv():
    report = build_weekly_report(get_feedback({"limit":5000}), days=7, owners=ISSUE_OWNERS)
    out = io.BytesIO(report_csv(report).encode("utf-8-sig"))
    return send_file(out, mimetype="text/csv", as_attachment=True, download_name="r5_weekly_brief.csv")

@app.get("/api/resolution-memory")
def resolution_memory():
    issue = request.args.get("issue_type")
    with connect() as c:
        sql = "SELECT * FROM resolution_memory"
        args = []
        if issue: sql += " WHERE issue_type=?"; args.append(issue)
        sql += " ORDER BY recorded_at DESC LIMIT 50"
        return jsonify([rowdict(r) for r in c.execute(sql,args).fetchall()])

@app.errorhandler(404)
def not_found(_):
    if request.path.startswith("/api/"): return jsonify({"error":"API endpoint not found"}),404
    return "Page not found",404

def seed_demo_data_if_empty():
    """Load clearly labeled fictional demo records on first local launch only."""
    with connect() as c:
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
    print("R5 Voice-of-Store Engine running at http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT",5000)), debug=False)
