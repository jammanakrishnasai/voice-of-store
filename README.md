# Voice of Store — R5 Operations Engine

A local Flask + SQLite prototype that turns store feedback into evidence-linked incident candidates and tracked corrective actions. The interface uses a clean purple theme and deliberately contains **no charts, pie charts, trend charts, or flowcharts**.

## Included
- 50 clearly fictional sample feedback records, auto-loaded into a fresh database.
- Overview of feedback, open incidents, and high-priority items.
- FIRST SIGNAL detection: repeated local issue clusters and high-risk single-report escalation.
- Original feedback preservation, linked incident evidence, and View-event audit history.
- Incident severity, owner/department, status, notes, and update audit trail.
- Sort FIRST SIGNAL and Action Tracker by High → Low, Low → High, most evidence, or most recent.
- Add feedback manually; import CSV or JSON.
- Weekly evidence brief and CSV export.
- Resolution memory for completed incidents.

## Run on Windows
1. Install Python 3.10+ (Python 3.13 is suitable; use a supported Python version if your environment has dependency issues).
2. Extract the ZIP. Open the folder that contains `app.py` in VS Code.
3. In the VS Code terminal run:
   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   python app.py
   ```
4. Open `http://127.0.0.1:5000`. Keep the terminal open while using the app.

To load sample data only on a fresh database, the default setting does this automatically. To disable demo seeding for a new database, set `R5_LOAD_DEMO=0`. The sample records are fictional and must not be presented as real customer/store data.

## Import data
Use **Add feedback → Import dataset** with CSV or JSON. Each record should include `store_id`, `timestamp`, `channel`, and one of `text`, `feedback`, or `comment`. Optional `id`, `issue_type`, and `severity` fields are accepted.

## Important challenge compliance note
This package implements a usable end-to-end prototype, but the official R5 problem statement, required taxonomy/store mappings, original dataset, three planted-incident labels, and `evaluate.py` were not present in the files available while this bundle was built. Therefore, exact official requirement coverage and evaluation score cannot be claimed. Before final submission, align the classifier, mandatory escalation rules, schema, and report output with those official files and run the official evaluator.
