"""Voice of Store — System Evaluation Tool.

Evaluates:
1. Issue Type Classification (Accuracy, Macro-F1, Weighted-F1, per-class breakdown)
2. Severity Level Classification (Accuracy, Macro-F1)
3. Actionable Prediction (Accuracy, Precision, Recall, F1)
4. Escalation Precision & Recall (TP, FP, FN, TN, Precision)
5. Planted Incident Lead-Time (Reporting store, actual start date, first detection, lead-time days)
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

from config.taxonomy import TAXONOMY, get_all_issue_types, get_department
from services.classifier import classify_feedback, classify_batch, is_mandatory_escalation

DEFAULT_SYNTHETIC_DATA = Path(__file__).resolve().parent / "data" / "eval_data_synthetic.csv"
DEFAULT_DEMO_DATA = Path(__file__).resolve().parent / "demo_data.csv"

def parse_dt(val: Any) -> Optional[datetime]:
    if not val:
        return None
    try:
        d = datetime.fromisoformat(str(val).replace("Z", "+00:00"))
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except Exception:
        return None

def load_data(filepath: Path | str) -> Tuple[List[Dict[str, Any]], str]:
    """Load evaluation records from CSV file with column auto-discovery."""
    p = Path(filepath)
    if not p.exists():
        raise FileNotFoundError(f"Evaluation file not found: {p}")

    records = []
    with open(p, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)

    if not records:
        raise ValueError(f"Evaluation file is empty: {p}")

    # Check if dataset is synthetic
    is_synth = any(
        "synthetic" in str(r.get("data_source", "")).lower() or
        "fictional" in str(r.get("demo_record", "")).lower() or
        "synthetic" in p.name.lower()
        for r in records
    )
    dataset_type = "SYNTHETIC / DEVELOPMENT DATA" if is_synth else "OFFICIAL LABELED DATASET"
    return records, dataset_type

def evaluate_dataset(records: List[Dict[str, Any]], seed: int = 42) -> Dict[str, Any]:
    """Run full evaluation pipeline over records."""
    np.random.seed(seed)
    
    # Extract texts and ground truths
    texts = []
    gt_issue_types = []
    gt_severities = []
    gt_actionables = []
    gt_escalations = []
    has_gt_issue = False
    has_gt_sev = False
    has_gt_act = False
    has_gt_esc = False
    
    for r in records:
        text = str(r.get("text") or r.get("original_text") or r.get("feedback") or r.get("comment") or "").strip()
        texts.append(text)
        
        # Ground truth labels
        it = r.get("issue_type")
        if it:
            has_gt_issue = True
            gt_issue_types.append(str(it).strip())
        else:
            gt_issue_types.append(None)
            
        sev = r.get("severity")
        if sev:
            has_gt_sev = True
            gt_severities.append(str(sev).strip().title())
        else:
            gt_severities.append(None)
            
        act = r.get("actionable")
        if act is not None and str(act).strip() != "":
            has_gt_act = True
            gt_actionables.append(1 if str(act).lower() in {"1", "true", "yes"} else 0)
        else:
            gt_actionables.append(None)
            
        esc = r.get("ground_truth_escalated") or r.get("escalated") or r.get("is_escalated")
        if esc is not None and str(esc).strip() != "":
            has_gt_esc = True
            gt_escalations.append(1 if str(esc).lower() in {"1", "true", "yes"} else 0)
        else:
            gt_escalations.append(None)

    # Run predictions
    predictions = classify_batch(texts)
    
    results: Dict[str, Any] = {
        "record_count": len(records),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "random_seed": seed,
        "metrics": {},
        "escalation_metrics": {},
        "planted_incidents": [],
    }

    # 1. Issue Type Classification Metrics
    if has_gt_issue and any(x is not None for x in gt_issue_types):
        valid_indices = [i for i, y in enumerate(gt_issue_types) if y is not None]
        y_true = [gt_issue_types[i] for i in valid_indices]
        y_pred = [predictions[i]["issue_type"] for i in valid_indices]
        
        acc = float(accuracy_score(y_true, y_pred))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        
        # Per-class summary
        unique_classes = sorted(list(set(y_true + y_pred)))
        per_class = {}
        for cls in unique_classes:
            cls_true = [1 if y == cls else 0 for y in y_true]
            cls_pred = [1 if y == cls else 0 for y in y_pred]
            support = sum(cls_true)
            if support > 0:
                p = float(precision_score(cls_true, cls_pred, zero_division=0))
                r = float(recall_score(cls_true, cls_pred, zero_division=0))
                f = float(f1_score(cls_true, cls_pred, zero_division=0))
                per_class[cls] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4), "support": support}
                
        results["metrics"]["issue_type"] = {
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "evaluated_samples": len(valid_indices),
            "per_class": per_class
        }
    else:
        results["metrics"]["issue_type"] = "N/A (ground truth issue_type labels not provided)"

    # 2. Severity Classification Metrics
    if has_gt_sev and any(x is not None for x in gt_severities):
        valid_indices = [i for i, y in enumerate(gt_severities) if y is not None]
        y_true = [gt_severities[i] for i in valid_indices]
        y_pred = [predictions[i]["severity"] for i in valid_indices]
        
        acc = float(accuracy_score(y_true, y_pred))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        
        results["metrics"]["severity"] = {
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "evaluated_samples": len(valid_indices)
        }
    else:
        results["metrics"]["severity"] = "N/A (ground truth severity labels not provided)"

    # 3. Actionable Prediction Metrics
    if has_gt_act and any(x is not None for x in gt_actionables):
        valid_indices = [i for i, y in enumerate(gt_actionables) if y is not None]
        y_true = [gt_actionables[i] for i in valid_indices]
        y_pred = [1 if predictions[i]["actionable"] else 0 for i in valid_indices]
        
        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        
        results["metrics"]["actionable"] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "evaluated_samples": len(valid_indices)
        }
    else:
        results["metrics"]["actionable"] = "N/A (ground truth actionable labels not provided)"

    # 4. Escalation Precision
    if has_gt_esc and any(x is not None for x in gt_escalations):
        valid_indices = [i for i, y in enumerate(gt_escalations) if y is not None]
        y_true = [gt_escalations[i] for i in valid_indices]
        # System decides escalation if mandatory safety trigger or critical severity
        y_pred = []
        for i in valid_indices:
            pred_it = predictions[i]["issue_type"]
            txt = texts[i]
            is_esc = is_mandatory_escalation(pred_it, txt) or predictions[i]["severity"] == "Critical"
            y_pred.append(1 if is_esc else 0)
            
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
        tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
        
        prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        
        results["escalation_metrics"] = {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "total_escalations_flagged": tp + fp,
            "ground_truth_escalations": tp + fn,
            "honest_assessment": (
                "100% precision attained on evaluated subset" if prec == 1.0 and fp == 0
                else f"Precision is {prec*100:.1f}% ({fp} false positive escalations observed)"
            )
        }
    else:
        results["escalation_metrics"] = "N/A (ground truth escalation labels not provided in dataset)"

    # 5. Planted Incident Lead-Time Analysis
    planted_cases = []
    for i, r in enumerate(records):
        is_planted = str(r.get("is_planted") or "").lower() in {"1", "true", "yes"} or "planted" in str(r.get("id", "")).lower()
        actual_start_str = r.get("actual_incident_start")
        
        if is_planted or actual_start_str:
            store_id = str(r.get("store_id") or "UNKNOWN").upper()
            detection_ts = parse_dt(r.get("timestamp"))
            actual_start = parse_dt(actual_start_str)
            
            lead_days = None
            if detection_ts and actual_start:
                diff_seconds = (detection_ts - actual_start).total_seconds()
                lead_days = round(diff_seconds / 86400.0, 2)
                
            planted_cases.append({
                "record_id": r.get("id", f"REC-{i+1}"),
                "reporting_store": store_id,
                "predicted_issue": predictions[i]["issue_type"],
                "predicted_severity": predictions[i]["severity"],
                "actual_start_date": actual_start.isoformat() if actual_start else "Unknown",
                "first_detection_date": detection_ts.isoformat() if detection_ts else "Unknown",
                "lead_time_days": lead_days,
            })
            
    if planted_cases:
        lead_times = [c["lead_time_days"] for c in planted_cases if c["lead_time_days"] is not None]
        avg_lead = round(float(np.mean(lead_times)), 2) if lead_times else None
        results["planted_incidents"] = {
            "total_planted_incidents": len(planted_cases),
            "mean_lead_time_days": avg_lead,
            "incidents": planted_cases[:10]  # Show top 10
        }
    else:
        results["planted_incidents"] = "N/A (planted incident start dates not present in dataset)"

    # 6. Combined Evaluation Score
    component_weights = {}
    if isinstance(results["metrics"].get("issue_type"), dict):
        component_weights["Issue Type Macro-F1"] = (results["metrics"]["issue_type"]["macro_f1"], 0.35)
    if isinstance(results["metrics"].get("severity"), dict):
        component_weights["Severity Macro-F1"] = (results["metrics"]["severity"]["macro_f1"], 0.20)
    if isinstance(results["metrics"].get("actionable"), dict):
        component_weights["Actionable F1"] = (results["metrics"]["actionable"]["f1"], 0.15)
    if isinstance(results.get("escalation_metrics"), dict):
        component_weights["Escalation Precision"] = (results["escalation_metrics"]["precision"], 0.15)
    if isinstance(results.get("planted_incidents"), dict) and results["planted_incidents"].get("mean_lead_time_days") is not None:
        lead_norm = min(max(results["planted_incidents"]["mean_lead_time_days"] / 3.0, 0.0), 1.0)
        component_weights["Planted Incident Lead Time"] = (lead_norm, 0.15)

    if component_weights:
        total_weight = sum(w for _, w in component_weights.values())
        score = sum(val * w for val, w in component_weights.values()) / total_weight
        results["combined_score"] = {
            "score_out_of_100": round(score * 100.0, 2),
            "normalized_score": round(score, 4),
            "components": {k: {"value": round(v, 4), "weight": round(w / total_weight, 4)} for k, (v, w) in component_weights.items()}
        }
    else:
        results["combined_score"] = "N/A (insufficient ground-truth components to calculate combined score)"

    return results

def format_text_report(results: Dict[str, Any], dataset_path: str, dataset_type: str) -> str:
    """Format evaluation output as a professional terminal report."""
    lines = []
    lines.append("=" * 78)
    lines.append("  VOICE OF STORE — SYSTEM EVALUATION REPORT")
    lines.append("=" * 78)
    lines.append(f"  Dataset Path    : {dataset_path}")
    lines.append(f"  Dataset Type    : {dataset_type}")
    lines.append(f"  Evaluated Rows  : {results['record_count']}")
    lines.append(f"  Timestamp (UTC) : {results['evaluated_at']}")
    lines.append("-" * 78)
    
    # 1. Classification Metrics
    lines.append("\n[1] CLASSIFICATION PERFORMANCE")
    lines.append("." * 78)
    m = results["metrics"]
    
    if isinstance(m.get("issue_type"), dict):
        it = m["issue_type"]
        lines.append(f"  Issue Type Accuracy : {it['accuracy'] * 100:.2f}%")
        lines.append(f"  Issue Type Macro F1 : {it['macro_f1']:.4f}")
        lines.append(f"  Issue Type W-F1     : {it['weighted_f1']:.4f}")
    else:
        lines.append(f"  Issue Type          : {m.get('issue_type')}")
        
    if isinstance(m.get("severity"), dict):
        sev = m["severity"]
        lines.append(f"  Severity Accuracy   : {sev['accuracy'] * 100:.2f}%")
        lines.append(f"  Severity Macro F1   : {sev['macro_f1']:.4f}")
    else:
        lines.append(f"  Severity            : {m.get('severity')}")

    if isinstance(m.get("actionable"), dict):
        act = m["actionable"]
        lines.append(f"  Actionable Accuracy : {act['accuracy'] * 100:.2f}%")
        lines.append(f"  Actionable Prec/Rec : {act['precision']:.4f} / {act['recall']:.4f}")
        lines.append(f"  Actionable F1       : {act['f1']:.4f}")
    else:
        lines.append(f"  Actionable          : {m.get('actionable')}")

    # 2. Escalation Precision
    lines.append("\n[2] ESCALATION PRECISION & SAFETY COMPLIANCE")
    lines.append("." * 78)
    esc = results.get("escalation_metrics")
    if isinstance(esc, dict):
        lines.append(f"  True Positives  (TP) : {esc['true_positives']}")
        lines.append(f"  False Positives (FP) : {esc['false_positives']}")
        lines.append(f"  False Negatives (FN) : {esc['false_negatives']}")
        lines.append(f"  True Negatives  (TN) : {esc['true_negatives']}")
        lines.append(f"  Escalation Precision : {esc['precision'] * 100:.2f}%")
        lines.append(f"  Escalation Recall    : {esc['recall'] * 100:.2f}%")
        lines.append(f"  Escalation F1-Score  : {esc['f1']:.4f}")
        lines.append(f"  Status Assessment    : {esc['honest_assessment']}")
    else:
        lines.append(f"  Escalation Metrics   : {esc}")

    # 3. Planted Incident Lead Times
    lines.append("\n[3] PLANTED INCIDENT DETECTION LEAD-TIME")
    lines.append("." * 78)
    pi = results.get("planted_incidents")
    if isinstance(pi, dict):
        lines.append(f"  Total Planted Incidents : {pi['total_planted_incidents']}")
        lines.append(f"  Mean Lead Time (Days)   : {pi['mean_lead_time_days']}")
        lines.append("\n  {:<10} {:<10} {:<24} {:<24} {:<10}".format(
            "Record", "Store", "Actual Start", "Detection Date", "Lead (Days)"
        ))
        lines.append("  " + "-" * 74)
        for inc in pi["incidents"][:6]:
            lines.append("  {:<10} {:<10} {:<24} {:<24} {:<10}".format(
                inc["record_id"][:10],
                inc["reporting_store"][:10],
                inc["actual_start_date"][:22],
                inc["first_detection_date"][:22],
                str(inc["lead_time_days"])
            ))
    else:
        lines.append(f"  Planted Incident Analysis : {pi}")

    # 4. Combined System Score
    lines.append("\n[4] COMBINED SYSTEM SCORE")
    lines.append("." * 78)
    cs = results.get("combined_score")
    if isinstance(cs, dict):
        lines.append(f"  Combined Evaluation Score : {cs['score_out_of_100']:.2f} / 100.00")
        lines.append(f"  Normalized Composite F1   : {cs['normalized_score']:.4f}")
        lines.append("  Component Breakdown:")
        for comp_name, comp_data in cs["components"].items():
            lines.append(f"    - {comp_name:<28}: {comp_data['value']:.4f} (effective weight: {comp_data['weight']*100:.1f}%)")
    else:
        lines.append(f"  Combined System Score     : {cs}")

    lines.append("\n" + "=" * 78)
    return "\n".join(lines)

def run_evaluation(data_path: Optional[str] = None, output_path: Optional[str] = None,
                   fmt: str = "text", seed: int = 42) -> Dict[str, Any]:
    """Programmatic entry point for running evaluation."""
    target_path = None
    if data_path:
        target_path = Path(data_path)
    elif DEFAULT_SYNTHETIC_DATA.exists():
        target_path = DEFAULT_SYNTHETIC_DATA
    elif DEFAULT_DEMO_DATA.exists():
        target_path = DEFAULT_DEMO_DATA
    else:
        raise FileNotFoundError("No evaluation dataset specified and default datasets not found.")

    records, dataset_type = load_data(target_path)
    results = evaluate_dataset(records, seed=seed)
    results["dataset_path"] = str(target_path)
    results["dataset_type"] = dataset_type

    if fmt == "json":
        output_str = json.dumps(results, indent=2)
        print(output_str)
    else:
        report_str = format_text_report(results, str(target_path), dataset_type)
        print(report_str)

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\n[INFO] Machine-readable evaluation results saved to: {out_p}")

    return results

def main():
    parser = argparse.ArgumentParser(description="Voice of Store Evaluation CLI")
    parser.add_argument("--data", type=str, default=None, help="Path to evaluation data CSV")
    parser.add_argument("--output", type=str, default=None, help="Path to write JSON evaluation metrics")
    parser.add_argument("--format", type=str, choices=["text", "json"], default="text", help="Output format")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    try:
        run_evaluation(
            data_path=args.data,
            output_path=args.output,
            fmt=args.format,
            seed=args.seed
        )
    except Exception as e:
        print(f"[ERROR] Evaluation failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
