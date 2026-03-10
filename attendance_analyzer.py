import os
import sys
import argparse
from datetime import datetime, timedelta, date
import pandas as pd
import yaml

# -----------------------------
# Utilities
# -----------------------------
TIME_FMT = "%H:%M"
DATE_FMT = "%Y-%m-%d"

def parse_time(t):
    if pd.isna(t) or t == "":
        return None
    return datetime.strptime(str(t).strip(), TIME_FMT).time()

def parse_date(d):
    return datetime.strptime(str(d).strip(), DATE_FMT).date()

def minutes_diff(t1, t2):
    """Return (t2 - t1) in minutes; t1,t2 are datetime.time"""
    dt1 = datetime.combine(date.today(), t1)
    dt2 = datetime.combine(date.today(), t2)
    return int((dt2 - dt1).total_seconds() // 60)

def ensure_folder(path):
    os.makedirs(path, exist_ok=True)

# -----------------------------
# Core Analysis
# -----------------------------
def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Normalize columns
    expected = {"employee_id","employee_name","date","check_in","check_out","status"}
    missing = expected - set(df.columns.str.lower())
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")
    # Standardize column names
    df.columns = [c.lower() for c in df.columns]
    df["date"] = df["date"].apply(parse_date)
    df["check_in"] = df["check_in"].apply(parse_time)
    df["check_out"] = df["check_out"].apply(parse_time)
    df["status"] = df["status"].astype(str).str.upper().str.strip()
    return df

def analyze(df: pd.DataFrame, cfg: dict, start_date=None, end_date=None) -> (pd.DataFrame, pd.DataFrame):
    # Filter by date range if provided
    if start_date:
        start_date = parse_date(start_date)
        df = df[df["date"] >= start_date]
    if end_date:
        end_date = parse_date(end_date)
        df = df[df["date"] <= end_date]

    ws = cfg["workday"]
    scoring = cfg["scoring"]

    sched_start = datetime.strptime(ws["start_time"], TIME_FMT).time()
    sched_end   = datetime.strptime(ws["end_time"], TIME_FMT).time()
    late_grace  = int(ws.get("late_grace_minutes", 0))
    early_grace = int(ws.get("early_grace_minutes", 0))
    min_hours   = float(ws.get("minimum_hours", 8))

    per_row = []
    for _, r in df.iterrows():
        status = r["status"]
        ci = r["check_in"]
        co = r["check_out"]

        late_minutes = None
        early_leave_minutes = None
        total_minutes = None

        is_absent = status == "ABSENT"
        is_present_like = status in {"PRESENT","WFH"}  # treat WFH as present for hours check

        if is_present_like and ci and co:
            total_minutes = minutes_diff(ci, co)

            # Late arrival
            if ci > sched_start:
                late_minutes = max(0, minutes_diff(sched_start, ci))
            else:
                late_minutes = 0

            # Early leave
            if co < sched_end:
                early_leave_minutes = max(0, minutes_diff(co, sched_end))
            else:
                early_leave_minutes = 0
        elif is_absent:
            late_minutes = None
            early_leave_minutes = None
            total_minutes = 0  # absent => 0
        else:
            # Missing punches: treat as present with unknown hours
            total_minutes = None

        # Infractions
        late_flag = is_present_like and late_minutes is not None and late_minutes > late_grace
        early_flag = is_present_like and early_leave_minutes is not None and early_leave_minutes > early_grace
        short_hours_flag = is_present_like and (total_minutes is not None) and (total_minutes < min_hours * 60)
        absent_flag = is_absent

        score = 0
        if late_flag:        score += scoring["late_weight"]
        if early_flag:       score += scoring["early_leave_weight"]
        if short_hours_flag: score += scoring["short_hours_weight"]
        if absent_flag:      score += scoring["absent_weight"]

        per_row.append({
            "employee_id": r["employee_id"],
            "employee_name": r["employee_name"],
            "date": r["date"],
            "status": status,
            "check_in": r["check_in"].strftime(TIME_FMT) if r["check_in"] else "",
            "check_out": r["check_out"].strftime(TIME_FMT) if r["check_out"] else "",
            "late_minutes": late_minutes if late_minutes is not None else "",
            "early_leave_minutes": early_leave_minutes if early_leave_minutes is not None else "",
            "total_minutes": total_minutes if total_minutes is not None else "",
            "late_flag": late_flag,
            "early_leave_flag": early_flag,
            "short_hours_flag": short_hours_flag,
            "absent_flag": absent_flag,
            "infraction_score": score
        })

    detailed = pd.DataFrame(per_row)

    # Summary per employee
    def safe_sum(series):
        # Handles empty/NaN gracefully
        return pd.to_numeric(series, errors="coerce").fillna(0).sum()

    summary = detailed.groupby(["employee_id","employee_name"], dropna=False).agg(
        days= ("date","count"),
        presents=("status", lambda s: (s.isin(["PRESENT","WFH"])).sum()),
        absences=("absent_flag","sum"),
        late_days=("late_flag","sum"),
        early_leave_days=("early_leave_flag","sum"),
        short_hours_days=("short_hours_flag","sum"),
        total_infraction_score=("infraction_score","sum"),
        total_late_minutes=("late_minutes", safe_sum),
        total_early_leave_minutes=("early_leave_minutes", safe_sum),
        total_work_minutes=("total_minutes", safe_sum),
    ).reset_index()

    # Overall totals row
    overall = pd.DataFrame([{
        "employee_id": "ALL",
        "employee_name": "All Employees",
        "days": summary["days"].sum(),
        "presents": summary["presents"].sum(),
        "absences": summary["absences"].sum(),
        "late_days": summary["late_days"].sum(),
        "early_leave_days": summary["early_leave_days"].sum(),
        "short_hours_days": summary["short_hours_days"].sum(),
        "total_infraction_score": summary["total_infraction_score"].sum(),
        "total_late_minutes": summary["total_late_minutes"].sum(),
        "total_early_leave_minutes": summary["total_early_leave_minutes"].sum(),
        "total_work_minutes": summary["total_work_minutes"].sum(),
    }])

    summary = pd.concat([summary, overall], ignore_index=True)
    return detailed, summary

def save_reports(detailed, summary, cfg, tag="latest"):
    out_dir = cfg["report"]["output_folder"]
    prefix  = cfg["report"]["output_prefix"]
    ensure_folder(out_dir)

    detailed_path = os.path.join(out_dir, f"{prefix}_{tag}_detailed.csv")
    summary_path  = os.path.join(out_dir, f"{prefix}_{tag}_summary.csv")

    detailed.to_csv(detailed_path, index=False)
    summary.to_csv(summary_path, index=False)
    return detailed_path, summary_path

# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Team Attendance Analyzer")
    parser.add_argument("--csv", default="data/attendance_sample.csv", help="Path to attendance CSV")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD (optional)")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD (optional)")
    parser.add_argument("--tag", default=None, help="Output filename tag (defaults to end date or 'latest')")
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = load_data(args.csv)

    detailed, summary = analyze(df, cfg, start_date=args.start, end_date=args.end)

    # Choose tag
    tag = args.tag or (args.end if args.end else "latest")
    d_path, s_path = save_reports(detailed, summary, cfg, tag=tag)

    print(f"✅ Detailed report saved to: {d_path}")
    print(f"✅ Summary report saved to:  {s_path}")

if __name__ == "__main__":
    main()
