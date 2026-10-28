"""SchemeSeva — Simple Data Audit, Verification, and Preprocessing Pipeline.

This script implements the initial data-preparation stage for the SchemeSeva
government scheme recommendation project:
  Step 1: Load and standardize raw scheme columns.
  Step 2: Audit records for duplicates, missing values, and data-quality issues.
  Step 3: Manage verification evidence and human review decisions.
  Step 4: Preprocess verified records (cleaning text, standardizing levels).
  Step 5: Export partitioned datasets and reports.

Designed for clarity and straightforward rewriting in Google Colab.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import urlparse
import pandas as pd

# Default project paths
RAW_DATA_PATH = Path("data/raw/gov_myscheme_data.csv")
PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")

AUDIT_SUMMARY_PATH = REPORTS_DIR / "audit_summary.csv"
RECORD_ISSUES_PATH = REPORTS_DIR / "record_issues.csv"
SCHEME_REVIEW_PATH = REPORTS_DIR / "scheme_review.csv"

VERIFIED_PATH = PROCESSED_DIR / "verified_schemes.csv"
PENDING_PATH = PROCESSED_DIR / "pending_schemes.csv"
INACTIVE_PATH = PROCESSED_DIR / "inactive_schemes.csv"
CLEAN_VERIFIED_PATH = PROCESSED_DIR / "verified_schemes_clean.csv"

# Recognized government domains in India
GOV_DOMAINS = (".gov.in", ".nic.in", ".gov")


# ==============================================================================
# Helper & Utility Functions
# ==============================================================================

def ensure_directories() -> None:
    """Create project directories if they do not exist."""
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(val: object) -> str:
    """Normalize whitespace without removing words, punctuation, or negations."""
    if pd.isna(val) or val is None:
        return ""
    text = re.sub(r"[\r\n\t]+", " ", str(val))
    return re.sub(r"\s+", " ", text).strip()


def is_valid_url(url: str) -> bool:
    """Return True if url has a valid HTTP/HTTPS scheme and network location."""
    if not url or not isinstance(url, str):
        return False
    p = urlparse(url.strip())
    return p.scheme in ("http", "https") and bool(p.netloc)


def is_gov_domain(url: str) -> bool:
    """Return True if URL host points to an official government domain."""
    if not is_valid_url(url):
        return False
    host = urlparse(url.strip()).netloc.lower().split(":")[0]
    return any(host.endswith(d) for d in GOV_DOMAINS) or "myscheme.gov.in" in host


def check_url_online(url: str, timeout: int = 5) -> tuple[bool, str]:
    """Optional basic HTTP response check for a source URL."""
    import requests
    try:
        r = requests.head(url, timeout=timeout, headers={"User-Agent": "SchemeSeva/1.0"}, allow_redirects=True)
        if r.status_code < 400:
            return True, f"HTTP {r.status_code}"
        if r.status_code in (403, 405):
            r = requests.get(url, timeout=timeout, headers={"User-Agent": "SchemeSeva/1.0"}, stream=True)
            return r.status_code < 400, f"HTTP {r.status_code}"
        return False, f"HTTP {r.status_code}"
    except Exception as exc:
        return False, f"Error: {type(exc).__name__}"


def standardize_level(val: str) -> str:
    """Map common government level variations to standard labels."""
    s = clean_text(val).lower()
    if s in ("central", "central government"):
        return "Central Government"
    if s in ("state", "state government", "state / ut government", "state/ut", "state / ut"):
        return "State / UT Government"
    return clean_text(val)


# ==============================================================================
# Step 1: Load and Standardize Dataset
# ==============================================================================

def load_dataset(filepath: Path | str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV file and map columns to a standard schema.
    
    Standard fields:
      - record_id, scheme_name, description, benefits, eligibility,
        application_process, department, level, source_url
    """
    path = Path(filepath)
    if not path.exists():
        ensure_directories()
        raise FileNotFoundError(
            f"Raw dataset not found at '{path}'.\n"
            "Please place 'gov_myscheme_data.csv' from https://github.com/Aryan-Pardeshi/gov-myscheme-dataset "
            f"into '{path.parent}' and rerun."
        )

    df = pd.read_csv(path, low_memory=False, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]

    field_map = {
        "scheme_name": ["Scheme Name", "scheme_name", "title", "name"],
        "description": ["Description", "description", "details", "about"],
        "benefits": ["Benefits", "benefits", "benefit"],
        "eligibility": ["Eligibility Criteria", "Eligibility (General)", "eligibility"],
        "application_process": ["Application Process", "application_process", "how_to_apply"],
        "department": ["State / UT / Ministry", "department", "ministry"],
        "level": ["Level", "level", "government_level"],
        "source_url": ["Official Link", "MyScheme URL", "source_url", "url", "link"],
    }

    lower_cols = {c.lower(): c for c in df.columns}
    resolved = {}
    for std_col, candidates in field_map.items():
        for cand in candidates:
            if cand.lower() in lower_cols:
                resolved[std_col] = lower_cols[cand.lower()]
                break

    if "scheme_name" not in resolved:
        raise ValueError(f"Dataset missing required scheme name column. Available columns: {list(df.columns)}")

    out = pd.DataFrame()
    out["record_id"] = (
        df[lower_cols["record_id"]].astype(str)
        if "record_id" in lower_cols
        else [f"SCH_{i + 1:04d}" for i in range(len(df))]
    )

    for std_col in [
        "scheme_name", "description", "benefits", "eligibility",
        "application_process", "department", "level", "source_url"
    ]:
        out[std_col] = (
            df[resolved[std_col]].fillna("").astype(str).map(clean_text)
            if std_col in resolved
            else ""
        )

    # Use MyScheme URL if primary official link is empty
    if "MyScheme URL" in df.columns:
        empty_url = out["source_url"].eq("")
        out.loc[empty_url, "source_url"] = (
            df.loc[empty_url, "MyScheme URL"].fillna("").astype(str).map(clean_text)
        )

    # Retain raw columns to prevent data loss
    for col in df.columns:
        if col not in out.columns:
            out[f"raw_{col}"] = df[col]
    return out


# ==============================================================================
# Step 2: Audit Dataset
# ==============================================================================

def audit_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Perform checks on records and generate summary and row-level issues."""
    issues: list[dict[str, str]] = []

    # 1. Exact duplicates across core scheme attributes
    dup_rows = df.duplicated(subset=["scheme_name", "description", "benefits"], keep=False)
    for _, r in df[dup_rows].iterrows():
        issues.append({
            "record_id": r["record_id"],
            "scheme_name": r["scheme_name"],
            "issue_type": "EXACT_DUPLICATE_ROW",
            "issue_detail": "Exact duplicate row found.",
        })

    # 2. Duplicate scheme name candidates (normalized text comparison)
    norm_names = df["scheme_name"].str.lower().str.replace(r"[^\w\s]", "", regex=True).str.strip()
    name_counts = norm_names.value_counts()
    dup_name_set = set(name_counts[name_counts > 1].index) - {""}

    for _, r in df.iterrows():
        rec_id, s_name, desc = r["record_id"], r["scheme_name"], r["description"]
        url, elig, bens = r["source_url"], r["eligibility"], r["benefits"]
        n_name = re.sub(r"[^\w\s]", "", clean_text(s_name).lower()).strip()

        if n_name in dup_name_set:
            issues.append({
                "record_id": rec_id,
                "scheme_name": s_name,
                "issue_type": "DUPLICATE_NAME_CANDIDATE",
                "issue_detail": f"Matches {name_counts[n_name]} candidate schemes.",
            })
        if not s_name:
            issues.append({"record_id": rec_id, "scheme_name": "(empty)", "issue_type": "MISSING_SCHEME_NAME", "issue_detail": "Scheme name is empty."})
        if not desc:
            issues.append({"record_id": rec_id, "scheme_name": s_name, "issue_type": "MISSING_DESCRIPTION", "issue_detail": "Description is empty."})
        elif len(desc) < 30:
            issues.append({"record_id": rec_id, "scheme_name": s_name, "issue_type": "SHORT_DESCRIPTION", "issue_detail": f"Description is only {len(desc)} characters."})
        if not url:
            issues.append({"record_id": rec_id, "scheme_name": s_name, "issue_type": "MISSING_SOURCE_URL", "issue_detail": "Source URL is missing."})
        elif not is_valid_url(url):
            issues.append({"record_id": rec_id, "scheme_name": s_name, "issue_type": "MALFORMED_SOURCE_URL", "issue_detail": f"Malformed URL: {url}"})
        if not elig:
            issues.append({"record_id": rec_id, "scheme_name": s_name, "issue_type": "MISSING_ELIGIBILITY", "issue_detail": "Eligibility text is empty."})
        if not bens:
            issues.append({"record_id": rec_id, "scheme_name": s_name, "issue_type": "MISSING_BENEFITS", "issue_detail": "Benefits text is empty."})
        if (not desc or len(desc) < 30) and (not elig or not bens):
            issues.append({
                "record_id": rec_id,
                "scheme_name": s_name,
                "issue_type": "INSUFFICIENT_INFORMATION",
                "issue_detail": "Record lacks sufficient information for recommendations.",
            })

    issues_df = (
        pd.DataFrame(issues)
        if issues
        else pd.DataFrame(columns=["record_id", "scheme_name", "issue_type", "issue_detail"])
    )

    summary_rows = [
        ("total_records", len(df)),
        ("total_columns", len(df.columns)),
        ("exact_duplicate_rows", int(dup_rows.sum())),
        ("potential_duplicate_name_records", int(norm_names.isin(dup_name_set).sum())),
        ("missing_scheme_name_count", int(df["scheme_name"].eq("").sum())),
        ("missing_description_count", int(df["description"].eq("").sum())),
        ("short_description_count (<30 chars)", int(df["description"].map(lambda x: 0 < len(x) < 30).sum())),
        ("missing_eligibility_count", int(df["eligibility"].eq("").sum())),
        ("missing_benefits_count", int(df["benefits"].eq("").sum())),
        ("missing_source_url_count", int(df["source_url"].eq("").sum())),
        ("malformed_source_url_count", int(df["source_url"].map(lambda u: bool(u) and not is_valid_url(u)).sum())),
        ("official_gov_domain_url_count", int(df["source_url"].map(is_gov_domain).sum())),
        ("total_issues_flagged", len(issues_df)),
    ]
    summary_df = pd.DataFrame(summary_rows, columns=["metric", "value"])
    return summary_df, issues_df


# ==============================================================================
# Step 3: Verification Workflow & Review Decisions
# ==============================================================================

def build_or_update_review(
    df: pd.DataFrame,
    review_path: Path | str = SCHEME_REVIEW_PATH,
    check_urls: bool = False,
) -> pd.DataFrame:
    """Create or update reports/scheme_review.csv preserving human decisions."""
    path = Path(review_path)
    saved_decisions: dict[str, dict[str, str]] = {}
    if path.exists():
        existing = pd.read_csv(path, dtype=str).fillna("")
        for _, r in existing.iterrows():
            rid = r.get("record_id", "").strip()
            if rid:
                saved_decisions[rid] = {
                    "verification_status": r.get("verification_status", "").strip(),
                    "reason": r.get("reason", "").strip(),
                    "official_evidence_url": r.get("official_evidence_url", "").strip(),
                    "reviewer_notes": r.get("reviewer_notes", "").strip(),
                }

    rows: list[dict[str, str]] = []
    for _, r in df.iterrows():
        rec_id, s_name, url = r["record_id"], r["scheme_name"], r["source_url"]
        if rec_id in saved_decisions and saved_decisions[rec_id]["verification_status"]:
            s = saved_decisions[rec_id]
            rows.append({
                "record_id": rec_id,
                "scheme_name": s_name,
                "original_source_url": url,
                "verification_status": s["verification_status"],
                "reason": s["reason"],
                "official_evidence_url": s["official_evidence_url"],
                "reviewer_notes": s["reviewer_notes"],
            })
            continue

        if not url:
            status, reason = "NEEDS_REVIEW", "Missing source URL; manual review required."
        elif not is_valid_url(url):
            status, reason = "NEEDS_REVIEW", f"Malformed source URL: '{url}'."
        elif not is_gov_domain(url):
            status, reason = "NEEDS_REVIEW", "Source domain is non-official; requires official verification."
        else:
            status, reason = "UNVERIFIED", "Official government domain identified; awaiting human reviewer confirmation."

        if check_urls and is_valid_url(url):
            _, msg = check_url_online(url)
            reason += f" [Check: {msg}]"

        rows.append({
            "record_id": rec_id,
            "scheme_name": s_name,
            "original_source_url": url,
            "verification_status": status,
            "reason": reason,
            "official_evidence_url": "",
            "reviewer_notes": "",
        })

    review_df = pd.DataFrame(rows)
    review_df.to_csv(path, index=False, encoding="utf-8-sig")
    return review_df


def apply_verification_decisions(
    df: pd.DataFrame,
    review_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Partition records based on review status into verified, pending, and inactive."""
    merged = df.merge(
        review_df[["record_id", "verification_status", "official_evidence_url", "reviewer_notes"]],
        on="record_id",
        how="left",
    )
    merged["verification_status"] = merged["verification_status"].fillna("UNVERIFIED").astype(str).str.strip()

    verified_mask = merged["verification_status"].eq("VERIFIED")
    inactive_mask = merged["verification_status"].eq("INACTIVE")
    pending_mask = ~verified_mask & ~inactive_mask

    return merged[verified_mask].copy(), merged[pending_mask].copy(), merged[inactive_mask].copy()


# ==============================================================================
# Step 4: Preprocessing Verified Records
# ==============================================================================

def preprocess_verified(verified_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Clean text, standardize government level, and create recommendation search_text."""
    cols = list(verified_df.columns)
    if "search_text" not in cols:
        cols.append("search_text")
    if verified_df.empty:
        return pd.DataFrame(columns=cols), []

    clean_df = verified_df.copy()
    for col in ["scheme_name", "description", "benefits", "eligibility", "application_process", "department"]:
        if col in clean_df.columns:
            clean_df[col] = clean_df[col].map(clean_text)

    clean_df["level"] = clean_df["level"].map(standardize_level)

    search_parts: list[str] = []
    excluded: list[str] = []
    for _, r in clean_df.iterrows():
        parts = [r["scheme_name"], r["description"], r["benefits"], r["eligibility"], r["application_process"]]
        joined = " ".join([p for p in parts if p]).strip()
        search_parts.append(joined)

    clean_df["search_text"] = search_parts
    usable = clean_df["search_text"].str.len() >= 20
    for _, r in clean_df[~usable].iterrows():
        excluded.append(f"Excluded {r['record_id']} ({r['scheme_name']}): search_text under 20 chars.")

    return clean_df[usable].reset_index(drop=True), excluded


# ==============================================================================
# Step 5: Export & Orchestration
# ==============================================================================

def run_pipeline(
    raw_path: Path | str = RAW_DATA_PATH,
    check_urls: bool = False,
) -> dict[str, int]:
    """Execute the complete audit -> verification -> preprocessing pipeline."""
    ensure_directories()
    raw_file = Path(raw_path)

    print(f"[Step 1/5] Loading dataset: {raw_file}")
    df = load_dataset(raw_file)
    print(f"  Loaded {len(df)} records across {len(df.columns)} columns.")

    print("[Step 2/5] Auditing records...")
    summary_df, issues_df = audit_dataset(df)
    summary_df.to_csv(AUDIT_SUMMARY_PATH, index=False)
    issues_df.to_csv(RECORD_ISSUES_PATH, index=False)
    print(f"  Saved audit summary ({len(summary_df)} metrics) and issues ({len(issues_df)} flagged).")

    print("[Step 3/5] Updating verification review sheet...")
    review_df = build_or_update_review(df, SCHEME_REVIEW_PATH, check_urls=check_urls)
    verified_df, pending_df, inactive_df = apply_verification_decisions(df, review_df)
    verified_df.to_csv(VERIFIED_PATH, index=False)
    pending_df.to_csv(PENDING_PATH, index=False)
    inactive_df.to_csv(INACTIVE_PATH, index=False)
    print(f"  Decisions: {len(verified_df)} verified, {len(pending_df)} pending, {len(inactive_df)} inactive.")

    print("[Step 4/5] Preprocessing verified records...")
    clean_verified_df, excluded = preprocess_verified(verified_df)
    clean_verified_df.to_csv(CLEAN_VERIFIED_PATH, index=False)
    print(f"  Prepared {len(clean_verified_df)} clean verified records for ML.")
    if excluded:
        print(f"  Notice: {len(excluded)} verified records excluded due to short text.")

    print("[Step 5/5] Pipeline execution finished.")
    return {
        "raw_records": len(df),
        "issues_flagged": len(issues_df),
        "verified_records": len(verified_df),
        "pending_records": len(pending_df),
        "inactive_records": len(inactive_df),
        "clean_verified_records": len(clean_verified_df),
    }


def main() -> int:
    """CLI entrypoint."""
    check_urls = "--check-urls" in sys.argv
    custom_input = None
    if "--input" in sys.argv:
        idx = sys.argv.index("--input")
        if idx + 1 < len(sys.argv):
            custom_input = sys.argv[idx + 1]

    target_path = Path(custom_input) if custom_input else RAW_DATA_PATH

    try:
        counts = run_pipeline(raw_path=target_path, check_urls=check_urls)
        print("\nExecution Summary:")
        for k, v in counts.items():
            print(f"  {k}: {v}")
        return 0
    except FileNotFoundError as err:
        print(f"\n[ERROR] {err}")
        return 1
    except Exception as err:
        print(f"\n[UNEXPECTED ERROR] {err}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
