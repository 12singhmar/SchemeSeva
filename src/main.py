import re
import sys
from pathlib import Path
from urllib.parse import urlparse
import pandas as pd

RAW_DATA_PATH = Path("data/raw/gov_myscheme_data.csv")
PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")

AUDIT_SUMMARY_PATH = REPORTS_DIR / "audit_summary.csv"
RECORD_ISSUES_PATH = REPORTS_DIR / "record_issues.csv"
CLEAN_SCHEMES_PATH = PROCESSED_DIR / "schemes_clean.csv"
SEARCH_SCHEMES_PATH = PROCESSED_DIR / "schemes_for_search.csv"


def ensure_directories():
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(val):
    if pd.isna(val) or val is None:
        return ""
    text = re.sub(r"[\r\n\t]+", " ", str(val))
    return re.sub(r"\s+", " ", text).strip()


def is_valid_url(url):
    if not url or not isinstance(url, str):
        return False
    parsed = urlparse(url.strip())
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def standardize_level(val):
    normalized = clean_text(val).lower()
    if normalized in ("central", "central government"):
        return "Central Government"
    if normalized in ("state", "state government", "state / ut government", "state/ut", "state / ut"):
        return "State / UT Government"
    return clean_text(val)


def load_dataset(filepath=RAW_DATA_PATH):
    path = Path(filepath)
    if not path.exists():
        ensure_directories()
        raise FileNotFoundError(
            f"Raw dataset not found at '{path}'.\n"
            "Please download 'gov_myscheme_data.csv' from https://github.com/Aryan-Pardeshi/gov-myscheme-dataset "
            f"and place it at '{path}'."
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

    # Use MyScheme portal link if the primary official link is blank
    if "MyScheme URL" in df.columns:
        empty_url = out["source_url"].eq("")
        out.loc[empty_url, "source_url"] = (
            df.loc[empty_url, "MyScheme URL"].fillna("").astype(str).map(clean_text)
        )

    # Keep all other original columns so no source data is lost
    for col in df.columns:
        if col not in out.columns:
            out[f"raw_{col}"] = df[col]

    return out


def audit_dataset(df):
    issues = []

    # Check for rows with identical values across all data fields (ignoring generated record_id)
    content_cols = [c for c in df.columns if c != "record_id"]
    dup_rows = df.duplicated(subset=content_cols, keep=False)
    for _, r in df[dup_rows].iterrows():
        issues.append({
            "record_id": r["record_id"],
            "scheme_name": r["scheme_name"],
            "issue_type": "EXACT_DUPLICATE_ROW",
            "issue_detail": "Identical values across all data columns found in another row.",
        })

    # Normalized scheme names help spot candidates that might be duplicates despite minor punctuation or case differences
    norm_names = df["scheme_name"].str.lower().str.replace(r"[^\w\s]", "", regex=True).str.strip()
    name_counts = norm_names.value_counts()
    dup_name_set = set(name_counts[name_counts > 1].index) - {""}

    for idx, r in df.iterrows():
        rec_id = r["record_id"]
        s_name = r["scheme_name"]
        desc = r["description"]
        url = r["source_url"]
        elig = r["eligibility"]
        bens = r["benefits"]
        n_name = norm_names.iloc[idx]

        if n_name in dup_name_set:
            issues.append({
                "record_id": rec_id,
                "scheme_name": s_name,
                "issue_type": "DUPLICATE_NAME_CANDIDATE",
                "issue_detail": f"Name matches {name_counts[n_name]} candidate schemes (verify if distinct).",
            })

        if not s_name:
            issues.append({
                "record_id": rec_id,
                "scheme_name": "(empty)",
                "issue_type": "MISSING_SCHEME_NAME",
                "issue_detail": "Scheme name is empty.",
            })

        if not desc:
            issues.append({
                "record_id": rec_id,
                "scheme_name": s_name,
                "issue_type": "MISSING_DESCRIPTION",
                "issue_detail": "Description is empty.",
            })
        elif len(desc) < 30:
            issues.append({
                "record_id": rec_id,
                "scheme_name": s_name,
                "issue_type": "SHORT_DESCRIPTION",
                "issue_detail": f"Description has only {len(desc)} characters.",
            })

        if not url:
            issues.append({
                "record_id": rec_id,
                "scheme_name": s_name,
                "issue_type": "MISSING_SOURCE_URL",
                "issue_detail": "Source URL is missing.",
            })
        elif not is_valid_url(url):
            issues.append({
                "record_id": rec_id,
                "scheme_name": s_name,
                "issue_type": "MALFORMED_SOURCE_URL",
                "issue_detail": f"Malformed URL syntax: '{url}'.",
            })

        if not elig:
            issues.append({
                "record_id": rec_id,
                "scheme_name": s_name,
                "issue_type": "MISSING_ELIGIBILITY",
                "issue_detail": "Eligibility text is empty.",
            })

        if not bens:
            issues.append({
                "record_id": rec_id,
                "scheme_name": s_name,
                "issue_type": "MISSING_BENEFITS",
                "issue_detail": "Benefits text is empty.",
            })

        if (not desc or len(desc) < 30) and (not elig or not bens):
            issues.append({
                "record_id": rec_id,
                "scheme_name": s_name,
                "issue_type": "INSUFFICIENT_INFORMATION",
                "issue_detail": "Record lacks sufficient text for useful recommendations.",
            })

    issues_df = pd.DataFrame(issues) if issues else pd.DataFrame(
        columns=["record_id", "scheme_name", "issue_type", "issue_detail"]
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
        ("total_issues_flagged", len(issues_df)),
    ]
    summary_df = pd.DataFrame(summary_rows, columns=["metric", "value"])
    return summary_df, issues_df


def preprocess_data(df):
    clean_df = df.copy()

    text_cols = ["scheme_name", "description", "benefits", "eligibility", "application_process", "department"]
    for col in text_cols:
        if col in clean_df.columns:
            clean_df[col] = clean_df[col].map(clean_text)

    clean_df["level"] = clean_df["level"].map(standardize_level)

    # Combine text fields to support keyword search and recommendation matching
    search_parts = []
    for _, row in clean_df.iterrows():
        fields = [row["scheme_name"], row["description"], row["benefits"], row["eligibility"], row["application_process"]]
        search_parts.append(" ".join([f for f in fields if f]))

    clean_df["search_text"] = search_parts

    search_cols = [
        "record_id", "scheme_name", "department", "level",
        "benefits", "eligibility", "search_text", "source_url"
    ]
    search_df = clean_df[[c for c in search_cols if c in clean_df.columns]].copy()

    return clean_df, search_df


def run_pipeline(raw_path=RAW_DATA_PATH):
    ensure_directories()
    raw_file = Path(raw_path)

    print(f"Loading dataset from: {raw_file}")
    df = load_dataset(raw_file)
    print(f"Loaded {len(df)} records.")

    print("Auditing data quality...")
    summary_df, issues_df = audit_dataset(df)
    summary_df.to_csv(AUDIT_SUMMARY_PATH, index=False)
    issues_df.to_csv(RECORD_ISSUES_PATH, index=False)
    print(f"Saved: {AUDIT_SUMMARY_PATH} ({len(summary_df)} metrics)")
    print(f"Saved: {RECORD_ISSUES_PATH} ({len(issues_df)} issues flagged)")

    print("Preprocessing dataset...")
    clean_df, search_df = preprocess_data(df)

    print("Exporting processed datasets...")
    clean_df.to_csv(CLEAN_SCHEMES_PATH, index=False)
    search_df.to_csv(SEARCH_SCHEMES_PATH, index=False)
    print(f"Saved: {CLEAN_SCHEMES_PATH} ({len(clean_df)} records)")
    print(f"Saved: {SEARCH_SCHEMES_PATH} ({len(search_df)} records)")

    return {
        "raw_records": len(df),
        "issues_flagged": len(issues_df),
        "clean_records": len(clean_df),
        "search_records": len(search_df),
    }


def main():
    custom_input = None
    if "--input" in sys.argv:
        idx = sys.argv.index("--input")
        if idx + 1 < len(sys.argv):
            custom_input = sys.argv[idx + 1]

    target_path = Path(custom_input) if custom_input else RAW_DATA_PATH
    try:
        counts = run_pipeline(raw_path=target_path)
        print("\nPipeline finished successfully:")
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
