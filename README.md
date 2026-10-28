# SchemeSeva — Data Audit, Verification, and Preprocessing

Stage 1 data preparation pipeline for **SchemeSeva**, an Indian government scheme recommendation system.

The goal of this stage is to audit raw government scheme data, coordinate human verification against official government sources, and preprocess verified records for future ML retrieval models.

---

## 1. Project Structure

```text
SchemeSeva/
├── data/
│   ├── raw/
│   │   └── gov_myscheme_data.csv       # Candidate raw dataset (place here)
│   └── processed/
│       ├── verified_schemes.csv        # Records confirmed by human review
│       ├── pending_schemes.csv         # Awaiting review or unverified
│       ├── inactive_schemes.csv        # Officially discontinued schemes
│       └── verified_schemes_clean.csv  # Preprocessed dataset ready for ML
├── reports/
│   ├── audit_summary.csv               # Dataset-level health and defect metrics
│   ├── record_issues.csv               # Row-by-row audit findings and anomalies
│   └── scheme_review.csv               # Verification review sheet for human decisions
├── src/
│   └── main.py                         # Complete standalone pipeline script
├── tests/
│   └── test_pipeline.py                # Unit test suite on isolated fixtures
├── requirements.txt                    # Minimal dependencies (pandas, requests)
└── README.md
```

---

## 2. Dataset Setup

This project uses the publicly available scheme catalogue from:
**https://github.com/Aryan-Pardeshi/gov-myscheme-dataset**

To set up:
1. Download `gov_myscheme_data.csv` from the repository above.
2. Place the file directly at:
   ```text
   data/raw/gov_myscheme_data.csv
   ```
3. Keep the raw file unchanged to ensure data provenance and reproducibility.

If the file is not present when running the script, the pipeline will display a clear error message and stop without creating fake records.

---

## 3. Five-Step Pipeline Architecture

The pipeline in `src/main.py` operates in five distinct, transparent steps suitable for reproduction in Google Colab notebook cells:

1. **Step 1: Load and Standardize (`load_dataset`)**
   - Validates file existence and reads the CSV safely.
   - Maps source columns into standard fields (`record_id`, `scheme_name`, `description`, `benefits`, `eligibility`, `application_process`, `department`, `level`, `source_url`).
   - Retains original columns (`raw_*`) to prevent data loss.

2. **Step 2: Audit (`audit_dataset`)**
   - Checks record counts, missing values, exact row duplicates, candidate duplicate names, short descriptions (<30 chars), malformed URLs, and records with insufficient information.
   - Exports `reports/audit_summary.csv` and `reports/record_issues.csv` without deleting records.

3. **Step 3: Verification Review (`build_or_update_review`, `apply_verification_decisions`)**
   - Identifies official government domains (`.gov.in`, `.nic.in`, `myscheme.gov.in`).
   - Generates/updates `reports/scheme_review.csv` with review statuses (`UNVERIFIED`, `NEEDS_REVIEW`).
   - **Crucial rule:** Records are **never** marked `VERIFIED` automatically. Human judgment and official evidence URLs are strictly required.
   - Partitions rows into `data/processed/verified_schemes.csv`, `pending_schemes.csv`, and `inactive_schemes.csv`.

4. **Step 4: Preprocessing (`preprocess_verified`)**
   - Preprocesses **only** records in `verified_schemes.csv`.
   - Cleans formatting and whitespace; standardizes government level labels (`Central Government`, `State / UT Government`).
   - Generates a composite `search_text` column without adding filler words like `"nan"` or `"None"`.
   - Preserves negations and eligibility constraints (e.g., `"not eligible"`, `"only"`, income limits).

5. **Step 5: Export (`run_pipeline`)**
   - Saves clean verified data to `data/processed/verified_schemes_clean.csv`.
   - Prints execution metrics and defect summaries.

---

## 4. Key Functions Reference

| Function | Primary Purpose |
| :--- | :--- |
| `ensure_directories()` | Automatically creates `data/raw`, `data/processed`, and `reports`. |
| `clean_text(val)` | Strips leading/trailing whitespace and normalizes repeated whitespace. |
| `is_valid_url(url)` | Checks if a URL has a valid HTTP/HTTPS scheme and network domain. |
| `is_gov_domain(url)` | Checks if the URL host matches official government portals (`.gov.in`, `.nic.in`, `myscheme.gov.in`). |
| `standardize_level(val)` | Maps variations of government level into standard labels. |
| `load_dataset(filepath)` | Reads the CSV and maps columns to a standard schema with stable IDs. |
| `audit_dataset(df)` | Evaluates 8 quality criteria and produces high-level and row-level audit reports. |
| `build_or_update_review(...)` | Populates review sheet while strictly preserving existing human reviewer edits. |
| `apply_verification_decisions(...)` | Partitions records based on human review decisions (`VERIFIED`, `PENDING`, `INACTIVE`). |
| `preprocess_verified(verified_df)` | Cleans text, standardizes labels, and generates `search_text` for verified schemes. |
| `run_pipeline(raw_path, check_urls)` | Orchestrates the entire five-step pipeline and prints execution counts. |

---

## 5. Execution Instructions

### Setup Environment
```bash
pip install -r requirements.txt
```

### Run the Pipeline
```bash
python src/main.py
```

*Optional flags:*
- `python src/main.py --check-urls`: Performs lightweight HTTP availability checks on source URLs.
- `python src/main.py --input path/to/dataset.csv`: Runs the pipeline against an alternative CSV file.

### Run Unit Tests
```bash
python -m unittest tests/test_pipeline.py -v
```

---

## 6. How to Perform Human Verification

1. Run the pipeline to generate `reports/scheme_review.csv`.
2. Open `reports/scheme_review.csv` in a spreadsheet editor.
3. For each candidate scheme:
   - Check the scheme against official government sources (MyScheme portal, Ministry website, gazette notifications).
   - If genuine and active: change `verification_status` to `VERIFIED`, enter the official source link in `official_evidence_url`, and add any notes in `reviewer_notes`.
   - If discontinued or superseded: change `verification_status` to `INACTIVE`.
   - If conflicting or incomplete: keep or set status to `NEEDS_REVIEW`.
4. Save the CSV file and re-run:
   ```bash
   python src/main.py
   ```
5. The pipeline reads your decisions and populates `data/processed/verified_schemes.csv` and `data/processed/verified_schemes_clean.csv`.

---

## 7. Google Colab Handoff Guide

When rewriting this in Google Colab:
- **Cell 1**: Install dependencies (`!pip install pandas requests`) and import standard libraries.
- **Cell 2**: Step 1 — Define `load_dataset` and column mapping.
- **Cell 3**: Step 2 — Define `audit_dataset` and run on loaded data.
- **Cell 4**: Step 3 — Define `build_or_update_review` and `apply_verification_decisions`.
- **Cell 5**: Step 4 & 5 — Define `clean_text`, `preprocess_verified`, and export clean CSV.

---

## 8. Current Limitations & Scope Boundary

- **No Synthetic Records**: This project operates strictly on real scheme records, never synthetic citizen or applicant tables.
- **No Automatic Verification**: Working links or government domains do not prove a scheme is active or described correctly. Human verification is strictly required.
- **Out of Scope for Stage 1**: Recommender models (TF-IDF / vector search), classification models, citizen profiling, and web UIs are intentionally omitted until a verified dataset is established.
