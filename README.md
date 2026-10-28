# SchemeSeva — Data Audit and Preprocessing Pipeline

SchemeSeva is an Indian government scheme recommendation system in development. This repository contains the data preparation pipeline that loads the published scheme catalogue, audits data quality, and preprocesses records into clean, searchable datasets for ML retrieval models.

---

## 1. Project Purpose

Before training recommendation models, we need a clean, structured, and auditable dataset. This pipeline:
1. Loads the published government scheme catalogue and maps varying column names to standard fields.
2. Audits data health (missing values, exact duplicates, candidate name duplicates, malformed URLs, and insufficient text).
3. Cleans whitespace, standardizes governance level labels, and constructs a unified `search_text` field without dropping records.
4. Exports cleaned datasets and audit reports for downstream ML retrieval.

---

## 2. Dataset Setup

The project uses the publicly available Indian government scheme catalogue published by Aryan Pardeshi:
- **Repository**: [https://github.com/Aryan-Pardeshi/gov-myscheme-dataset](https://github.com/Aryan-Pardeshi/gov-myscheme-dataset)
- **Source file**: `gov_myscheme_data.csv`

### Placement Instructions
1. Download `gov_myscheme_data.csv` from the link above.
2. Place the file at:
   ```text
   data/raw/gov_myscheme_data.csv
   ```
3. Keep the raw file unchanged to ensure data provenance and reproducibility.

If the file is absent, running the script will cleanly stop and explain where to place it.

---

## 3. Project Structure

```text
SchemeSeva/
├── data/
│   ├── raw/
│   │   └── gov_myscheme_data.csv       # Place the raw dataset here
│   └── processed/
│       ├── schemes_clean.csv           # Cleaned structured dataset
│       └── schemes_for_search.csv      # Search-ready dataset with search_text
├── reports/
│   ├── audit_summary.csv               # Dataset-level defect and health metrics
│   └── record_issues.csv               # Specific issues flagged per record
├── src/
│   └── main.py                         # Complete standalone pipeline script (~240 lines)
├── tests/
│   └── test_pipeline.py                # Unit test suite covering all edge cases
├── requirements.txt                    # Minimal dependency (pandas)
├── README.md
└── .gitignore
```

---

## 4. Dependencies and Installation

The pipeline requires Python 3.9+ and **pandas**:

```bash
pip install -r requirements.txt
```

---

## 5. Execution Commands

### Run the Pipeline
```bash
python src/main.py
```

*Optional parameter for a custom input file:*
```bash
python src/main.py --input path/to/another_dataset.csv
```

### Run Unit Tests
```bash
python -m unittest tests/test_pipeline.py -v
```

---

## 6. Output Files

| File | Purpose |
| :--- | :--- |
| `reports/audit_summary.csv` | Summarizes high-level quality metrics (total rows, duplicates, missing counts per field, malformed URLs). |
| `reports/record_issues.csv` | Identifies specific quality issues flagged for each record (e.g. duplicate candidates, missing eligibility, short descriptions). |
| `data/processed/schemes_clean.csv` | Full cleaned dataset with normalized whitespace, standardized `level`, and stable `record_id`. |
| `data/processed/schemes_for_search.csv` | Compact dataset containing `record_id`, `scheme_name`, `department`, `level`, `benefits`, `eligibility`, `source_url`, and the combined `search_text`. |

---

## 7. Dataset Limitations

- **Published Snapshot**: The dataset is a point-in-time snapshot of schemes. Benefits, eligibility thresholds, and active status may change over time on official portals.
- **No Manual Verification**: The pipeline uses the published catalogue directly. A working link or government domain name does not guarantee that a scheme is active or that rules have not been amended.
- **No Synthetic Records**: This project uses only authentic government scheme records, never synthetic citizen or applicant tables.

---

## 8. Five-Cell Google Colab Rewrite Plan

To reproduce this pipeline in Google Colab in approximately four hours:

- **Cell 1: Setup**
  ```python
  import re
  from pathlib import Path
  from urllib.parse import urlparse
  import pandas as pd
  ```
- **Cell 2: Load Data**
  Define `load_dataset(filepath)` with column mapping to standard fields (`record_id`, `scheme_name`, `description`, `benefits`, `eligibility`, `application_process`, `department`, `level`, `source_url`).
- **Cell 3: Audit**
  Define `audit_dataset(df)` to check duplicates, nulls, short text, and malformed URLs. Export `audit_summary.csv` and `record_issues.csv`.
- **Cell 4: Preprocessing**
  Define `clean_text`, `standardize_level`, and `preprocess_data(df)`. Construct `search_text` without inserting `"nan"` or `"None"`.
- **Cell 5: Export**
  Export `schemes_clean.csv` and `schemes_for_search.csv`, then display `df.head()` and key metrics.
