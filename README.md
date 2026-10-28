# SchemeSeva

SchemeSeva is an Indian government scheme recommendation system. This repository contains the data preparation pipeline: loading the scheme catalogue, running data quality audit checks, and cleaning the text for recommendation models.

## How it works

1. **Load data**: Reads the raw scheme CSV, maps various column names to standard fields (`scheme_name`, `description`, `benefits`, `eligibility`, `application_process`, `department`, `level`, `source_url`), and assigns a unique `record_id` to each row.
2. **Audit**: Checks the data for exact duplicate rows, similar scheme names, missing fields, short descriptions, and malformed URLs. Outputs summary metrics and a list of flagged issues without deleting any records.
3. **Preprocess**: Cleans up whitespace, standardizes government levels (`Central Government` vs `State / UT Government`), and combines available text into a single `search_text` column.
4. **Export**: Saves clean and search-ready CSV files for downstream modeling.

## Project Structure

```text
SchemeSeva/
├── data/
│   ├── raw/                         # Place raw gov_myscheme_data.csv here
│   └── processed/
│       ├── schemes_clean.csv        # Cleaned dataset with all structured fields
│       └── schemes_for_search.csv   # Search dataset with search_text
├── reports/
│   ├── audit_summary.csv            # Summary metrics and issue counts
│   └── record_issues.csv            # Row-by-row issues flagged during audit
├── src/
│   └── main.py                      # Main pipeline script
├── tests/
│   └── test_pipeline.py             # Unit tests using test fixtures
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

Download `gov_myscheme_data.csv` from [Aryan-Pardeshi/gov-myscheme-dataset](https://github.com/Aryan-Pardeshi/gov-myscheme-dataset) and place it in `data/raw/`:

```text
data/raw/gov_myscheme_data.csv
```

## Running the pipeline

```bash
python src/main.py
```

To run on a different CSV file:
```bash
python src/main.py --input path/to/file.csv
```

## Running tests

```bash
python -m unittest tests/test_pipeline.py
```

## Output files

- `reports/audit_summary.csv`: Overall counts of records, duplicates, missing values, and invalid links.
- `reports/record_issues.csv`: Individual records flagged for issues (e.g., short descriptions, missing eligibility, potential duplicate names).
- `data/processed/schemes_clean.csv`: Cleaned dataset with normalized text and standardized levels.
- `data/processed/schemes_for_search.csv`: Compact dataset with `record_id`, `scheme_name`, `department`, `level`, `benefits`, `eligibility`, `source_url`, and `search_text`.
