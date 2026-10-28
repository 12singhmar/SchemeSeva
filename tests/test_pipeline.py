"""Unit tests for simplified SchemeSeva audit and preprocessing pipeline.

Uses a temporary, isolated synthetic fixture to test edge cases without
touching real government scheme datasets.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
import pandas as pd

# Ensure 'src' is importable
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from main import (
    clean_text,
    is_valid_url,
    standardize_level,
    load_dataset,
    audit_dataset,
    preprocess_data,
    run_pipeline,
)


class TestSimplifiedSchemeSevaPipeline(unittest.TestCase):
    """Test suite covering data loading, audit checks, and preprocessing."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.temp_dir.name)

        # Create a small, controlled synthetic test fixture
        self.fixture_csv = self.dir_path / "fixture_schemes.csv"
        fixture_data = [
            {
                "Scheme Name": "PM Kisan Samman Nidhi",
                "Description": "Income support scheme providing 6000 rupees annually to eligible small and marginal farmer families.",
                "Benefits": "Rs 6000 per year paid in three equal four-monthly installments directly into bank accounts.",
                "Eligibility Criteria": "Small and marginal farmers owning cultivable land; not eligible if institutional landholder.",
                "Application Process": "Apply online at pmkisan.gov.in or through local CSC centers.",
                "State / UT / Ministry": "Ministry of Agriculture and Farmers Welfare",
                "Level": "Central Government",
                "Official Link": "https://pmkisan.gov.in",
            },
            {
                # Duplicate candidate of row 1 (same name, minor case/punctuation difference)
                "Scheme Name": "pm kisan samman nidhi!",
                "Description": "Alternative record for PM Kisan with identical scheme name.",
                "Benefits": "Financial assistance to farmers.",
                "Eligibility Criteria": "Farmers only.",
                "Application Process": "Online portal.",
                "State / UT / Ministry": "Agriculture",
                "Level": "central",
                "Official Link": "https://pmkisan.gov.in/portal",
            },
            {
                # Exact duplicate of row 1 (for testing exact duplicate row detection)
                "Scheme Name": "PM Kisan Samman Nidhi",
                "Description": "Income support scheme providing 6000 rupees annually to eligible small and marginal farmer families.",
                "Benefits": "Rs 6000 per year paid in three equal four-monthly installments directly into bank accounts.",
                "Eligibility Criteria": "Small and marginal farmers owning cultivable land; not eligible if institutional landholder.",
                "Application Process": "Apply online at pmkisan.gov.in or through local CSC centers.",
                "State / UT / Ministry": "Ministry of Agriculture and Farmers Welfare",
                "Level": "Central Government",
                "Official Link": "https://pmkisan.gov.in",
            },
            {
                # Short description
                "Scheme Name": "State Dairy Loan Scheme",
                "Description": "Short text.",
                "Benefits": "Subsidy on cattle loans.",
                "Eligibility Criteria": "Rural dairy producers only.",
                "Application Process": "Visit district cooperative bank.",
                "State / UT / Ministry": "Punjab Animal Husbandry",
                "Level": "State / UT Government",
                "Official Link": "http://dairy.punjab.gov.in",
            },
            {
                # Missing eligibility and malformed URL
                "Scheme Name": "Rural Housing Grant",
                "Description": "Housing assistance for rural homeless households with low income.",
                "Benefits": "Construction grant of Rs 1,20,000.",
                "Eligibility Criteria": "",  # missing
                "Application Process": "Gram Panchayat recommendation.",
                "State / UT / Ministry": "Rural Development",
                "Level": "State",
                "Official Link": "not-a-valid-url",  # malformed
            },
            {
                # Insufficient information (missing description, eligibility, benefits)
                "Scheme Name": "Old Incomplete Scheme",
                "Description": "",
                "Benefits": "",
                "Eligibility Criteria": "",
                "Application Process": "",
                "State / UT / Ministry": "Unknown",
                "Level": "State",
                "Official Link": "",
            },
        ]
        pd.DataFrame(fixture_data).to_csv(self.fixture_csv, index=False)

    def tearDown(self):
        self.temp_dir.cleanup()

    # --- Loading & Column Mapping Tests ---

    def test_load_dataset_valid_csv(self):
        df = load_dataset(self.fixture_csv)
        self.assertEqual(len(df), 6)
        expected_cols = [
            "record_id", "scheme_name", "description", "benefits",
            "eligibility", "application_process", "department", "level", "source_url"
        ]
        for col in expected_cols:
            self.assertIn(col, df.columns)
        self.assertTrue(df["record_id"].iloc[0].startswith("SCH_"))

    def test_load_dataset_missing_file_raises_clear_error(self):
        missing_path = self.dir_path / "non_existent.csv"
        with self.assertRaises(FileNotFoundError) as ctx:
            load_dataset(missing_path)
        self.assertIn("Raw dataset not found", str(ctx.exception))
        self.assertIn("https://github.com/Aryan-Pardeshi/gov-myscheme-dataset", str(ctx.exception))

    def test_load_dataset_missing_scheme_name_raises_error(self):
        bad_csv = self.dir_path / "bad.csv"
        pd.DataFrame({"RandomColumn": [1, 2], "Details": ["A", "B"]}).to_csv(bad_csv, index=False)
        with self.assertRaises(ValueError) as ctx:
            load_dataset(bad_csv)
        self.assertIn("missing required scheme name", str(ctx.exception))

    def test_load_dataset_empty_dataframe(self):
        empty_csv = self.dir_path / "empty.csv"
        pd.DataFrame(columns=["Scheme Name", "Description", "Benefits"]).to_csv(empty_csv, index=False)
        df = load_dataset(empty_csv)
        self.assertEqual(len(df), 0)
        self.assertIn("scheme_name", df.columns)
        self.assertIn("record_id", df.columns)

    # --- Audit Tests ---

    def test_audit_flags_exact_duplicates(self):
        df = load_dataset(self.fixture_csv)
        _, issues_df = audit_dataset(df)
        exact_dups = issues_df[issues_df["issue_type"] == "EXACT_DUPLICATE_ROW"]
        self.assertGreaterEqual(len(exact_dups), 2)

    def test_audit_flags_duplicate_name_candidates(self):
        df = load_dataset(self.fixture_csv)
        _, issues_df = audit_dataset(df)
        name_dups = issues_df[issues_df["issue_type"] == "DUPLICATE_NAME_CANDIDATE"]
        self.assertGreaterEqual(len(name_dups), 2)

    def test_audit_missing_values_and_short_descriptions(self):
        df = load_dataset(self.fixture_csv)
        summary_df, issues_df = audit_dataset(df)
        metrics = dict(zip(summary_df["metric"], summary_df["value"]))

        self.assertEqual(metrics["total_records"], 6)
        self.assertGreater(metrics["short_description_count (<30 chars)"], 0)
        self.assertGreater(metrics["missing_eligibility_count"], 0)
        self.assertGreater(metrics["missing_benefits_count"], 0)

        short_desc = issues_df[issues_df["issue_type"] == "SHORT_DESCRIPTION"]
        self.assertTrue(any(short_desc["scheme_name"].str.contains("Dairy")))

    def test_audit_malformed_and_missing_urls(self):
        df = load_dataset(self.fixture_csv)
        _, issues_df = audit_dataset(df)

        malformed = issues_df[issues_df["issue_type"] == "MALFORMED_SOURCE_URL"]
        self.assertTrue(any(malformed["scheme_name"].str.contains("Housing")))

        missing_url = issues_df[issues_df["issue_type"] == "MISSING_SOURCE_URL"]
        self.assertTrue(any(missing_url["scheme_name"].str.contains("Old Incomplete")))

    def test_audit_does_not_drop_records(self):
        df = load_dataset(self.fixture_csv)
        summary_df, issues_df = audit_dataset(df)
        # Original df length must remain intact
        self.assertEqual(len(df), 6)
        self.assertGreater(len(issues_df), 0)

    # --- Cleaning & Preprocessing Tests ---

    def test_clean_text_normalizes_whitespace(self):
        raw = "  Eligible for small farmers\n\twith land < 2 hectares.   Not eligible for institutional landholders.  "
        cleaned = clean_text(raw)
        self.assertEqual(cleaned, "Eligible for small farmers with land < 2 hectares. Not eligible for institutional landholders.")

    def test_preserve_negations_and_criteria(self):
        raw = "Small farmers only; not eligible if age > 60 or income > 2,00,000."
        cleaned = clean_text(raw)
        self.assertIn("not eligible", cleaned)
        self.assertIn("only", cleaned)
        self.assertIn("age > 60", cleaned)
        self.assertIn("income > 2,00,000", cleaned)

    def test_standardize_level(self):
        self.assertEqual(standardize_level("central"), "Central Government")
        self.assertEqual(standardize_level("Central Government"), "Central Government")
        self.assertEqual(standardize_level("state"), "State / UT Government")
        self.assertEqual(standardize_level("State / UT Government"), "State / UT Government")
        self.assertEqual(standardize_level("State"), "State / UT Government")

    def test_search_text_generation_and_no_nan_or_none(self):
        df = load_dataset(self.fixture_csv)
        clean_df, search_df = preprocess_data(df)

        self.assertEqual(len(clean_df), 6)
        self.assertEqual(len(search_df), 6)
        self.assertIn("search_text", search_df.columns)

        for text in search_df["search_text"]:
            words = text.split()
            self.assertNotIn("nan", words)
            self.assertNotIn("None", words)

    def test_preprocessing_retains_all_records(self):
        df = load_dataset(self.fixture_csv)
        clean_df, search_df = preprocess_data(df)
        # Incomplete schemes are retained without silent dropping
        self.assertEqual(len(clean_df), len(df))
        self.assertEqual(len(search_df), len(df))

    # --- End-to-end Pipeline Execution Test ---

    def test_run_pipeline_exports_all_four_files(self):
        temp_raw = self.dir_path / "data" / "raw" / "gov_myscheme_data.csv"
        temp_raw.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(pd.read_csv(self.fixture_csv)).to_csv(temp_raw, index=False)

        import main
        orig_raw = main.RAW_DATA_PATH
        orig_proc = main.PROCESSED_DIR
        orig_rep = main.REPORTS_DIR

        try:
            main.RAW_DATA_PATH = temp_raw
            main.PROCESSED_DIR = self.dir_path / "data" / "processed"
            main.REPORTS_DIR = self.dir_path / "reports"
            main.AUDIT_SUMMARY_PATH = main.REPORTS_DIR / "audit_summary.csv"
            main.RECORD_ISSUES_PATH = main.REPORTS_DIR / "record_issues.csv"
            main.CLEAN_SCHEMES_PATH = main.PROCESSED_DIR / "schemes_clean.csv"
            main.SEARCH_SCHEMES_PATH = main.PROCESSED_DIR / "schemes_for_search.csv"

            counts = run_pipeline(raw_path=temp_raw)
            self.assertEqual(counts["raw_records"], 6)
            self.assertEqual(counts["clean_records"], 6)
            self.assertEqual(counts["search_records"], 6)

            # Check that all four expected files exist
            self.assertTrue(main.AUDIT_SUMMARY_PATH.exists())
            self.assertTrue(main.RECORD_ISSUES_PATH.exists())
            self.assertTrue(main.CLEAN_SCHEMES_PATH.exists())
            self.assertTrue(main.SEARCH_SCHEMES_PATH.exists())
        finally:
            main.RAW_DATA_PATH = orig_raw
            main.PROCESSED_DIR = orig_proc
            main.REPORTS_DIR = orig_rep
            main.AUDIT_SUMMARY_PATH = orig_rep / "audit_summary.csv"
            main.RECORD_ISSUES_PATH = orig_rep / "record_issues.csv"
            main.CLEAN_SCHEMES_PATH = orig_proc / "schemes_clean.csv"
            main.SEARCH_SCHEMES_PATH = orig_proc / "schemes_for_search.csv"


if __name__ == "__main__":
    unittest.main()
