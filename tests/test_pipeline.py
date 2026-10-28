"""Unit tests for SchemeSeva audit, verification, and preprocessing pipeline.

Uses a temporary, clearly labelled test dataset fixture. Does not touch or
mix with real government scheme datasets.
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
    is_gov_domain,
    standardize_level,
    load_dataset,
    audit_dataset,
    build_or_update_review,
    apply_verification_decisions,
    preprocess_verified,
    run_pipeline,
)


class TestSchemeSevaPipeline(unittest.TestCase):
    """Test suite covering data loading, audit checks, verification, and preprocessing."""

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
                # Non-gov URL and short description
                "Scheme Name": "State Dairy Loan Scheme",
                "Description": "Short text.",
                "Benefits": "Subsidy on cattle loans.",
                "Eligibility Criteria": "Rural dairy producers only.",
                "Application Process": "Visit district cooperative bank.",
                "State / UT / Ministry": "Punjab Animal Husbandry",
                "Level": "State / UT Government",
                "Official Link": "http://private-dairy-loans.org/info",
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

    # --- Loading & Schema tests ---

    def test_missing_dataset_raises_clear_error(self):
        missing_path = self.dir_path / "non_existent.csv"
        with self.assertRaises(FileNotFoundError) as ctx:
            load_dataset(missing_path)
        self.assertIn("Raw dataset not found", str(ctx.exception))

    def test_missing_scheme_name_column_raises_value_error(self):
        bad_csv = self.dir_path / "bad.csv"
        pd.DataFrame({"RandomCol": [1, 2], "Details": ["A", "B"]}).to_csv(bad_csv, index=False)
        with self.assertRaises(ValueError) as ctx:
            load_dataset(bad_csv)
        self.assertIn("missing required scheme name", str(ctx.exception))

    def test_load_dataset_maps_columns_and_assigns_ids(self):
        df = load_dataset(self.fixture_csv)
        self.assertEqual(len(df), 6)
        expected_cols = [
            "record_id", "scheme_name", "description", "benefits",
            "eligibility", "application_process", "department", "level", "source_url"
        ]
        for col in expected_cols:
            self.assertIn(col, df.columns)
        self.assertTrue(df["record_id"].iloc[0].startswith("SCH_"))

    # --- URL and domain check tests ---

    def test_is_valid_url(self):
        self.assertTrue(is_valid_url("https://pmkisan.gov.in"))
        self.assertTrue(is_valid_url("http://myscheme.gov.in/schemes"))
        self.assertFalse(is_valid_url("not-a-url"))
        self.assertFalse(is_valid_url(""))
        self.assertFalse(is_valid_url(None))

    def test_is_gov_domain(self):
        self.assertTrue(is_gov_domain("https://pmkisan.gov.in/portal"))
        self.assertTrue(is_gov_domain("https://serviceonline.nic.in/service"))
        self.assertTrue(is_gov_domain("https://www.myscheme.gov.in/schemes/pm-kisan"))
        self.assertFalse(is_gov_domain("http://private-dairy-loans.org/info"))
        self.assertFalse(is_gov_domain("https://commercial-scheme-blog.com"))

    # --- Audit tests ---

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

    def test_audit_flags_short_description_and_missing_fields(self):
        df = load_dataset(self.fixture_csv)
        summary_df, issues_df = audit_dataset(df)

        metrics = dict(zip(summary_df["metric"], summary_df["value"]))
        self.assertEqual(metrics["total_records"], 6)
        self.assertGreater(metrics["short_description_count (<30 chars)"], 0)
        self.assertGreater(metrics["missing_eligibility_count"], 0)
        self.assertGreater(metrics["malformed_source_url_count"], 0)

        short_desc_issues = issues_df[issues_df["issue_type"] == "SHORT_DESCRIPTION"]
        self.assertTrue(any(short_desc_issues["scheme_name"].str.contains("Dairy")))

        malformed_url_issues = issues_df[issues_df["issue_type"] == "MALFORMED_SOURCE_URL"]
        self.assertTrue(any(malformed_url_issues["scheme_name"].str.contains("Housing")))

    # --- Verification & Human Decision tests ---

    def test_initial_review_sheet_never_marks_verified_or_inactive(self):
        df = load_dataset(self.fixture_csv)
        review_file = self.dir_path / "scheme_review.csv"
        review_df = build_or_update_review(df, review_path=review_file, check_urls=False)

        statuses = review_df["verification_status"].tolist()
        self.assertNotIn("VERIFIED", statuses)
        self.assertNotIn("INACTIVE", statuses)
        # Gov URLs are UNVERIFIED pending review; missing/non-gov are NEEDS_REVIEW
        self.assertTrue(all(s in ("UNVERIFIED", "NEEDS_REVIEW") for s in statuses))

    def test_unverified_records_cannot_enter_verified_dataset(self):
        df = load_dataset(self.fixture_csv)
        review_file = self.dir_path / "scheme_review.csv"
        review_df = build_or_update_review(df, review_path=review_file, check_urls=False)

        # With no human review, verified_df must be empty
        verified_df, pending_df, inactive_df = apply_verification_decisions(df, review_df)
        self.assertEqual(len(verified_df), 0)
        self.assertEqual(len(pending_df), 6)
        self.assertEqual(len(inactive_df), 0)

    def test_human_decisions_applied_and_preserved(self):
        df = load_dataset(self.fixture_csv)
        review_file = self.dir_path / "scheme_review.csv"
        review_df = build_or_update_review(df, review_path=review_file, check_urls=False)

        # Simulate human reviewer entering decisions:
        # Mark SCH_0001 as VERIFIED with official evidence URL
        # Mark SCH_0006 as INACTIVE
        review_df.loc[review_df["record_id"] == "SCH_0001", "verification_status"] = "VERIFIED"
        review_df.loc[review_df["record_id"] == "SCH_0001", "official_evidence_url"] = "https://pmkisan.gov.in"
        review_df.loc[review_df["record_id"] == "SCH_0001", "reviewer_notes"] = "Confirmed against Ministry of Agriculture portal."

        review_df.loc[review_df["record_id"] == "SCH_0006", "verification_status"] = "INACTIVE"
        review_df.loc[review_df["record_id"] == "SCH_0006", "reviewer_notes"] = "Scheme discontinued."
        review_df.to_csv(review_file, index=False)

        # Test preserving human decisions on re-read/update
        updated_review = build_or_update_review(df, review_path=review_file, check_urls=False)
        sch1 = updated_review[updated_review["record_id"] == "SCH_0001"].iloc[0]
        self.assertEqual(sch1["verification_status"], "VERIFIED")
        self.assertEqual(sch1["reviewer_notes"], "Confirmed against Ministry of Agriculture portal.")

        sch6 = updated_review[updated_review["record_id"] == "SCH_0006"].iloc[0]
        self.assertEqual(sch6["verification_status"], "INACTIVE")

        # Partitioning
        verified_df, pending_df, inactive_df = apply_verification_decisions(df, updated_review)
        self.assertEqual(len(verified_df), 1)
        self.assertEqual(len(inactive_df), 1)
        self.assertEqual(len(pending_df), 4)
        self.assertEqual(verified_df.iloc[0]["record_id"], "SCH_0001")

    # --- Preprocessing tests ---

    def test_clean_text_normalizes_spaces_and_preserves_content(self):
        raw = "  Eligible for small farmers\n\twith land < 2 hectares.   Not eligible for institutional landholders.  "
        cleaned = clean_text(raw)
        self.assertEqual(cleaned, "Eligible for small farmers with land < 2 hectares. Not eligible for institutional landholders.")
        self.assertIn("Not eligible", cleaned)

    def test_standardize_level(self):
        self.assertEqual(standardize_level("central"), "Central Government")
        self.assertEqual(standardize_level("Central Government"), "Central Government")
        self.assertEqual(standardize_level("state"), "State / UT Government")
        self.assertEqual(standardize_level("State / UT Government"), "State / UT Government")
        self.assertEqual(standardize_level("State"), "State / UT Government")

    def test_preprocessing_search_text_and_no_none_or_nan(self):
        # Prepare a mock verified row
        mock_verified = pd.DataFrame([{
            "record_id": "SCH_0001",
            "scheme_name": " PM Kisan ",
            "description": "Income support for small farmers.\nAnnual grant of 6000 rupees. ",
            "benefits": "6000 rupees direct transfer.",
            "eligibility": "Small and marginal farmers only; not eligible if income tax payer.",
            "application_process": "Apply via official portal.",
            "department": "Agriculture",
            "level": "central",
            "source_url": "https://pmkisan.gov.in",
            "verification_status": "VERIFIED",
        }])

        cleaned_df, excluded = preprocess_verified(mock_verified)
        self.assertEqual(len(cleaned_df), 1)
        self.assertEqual(len(excluded), 0)

        row = cleaned_df.iloc[0]
        self.assertEqual(row["level"], "Central Government")
        self.assertIn("search_text", cleaned_df.columns)

        search_txt = row["search_text"]
        # Must not contain string literals like "nan" or "None"
        self.assertNotIn("nan", search_txt.lower())
        self.assertNotIn("none", search_txt.lower())
        # Must preserve negations and criteria
        self.assertIn("not eligible", search_txt.lower())
        self.assertIn("income tax payer", search_txt.lower())

    # --- End-to-end run_pipeline test on temporary workspace ---

    def test_run_pipeline_end_to_end_on_fixture(self):
        # Create a mini project layout in temp dir
        temp_raw = self.dir_path / "data" / "raw" / "gov_myscheme_data.csv"
        temp_raw.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(pd.read_csv(self.fixture_csv)).to_csv(temp_raw, index=False)

        # Run pipeline pointing to temp fixture
        # Patch paths temporarily in main module
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
            main.SCHEME_REVIEW_PATH = main.REPORTS_DIR / "scheme_review.csv"
            main.VERIFIED_PATH = main.PROCESSED_DIR / "verified_schemes.csv"
            main.PENDING_PATH = main.PROCESSED_DIR / "pending_schemes.csv"
            main.INACTIVE_PATH = main.PROCESSED_DIR / "inactive_schemes.csv"
            main.CLEAN_VERIFIED_PATH = main.PROCESSED_DIR / "verified_schemes_clean.csv"

            counts = run_pipeline(raw_path=temp_raw, check_urls=False)
            self.assertEqual(counts["raw_records"], 6)
            self.assertEqual(counts["verified_records"], 0)  # Honest unreviewed state
            self.assertEqual(counts["pending_records"], 6)
            self.assertTrue(main.AUDIT_SUMMARY_PATH.exists())
            self.assertTrue(main.RECORD_ISSUES_PATH.exists())
            self.assertTrue(main.SCHEME_REVIEW_PATH.exists())
            self.assertTrue(main.VERIFIED_PATH.exists())
            self.assertTrue(main.PENDING_PATH.exists())
            self.assertTrue(main.INACTIVE_PATH.exists())
            self.assertTrue(main.CLEAN_VERIFIED_PATH.exists())
        finally:
            main.RAW_DATA_PATH = orig_raw
            main.PROCESSED_DIR = orig_proc
            main.REPORTS_DIR = orig_rep
            main.AUDIT_SUMMARY_PATH = orig_rep / "audit_summary.csv"
            main.RECORD_ISSUES_PATH = orig_rep / "record_issues.csv"
            main.SCHEME_REVIEW_PATH = orig_rep / "scheme_review.csv"
            main.VERIFIED_PATH = orig_proc / "verified_schemes.csv"
            main.PENDING_PATH = orig_proc / "pending_schemes.csv"
            main.INACTIVE_PATH = orig_proc / "inactive_schemes.csv"
            main.CLEAN_VERIFIED_PATH = orig_proc / "verified_schemes_clean.csv"


if __name__ == "__main__":
    unittest.main()
