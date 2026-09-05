import io
import unittest
import zipfile

import pandas as pd

from gsc_analysis import analyse_bundle, parse_ctr, prepare_queries, read_gsc_zip


def make_export(include_dates=True, empty_pages=False):
    queries = pd.DataFrame({
        "Top queries": ["seo audit", "technical seo audit", "seo consultant"],
        "Clicks": [5, 2, 1],
        "Impressions": [100, 80, 60],
        "CTR": ["5%", "2.5%", "1.67%"],
        "Position": [5, 12, 18],
    })
    pages = pd.DataFrame({
        "Top pages": [] if empty_pages else ["https://example.com/seo-audit/", "https://example.com/seo-consultant/"],
        "Clicks": [] if empty_pages else [7, 1],
        "Impressions": [] if empty_pages else [180, 60],
        "CTR": [] if empty_pages else ["3.89%", "1.67%"],
        "Position": [] if empty_pages else [8, 18],
    })
    dates = pd.DataFrame({
        "Date": pd.date_range("2026-01-01", periods=6),
        "Clicks": [1, 2, 2, 3, 4, 5],
        "Impressions": [10, 12, 15, 20, 25, 30],
        "CTR": ["10%"] * 6,
        "Position": [10] * 6,
    })
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("Queries.csv", queries.to_csv(index=False))
        archive.writestr("Pages.csv", pages.to_csv(index=False))
        if include_dates:
            archive.writestr("Dates.csv", dates.to_csv(index=False))
    return buffer.getvalue()


class AnalysisTests(unittest.TestCase):
    def test_ctr_formats(self):
        self.assertAlmostEqual(parse_ctr("12.5%"), 0.125)
        self.assertAlmostEqual(parse_ctr(12.5), 0.125)
        self.assertAlmostEqual(parse_ctr(0.125), 0.125)

    def test_zip_is_detected_by_columns(self):
        bundle = read_gsc_zip(make_export())
        self.assertEqual(set(bundle.tables), {"queries", "pages", "dates"})

    def test_nominal_analysis_builds_work_queues(self):
        analysis = analyse_bundle(read_gsc_zip(make_export()), ["example"])
        self.assertEqual(len(analysis["queries"]), 3)
        self.assertTrue(analysis["trend"]["available"])
        self.assertIn("quick_wins", analysis["queues"])

    def test_empty_page_export_has_clear_error(self):
        with self.assertRaisesRegex(ValueError, "page table contains no usable rows"):
            analyse_bundle(read_gsc_zip(make_export(empty_pages=True)), [])

    def test_invalid_zip_has_clear_error(self):
        with self.assertRaisesRegex(ValueError, "not a valid ZIP"):
            read_gsc_zip(b"not a zip")

    def test_business_signals_work_across_industries(self):
        queries = pd.DataFrame({
            "query": [
                "washing machine not working",
                "are solar panels worth it",
                "hubspot vs salesforce",
                "increase restaurant bookings",
                "blue cotton shirts",
            ],
            "clicks": [1, 1, 1, 1, 1],
            "impressions": [10, 10, 10, 10, 10],
            "ctr": ["10%"] * 5,
            "position": [10] * 5,
        })
        result = prepare_queries(queries, [])
        signals = dict(zip(result["query"], result["business_signal"]))
        self.assertEqual(signals["washing machine not working"], "Problem")
        self.assertEqual(signals["are solar panels worth it"], "Objection")
        self.assertEqual(signals["hubspot vs salesforce"], "Comparison")
        self.assertEqual(signals["increase restaurant bookings"], "Desired outcome")
        self.assertEqual(signals["blue cotton shirts"], "Other")

    def test_overlapping_and_custom_business_signals_are_retained(self):
        queries = pd.DataFrame({
            "query": ["best affordable crm", "prevent stockouts"],
            "clicks": [1, 1], "impressions": [10, 10], "ctr": ["10%", "10%"], "position": [10, 10],
        })
        custom = {"Problem": ["stockouts"]}
        result = prepare_queries(queries, [], custom).set_index("query")
        self.assertEqual(result.loc["best affordable crm", "business_signal"], "Comparison")
        self.assertEqual(result.loc["best affordable crm", "all_business_signals"], "Comparison | Objection")
        self.assertIn("Problem", result.loc["prevent stockouts", "all_business_signals"])
        self.assertIn("Desired outcome", result.loc["prevent stockouts", "all_business_signals"])


if __name__ == "__main__":
    unittest.main()
