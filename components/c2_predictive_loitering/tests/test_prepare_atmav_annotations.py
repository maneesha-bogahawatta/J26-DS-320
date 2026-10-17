"""Tests for the ATMA-V temporal annotation parser."""

import unittest

from components.c2_predictive_loitering.data_processing.prepare_atmav_annotations import (
    parse_annotation_line,
)


class TestAtmavAnnotationParser(unittest.TestCase):
    def test_normal_video(self):
        record = parse_annotation_line("12.mp4 910 24 -1 -1")

        self.assertEqual(record["video_name"], "12.mp4")
        self.assertEqual(record["total_frames"], 910)
        self.assertEqual(record["fps"], 24.0)
        self.assertEqual(record["intervals"], [])
        self.assertEqual(record["annotation_warning"], "")

    def test_single_anomaly_interval(self):
        record = parse_annotation_line("3.mp4 737 30 1 737")

        self.assertEqual(record["intervals"], [(1, 737)])
        self.assertEqual(record["annotation_warning"], "")

    def test_multiple_anomaly_intervals(self):
        record = parse_annotation_line(
            "28.mp4 945 25 92 321 807 945"
        )

        self.assertEqual(
            record["intervals"],
            [(92, 321), (807, 945)],
        )

    def test_incomplete_final_interval_is_flagged(self):
        record = parse_annotation_line(
            "93.mp4 2973 1 82 249 492 1129 2600"
        )

        self.assertEqual(
            record["intervals"],
            [(82, 249), (492, 1129)],
        )
        self.assertIn("2600", record["annotation_warning"])

    def test_rejects_reversed_interval(self):
        with self.assertRaises(ValueError):
            parse_annotation_line("sample.mp4 500 25 300 100")

    def test_rejects_incomplete_line(self):
        with self.assertRaises(ValueError):
            parse_annotation_line("sample.mp4 500 25")


if __name__ == "__main__":
    unittest.main()