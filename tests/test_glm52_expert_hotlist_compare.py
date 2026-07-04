import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

from benchmarks import glm52_expert_hotlist_compare as compare


class TestGlm52ExpertHotlistCompare(unittest.TestCase):
    def test_parse_ds4_c_initializer(self):
        entries = compare.parse_hotlist_text(
            """
            /* Generated. */
            static const uint16_t ds4_default_streaming_hotlist_glm52[][2] = {
                {50, 118},
                {72, 53},
                {50, 118},
            };
            """
        )

        self.assertEqual([entry.pair for entry in entries], [(50, 118), (72, 53)])
        self.assertEqual([entry.rank for entry in entries], [1, 2])
        self.assertIsNone(entries[0].hits)
        self.assertIsNone(entries[0].weight)

    def test_parse_mlx_text_hotlist(self):
        entries = compare.parse_hotlist_text(
            """
            # mlx_lm GLM DSA expert hotlist v1
            # columns: layer expert hits weight
            1 3 7 2.5
            1 4 2 0.25
            """
        )

        self.assertEqual([entry.pair for entry in entries], [(1, 3), (1, 4)])
        self.assertEqual(entries[0].hits, 7)
        self.assertEqual(entries[0].weight, 2.5)

    def test_compare_hotlists_reports_top_n_overlap(self):
        left = compare.parse_hotlist_text(
            """
            1 3 7 2.5
            1 4 2 0.25
            2 8 1 0.1
            """
        )
        right = compare.parse_hotlist_text(
            """
            {1, 4},
            {1, 3},
            {9, 9},
            """
        )

        summary = compare.compare_hotlists(left, right, top_ns=[1, 2, 3])
        self.assertEqual(summary["left_entries"], 3)
        self.assertEqual(summary["right_entries"], 3)
        self.assertEqual(summary["rows"][0]["overlap"], 0)
        self.assertEqual(summary["rows"][1]["overlap"], 2)
        self.assertEqual(summary["rows"][1]["jaccard"], 1.0)
        self.assertEqual(summary["rows"][2]["overlap"], 2)
        self.assertIn("top_n\tleft_count", compare.format_table(summary))

    def test_main_writes_json_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            left = tmp_path / "left.hotlist"
            right = tmp_path / "right.inc"
            out = tmp_path / "summary.json"
            left.write_text("1 3 7 2.5\n", encoding="utf-8")
            right.write_text("{1, 3},\n", encoding="utf-8")

            old_argv = sys.argv
            sys.argv = [
                "glm52_expert_hotlist_compare.py",
                "--left",
                str(left),
                "--right",
                str(right),
                "--top-ns",
                "1",
                "--json-output",
                str(out),
            ]
            try:
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    compare.main()
                self.assertIn("top_n\tleft_count", stdout.getvalue())
            finally:
                sys.argv = old_argv

            loaded = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(loaded["rows"][0]["overlap"], 1)


if __name__ == "__main__":
    unittest.main()
