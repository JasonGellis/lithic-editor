"""Tests for the evaluation harness: registration, CSV, report, and argument handling."""

import numpy as np

from tools.dpi_eval import metrics, report
from tools.dpi_eval.run import parse_args
from tools.dpi_eval.strategies import STRATEGIES


def _drawing_mask(shape=(300, 260)):
    mask = np.zeros(shape, dtype=bool)
    mask[40:260, 40:46] = mask[40:260, 214:220] = True
    mask[40:46, 40:220] = mask[254:260, 40:220] = True
    mask[120:126, 60:200] = True
    return mask


class TestRegistration:
    def test_shift_is_recovered(self):
        reference = _drawing_mask()
        shifted = np.roll(np.roll(reference, 7, axis=0), -5, axis=1)
        warp, cc = metrics.estimate_registration(shifted, reference)
        aligned = metrics.apply_registration(shifted, warp, reference.shape)
        _, _, f1 = metrics.tolerance_scores(metrics.skeleton(aligned), metrics.skeleton(reference), 2)
        assert cc > 0.9
        assert f1 > 0.95

    def test_fit_to_shape_pads_and_crops(self):
        mask = np.ones((10, 10), dtype=bool)
        padded = metrics.fit_to_shape(mask, (15, 12))
        cropped = metrics.fit_to_shape(mask, (5, 5))
        assert padded.shape == (15, 12) and padded[:10, :10].all() and not padded[10:, :].any()
        assert cropped.shape == (5, 5) and cropped.all()


class TestReport:
    def _rows(self):
        return [
            {"source": "a", "strategy": "native", "dpi": 600, "status": "ok", "runtime_s": 1.0,
             "output_width": 10, "output_height": 10, "residual_ripples": 3},
            {"source": "a", "strategy": "adaptive", "dpi": 300, "status": "ok", "runtime_s": 0.5,
             "output_width": 10, "output_height": 10, "residual_ripples": 2,
             "ref_precision": 0.9, "ref_recall": 0.8, "ref_f1": 0.85, "ref_structural_loss": 0.1,
             "ref_cortex_ratio": float("nan")},
            {"source": "a", "strategy": "adaptive", "dpi": 150, "status": "error", "error": "boom"},
        ]

    def test_csv_round_trip(self, tmp_path):
        path = tmp_path / "summary.csv"
        report.write_csv(self._rows(), path)
        rows = report.read_csv(path)
        assert [r["strategy"] for r in rows] == ["native", "adaptive", "adaptive"]
        assert rows[1]["ref_f1"] == 0.85
        assert np.isnan(rows[1]["ref_cortex_ratio"])
        assert rows[2]["status"] == "error" and rows[2]["error"] == "boom"

    def test_summarize_skips_errors_and_nans(self):
        summary = report.summarize(self._rows(), "ref")
        assert summary[("adaptive", 300)]["f1"] == 0.85
        assert "cortex_ratio" not in summary[("adaptive", 300)]
        assert ("adaptive", 150) not in summary

    def test_report_names_reference_and_failures(self, tmp_path):
        text = report.write_report(self._rows(), tmp_path / "report.md", ["native", "adaptive"],
                                   [600, 300, 150], None, "native")
        assert "`native` at 600 DPI" in text
        assert "| adaptive | " in text
        assert "Cells that did not run" in text and "boom" in text
        assert "Ground truth" not in text

    def test_montage_has_readable_labels(self, tmp_path):
        path = tmp_path / "montage.png"
        report.write_montage([("input 600", _drawing_mask()), ("native", _drawing_mask())], path)
        from PIL import Image
        with Image.open(path) as im:
            assert im.size[0] > 0 and im.size[1] > 300
            bar = np.array(im)[:report._LABEL_HEIGHT]
            assert (bar < 100).any(), "no label text drawn"


class TestArguments:
    def test_reference_strategy_runs_first(self):
        args = parse_args(["--quick", "--strategies", "adaptive", "resample", "--reference-strategy", "native"])
        assert args.strategies[0] == "native"
        assert set(args.strategies) == {"native", "adaptive", "resample"}
        assert args.sources == ["369", "371"]

    def test_dpis_are_sorted_descending_and_include_the_source(self):
        args = parse_args(["--dpis", "75", "600", "150"])
        assert args.dpis == [600, 150, 75]

    def test_config_option_sets_environment(self, tmp_path, monkeypatch):
        import os
        monkeypatch.delenv("LITHIC_EDITOR_CONFIG", raising=False)
        path = tmp_path / "c.yaml"
        path.write_text("smoothing:\n  enabled: false\n")
        parse_args(["--quick", "--config", str(path)])
        assert os.environ["LITHIC_EDITOR_CONFIG"] == str(path.resolve())

    def test_all_strategies_share_the_signature(self):
        import inspect
        for name, fn in STRATEGIES.items():
            params = list(inspect.signature(fn).parameters)
            assert params == ["variant_path", "dpi", "workdir"], name
