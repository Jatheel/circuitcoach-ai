"""Unit tests for the pure OpenCV helpers in scripts/live_detect.py (no model, no camera).

Run: pytest tests/test_live_detect.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from live_detect import (FpsMeter, UniqueCounter, apply_clahe, class_color,  # noqa: E402
                         draw_detections, extract_dets, is_live, parse_source, sharpness)


def checkerboard(h=480, w=640, cell=16):
    y, x = np.indices((h, w))
    g = (((x // cell) + (y // cell)) % 2 * 255).astype(np.uint8)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def test_sharp_scores_higher_than_blurred():
    img = checkerboard()
    blurred = cv2.GaussianBlur(img, (21, 21), 6)
    assert sharpness(img) > 10 * sharpness(blurred)


def test_sharpness_is_resolution_normalised():
    img = checkerboard(480, 640, 16)
    big = cv2.resize(img, (1280, 960), interpolation=cv2.INTER_NEAREST)
    assert sharpness(big) == pytest.approx(sharpness(img), rel=0.25)


def test_flat_image_has_zero_sharpness():
    assert sharpness(np.full((100, 100, 3), 128, np.uint8)) == 0.0


def test_clahe_keeps_shape_and_dtype_and_boosts_contrast():
    rng = np.random.default_rng(0)
    dark = rng.integers(40, 70, (240, 320, 3), dtype=np.uint8)  # dim, low-contrast frame
    out = apply_clahe(dark)
    assert out.shape == dark.shape and out.dtype == np.uint8
    assert out.std() > dark.std()


def test_unique_counter_needs_min_hits_and_counts_once():
    c = UniqueCounter(min_hits=3)
    for _ in range(2):
        c.update([1], ["Resistor"])
    assert c.total() == 0  # seen twice, not yet counted
    for _ in range(10):
        c.update([1], ["Resistor"])
    assert c.counts() == {"Resistor": 1}


def test_unique_counter_ignores_none_and_separates_classes():
    c = UniqueCounter(min_hits=1)
    c.update([None, None], ["LED", "LED"])
    assert c.total() == 0
    c.update([1, 2, 3], ["LED", "LED", "Buzzer"])
    assert c.counts() == {"LED": 2, "Buzzer": 1}
    assert list(c.counts()) == ["LED", "Buzzer"]  # most common first


def test_unique_counter_keeps_first_counted_class():
    c = UniqueCounter(min_hits=1)
    c.update([7], ["Diode"])
    c.update([7], ["Zener-Diode"])  # same track relabelled later
    assert c.counts() == {"Diode": 1}


def test_fps_meter_constant_rate():
    m = FpsMeter(alpha=0.5)
    assert m.tick(0.0) is None
    for i in range(1, 20):
        fps = m.tick(i * 0.05)
    assert fps == pytest.approx(20.0)


def test_fps_meter_smooths_a_spike():
    m = FpsMeter(alpha=0.1)
    m.tick(0.0); m.tick(0.1)  # 10 fps
    assert m.tick(0.101) < 200  # one 1 ms frame does not jump to 1000 fps


def test_parse_source_and_is_live():
    assert parse_source("0") == 0 and is_live(0)
    assert parse_source("clip.mp4") == "clip.mp4" and not is_live("clip.mp4")
    assert is_live("http://192.168.1.20:4747/video")


def test_class_colors_are_valid_and_distinct():
    cols = [class_color(i) for i in range(61)]
    assert all(all(0 <= v <= 255 for v in c) for c in cols)
    assert len(set(cols)) == 61


def test_draw_detections_draws_inside_box_and_handles_edges():
    img = np.zeros((200, 300, 3), np.uint8)
    dets = [(50, 60, 150, 160, 0.9, 3, 12), (0, 0, 40, 30, 0.5, 5, None),
            (280, 180, 299, 199, 0.45, 7, 4)]
    names = {i: f"class{i}" for i in range(61)}
    out = draw_detections(img, dets, names)
    assert out is img and img.sum() > 0
    assert tuple(int(v) for v in img[110, 50]) == class_color(3)  # left box edge


class _T:  # tiny stand-in for a torch tensor
    def __init__(self, a): self.a = np.asarray(a)
    def cpu(self): return self
    def numpy(self): return self.a


class _Boxes:
    def __init__(self, xyxy, conf, cls, ids):
        self.xyxy, self.conf, self.cls = _T(xyxy), _T(conf), _T(cls)
        self.id = None if ids is None else _T(ids)
    def __len__(self): return len(self.conf.a)


class _Result:
    def __init__(self, boxes): self.boxes = boxes


def test_extract_dets_filters_conf_and_handles_missing_ids():
    b = _Boxes([[0, 0, 10, 10], [5, 5, 20, 20]], [0.9, 0.2], [1, 2], None)
    d = extract_dets(_Result(b), 0.4)
    assert len(d) == 1 and d[0][4] == pytest.approx(0.9) and d[0][5] == 1 and d[0][6] is None
    b = _Boxes([[0, 0, 10, 10]], [0.8], [3], [42])
    assert extract_dets(_Result(b), 0.4)[0][6] == 42
    assert extract_dets(_Result(None), 0.4) == []
