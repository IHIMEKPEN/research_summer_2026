from s2r.models.qwen_reasoner import QwenReasoner
from s2r.models.qwen_vl import QwenVLBackend
from s2r.models.yolo_detector import YOLODetector, encode_image_stub
from s2r.models.registry import build_detector, build_reasoner


def test_yolo_mock_detects_phased_objects():
    det = YOLODetector(mock=True)
    frame = det.infer(encode_image_stub())
    assert frame.backend.endswith("mock") or "yolo" in frame.backend
    assert isinstance(frame.caption, str)


def test_qwen_reasoner_mock_plan():
    r = QwenReasoner(mock=True)
    from s2r.models.base import PerceptionFrame, Detection

    out = r.plan(
        "bring me a pen",
        PerceptionFrame(objects_of_interest=["pen"], detections=[Detection("pen", 0.9, [0, 0, 1, 1])], caption="pen"),
        {"holding_pen": False},
        "locate",
    )
    assert out.intent in {"approach_table", "grasp_pen", "locate_pen", "explore"}
    assert out.allow_motion in {True, False}


def test_qwen_vl_mock():
    vlm = QwenVLBackend(mock=True)
    out = vlm.infer(encode_image_stub(), "find pen")
    assert out.caption


def test_registry_builds_mock_from_config():
    cfg = {
        "models": {
            "detector": {"mock": True, "backend": "yolo"},
            "reasoner": {"mock": True, "backend": "qwen"},
        }
    }
    assert build_detector(cfg).infer(encode_image_stub()).caption
    assert build_reasoner(cfg).plan("x", None).intent
