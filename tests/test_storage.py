import json
from pathlib import Path

import pytest


def test_safe_child_rejects_traversal(tmp_path: Path) -> None:
    from medical_evaluation.storage import safe_child

    with pytest.raises(ValueError, match="outside storage root"):
        safe_child(tmp_path, "../escape.json")


def test_atomic_write_json_replaces_complete_document(tmp_path: Path) -> None:
    from medical_evaluation.storage import atomic_write_json

    path = tmp_path / "nested" / "item.json"
    atomic_write_json(path, {"version": 1})
    atomic_write_json(path, {"version": 2, "text": "中文"})

    assert json.loads(path.read_text(encoding="utf-8")) == {"version": 2, "text": "中文"}
    assert list(path.parent.glob("*.tmp")) == []
