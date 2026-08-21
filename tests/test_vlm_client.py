from __future__ import annotations

import json

import httpx

from medical_evaluation.vlm.client import QwenVlmClient
from medical_evaluation.vlm.schemas import VlmReviewRequest


def make_review_request() -> VlmReviewRequest:
    return VlmReviewRequest(
        checkpoint_id="cp_09",
        checkpoint_name="安装支架",
        criteria=["支架居中"],
        deterministic_status="incorrect",
        reason_code="frame_not_centered",
        features={"frame_center_offset": 0.2},
    )


def test_valid_structured_response_is_returned_without_score_override() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        content = json.dumps(
            {
                "evidence_supported": True,
                "semantic_status": "supports",
                "reason_zh": "支架明显偏离中心。",
                "suggestion_zh": "调整支架四周张力。",
                "cited_evidence_indices": [],
            },
            ensure_ascii=False,
        )
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    client = QwenVlmClient(
        "http://local/v1",
        "Qwen3-VL-4B-Instruct",
        transport=httpx.MockTransport(handler),
    )

    result = client.review(make_review_request())

    assert calls == 1
    assert result.source == "qwen"
    assert result.score_override is None


def test_invalid_json_retries_once_then_returns_template() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"choices": [{"message": {"content": "not-json"}}]})

    client = QwenVlmClient(
        "http://local/v1",
        "Qwen3-VL-4B-Instruct",
        transport=httpx.MockTransport(handler),
    )

    result = client.review(make_review_request())

    assert calls == 2
    assert result.source == "template_fallback"
    assert result.score_override is None
    assert "支架" in result.reason_zh


def test_forbidden_score_field_is_rejected_and_falls_back() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        content = json.dumps(
            {
                "evidence_supported": True,
                "semantic_status": "supports",
                "reason_zh": "结论",
                "suggestion_zh": "建议",
                "cited_evidence_indices": [],
                "score": 100,
            }
        )
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    client = QwenVlmClient(
        "http://local/v1",
        "Qwen3-VL-4B-Instruct",
        transport=httpx.MockTransport(handler),
    )

    assert client.review(make_review_request()).source == "template_fallback"


def test_service_down_returns_template_after_retry() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("service unavailable", request=request)

    client = QwenVlmClient(
        "http://local/v1",
        "Qwen3-VL-4B-Instruct",
        transport=httpx.MockTransport(handler),
    )

    result = client.review(make_review_request())

    assert calls == 2
    assert result.source == "template_fallback"


def test_only_configured_maximum_number_of_images_is_sent(tmp_path) -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        content = json.dumps(
            {
                "evidence_supported": False,
                "semantic_status": "uncertain",
                "reason_zh": "证据不足",
                "suggestion_zh": "人工复核",
                "cited_evidence_indices": [],
            }
        )
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    images = []
    for index in range(4):
        path = tmp_path / f"{index}.jpg"
        path.write_bytes(b"jpeg")
        images.append(path)
    request = make_review_request().model_copy(update={"evidence_images": images})
    client = QwenVlmClient(
        "http://local/v1",
        "Qwen3-VL-4B-Instruct",
        max_images=2,
        transport=httpx.MockTransport(handler),
    )

    client.review(request)

    content = captured["messages"][1]["content"]
    assert len([item for item in content if item["type"] == "image_url"]) == 2
