from __future__ import annotations

import json

import httpx
import pytest

from medical_evaluation.vlm.client import QwenVlmClient, template_fallback
from medical_evaluation.vlm.schemas import VlmReviewRequest


def make_review_request() -> VlmReviewRequest:
    return VlmReviewRequest(
        checkpoint_id="cp_09",
        checkpoint_name="安装支架",
        criteria=["支架居中"],
        deterministic_status="incorrect",
        reason_code="frame_not_centered_on_oral_region",
        features={"frame_oral_center_offset": 0.2},
    )


def test_fallback_messages_match_current_cp02_cp09_cp10_cp11_rules() -> None:
    assert "第二大" in template_fallback("wrong_punch_hole_selected").reason_zh
    residue = template_fallback("residue_not_cleaned")
    assert "CP02最后可靠可见状态" in residue.reason_zh
    assert "绿色残留" in residue.reason_zh
    assert "稳定" in template_fallback("frame_not_stabilized").reason_zh
    assert "牙线" in template_fallback("dental_floss_not_observed").reason_zh
    cp11 = template_fallback("final_position_incorrect")
    assert "白色支架" in cp11.reason_zh
    assert "鼻部" in cp11.reason_zh
    assert "口鼻" not in cp11.reason_zh + cp11.suggestion_zh


def test_cp08_commentary_uses_shape_proxy_without_claiming_tip_type() -> None:
    review = template_fallback("criteria_satisfied", checkpoint_id="cp_08")

    assert "符合约定外形" in review.reason_zh
    assert "长柄细杆弯曲端" in review.reason_zh
    assert "钝头" not in review.reason_zh + review.suggestion_zh
    assert "尖锐" not in review.reason_zh + review.suggestion_zh


def test_cp08_qwen_request_forbids_blunt_or_sharp_claims() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        content = json.dumps(
            {
                "evidence_supported": True,
                "semantic_status": "supports",
                "reason_zh": "符合约定外形，长柄细杆弯曲端证据一致。",
                "suggestion_zh": "继续保持规范操作。",
                "cited_evidence_indices": [],
            },
            ensure_ascii=False,
        )
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    request = make_review_request().model_copy(
        update={"checkpoint_id": "cp_08", "reason_code": "criteria_satisfied"}
    )
    client = QwenVlmClient(
        "http://local/v1",
        "Qwen3-VL-4B-Instruct",
        transport=httpx.MockTransport(handler),
    )

    client.review(request)

    details = captured["messages"][1]["content"][0]["text"]
    assert "符合约定外形" in details
    assert "长柄细杆弯曲端" in details
    assert "不能声称钝头或尖锐" in details
    for forbidden_claim in ("圆钝", "钝性", "尖头", "锐利", "针尖", "探针"):
        assert forbidden_claim in details


@pytest.mark.parametrize(
    "tip_claim",
    ["钝头", "尖锐", "圆钝", "钝性", "尖头", "锐利", "针尖", "探针"],
)
def test_cp08_qwen_tip_claim_is_rejected_and_uses_safe_fallback(
    tip_claim: str,
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        content = json.dumps(
            {
                "evidence_supported": True,
                "semantic_status": "supports",
                "reason_zh": (
                    "器具符合约定外形，长柄细杆弯曲端证据一致；"
                    f"已经证明使用的是{tip_claim}器具。"
                ),
                "suggestion_zh": f"继续使用{tip_claim}器具。",
                "cited_evidence_indices": [],
            },
            ensure_ascii=False,
        )
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    request = make_review_request().model_copy(
        update={"checkpoint_id": "cp_08", "reason_code": "criteria_satisfied"}
    )
    client = QwenVlmClient(
        "http://local/v1",
        "Qwen3-VL-4B-Instruct",
        transport=httpx.MockTransport(handler),
    )

    result = client.review(request)

    assert result.source == "template_fallback"
    assert "符合约定外形" in result.reason_zh
    assert "长柄细杆弯曲端" in result.reason_zh
    assert tip_claim not in result.reason_zh + result.suggestion_zh


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


def test_system_prompt_scopes_review_to_rubber_dam_isolation() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        content = json.dumps(
            {
                "evidence_supported": True,
                "semantic_status": "supports",
                "reason_zh": "结论",
                "suggestion_zh": "建议",
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

    client.review(make_review_request())

    assert captured["messages"][0]["content"] == (
        "你是牙科操作考核的证据点评助手，现在需要对橡皮障隔离技术相关操作进行点评。\n"
        "只能使用请求中给出的考核标准、确定性规则结论、特征和证据图。\n"
        "必须引用使用过的证据图索引；证据不足时明确说明不确定。\n"
        "只能返回符合给定字段的 JSON，不要 Markdown，不要额外字段。\n"
        "绝对不能给出、修改或建议任何分数；确定性规则结论不可被覆盖。"
    )
    response_format = captured["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "vlm_review"
    schema = response_format["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    assert {
        "evidence_supported",
        "semantic_status",
        "reason_zh",
        "suggestion_zh",
        "cited_evidence_indices",
    } <= set(schema["required"])


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


def test_missing_evidence_image_returns_template_without_aborting(tmp_path) -> None:
    request = make_review_request().model_copy(
        update={"evidence_images": [tmp_path / "missing.jpg"]}
    )
    client = QwenVlmClient("http://local/v1", "Qwen3-VL-4B-Instruct")

    result = client.review(request)

    assert result.source == "template_fallback"
    assert result.score_override is None
