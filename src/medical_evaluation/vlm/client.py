from __future__ import annotations

import base64
import hashlib
import json
import logging
import mimetypes
from pathlib import Path

import httpx
from pydantic import ValidationError

from medical_evaluation.vlm.schemas import VlmReview, VlmReviewRequest

LOGGER = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是牙科操作考核的证据点评助手，现在需要对橡皮障隔离技术相关操作进行点评。
只能使用请求中给出的考核标准、确定性规则结论、特征和证据图。
必须引用使用过的证据图索引；证据不足时明确说明不确定。
只能返回符合给定字段的 JSON，不要 Markdown，不要额外字段。
绝对不能给出、修改或建议任何分数；确定性规则结论不可被覆盖。"""


class QwenVlmClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        timeout_sec: float = 30,
        max_images: int = 3,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if timeout_sec <= 0:
            raise ValueError("timeout_sec must be positive")
        if max_images < 0:
            raise ValueError("max_images must be non-negative")
        self.endpoint = f"{base_url.rstrip('/')}/chat/completions"
        self.model = model
        self.max_images = max_images
        self.client = httpx.Client(timeout=timeout_sec, transport=transport)

    def review(self, request: VlmReviewRequest) -> VlmReview:
        selected_images = request.evidence_images[: self.max_images]
        for attempt in range(2):
            try:
                response = self.client.post(
                    self.endpoint,
                    json=self._payload(request, selected_images, repair=attempt == 1),
                )
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                review = VlmReview.model_validate_json(content)
                if any(
                    index < 0 or index >= len(selected_images)
                    for index in review.cited_evidence_indices
                ):
                    raise ValueError("response cites an unavailable evidence image")
                digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
                LOGGER.info("local VLM review model=%s response_sha256=%s", self.model, digest)
                return review
            except (
                OSError,
                httpx.HTTPError,
                KeyError,
                TypeError,
                ValueError,
                ValidationError,
            ) as exc:
                LOGGER.warning(
                    "local VLM review attempt failed model=%s attempt=%d error=%s",
                    self.model,
                    attempt + 1,
                    type(exc).__name__,
                )
        return template_fallback(request.reason_code)

    def _payload(
        self,
        request: VlmReviewRequest,
        images: list[Path],
        *,
        repair: bool,
    ) -> dict[str, object]:
        details = {
            "checkpoint_id": request.checkpoint_id,
            "checkpoint_name": request.checkpoint_name,
            "criteria": request.criteria,
            "deterministic_status": request.deterministic_status,
            "reason_code": request.reason_code,
            "features": request.features,
            "evidence_indices": list(range(len(images))),
        }
        instruction = "请严格返回 JSON。"
        if repair:
            instruction = "上一次响应未通过 JSON 模式校验。请修复格式，只返回合法 JSON。"
        content: list[dict[str, object]] = [
            {
                "type": "text",
                "text": f"{instruction}\n输入证据：{json.dumps(details, ensure_ascii=False)}",
            }
        ]
        content.extend(
            {
                "type": "image_url",
                "image_url": {"url": _image_data_url(path)},
            }
            for path in images
        )
        return {
            "model": self.model,
            "temperature": 0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "vlm_review",
                    "strict": True,
                    "schema": VlmReview.model_json_schema(),
                },
            },
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
        }


def _image_data_url(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def template_fallback(reason_code: str) -> VlmReview:
    templates = {
        "criteria_satisfied": (
            "现有证据支持确定性规则的通过结论。",
            "继续保持当前规范操作。",
        ),
        "missing_frame_oral_evidence": (
            "支架相对口腔参考区域的位置证据不足。",
            "请补充清晰支架提示和唯一口腔参考框后复核。",
        ),
        "frame_not_centered_on_oral_region": (
            "支架位置未达到居中要求。",
            "检查支架四周张力后重新调整至中央。",
        ),
        "unreliable_stage_presence": (
            "无法可靠确认本阶段是否出现橡皮布。",
            "请检查阶段时间范围和视频清晰度。",
        ),
        "rubber_dam_not_observed": (
            "本阶段未观察到橡皮布，操作没有完成。",
            "请完成橡皮布调整并将其撑开至支架。",
        ),
        "unreliable_final_presence": (
            "无法可靠确认阶段末尾的橡皮布状态。",
            "请提供末尾清晰且无遮挡的证据帧。",
        ),
        "rubber_dam_missing_at_end": (
            "阶段中出现过橡皮布，但末尾没有保持在位。",
            "重新完成末尾调整并确认橡皮布稳定覆盖支架。",
        ),
        "missing_required_evidence": (
            "末尾橡皮布、鼻部或支架可见性证据不足。",
            "请补充能够同时观察这些区域的清晰末尾证据。",
        ),
        "final_position_incorrect": (
            "橡皮布最终位置未同时满足规则要求。",
            "重新调整游离缘，保证口鼻无遮挡并充分撑开。",
        ),
    }
    reason, suggestion = templates.get(
        reason_code,
        ("本地点评模型不可用，保留确定性规则结论。", "请根据证据图和特征进行人工复核。"),
    )
    return VlmReview(
        evidence_supported=False,
        semantic_status="uncertain",
        reason_zh=reason,
        suggestion_zh=suggestion,
        cited_evidence_indices=[],
        source="template_fallback",
    )
