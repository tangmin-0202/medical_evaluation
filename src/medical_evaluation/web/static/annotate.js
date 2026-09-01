const root = document.querySelector("#annotation-app");
const videoId = root.dataset.videoId;
const annotationGuides = JSON.parse(document.querySelector("#annotation-guides").textContent);
let annotations;
let dragStart = null;

function applyPromptGuide(checkpointId) {
  const guide = annotationGuides[checkpointId];
  if (!guide) return;
  const options = document.querySelector("#object-options");
  options.innerHTML = guide.objects.map((objectId) => `<option value="${objectId}"></option>`).join("");
  document.querySelector("#object-id").value = guide.objects[0];
  document.querySelector("#prompt-guide").textContent = guide.hint;
}

function clampBoundary(value, previousEnd, nextStart) {
  const epsilon = 0.05;
  return Math.max(previousEnd + epsilon, Math.min(value, nextStart - epsilon));
}

function render() {
  const duration = Math.max(...annotations.steps.map((step) => step.time_range.end_sec));
  const timeline = document.querySelector("#timeline");
  timeline.innerHTML = annotations.steps.map((step, index) => {
    const width = (step.time_range.end_sec - step.time_range.start_sec) / duration * 100;
    const left = step.time_range.start_sec / duration * 100;
    return `<div class="timeline-step" data-checkpoint-id="${step.checkpoint_id}" style="--index:${index};left:${left}%;width:${width}%" title="${step.checkpoint_id}">
      ${index + 1}${index < 10 ? `<button class="boundary" data-boundary="${index}" aria-label="调整边界"></button>` : ""}
    </div>`;
  }).join("");
  timeline.querySelectorAll(".timeline-step").forEach((item) => item.addEventListener("click", () => {
    applyPromptGuide(item.dataset.checkpointId);
  }));

  document.querySelector("#step-list").innerHTML = annotations.steps.map((step, index) => `
    <article class="panel step-card">
      <strong>${step.checkpoint_id}</strong>
      <label>开始（秒）<input type="number" step="0.05" data-index="${index}" data-field="start_sec" value="${step.time_range.start_sec}"></label>
      <label>结束（秒）<input type="number" step="0.05" data-index="${index}" data-field="end_sec" value="${step.time_range.end_sec}"></label>
      <label>结果<select data-index="${index}" data-field="label">
        ${["needs_review", "correct", "incorrect", "incomplete"].map((value) => `<option ${value === step.label ? "selected" : ""}>${value}</option>`).join("")}
      </select></label>
      <label>原因<input data-index="${index}" data-field="reason" value="${step.reason}"></label>
    </article>`).join("");
  bindInputs();
  bindBoundaries(duration);
  renderPrompts();
}

function renderPrompts() {
  const surface = document.querySelector("#prompt-surface");
  surface.querySelectorAll(".prompt-marker").forEach((item) => item.remove());
  annotations.prompts.forEach((prompt) => {
    const marker = document.createElement("span");
    marker.className = `prompt-marker ${prompt.kind}`;
    if (prompt.kind === "point") {
      marker.style.left = `${prompt.x * 100}%`;
      marker.style.top = `${prompt.y * 100}%`;
    } else if (prompt.kind === "box") {
      marker.style.left = `${prompt.x1 * 100}%`;
      marker.style.top = `${prompt.y1 * 100}%`;
      marker.style.width = `${(prompt.x2 - prompt.x1) * 100}%`;
      marker.style.height = `${(prompt.y2 - prompt.y1) * 100}%`;
    }
    surface.append(marker);
  });
}

function bindInputs() {
  document.querySelectorAll("[data-field]").forEach((input) => input.addEventListener("change", () => {
    const step = annotations.steps[Number(input.dataset.index)];
    if (["start_sec", "end_sec"].includes(input.dataset.field)) step.time_range[input.dataset.field] = Number(input.value);
    else step[input.dataset.field] = input.value;
    render();
  }));
}

function bindBoundaries(duration) {
  document.querySelectorAll(".boundary").forEach((handle) => handle.addEventListener("pointerdown", () => {
    const moveBoundary = (move) => {
      const index = Number(handle.dataset.boundary);
      const rectangle = document.querySelector("#timeline").getBoundingClientRect();
      const proposed = (move.clientX - rectangle.left) / rectangle.width * duration;
      const previousStart = annotations.steps[index].time_range.start_sec;
      const nextEnd = annotations.steps[index + 1].time_range.end_sec;
      const boundary = clampBoundary(proposed, previousStart, nextEnd);
      annotations.steps[index].time_range.end_sec = boundary;
      annotations.steps[index + 1].time_range.start_sec = boundary;
      document.querySelector(`[data-index="${index}"][data-field="end_sec"]`).value = boundary.toFixed(2);
      document.querySelector(`[data-index="${index + 1}"][data-field="start_sec"]`).value = boundary.toFixed(2);
    };
    const finishBoundary = () => {
      window.removeEventListener("pointermove", moveBoundary);
      window.removeEventListener("pointerup", finishBoundary);
      render();
    };
    window.addEventListener("pointermove", moveBoundary);
    window.addEventListener("pointerup", finishBoundary);
  }));
}

document.querySelector("#save-segments").addEventListener("click", async () => {
  const response = await fetch(`/api/videos/${videoId}/segments`, {
    method: "PUT",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({actor: document.querySelector("#actor").value, steps: annotations.steps}),
  });
  const payload = await response.json();
  document.querySelector("#annotation-message").textContent = response.ok ? "已保存" : (payload.detail || "保存失败");
  if (response.ok) annotations = payload;
});

function normalizedPosition(event) {
  const rectangle = document.querySelector("#prompt-surface").getBoundingClientRect();
  return {
    x: Math.max(0, Math.min(1, (event.clientX - rectangle.left) / rectangle.width)),
    y: Math.max(0, Math.min(1, (event.clientY - rectangle.top) / rectangle.height)),
  };
}

document.querySelector("#prompt-surface").addEventListener("pointerdown", (event) => {
  dragStart = normalizedPosition(event);
  event.currentTarget.setPointerCapture(event.pointerId);
});

document.querySelector("#prompt-surface").addEventListener("pointerup", (event) => {
  const end = normalizedPosition(event);
  const common = {
    video_id: videoId,
    frame_time_sec: document.querySelector("#annotation-video").currentTime,
    object_id: document.querySelector("#object-id").value,
  };
  if (document.querySelector("#prompt-mode").value === "point") {
    annotations.prompts.push({kind: "point", ...common, x: end.x, y: end.y, positive: true});
  } else if (dragStart && Math.abs(end.x - dragStart.x) > 0.005 && Math.abs(end.y - dragStart.y) > 0.005) {
    annotations.prompts.push({
      kind: "box", ...common,
      x1: Math.min(dragStart.x, end.x), y1: Math.min(dragStart.y, end.y),
      x2: Math.max(dragStart.x, end.x), y2: Math.max(dragStart.y, end.y),
    });
  }
  dragStart = null;
  renderPrompts();
});

document.querySelector("#clear-prompts").addEventListener("click", () => {
  annotations.prompts = [];
  renderPrompts();
});

document.querySelector("#save-prompts").addEventListener("click", async () => {
  const response = await fetch(`/api/videos/${videoId}/prompts`, {
    method: "PUT",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({actor: document.querySelector("#actor").value, prompts: annotations.prompts}),
  });
  const payload = await response.json();
  document.querySelector("#annotation-message").textContent = response.ok ? "SAM 提示已保存" : (payload.detail || "保存失败");
  if (response.ok) annotations = payload;
});

fetch(`/api/videos/${videoId}/segments`).then((response) => response.json()).then((payload) => {
  annotations = payload;
  render();
  applyPromptGuide(root.dataset.defaultCheckpoint);
});
