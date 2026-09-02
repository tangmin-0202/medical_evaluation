function scoreLabel(summary) {
  if (summary.evaluated_count < summary.total_count) {
    return summary.provisional_score === null
      ? `${summary.provisional_minimum_score.toFixed(1)}–${summary.provisional_maximum_score.toFixed(1)}`
      : summary.provisional_score.toFixed(1);
  }
  return summary.final_score === null
    ? `${summary.minimum_score.toFixed(1)}–${summary.maximum_score.toFixed(1)}（待复核）`
    : `${summary.final_score.toFixed(1)} / 100`;
}

document.querySelectorAll(".review-form").forEach((form) => form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const data = new FormData(form);
  const jobId = document.querySelector("#report-app").dataset.jobId;
  const response = await fetch(`/api/reports/${jobId}/${form.dataset.reviewCheckpointId}/review`, {
    method: "PUT",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(Object.fromEntries(data)),
  });
  const payload = await response.json();
  if (!response.ok) {
    window.alert(payload.detail || "复核保存失败");
    return;
  }
  document.querySelector("#score-label").textContent = scoreLabel(payload.summary);
  window.location.reload();
}));
