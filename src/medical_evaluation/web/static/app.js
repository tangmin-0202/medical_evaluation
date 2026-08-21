async function createJob(options) {
  const message = document.querySelector("#message");
  if (message) message.textContent = "正在创建任务…";
  const response = await fetch("/api/jobs", options);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.detail || "创建任务失败");
  window.location.assign(`/jobs/${payload.job_id}`);
}

document.querySelectorAll("[data-preset-id]").forEach((button) => {
  button.addEventListener("click", () => createJob({
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({preset_id: button.dataset.presetId}),
  }).catch((error) => { document.querySelector("#message").textContent = error.message; }));
});

const uploadForm = document.querySelector("#upload-form");
if (uploadForm) uploadForm.addEventListener("submit", (event) => {
  event.preventDefault();
  createJob({method: "POST", body: new FormData(uploadForm)})
    .catch((error) => { document.querySelector("#message").textContent = error.message; });
});

const jobRoot = document.querySelector("[data-job-id]");
if (jobRoot) {
  const poll = async () => {
    const response = await fetch(`/api/jobs/${jobRoot.dataset.jobId}`);
    const job = await response.json();
    document.querySelector("#status").textContent = job.status;
    document.querySelector("#percent").textContent = `${Math.round(job.progress * 100)}%`;
    document.querySelector("#progress-bar").style.width = `${job.progress * 100}%`;
    document.querySelector("#checkpoint").textContent = job.current_checkpoint || "";
    document.querySelector("#error").textContent = job.error || "";
    if (job.status === "complete") document.querySelector("#report-link").classList.remove("hidden");
    if (!["complete", "failed"].includes(job.status)) window.setTimeout(poll, 500);
  };
  poll();
}
