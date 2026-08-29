# SAM2 Windowed Stage Input Design

## Problem

The real CP09 smoke test passes the complete success MP4 to
`SAM2VideoPredictor.init_state`. The official loader decodes and resizes every
video frame and then moves the tensor to the compute device. For the 252 MB
success video this attempted a 175.12 GiB CUDA allocation on a 23.56 GiB RTX
A5000. The current one-FPS retry does not reduce initialization memory because
sampling happens only after the complete video has been loaded.

## Goal

Introduce a checkpoint-agnostic windowed input mechanism in `Sam2Backend` so
each of the eventual eleven assessment stages processes only its own time range.
The current vertical slice connects and validates CP09 only. Reviewed times,
scoring rules, and audit outputs remain unchanged.

## Non-goals

- Do not implement the remaining ten checkpoint feature extractors now.
- Do not change CP09 thresholds or reviewed JSON annotations.
- Do not add SAM3, require FFmpeg, or retain a second long-form video.
- Do not change the fake pipeline or web-server mode.

## Selected Approach

For every `Sam2Backend.track` call, export a temporary, numerically named JPEG
sequence from that call's `TimeRange`. It contains frames sampled uniformly at
`sample_fps` plus the nearest original frame for every reviewed point, box, or
mask prompt. Duplicate source frames are removed and sorted. A timeline records
each local SAM2 index, original frame index, and original timestamp.

This belongs in the generic adapter rather than the CP09 script. CP01 through
CP11 can reuse it later with their per-video intervals. The SAM2 weights stay
loaded while stages run sequentially; predictor state and temporary frames are
reset between stages.

Rejected alternatives:

- A reduced-FPS FFmpeg clip adds a dependency and less transparent timestamp
  and keyframe behavior.
- `offload_video_to_cpu=True` still decodes the complete video and may require
  roughly 175 GiB of host memory.

## Components

### Windowed frame preparation

A focused video utility builds the ordered source-frame list, decodes each
selected frame with OpenCV, and writes `00000.jpg`, `00001.jpg`, and so on into
a temporary directory. It returns source metadata and an immutable
local-to-source timeline. Prompt frames are mandatory even when off-grid. A
prompt outside the stage interval fails with an actionable error.

### Generic SAM2 adapter

`Sam2Backend.track` initializes the predictor with the JPEG directory instead
of the MP4. Prompt local indices come from the timeline; coordinates still use
the original frame dimensions. Returned local indices are mapped back to source
timestamps before `FrameMasks` are built, so extractors, evidence writers,
judges, and report schemas do not change.

State reset and temporary-directory cleanup run in `finally` blocks after
success, prompt failure, propagation failure, or CUDA OOM.

### Eleven-stage execution model

The future full pipeline obtains eleven per-video intervals from reviewed
annotations or a stage localizer. It invokes the same loaded backend
sequentially for stages needing segmentation. Each call prepares and cleans only
its stage window. Stages needing sequence classification instead of masks can
use another extractor. Per-video intervals handle timing differences without
global fixed timestamps.

### OOM degradation

The smoke script keeps one retry. A retry at one FPS now rebuilds a smaller JPEG
sequence and materially reduces memory. Mandatory prompt frames remain present.

## Data Flow

```text
video + per-video stage interval + reviewed prompts
  -> sampled frames union prompt frames
  -> temporary JPEG sequence + source timeline
  -> SAM2 prompting and bidirectional propagation
  -> source timestamp restoration
  -> stage features, decision, and overlays
```

## Error Handling

- Reject non-positive rates and out-of-video stage ranges.
- Reject reviewed prompts outside the requested stage range.
- Fail if a selected frame cannot be decoded or written.
- Always reset predictor state and remove temporary inputs.
- Retry CUDA OOM once at at most one FPS; surface the second OOM.

## Verification

Automated tests prove that `init_state` receives a JPEG directory rather than
the MP4, only window and mandatory prompt frames are written, prompt indices and
result timestamps map correctly, duplicates are removed, cleanup works after
success and failure, lower FPS reduces non-prompt frames, non-CP09 ranges use the
same backend, and the existing suite and Ruff checks remain green.

The server CP09 smoke succeeds only after SAM2 finishes on physical GPU 7,
writes `summary.json` and `decision.json`, and the first/middle/last overlays
cover the rubber dam frame rather than hands, rubber sheet, face, or background.
