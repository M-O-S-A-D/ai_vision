# Ollama Vision Test Project

This repository contains a Python-based real-time computer vision prototype that connects:

- a local webcam feed,
- an Ollama vision-capable model (for example `llava`), and
- a lightweight web dashboard for live monitoring.

The script captures frames, preprocesses them, sends images to Ollama for scene description, and stores performance/session logs.

## Project status

The project has been tested, and test run artifacts are included under `vision_logs/`.

Included sample sessions:

- `vision_logs/session_20260304_010408/`
- `vision_logs/session_20260304_011939/`

These folders contain saved outputs such as analysis history and performance metrics from actual runs.

## Main behavior

- Captures camera frames continuously (targeting 1280x720 MJPG).
- Selects sharper frames from a short rolling buffer.
- Enhances frames before AI inference.
- Sends frames to Ollama with streaming responses.
- Serves a web UI on port `5000` to show live feed and AI text.
- Writes structured logs (`JSONL`, `CSV`, and summary `JSON`) for each session.

## File and folder guide

### Root files

- `ollama_vision.py`
  - Main application entry point.
  - Starts camera capture thread, AI analysis thread, and HTTP server.
  - Exposes endpoints such as `/`, `/state`, `/stats`, `/config`, `/pause`, and `/export`.
  - Creates per-session logs and summary files.

- `ollama_vision.py.bak`
  - Backup copy of the main script from an earlier state.

- `ollama_vision_v2.bak`
  - Additional backup/versioned copy of the script.

- `README.md`
  - This documentation file.

### Runtime/generated folders

- `__pycache__/`
  - Python bytecode cache generated automatically by the interpreter.

- `vision_logs/`
  - Parent folder for session output.
  - Each run is stored in a timestamped subfolder named like `session_YYYYMMDD_HHMMSS`.

#### Session folder contents

Each session folder (for example `vision_logs/session_20260304_010408/`) contains:

- `analysis_log.jsonl`
  - Line-delimited JSON entries.
  - One record per AI analysis with fields like timestamp, model, inference time, scene-change percentage, token count, CPU/memory stats, and generated description text.

- `performance.csv`
  - Tabular performance log for charting/analysis.
  - Includes columns such as inference latency, sharpness, token count, frame count, and system usage.

- `session_summary.json`
  - End-of-run summary report.
  - Includes aggregate metrics (average/min/max/median inference time, analyses per minute), run duration, camera info, and in-memory history snapshot.

- `frames/` (present when frame saving is enabled and enough analyses occur)
  - Saved JPEG frames representing what was sent to the AI model during the session.

## Notes

- The script currently defaults to an Ollama server URL of `http://localhost:11434`.
- In `ollama_vision.py`, `LOG_DIR` is configured to a user path (`~/Downloads/NOETIC/vision_logs`).
  - In this repository, `vision_logs/` contains collected test artifacts for inspection.