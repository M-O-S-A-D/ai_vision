#!/usr/bin/env python3
"""
Ollama Vision — Optimized Real-time Camera Analysis

Architecture (3 threads, fully decoupled):
  T1  Camera capture  — 720p MJPG @ 30fps, smooth live feed
  T2  AI analysis     — streaming responses, continuous cycle
  T3  HTTP server     — web UI on :5000, live-streaming text

Performance optimizations:
  • MJPG capture at native 1280x720 (no YUYV→RGB conversion)
  • Image enhancement: auto-brightness, contrast, sharpening
  • Sharpest-frame selection from rolling buffer
  • Downscale to 512px for AI (optimal for vision models)
  • Streaming Ollama responses (text appears word-by-word)
  • Model pre-warming at startup
  • Continuous analysis (no idle gap after long inference)
  • Low scene-change threshold (catches motion faster)
"""

import cv2
import base64
import requests
import threading
import time
import json
import sys
import os
import logging
import numpy as np
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime
from collections import deque

# ── Config ──────────────────────────────────────────────────────────────────
OLLAMA_URL          = "http://localhost:11434"
PREFERRED_MODELS    = ["llava", "bakllava", "llava:13b", "moondream"]
PROMPT              = "What is in this image? Describe the scene accurately."
MIN_INTERVAL        = 0.5          # min seconds between analyses
WEB_PORT            = 5000
SCENE_CHANGE_THRESH = 5.0          # % change to force re-analysis
CPU_TIMEOUT         = 300          # 5min timeout for CPU inference
AI_IMAGE_WIDTH      = 768          # px sent to model (bigger = more accurate)
DISPLAY_WIDTH       = 1280         # px for browser live feed
DISPLAY_QUALITY     = 80           # JPEG quality for browser feed
AI_QUALITY          = 95           # JPEG quality for model (high = accurate)
SHARP_BUFFER_SIZE   = 5            # pick sharpest from last N frames
LOG_DIR             = Path.home() / "Downloads" / "NOETIC" / "vision_logs"
SAVE_AI_FRAMES      = True         # save what the AI actually sees
FRAME_SAVE_INTERVAL = 10           # save AI frame every N analyses
# ────────────────────────────────────────────────────────────────────────────


# ── Logging setup ───────────────────────────────────────────────────────────
def setup_logging():
    """Create log directory and configure file + console logging."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    frames_dir = LOG_DIR / "frames"
    frames_dir.mkdir(exist_ok=True)

    # Session timestamp for this run
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = LOG_DIR / f"session_{session_id}"
    session_dir.mkdir(exist_ok=True)
    (session_dir / "frames").mkdir(exist_ok=True)

    # Main log file (JSONL — one JSON object per line)
    log_file = session_dir / "analysis_log.jsonl"
    # Session summary (written on exit)
    summary_file = session_dir / "session_summary.json"
    # Performance CSV for easy charting
    perf_file = session_dir / "performance.csv"

    # Write CSV header
    with open(perf_file, "w") as f:
        f.write("timestamp,analysis_num,model,inference_ms,scene_change_pct,"
                "sharpness,tokens,frame_count,description_length,"
                "cpu_pct,mem_used_mb,prompt\n")

    # Python logger for console + file
    logger = logging.getLogger("vision")
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(session_dir / "debug.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"
    ))
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    logger.addHandler(ch)

    return session_id, session_dir, log_file, summary_file, perf_file, logger


def get_system_stats():
    """Read CPU and memory usage from /proc (Linux)."""
    cpu_pct = 0.0
    mem_mb = 0.0
    try:
        # CPU: average from /proc/loadavg (1-min load / num cores)
        with open("/proc/loadavg") as f:
            load1 = float(f.read().split()[0])
        ncpu = os.cpu_count() or 1
        cpu_pct = round((load1 / ncpu) * 100, 1)
    except Exception:
        pass
    try:
        # Memory: from /proc/meminfo
        with open("/proc/meminfo") as f:
            lines = f.readlines()
        info = {}
        for line in lines:
            parts = line.split()
            if parts[0] in ("MemTotal:", "MemAvailable:"):
                info[parts[0]] = int(parts[1])  # kB
        total = info.get("MemTotal:", 0)
        avail = info.get("MemAvailable:", 0)
        mem_mb = round((total - avail) / 1024, 1)
    except Exception:
        pass
    return cpu_pct, mem_mb


def log_analysis(entry):
    """Append one analysis result to JSONL log and CSV."""
    try:
        with open(log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass
    try:
        with open(perf_file, "a") as f:
            f.write(
                f"{entry['timestamp']},{entry['analysis_num']},"
                f"{entry['model']},{entry['inference_ms']},"
                f"{entry['scene_change_pct']},{entry['sharpness']},"
                f"{entry.get('tokens', 0)},{entry['frame_count']},"
                f"{entry['description_length']},"
                f"{entry.get('cpu_pct', 0)},{entry.get('mem_used_mb', 0)},"
                f"\"{entry.get('prompt', '')[:60]}\"\n"
            )
    except Exception:
        pass


def save_session_summary():
    """Write final session summary JSON on exit."""
    elapsed = time.time() - session_start_time
    times = [e.get("ms", 0) for e in state["history"] if isinstance(e.get("ms"), (int, float)) and e["ms"] > 0]
    summary = {
        "session_id":      session_id,
        "started_at":      datetime.fromtimestamp(session_start_time).isoformat(),
        "duration_s":      round(elapsed, 1),
        "model":           state["model"],
        "camera_info":     state["camera_info"],
        "total_frames":    state["frame_count"],
        "total_analyses":  state["analysis_count"],
        "ai_image_width":  AI_IMAGE_WIDTH,
        "ai_quality":      AI_QUALITY,
        "config": {
            "prompt":       prompt_text,
            "interval":     state["interval"],
            "scene_thresh": SCENE_CHANGE_THRESH,
        },
        "performance": {
            "avg_inference_ms": round(sum(times) / len(times), 1) if times else 0,
            "min_inference_ms": min(times) if times else 0,
            "max_inference_ms": max(times) if times else 0,
            "median_inference_ms": round(sorted(times)[len(times)//2], 1) if times else 0,
            "analyses_per_min":  round(len(times) / max(elapsed / 60, 0.01), 1),
        },
        "history": state["history"],
    }
    try:
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"📊 Session summary saved → {summary_file}")
    except Exception as e:
        logger.error(f"Failed to save summary: {e}")


# Globals set by setup_logging() in __main__
session_id = ""
session_dir = Path(".")
log_file = Path(".")
summary_file = Path(".")
perf_file = Path(".")
logger = logging.getLogger("vision")
session_start_time = time.time()


def detect_models():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


def pick_best_model(available):
    short = [n.split(":")[0] for n in available]
    for pref in PREFERRED_MODELS:
        base = pref.split(":")[0]
        if pref in available:
            return pref
        if base in short:
            return available[short.index(base)]
    return available[0] if available else "llava"


def prewarm_model(model):
    """Load model into memory with a tiny no-image request."""
    print(f"   🔥 Pre-warming {model}…", end=" ", flush=True)
    t0 = time.time()
    try:
        r = requests.post(f"{OLLAMA_URL}/api/generate", json={
            "model": model, "prompt": "hi", "stream": False,
        }, timeout=120)
        r.raise_for_status()
        print(f"ready in {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"skip ({e})")


# ── Shared state ─────────────────────────────────────────────────────────────
prompt_lock = threading.Lock()
prompt_text = PROMPT

state = {
    # frames — separate for display vs AI
    "display_b64":      None,       # 720p JPEG for browser
    "ai_frame_b64":     None,       # 512px enhanced JPEG for model
    # AI output
    "description":      "Starting camera…",
    "streaming_text":   "",         # partial response while generating
    "model":            "llava",
    "interval":         MIN_INTERVAL,
    "running":          True,
    "paused":           False,
    "analysing":        False,
    "frame_count":      0,
    "analysis_count":   0,
    "analysis_ms":      0,          # last inference time
    "error":            None,
    "available_models": [],
    "history":          [],
    "scene_change_pct": 0.0,
    "sharpness":        0.0,
    "camera_info":      "",
}


# ── Image enhancement ────────────────────────────────────────────────────────
def enhance_frame(frame):
    """Light brightness normalization only — keep image natural for the AI."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Light CLAHE — just enough to see in dark rooms, no artifacts
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def frame_sharpness(frame):
    """Laplacian variance — higher = sharper."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


# ── Ollama streaming call ────────────────────────────────────────────────────
def analyse_frame_streaming(frame_b64: str) -> str:
    """Send frame to Ollama with streaming — updates state['streaming_text'] live."""
    with prompt_lock:
        current_prompt = prompt_text

    try:
        payload = {
            "model":   state["model"],
            "prompt":  current_prompt,
            "images":  [frame_b64],
            "stream":  True,
            "options": {
                "num_predict": 300,     # allow enough tokens for detail
                "temperature": 0.1,     # very factual, minimal hallucination
                "top_p":       0.9,
            },
        }

        state["streaming_text"] = ""
        state["_token_count"] = 0
        full_text = ""
        token_count = 0

        with requests.post(
            f"{OLLAMA_URL}/api/generate",
            json=payload,
            timeout=CPU_TIMEOUT,
            stream=True,
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not state["running"]:
                    break
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                    token = chunk.get("response", "")
                    full_text += token
                    token_count += 1
                    state["streaming_text"] = full_text
                    if chunk.get("done"):
                        break
                except json.JSONDecodeError:
                    continue

        state["_token_count"] = token_count
        return full_text.strip() or "No response"

    except requests.exceptions.ConnectionError:
        return "❌ Cannot connect to Ollama — run: ollama serve"
    except requests.exceptions.Timeout:
        return f"⏱️ Timed out ({CPU_TIMEOUT}s). Try moondream (faster on CPU)."
    except requests.exceptions.HTTPError as e:
        code = e.response.status_code if e.response is not None else "?"
        if code == 404:
            return f"❌ Model '{state['model']}' not found → ollama pull {state['model']}"
        return f"❌ HTTP {code}: {e}"
    except Exception as e:
        return f"Error: {e}"


# ── Scene change detection ───────────────────────────────────────────────────
_prev_gray = None


def scene_changed(gray_small):
    """Return (changed, pct) from a pre-computed small grayscale frame."""
    global _prev_gray
    if _prev_gray is None:
        _prev_gray = gray_small
        return True, 100.0

    diff = cv2.absdiff(_prev_gray, gray_small)
    pct = (np.count_nonzero(diff > 20) / diff.size) * 100
    _prev_gray = gray_small
    return pct > SCENE_CHANGE_THRESH, round(pct, 1)


# ── Thread 1: Camera capture ────────────────────────────────────────────────
def camera_loop():
    # Try MJPG at 720p first (best for this camera)
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        # Fallback
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        state["description"] = "❌ Cannot open camera"
        state["error"] = "no_camera"
        return

    # Force MJPG codec for max performance (no YUYV→RGB on CPU)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    # Enable auto-exposure (3 = auto on V4L2)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
    # Boost brightness & gain for low-light
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)
    cap.set(cv2.CAP_PROP_GAIN, 200)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    codec = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec_str = "".join([chr((codec >> 8*i) & 0xFF) for i in range(4)])
    info = f"{actual_w}x{actual_h} @ {actual_fps:.0f}fps [{codec_str}]"
    state["camera_info"] = info
    print(f"   📷 Camera: {info}")
    state["description"] = f"Camera ready ({info}) — model: {state['model']}"

    # Rolling buffer for sharpest-frame selection
    sharp_buf = deque(maxlen=SHARP_BUFFER_SIZE)

    while state["running"]:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.03)
            continue

        state["frame_count"] += 1

        # ── Brighten display frame for dark rooms ──
        disp_frame = frame
        gray_check = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray_check)
        if mean_brightness < 80:  # dark scene — boost for display
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            if mean_brightness < 50:
                boost = min(2.0, 90.0 / max(mean_brightness, 1))
                l = np.clip(l.astype(np.float32) * boost, 0, 255).astype(np.uint8)
            disp_frame = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

        # ── Display frame (720p, fast encode) ──
        _, disp_buf = cv2.imencode(
            ".jpg", disp_frame, [cv2.IMWRITE_JPEG_QUALITY, DISPLAY_QUALITY]
        )
        state["display_b64"] = base64.b64encode(disp_buf).decode("utf-8")

        # ── AI frame: track sharpness, keep sharpest ──
        sharpness = frame_sharpness(frame)
        state["sharpness"] = round(sharpness, 1)
        sharp_buf.append((sharpness, frame.copy()))

        # Pick sharpest from buffer → enhance → downscale → encode for AI
        best_frame = max(sharp_buf, key=lambda x: x[0])[1]

        # Downscale for AI
        h, w = best_frame.shape[:2]
        scale = AI_IMAGE_WIDTH / w
        ai_frame = cv2.resize(best_frame, (AI_IMAGE_WIDTH, int(h * scale)),
                              interpolation=cv2.INTER_AREA)

        # Enhance
        ai_frame = enhance_frame(ai_frame)

        _, ai_buf = cv2.imencode(
            ".jpg", ai_frame, [cv2.IMWRITE_JPEG_QUALITY, AI_QUALITY]
        )
        state["ai_frame_b64"] = base64.b64encode(ai_buf).decode("utf-8")

        # Scene change detection (on small grayscale)
        gray_small = cv2.cvtColor(
            cv2.resize(frame, (160, 120)), cv2.COLOR_BGR2GRAY
        )
        changed, pct = scene_changed(gray_small)
        state["scene_change_pct"] = pct

        time.sleep(0.033)  # ~30 FPS

    cap.release()


# ── Thread 2: AI analysis (continuous) ──────────────────────────────────────
def analysis_loop():
    time.sleep(1.5)  # let camera warm up

    while state["running"]:
        if state["paused"]:
            time.sleep(0.3)
            continue

        frame = state.get("ai_frame_b64")
        if not frame:
            time.sleep(0.2)
            continue

        # Always analyze — scene change just controls urgency
        state["analysing"] = True
        t0 = time.time()
        desc = analyse_frame_streaming(frame)
        elapsed_ms = int((time.time() - t0) * 1000)

        state["description"]    = desc
        state["streaming_text"] = ""
        state["analysis_ms"]    = elapsed_ms
        state["analysis_count"] += 1
        state["analysing"]      = False

        # ── System stats ──
        cpu_pct, mem_mb = get_system_stats()

        # ── Build log entry ──
        now = datetime.now()
        with prompt_lock:
            cur_prompt = prompt_text
        entry = {
            "timestamp":        now.isoformat(),
            "analysis_num":     state["analysis_count"],
            "model":            state["model"],
            "inference_ms":     elapsed_ms,
            "scene_change_pct": state["scene_change_pct"],
            "sharpness":        state["sharpness"],
            "tokens":           state.get("_token_count", 0),
            "frame_count":      state["frame_count"],
            "description":      desc,
            "description_length": len(desc),
            "prompt":           cur_prompt,
            "cpu_pct":          cpu_pct,
            "mem_used_mb":      mem_mb,
        }

        # ── Write to log files ──
        log_analysis(entry)
        logger.debug(
            f"Analysis #{state['analysis_count']}: {elapsed_ms}ms, "
            f"{state.get('_token_count', 0)} tokens, Δ={state['scene_change_pct']}%, "
            f"CPU={cpu_pct}%, MEM={mem_mb}MB"
        )

        # ── Save AI frame periodically so we can see what the model saw ──
        if SAVE_AI_FRAMES and state["analysis_count"] % FRAME_SAVE_INTERVAL == 0:
            try:
                frame_path = session_dir / "frames" / f"analysis_{state['analysis_count']:05d}.jpg"
                raw = base64.b64decode(frame)
                with open(frame_path, "wb") as f:
                    f.write(raw)
            except Exception:
                pass

        # History (in-memory for UI)
        state["history"].insert(0, {
            "time":        now.strftime("%H:%M:%S"),
            "description": desc,
            "change_pct":  state["scene_change_pct"],
            "ms":          elapsed_ms,
            "tokens":      state.get("_token_count", 0),
            "cpu_pct":     cpu_pct,
            "mem_mb":      mem_mb,
        })
        if len(state["history"]) > 100:
            state["history"] = state["history"][:100]

        # Short pause between analyses (but don't wait long —
        # the inference itself is the bottleneck)
        time.sleep(max(MIN_INTERVAL, state["interval"]))


# ── HTML / JS ────────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Ollama Vision — Live</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Space+Grotesk:wght@300;500;700&display=swap');
  :root {
    --bg:#0a0a0f;--panel:#111118;--border:#1e1e2e;
    --accent:#7c3aed;--accent2:#06b6d4;--text:#e2e8f0;
    --muted:#64748b;--green:#10b981;--red:#f43f5e;--amber:#f59e0b;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--text);font-family:'Space Grotesk',sans-serif;min-height:100vh;display:flex;flex-direction:column}

  header{padding:1rem 2rem;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:1rem;background:var(--panel)}
  .logo{font-size:1.1rem;font-weight:700;letter-spacing:-.02em} .logo span{color:var(--accent2)}
  .status-dot{width:8px;height:8px;border-radius:50%;background:var(--green);animation:pulse 2s infinite;margin-left:auto}
  .status-dot.busy{background:var(--accent)} .status-dot.paused{background:var(--amber);animation:none}
  @keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.5;transform:scale(1.3)}}
  .status-label{font-size:.8rem;color:var(--muted);font-family:'DM Mono',monospace}
  .perf-badge{font-family:'DM Mono',monospace;font-size:.7rem;color:var(--accent2);background:var(--panel);border:1px solid var(--border);padding:.2rem .5rem;border-radius:4px}

  main{flex:1;display:grid;grid-template-columns:1fr 1fr;gap:0}
  .panel{padding:1.5rem;border-right:1px solid var(--border);overflow-y:auto} .panel:last-child{border-right:none}
  .panel-title{font-size:.7rem;font-weight:500;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);margin-bottom:.8rem;font-family:'DM Mono',monospace}

  #camera-feed{width:100%;border-radius:8px;border:1px solid var(--border);background:#000;display:block;aspect-ratio:16/9;object-fit:cover}

  .meta{margin-top:.8rem;display:flex;gap:.5rem;flex-wrap:wrap}
  .meta-item{font-family:'DM Mono',monospace;font-size:.7rem;color:var(--muted);background:var(--panel);border:1px solid var(--border);padding:.25rem .5rem;border-radius:4px}
  .meta-item strong{color:var(--accent2)}

  .description-box{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:1.2rem;min-height:100px;font-size:.95rem;line-height:1.6;position:relative;transition:border-color .3s;overflow:hidden;white-space:pre-wrap}
  .description-box.thinking{border-color:var(--accent)}
  .description-box.thinking::after{content:'▊';animation:blink .6s infinite;color:var(--accent)}
  @keyframes blink{0%,100%{opacity:1}50%{opacity:0}}

  .controls{margin-top:1.2rem;display:flex;flex-direction:column;gap:.7rem}
  label{font-size:.78rem;color:var(--muted);display:block;margin-bottom:.2rem}
  select,input[type=range]{width:100%;background:var(--panel);border:1px solid var(--border);color:var(--text);padding:.45rem .7rem;border-radius:6px;font-family:'DM Mono',monospace;font-size:.82rem;outline:none;cursor:pointer}
  select:focus{border-color:var(--accent)}
  .range-val{font-family:'DM Mono',monospace;font-size:.78rem;color:var(--accent2)}
  textarea{width:100%;background:var(--panel);border:1px solid var(--border);color:var(--text);padding:.45rem .7rem;border-radius:6px;font-family:'DM Mono',monospace;font-size:.82rem;outline:none;resize:vertical;min-height:55px}
  textarea:focus{border-color:var(--accent)}

  .btn-row{display:flex;gap:.5rem;flex-wrap:wrap}
  button{background:var(--accent);color:#fff;border:none;padding:.5rem 1rem;border-radius:6px;font-family:'Space Grotesk',sans-serif;font-weight:600;font-size:.82rem;cursor:pointer;transition:opacity .2s}
  button:hover{opacity:.85} button.secondary{background:var(--border);color:var(--text)} button.warn{background:var(--amber);color:#000}

  .history{margin-top:.6rem;display:flex;flex-direction:column;gap:.5rem;max-height:260px;overflow-y:auto}
  .history-item{background:var(--panel);border:1px solid var(--border);border-radius:6px;padding:.5rem .7rem;font-size:.78rem;line-height:1.5;color:var(--muted);animation:fadeIn .3s ease}
  @keyframes fadeIn{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:none}}
  .history-item .ts{font-family:'DM Mono',monospace;font-size:.68rem;color:var(--accent);margin-bottom:.15rem}
  .history-item .perf{font-family:'DM Mono',monospace;font-size:.62rem;color:var(--muted);float:right}

  @media(max-width:700px){main{grid-template-columns:1fr}.panel{border-right:none;border-bottom:1px solid var(--border)}}
</style>
</head>
<body>
<header>
  <div class="logo">ollama <span>vision</span></div>
  <div class="status-label" id="status-label">initialising</div>
  <div class="perf-badge" id="perf-badge">—</div>
  <div class="status-dot" id="status-dot"></div>
</header>
<main>
  <div class="panel">
    <div class="panel-title">📷 Live Feed</div>
    <img id="camera-feed" src="" alt="Camera feed">
    <div class="meta">
      <div class="meta-item">Frames: <strong id="frame-count">0</strong></div>
      <div class="meta-item">Analyses: <strong id="analysis-count">0</strong></div>
      <div class="meta-item">Model: <strong id="model-label">—</strong></div>
      <div class="meta-item">Δ scene: <strong id="scene-pct">—</strong></div>
      <div class="meta-item">Sharpness: <strong id="sharpness">—</strong></div>
      <div class="meta-item" id="cam-info">—</div>
    </div>
  </div>
  <div class="panel">
    <div class="panel-title">🤖 AI Description</div>
    <div class="description-box" id="desc-box">Waiting for camera…</div>
    <div class="controls">
      <div>
        <label>Model</label>
        <select id="model-select"></select>
      </div>
      <div>
        <label>Min interval between analyses: <span class="range-val" id="interval-val">0.5s</span></label>
        <input type="range" id="interval" min="0.5" max="10" step="0.5" value="0.5">
      </div>
      <div>
        <label>Custom prompt</label>
        <textarea id="prompt-input">What is in this image? Describe the scene accurately.</textarea>
      </div>
      <div class="btn-row">
        <button id="apply-btn">Apply</button>
        <button id="pause-btn" class="warn">⏸ Pause</button>
        <button id="export-btn" class="secondary">💾 Export</button>
      </div>
    </div>
    <div class="panel-title" style="margin-top:1.2rem">📝 History</div>
    <div class="history" id="history"></div>
  </div>
</main>
<script>
const feed=document.getElementById('camera-feed'),descBox=document.getElementById('desc-box'),
dot=document.getElementById('status-dot'),lbl=document.getElementById('status-label'),
fc=document.getElementById('frame-count'),ac=document.getElementById('analysis-count'),
modelLbl=document.getElementById('model-label'),scenePct=document.getElementById('scene-pct'),
sharpEl=document.getElementById('sharpness'),camInfo=document.getElementById('cam-info'),
historyEl=document.getElementById('history'),slider=document.getElementById('interval'),
iVal=document.getElementById('interval-val'),modelSel=document.getElementById('model-select'),
pauseBtn=document.getElementById('pause-btn'),exportBtn=document.getElementById('export-btn'),
perfBadge=document.getElementById('perf-badge');

slider.addEventListener('input',()=>iVal.textContent=slider.value+'s');
let lastDesc='',isPaused=false,modelsInit=false;

async function poll(){
  try{
    const r=await fetch('/state');const s=await r.json();
    if(s.display_b64) feed.src='data:image/jpeg;base64,'+s.display_b64;

    if(!modelsInit&&s.available_models&&s.available_models.length){
      modelSel.innerHTML='';
      for(const m of s.available_models){const o=document.createElement('option');o.value=m;o.textContent=m;if(m===s.model)o.selected=true;modelSel.appendChild(o)}
      modelsInit=true;
    }

    // Show streaming text live while analysing
    if(s.analysing&&s.streaming_text){
      descBox.textContent=s.streaming_text;
      descBox.className='description-box thinking';
    } else if(s.description&&s.description!==lastDesc){
      descBox.textContent=s.description;
      descBox.className='description-box';
      if(lastDesc&&!lastDesc.startsWith('❌')&&!lastDesc.startsWith('Starting')&&!lastDesc.startsWith('Camera ready')){
        const item=document.createElement('div');item.className='history-item';
        item.innerHTML='<span class="perf">'+s.analysis_ms+'ms</span><div class="ts">'+new Date().toLocaleTimeString()+'</div>'+lastDesc;
        historyEl.prepend(item);if(historyEl.children.length>40)historyEl.lastChild.remove();
      }
      lastDesc=s.description;
    } else if(!s.analysing){
      descBox.className='description-box';
    }

    if(s.paused){dot.className='status-dot paused';lbl.textContent='paused'}
    else if(s.analysing){dot.className='status-dot busy';lbl.textContent='analysing…'}
    else{dot.className='status-dot';lbl.textContent='watching'}

    fc.textContent=s.frame_count;ac.textContent=s.analysis_count;
    modelLbl.textContent=s.model;scenePct.textContent=s.scene_change_pct+'%';
    sharpEl.textContent=s.sharpness;
    if(s.camera_info)camInfo.textContent=s.camera_info;
    perfBadge.textContent=s.analysis_ms?s.analysis_ms+'ms':'—';
    isPaused=s.paused;
    pauseBtn.textContent=isPaused?'▶ Resume':'⏸ Pause';
    pauseBtn.className=isPaused?'secondary':'warn';
  }catch(e){lbl.textContent='server error'}
  setTimeout(poll,s_analysing?300:600);
}
let s_analysing=false;

async function pollWrap(){
  try{
    const r=await fetch('/state');const s=await r.json();s_analysing=s.analysing;
    // Same as poll but we already have s
    if(s.display_b64) feed.src='data:image/jpeg;base64,'+s.display_b64;
    if(!modelsInit&&s.available_models&&s.available_models.length){
      modelSel.innerHTML='';
      for(const m of s.available_models){const o=document.createElement('option');o.value=m;o.textContent=m;if(m===s.model)o.selected=true;modelSel.appendChild(o)}
      modelsInit=true;
    }
    if(s.analysing&&s.streaming_text){descBox.textContent=s.streaming_text;descBox.className='description-box thinking'}
    else if(s.description&&s.description!==lastDesc){
      descBox.textContent=s.description;descBox.className='description-box';
      if(lastDesc&&!lastDesc.startsWith('❌')&&!lastDesc.startsWith('Starting')&&!lastDesc.startsWith('Camera ready')){
        const item=document.createElement('div');item.className='history-item';
        item.innerHTML='<span class="perf">'+s.analysis_ms+'ms</span><div class="ts">'+new Date().toLocaleTimeString()+'</div>'+lastDesc;
        historyEl.prepend(item);if(historyEl.children.length>40)historyEl.lastChild.remove();
      }
      lastDesc=s.description;
    } else if(!s.analysing){descBox.className='description-box'}
    if(s.paused){dot.className='status-dot paused';lbl.textContent='paused'}
    else if(s.analysing){dot.className='status-dot busy';lbl.textContent='analysing…'}
    else{dot.className='status-dot';lbl.textContent='watching'}
    fc.textContent=s.frame_count;ac.textContent=s.analysis_count;
    modelLbl.textContent=s.model;scenePct.textContent=s.scene_change_pct+'%';
    sharpEl.textContent=s.sharpness;
    if(s.camera_info)camInfo.textContent=s.camera_info;
    perfBadge.textContent=s.analysis_ms?s.analysis_ms+'ms':'—';
    isPaused=s.paused;
    pauseBtn.textContent=isPaused?'▶ Resume':'⏸ Pause';
    pauseBtn.className=isPaused?'secondary':'warn';
  }catch(e){lbl.textContent='server error'}
  setTimeout(pollWrap,s_analysing?300:600);
}

document.getElementById('apply-btn').addEventListener('click',async()=>{
  await fetch('/config',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({model:modelSel.value,interval:parseFloat(slider.value),prompt:document.getElementById('prompt-input').value.trim()})});
});
pauseBtn.addEventListener('click',async()=>{await fetch('/pause',{method:'POST'})});
exportBtn.addEventListener('click',async()=>{
  try{const r=await fetch('/export');const b=await r.blob();const u=URL.createObjectURL(b);
  const a=document.createElement('a');a.href=u;a.download='ollama_vision_history.json';a.click();URL.revokeObjectURL(u)}catch(e){alert('Export failed')}
});

pollWrap();
</script>
</body>
</html>
"""


# ── HTTP handler ─────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/":
            self._respond(200, "text/html", HTML.encode())
        elif self.path == "/state":
            out = {k: v for k, v in state.items()}
            del out["running"]
            # Don't send ai_frame to browser (save bandwidth)
            out.pop("ai_frame_b64", None)
            self._respond(200, "application/json", json.dumps(out).encode())
        elif self.path == "/stats":
            elapsed = time.time() - session_start_time
            times = [e["ms"] for e in state["history"] if isinstance(e.get("ms"), (int, float)) and e["ms"] > 0]
            tokens = [e.get("tokens", 0) for e in state["history"] if e.get("tokens", 0) > 0]
            cpu_vals = [e.get("cpu_pct", 0) for e in state["history"] if e.get("cpu_pct")]
            mem_vals = [e.get("mem_mb", 0) for e in state["history"] if e.get("mem_mb")]
            stats = {
                "session_id":       session_id,
                "uptime_s":         round(elapsed, 1),
                "model":            state["model"],
                "camera":           state["camera_info"],
                "total_frames":     state["frame_count"],
                "total_analyses":   state["analysis_count"],
                "log_dir":          str(session_dir),
                "inference": {
                    "avg_ms":     round(sum(times)/len(times), 1) if times else 0,
                    "min_ms":     min(times) if times else 0,
                    "max_ms":     max(times) if times else 0,
                    "median_ms":  round(sorted(times)[len(times)//2], 1) if times else 0,
                    "per_min":    round(len(times)/max(elapsed/60, 0.01), 1),
                },
                "tokens": {
                    "avg":  round(sum(tokens)/len(tokens), 1) if tokens else 0,
                    "total": sum(tokens),
                },
                "system": {
                    "cpu_avg_pct": round(sum(cpu_vals)/len(cpu_vals), 1) if cpu_vals else 0,
                    "mem_avg_mb":  round(sum(mem_vals)/len(mem_vals), 1) if mem_vals else 0,
                },
            }
            self._respond(200, "application/json", json.dumps(stats, indent=2).encode())

        elif self.path == "/export":
            data = json.dumps({
                "exported_at": datetime.now().isoformat(),
                "model": state["model"],
                "entries": state["history"],
            }, indent=2)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Disposition",
                             'attachment; filename="ollama_vision_history.json"')
            self.end_headers()
            self.wfile.write(data.encode())
        else:
            self._respond(404, "text/plain", b"Not found")

    def do_POST(self):
        if self.path == "/config":
            body = self._read_json()
            global prompt_text
            if "model"    in body: state["model"]    = body["model"]
            if "interval" in body: state["interval"] = max(0.5, float(body["interval"]))
            if "prompt"   in body:
                with prompt_lock:
                    prompt_text = body["prompt"]
            self._respond(200, "application/json", b'{"ok":true}')
        elif self.path == "/pause":
            state["paused"] = not state["paused"]
            self._respond(200, "application/json",
                          json.dumps({"paused": state["paused"]}).encode())
        else:
            self._respond(404, "text/plain", b"Not found")

    def _read_json(self):
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length)) if length else {}

    def _respond(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.end_headers()
        self.wfile.write(body)


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ── Set up logging ──
    session_id, session_dir, log_file, summary_file, perf_file, logger = setup_logging()
    session_start_time = time.time()

    logger.info("🔍 Detecting Ollama models…")
    available = detect_models()
    state["available_models"] = available if available else PREFERRED_MODELS

    if available:
        best = pick_best_model(available)
        state["model"] = best
        logger.info(f"   Models: {', '.join(available)}")
        logger.info(f"   Selected: {best}")
    else:
        state["model"] = "llava"
        logger.warning("   ⚠ Cannot reach Ollama. Defaulting to llava.")
        logger.info("   Start it with:  ollama serve")

    # Pre-warm: loads model weights into RAM (first inference is slow otherwise)
    prewarm_model(state["model"])

    logger.info(f"""
╔══════════════════════════════════════════════════╗
║         Ollama Vision — Optimized                ║
╠══════════════════════════════════════════════════╣
║  Model     : {state['model']:<35}║
║  Web UI    : http://localhost:{WEB_PORT:<18}║
║  AI image  : {AI_IMAGE_WIDTH}px wide, enhanced, sharpest-frame  ║
║  Camera    : 1280x720 MJPG target                ║
║  Streaming : tokens appear live                  ║
║  Interval  : {MIN_INTERVAL}s min (continuous analysis)       ║
╠══════════════════════════════════════════════════╣
║  Logs      : {str(session_dir):<35}║
║  Stats API : http://localhost:{WEB_PORT}/stats{' '*13}║
║  Press Ctrl+C to stop                             ║
╚══════════════════════════════════════════════════╝
""")

    threading.Thread(target=camera_loop,   daemon=True, name="camera").start()
    threading.Thread(target=analysis_loop, daemon=True, name="analysis").start()

    server = HTTPServer(("0.0.0.0", WEB_PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        state["running"] = False
        logger.info("")
        save_session_summary()
        logger.info("👋 Stopped.")
