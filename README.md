# AI.game.agent
# AI Game Agent

A local, AI-driven computer-use agent that observes games through screen capture, understands game state using computer vision and multimodal AI, and can interact with the game through mouse and keyboard input.

The project combines traditional computer vision, OCR, state machines, local LLM/VLM reasoning, and automated input to create a modular autonomous game agent.

The initial target is **World of Warcraft**, with the long-term goal of providing a generic framework that can operate different GUI-based games and applications.

## Key Features

* 🖥️ Real-time screen capture and visual perception
* 👁️ Computer vision, OCR, object detection and multimodal AI
* 🧠 Local LLM/VLM reasoning through Ollama
* 🎮 Mouse and keyboard automation
* 🔄 State-machine-based gameplay control
* 🗺️ Planning, navigation and recovery
* 🧠 Short-term and persistent agent memory
* 🛑 Independent emergency-stop and safety mechanisms
* 📊 Detailed decision and action logging

## Testing and Development Workflow

The project follows a strict Test-Driven Development (TDD) lifecycle.

**Unit and Integration Testing:**
The unit and integration test suite is managed by `pytest`. The stable command to run all tests is:

\`\`\`bash
uv run pytest
\`\`\`

The currently detected version of `pytest` is 9.1.1. Remember to run tests from the root directory of the workspace.

* 🔬 Replay and testing of recorded sessions
* 🧩 Game-independent core with pluggable game adapters
* 🏠 Designed for local execution and privacy

## Architecture

```text
Game
  │
  ▼
Screen Capture
  │
  ▼
Perception
(CV / OCR / VLM)
  │
  ▼
Game State
  │
  ▼
Decision & Planning
(Rules + AI)
  │
  ▼
Action Executor
(Mouse / Keyboard)
  │
  ▼
Game
```

The system is intentionally designed so that fast, deterministic computer-vision and state-machine logic handles routine operations, while AI models are used for higher-level interpretation, planning, and unexpected situations.

## Goals

The project aims to explore how modern multimodal AI and local computer-use agents can interact with complex graphical environments without requiring direct access to a game's internal APIs or memory.

The architecture is intended to remain generic enough to support additional games and GUI applications beyond the initial World of Warcraft implementation.

## Development Setup

The project is managed with [uv](https://docs.astral.sh/uv/) and requires Python 3.11+.

```powershell
# Create the virtual environment and install the package with dev dependencies
uv sync --extra dev
```

`uv sync` installs the project in editable mode into `.venv/` and activates it for all `uv run` commands below.

## Phase 1 — Screen Observation

The first milestone of the project (see `docs/high-level-development-plan.md`,
section 15) ships with this release:

* **Screen capture** — `Capture` facade over a swappable backend. The `mss`
  backend captures real screens on Windows; a deterministic `mock` backend
  supports CI and replay.
* **Configurable capture region** — `capture.region: {x, y, width, height}`
  in `config/default.yaml` selects a sub-rectangle.
* **Optional downscaling** — `capture.scale` (1.0 = no scaling) resizes each
  frame with deterministic nearest-neighbor sampling after region extraction.
* **FPS pacing + measurement** — `Capture.wait_next_frame()` paces the loop
  per `capture.fps`; `FpsMeter` reports the measured rate over a run.
* **Recording** — `Recorder` writes each frame as PNG to `record.directory`,
  rotating out the oldest files beyond `record.max_files`.
* **Screenshot viewer (CLI)** — `ai-game-agent` exposes three subcommands:
  `capture` (single frame), `observe` (loop + measured FPS), and
  `analyze` (one frame through the Phase 2 perception pipeline, JSON out).

### CLI usage

```powershell
# One screenshot from the primary monitor, saved to ./screenshots
uv run ai-game-agent capture --backend mss --out shots/

# Select a different monitor (1=primary default, 2+=secondary, 0=whole virtual screen)
uv run ai-game-agent capture --backend mss --monitor 2 --out shots/

# A 30-frame capture loop with measured FPS (no recording)
uv run ai-game-agent observe --backend mss --frames 30 --fps 30

# Headless / CI: mock backend, no pacing, record 5 frames
uv run ai-game-agent observe --backend mock --frames 5 --fps 0 --record --out rec/

# Region of interest (x,y,width,height)
uv run ai-game-agent capture --backend mss --region 0,0,1920,1080 --out shots/
```

### Phase 2: perception (`analyze`)

`analyze` captures one frame and runs it through the perception pipeline —
UI region detection, template matching, OCR, and color-blob object detection
— printing the structured `Observation` as JSON on stdout.

```powershell
# One frame from the mock backend, printed as JSON (CI-friendly, headless)
uv run ai-game-agent analyze --backend mock --fps 0

# Real capture from screen region 640,360 at 1920x1080, scaled 0.5x
uv run ai-game-agent analyze --backend mss --region 640,360,1920,1080 --scale 0.5

# Disable individual subsystems (OCR needs the "ocr" extra + Tesseract)
uv run ai-game-agent analyze --no-ocr --no-objects --no-templates
```

Shared options with `capture`/`observe`: `--backend` (default `mss`), `--monitor` (1=primary default, 2+=secondary, 0=whole virtual screen), `--config`,
`--region`, `--scale`, `--fps`, `--no-templates`, `--no-ocr`, `--no-objects`,
`--pretty`, `-o/--out`. Errors go to `stderr` with an `error:` prefix and exit 1.

`--fps 0` disables pacing (no sleeps) and is the right choice for tests,
CI, and fast debug runs. All CLI options default to the values in
`config/default.yaml` where applicable.

## Running the Unit Tests

All commands are run from the repository root.

### Run the full test suite

```powershell
uv run pytest
```

### Run a single test file

```powershell
uv run pytest tests/test_config.py
```

### Run a single test by node ID

```powershell
uv run pytest tests/test_capture.py::test_capture_applies_region
```

### Run only tests matching a keyword

```powershell
uv run pytest -k region
```

### Verbose output with traceback detail

```powershell
uv run pytest -v --tb=short
```

### Stop at the first failure (TDD red/green loop)

```powershell
uv run pytest -x
```

### End-to-end tests

```powershell
uv run pytest -m e2e       # only E2E
uv run pytest -m "not e2e" # unit + integration only
```

The E2E suite (`tests/e2e/`) spawns the real CLI (`python -m ai_game_agent`) as
a subprocess and asserts on exit codes, stdout/stderr routing, and files on
disk — the observable contract a user or CI job experiences. It is headless
(`--backend mock --fps 0`), so it runs in CI without a desktop session.

Notes:

* Tests are deterministic and headless — they use the `mock` capture backend and injected clocks, so they run in CI without a display, a game, or Ollama.
* The package is laid out as `src/ai_game_agent`; pytest picks this up automatically via `pythonpath = ["src"]` in `pyproject.toml`, so no manual `pip install` step is required when using `uv run`.
* If you are not using uv, the equivalent flow is: `python -m venv .venv`, activate the venv, `pip install -e ".[dev]"`, then `pytest`.

## Configuration

Runtime configuration lives in `config/` (YAML) and is loaded through `ai_game_agent.config.load_config()`. See `config/default.yaml` for the full schema with comments.

> **Note:** Automated interaction with online games may violate the terms of service of the game being controlled. This project is primarily intended as an exploration of local AI agents, computer vision, multimodal reasoning, and GUI automation.
