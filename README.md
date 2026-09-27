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

> **Note:** Automated interaction with online games may violate the terms of service of the game being controlled. This project is primarily intended as an exploration of local AI agents, computer vision, multimodal reasoning, and GUI automation.
