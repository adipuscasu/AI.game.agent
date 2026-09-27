# AGENTS.md

## Project Overview

This repository contains a local, AI-driven computer-use agent designed to observe graphical applications, understand their state, make decisions, and interact through mouse and keyboard input.

The initial target environment is **World of Warcraft**, but the architecture must remain generic enough to support additional games and GUI applications in the future.

The system combines:

* Screen capture
* Computer vision
* OCR
* Object detection
* Multimodal vision-language models
* Local LLM/VLM reasoning through Ollama
* Deterministic state machines
* Planning and decision-making
* Mouse and keyboard automation
* Persistent and episodic memory
* Safety controls
* Logging and replay

The primary development objective is to build a reliable **local computer-use agent**, not a collection of game-specific automation scripts.

---

# Core Engineering Principles

## 1. Local First

Prefer local execution whenever practical.

The preferred AI backend is **Ollama** running locally.

Do not introduce cloud AI APIs unless:

1. there is a clear technical reason,
2. the dependency is optional,
3. the local implementation cannot reasonably provide the required capability.

Never hard-code API keys, credentials, tokens, or other secrets.

---

## 2. Keep the Architecture Modular

Separate the following concerns:

```text
Screen Capture
      ↓
Perception
      ↓
Game State
      ↓
Decision / Planning
      ↓
Action Execution
```

Do not allow one layer to become tightly coupled to another.

For example:

* perception must not directly press keys
* the LLM must not directly execute OS commands
* game adapters must not contain generic screen-capture infrastructure
* the action executor must not contain game-specific decision logic

---

## 3. Prefer Deterministic Logic Over AI

Use deterministic code whenever the problem is deterministic.

Examples:

* keyboard timing
* action queues
* cooldown timers
* state transitions
* UI template matching
* known UI detection
* input validation
* safety limits

Use AI when interpretation or reasoning is genuinely required.

The preferred architecture is:

```text
Fast deterministic processing
            │
            ▼
       Known state?
        /       \
      yes        no
       │          │
       ▼          ▼
 State Machine    AI
       │          │
       └────┬─────┘
            ▼
       Action Plan
```

Do not send every screen frame to an LLM/VLM unless there is a demonstrated requirement for doing so.

---

# Repository Structure

Prefer the following logical separation:

```text
agent/
    Agent orchestration
    Planning
    Decision making
    Memory

perception/
    Screen capture
    OpenCV
    OCR
    Object detection
    VLM integration

state/
    Game state
    State machine
    State transitions

actions/
    Keyboard
    Mouse
    Action queue
    Safety controls

games/
    Generic game adapter
    Game-specific implementations

ai/
    Ollama integration
    Prompt definitions
    Structured AI schemas

ui/
    Monitoring/debugging UI

logging/
    Event logging
    Session recording
    Replay

tests/
    Unit tests
    Integration tests
    Replay tests
```

The exact directory structure may evolve, but responsibilities must remain clearly separated.

---

# Game Adapter Architecture

Game-specific code must live under the appropriate game adapter.

For example:

```text
games/
    base.py

    wow/
        adapter.py
        perception.py
        states.py
        actions.py
```

The generic agent must not contain assumptions specific to World of Warcraft.

Use abstractions such as:

```python
class GameAdapter:
    def observe(self) -> GameState:
        ...

    def available_actions(self) -> list[Action]:
        ...

    def execute(self, action: Action) -> None:
        ...

    def is_safe_to_continue(self) -> bool:
        ...
```

Additional games should be implementable without modifying the core agent.

---

# AI Integration

## Structured Output

AI models should communicate with the application through structured data.

Prefer:

```json
{
  "action": "ATTACK",
  "target": "nearest_hostile",
  "confidence": 0.94
}
```

over:

```text
I think we should probably attack the nearest enemy.
```

Validate all model output before it reaches the action executor.

Never blindly execute arbitrary model-generated:

* Python
* PowerShell
* shell commands
* filesystem operations
* OS commands
* mouse/keyboard API calls

The model proposes an action; application code validates and executes it.

---

# Ollama

Ollama is the preferred local model runtime.

Model names must be configurable.

Do not hard-code a specific model into application logic.

Example configuration:

```yaml
ai:
  provider: ollama
  model: qwen3.8:27b
  base_url: http://localhost:11434
```

Model-specific behavior should be isolated behind the AI abstraction.

The application should remain usable if the model changes.

---

# Vision Processing

Use the cheapest reliable perception mechanism.

Preferred order:

1. deterministic pixel/UI detection
2. template matching
3. OCR
4. object detection
5. VLM reasoning

Do not use a VLM to solve a problem that can be reliably solved with OpenCV.

For expensive visual operations:

* crop to relevant regions
* resize where appropriate
* avoid unnecessary frame processing
* cache results where possible
* process only when state changes

---

# State Machine

State transitions should be explicit and testable.

Example:

```text
IDLE
 │
 ├── target detected → TARGETING
 │
 ├── loot detected → LOOTING
 │
 ├── player dead → DEAD
 │
 └── unknown situation → AI_ANALYSIS
```

Avoid deeply nested conditionals when a state machine is more appropriate.

Each state should define:

* entry conditions
* allowed actions
* exit conditions
* timeout
* recovery behavior

Unexpected states must have an explicit fallback.

---

# Action Executor

All mouse and keyboard control must go through a single action-execution abstraction.

Do not call OS input APIs directly from:

* AI code
* perception code
* state-management code
* game-specific decision logic

The executor is responsible for:

* action validation
* action sequencing
* key/button state tracking
* timing
* cancellation
* timeouts
* emergency release

Example:

```text
Decision
   ↓
Action
   ↓
Validation
   ↓
Action Queue
   ↓
Executor
   ↓
Mouse / Keyboard
```

---

# Safety Requirements

Safety mechanisms are mandatory.

The agent must have an emergency-stop mechanism independent of the AI.

Emergency stop must:

1. stop the decision loop
2. cancel pending actions
3. release all held keyboard keys
4. release mouse buttons
5. prevent further input
6. record the stop event

Actions must have bounded durations.

Avoid unbounded:

```python
key_down("w")
```

without a corresponding timeout/release mechanism.

Prefer abstractions that guarantee release even if an exception occurs.

---

# Human-in-the-Loop

Support multiple operating modes:

```text
OBSERVE_ONLY
ASSISTED
SEMI_AUTONOMOUS
AUTONOMOUS
```

`OBSERVE_ONLY` should never generate mouse/keyboard input.

`ASSISTED` should allow the agent to propose actions without automatically executing them.

This mode should be used extensively during development and debugging.

---

# Logging

Important decisions and actions must be observable.

Log:

* timestamps
* current state
* perception results
* AI requests
* AI responses
* confidence
* selected actions
* executed actions
* errors
* recovery events
* emergency stops

Avoid logging:

* API keys
* authentication tokens
* passwords
* personal information
* unnecessary sensitive data

Logs should make it possible to answer:

> Why did the agent perform this action?

---

# Replayability

Where practical, record enough information to reproduce an agent decision.

A replay session should be able to feed recorded observations into:

```text
Perception
    ↓
State Machine
    ↓
Decision Engine
```

without controlling the real application.

Replay testing should become the primary mechanism for regression testing perception and decision logic.

---

# Testing

## Test-Driven Development (TDD)

This project is developed using **Test-Driven Development (TDD)**.

TDD is mandatory for all new features and bug fixes, not optional:

1. **Red** — write a failing test that describes the desired behavior before writing the implementation.
2. **Green** — write the minimum code required to make the test pass.
3. **Refactor** — improve the implementation while keeping all tests green.

Additional TDD rules:

* Do not write production code without a corresponding failing test first.
* Do not commit an implementation without its tests passing.
* Bug fixes require a regression test that fails before the fix and passes after it.
* Tests define the expected behavior; production code must conform to the tests, not the other way around.
* Tests must be deterministic, independent, and runnable in CI without a desktop session or a live game.

## Unit Tests

Test independently:

* state transitions
* action validation
* action serialization
* AI response parsing
* configuration
* safety mechanisms
* game-state transformations

## Integration Tests

Test:

* Ollama communication
* screen capture
* perception pipeline
* action executor
* game adapter

## Replay Tests

Prefer recorded screenshots/session data over requiring the actual game to run in CI.

CI must not depend on an interactive desktop session.

---

# Error Handling

Never silently ignore errors.

Errors should be classified where possible:

```text
PERCEPTION_ERROR
AI_ERROR
MODEL_TIMEOUT
INVALID_AI_RESPONSE
ACTION_ERROR
INPUT_ERROR
STATE_ERROR
GAME_ADAPTER_ERROR
CONFIGURATION_ERROR
```

Recoverable errors should have bounded retry behavior.

Do not create infinite retry loops.

If recovery fails, transition to a safe paused state.

---

# Performance

Performance matters because perception and AI inference may run continuously.

Measure rather than guess.

Track:

* screen capture latency
* perception latency
* OCR latency
* object detection latency
* VLM latency
* LLM latency
* action latency
* GPU utilization
* VRAM usage
* CPU utilization
* memory usage

Do not optimize prematurely, but preserve clear boundaries that allow optimization later.

---

# Configuration

Runtime configuration belongs outside source code.

Examples:

```text
config/
    default.yaml
    development.yaml
```

Configuration should cover:

* AI provider
* model
* Ollama endpoint
* capture region
* capture frequency
* confidence thresholds
* action timeouts
* safety settings
* logging
* game-specific parameters

Never commit secrets.

Provide configuration examples instead.

---

# Dependencies

Keep dependencies minimal.

Before adding a dependency:

1. determine whether the functionality is already available
2. evaluate maintenance status
3. consider platform compatibility
4. consider licensing
5. consider runtime overhead
6. determine whether the dependency is actually necessary

Do not add large frameworks for small pieces of functionality.

---

# Code Quality

Prefer:

* small focused modules
* explicit interfaces
* type hints
* meaningful names
* deterministic behavior
* dependency injection where appropriate
* testable functions
* clear error handling

Avoid:

* global mutable state
* hidden side effects
* magic constants
* duplicated input logic
* direct AI-to-OS control
* game-specific assumptions in generic modules

---

# Documentation

Document architectural decisions that are not obvious from the code.

When introducing a significant architectural change, update the relevant documentation.

Important documentation should include:

* setup
* architecture
* configuration
* supported models
* development workflow
* testing
* troubleshooting
* game adapters
* safety controls

Prefer diagrams for complex control flows.

---

# Git Practices

Keep commits focused.

Prefer:

```text
feat: add screen capture abstraction
feat: add Ollama VLM client
feat: implement combat state machine
fix: release held keys on action timeout
test: add replay tests for target detection
```

Avoid large commits mixing unrelated changes.

Do not commit:

* credentials
* local model files
* game recordings unless explicitly intended
* large generated artifacts
* machine-specific configuration
* temporary debugging files

---

# Development Workflow

Before implementing a feature (TDD):

1. Understand the relevant architecture.
2. Identify the correct layer.
3. Check existing abstractions.
4. Avoid duplicating functionality.
5. Define the smallest useful change.
6. Write a failing test for the desired behavior (red).
7. Implement the minimum code to make the test pass (green).
8. Refactor while keeping tests green.
9. Run the full relevant test suite.
10. Update documentation if the architecture changed.

When debugging:

1. reproduce the problem
2. capture relevant logs/state
3. determine which layer is responsible
4. fix the underlying abstraction
5. add a regression test where practical

Do not work around architectural problems by adding special cases to unrelated components.

---

# AI Agent Development Rules

When modifying this repository, an AI coding agent should:

* inspect existing code before creating new abstractions
* preserve existing public interfaces unless there is a reason to change them
* avoid unnecessary rewrites
* keep changes localized
* explain significant architectural changes
* follow TDD: write a failing test before implementation, then the minimum code to pass
* never introduce credentials
* never disable safety mechanisms to make a test pass
* never bypass the action-executor abstraction
* never allow model output to execute arbitrary code
* prefer deterministic solutions where appropriate

The coding agent should treat AI-generated code as untrusted until validated by tests and review.

---

# Definition of Done

A feature is considered complete when:

* the implementation is integrated into the correct architectural layer
* public interfaces are documented
* errors are handled
* relevant tests exist, were written first (TDD), and pass
* the full relevant test suite is green
* safety requirements are satisfied
* no secrets are introduced
* logging is sufficient for debugging
* configuration is externalized where appropriate
* existing functionality continues to work
* documentation is updated when necessary

---

# Primary Goal

Build a robust, modular **local computer-use agent** capable of understanding and interacting with graphical environments.

World of Warcraft is the initial environment, not the architectural boundary.

The long-term objective is a reusable framework in which:

```text
             Generic AI Agent
                    │
        ┌───────────┼───────────┐
        │           │           │
       WoW        Game B      Game C
        │           │           │
        ▼           ▼           ▼
     Adapter      Adapter     Adapter
```

The core agent should remain independent of any particular game whenever practical.
