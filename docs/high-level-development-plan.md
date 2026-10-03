# AI Game Agent — High-Level Development Plan

## 1. Objective

Build a local AI-driven computer agent capable of observing a game through screen capture, understanding the current game state, deciding what action should be taken, and controlling the mouse and keyboard.

The initial target is **World of Warcraft**, but the architecture should remain sufficiently generic that the same agent could eventually operate other GUI-based games.

The agent should operate entirely locally where practical, using the available RTX 5090 and Ollama-hosted vision/language models.

### Core loop

```text
┌─────────────────┐
│     Game        │
└────────┬────────┘
         │
         │ screenshot
         ▼
┌─────────────────┐
│ Perception      │
│ CV / OCR / VLM  │
└────────┬────────┘
         │
         │ game state
         ▼
┌─────────────────┐
│ State Manager   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Decision Engine │
│ Rules + AI      │
└────────┬────────┘
         │
         │ action
         ▼
┌─────────────────┐
│ Action Executor │
│ Mouse / Keyboard│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│     Game        │
└─────────────────┘
```

The system should operate as a continuous observe → understand → decide → act → observe loop.

---

# 2. Design Principles

## 2.1 Local-first

Prefer local processing wherever practical.

Primary components:

* Local screen capture
* Local computer vision
* Local OCR
* Local VLM/LLM through Ollama
* Local action execution
* Local logging and telemetry

Cloud AI APIs should be optional rather than required.

## 2.2 AI should not control every action

The LLM/VLM should not be asked to make decisions on every frame.

Instead:

* deterministic code handles frequent/simple operations
* computer vision handles object detection
* state machines handle predictable behavior
* the AI handles interpretation, planning, and unusual situations

This reduces latency and GPU usage while improving reliability.

## 2.3 Modular architecture

The system should separate:

* screen acquisition
* perception
* game-state representation
* decision making
* planning
* action execution
* safety controls
* logging
* UI/debugging

A game-specific adapter should sit above the generic agent framework.

---

# 3. Proposed Technology Stack

## Runtime

* Windows 11
* Python
* Docker where useful
* Git
* VS Code

## AI

* Ollama
* Local multimodal model
* Local text/reasoning model
* Potential future support for multiple models

Candidate models should be benchmarked rather than hard-coded.

Possible architecture:

```text
Fast CV
   │
   ├── OpenCV
   ├── YOLO
   └── OCR
        │
        ▼
Game State
        │
        ▼
Local VLM / LLM
        │
        ▼
Structured Action
```

## Screen capture

Evaluate:

* MSS
* DXGI/Desktop Duplication
* Windows Graphics Capture

The capture mechanism should eventually support high-frequency screenshots without unnecessarily copying the entire desktop.

## Input

Provide an abstraction layer over:

* keyboard
* mouse movement
* mouse clicks
* mouse buttons
* key combinations
* key holds
* configurable delays

Possible initial implementations:

* PyAutoGUI
* PyDirectInput
* Windows SendInput API

The input layer should be replaceable without changing the rest of the system.

## Computer Vision

Potential components:

* OpenCV
* YOLO
* template matching
* image segmentation
* OCR
* VLM

Use traditional CV where deterministic recognition is sufficient.

Use the VLM when interpretation requires contextual reasoning.

---

# 4. System Architecture

## 4.1 Screen Capture

Responsible for acquiring the current game image.

Requirements:

* configurable capture region
* configurable frame rate
* optional downscaling
* screenshot timestamps
* optional frame differencing
* optional region-of-interest capture

Example:

```text
Desktop
   │
   ▼
Screen Capture
   │
   ├── Full frame
   ├── Game viewport
   ├── Minimap
   ├── Action bars
   └── UI regions
```

The agent should avoid sending the entire screen to the VLM when only a small region is relevant.

---

# 5. Perception Layer

The perception layer converts pixels into structured information.

Example:

```json
{
  "player": {
    "health": 0.87,
    "mana": 0.64,
    "position": null,
    "in_combat": true
  },
  "target": {
    "present": true,
    "health": 0.42,
    "hostile": true
  },
  "environment": {
    "npc_detected": false,
    "loot_detected": true
  },
  "ui": {
    "dead": false,
    "inventory_full": false
  }
}
```

## Detection mechanisms

### OCR

Useful for:

* quest text
* NPC names
* item names
* error messages
* combat messages
* UI state

### Template matching

Useful for:

* known UI icons
* buttons
* status indicators
* common visual states

### Object detection

Useful for:

* enemies
* NPCs
* interactable objects
* loot
* environmental landmarks

### VLM

Useful for:

* interpreting unfamiliar situations
* understanding screenshots holistically
* identifying UI elements
* deciding what visual information is relevant

---

# 6. Game State Manager

The state manager should maintain the agent's current understanding of the game.

Possible states:

```text
STARTING
IDLE
NAVIGATING
SEARCHING
TARGETING
COMBAT
LOOTING
RECOVERING
DEAD
INVENTORY_FULL
QUESTING
UNEXPECTED_STATE
PAUSED
EMERGENCY_STOP
```

The state machine should be deterministic wherever possible.

Example:

```text
IDLE
 │
 ├── enemy detected ──────► TARGETING
 │
 ├── loot detected ───────► LOOTING
 │
 ├── dead ────────────────► DEAD
 │
 └── unexpected ──────────► AI_ANALYSIS
```

---

# 7. Decision Engine

The decision engine combines:

1. deterministic rules
2. current game state
3. short-term objectives
4. AI reasoning

The AI should return structured actions rather than arbitrary text.

Example:

```json
{
  "action": "attack",
  "target": "nearest_hostile",
  "confidence": 0.94
}
```

Other possible actions:

```text
MOVE
TURN
TARGET
ATTACK
CAST
LOOT
INTERACT
WAIT
RETREAT
HEAL
OPEN_INVENTORY
CLOSE_WINDOW
ANALYZE_SCREEN
REQUEST_HUMAN
```

---

# 8. Planning Layer

Separate tactical decisions from longer-term goals.

Example:

```text
Goal:
    Grind mobs in a specified area

Plan:
    1. Navigate to area
    2. Locate suitable target
    3. Engage target
    4. Execute combat rotation
    5. Loot
    6. Check health/mana
    7. Find next target
    8. Repeat
```

The planner should be able to recover from interruptions.

For example:

```text
Combat
   │
   ├── player dies
   │       └──► recovery plan
   │
   ├── inventory full
   │       └──► inventory plan
   │
   ├── unexpected NPC
   │       └──► AI analysis
   │
   └── normal completion
           └──► next target
```

---

# 9. Action Executor

The action executor translates structured actions into real input.

Example:

```text
Decision:
    MOVE_FORWARD
    duration = 1.7 sec

        ↓

Action Executor

        ↓

Keyboard:
    W DOWN
    wait 1.7 sec
    W UP
```

The executor should provide:

* action queue
* cancellation
* priority actions
* configurable delays
* key-state tracking
* mouse-state tracking
* emergency release of all keys/buttons

The last item is particularly important.

If the agent crashes while holding a key, the executor should be able to release all currently held keys/buttons.

---

# 10. Safety System

The agent should have a hard emergency-stop mechanism independent of the AI.

Possible activation:

```text
F12
Ctrl + Alt + Pause
```

Emergency stop should:

1. stop the decision loop
2. cancel queued actions
3. release all keyboard keys
4. release mouse buttons
5. disable further input
6. record the event

Additional safeguards:

* maximum action duration
* maximum movement duration
* maximum number of consecutive actions
* inactivity timeout
* confidence thresholds
* human confirmation for dangerous/unknown states

---

# 11. Memory

The agent should maintain several levels of memory.

## Short-term memory

Current situation:

```text
Current target
Current combat
Recent actions
Recent screenshots
Current state
```

## Episodic memory

Useful events:

```text
At location X:
    mob type Y was encountered
    combat succeeded
    route took approximately N seconds
```

## Long-term knowledge

Game-specific information:

```text
Abilities
Cooldowns
NPC locations
Routes
Known enemies
Known UI layouts
Known recovery procedures
```

Initially this can be stored in JSON/YAML.

Later it could use SQLite or a vector database.

---

# 12. AI Interaction Protocol

The AI should communicate using structured schemas rather than free-form instructions.

Example:

```json
{
  "state": "COMBAT",
  "observation": {
    "player_hp": 0.72,
    "target_hp": 0.31,
    "target_distance": 8
  },
  "recommended_action": {
    "type": "CAST",
    "ability": "ABILITY_X"
  },
  "confidence": 0.91,
  "reason": "Target is within range and player has sufficient resource."
}
```

The application validates the response before executing it.

The AI must never directly execute arbitrary Python, shell commands, or OS commands.

---

# 13. Human-in-the-Loop Mode

Before autonomous operation, implement a mode where the AI proposes actions but does not execute them automatically.

Example:

```text
AI:
    Target detected.
    Recommended action: ATTACK

[Execute] [Reject] [Pause]
```

This allows the agent to be tested safely.

Modes:

```text
OBSERVE_ONLY
ASSISTED
SEMI_AUTONOMOUS
AUTONOMOUS
```

---

# 14. Logging and Replay

Every action should be logged.

Example:

```text
09:21:31 SCREEN
09:21:31 STATE = SEARCHING
09:21:32 TARGET DETECTED
09:21:32 AI = ATTACK
09:21:32 KEY = 1
09:21:34 TARGET HP = 0
09:21:34 LOOT DETECTED
09:21:35 ACTION = INTERACT
```

Store:

* timestamps
* screenshots where appropriate
* detected state
* AI decisions
* confidence
* executed actions
* errors
* recovery events

This makes debugging dramatically easier.

A replay mode should eventually allow the agent to process previously recorded sessions without controlling the real game.

---

# 15. Development Phases

## Phase 1 — Screen Observation

Build:

* screen capture
* configurable capture region
* screenshot viewer
* FPS measurement
* recording

Goal:

> Reliably observe the game without interacting with it.

---

## Phase 2 — Basic Computer Vision

Implement:

* UI detection
* OCR
* template matching
* basic object detection

Goal:

> Convert screenshots into useful structured information.

---

## Phase 3 — Input Control

Implement:

* mouse
* keyboard
* action queue
* key-state tracking
* emergency stop

Goal:

> Reliably perform deterministic actions.

---

## Phase 4 — State Machine

Implement:

* state detection
* transitions
* recovery
* timeouts

Goal:

> Create an autonomous deterministic agent without an LLM.

---

## Phase 5 — Local AI Integration

Connect the state manager to Ollama.

Start with:

```text
Screenshot
    ↓
VLM
    ↓
Structured observation
```

Then:

```text
Game State
    ↓
LLM
    ↓
Structured decision
```

Goal:

> Use AI only where deterministic logic is insufficient.

---

## Phase 6 — Assisted Mode

Implement:

```text
Observe
  ↓
Analyze
  ↓
Suggest action
  ↓
Human approval
  ↓
Execute
```

Goal:

> Validate AI decisions without fully autonomous control.

---

## Phase 7 — Autonomous Mode

Implement:

```text
Observe
  ↓
Understand
  ↓
Plan
  ↓
Act
  ↓
Verify
  ↓
Recover
```

Goal:

> Operate autonomously for extended periods while remaining observable and interruptible.

---

# 16. Performance Optimization

Once the prototype works, optimize the perception/decision loop.

Do not use the VLM for everything.

Target architecture:

```text
              Screenshot
                  │
                  ▼
             Fast CV layer
                  │
          ┌───────┴────────┐
          │                │
       Known state       Unknown
          │                │
          ▼                ▼
      State machine       VLM
          │                │
          └───────┬────────┘
                  ▼
             Action plan
                  │
                  ▼
             Input layer
```

Potential optimization:

* CV: 10–30 Hz
* state machine: 10–30 Hz
* VLM: only when necessary
* LLM planning: seconds/minutes rather than every frame

---

# 17. Generic Agent API

The long-term goal should be a generic interface.

```python
class GameAdapter:
    def observe(self) -> GameState:
        ...

    def available_actions(self) -> list[Action]:
        ...

    def execute(self, action: Action):
        ...

    def is_safe_to_continue(self) -> bool:
        ...
```

Then WoW becomes one implementation:

```text
GameAgent
    │
    ├── WoWAdapter
    ├── GameAdapter2
    └── GameAdapter3
```

This keeps the core computer-use framework independent of any particular game.

---

# 18. Suggested Project Structure

```text
ai-game-agent/
│
├── agent/
│   ├── agent.py
│   ├── planner.py
│   ├── decision_engine.py
│   └── memory.py
│
├── perception/
│   ├── capture.py
│   ├── opencv.py
│   ├── ocr.py
│   ├── object_detection.py
│   └── vlm.py
│
├── state/
│   ├── state_machine.py
│   ├── game_state.py
│   └── transitions.py
│
├── actions/
│   ├── executor.py
│   ├── keyboard.py
│   ├── mouse.py
│   └── safety.py
│
├── games/
│   ├── base.py
│   └── wow/
│       ├── adapter.py
│       ├── perception.py
│       ├── states.py
│       └── actions.py
│
├── ai/
│   ├── ollama.py
│   ├── prompts.py
│   └── schemas.py
│
├── ui/
│   └── dashboard.py
│
├── logging/
│   ├── events.py
│   └── recorder.py
│
├── tests/
│
├── config/
│
└── main.py
```

---

# 19. Initial MVP

The first useful prototype should be deliberately small.

### MVP capabilities

1. Capture the WoW window.
2. Display/record screenshots.
3. Detect player/UI regions.
4. Detect a target.
5. Read basic health information.
6. Detect whether the player is in combat.
7. Execute a small set of keyboard/mouse actions.
8. Implement an emergency stop.
9. Implement a deterministic state machine.
10. Ask the local VLM for help when the state becomes ambiguous.
11. Log every observation and action.

The MVP should **not** initially attempt full autonomous gameplay.

---

# 20. Success Criteria

The project should be evaluated using measurable metrics.

### Perception

* target detection accuracy
* UI recognition accuracy
* OCR accuracy
* state detection accuracy

### Decision making

* valid action rate
* invalid action rate
* recovery success rate
* AI confidence calibration

### Performance

* screenshot latency
* perception latency
* VLM latency
* end-to-end action latency
* GPU utilization
* VRAM usage

### Reliability

* autonomous runtime
* unexpected-state recovery
* crash recovery
* emergency-stop reliability

---

# 21. Future Extensions

Once the basic system works:

* minimap/navigation understanding
* visual path planning
* learned navigation
* automatic quest interpretation
* inventory management
* NPC interaction
* long-term objectives
* persistent world model
* reinforcement learning
* multimodal memory
* multiple cooperating AI agents
* web/game knowledge retrieval
* GUI dashboard
* remote monitoring
* session replay
* automatic performance benchmarking

---

# 22. Long-Term Architecture

The eventual system could evolve into a general-purpose **local computer-use agent for games**:

```text
                         ┌──────────────────┐
                         │    Human User    │
                         └────────┬─────────┘
                                  │
                            Goals / commands
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │     Agent Orchestrator   │
                    └────────────┬─────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
        Perception            Memory             Planner
              │                  │                  │
              └──────────────────┼──────────────────┘
                                 │
                                 ▼
                       ┌──────────────────┐
                       │ Decision Engine  │
                       └────────┬─────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │ Action Executor  │
                       └────────┬─────────┘
                                │
                                ▼
                         ┌──────────────┐
                         │ Game / GUI   │
                         └──────┬───────┘
                                │
                                └──────► Observation
```

The key architectural objective is to make the **game itself just another environment**.

The same agent framework could eventually observe and operate:

* games
* desktop applications
* web applications
* emulators
* development tools
* other graphical environments

while the environment-specific adapter provides the knowledge required to interpret and operate each application.

---

# 23. Recommended First Milestone

Build a **read-only WoW observer** first.

It should:

```text
WoW
 │
 ▼
Screen Capture
 │
 ▼
Screenshot
 │
 ├── OpenCV
 ├── OCR
 └── Local VLM
       │
       ▼
   Game State
       │
       ▼
   Dashboard
```

No keyboard or mouse control should initially be enabled.

Once the system can reliably answer:

> "What is happening on the screen right now?"

the control layer can be added.

This separation will make debugging considerably easier because perception errors and action errors can be isolated independently.
