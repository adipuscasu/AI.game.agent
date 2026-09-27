#!/usr/bin/env bash
# validate-perception.sh
# PostToolUse hook (non-Windows). Mirrors validate-perception.ps1: when a file
# under src/ai_game_agent/perception/ is created or edited, auto-run ruff on it
# and the matching test file, then report the result back. Non-blocking:
# always exits 0; findings are delivered via the `systemMessage` output field.
set -u

raw="$(cat)"
[ -z "$raw" ] && exit 0

path="$(printf '%s' "$raw" | python3 -c '
import sys, json
try:
    ti = json.load(sys.stdin).get("tool_input", {}) or {}
except Exception:
    sys.exit(0)
for k in ("filePath", "path", "file_path", "target_file"):
    if ti.get(k):
        print(ti[k]); break
')"
[ -z "$path" ] && exit 0

case "$path" in
  */src/ai_game_agent/perception/*) : ;;
  *) exit 0 ;;
esac

stem="$(basename "$path")"; stem="${stem%.*}"
declare -A testMap=(
  [template]=test_template_matcher.py
  [ui]=test_ui_zones.py
  [objects]=test_objects.py
  [ocr]=test_ocr.py
  [pipeline]=test_pipeline.py
  [observation]=test_observation.py
)
testFile="${testMap[$stem]:-test_${stem}.py}"

notes=()
if [ -f "$path" ]; then
  if out="$(uv run ruff check "$path" 2>&1)" && [ $? -eq 0 ]; then
    notes+=("ruff: OK")
  else
    notes+=("ruff: FAIL -> $out")
  fi
fi

if [ -f "tests/$testFile" ]; then
  if out="$(uv run pytest "tests/$testFile" -q 2>&1)" && [ $? -eq 0 ]; then
    notes+=("pytest $testFile: OK")
  else
    notes+=("pytest $testFile: FAIL -> $out")
  fi
else
  notes+=("pytest: $testFile not found yet - TDD: write the red test first")
fi

msg="Perception file touched (${stem}). Auto-checks:
$(printf '%s\n' "${notes[@]}")"

printf '%s' "$msg" | python3 -c 'import sys, json; print(json.dumps({"continue": True, "systemMessage": sys.stdin.read()}))'
exit 0
