# validate-perception.ps1
# PostToolUse hook (Windows). When a file under src/ai_game_agent/perception/ is
# created or edited, auto-run ruff on it and the matching test file, then report
# the result back to the agent. Non-blocking: always exits 0; findings are
# delivered via the `systemMessage` output field.

$ErrorActionPreference = 'Continue'

$raw = [Console]::In.ReadToEnd()
if (-not $raw) { exit 0 }

$toolInput = $null
try {
    $data = $raw | ConvertFrom-Json
    $toolInput = $data.tool_input
} catch { exit 0 }

if (-not $toolInput) { exit 0 }

# Pull the target file path from common tool-input keys.
$path = $null
foreach ($k in @('filePath', 'path', 'file_path', 'target_file')) {
    if ($toolInput.PSObject.Properties.Name -contains $k) {
        $path = [string]$toolInput.$k
        if ($path) { break }
    }
}
if (-not $path) { exit 0 }

# Only act on perception-layer files.
$normalized = ($path -replace '\\', '/').ToLower()
if ($normalized -notmatch 'src/ai_game_agent/perception/') { exit 0 }

# Map a perception module stem to its test file.
$stem = [System.IO.Path]::GetFileNameWithoutExtension((Split-Path $path -Leaf))
$testMap = @{
    template    = 'test_template_matcher.py'
    ui          = 'test_ui_zones.py'
    objects     = 'test_objects.py'
    ocr         = 'test_ocr.py'
    pipeline    = 'test_pipeline.py'
    observation = 'test_observation.py'
}
$testFile = if ($testMap.ContainsKey($stem)) { $testMap[$stem] } else { "test_$stem.py" }

$notes = [System.Collections.Generic.List[string]]::new()

# ruff check on the touched file.
if (Test-Path $path) {
    $ruffOut = (& uv run ruff check $path 2>&1 | Out-String).Trim()
    if ($LASTEXITCODE -ne 0) { $notes.Add("ruff: FAIL -> $ruffOut") }
    else { $notes.Add("ruff: OK") }
}

# pytest the matching test file (if present).
$testPath = Join-Path "tests" $testFile
if (Test-Path $testPath) {
    $pyOut = (& uv run pytest $testPath -q 2>&1 | Out-String).Trim()
    if ($LASTEXITCODE -ne 0) { $notes.Add("pytest ${testFile} FAIL -> $pyOut") }
    else { $notes.Add("pytest ${testFile} OK") }
} else {
    $notes.Add("pytest: ${testFile} not found yet - TDD: write the red test first")
}

$message = "Perception file touched ($stem). Auto-checks:`n" + ($notes -join "`n")

$result = [ordered]@{
    continue      = $true
    systemMessage = $message
}
$result | ConvertTo-Json -Compress | Write-Output
exit 0
