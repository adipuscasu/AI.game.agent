# Single mock screenshot → expect "saved: ...\*.png" and exit 0
```
uv run python -m ai_game_agent capture --backend mock --out shots
```
$LASTEXITCODE   # → 0

# Region + scale pipeline: 32×16 region scaled 0.5 → expect a 16×8 PNG
uv run python -m ai_game_agent capture --backend mock --region 10,20,32,16 --scale 0.5 --out shots

# Observe loop, no pacing, 5 frames → expect "frames=5 fps=..."
uv run python -m ai_game_agent observe --backend mock --frames 5 --fps 0

# Recording + rotation: 10 frames, keep max 3 → exactly 3 PNGs must remain in rec/
uv run python -m ai_game_agent observe --backend mock --frames 10 --fps 0 --record --record-max-files 3 --out rec

2. Real capture (mss backend, needs a display)
```
uv run python -m ai_game_agent capture --backend mss --out shots
```
uv run python -m ai_game_agent capture --backend mss --region 0,0,3840,2160 --out shots
uv run python -m ai_game_agent observe --backend mss --frames 30 --fps 30

> **Note (this PC — 2 monitors):** By default `capture` grabs the *primary*
> monitor only (3840 × 2160), which is what you usually want — no black bands
> around the secondary. To grab a different monitor use `--monitor`
> (1 = primary, 2+ = the (n-1)th secondary, 0 = the whole 5760×2160 virtual
> screen):
>
> ```powershell
> uv run python -m ai_game_agent capture --backend mss --monitor 2 --out shots   # secondary (Dell U2412M, 1920×1200)
> uv run python -m ai_game_agent capture --backend mss --monitor 0 --out shots   # whole virtual screen
> ```
>
> If a *primary*-monitor capture is also fully black, disable DP HDR in the
> Windows display settings — HDR mode can make screen capture render black.

> **mss 10.x compatibility:** `MssBackend` works with mss 10+ (verified on
> 10.2.0): `grab()` passes `monitors[<selector>]` explicitly (the monitor
> argument is required in mss 10) and handles both the dict-style `shot.size`
> (mss 9) and the `Size` object (mss 10+). If you see
> `MSS.grab() missing 1 required positional argument: 'monitor'` or
> `tuple indices must be integers`, the running code predates that fix —
> update the package, not your command.