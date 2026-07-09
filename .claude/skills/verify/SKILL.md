---
name: verify
description: How to verify the KROK TUG-test tracker — run the Playwright e2e suite; manual preview recipe for visual checks.
---

# Verifying KROK

**Primary: `npm test`** (Playwright, ~15s, offline-safe). The suite in
`tests/tug.spec.js` drives the full TUG flow in `movenet.html` with a stubbed
camera/detector and Playwright's fake clock: voice phrases + beeps per phase,
start gate (calibration refuses a standing person, re-prompts), end gate
(crouching at the chair ≠ seated), both 30s timeouts, manual stop, restart,
auto-download of the JSON, and fullscreen HUD layout on desktop/mobile viewports.

Key mechanics if you extend the tests:
- Stubs go in `page.addInitScript` (getUserMedia → canvas.captureStream, `tf`,
  `poseDetection`); CDN requests are route-aborted. The detector returns
  `window.__pose`, which tests set per phase via `setPose(page, POSES.x)`.
- Poses MUST include knees (kp 13/14): the FSM's `isSeated()` compares
  knees-vs-hips distance to torso length (seated ≈ knees at hip level).
  Baseline geometry: torso 100px, seated hips y=400/knees 410, standing
  hips 340/knees 440, stand threshold 385.
- `page.clock.install()` AFTER load (camera/model init needs the real clock),
  BEFORE clicking Start; advance with `clock.runFor` in ≤1s steps so rAF and
  EMA smoothing (alpha 0.2) actually tick.
- `detector`/`startTime` are top-level `let` — unreachable via `window`; only
  function declarations (`init`, `playBeep`, `speak`) and the CDN globals can
  be replaced from outside. That's why stubs replace `poseDetection.createDetector`.

Manual visual check: `python3 -m http.server 8742` (or `preview_start` with
`krok-static` from `.claude/launch.json`) → open `/movenet.html`. The preview
browser denies the camera — the same addInitScript-style stubs pasted into
`preview_eval` + re-calling global `init()` get past it.
