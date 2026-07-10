#!/usr/bin/env python3
"""
TUG (Timed Up and Go) analyser — JSON in, JSON out.

Accepts a MoveNet pose recording and returns a full TUG report as JSON.

  Input : a JSON array of frames
          [{ "timestamp": <ms>, "keypoints": [[x,y] x17] }, ...]
          COCO-17 keypoint order, image pixels, y grows downward.
          (also accepts {"frames":[...]} or {"poses":[...]} wrappers)

  Output: a JSON object with the TUG time, phase breakdown, gait, turn and
          symmetry metrics — see build_report() for the exact shape.

Usage
  python3 tug.py recording.json                 # -> JSON on stdout
  python3 tug.py recording.json -o report.json  # -> JSON to a file
  cat recording.json | python3 tug.py           # -> reads stdin
  python3 tug.py rec.json --distance 3          # override walk distance (m)

As a library
  from tug import analyze
  report = analyze(frames, walk_distance_m=3.0)
"""
import json
import sys
import math
import statistics

# ---- COCO-17 keypoint indices --------------------------------------------
NOSE = 0
LSH, RSH = 5, 6
LHIP, RHIP = 11, 12
LKN, RKN = 13, 14
LAN, RAN = 15, 16

DEFAULT_WALK_DISTANCE_M = 3.0   # standard TUG walk each way


# ---- small signal helpers -------------------------------------------------
def _median_filter(seq, w=3):
    out, n = [], len(seq)
    for i in range(n):
        a = max(0, i - w)
        b = min(n, i + w + 1)
        out.append(statistics.median(seq[a:b]))
    return out


def _ema(x, alpha=0.25):
    out = list(x)
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


def _build_signals(frames):
    t = [f["timestamp"] for f in frames]

    def mid(i, j, axis):
        return [(f["keypoints"][i][axis] + f["keypoints"][j][axis]) / 2 for f in frames]

    hipX = _median_filter(mid(LHIP, RHIP, 0))
    hipY = _median_filter(mid(LHIP, RHIP, 1))
    shY = _median_filter(mid(LSH, RSH, 1))
    knY = _median_filter(mid(LKN, RKN, 1))
    torso = [max(hipY[i] - shY[i], 1.0) for i in range(len(frames))]
    # seated ratio: knees ~hip level -> seated (~0); hips a thigh above knees -> standing (~0.8)
    seated = _median_filter([(knY[i] - hipY[i]) / torso[i] for i in range(len(frames))], 4)
    # signed shoulder vector (x of left minus right): flips sign on the 180 turn
    shVec = _median_filter([f["keypoints"][LSH][0] - f["keypoints"][RSH][0] for f in frames], 4)
    return dict(t=t, hipX=hipX, hipY=hipY, torso=torso, seated=seated, shVec=shVec)


# ---- phase detection ------------------------------------------------------
def _detect_phases(s):
    t, hipX, seated, shVec = s["t"], s["hipX"], s["seated"], s["shVec"]
    n = len(t)
    hipXs = _ema(hipX, 0.2)

    # away peak = turn-around location
    i_peak = max(range(n), key=lambda i: hipXs[i])
    peakX = hipXs[i_peak]

    # chair X baseline, from clearly-seated frames only
    seated_x = [hipXs[i] for i in range(i_peak) if seated[i] < 0.45]
    baseX = statistics.median(seated_x) if seated_x else hipXs[0]
    span = max(peakX - baseX, 1.0)

    # chair departure: the walk-out is a sustained climb to the peak, so scan
    # backward from the peak to the last frame still near the chair (robust to
    # isolated early tracking glitches a forward scan would latch onto)
    dep_thr = baseX + 0.15 * span
    i_dep = 0
    for i in range(i_peak, 0, -1):
        if hipXs[i] < dep_thr:
            i_dep = i
            break
    # stand onset (TUG t0): last clearly-seated frame at/just before departure
    i_stand = i_dep
    for i in range(i_dep, 0, -1):
        if seated[i] < 0.45:
            i_stand = i
            break

    # return home: first frame back near the chair after the peak
    return_thr = baseX + 0.15 * span
    i_return = n - 1
    for i in range(i_peak, n):
        if hipXs[i] < return_thr:
            i_return = i
            break
    # sit completed: seated ratio drops and stays low after arriving home
    i_sit = n - 1
    for i in range(i_return, n):
        if seated[i] < 0.5 and statistics.median(seated[i:min(i + 10, n)]) < 0.55:
            i_sit = i
            break

    # turn: orientation sign flip of the smoothed shoulder vector, near the peak
    shv = _ema(shVec, 0.25)
    i_flip = i_peak
    for i in range(i_stand, n - 1):
        if shv[i] <= 0 and shv[i + 1] > 0:
            i_flip = i
            break
    # turn boundaries from hip-X velocity around the peak
    vel = [0.0] + [(hipXs[i] - hipXs[i - 1]) / max(t[i] - t[i - 1], 1) for i in range(1, n)]
    vels = _ema(vel, 0.15)
    vmax = max(vels[i_stand:i_peak]) if i_peak > i_stand else 1
    i_turn_start = i_peak
    for i in range(i_peak, i_stand, -1):
        if vels[i] > 0.35 * vmax:      # last frame still clearly advancing outward
            i_turn_start = i
            break
    vmin = min(vels[i_peak:i_return]) if i_return > i_peak else -1
    i_turn_end = i_peak
    for i in range(i_peak, i_return):
        if vels[i] < 0.35 * vmin:      # first frame clearly walking back
            i_turn_end = i
            break

    return dict(i_stand=i_stand, i_turn_start=i_turn_start, i_flip=i_flip,
                i_turn_end=i_turn_end, i_return=i_return, i_sit=i_sit,
                i_peak=i_peak, baseX=baseX, peakX=peakX, span=span)


# ---- derived metrics ------------------------------------------------------
def _count_steps(s, i0, i1):
    """Approx alternating foot events from L-R ankle vertical difference."""
    t, torso = s["t"], s["torso"]
    frames = s["_frames"]
    if i1 - i0 < 6:
        return 0
    diff = _ema([(frames[i]["keypoints"][LAN][1] - frames[i]["keypoints"][RAN][1]) / torso[i]
                 for i in range(len(t))], 0.35)
    seg = diff[i0:i1]
    m = statistics.median(seg)
    sign = [1 if v - m > 0 else -1 for v in seg]
    thr, last, cnt = 0.04, sign[0], 0
    for k in range(1, len(seg)):
        if sign[k] != last and abs(seg[k] - m) > thr:
            cnt += 1
            last = sign[k]
    return cnt


def _limb_symmetry(frames, i0, i1):
    """L/R symmetry (%) from vertical knee & ankle excursion during gait."""
    if i1 - i0 < 8:
        return None, None

    def excursion(idx):
        vals = []
        for f in frames[i0:i1]:
            k = f["keypoints"]
            hipY = (k[LHIP][1] + k[RHIP][1]) / 2
            shY = (k[LSH][1] + k[RSH][1]) / 2
            torso = max(hipY - shY, 1.0)
            vals.append((k[idx][1] - hipY) / torso)
        return statistics.pstdev(_median_filter(vals, 2))

    def sym(a, b):
        return round(100 * (1 - abs(a - b) / max(a + b, 1e-9)))

    return sym(excursion(LKN), excursion(RKN)), sym(excursion(LAN), excursion(RAN))


def _trunk_tilt(frames, i0, i1):
    """Median absolute lateral trunk lean (deg); 0 = level shoulders."""
    if i1 - i0 < 4:
        return None
    angs = []
    for f in frames[i0:i1]:
        k = f["keypoints"]
        dx = abs(k[LSH][0] - k[RSH][0])   # abs -> tilt vs horizontal, facing-independent
        dy = k[LSH][1] - k[RSH][1]
        if dx < 1:
            continue
        angs.append(abs(math.degrees(math.atan2(dy, dx))))
    return round(statistics.median(_median_filter(angs, 3)), 1) if angs else None


def _risk_band(seconds):
    if seconds < 10:
        return "normal", "Normal mobility (<10 s)"
    if seconds < 20:
        return "independent", "Mostly independent (10-20 s)"
    if seconds < 30:
        return "caution", "Variable mobility, some fall risk (20-30 s)"
    return "high", "High fall risk (>=30 s)"


# ---- public API -----------------------------------------------------------
def _normalize_input(data):
    """Accept a bare list, or a {frames|poses|data:[...]} wrapper."""
    if isinstance(data, dict):
        for key in ("frames", "poses", "data", "keypoints_sequence"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if not isinstance(data, list) or not data:
        raise ValueError("expected a non-empty JSON array of pose frames")
    for f in data:
        if "timestamp" not in f or "keypoints" not in f:
            raise ValueError("each frame needs 'timestamp' and 'keypoints'")
        if len(f["keypoints"]) < 17:
            raise ValueError("keypoints must be COCO-17 ([x,y] x17)")
    return sorted(data, key=lambda f: f["timestamp"])


def analyze(frames, walk_distance_m=DEFAULT_WALK_DISTANCE_M):
    """Analyse a pose recording and return the TUG report dict."""
    frames = _normalize_input(frames)
    s = _build_signals(frames)
    s["_frames"] = frames
    t = s["t"]
    ph = _detect_phases(s)

    def T(i):
        return t[i] / 1000.0

    t_stand = T(ph["i_stand"])
    t_turn0 = T(ph["i_turn_start"])
    t_flip = T(ph["i_flip"])
    t_turn1 = T(ph["i_turn_end"])
    t_return = T(ph["i_return"])
    t_sit = T(ph["i_sit"])

    total = t_sit - t_stand
    turn = t_turn1 - t_turn0
    walk_out = t_turn0 - t_stand
    walk_back = t_return - t_turn1
    sit_phase = t_sit - t_return

    # sit-to-stand sub-phase: rise until fully standing (seated ratio > 0.75)
    i_up = ph["i_stand"]
    for i in range(ph["i_stand"], ph["i_turn_start"]):
        if s["seated"][i] > 0.75:
            i_up = i
            break
    stand_up = T(i_up) - t_stand
    walk_out_pure = t_turn0 - T(i_up)   # translation only, for gait speed

    gait_out = walk_distance_m / walk_out_pure if walk_out_pure > 0 else 0.0
    gait_back = walk_distance_m / walk_back if walk_back > 0 else 0.0
    gait_avg = (2 * walk_distance_m) / (walk_out_pure + walk_back) if (walk_out_pure + walk_back) > 0 else 0.0

    steps_out = _count_steps(s, i_up, ph["i_turn_start"])
    steps_turn = _count_steps(s, ph["i_turn_start"], ph["i_turn_end"])
    steps_back = _count_steps(s, ph["i_turn_end"], ph["i_return"])
    total_steps = steps_out + steps_turn + steps_back
    walk_time = walk_out_pure + turn + walk_back
    cadence = round(total_steps / walk_time * 60) if walk_time > 0 else 0

    knee_out, ankle_out = _limb_symmetry(frames, i_up, ph["i_turn_start"])
    knee_back, ankle_back = _limb_symmetry(frames, ph["i_turn_end"], ph["i_return"])
    walk_speed_sym = round(100 * (1 - abs(gait_out - gait_back) / max(gait_out + gait_back, 1e-9)))
    lean_out = _trunk_tilt(frames, i_up, ph["i_turn_start"])
    lean_back = _trunk_tilt(frames, ph["i_turn_end"], ph["i_return"])
    turn_dir = "clockwise" if s["shVec"][ph["i_turn_end"]] > 0 else "counter-clockwise"
    turn_speed = round(180.0 / turn) if turn > 0 else 0

    lo = max(ph["i_stand"] - 220, 0)
    hi = max(ph["i_stand"] - 40, lo + 1)
    seated_region = [s["hipX"][i] for i in range(lo, hi) if s["seated"][i] < 0.45]
    sway = (statistics.pstdev(seated_region) / statistics.median(s["torso"])
            if len(seated_region) > 5 else 0.0)

    dt = statistics.median([t[i + 1] - t[i] for i in range(len(t) - 1)]) if len(t) > 1 else 0
    risk_key, risk_label = _risk_band(total)

    return dict(
        recording=dict(frames=len(frames), duration_s=round(t[-1] / 1000, 2),
                       fps=round(1000 / dt) if dt else None,
                       walk_distance_m=walk_distance_m),
        tug_time_s=round(total, 2),
        risk=dict(key=risk_key, label=risk_label),
        phases=dict(sit_to_stand_s=round(stand_up, 2), walk_out_s=round(walk_out, 2),
                    turn_s=round(turn, 2), walk_back_s=round(walk_back, 2),
                    sit_down_s=round(sit_phase, 2)),
        timestamps=dict(stand=round(t_stand, 2), turn_start=round(t_turn0, 2),
                        turn_mid=round(t_flip, 2), turn_end=round(t_turn1, 2),
                        return_home=round(t_return, 2), seated=round(t_sit, 2)),
        gait=dict(speed_out_ms=round(gait_out, 2), speed_back_ms=round(gait_back, 2),
                  speed_avg_ms=round(gait_avg, 2), steps_out=steps_out,
                  steps_turn=steps_turn, steps_back=steps_back, total_steps=total_steps,
                  cadence_spm=cadence,
                  note="steps and cadence are approximate (depth walking, no confidence scores)"),
        turn=dict(direction=turn_dir, duration_s=round(turn, 2), speed_deg_s=turn_speed),
        symmetry=dict(walk_speed_pct=walk_speed_sym,
                      knee_motion_out_pct=knee_out, knee_motion_back_pct=knee_back,
                      ankle_motion_out_pct=ankle_out, ankle_motion_back_pct=ankle_back,
                      trunk_lean_out_deg=lean_out, trunk_lean_back_deg=lean_back),
        seated_sway_pct_torso=round(sway * 100, 1),
    )


def main(argv):
    args = [a for a in argv[1:]]
    inp = out = None
    dist = DEFAULT_WALK_DISTANCE_M
    i = 0
    while i < len(args):
        a = args[i]
        if a in ("-o", "--out"):
            out = args[i + 1]; i += 2
        elif a in ("-d", "--distance"):
            dist = float(args[i + 1]); i += 2
        elif a in ("-h", "--help"):
            print(__doc__); return 0
        else:
            inp = a; i += 1

    raw = open(inp, encoding="utf-8").read() if inp else sys.stdin.read()
    try:
        report = analyze(json.loads(raw), walk_distance_m=dist)
    except (ValueError, KeyError, IndexError) as e:
        json.dump({"error": str(e)}, sys.stderr); sys.stderr.write("\n")
        return 1

    text = json.dumps(report, indent=2)
    if out:
        open(out, "w", encoding="utf-8").write(text)
        sys.stderr.write(f"wrote {out}\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
