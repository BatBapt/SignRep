#!/usr/bin/env python3
"""SignRep sign priors from the MediaPipe npz files of signrep_pose_l1.py.

Used as a module by the training script (priors are computed on the fly for each
16-frame clip), and as a CLI that computes the normalisation statistics over the
L1 videos and prints coverage numbers as a sanity check.

Priors per frame (NaN = invalid, masked from the loss):
    hand_kpt    (2, 21, 3)      hand keypoints, wrist-centred, / hand size        (world coords)
    hand_angle  (2, 19, 2)      cos/sin of 15 finger-joint bends + 4 finger spreads (world coords)
    hand_dist   (2, 5, 11, 3)   fingertip -> knuckles, wrist, fingertips, / hand size (paper: 5x11x3)
    body_kpt    (17, 3)         face (0-10), shoulders, elbows, wrists, body frame
    body_angle  (6, 2)          cos/sin of shoulder, elbow, wrist angles (2D, signed)
    inter_dist  (12, 22, 2)     wrist + 5 fingertips per hand -> face, shoulders, elbows,
                                chest, other hand (2D, body frame; paper: 12x22)
First axis of hand priors: 0 = signer's left hand, 1 = right hand.

Body frame: origin = shoulder midpoint of each frame, unit = median shoulder width
of the video (pixels), so crops and colour augmentations of the image do not change
the targets. Hand size = median wrist -> middle-finger knuckle length of the video.
Points below the confidence threshold (pose visibility, hand score) are NaN.

Activity (per clip, paper heuristic): a hand is inactive if it is missing or stays
below the rest line and barely moves during the clip.
"""

import argparse
from pathlib import Path

import numpy as np

PRIORS = ("hand_kpt", "hand_angle", "hand_dist", "body_kpt", "body_angle", "inter_dist")

UPPER = list(range(11)) + [11, 12, 13, 14, 15, 16]   # face, shoulders, elbows, wrists
TIPS = [4, 8, 12, 16, 20]
KNUCKLES = [1, 5, 9, 13, 17]
FINGERS = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12], [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]]
HANDS = ("lh", "rh")


def load_pose(path):
    with np.load(path) as d:
        return {k: d[k] for k in d.files}


def _angle3(a, b, c):
    """Unsigned angle at b between b->a and b->c, as (cos, sin). Inputs (..., 3)."""
    u, v = a - b, c - b
    cos = (u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1) + 1e-8)
    cos = np.clip(cos, -1, 1)
    return np.stack([cos, np.sqrt(1 - cos ** 2)], -1)


def _angle2(a, b, c):
    """Signed 2D angle at b from b->a to b->c, as (cos, sin). Inputs (..., 2)."""
    u, v = a - b, c - b
    ang = np.arctan2(u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0], (u * v).sum(-1))
    return np.stack([np.cos(ang), np.sin(ang)], -1)


def _masked(d, thr):
    """Pose in pixels (T, 33, 3) and hands (image px, world), NaN below the threshold."""
    w, h = float(d["width"]), float(d["height"])
    scale = np.array([w, h, w], np.float32)        # MediaPipe z uses the x scale
    pose = d["pose"][..., :3] * scale
    pose[d["pose"][..., 3] < thr] = np.nan
    hands = {}
    for name in HANDS:
        bad = ~(d[f"{name}_score"] >= thr)          # NaN score -> bad
        img = d[name] * scale
        world = d[f"{name}_world"].copy()
        img[bad] = np.nan
        world[bad] = np.nan
        hands[name] = (img, world)
    return pose, hands


def video_constants(d, thr=0.5):
    """Median shoulder width (px) and hand size (m) of a video."""
    pose, hands = _masked(d, thr)
    shoulder = np.nanmedian(np.linalg.norm(pose[:, 11, :2] - pose[:, 12, :2], axis=-1))
    sizes = [np.linalg.norm(world[:, 9] - world[:, 0], axis=-1) for _, world in hands.values()]
    hand = np.nanmedian(np.concatenate(sizes))
    return {"shoulder": float(shoulder), "hand": float(hand)}


def compute_priors(d, const, start=0, end=None, thr=0.5):
    """Priors for frames [start, end) of one video -> dict of (T, ...) float32 arrays + wrist (T, 2, 2)."""
    sl = slice(start, end)
    d = {k: (v[sl] if isinstance(v, np.ndarray) and v.ndim > 0 else v) for k, v in d.items()}
    pose, hands = _masked(d, thr)
    sw, hs = const["shoulder"], const["hand"]

    origin = (pose[:, 11, :2] + pose[:, 12, :2]) / 2                      # (T, 2)
    z0 = (pose[:, 11, 2] + pose[:, 12, 2]) / 2

    def body2d(p):
        return (p - origin[:, None]) / sw

    out = {}
    body_xy = body2d(pose[:, UPPER, :2])
    body_z = (pose[:, UPPER, 2] - z0[:, None]) / sw
    out["body_kpt"] = np.concatenate([body_xy, body_z[..., None]], -1)

    hk, ha, hd, pts = [], [], [], []
    for name in HANDS:
        img, world = hands[name]
        rel = (world - world[:, :1]) / hs
        hk.append(rel)
        tips = rel[:, TIPS]
        refs = rel[:, KNUCKLES + [0] + TIPS]
        hd.append(refs[:, None] - tips[:, :, None])                      # (T, 5, 11, 3)
        bends = [_angle3(world[:, f[j - 1]], world[:, f[j]], world[:, f[j + 1]])
                 for f in FINGERS for j in (1, 2, 3)]
        prox = [world[:, f[2]] - world[:, f[1]] for f in FINGERS]
        spreads = [_angle3(prox[i], np.zeros_like(prox[i]), prox[i + 1]) for i in range(4)]
        ha.append(np.stack(bends + spreads, 1))                           # (T, 19, 2)
        pts.append(body2d(img[:, [0] + TIPS, :2]))                        # (T, 6, 2)
    out["hand_kpt"] = np.stack(hk, 1)
    out["hand_angle"] = np.stack(ha, 1)
    out["hand_dist"] = np.stack(hd, 1)

    b = body2d(pose[..., :2])                                             # (T, 33, 2)
    lh_mcp = body2d(hands["lh"][0][:, 9:10, :2])[:, 0]
    rh_mcp = body2d(hands["rh"][0][:, 9:10, :2])[:, 0]
    out["body_angle"] = np.stack([
        _angle2(b[:, 13], b[:, 11], b[:, 12]),     # left shoulder
        _angle2(b[:, 14], b[:, 12], b[:, 11]),     # right shoulder
        _angle2(b[:, 11], b[:, 13], b[:, 15]),     # left elbow
        _angle2(b[:, 12], b[:, 14], b[:, 16]),     # right elbow
        _angle2(b[:, 13], b[:, 15], lh_mcp),       # left wrist
        _angle2(b[:, 14], b[:, 16], rh_mcp),       # right wrist
    ], 1)

    common = np.concatenate([b[:, :11], b[:, 11:15], np.zeros_like(b[:, :1])], 1)   # (T, 16, 2)
    inter = []
    for own, other in ((pts[0], pts[1]), (pts[1], pts[0])):
        targets = np.concatenate([common, other], 1)                      # (T, 22, 2)
        inter.append(targets[:, None] - own[:, :, None])                  # (T, 6, 22, 2)
    out["inter_dist"] = np.concatenate(inter, 1)                          # (T, 12, 22, 2)

    out = {k: v.astype(np.float32) for k, v in out.items()}
    out["wrist"] = np.stack([pts[0][:, 0], pts[1][:, 0]], 1).astype(np.float32)   # (T, 2, 2)
    return out


def hand_activity(wrist, rest_y=1.0, move_thr=0.1):
    """wrist (T, 2, 2) body frame (y down) -> (2,) activity 0/1 per hand for one clip."""
    act = np.zeros(2, np.float32)
    for s in range(2):
        w = wrist[:, s]
        w = w[np.isfinite(w).all(-1)]
        if len(w) < len(wrist) // 2:               # mostly missing = out of frame = resting
            continue
        raised = (w[:, 1] < rest_y).any()
        moved = np.linalg.norm(w.max(0) - w.min(0)) > move_thr
        act[s] = float(raised or moved)
    return act


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pose-dir", required=True)
    p.add_argument("--out", required=True, help="statistics npz")
    p.add_argument("--video-list", default=None, help="text file of video stems (e.g. train split)")
    p.add_argument("--every", type=int, default=5, help="use one frame every N for the statistics")
    p.add_argument("--thr", type=float, default=0.5)
    p.add_argument("--clip-len", type=int, default=16)
    args = p.parse_args()

    files = sorted(Path(args.pose_dir).glob("*.npz"))
    files = [f for f in files if not f.name.endswith(".tmp.npz")]
    if args.video_list:
        keep = set(Path(args.video_list).read_text().split())
        files = [f for f in files if f.stem in keep]
    print(f"[stats] {len(files)} videos")

    s1, s2, cnt = {}, {}, {}
    nf = 0
    acts = []
    for f in files:
        d = load_pose(f)
        const = video_constants(d, args.thr)
        pr = compute_priors(d, const, thr=args.thr)
        for k in PRIORS:
            x = pr[k][::args.every].astype(np.float64)
            ok = np.isfinite(x)
            x = np.where(ok, x, 0)
            s1[k] = s1.get(k, 0) + x.sum(0)
            s2[k] = s2.get(k, 0) + (x ** 2).sum(0)
            cnt[k] = cnt.get(k, 0) + ok.sum(0)
        nf += len(pr["wrist"][::args.every])
        n = len(pr["wrist"]) // args.clip_len
        acts += [hand_activity(pr["wrist"][i * args.clip_len:(i + 1) * args.clip_len]) for i in range(n)]
        print(f"  {f.stem}: {len(pr['wrist'])} frames, shoulder {const['shoulder']:.0f} px, "
              f"hand {const['hand'] * 100:.1f} cm")

    stats = {}
    for k in PRIORS:
        c = np.maximum(cnt[k], 1)
        mean = s1[k] / c
        std = np.sqrt(np.maximum(s2[k] / c - mean ** 2, 0))
        std = np.where(std < 1e-6, 1.0, std)       # constant elements (e.g. tip to itself)
        stats[f"{k}_mean"] = mean.astype(np.float32)
        stats[f"{k}_std"] = std.astype(np.float32)
        print(f"[stats] {k:10s} shape {mean.shape}  valid {cnt[k].mean() / max(nf, 1):.3f}")
    acts = np.array(acts)
    print(f"[stats] activity (clips of {args.clip_len}): left {acts[:, 0].mean():.3f}  "
          f"right {acts[:, 1].mean():.3f}")
    np.savez(args.out, **stats)
    print(f"[stats] saved {args.out}")


if __name__ == "__main__":
    main()