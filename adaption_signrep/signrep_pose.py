#!/usr/bin/env python3
"""MediaPipe pose + hands on every frame (25 fps) of the L1 videos, for SignRep priors.

One npz per video:
    pose                (T, 33, 4) float32  x, y (normalised image coords), z, visibility
    lh, rh              (T, 21, 3) float32  hand landmarks, x, y normalised, z relative to wrist
    lh_world, rh_world  (T, 21, 3) float32  hand world landmarks (metres)
    lh_score, rh_score  (T,)       float32  hand detection score
    fps, width, height, n_frames
Missing values are NaN.

Left / right are the signer's own hands. Each detected hand goes to the nearest
pose wrist (15 = left, 16 = right) instead of MediaPipe's handedness label,
which assumes a mirrored image.

Videos are split over SLURM array tasks: task k processes videos[k::num_tasks].
Model files come from the MEDIAPIPE_MODELS_DIR environment variable.
"""

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")

import av  # noqa: E402
import mediapipe as mp  # noqa: E402
import numpy as np  # noqa: E402
from mediapipe.tasks.python import vision  # noqa: E402

MODELS_DIR = os.environ.get("MEDIAPIPE_MODELS_DIR",
                            "/hpchome/baptiste26/models_weights/mediapipe")
POSE_MODEL = "pose_landmarker_full.task"
HAND_MODEL = "hand_landmarker.task"
VIDEO_EXTS = (".mp4", ".mkv", ".mov", ".avi", ".webm")
L_WRIST, R_WRIST = 15, 16


def iter_sampled_frames(path, fps):
    """Yield RGB uint8 frames (H, W, 3) at the requested sampling rate."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        time_base = float(stream.time_base)
        next_t = 0.0
        step = 1.0 / fps
        for frame in container.decode(video=0):
            if frame.pts is None:
                continue
            t = frame.pts * time_base
            if t + 1e-6 < next_t:
                continue
            yield frame.to_ndarray(format="rgb24")
            next_t += step
            while t + 1e-6 >= next_t:
                next_t += step


def make_landmarkers(models_dir):
    """New landmarkers per video: VIDEO mode keeps tracking state and needs increasing timestamps."""
    base = mp.tasks.BaseOptions
    mode = vision.RunningMode.VIDEO
    pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=base(model_asset_path=str(Path(models_dir) / POSE_MODEL)),
        running_mode=mode, num_poses=1))
    hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=base(model_asset_path=str(Path(models_dir) / HAND_MODEL)),
        running_mode=mode, num_hands=2))
    return pose, hands


def assign_hands(wrists, pose_wrists):
    """wrists (n, 2) and pose_wrists (2, 2) [left, right] in pixels -> side per hand (0 = left, 1 = right)."""
    n = len(wrists)
    if n == 0:
        return []
    d = np.linalg.norm(wrists[:, None, :] - pose_wrists[None, :, :], axis=-1)   # (n, 2)
    d = np.where(np.isnan(d), np.inf, d)
    if np.isinf(d).all():
        # no pose wrist: the signer faces the camera, so their right hand is on the image left
        order = np.argsort(wrists[:, 0])
        sides = [0] * n
        sides[order[0]] = 1
        return sides
    if n == 1:
        return [int(np.argmin(d[0]))]
    keep = d[0, 0] + d[1, 1] <= d[0, 1] + d[1, 0]
    return [0, 1] if keep else [1, 0]


def process_video(path, args):
    frames_iter = iter_sampled_frames(path, args.fps)
    out = {k: [] for k in ("pose", "lh", "rh", "lh_world", "rh_world", "lh_score", "rh_score")}
    nan_pose = np.full((33, 4), np.nan, np.float32)
    nan_hand = np.full((21, 3), np.nan, np.float32)
    h = w = None
    t0 = time.time()

    pose_lm, hand_lm = make_landmarkers(args.models_dir)
    with pose_lm, hand_lm:
        for i, frame in enumerate(frames_iter):
            if args.max_frames and i >= args.max_frames:
                break
            h, w = frame.shape[:2]
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame))
            ts = int(round(i * 1000 / args.fps))

            res = pose_lm.detect_for_video(image, ts)
            pose = nan_pose
            if res.pose_landmarks:
                pose = np.array([[p.x, p.y, p.z, p.visibility] for p in res.pose_landmarks[0]],
                                np.float32)
            out["pose"].append(pose)

            res = hand_lm.detect_for_video(image, ts)
            hands = [np.array([[p.x, p.y, p.z] for p in lms], np.float32) for lms in res.hand_landmarks]
            worlds = [np.array([[p.x, p.y, p.z] for p in lms], np.float32)
                      for lms in res.hand_world_landmarks]
            scores = [c[0].score for c in res.handedness]
            px = np.array([w, h], np.float32)
            wrists = np.array([hd[0, :2] * px for hd in hands]).reshape(-1, 2)
            sides = assign_hands(wrists, pose[[L_WRIST, R_WRIST], :2] * px)

            side_data = {0: (nan_hand, nan_hand, np.nan), 1: (nan_hand, nan_hand, np.nan)}
            for hd, wd, sc, s in zip(hands, worlds, scores, sides):
                side_data[s] = (hd, wd, sc)
            for s, name in ((0, "lh"), (1, "rh")):
                hd, wd, sc = side_data[s]
                out[name].append(hd)
                out[f"{name}_world"].append(wd)
                out[f"{name}_score"].append(sc)

            if (i + 1) % 5000 == 0:
                print(f"  {path.stem}: {i + 1} frames, {(i + 1) / (time.time() - t0):.1f} fps", flush=True)

    if not out["pose"]:
        return None
    result = {k: np.asarray(v, np.float32) for k, v in out.items()}
    result.update(fps=np.float32(args.fps), width=np.int32(w), height=np.int32(h),
                  n_frames=np.int32(len(out["pose"])))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--models-dir", default=MODELS_DIR)
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--task-id", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    p.add_argument("--num-tasks", type=int, default=1)
    p.add_argument("--limit", type=int, default=None, help="first N videos only (testing)")
    p.add_argument("--max-frames", type=int, default=None, help="first N frames only (testing)")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    videos = sorted(f for f in Path(args.video_dir).iterdir() if f.suffix.lower() in VIDEO_EXTS)
    if args.limit:
        videos = videos[:args.limit]
    videos = videos[args.task_id::args.num_tasks]
    print(f"[main] task {args.task_id}/{args.num_tasks}: {len(videos)} videos")

    for k, path in enumerate(videos, 1):
        out_path = out_dir / f"{path.stem}.npz"
        if out_path.exists() and not args.overwrite:
            print(f"[{k}/{len(videos)}] skip {path.stem}")
            continue
        t0 = time.time()
        result = process_video(path, args)
        if result is None:
            print(f"[{k}/{len(videos)}] {path.stem}: no frames, skipped")
            continue
        tmp = out_dir / f"{path.stem}.tmp.npz"
        np.savez(tmp, **result)
        os.replace(tmp, out_path)                     # no half-written file if the job is killed
        n = int(result["n_frames"])
        pose_ok = 1 - np.isnan(result["pose"][:, 0, 0]).mean()
        lh_ok = 1 - np.isnan(result["lh_score"]).mean()
        rh_ok = 1 - np.isnan(result["rh_score"]).mean()
        print(f"[{k}/{len(videos)}] {path.stem}: {n} frames in {time.time() - t0:.0f} s | "
              f"pose {pose_ok:.3f} lh {lh_ok:.3f} rh {rh_ok:.3f}", flush=True)


if __name__ == "__main__":
    main()