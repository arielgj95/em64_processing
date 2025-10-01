#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Acoustic 3-D beamforming over a room point cloud with a step-by-step interactive viewer
(no audio while interacting) and a rotating-camera MP4 export (with audio).

What this does:
- Align EM64 array frame to room frame (axis mapping + optional yaw/pitch/roll).
- Build a full-sphere direction grid (Fibonacci sphere).
- Compute broadband energy per STFT frame with:
    * DAS / SRP-PHAT over RAW 64ch
    * MAX-rE and MVDR over HOA (6th order, ACN/SN3D or N3D)
- Color the point cloud interior by directional energy (distance-attenuated).
- Interactive "stepper" shows exactly one frame at a time; you press keys to
  advance/rewind. Every step **resets the camera** to the microphone pose
  (so each frame starts from the same viewpoint).
- Export MP4: camera fixed at mic position and slowly rotates by **360°** while
  audio is muxed and the acoustic map is synchronized to frames.

Controls (step viewer):
  N or Space  : next frame
  P           : previous frame
  G           : +10 frames
  H           : -10 frames
  S           : save screenshot (PNG)
  Q or Esc    : quit

Author: you + ChatGPT
"""

import os, sys, time, math, threading, tempfile, copy, logging, traceback, random
import numpy as np
import soundfile as sf
from tqdm import tqdm
from scipy.signal import stft, istft, get_window, butter, filtfilt
from scipy import special as spsp
import open3d as o3d
from open3d.visualization import rendering as o3dr
import imageio
from moviepy.editor import AudioFileClip, VideoFileClip
from matplotlib import cm
from matplotlib import colormaps as mpl_cmaps

# ---------------------------------
# Logging
# ---------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("Acoustic3d")

C_SOUND = 343.0  # m/s
DEFAULT_CMAP = "inferno"

_CACHED_CMAPS = {}

def get_cmap_cached(name):
    c = _CACHED_CMAPS.get(name)
    if c is None:
        c = mpl_cmaps.get_cmap(name)          # <- use this
        _CACHED_CMAPS[name] = c
    return c

# --------------------------------------------------------------------------------------
# Camera helpers (NOTE: supports distinct horizontal & vertical FOVs)
# --------------------------------------------------------------------------------------

def _unit_dirs_from_equirectangular(w, h):
    """
    Build a (H,W,3) array of unit direction vectors for equirectangular pixels.
    Convention:
      - phi (azimuth)  ∈ [-π, +π) maps to x-axis horizontally, left→right.
      - theta (polar)  ∈ [0,  π]   maps to y-axis vertically, top→bottom.
      - Spherical-to-Cartesian: x=cos(phi)*sin(theta), y=sin(phi)*sin(theta), z=cos(theta).
    Returns float32 unit vectors.
    """
    yy = np.linspace(0.0, 1.0, h, endpoint=False)
    xx = np.linspace(0.0, 1.0, w, endpoint=False)
    u, v = np.meshgrid(xx, yy)  # u: [0,1) -> phi, v: [0,1) -> theta
    phi   = (u * 2.0 * np.pi) - np.pi          # [-π, +π)
    theta = v * np.pi                           # [0, π]
    st = np.sin(theta); ct = np.cos(theta)
    cp = np.cos(phi);   sp = np.sin(phi)
    # Room convention consistent with cart2sph_dirs (x,y,z)
    x = cp * st
    y = sp * st
    z = ct
    dirs = np.stack([x, y, z], axis=-1).astype(np.float32, copy=False)
    # Safety normalize (grid is already unit-length, but be robust)
    n = np.linalg.norm(dirs, axis=-1, keepdims=True) + 1e-12
    return dirs / n

def _precompute_equirectangular_nn(dirs_grid, w=2048, h=1024, knn=1, chunk=131072):
    """
    For each equirectangular pixel, find nearest (or top-k) directions in dirs_grid by cosine similarity.
    Returns:
      idx: (H,W,knn) int32 indices into dirs_grid
      wts: (H,W,knn) float32 weights that sum to 1 along the last axis
    """
    # Flatten grid
    H, W = h, w
    raydirs = _unit_dirs_from_equirectangular(W, H).reshape(-1, 3).astype(np.float32, copy=False)  # (P,3)
    D = dirs_grid.astype(np.float32, copy=False)  # (K,3) unit
    P, K = raydirs.shape[0], D.shape[0]

    idx_out = np.empty((P, knn), dtype=np.int32)
    wts_out = np.empty((P, knn), dtype=np.float32)

    # Cosine similarity chunks
    for p0 in tqdm(range(0, P, chunk), desc="equirect NN", unit="px"):
        p1 = min(P, p0 + chunk)
        R = raydirs[p0:p1, :]          # (Pc,3)
        # (Pc,K) similarities
        S = R @ D.T                    # cosine ~ dot (already unit)
        # top-k
        topk_idx = np.argpartition(-S, kth=knn-1, axis=1)[:, :knn]  # (Pc,knn) unordered
        # reorder exact by value
        row = np.arange(topk_idx.shape[0])[:, None]
        topk_vals = S[row, topk_idx]
        ord = np.argsort(-topk_vals, axis=1)
        sel_idx = topk_idx[row, ord]          # (Pc,knn)
        sel_val = np.take_along_axis(topk_vals, ord, axis=1)

        # positive weights from similarities (ReLU), then normalize
        w = np.maximum(sel_val, 0.0)
        s = np.sum(w, axis=1, keepdims=True) + 1e-12
        w = (w / s).astype(np.float32, copy=False)

        idx_out[p0:p1, :] = sel_idx.astype(np.int32, copy=False)
        wts_out[p0:p1, :] = w

    idx_out = idx_out.reshape(H, W, knn)
    wts_out = wts_out.reshape(H, W, knn)
    return idx_out, wts_out

def export_360_pointcloud_equirect(
    pack, out_mp4, width=2048, height=1024, fps_out=12,
    cmap_name="inferno", per_frame_norm=False, color_floor=0.03,
    base_geom_gray=0.25, splat_px=1.25,
    # NEW: audio controls (same behavior as your other exporters)
    audio_wav_override=None, use_hoa_binaural=True, order=6, azL=-30.0, azR=+30.0, el=0.0,
    # NEW: speed controls to match rotating video defaults
    encoder_preset="ultrafast", crf=23
):
    """
    Render an equirectangular 360° video of the point cloud, colored by acoustic energy.
    Now supports optional stereo audio muxing (override WAV or HOA→binaural),
    and lets you choose a faster encoder preset/CRF for speed.
    """
    import imageio, os
    from matplotlib import colormaps as mpl_cmaps

    pcd      = pack["pcd"]
    pts_room = np.asarray(pcd.points, np.float64)
    mic_pos  = np.asarray(pack["mic_pos_room"], np.float64)
    R_ra     = np.asarray(pack["mic_R_room"],  np.float64)   # ROOM->ARRAY (your variable name)
    E        = pack["E"]                    # (T,K)
    hop_s    = pack["hop_s"]
    idx_dir  = pack["idx_dir"]              # (N,) per-point direction bin (ARRAY frame)
    # small speed win: reuse cached cmap object
    cmap     = get_cmap_cached(cmap_name)

    # Precompute per-point ARRAY-local directions and distances (static)
    v_room   = pts_room - mic_pos[None, :]
    v_local  = v_room @ R_ra
    dists    = np.linalg.norm(v_local, axis=1)
    mask     = dists > 1e-9
    dirs_loc = np.zeros_like(v_local)
    dirs_loc[mask] = v_local[mask] / dists[mask, None]

    # Project point directions to equirect pixels (static)
    x,y,z    = dirs_loc[:,0], dirs_loc[:,1], dirs_loc[:,2]
    thetas   = np.arccos(np.clip(z, -1.0, 1.0))                  # [0,π] from +Z
    phis     = np.arctan2(y, x)                                  # [-π,π] from +X
    px       = np.clip(((phis + np.pi) / (2*np.pi) * width ).astype(np.int32),  0, width-1)
    py       = np.clip(((thetas     /  np.pi)  * height).astype(np.int32),      0, height-1)
    lin_idx  = (py * width + px)

    # Z-buffer once: for each pixel, keep nearest point
    #order      = np.lexsort((dists, lin_idx))     # sort by pixel then distance
    #lin_sorted = lin_idx[order]
    #keep_first = np.concatenate(([True], lin_sorted[1:] != lin_sorted[:-1]))
    #keep_idx   = order[keep_first]                # indices of nearest points per pixel
    #kx, ky     = px[keep_idx], py[keep_idx]
    sort_idx = np.lexsort((dists, lin_idx))
    lin_sorted = lin_idx[sort_idx]
    keep_first = np.concatenate(([True], lin_sorted[1:] != lin_sorted[:-1]))
    keep_idx = sort_idx[keep_first]
    kx, ky = px[keep_idx], py[keep_idx]

    # Energy frames at output fps
    E_dst      = resample_energy_frames(E, src_fps=1.0/hop_s, dst_fps=fps_out)
    T_out      = E_dst.shape[0]
    duration_s = T_out / float(fps_out)
    vmax_g     = max(1e-12, np.percentile(E_dst, 90))

    # Decide audio & temp path
    need_audio = (audio_wav_override is not None) or (use_hoa_binaural and (pack.get("hoa_pack") is not None))
    tmp_video  = out_mp4 if not need_audio else out_mp4.replace(".mp4", "_noaudio.mp4")
    os.makedirs(os.path.dirname(out_mp4) or ".", exist_ok=True)

    # Faster encoder settings to match rotating video speed
    writer = imageio.get_writer(
        tmp_video, fps=fps_out, codec="libx264",
        ffmpeg_params=["-preset", str(encoder_preset), "-crf", str(crf), "-pix_fmt", "yuv420p"]
    )

    try:
        for ti in tqdm(range(T_out), desc="equirect PC frames", unit="frm"):
            # Base background
            frame = np.full((height, width, 3), base_geom_gray, np.float32)

            # Per-point energy -> colors
            e_pt  = E_dst[ti][idx_dir]  # (N,)
            vmax  = max(1e-12, np.percentile(e_pt, 90)) if per_frame_norm else vmax_g
            v     = np.clip(e_pt / vmax, 0, 1)
            if color_floor > 0:
                v = v * (1 - color_floor) + color_floor
            cols_pts = cmap(v).astype(np.float32)[:, :3]

            # Paint nearest point per pixel
            frame[ky, kx, :] = cols_pts[keep_idx]

            # Optional micro-splat (few vectorized passes; cost ~O(#offsets))
            if splat_px and splat_px > 0:
                r = int(round(splat_px))
                offsets = [(dx,dy) for dx in range(-r, r+1) for dy in range(-r, r+1) if (dx,dy)!=(0,0)]
                for dx, dy in offsets:
                    x2 = np.clip(kx + dx, 0, width - 1)
                    y2 = np.clip(ky + dy, 0, height - 1)
                    frame[y2, x2, :] = cols_pts[keep_idx]

            writer.append_data((np.clip(frame, 0, 1) * 255).astype(np.uint8))
    finally:
        writer.close()

    # Mux audio if requested
    if need_audio:
        if audio_wav_override:
            stereo_wav = _ensure_stereo_from_override(audio_wav_override, prefer_channel0=True)
        else:
            y, fs = decode_binaural_from_hoa_pack(pack["hoa_pack"], order, azL, azR, el)
            stereo_wav = os.path.splitext(out_mp4)[0] + "_stereo.wav"
            sf.write(stereo_wav, y, fs)

        log.info("[stage] Mux audio into equirect point-cloud MP4")
        aud  = AudioFileClip(stereo_wav).subclip(0, duration_s)
        clip = VideoFileClip(tmp_video).set_audio(aud)
        clip.write_videofile(out_mp4, codec="libx264", audio_codec="aac",
                             fps=fps_out, preset=encoder_preset, threads=os.cpu_count() or 4, verbose=True)
        clip.close(); aud.close()
        try:
            if tmp_video != out_mp4 and os.path.exists(tmp_video):
                os.remove(tmp_video)
        except:
            pass

    log.info("[done] 360° equirect point-cloud video -> %s", out_mp4)
    return out_mp4

def export_360_equirectangular_video(
    pack, out_mp4, width=2048, height=1024, fps_out=24,
    cmap_name="inferno", knn=4, per_frame_norm=False, color_floor=0.03,
    audio_wav_override=None, use_hoa_binaural=True, order=6, azL=-30.0, azR=+30.0, el=0.0
):
    """
    Write an equirectangular 360° video with optional stereo audio:
      - audio_wav_override: any wav path (we downmix/duplicate to stereo)
      - else, if use_hoa_binaural and pack['hoa_pack'] exists: decode binaural from HOA
      - else: no audio
    """
    assert "dirs_grid" in pack, "pack missing 'dirs_grid' (update build_acoustic_3d return)."
    E = pack["E"]                      # (T,K)
    dirs_grid = pack["dirs_grid"]      # (K,3)

    # Resample energies to output fps (if needed)
    hop_s = pack["hop_s"]
    E_dst = resample_energy_frames(E, src_fps=1.0/hop_s, dst_fps=fps_out)  # (T',K)
    T_out = E_dst.shape[0]
    duration_s = T_out / float(fps_out)

    # Precompute pixel→direction lookup + weights
    idx_map, wts_map = _precompute_equirectangular_nn(dirs_grid, w=width, h=height, knn=knn)

    # Global range (if not per-frame)
    vmin = 0.0
    vmax_global = max(1e-12, np.percentile(E_dst, 90))

    # Decide whether we need a temp (no-audio) mp4
    need_audio = (audio_wav_override is not None) or (use_hoa_binaural and (pack.get("hoa_pack") is not None))
    tmp_video = out_mp4 if not need_audio else out_mp4.replace(".mp4", "_noaudio.mp4")

    # Write video frames
    os.makedirs(os.path.dirname(out_mp4) or ".", exist_ok=True)
    writer = imageio.get_writer(
        tmp_video, fps=fps_out, codec="libx264",
        ffmpeg_params=["-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p"]
    )
    cmap = mpl_cmaps.get_cmap(cmap_name)

    try:
        for ti in tqdm(range(T_out), desc="write 360 equirect frames", unit="frm"):
            e_dir = E_dst[ti]  # (K,)

            # Weighted splat to pixels
            e_knn = e_dir[idx_map]                 # (H,W,knn)
            e_pix = np.sum(e_knn * wts_map, axis=-1)  # (H,W)

            # Normalize
            vmax = max(1e-12, np.percentile(e_pix, 90)) if per_frame_norm else vmax_global
            v = np.clip(e_pix / vmax, 0.0, 1.0)
            if color_floor > 0.0:
                v = v * (1.0 - color_floor) + color_floor

            rgb = (cmap(v)[:, :, :3] * 255.0).astype(np.uint8)  # (H,W,3)
            writer.append_data(rgb)
    finally:
        writer.close()

    # Mux audio (if any)
    if need_audio:
        # Build stereo wav
        if audio_wav_override:
            stereo_wav = _ensure_stereo_from_override(audio_wav_override, prefer_channel0=True)
        else:
            y, fs = decode_binaural_from_hoa_pack(pack["hoa_pack"], order, azL, azR, el)
            stereo_wav = os.path.splitext(out_mp4)[0] + "_stereo.wav"
            sf.write(stereo_wav, y, fs)

        log.info("[stage] Mux audio into 360 MP4")
        aud = AudioFileClip(stereo_wav).subclip(0, duration_s)
        clip = VideoFileClip(tmp_video).set_audio(aud)
        clip.write_videofile(out_mp4, codec="libx264", audio_codec="aac",
                             fps=fps_out, preset="medium", threads=os.cpu_count() or 4, verbose=True)
        clip.close(); aud.close()

        # cleanup temp (and optional wav if it was a temp)
        try:
            if tmp_video != out_mp4 and os.path.exists(tmp_video):
                os.remove(tmp_video)
        except: pass

    log.info("[done] 360° equirect video -> %s", out_mp4)
    return out_mp4

def _lookat_from_o3d_params(params):
    Rcw = params.extrinsic[:3, :3]
    t   = params.extrinsic[:3,  3]
    eye   = -Rcw.T @ t
    right =  Rcw.T @ np.array([1.0, 0.0, 0.0])
    up    =  Rcw.T @ np.array([0.0, 1.0, 0.0])
    front = -Rcw.T @ np.array([0.0, 0.0, 1.0])

    # # 1) Ensure right-handed (if left-handed, flip up)
    # if np.dot(np.cross(up, front), right) < 0:
    #     up = -up
    #
    # # 2) Remove accidental 180° roll: prefer up to point roughly toward +Y (room green)
    # y_ref = np.array([0.0, 1.0, 0.0])
    # u_ref = y_ref - np.dot(y_ref, front) * front  # project +Y onto plane ⟂ front
    # nrm = np.linalg.norm(u_ref)
    # if nrm > 1e-9:
    #     u_ref /= nrm
    #     if np.dot(up, u_ref) < 0.0:
    #         up = -up

    return eye, front, up

def _spin_camera_inplace(params, axis='z'):
    if axis == 'x': Q = np.diag([ 1.0,-1.0,-1.0])  # 180° about world X
    elif axis == 'y': Q = np.diag([-1.0, 1.0,-1.0]) # 180° about world Y
    elif axis == 'z': Q = np.diag([-1.0,-1.0, 1.0]) # 180° about world Z
    else: raise ValueError("axis must be 'x','y','z'")
    extr = params.extrinsic.copy()
    extr[:3,:3] = Q @ extr[:3,:3]
    extr[:3, 3] = Q @ extr[:3, 3]
    params.extrinsic = extr
    return params

def _o3d_cam_from_room_axes(eye, width=1280, height=720, fov_h_deg=90.0, fov_v_deg=50.0):
    """Camera at `eye`, oriented by the POINT-CLOUD/ROOM frame (front=+Z, up=+Y)."""
    front_room = np.array([0.0, 0.0, 1.0], dtype=float)  # +Zpc
    up_room    = np.array([0.0, 1.0, 0.0], dtype=float)  # +Ypc
    return _cam_params_from_pose(
        np.asarray(eye, dtype=float), front_room, up_room,
        width=width, height=height, fov_h_deg=fov_h_deg, fov_v_deg=fov_v_deg
    )

def precompute_point_colors(
    E_tk, point_dir_idx, w_dist, cmap_name=DEFAULT_CMAP,
    per_frame_norm=True, color_floor=0.03, global_vmin=0.0, global_vmax=None
):
    """
    Returns a list of length T. Each item is (N,3) float64 in [0,1] for the point cloud colors.
    """
    T, K = E_tk.shape
    N = point_dir_idx.shape[0]
    if global_vmax is None:
        global_vmax = max(1e-12, np.percentile(E_tk, 90))
    colors = []
    cmap = mpl_cmaps.get_cmap(cmap_name)
    ones = np.ones_like(w_dist)

    for ti in tqdm(range(T), desc="Precompute point colors", unit="frm"):
        e_dir = E_tk[ti]
        vals = e_dir[point_dir_idx] * w_dist
        if per_frame_norm:
            vmax = max(1e-12, np.percentile(vals, 90))
            v = np.clip(vals / vmax, 0.0, 1.0)
        else:
            v = np.clip((vals - global_vmin) / (global_vmax - global_vmin + 1e-12), 0.0, 1.0)
        if color_floor > 0.0:
            v = v * (1.0 - color_floor) + color_floor
        cols = cmap(v)[:, :3].astype(np.float64, copy=False)
        colors.append(cols)
    return colors

def _o3d_cam_from_mic_pose(mic_pos_room, mic_R_room, width=1280, height=720, fov_h_deg=90.0, fov_v_deg=50.0):
    eye = np.asarray(mic_pos_room, dtype=float)

    # Mic axes (left-handed device):
    # +x = down, +y = forward, +z = right
    # For a camera aligned to the mic:
    eye   = np.asarray(mic_pos_room, dtype=float)
    front = mic_R_room @ np.array([1.0, 0.0, 0.0])  # mic +x (forward)
    up    = mic_R_room @ np.array([0.0, 0.0, 1.0])  # mic +z (up)

    f = front / (np.linalg.norm(front) + 1e-12)
    u0 = up    / (np.linalg.norm(up)    + 1e-12)

    # right = up × front ; up_cam = front × right
    r = np.cross(u0, f); r /= (np.linalg.norm(r) + 1e-12)
    u = np.cross(f, r);  u /= (np.linalg.norm(u) + 1e-12)

    Rcw = np.stack([r, u, -f], axis=0)
    t = -Rcw @ eye

    extrinsic = np.eye(4, dtype=float); extrinsic[:3,:3] = Rcw; extrinsic[:3,3] = t
    fx = 0.5 * width  / np.tan(np.deg2rad(fov_h_deg) * 0.5)
    fy = 0.5 * height / np.tan(np.deg2rad(fov_v_deg) * 0.5)
    cx = (width - 1) * 0.5; cy = (height - 1) * 0.5
    intr = o3d.camera.PinholeCameraIntrinsic(); intr.set_intrinsics(int(width), int(height), fx, fy, cx, cy)

    params = o3d.camera.PinholeCameraParameters(); params.intrinsic = intr; params.extrinsic = extrinsic
    return params

def _flip_camera_yz_inplace(params):
    """
    Rotate the camera basis 180° around world X (flip Y and Z).
    This changes orientation only; the eye stays fixed.
    """
    Q = np.diag([1.0, -1.0, -1.0])
    extr = params.extrinsic.copy()
    extr[:3, :3] = Q @ extr[:3, :3]
    extr[:3,  3] = Q @ extr[:3,  3]
    params.extrinsic = extr
    return params





def _cam_params_from_pose(eye, front_room, up_room, width=1280, height=720,
                          fov_h_deg=90.0, fov_v_deg=50.0):
    f = np.asarray(front_room, dtype=float)
    f = f / (np.linalg.norm(f) + 1e-12)

    up = np.asarray(up_room, dtype=float)
    up = up / (np.linalg.norm(up) + 1e-12)

    # Right-handed camera basis:
    # right = up × front     (NOT front × up)
    # up_cam = front × right
    r = np.cross(up, f); r /= (np.linalg.norm(r) + 1e-12)
    u = np.cross(f, r);  u /= (np.linalg.norm(u) + 1e-12)

    # world -> camera, camera looks down -Z (Open3D convention)
    Rcw = np.stack([r, u, -f], axis=0)
    t = -Rcw @ np.asarray(eye, dtype=float)

    extrinsic = np.eye(4, dtype=float)
    extrinsic[:3, :3] = Rcw
    extrinsic[:3, 3]  = t

    fx = 0.5 * width  / np.tan(np.deg2rad(fov_h_deg) * 0.5)
    fy = 0.5 * height / np.tan(np.deg2rad(fov_v_deg) * 0.5)
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5

    intr = o3d.camera.PinholeCameraIntrinsic()
    intr.set_intrinsics(int(width), int(height), fx, fy, cx, cy)

    params = o3d.camera.PinholeCameraParameters()
    params.intrinsic = intr
    params.extrinsic = extrinsic

    det = np.linalg.det(Rcw)
    log.info("[camera] Rcw det=%.6f (should be +1)", det)
    return params

# ----------------------------
# Basic utilities
# ----------------------------
def rpy_to_R(yaw, pitch, roll):
    """Right-handed rotations about z (yaw), y (pitch), x (roll). Angles in radians."""
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr,  cr]])
    return Rz @ Ry @ Rx

def show_pointcloud_with_frames(pcd, mic_pos_room, mic_R_room,
                                room_size=0.5, mic_size=0.5):
    geoms = [pcd]

    # Room/frame at origin
    room_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=room_size, origin=[0, 0, 0]
    )
    geoms.append(room_axes)

    # Mic frame at its pose
    mic_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=mic_size)
    T = np.eye(4)
    T[:3, :3] = np.asarray(mic_R_room, float)
    T[:3,  3] = np.asarray(mic_pos_room, float)
    mic_axes.transform(T)
    geoms.append(mic_axes)

    print("Room frame: RH (+X red, +Y green, +Z blue).")
    det = np.linalg.det(mic_R_room)
    print(f"Mic frame det={det:.6f} → {'RH' if det>0 else 'LH'}.")

    o3d.visualization.draw_geometries(geoms)


# def build_axis_mapping_R(spec=None):
#     """
#     Fixed axis mapping from EM64 array frame to room frame.
#     No longer uses the problematic string specification.
#     """
#     # EM64: +X=forward, +Y=right, +Z=up
#     # Room: +X=left, +Y=up, +Z=forward
#     # Transformation: [room_x, room_y, room_z] = R @ [array_x, array_y, array_z]
#
#     R = np.array([
#         [0, -1, 0],  # room_x = -array_y (right becomes left)
#         [0, 0, 1],  # room_y = +array_z (up stays up)
#         [1, 0, 0],  # room_z = +array_x (forward stays forward)
#     ], dtype=np.float64)
#
#     det = np.linalg.det(R)
#     print(f"[axis_mapping] Determinant: {det:.6f} (should be +1.0)")
#     return R



def build_axis_mapping_R(spec="x,-y,z"):
    """
    Build the base rotation mapping EM64 array axes -> room axes.
    spec "x,-y,z" means:
      room_x = +array_x, room_y = -array_y, room_z = +array_z  (your stated case)
    Returns R such that v_room = R @ v_array.
    """
    tokens = [t.strip().lower() for t in spec.split(",")]
    if len(tokens) != 3:
        raise ValueError("Axis spec must have 3 comma-separated entries like 'x,-y,z'")
    basis = {
        "+x": np.array([1,0,0]), "x": np.array([1,0,0]),
        "-x": np.array([-1,0,0]),
        "+y": np.array([0,1,0]), "y": np.array([0,1,0]),
        "-y": np.array([0,-1,0]),
        "+z": np.array([0,0,1]), "z": np.array([0,0,1]),
        "-z": np.array([0,0,-1]),
    }
    cols = []
    used = set()
    for t in tokens:
        if t not in basis: raise ValueError(f"Bad axis token '{t}'. Use x,-x,y,-y,z,-z.")
        v = basis[t]
        ab = t.replace("-", "")
        if ab in used:
            raise ValueError(f"Axis '{ab}' used more than once.")
        used.add(ab)
        cols.append(v)
    R = np.stack(cols, axis=1).astype(np.float64)  # columns are images of array basis in room coords
    det = np.linalg.det(R)
    if abs(det - 1.0) > 1e-9:
        log.warning("Axis mapping det != 1 (%.6f). Check spec.", det)
    return R

def cart2sph_dirs(dirs, phi_flip=False):
    """ dirs: (K,3) unit vectors (x,y,z). Returns theta from +Z, phi from +X CCW.
        If phi_flip=True, mirror azimuth (needed when array frame is LH). """
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    phi   = np.arctan2(y, x)
    if phi_flip:
        phi = -phi
    return theta, phi

def fibonacci_sphere(n_dirs=4096, z_clip=None):
    """ Quasi-uniform unit vectors on sphere; optionally clip to |z| <= z_clip. """
    ga = np.pi * (3.0 - np.sqrt(5.0))
    i = np.arange(n_dirs, dtype=np.float64)
    z = 1.0 - 2.0 * (i + 0.5) / n_dirs
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    phi = ga * i
    x = r * np.cos(phi); y = r * np.sin(phi)
    dirs = np.stack([x,y,z], axis=1).astype(np.float64)
    if z_clip is not None:
        m = (np.abs(z) <= z_clip)
        dirs = dirs[m]
    return dirs

def normalize01(arr, vmin=None, vmax=None, eps=1e-12):
    if vmin is None: vmin = 0.0
    if vmax is None:
        vmax = np.percentile(arr, 90)
        if vmax < eps: vmax = eps
    return np.clip((arr - vmin) / (vmax - vmin + eps), 0.0, 1.0)

def colormap_values(vals01, cmap_name=DEFAULT_CMAP):
    """ vals01: (N,) in [0,1] returns colors as (N,3) in [0,1] """
    rgba = mpl_cmaps.get_cmap(cmap_name)(vals01)
    return rgba[:, :3]

# ----------------------------
# EM64 geometry
# ----------------------------
def load_em64_geometry(csv_path):
    """
    Reads em64_geom.csv with columns:
      'mic','mic X (m)','mic Y (m)','mic Z (m)','Theta (degrees)','Phi (degrees)','Quad. Weight'
    Returns:
      pos: (M,3) meters (centered)
      wq:  (M,) quadrature weights
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    def g(col):
        for c in df.columns:
            if c.strip().lower() == col.strip().lower():
                return df[c].to_numpy()
        raise KeyError(f"Column '{col}' not found in {csv_path}. Found: {list(df.columns)}")
    x = g("mic X (m)").astype(np.float64)
    y = g("mic Y (m)").astype(np.float64)
    z = g("mic Z (m)").astype(np.float64)
    wq = g("Quad. Weight").astype(np.float64)
    pos = np.stack([x, y, z], axis=1)
    pos = pos - np.mean(pos, axis=0, keepdims=True)  # center for plane-wave steering
    return pos, wq

# ----------------------------
# STFT & preprocessing
# ----------------------------
def stft_multi(x, fs, map_fps=12, win_mult=2, nfft_min=1024, nfft_max=None, force_float32=True):
    N, C = x.shape
    if force_float32 and x.dtype != np.float32:
        x = x.astype(np.float32, copy=False)

    hop = max(1, int(round(fs / float(map_fps))))
    win = 2 * hop                                  # <-- EXACTLY 50% overlap
    window = get_window("hann", win, fftbins=True)

    # nfft can be >= win (power-of-two if you like), but nperseg stays == win
    nfft = max(nfft_min, 1 << int(np.ceil(np.log2(win)))) if nfft_max is None else min(
        max(nfft_min, 1 << int(np.ceil(np.log2(win)))), int(nfft_max)
    )

    X_list = []
    f = t = None
    for c in tqdm(range(C), desc="STFT per channel", unit="ch"):
        f, t, Z = stft(x[:, c], fs=fs, window=window, nperseg=win,
                       noverlap=hop, nfft=nfft,
                       boundary=None, padded=False, return_onesided=True)
        X_list.append(Z)
    X = np.stack(X_list, axis=-1)  # (F,T,C)
    hop_s = hop / float(fs)
    stft_t0 = float(t[0]) if len(t) else 0.0
    return X, f.astype(np.float32), t.astype(np.float32), hop_s, stft_t0, window

def istft_multi(Y_ftc, fs, hop_s, window):
    """ Inverse STFT for (F,T,C=1 or 2); returns (N,C). """
    hop = int(round(hop_s * fs))
    F, T, C = Y_ftc.shape
    outs = []
    for c in range(C):
        _, y = istft(Y_ftc[:, :, c], fs=fs, window=window, nperseg=len(window),
                     noverlap=len(window) - hop, input_onesided=True, boundary=False)
        outs.append(y.astype(np.float32, copy=False))
    L = max(map(len, outs))
    outs = [np.pad(o, (0, L - len(o)), mode='constant') for o in outs]
    y_nc = np.stack(outs, axis=-1)
    return y_nc

def bandpass_inplace(x, fs, lo=150.0, hi=8000.0, order=4):
    b, a = butter(order, [lo/(fs*0.5), hi/(fs*0.5)], btype="bandpass")
    for c in range(x.shape[1]):
        x[:, c] = filtfilt(b, a, x[:, c])

# ----------------------------
# HOA helpers (ACN/N3D complex)
# ----------------------------
def acn_index(l, m): return l * l + l + m

def sh_complex_n3d_matrix(order, theta, phi):
    """ Complex SH (N3D) up to 'order' in ACN order. Y: (K, (order+1)^2) """
    K = theta.shape[0]
    C = (order + 1) ** 2
    Y = np.zeros((K, C), dtype=np.complex128)
    ct = np.cos(theta)
    for l in range(order + 1):
        P_l0 = spsp.lpmv(0, l, ct)
        N_l0 = np.sqrt((2 * l + 1) / (4 * np.pi))
        Y[:, acn_index(l, 0)] = (N_l0 * P_l0).astype(np.complex128)
        for m in range(1, l + 1):
            P_lm = spsp.lpmv(m, l, ct)
            N_lm = np.sqrt((2*l + 1) / (4*np.pi) * spsp.factorial(l - m) / spsp.factorial(l + m))
            base = N_lm * P_lm
            e_imphi = np.exp(1j * m * phi)
            Y_pos = base * e_imphi
            Y_neg = ((-1)**m) * np.conj(Y_pos)
            Y[:, acn_index(l,  m)] = Y_pos
            Y[:, acn_index(l, -m)] = Y_neg
    return Y

def maxre_weights(order):
    alpha = np.pi / (2 * order + 2)
    ca = np.cos(alpha)
    return np.array([spsp.eval_legendre(l, ca) for l in range(order + 1)], dtype=np.float64)

def expand_order_weights_per_channel(order_weights):
    ws = []
    for l, g in enumerate(order_weights):
        ws.extend([g] * (2*l + 1))
    return np.array(ws, dtype=np.float64)

def sn3d_to_n3d_inplace(hoa, order):
    """ACN/SN3D -> ACN/N3D inplace."""
    idx = 0
    for l in range(order + 1):
        s = np.sqrt(2 * l + 1)
        n_ch = 2 * l + 1
        hoa[:, idx:idx + n_ch] *= s
        idx += n_ch

def acn_real_to_complex_n3d_block(X_block_n3d, order):
    """ Real ACN/N3D -> Complex N3D (per STFT bin slice). """
    L, C = X_block_n3d.shape
    Xc = np.zeros((L, C), dtype=np.complex128)
    for l in range(order + 1):
        i0 = acn_index(l, 0)
        Xc[:, i0] = X_block_n3d[:, i0].astype(np.complex128)
        for m in range(1, l + 1):
            ic = acn_index(l,  m); is_ = acn_index(l, -m)
            rp = X_block_n3d[:, ic]; rn = X_block_n3d[:, is_]
            Xc[:, ic]  = ((-1)**m) * (rp - 1j*rn) / np.sqrt(2.0)
            Xc[:, is_] =               (rp + 1j*rn) / np.sqrt(2.0)
    return Xc

def real_to_complex_hoa_stft(X_ftc, order):
    """ Apply real->complex mapping to an STFT cube X (F,T,C) in ACN/N3D. Returns complex (F,T,C). """
    F, T, C = X_ftc.shape
    Xc = np.zeros_like(X_ftc, dtype=np.complex128)
    for f in range(F):
        Xc[f] = acn_real_to_complex_n3d_block(X_ftc[f], order)
    return Xc

# ----------------------------
# Beamforming (HOA)
# ----------------------------
def precompute_hoa_steering(order, theta_flat, phi_flat):
    Yc = sh_complex_n3d_matrix(order, theta_flat, phi_flat)  # (K,C)
    g_l = maxre_weights(order)
    g_per_ch = expand_order_weights_per_channel(g_l)
    W_maxre = np.conj(Yc) * g_per_ch[None, :]
    return Yc, g_per_ch, W_maxre

def hoa_broadband_power_maxre(Xc_ftc, W_maxre, f, f_lo=300.0, f_hi=8000.0, phat=False):
    F, T, C = Xc_ftc.shape
    K = W_maxre.shape[0]
    sel = np.where((f >= f_lo) & (f <= f_hi))[0]
    if len(sel) == 0: sel = np.arange(F)
    E = np.zeros((T, K), dtype=np.float64)
    Wt = W_maxre.T  # (C,K)
    for ti in tqdm(range(T), desc="MAX-rE (HOA) frames", unit="frm"):
        X = Xc_ftc[sel, ti, :]  # (F_sel, C)
        if phat:
            X = X / (np.abs(X) + 1e-12)
        Y = X @ Wt               # (F_sel,K)
        P = np.mean(np.abs(Y)**2, axis=0)
        E[ti] = P.real
    return E

def hoa_broadband_power_mvdr(Xc_ftc, Yc, g_per_ch, f, map_fps, tau_s=0.25,
                             f_lo=300.0, f_hi=8000.0, lam=1e-3, use_maxre_taper=True):
    F, T, C = Xc_ftc.shape
    K = Yc.shape[0]
    sel = np.where((f >= f_lo) & (f <= f_hi))[0]
    if len(sel) == 0: sel = np.arange(F)
    beta = (1.0 / map_fps) / max(tau_s, 1e-6)
    beta = np.clip(beta, 0.01, 0.5)
    E = np.zeros((T, K), dtype=np.float64)
    R = np.zeros((F, C, C), dtype=np.complex128)
    eyeC = np.eye(C, dtype=np.complex128)
    A = np.conj(Yc)
    if use_maxre_taper:
        A = A * g_per_ch[None, :]
    for ti in tqdm(range(T), desc="MVDR (HOA) frames", unit="frm"):
        X_ft = Xc_ftc[:, ti, :]
        for fi in sel:
            x = X_ft[fi:fi+1, :]
            xxH = (x.conj().T @ x)
            R[fi] = (1 - beta) * R[fi] + beta * xxH
        Pk = np.zeros((K,), dtype=np.float64)
        for fi in sel:
            Rf = R[fi]
            tr = np.trace(Rf).real
            dl = lam * (tr / max(C, 1))
            Rf_dl = Rf + dl * eyeC
            try:
                Rinv = np.linalg.inv(Rf_dl)
            except np.linalg.LinAlgError:
                Rinv = np.linalg.pinv(Rf_dl)
            denom = np.einsum('kc,cd,kd->k', np.conj(A), Rinv, A).real
            Pk += 1.0 / np.clip(denom, 1e-12, None)
        Pk /= max(1, len(sel))
        E[ti] = Pk
    return E

# ----------------------------
# Beamforming (RAW EM64: DAS / SRP-PHAT)
# ----------------------------
def steering_delays(mic_pos_m, dirs, c=C_SOUND):
    """ Plane wave: tau_{m,k} = (r_m · s_k) / c. mic_pos_m: (M,3), dirs: (K,3). Returns (K,M). """
    return (mic_pos_m[None, :, :] @ dirs[:, :, None]).squeeze(-1) / c

def das_broadband_power(X_ftm, f, tau_km, wq=None, f_lo=300.0, f_hi=8000.0,
                        phat=True, n_bands=16, chunk_K=4096):
    F, T, M = X_ftm.shape
    K, M2 = tau_km.shape
    assert M2 == M
    sel = np.where((f >= f_lo) & (f <= f_hi))[0]
    if sel.size == 0: sel = np.arange(F)
    f_sel = f[sel].astype(np.float32)

    fmin, fmax = float(f_sel[0]), float(f_sel[-1])
    edges = np.exp(np.linspace(np.log(fmin), np.log(fmax), n_bands + 1))
    bands = []
    for bi in range(n_bands):
        lo, hi = edges[bi], edges[bi+1]
        msk = (f_sel >= lo) & (f_sel < hi)
        idx = sel[msk]
        if idx.size == 0:
            centre = 0.5 * (lo + hi)
            idx = sel[np.argmin(np.abs(f_sel - centre))][None]
        bands.append(idx)

    if wq is None: wqv = np.ones((M,), dtype=np.float32)
    else:          wqv = wq.astype(np.float32, copy=False)
    wqv_c = (wqv.astype(np.float32)).astype(np.complex64, copy=False)
    tau_km = tau_km.astype(np.float32, copy=False)

    E = np.zeros((T, K), dtype=np.float64)
    two_pi = np.float32(2.0 * np.pi)

    for ti in tqdm(range(T), desc="DAS frames", unit="frm"):
        X_fm = X_ftm[:, ti, :].astype(np.complex64, copy=False)
        if phat:
            X_fm = X_fm / (np.abs(X_fm) + 1e-12).astype(np.float32, copy=False)
        acc_Pk = np.zeros((K,), dtype=np.float64)
        for idx in bands:
            nb = idx.size
            for k0 in range(0, K, chunk_K):
                k1 = min(K, k0 + chunk_K)
                tau_chunk = tau_km[k0:k1, :]  # (Kc,M)
                P_chunk = 0.0
                for b in idx:
                    fb = np.float32(f[b])
                    S_b = np.exp(+1j * two_pi * fb * tau_chunk, dtype=np.complex64)
                    x_b = (wqv_c * X_fm[b, :].astype(np.complex64, copy=False))
                    y = S_b @ x_b
                    P_chunk += np.abs(y)**2
                P_chunk = (P_chunk / float(nb)).astype(np.float64, copy=False)
                acc_Pk[k0:k1] += P_chunk
        E[ti, :] = acc_Pk / float(len(bands))
    return E

# ----------------------------
# Point cloud loading & mapping
# ----------------------------
'''
def load_point_cloud(path, voxel_size=None):
    log.info("[stage] Loading point cloud: %s", path)
    pcd = o3d.io.read_point_cloud(path)
    if voxel_size and voxel_size > 0:
        log.info("  Downsampling (voxel_size=%.4f)", voxel_size)
        pcd = pcd.voxel_down_sample(voxel_size)
    if not pcd.has_points():
        raise RuntimeError(f"Point cloud has no points: {path}")
    log.info("  points: %d", np.asarray(pcd.points).shape[0])
    return pcd
'''
def load_point_cloud(path, voxel_size=None, flip_x = False):
    log.info("[stage] Loading point cloud: %s", path)
    pcd = o3d.io.read_point_cloud(path)
    if flip_x:
        points = np.asarray(pcd.points)
        points[:, 0] *= -1
        pcd.points = o3d.utility.Vector3dVector(points)
    if voxel_size and voxel_size > 0:
        log.info("  Downsampling (voxel_size=%.4f)", voxel_size)
        pcd = pcd.voxel_down_sample(voxel_size)
    if not pcd.has_points():
        raise RuntimeError(f"Point cloud has no points: {path}")
    log.info("  points: %d", np.asarray(pcd.points).shape[0])
    return pcd

# def compute_point_local_dirs_and_dists(points_room, mic_pos_room, mic_R_room):
#     """
#     Fixed version that properly transforms directions to room frame.
#     """
#     # Vector from mic to each point in room coordinates
#     vec_room = points_room - mic_pos_room[None, :]
#
#     # These are already in room coordinates - don't transform them!
#     # (Your original code was transforming to array frame, which was wrong)
#
#     dists = np.linalg.norm(vec_room, axis=1)
#     dirs_room = np.zeros_like(vec_room)
#     nz = dists > 1e-9
#     dirs_room[nz] = vec_room[nz] / dists[nz, None]
#
#     return dirs_room, dists


def compute_point_local_dirs_and_dists(points_room, mic_pos_room, mic_R_room):
    """
    Convert room points to ARRAY-LOCAL unit directions and distances relative to mic origin.
    mic_R_room maps ARRAY -> ROOM. We need ARRAY local: s_local = R^T * (p_room - mic_pos_room).
    """
    ########vec_room = points_room - mic_pos_room[None, :]
    ########vec_local = vec_room @ mic_R_room.T
    vec_room = points_room - mic_pos_room[None, :]
    vec_local = vec_room @ mic_R_room
    dists = np.linalg.norm(vec_local, axis=1)
    dirs_local = np.zeros_like(vec_local)
    nz = dists > 1e-9
    dirs_local[nz] = vec_local[nz] / dists[nz, None]
    return dirs_local, dists

def assign_points_to_dirs(dirs_local_points, dirs_grid, chunk=250_000):
    """ For each point direction (N,3) assign nearest direction on sphere (K,3) by cosine similarity. """
    N = dirs_local_points.shape[0]
    K = dirs_grid.shape[0]
    idx = np.empty((N,), dtype=np.int32)
    dp_dirs = dirs_grid.astype(np.float32)
    for i0 in tqdm(range(0, N, chunk), desc="Assign dirs", unit="pts"):
        i1 = min(N, i0 + chunk)
        S = dirs_local_points[i0:i1, :].astype(np.float32, copy=False)
        dots = S @ dp_dirs.T
        idx[i0:i1] = np.argmax(dots, axis=1)
    return idx

def distance_attenuation(dists, mode="inverse_square", min_dist=0.5, max_boost=10.0):
    if mode is None or mode == "none": return np.ones_like(dists, dtype=np.float64)
    d = np.maximum(dists, min_dist)
    w = 1.0 / (d * d)
    w = w / (1.0 / (min_dist * min_dist))
    w = np.clip(w, 0.0, max_boost)
    return w.astype(np.float64)

# ----------------------------
# Energy → colors per frame
# ----------------------------
def build_colors_for_frame(e_dir, point_dir_idx, w_dist, vmin, vmax, cmap_name=DEFAULT_CMAP):
    vals = e_dir[point_dir_idx] * w_dist
    vals01 = normalize01(vals, vmin=vmin, vmax=vmax)
    cols = colormap_values(vals01, cmap_name=cmap_name)
    return cols.astype(np.float64, copy=False)

# ----------------------------
# Directional energies driver
# ----------------------------
#################################
def compute_directional_energies(beamformer, raw_file=None, hoa_file=None, geom_csv=None,
                                 map_fps=12, f_lo=300.0, f_hi=8000.0, order=6,
                                 input_norm="SN3D", phat=True, mvdr_tau_s=0.25,
                                 mvdr_lambda=1e-3, mvdr_use_maxre=True, dirs_grid=None,
                                 bandpass_pre=True, lh_to_rh_phi_flip=False):
    '''
    def compute_directional_energies(beamformer, dirs_grid_array_frame, mic_R_room,
                                   raw_file=None, hoa_file=None, geom_csv=None,
                                    map_fps=12, f_lo=300.0, f_hi=8000.0, order=6,
                                    input_norm="SN3D", phat=True, mvdr_tau_s=0.25,
                                    mvdr_lambda=1e-3, mvdr_use_maxre=True, dirs_grid=None,
                                    bandpass_pre=True):
    '''
    """
    Returns: E: (T, K) float64 energies, hop_s, fs, (optionally also HOA STFT pack for decoding)
    """
    assert dirs_grid is not None, "dirs_grid must be provided"


    ###############################
    # Transform the beamforming directions from array frame to room frame

    #dirs_grid_room = (mic_R_room @ dirs_grid_array_frame.T).T

    # Ensure unit vectors
    #norms = np.linalg.norm(dirs_grid_room, axis=1, keepdims=True)
    #dirs_grid_room = dirs_grid_room / np.maximum(norms, 1e-12)

    if beamformer.lower() in ("das", "srp-phat", "srp"):
        if raw_file is None or geom_csv is None:
            raise ValueError("DAS requires raw_file and geom_csv.")
        log.info("[stage] Loading RAW 64: %s", raw_file)
        raw, fs_raw = sf.read(raw_file, always_2d=True)
        log.info("  raw shape: %s @ %d Hz", raw.shape, fs_raw)
        if bandpass_pre:
            log.info("[stage] Band-pass RAW (%.0f..%.0f Hz)", f_lo, f_hi)
            bandpass_inplace(raw, fs_raw, lo=f_lo, hi=f_hi, order=4)
        log.info("[stage] STFT RAW 64 (map_fps=%s)", map_fps)
        Xraw_ftm, f, _, hop_s, _, _ = stft_multi(raw, fs_raw, map_fps=map_fps)
        log.info("[stage] Load EM64 geometry: %s", geom_csv)
        mic_pos, wq = load_em64_geometry(geom_csv)
        tau_km = steering_delays(mic_pos, dirs_grid)  # ARRAY frame
        log.info("[stage] DAS/SRP-PHAT energy...")
        E = das_broadband_power(Xraw_ftm, f, tau_km, wq=wq, f_lo=f_lo, f_hi=f_hi, phat=phat)
        return E, hop_s, fs_raw, None  # no HOA STFT pack

    elif beamformer.lower() in ("maxre", "mvdr"):

        if hoa_file is None:
            raise ValueError(f"{beamformer} requires hoa_file.")
        log.info("[stage] Loading HOA: %s", hoa_file)
        hoa, fs_hoa = sf.read(hoa_file, always_2d=True)
        C_hoa = hoa.shape[1]
        if C_hoa != (order + 1) ** 2:
            raise ValueError(f"HOA channels {C_hoa} != {(order+1)**2} for order={order}")
        log.info("  HOA shape: %s @ %d Hz", hoa.shape, fs_hoa)
        if bandpass_pre:
            log.info("[stage] Band-pass HOA (%.0f..%.0f Hz)", f_lo, f_hi)
            bandpass_inplace(hoa, fs_hoa, lo=f_lo, hi=f_hi, order=4)
        if input_norm.upper() == "SN3D":
            log.info("[stage] HOA SN3D -> N3D")
            sn3d_to_n3d_inplace(hoa, order)
        log.info("[stage] STFT HOA (map_fps=%s)", map_fps)
        Xhoa_ftc, f, _, hop_s, _, window = stft_multi(hoa, fs_hoa, map_fps=map_fps)
        log.info("[stage] HOA real->complex (per bin)")
        Xhoa_cplx = real_to_complex_hoa_stft(Xhoa_ftc, order)
        theta, phi = cart2sph_dirs(dirs_grid, phi_flip=lh_to_rh_phi_flip)
        Yc, g_per_ch, W_maxre = precompute_hoa_steering(order, theta, phi)
        if beamformer.lower() == "maxre":
            log.info("[stage] MAX-rE energy...")
            E = hoa_broadband_power_maxre(Xhoa_cplx, W_maxre, f, f_lo=f_lo, f_hi=f_hi, phat=False)
        else:
            log.info("[stage] MVDR energy...")
            E = hoa_broadband_power_mvdr(
                Xhoa_cplx, Yc, g_per_ch, f, map_fps=map_fps, tau_s=mvdr_tau_s,
                f_lo=f_lo, f_hi=f_hi, lam=mvdr_lambda, use_maxre_taper=mvdr_use_maxre
            )
        pack = dict(fs=fs_hoa, hop_s=hop_s, window=window, Xhoa_cplx=Xhoa_cplx, Yc=Yc, g=g_per_ch)
        return E, hop_s, fs_hoa, pack
    else:
        raise ValueError(f"Unknown beamformer: {beamformer}")

# ----------------------------
# Resample energy frames to a target FPS
# ----------------------------
def resample_energy_frames(E_tk, src_fps, dst_fps):
    T_src = E_tk.shape[0]
    if abs(src_fps - dst_fps) < 1e-9:
        return E_tk
    dur = T_src / float(src_fps)
    T_dst = int(np.round(dur * float(dst_fps)))
    t_src = np.linspace(0, dur, T_src, endpoint=False)
    t_dst = np.linspace(0, dur, T_dst, endpoint=False)
    idx = np.clip(np.searchsorted(t_src, t_dst, side='right') - 1, 0, T_src - 1)
    return E_tk[idx]

# ----------------------------
# Simple binaural (static head) from HOA via MAX-rE beams (±30°)
# ----------------------------
def decode_binaural_from_hoa_pack(
    hoa_pack, order, azimuth_left_deg=-30.0, azimuth_right_deg=+30.0, elev_deg=0.0
):
    """
    Create a stereo track by steering two MAX-rE beams in HOA STFT domain (no HRTF).
    - azimuth/elevation defined in array frame (phi from +X, theta from +Z).
    """
    if hoa_pack is None:
        raise ValueError("HOA pack required for binaural decoding (use MAX-rE or MVDR path).")
    fs = hoa_pack["fs"]; hop_s = hoa_pack["hop_s"]; window = hoa_pack["window"]
    Xc = hoa_pack["Xhoa_cplx"]  # (F,T,C)
    g = hoa_pack["g"]

    def steer_for(az_deg, el_deg):
        phi = np.deg2rad(az_deg)
        theta = np.deg2rad(90.0 - el_deg)
        Y_one = sh_complex_n3d_matrix(order, np.array([theta]), np.array([phi]))  # (1,C)
        W = np.conj(Y_one) * g[None, :]  # (1,C)
        return W.reshape(-1)  # (C,)

    wL = steer_for(azimuth_left_deg, elev_deg)
    wR = steer_for(azimuth_right_deg, elev_deg)

    F, T, C = Xc.shape
    Y_stft = np.zeros((F, T, 2), dtype=np.complex128)
    for ti in tqdm(range(T), desc="Decode binaural (MAX-rE beams)", unit="frm"):
        X_ftc = Xc[:, ti, :]                   # (F,C)
        Y_stft[:, ti, 0] = X_ftc @ wL.conj()   # Left
        Y_stft[:, ti, 1] = X_ftc @ wR.conj()   # Right

    y_lr = istft_multi(Y_stft, fs, hop_s, window)  # (N,2)
    peak = np.max(np.abs(y_lr)) + 1e-9
    if peak > 0.99:
        y_lr = 0.99 * y_lr / peak
    return y_lr, fs

# ----------------------------
# Interactive STEP-BY-STEP viewer (no audio)
# ----------------------------
class StepViewer:
    def __init__(self, pcd, E_tk, point_dir_idx, w_dist, cmap_name=DEFAULT_CMAP,
                 global_vmin=None, global_vmax=None, window_title="Acoustic 3D Stepper",
                 point_size=3.0, bg_color=(0.12, 0.12, 0.12),
                 mic_pos_room=None, mic_R_room=None,
                 fov_h_deg=90.0, fov_v_deg=50.0, win_w=1280, win_h=720,
                 normalize_mode="per_frame", color_floor=0.03,
                 colors_list=None,          # <-- NEW: precomputed colors [T x (N,3)]
                 room_oriented_camera=True,  # <-- NEW: use point-cloud frame for camera
                 hop_s=None, step_seconds=5.0,  camera_eye_offset_room=(0.0, 0.0, 0.0),
                 ):
        self.hop_s = float(hop_s)
        self.step_frames = max(1, int(round(step_seconds / self.hop_s)))
        self.pcd = pcd
        self.E = E_tk
        self.colors_list = colors_list
        self.idx = point_dir_idx
        self.wd_base = w_dist
        self.use_atten = True
        self.T, self.K = E_tk.shape
        self.cmap = cmap_name
        self.norm_mode = normalize_mode
        self.color_floor = float(color_floor)
        self.vmin_g = 0.0 if global_vmin is None else float(global_vmin)
        self.vmax_g = np.percentile(E_tk, 90) if global_vmax is None else float(global_vmax)
        self.window_title = window_title
        self.point_size = float(point_size)
        self.bg_color = np.array(bg_color, dtype=np.float32)
        self.frame = 0
        self.gain = 1.0
        self.room_oriented_camera = bool(room_oriented_camera)
        self._cam_off = np.asarray(camera_eye_offset_room, dtype=float)

        self._mic_pos_room = np.asarray(mic_pos_room, dtype=float) if mic_pos_room is not None else None
        self._mic_R_room   = np.asarray(mic_R_room,  dtype=float) if mic_R_room  is not None else None
        self._win_w = int(win_w); self._win_h = int(win_h)
        self._fov_h = float(fov_h_deg); self._fov_v = float(fov_v_deg)

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name=window_title, width=win_w, height=win_h)

        # **Ensure no animation is registered** (prevents auto-advance)
        self.vis.register_animation_callback(None)

        opt = self.vis.get_render_option()
        opt.background_color = self.bg_color
        opt.point_size = self.point_size
        if not self.pcd.has_colors():
            self.pcd.paint_uniform_color([0.7, 0.7, 0.7])
        self.vis.add_geometry(self.pcd)
        # Mic axes at mic pose (ARRAY->ROOM rotation) + a small sphere marker
        if self._mic_pos_room is not None:
            if self._mic_R_room is None:
                self._mic_R_room = np.eye(3, dtype=float)
            T_mic = self._make_T(self._mic_R_room, self._mic_pos_room)

            self.axes_mic = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4)
            self.axes_mic.transform(T_mic)
            self.vis.add_geometry(self.axes_mic)

            self.mic_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
            self.mic_sphere.compute_vertex_normals()
            self.mic_sphere.paint_uniform_color([0.95, 0.20, 0.20])  # red dot
            self.mic_sphere.transform(self._make_T(np.eye(3), self._mic_pos_room))
            self.vis.add_geometry(self.mic_sphere)

        #self.axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        #self.vis.add_geometry(self.axes)

        self.axes_room = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6)
        self.vis.add_geometry(self.axes_room)


        # Show something first (frame to bbox), then go to the fixed mic position with PC orientation.
        ###self._frame_bbox()
        ###self.vis.poll_events(); self.vis.update_renderer()
        self._reset_to_mic()

        # Initial colors
        self._apply_colors(self.frame, redraw=True)

        # Key bindings (unchanged)
        self.vis.register_key_callback(ord('N'), self._step_forward)
        self.vis.register_key_callback(32,       self._step_forward)   # Space
        self.vis.register_key_callback(ord('P'), self._step_back)
        self.vis.register_key_callback(ord('G'), self._step_fwd_1)
        self.vis.register_key_callback(ord('H'), self._step_back_1)
        self.vis.register_key_callback(ord('S'), self._screenshot)
        self.vis.register_key_callback(ord('Q'), self._quit)
        self.vis.register_key_callback(256,      self._quit)   # ESC
        self.vis.register_key_callback(ord('F'), self._frame_bbox)
        self.vis.register_key_callback(ord('M'), self._reset_to_mic)
        self.vis.register_key_callback(ord('Z'), self._darker)
        self.vis.register_key_callback(ord('X'), self._brighter)
        self.vis.register_key_callback(ord('A'), self._toggle_atten)
        self.vis.register_key_callback(ord('I'), self._print_ref_frames)

    def _print_ref_frames(self, vis=None):
        try:
            vc = self.vis.get_view_control()
            params = vc.convert_to_pinhole_camera_parameters()
            Rcw = params.extrinsic[:3, :3]
            t = params.extrinsic[:3, 3]

            # Camera pose in ROOM coords
            eye = -Rcw.T @ t
            r = Rcw.T @ np.array([1.0, 0.0, 0.0])  # right (+X_cam) in ROOM
            u = Rcw.T @ np.array([0.0, 1.0, 0.0])  # up    (+Y_cam) in ROOM
            f = -Rcw.T @ np.array([0.0, 0.0, 1.0])  # front (-Z_cam) in ROOM

            def fmt(v):
                return np.array2string(v, precision=4, floatmode="fixed")

            log.info("[room] frame: +X red, +Y green, +Z blue at origin")
            if self._mic_pos_room is not None:
                log.info("[mic]  pos = %s", fmt(self._mic_pos_room))
            if self._mic_R_room is not None:
                log.info("[mic]  R (ARRAY->ROOM) =\n%s",
                         np.array2string(self._mic_R_room, precision=4, floatmode="fixed"))
            dot_fz = float(np.dot(f, np.array([0.0, 0.0, 1.0])))
            dot_uy = float(np.dot(u, np.array([0.0, 1.0, 0.0])))
            dot_rx = float(np.dot(r, np.array([1.0, 0.0, 0.0])))
            log.info("[cam]  eye   = %s", fmt(eye))
            log.info("[cam]  front = %s  (front·+Z = %.3f)", fmt(f), dot_fz)
            log.info("[cam]  up    = %s  (up·+Y    = %.3f)", fmt(u), dot_uy)
            log.info("[cam]  right = %s  (right·+X = %.3f)", fmt(r), dot_rx)
            if self._mic_pos_room is not None:
                log.info("[delta] |eye - mic_pos| = %.6f m", float(np.linalg.norm(eye - self._mic_pos_room)))
        except Exception as e:
            log.warning("[debug] print frames failed: %s", e)
        return True

    def _toggle_atten(self, vis=None):
        self.use_atten = not self.use_atten
        logging.getLogger("acoustic3d_stepper").info("[stepper] Distance attenuation: %s",
                                                     "ON" if self.use_atten else "OFF")
        self._apply_colors(self.frame, redraw=True)
        return True


    def _make_T(self, R, t):
        T = np.eye(4, dtype=float)
        T[:3, :3] = np.asarray(R, dtype=float)
        T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
        return T

    # def _cam_params_from_pose(self, eye, front_room, up_room, width, height, fov_h_deg, fov_v_deg):
    #     """Fixed camera parameter computation."""
    #     import open3d as o3d
    #
    #     eye = np.asarray(eye, dtype=float)
    #     front = np.asarray(front_room, dtype=float)
    #     up = np.asarray(up_room, dtype=float)
    #
    #     # Normalize
    #     front = front / np.linalg.norm(front)
    #     up = up / np.linalg.norm(up)
    #
    #     # Make up orthogonal to front
    #     up = up - np.dot(up, front) * front
    #     up = up / np.linalg.norm(up)
    #
    #     # Right-handed basis
    #     right = np.cross(up, front)
    #     up = np.cross(front, right)  # Ensure perfect orthogonality
    #
    #     # Camera matrix (world to camera)
    #     Rcw = np.column_stack([right, up, -front])  # Camera looks down -Z
    #     t = -Rcw @ eye
    #
    #     extrinsic = np.eye(4, dtype=float)
    #     extrinsic[:3, :3] = Rcw
    #     extrinsic[:3, 3] = t
    #
    #     # Intrinsics
    #     fx = 0.5 * width / np.tan(np.deg2rad(fov_h_deg) * 0.5)
    #     fy = 0.5 * height / np.tan(np.deg2rad(fov_v_deg) * 0.5)
    #     cx = (width - 1) * 0.5
    #     cy = (height - 1) * 0.5
    #
    #     intrinsic = o3d.camera.PinholeCameraIntrinsic()
    #     intrinsic.set_intrinsics(int(width), int(height), fx, fy, cx, cy)
    #
    #     params = o3d.camera.PinholeCameraParameters()
    #     params.intrinsic = intrinsic
    #     params.extrinsic = extrinsic
    #
    #     return params

    # def _reset_to_mic(self, vis=None):
    #     """Fixed camera reset that properly orients the view."""
    #     try:
    #         if self._mic_pos_room is None:
    #             return True
    #
    #         eye = self._mic_pos_room + self._cam_off
    #
    #         # Look along +Z (blue/forward), up is +Y (green)
    #         front_room = np.array([0.0, 0.0, 1.0])  # +Z blue
    #         up_room = np.array([0.0, 1.0, 0.0])  # +Y green
    #
    #         # Create proper camera parameters
    #         cam = self._cam_params_from_pose(
    #             eye=eye,
    #             front_room=front_room,
    #             up_room=up_room,
    #             width=self._win_w,
    #             height=self._win_h,
    #             fov_h_deg=self._fov_h,
    #             fov_v_deg=self._fov_v
    #         )
    #
    #         vc = self.vis.get_view_control()
    #         vc.convert_from_pinhole_camera_parameters(cam, allow_arbitrary=True)
    #
    #     except Exception as e:
    #         print(f"[camera] Reset failed: {e}")
    #     return True

    '''
    def _reset_to_mic(self, vis=None):
        try:
            if self._mic_pos_room is None:
                return True

            eye = self._mic_pos_room + self._cam_off
            if self.room_oriented_camera:
                cam = _cam_params_from_pose(
                    eye=eye,
                    front_room=np.array([0, 0, -1]),  # +Z_room forward (blue)
                    up_room=np.array([0, -1, 0]),
                    width=self._win_w, height=self._win_h,
                    fov_h_deg=self._fov_h, fov_v_deg=self._fov_v
                )
                eye0, front0, up0 = _lookat_from_o3d_params(cam)
                cam = _cam_params_from_pose(eye0, front0, up0, width=self._win_w, height=self._win_h,
                                            fov_h_deg=self._fov_h, fov_v_deg=self._fov_v)
            else:
                cam = _o3d_cam_from_mic_pose(
                    eye, self._mic_R_room,
                    width=self._win_w, height=self._win_h,
                    fov_h_deg=self._fov_h, fov_v_deg=self._fov_v
                )

            # NEW: Fix flip due to det=-1 and observed orientation
            #_flip_camera_yz_inplace(cam)  # Flips Y/Z to correct upside down
            #_spin_camera_inplace(cam, 'x')  # 180° around X to fix opposite direction (adjust 'x'/'y'/'z' if needed)

            vc = self.vis.get_view_control()
            vc.convert_from_pinhole_camera_parameters(cam, allow_arbitrary=True)

            # NEW: Force update after reset
            self.vis.poll_events()
            self.vis.update_renderer()
        except Exception as e:
            log.warning("[camera] Mic reset failed: %s", e)
        return True
    '''

    def _reset_to_mic(self, vis=None):
        try:
            if self._mic_pos_room is None:
                return True
            eye = self._mic_pos_room + self._cam_off
            front = np.array([0.0, 0.0, 1.0], dtype=float)
            up = np.array([0.0, 1.0, 0.0], dtype=float)
            '''
            cam = _cam_params_from_pose(
                eye=eye,
                front_room=np.array([0, 0, 1]),  # look along +Z_room
                up_room=np.array([0, 1, 0]),  # +Y_room up
                width=self._win_w, height=self._win_h,
                fov_h_deg=self._fov_h, fov_v_deg=self._fov_v
            )
            vc = self.vis.get_view_control()
            vc.convert_from_pinhole_camera_parameters(cam, allow_arbitrary=True)
            '''
            # Drive the view explicitly (center = eye + front)
            vc = self.vis.get_view_control()
            vc.set_front(front.tolist())
            vc.set_up(up.tolist())
            vc.set_lookat((eye + front).tolist())
            self.vis.poll_events();
            self.vis.update_renderer()
        except Exception as e:
            log.warning("[camera] Mic reset failed: %s", e)
        return True

    def _apply_colors(self, frame_idx, redraw=True):
        if self.colors_list is not None:
            cols = self.colors_list[frame_idx]
        else:
            # fallback: compute on the fly
            e_dir = self.E[frame_idx]
            vals = e_dir[self.idx] * (self.wd_base if self.use_atten else 1.0)
            if self.norm_mode == "per_frame":
                vmax = max(1e-12, np.percentile(vals, 90))
                v = np.clip((self.gain * vals) / vmax, 0.0, 1.0)
            else:
                v = np.clip((self.gain * (vals - self.vmin_g)) / (self.vmax_g - self.vmin_g + 1e-12), 0.0, 1.0)
            if self.color_floor > 0.0:
                v = v * (1.0 - self.color_floor) + self.color_floor
            cols = mpl_cmaps.get_cmap(self.cmap)(v).astype(np.float32, copy=False)[:, :3]

        self.pcd.colors = o3d.utility.Vector3dVector(cols)
        if redraw:
            self.vis.update_geometry(self.pcd)
            self.vis.poll_events()
            self.vis.update_renderer()
        logging.getLogger("acoustic3d_stepper").info(
            "[stepper] Frame %d/%d", frame_idx+1, self.T
        )


    def _jump(self, delta):
        self.frame = (self.frame + delta) % self.T
        self._reset_to_mic()
        self._apply_colors(self.frame, redraw=True)
        return True

    def _step_forward(self, vis=None):  return self._jump(+self.step_frames)
    def _step_back(self, vis=None):     return self._jump(-self.step_frames)
    def _step_fwd_1(self, vis=None):    return self._jump(+1)
    def _step_back_1(self, vis=None):   return self._jump(-1)

    def _frame_bbox(self, vis=None):
        # Simple “frame to point cloud” default; rely on Open3D’s default view.
        # If you want a real fit, replace with custom set_lookat/front/up logic.
        return True

    def _next(self, vis=None):
        self.frame = (self.frame + 1) % self.T
        self._reset_to_mic()
        self._apply_colors(self.frame, redraw=True)
        return True

    def _prev(self, vis=None):
        self.frame = (self.frame - 1) % self.T
        self._reset_to_mic()
        self._apply_colors(self.frame, redraw=True)
        return True

    def _skip_fwd_10(self, vis=None):
        self.frame = (self.frame + 10) % self.T
        self._reset_to_mic()
        self._apply_colors(self.frame, redraw=True)
        return True

    def _skip_back_10(self, vis=None):
        self.frame = (self.frame - 10) % self.T
        self._reset_to_mic()
        self._apply_colors(self.frame, redraw=True)
        return True

    def _darker(self, vis=None):
        self.gain /= 1.2
        self._apply_colors(self.frame, redraw=True)
        return True

    def _brighter(self, vis=None):
        self.gain *= 1.2
        self._apply_colors(self.frame, redraw=True)
        return True

    def _screenshot(self, vis=None):
        fn = f"snap_{self.frame:05d}.png"
        self.vis.capture_screen_image(fn, do_render=True)
        log.info("[stepper] Saved %s", fn)
        return True

    def _quit(self, vis=None):
        self._running = False
        return True

    def run(self):
        # tiny manual loop so our key callbacks work as intended
        self._running = True
        try:
            while self._running:
                if not self.vis.poll_events():
                    break
                self.vis.update_renderer()
                time.sleep(0.01)
        finally:
            self.vis.destroy_window()
        return {"last_frame": self.frame, "total_frames": self.T}


# ----------------------------
# Rotating camera path (anchored at mic)
# ----------------------------
def build_rotating_camera_path(
    T_frames, fps, mic_pos_room, mic_R_room,
    rotate_axis="y", period_seconds=30.0,
    keep_level=True,
    pitch_amp_deg=0.0,
    jitter_deg=0.5, seed=13,
    width=1280, height=720,
    fov_h_deg=90.0, fov_v_deg=50.0,
    eye_offset_room=(0.0, 0.0, 0.0),
    tilt_down_deg=0.0,   # constant downward tilt (deg)
):
    """
    Camera fixed at mic position; orientation rotates steadily.
    - rotate_axis: 'x' | 'y' | 'z'
    - period_seconds: time to complete 360°
    - tilt_down_deg: constant downward tilt (elevation), positive looks down.
    """
    rng = random.Random(seed)
    params_list = []

    base_front = np.array([0.0, 0.0, 1.0])  # look along +Z
    base_up    = np.array([0.0, 1.0, 0.0])  # +Y up
    omega = 2.0 * math.pi / max(period_seconds, 1e-6)

    # pick the axis unit vector in ROOM frame
    ax = {"x": np.array([1.0,0,0]), "y": np.array([0,1.0,0]), "z": np.array([0,0,1.0])}[rotate_axis.lower()]

    def R_axis(angle):
        # minimal axis-rotation via composing rpy (use yours for stability)
        if rotate_axis.lower() == "x": return rpy_to_R(0.0, 0.0, angle)
        if rotate_axis.lower() == "y": return rpy_to_R(0.0, angle, 0.0)
        return rpy_to_R(angle, 0.0, 0.0)  # 'z'

    base_front = np.array([0.0, 0.0, 1.0])  # look along +Z
    base_up    = np.array([0.0, 1.0, 0.0])  # +Y up

    # constant "look down" tilt by rotating about +X
    Rx_tilt = rpy_to_R(0.0, 0.0, math.radians(tilt_down_deg))
    front0 = Rx_tilt @ base_front
    up0    = Rx_tilt @ base_up

    for i in range(T_frames):
        t = i / float(fps)
        angle = omega * t
        jitter = math.radians(jitter_deg) * math.sin(2*math.pi*0.1*t + rng.random()*2*math.pi)
        R = R_axis(angle + jitter)

        # optional gentle head-bob around X to keep it natural (default 0)
        if pitch_amp_deg > 0:
            pitch = math.radians(pitch_amp_deg) * math.sin(2*math.pi*(1.0/period_seconds)*t)
            R = rpy_to_R(0.0, 0.0, 0.0) @ R @ rpy_to_R(0.0, pitch, 0.0)  # small extra around Y

        front = R @ front0
        up    = base_up if keep_level else R @ up0

        params = _cam_params_from_pose(
            eye=np.asarray(mic_pos_room, dtype=float) + np.asarray(eye_offset_room, dtype=float),
            front_room=front,
            up_room=up,
            width=width, height=height, fov_h_deg=fov_h_deg, fov_v_deg=fov_v_deg
        )
        #_flip_camera_yz_inplace(params)
        #_spin_camera_inplace(params, 'z')
        params_list.append(params)
    return params_list

# ----------------------------
# Rendering frames with camera path
# ----------------------------
def render_frames_offscreen_fast(
    pcd_legacy, E_dst, idx_dir, w_dist, camera_params, out_mp4_path,
    width=1280, height=720, point_size=2.0, bg_color=(0,0,0), fps=24, cmap_name="inferno",
    per_frame_norm=True, color_floor=0.03,
):
    # Try tensor path; else use legacy geometry (slower but works everywhere)
    try:
        from open3d import core as o3c
        TENSOR_OK = True
    except Exception:
        log.warning("[render] Open3D Tensor API not available; falling back to legacy path.")
        TENSOR_OK = False

    vmin = 0.0
    vmax_global = max(1e-12, np.percentile(E_dst, 90))  # keep global just in case

    renderer = o3dr.OffscreenRenderer(width, height)
    scene = renderer.scene
    scene.set_background(np.array([*bg_color, 1.0], np.float32))  # RGBA

    mat = o3dr.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = float(point_size)

    cam = scene.camera

    # Projection from first camera params
    intr = camera_params[0].intrinsic
    fx, fy = intr.get_focal_length() if hasattr(intr, "get_focal_length") else (
        intr.intrinsic_matrix[0,0], intr.intrinsic_matrix[1,1]
    )
    fov_y = 2.0 * np.degrees(np.arctan(0.5 * height / max(float(fy), 1e-6)))
    cam.set_projection(fov_y, width/height, 0.01, 1000.0, o3dr.Camera.FovType.Vertical)

    writer = imageio.get_writer(
        out_mp4_path, fps=fps, codec="libx264",
        ffmpeg_params=["-preset","ultrafast","-crf","23","-pix_fmt","yuv420p"]
    )

    cmap = mpl_cmaps.get_cmap(cmap_name)
    vmin = 0.0
    vmax = max(1e-12, np.percentile(E_dst, 90))

    geom_name = "pcd"

    if TENSOR_OK:
        # --- fast tensor path ---
        pts = np.asarray(pcd_legacy.points, dtype=np.float32)
        cols0 = np.full((pts.shape[0], 3), 0.1, np.float32)
        tpcd = o3d.t.geometry.PointCloud(o3c.Tensor(pts))
        tpcd.point["colors"] = o3c.Tensor(cols0)
        scene.scene.add_geometry(geom_name, tpcd, mat)

        try:
            T = E_dst.shape[0]
            for ti in range(T):
                params = camera_params[min(ti, len(camera_params)-1)]
                #Rcw = params.extrinsic[:3,:3]; t = params.extrinsic[:3,3]
                #eye   = -Rcw.T @ t
                #front = -(Rcw.T @ np.array([0,0,1.0]))
                #up    =  (Rcw.T @ np.array([0,1.0,0.0]))
                #cam.look_at(eye + front, eye, up)
                eye, front, up = _lookat_from_o3d_params(params)
                cam.look_at(eye + front, eye, up)

                vals = (E_dst[ti][idx_dir] * w_dist).astype(np.float32)
                if per_frame_norm:
                    vmax = max(1e-12, np.percentile(vals, 90))
                else:
                    vmax = vmax_global

                v = np.clip((vals - vmin) / (vmax - vmin + 1e-12), 0.0, 1.0)
                if color_floor > 0.0:
                    v = v * (1.0 - color_floor) + color_floor
                cols = cmap(v).astype(np.float32)[:, :3]
                tpcd.point["colors"] = o3d.core.Tensor(cols, o3d.core.Dtype.Float32)
                scene.scene.update_geometry(geom_name, tpcd, o3dr.Scene.UPDATE_COLORS_FLAG)

                img = renderer.render_to_image()
                writer.append_data(np.asarray(img))
        finally:
            writer.close()
            del renderer

    else:
        # --- legacy path: re-add geometry each frame (works without open3d.core) ---
        try:
            T = E_dst.shape[0]
            for ti in range(T):
                params = camera_params[min(ti, len(camera_params)-1)]
                Rcw = params.extrinsic[:3,:3]; t = params.extrinsic[:3,3]
                eye   = -Rcw.T @ t
                front = -(Rcw.T @ np.array([0,0,1.0]))
                up    =  (Rcw.T @ np.array([0,1.0,0.0]))
                cam.look_at(eye + front, eye, up)

                vals = (E_dst[ti][idx_dir] * w_dist).astype(np.float32)
                v = np.clip((vals - vmin) / (vmax - vmin + 1e-12), 0.0, 1.0)
                cols = cmap(v).astype(np.float32)[:, :3]
                pcd_legacy.colors = o3d.utility.Vector3dVector(cols)

                # Re-add geometry (no UPDATE_COLORS_FLAG available for legacy in offscreen)
                if scene.scene.has_geometry(geom_name):
                    scene.scene.remove_geometry(geom_name)
                scene.scene.add_geometry(geom_name, pcd_legacy, mat)

                img = renderer.render_to_image()
                writer.append_data(np.asarray(img))
        finally:
            writer.close()
            del renderer

def render_frames_open3d_streaming(
        pcd, E_dst, idx_dir, w_dist, camera_params, out_mp4_path, width=1280, height=720,
        point_size=3.0, bg_color=(0, 0, 0), fps=24, add_axes=True, cmap_name=DEFAULT_CMAP):
    import imageio, numpy as np, open3d as o3d

    vmin = 0.0
    vmax = max(1e-12, np.percentile(E_dst, 90))

    os.makedirs(os.path.dirname(out_mp4_path) or ".", exist_ok=True)
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=height)
    opt = vis.get_render_option()
    opt.background_color = np.asarray(bg_color, dtype=np.float32)
    opt.point_size = float(point_size)

    vis.add_geometry(pcd)
    if add_axes:
        vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5))

    ctr = vis.get_view_control()
    writer = imageio.get_writer(
        out_mp4_path, fps=fps, codec="libx264",
        ffmpeg_params=["-preset", "ultrafast", "-crf", "23", "-pix_fmt", "yuv420p"]
    )
    try:
        T = E_dst.shape[0]
        for ti in tqdm(range(T), desc="Render frames", unit="frm"):
            cam_idx = min(ti, len(camera_params) - 1)
            if cam_idx >= 0:
                try:
                    ctr.convert_from_pinhole_camera_parameters(camera_params[cam_idx], allow_arbitrary=True)
                except Exception:
                    pass

            cols = build_colors_for_frame(E_dst[ti], idx_dir, w_dist, vmin, vmax, cmap_name=cmap_name)
            pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float32, copy=False))
            vis.update_geometry(pcd)
            vis.poll_events(); vis.update_renderer()
            img = vis.capture_screen_float_buffer(do_render=True)
            frame = (np.asarray(img) * 255.0).astype(np.uint8)
            writer.append_data(frame)
    finally:
        writer.close()
        vis.destroy_window()


def export_acoustic_video_rotating_camera(
        pcd, E_dst, idx_dir, w_dist, fps_out, cmap_name,
        out_mp4_path, camera_params_list,
        stereo_wav=None, point_size=3.0, bg=(0,0,0)):
    '''
    vmin = 0.0
    vmax = max(1e-12, np.percentile(E_dst, 99.5))
    frames = []
    for ti in tqdm(range(E_dst.shape[0]), desc="Build frame colors", unit="frm"):
        frames.append(build_colors_for_frame(E_dst[ti], idx_dir, w_dist, vmin, vmax, cmap_name))
    '''
    tmp_video = out_mp4_path if stereo_wav is None else out_mp4_path.replace(".mp4", "_noaudio.mp4")
    '''
    render_frames_open3d_streaming(
        pcd, E_dst, idx_dir, w_dist, camera_params_list, tmp_video,
        fps=fps_out, point_size=point_size, bg_color=bg, cmap_name=cmap_name
    )
    '''
    render_frames_offscreen_fast(
        pcd, E_dst, idx_dir, w_dist, camera_params_list, tmp_video,
        width=1280, height=720, point_size=point_size, bg_color=bg,
        fps=fps_out, cmap_name=cmap_name, per_frame_norm=False, color_floor=0.03
    )
    if stereo_wav is None:
        log.info("[done] Wrote %s", out_mp4_path)
        return out_mp4_path

    log.info("[stage] Mux audio into MP4")
    aud = AudioFileClip(stereo_wav)
    clip = VideoFileClip(tmp_video).set_audio(aud)
    clip.write_videofile(out_mp4_path, codec="libx264", audio_codec="aac",
                         fps=fps_out, preset="medium", threads=os.cpu_count() or 4, verbose=True)
    clip.close(); aud.close()
    if os.path.exists(tmp_video) and tmp_video != out_mp4_path:
        os.remove(tmp_video)
    log.info("[done] Wrote %s", out_mp4_path)
    return out_mp4_path

# ----------------------------
# Orchestration (3-D pipeline)
# ----------------------------
def build_acoustic_3d(
    pointcloud_path,
    beamformer="das",           # "das", "maxre", "mvdr"
    raw_wav_path=None,          # for DAS
    hoa_wav_path=None,          # for HOA beamformers + binaural
    em64_geom_csv=None,         # for DAS
    # Pose / alignment
    mic_pos_room=(0.0, 0.0, 2),
    axis_spec = "z,-x,y" ,         # EM64 array -> room mapping (your case)
    mic_rpy_deg=(0.0, 0.0, 0),# extra yaw,pitch,roll if needed
    # Beamforming / grids
    map_fps=12, n_dirs=4096, f_lo=300.0, f_hi=8000.0, order=6, input_norm="SN3D",
    phat=True, mvdr_tau_s=0.25, mvdr_lambda=1e-3, mvdr_use_maxre=True,
    # Rendering / mapping
    voxel_size=None, dist_mode="none", min_dist=0.5, max_boost=10.0,
    energy_norm="global", cmap_name=DEFAULT_CMAP,
):
    pcd = load_point_cloud(pointcloud_path, voxel_size=voxel_size)
    points_room = np.asarray(pcd.points, dtype=np.float64)

    log.info("[stage] Build EM64->room rotation (axis_spec=%s, rpy=%s)", axis_spec, mic_rpy_deg)
    R_axes = build_axis_mapping_R(axis_spec)
    yaw, pitch, roll = [np.deg2rad(x) for x in mic_rpy_deg]
    R_extra = rpy_to_R(yaw, pitch, roll)
    mic_R = R_axes @ R_extra  # ARRAY -> ROOM
    mic_pos = np.asarray(mic_pos_room, dtype=np.float64)

    log.info("[align] Axis map columns (array x,y,z -> room): %s", axis_spec)
    log.info("[align] mic_R (ARRAY->ROOM) det=%.6f\n%s", np.linalg.det(mic_R),
             np.array2string(mic_R, precision=3, floatmode="fixed"))

    log.info("[stage] Build direction grid (full sphere) n_dirs=%d", n_dirs)
    dirs_grid = fibonacci_sphere(n_dirs)  # (K,3) ARRAY frame
    K = dirs_grid.shape[0]
    log.info("  directions K=%d", K)

    log.info("[stage] Compute directional energies (%s)", beamformer.upper())
    ##########lh_to_rh_phi_flip = (np.linalg.det(mic_R) < 0)
    # Eigenmike physical frame is LH; Ambisonics SH expects RH → mirror azimuth for HOA
    array_is_LH = False
    lh_to_rh_phi_flip = array_is_LH
    E_tk, hop_s, fs, hoa_pack = compute_directional_energies(
        beamformer=beamformer,
        raw_file=raw_wav_path, hoa_file=hoa_wav_path, geom_csv=em64_geom_csv,
        map_fps=map_fps, f_lo=f_lo, f_hi=f_hi, order=order, input_norm=input_norm, phat=phat,
        mvdr_tau_s=mvdr_tau_s, mvdr_lambda=mvdr_lambda, mvdr_use_maxre=mvdr_use_maxre,
        dirs_grid=dirs_grid, bandpass_pre=True,
        lh_to_rh_phi_flip=lh_to_rh_phi_flip,  # <-- IMPORTANT
    )
    '''
    E_tk, hop_s, fs, hoa_pack = compute_directional_energies(
        beamformer=beamformer, raw_file=raw_wav_path, hoa_file=hoa_wav_path, geom_csv=em64_geom_csv,
        map_fps=map_fps, f_lo=f_lo, f_hi=f_hi, order=order, input_norm=input_norm, phat=phat,
        mvdr_tau_s=mvdr_tau_s, mvdr_lambda=mvdr_lambda, mvdr_use_maxre=mvdr_use_maxre,
        dirs_grid=dirs_grid, bandpass_pre=True,
    )
    '''
    T = E_tk.shape[0]
    log.info("  frames T=%d, hop=%.3fs, duration≈%.2fs", T, hop_s, T*hop_s)

    log.info("[stage] Map points to direction bins (ARRAY local) + distances")
    dirs_local_pts, dists = compute_point_local_dirs_and_dists(points_room, mic_pos, mic_R)
    idx_dir = assign_points_to_dirs(dirs_local_pts, dirs_grid, chunk=4096)
    w_dist = distance_attenuation(dists, mode=dist_mode, min_dist=min_dist, max_boost=max_boost)

    return dict(pcd=pcd, E=E_tk, hop_s=hop_s, fs=fs, hoa_pack=hoa_pack,
                idx_dir=idx_dir, w_dist=w_dist, cmap=cmap_name,
                mic_R_room=mic_R, mic_pos_room=mic_pos, dirs_grid=dirs_grid)

# ----------------------------
# Helpers for audio in export
# ----------------------------
def _ensure_stereo_from_override(wav_path, prefer_channel0=True):
    """Load any WAV and return a stereo float32 (N,2) and fs. Save temp stereo WAV for mux."""
    x, fs = sf.read(wav_path, always_2d=True)
    if x.shape[1] == 1:
        y = np.repeat(x, 2, axis=1)
    elif x.shape[1] >= 2:
        if prefer_channel0:
            y = np.stack([x[:,0], x[:,0]], axis=-1)
        else:
            mono = np.mean(x, axis=1, keepdims=True)
            y = np.repeat(mono, 2, axis=1)
    else:
        raise RuntimeError("Invalid audio file with zero channels.")
    peak = np.max(np.abs(y)) + 1e-9
    if peak > 0.99:
        y = 0.99 * y / peak
    tmp = os.path.splitext(wav_path)[0] + "_stereo_tmp.wav"
    sf.write(tmp, y.astype(np.float32, copy=False), fs)
    log.info("[audio] Override: %s -> stereo @ %d Hz (saved %s)", wav_path, fs, tmp)
    return tmp

# ----------------------------
# Runners
# ----------------------------
def run_stepper_viewer(
    pointcloud_path, beamformer, raw_wav_path=None, hoa_wav_path=None, em64_geom_csv=None,
    mic_pos_room=(0,0,2), axis_spec = "z,-x,y" , mic_rpy_deg=(0,0,0),
    map_fps=12, n_dirs=2048, f_lo=300, f_hi=8000, order=6, input_norm="SN3D",
    phat=True, mvdr_tau_s=0.25, mvdr_lambda=1e-3, mvdr_use_maxre=True,
    voxel_size=None, dist_mode="none", min_dist=0.5, max_boost=10.0,
    point_size=3.0, bg=(0,0,0),
    fov_h_deg=90.0, fov_v_deg=50.0,
):
    """
    Build pipeline and open the NO-AUDIO stepper viewer. Returns (pack, result).
    """
    pack = build_acoustic_3d(
        pointcloud_path=pointcloud_path,
        beamformer=beamformer, raw_wav_path=raw_wav_path, hoa_wav_path=hoa_wav_path, em64_geom_csv=em64_geom_csv,
        mic_pos_room=mic_pos_room, axis_spec=axis_spec, mic_rpy_deg=mic_rpy_deg,
        map_fps=map_fps, n_dirs=n_dirs, f_lo=f_lo, f_hi=f_hi, order=order, input_norm=input_norm,
        phat=phat, mvdr_tau_s=mvdr_tau_s, mvdr_lambda=mvdr_lambda, mvdr_use_maxre=mvdr_use_maxre,
        voxel_size=voxel_size, dist_mode=dist_mode, min_dist=min_dist, max_boost=max_boost,
    )

    viewer = StepViewer(
        pcd=pack["pcd"], E_tk=pack["E"], point_dir_idx=pack["idx_dir"], w_dist=pack["w_dist"],
        cmap_name=pack["cmap"], point_size=point_size, bg_color=(0.12, 0.12, 0.12),
        mic_pos_room=pack["mic_pos_room"], mic_R_room=pack["mic_R_room"],
        fov_h_deg=fov_h_deg, fov_v_deg=fov_v_deg, win_w=1280, win_h=720,
        normalize_mode="global", color_floor=0.03,
        colors_list=None,
        room_oriented_camera=True,  # <-- use mic orientation
        hop_s=pack["hop_s"],
        step_seconds=5.0,
    )
    result = viewer.run()
    return pack, result

def export_rotating_video(
    pack, out_mp4,
    fps_out=24, yaw_rate_deg=None, yaw_total_deg=360.0, pitch_amp_deg=5.0, pitch_freq_hz=0.05, jitter_deg=1.0, seed=13,
    width=1280, height=720, fov_h_deg=90.0, fov_v_deg=50.0, point_size=3.0, bg=(0,0,0),
    audio_wav_override=None, use_hoa_binaural=True, order=6, azL=-30.0, azR=+30.0, el=0.0
):
    """
    Export a rotating-camera MP4 from artifacts in `pack`.
    - If `audio_wav_override` is provided, it is downmixed/duplicated to stereo and muxed.
    - Else, if `use_hoa_binaural` and HOA pack exists, a stereo is decoded via MAX-rE beams.
    - Camera rotates a total of `yaw_total_deg` by default (360°).
    """
    pcd = pack["pcd"]; E = pack["E"]; hop_s = pack["hop_s"]; hoa_pack = pack["hoa_pack"]
    idx_dir = pack["idx_dir"]; w_dist = pack["w_dist"]; cmap = pack["cmap"]
    mic_pos = pack["mic_pos_room"]; mic_R = pack["mic_R_room"]

    # Energy frames at output fps
    E_dst = resample_energy_frames(E, src_fps=1.0/hop_s, dst_fps=fps_out)

    # Rotating camera path (anchored at mic, full 360° unless yaw_rate is specified)
    T_frames = E_dst.shape[0]
    dur_s = T_frames / float(fps_out)
    period_seconds = 360.0 * dur_s / max(yaw_total_deg, 1e-6)
    cam_list = build_rotating_camera_path(
        T_frames, fps_out, mic_pos, mic_R,
        rotate_axis="y", period_seconds=period_seconds,
        keep_level=True, pitch_amp_deg=0.0, jitter_deg=0.3,
        width=width, height=height, fov_h_deg=fov_h_deg, fov_v_deg=fov_v_deg,
        eye_offset_room=(0.0, 0.0, 0.0),
        tilt_down_deg=0.0,  # was 30.0, now start looking straight
    )

    # Decide audio
    stereo_wav = None
    if audio_wav_override:
        stereo_wav = _ensure_stereo_from_override(audio_wav_override, prefer_channel0=True)
    elif use_hoa_binaural and (hoa_pack is not None):
        y, fs = decode_binaural_from_hoa_pack(hoa_pack, order, azL, azR, el)
        stereo_wav = os.path.splitext(out_mp4)[0] + "_stereo.wav"
        sf.write(stereo_wav, y, fs)
        log.info("[audio] Wrote binaural stereo -> %s", stereo_wav)
    else:
        log.info("[audio] No audio will be muxed.")

    return export_acoustic_video_rotating_camera(
        pcd, E_dst, idx_dir, w_dist, fps_out, cmap, out_mp4,
        camera_params_list=cam_list, stereo_wav=stereo_wav, point_size=point_size, bg=bg
    )


def plot_reference_frames(mic_pos_room, mic_R_room, scale=0.5, save_path=None, show=True):
    """
    Quick 3D preview of ROOM and MIC frames (in ROOM coords).
    - ROOM: origin at (0,0,0), +X red, +Y green, +Z blue (right-handed).
    - MIC: origin at mic_pos_room, axes are columns of mic_R_room.

    Labels over each frame state which frame it is and whether it's RH or LH.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)

    # Handedness: sign of det
    def handed(R):
        return "right-handed" if np.linalg.det(R) > 0 else "left-handed"

    room_R = np.eye(3)
    mic_R  = np.asarray(mic_R_room, dtype=float)
    mic_pos = np.asarray(mic_pos_room, dtype=float)

    log.info("[ref] ROOM frame: right-handed (det=+1)")
    log.info("[ref] MIC  frame: %s (det=%.6f)", handed(mic_R), float(np.linalg.det(mic_R)))

    fig = plt.figure(figsize=(5.2, 4.8))
    ax  = fig.add_subplot(111, projection='3d')
    ax.set_title("Reference frames (in ROOM coordinates)")

    def draw_axes(ax, origin, Rcols, s=0.5, alpha=1.0):
        # Rcols: 3x3 whose columns are +X,+Y,+Z directions in ROOM coords
        colors = ['r','g','b']
        for i in range(3):
            v = Rcols[:, i] * s
            ax.quiver(origin[0], origin[1], origin[2],
                      v[0], v[1], v[2],
                      color=colors[i], linewidth=2, arrow_length_ratio=0.15, alpha=alpha)

    # Draw ROOM at origin
    draw_axes(ax, np.zeros(3), room_R, s=scale)
    ax.text(0, 0, scale*1.15, "Room (RH)", ha='center', va='bottom', fontsize=10)

    # Draw MIC at its pose
    draw_axes(ax, mic_pos, mic_R, s=scale)
    ax.text(*(mic_pos + np.array([0,0,scale*1.15])), f"Mic ({handed(mic_R).split('-')[0]}H)",
            ha='center', va='bottom', fontsize=10)

    # Nice equal aspect around both frames
    pts = []
    for R, o in [(room_R, np.zeros(3)), (mic_R, mic_pos)]:
        for i in range(3):
            pts.append(o)
            pts.append(o + R[:, i]*scale)
    P = np.vstack(pts)
    mins, maxs = P.min(0), P.max(0)
    ctr = (mins + maxs) / 2.0
    rad = (maxs - mins).max() * 0.6 + 1e-6
    ax.set_xlim(ctr[0]-rad, ctr[0]+rad)
    ax.set_ylim(ctr[1]-rad, ctr[1]+rad)
    ax.set_zlim(ctr[2]-rad, ctr[2]+rad)
    ax.set_xlabel("X (room)"); ax.set_ylabel("Y (room)"); ax.set_zlabel("Z (room)")
    ax.view_init(elev=18, azim=35)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        log.info("[ref] Saved preview: %s", save_path)

    if show:
        plt.show(block=False); plt.pause(2.5); plt.close(fig)
    else:
        plt.close(fig)

def build_correct_transformation():
    # EM64: +X=forward, +Y=right, +Z=up
    # Room: +X=left, +Y=up, +Z=forward
    R = np.array([
        [ 0, -1,  0],  # room_x = -array_y
        [ 0,  0,  1],  # room_y = +array_z
        [ 1,  0,  0],  # room_z = +array_x
    ], dtype=np.float64)
    # This gives det = +1 (proper rotation)
    return R

# ----------------------------
# Example usage (edit paths)
# ----------------------------
if __name__ == "__main__":
    # ---- EDIT THESE ----
    POINTCLOUD = r"/home/agjaci-iit.local/em64_processing/limerick_pc.ply"  # your .ply
    RAW64 = r"/media/agjaci/Extreme SSD/ICO_Production_Limerick_Data_Backup/ICO_day1/Atmos-Movement-Winter_T3_Eigenmike_raw.wav"   # 64ch (for DAS) or None
    HOA = r"/media/agjaci/Extreme SSD/ICO_Production_Limerick_Data_Backup/ICO_day1/Atmos-Movement-Winter_T3_Eigenmike_hoa.wav"     # 49ch ACN/SN3D/N3D 6th order (for MAX-rE/MVDR) or None
    GEOM = "em64_geom.csv"                # EM64 geometry (for DAS)
    BEAM = "maxre"                        # "das" | "maxre" | "mvdr"
    # --------------------

    MIC_POS     = (0, 2, 0)
    mic_R_room = build_correct_transformation()
    MIC_RPY_DEG = (0, 0, 0)
    axis_spec = "z,-x,y" #####"y,z,x" #"-y,z,x" #"-y,z,x" #"z,-x,y"  # "-y,z,x" "z,-x,y"  # col0=+Z, col1=+X, col2=+Y  → matches the matrix above
    R = build_axis_mapping_R("z,-x,y")
    ex, ey, ez = R[:, 0], R[:, 1], R[:, 2]  # images of array +x,+y,+z in room
    print("x_mic ->", ex, " expected +Z_room")
    print("y_mic ->", ey, " expected -X_room")
    print("z_mic ->", ez, " expected +Y_room")
    print("det =", np.linalg.det(R))
    R_extra = rpy_to_R(0, 0, 0)
    mic_R_room = R @ R_extra
    mic_pos_room = (0, 2, 0)
    pcd = o3d.io.read_point_cloud(POINTCLOUD)

    show_pointcloud_with_frames(pcd, mic_pos_room, mic_R_room)


    try:
        log.info("=== Acoustic 3D (Stepper) ===")
        pack, result = run_stepper_viewer(
            pointcloud_path=POINTCLOUD,
            beamformer=BEAM,
            raw_wav_path=RAW64 if BEAM=="das" else None,
            hoa_wav_path=HOA   if BEAM in ("maxre","mvdr") else None,
            em64_geom_csv=GEOM if BEAM=="das" else None,
            mic_pos_room=(0,2,0),
            axis_spec= axis_spec ,  #### Suggested by chatgpt to avoid det=-1 and reflection: "-y,z,-x" , original: "-y,z,x"   # <- FIXED
            mic_rpy_deg=(0,0,0),
            map_fps=12, n_dirs=2048,          # finer grid helps separation
            f_lo=200, f_hi=8000, order=6, input_norm="SN3D",
            phat=True, mvdr_tau_s=0.25, mvdr_lambda=5e-3, mvdr_use_maxre=True,
            voxel_size=0.05, point_size=3.0, bg=(0,0,0),
            fov_h_deg=90.0, fov_v_deg=50.0,
            # turn OFF attenuation initially for clarity
            dist_mode="none", min_dist=0.5, max_boost=10.0
        )
        log.info("[stepper] Closed at frame %d / %d", result["last_frame"], result["total_frames"])
        '''
        OUT_MP4 = "acoustic_3d_rotating.mp4"
        export_rotating_video(
            pack, out_mp4=OUT_MP4,
            fps_out=12, yaw_rate_deg=None, yaw_total_deg=360.0,
            pitch_amp_deg=4.0, pitch_freq_hz=0.04, jitter_deg=0.8, seed=13,
            width=1280, height=720, fov_h_deg=360.0, fov_v_deg=100.0, point_size=3.0, bg=(0,0,0),
            audio_wav_override=None,          # <- use HOA binaural
            use_hoa_binaural=True, order=6, azL=-30.0, azR=+30.0, el=0.0
        )

        OUT_360 = "acoustic_360_equirect.mp4"
        export_360_equirectangular_video(
            pack, OUT_360,
            width=2048, height=1024, fps_out=12,
            cmap_name="inferno", knn=4, per_frame_norm=False, color_floor=0.03,
            audio_wav_override=None,  # or path/to/your.wav
            use_hoa_binaural=True, order=6, azL=-30.0, azR=+30.0, el=0.0
        )
        '''
        OUT_EQ_PC = "acoustic_360_equirect_pointcloud_new.mp4"
        export_360_pointcloud_equirect(pack, OUT_EQ_PC, width=2048, height=1024, fps_out=12,
                                       cmap_name="inferno", per_frame_norm=False, color_floor=0.03)
    except Exception as e:
        log.error("[main] Error: %s\n%s", e, traceback.format_exc())
