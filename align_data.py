#!/usr/bin/env python3
import os
import numpy as np
import soundfile as sf
import pandas as pd
import matplotlib.pyplot as plt

import scipy.signal as sig
from scipy.signal import stft, get_window
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter
import scipy.special as spsp
from tqdm import tqdm
import scipy.ndimage as ndi

from tqdm import tqdm
import matplotlib.cm as cm
from moviepy.editor import VideoFileClip, VideoClip

C_SOUND = 343.0

# =========================
# Alignment utilities (unchanged)
# =========================

def _to_mono(x):
    """(N,) or (N, C) -> (N,)"""
    return x if x.ndim == 1 else np.mean(x, axis=1)

def _moving_envelope(y, fs, win_ms=5.0):
    """Rectified + moving-average envelope."""
    y = np.asarray(y, dtype=np.float64)
    y = np.abs(y)
    w = max(1, int(round(win_ms * 1e-3 * fs)))
    return sig.lfilter(np.ones(w) / w, [1.0], y)

def _highpass(y, fs, fc=300.0, order=4):
    """Gentle HP to de-emphasize rumble before onset detection."""
    b, a = sig.butter(order, fc / (fs * 0.5), btype="highpass")
    return sig.filtfilt(b, a, y)

def find_clap_onset(y, fs, search_start_s=0.0, search_end_s=None):
    """Find the clap onset as the first strong peak in the smoothed envelope. Returns sample index."""
    n = len(y)
    i0 = int(round(search_start_s * fs))
    i1 = n if search_end_s is None else min(n, int(round(search_end_s * fs)))
    y_seg = y[i0:i1]
    y_seg = _highpass(y_seg, fs, 300.0)
    env = _moving_envelope(y_seg, fs, win_ms=5.0)
    prom = max(1e-6, 2.5 * np.std(env))
    dist = max(1, int(0.05 * fs))
    peaks, _ = sig.find_peaks(env, prominence=prom, distance=dist)
    onset_local = int(np.argmax(env)) if len(peaks) == 0 else int(peaks[0])
    return i0 + onset_local

def load_video_audio_mono(video_path):
    """Load the camera audio as mono float32 in [-1, 1] and its sample rate."""
    clip = VideoFileClip(video_path)
    audio = clip.audio.to_soundarray(fps=clip.audio.fps)
    fs = int(clip.audio.fps)
    audio_mono = _to_mono(audio).astype(np.float32)
    duration = clip.duration
    clip.close()
    return audio_mono, fs, duration

def align_hoa_to_video(
    ambisonic_wav_path,
    raw_wav_path,
    video_path,
    output_hoa_path,
    output_raw_path,
    output_video_path,
    raw_onset_channel=0,
    hoa_onset_channel=0,
    pre_roll_s=0.5,
    search_window_s=None,
):
    """
    Aligns the HOA WAV to the video based on a clap, and writes aligned files.
    Returns a dict with offsets and paths.
    """
    hoa, fs_hoa = sf.read(ambisonic_wav_path, always_2d=True)
    n_hoa, c_hoa = hoa.shape
    if c_hoa < hoa_onset_channel + 1:
        raise ValueError(f"Requested HOA channel {hoa_onset_channel} not available (C={c_hoa}).")
    ch0 = hoa[:, hoa_onset_channel].astype(np.float64)

    cam_audio, fs_cam, vid_duration = load_video_audio_mono(video_path)
    print("FS cam", fs_cam)
    n_cam = len(cam_audio)

    raw, fs_raw = sf.read(raw_wav_path, always_2d=True)
    print("FS raw", fs_raw)
    n_raw, c_raw = raw.shape
    if raw_onset_channel < 0 or raw_onset_channel >= c_raw:
        raise ValueError(f"Requested RAW channel {raw_onset_channel} not available (C={c_raw}).")
    ch0_raw = raw[:, raw_onset_channel].astype(np.float64)

    hoa_search_end = min(search_window_s, n_hoa / fs_hoa) if search_window_s is not None else None
    cam_search_end = min(search_window_s, n_cam / fs_cam) if search_window_s is not None else None
    raw_search_end = min(search_window_s, n_raw / fs_raw) if search_window_s is not None else None

    hoa_onset = find_clap_onset(ch0, fs_hoa, 0.0, hoa_search_end)
    cam_onset = find_clap_onset(cam_audio, fs_cam, 0.0, cam_search_end)
    raw_onset = find_clap_onset(ch0_raw, fs_raw, 0.0, raw_search_end)

    hoa_start = max(0, hoa_onset - int(round(pre_roll_s * fs_hoa)))
    cam_start_s = max(0.0, (cam_onset / fs_cam) - pre_roll_s)
    raw_start = max(0, int(round(raw_onset - pre_roll_s * fs_raw)))

    hoa_len_s = (n_hoa - hoa_start) / fs_hoa
    cam_len_s_audio = (n_cam / fs_cam) - cam_start_s
    raw_len_s = (n_raw - raw_start) / fs_raw

    aligned_len_s = max(0.0, min(hoa_len_s, cam_len_s_audio, vid_duration - cam_start_s, raw_len_s))
    if aligned_len_s <= 0:
        raise RuntimeError("No overlap after clap-based trimming; check your inputs or pre_roll_s.")

    hoa_end = hoa_start + int(round(aligned_len_s * fs_hoa))
    hoa_aligned = hoa[hoa_start:hoa_end, :]
    sf.write(output_hoa_path, hoa_aligned, fs_hoa, subtype="PCM_24")

    raw_end = raw_start + int(round(aligned_len_s * fs_raw))
    raw_aligned = raw[raw_start:raw_end, :]
    sf.write(output_raw_path, raw_aligned, fs_raw, subtype="PCM_24")


    vid = VideoFileClip(video_path)
    t0 = max(0.0, min(vid.duration, cam_start_s))
    t1 = max(t0, min(vid.duration, cam_start_s + aligned_len_s))
    sub = vid.subclip(t0, t1)
    sub.write_videofile(
        output_video_path,
        codec="libx264",
        audio_codec="aac",
        preset="medium",
        threads=os.cpu_count() or 4,
        fps=sub.fps,
        verbose=False,
        logger=None,
    )
    sub.close()
    vid.close()


    offset_cam_hoa = (cam_onset / fs_cam) - (hoa_onset / fs_hoa)
    offset_raw_hoa = (raw_onset / fs_raw) - (hoa_onset / fs_hoa)
    offset_cam_raw = (cam_onset / fs_cam) - (raw_onset / fs_raw)
    return {
        "hoa_onset_sample": hoa_onset,
        "raw_onset_sample": raw_onset,
        "cam_onset_sample": cam_onset,
        "hoa_fs": fs_hoa,
        "raw_fs": fs_raw,
        "cam_fs": fs_cam,
        "offset_seconds_cam_minus_hoa": float(offset_cam_hoa),
        "offset_seconds_raw_minus_hoa": float(offset_raw_hoa),
        "offset_seconds_cam_minus_raw": float(offset_cam_raw),
        "aligned_duration_seconds": float(aligned_len_s),
        "hoa_start_sample": hoa_start,
        "raw_start_sample": raw_start,
        "video_subclip_start_s": float(t0),
        "video_subclip_end_s": float(t1),
        "output_hoa_path": output_hoa_path,
        "output_raw_path": output_raw_path,
        "output_video_path": output_video_path,
    }

'''
def plot_alignment_before_after(
    ambisonic_wav_path,
    video_path,
    hoa_onset_channel=0,/home/agjaci-iit.local/miniconda3/envs/em64_analysis/bin/python /home/agjaci-iit.local/em64_processing/align_data.py

    pre_roll_s=0.5,
    window_s=0.5,
    search_window_s=None,
):
    """Simple before/after visualization around the clap."""
    hoa, fs_hoa = sf.read(ambisonic_wav_path, always_2d=True)
    ch0 = hoa[:, hoa_onset_channel].astype(np.float64)
    cam, fs_cam, _ = load_video_audio_mono(video_path)

    hoa_search_end = None if search_window_s is None else min(search_window_s, len(ch0)/fs_hoa)
    cam_search_end = None if search_window_s is None else min(search_window_s, len(cam)/fs_cam)

    hoa_on = find_clap_onset(ch0, fs_hoa, 0.0, hoa_search_end)
    cam_on = find_clap_onset(cam, fs_cam, 0.0, cam_search_end)

    hoa_w0 = max(0, hoa_on - int(round(pre_roll_s*fs_hoa)))
    cam_w0 = max(0, cam_on - int(round(pre_roll_s*fs_cam)))
    hoa_w1 = min(len(ch0), hoa_on + int(round(window_s*fs_hoa)))
    cam_w1 = min(len(cam), cam_on + int(round(window_s*fs_cam)))

    t_hoa_before = np.arange(hoa_w0, hoa_w1) / fs_hoa
    t_cam_before = np.arange(cam_w0, cam_w1) / fs_cam

    plt.figure(figsize=(9, 4))
    plt.title("BEFORE alignment: camera vs HOA ch0 (around clap)")
    plt.plot(t_cam_before, cam[cam_w0:cam_w1], alpha=0.8, label="Camera audio")
    plt.plot(t_hoa_before, ch0[hoa_w0:hoa_w1], alpha=0.8, label="HOA ch0")
    plt.axvline(cam_on / fs_cam, linestyle="--", alpha=0.6)
    plt.axvline(hoa_on / fs_hoa, linestyle="--", alpha=0.6)
    plt.xlabel("Time (s)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    hoa_rel = np.arange(hoa_w0, hoa_w1) / fs_hoa - (hoa_on / fs_hoa)
    cam_rel = np.arange(cam_w0, cam_w1) / fs_cam - (cam_on / fs_cam)

    plt.figure(figsize=(9, 4))
    plt.title("AFTER alignment: re-centered on detected clap onset")
    plt.plot(cam_rel, cam[cam_w0:cam_w1], alpha=0.8, label="Camera audio")
    plt.plot(hoa_rel, ch0[hoa_w0:hoa_w1], alpha=0.8, label="HOA ch0")
    plt.axvline(0.0, linestyle="--", alpha=0.6)
    plt.xlabel("Relative time from onset (s)")
    plt.legend()
    plt.tight_layout()
    plt.show()
'''

# =========================
# Geometry / FOV / Rotations
# =========================
def load_em64_geometry(csv_path):
    """
    Reads em64_geom.csv with columns:
      'mic','mic X (m)','mic Y (m)','mic Z (m)','Theta (degrees)','Phi (degrees)','Quad. Weight'
    Returns:
      pos: (M,3) in meters, centered at array origin
      wq: (M,) quadrature weights (float)
    """
    df = pd.read_csv(csv_path)
    # Defensive: accept either exact headers or stripped variants
    def g(col):
        for c in df.columns:
            if c.strip().lower() == col.strip().lower():
                return df[c].to_numpy()
        raise KeyError(f"Column '{col}' not found in {csv_path}. Found: {list(df.columns)}")

    x = g("mic X (m)")
    y = g("mic Y (m)")
    z = g("mic Z (m)")
    wq = g("Quad. Weight").astype(float)
    pos = np.stack([x, y, z], axis=1).astype(float)

    # Re-center (just in case CSV is not zero-mean in geometry)
    pos = pos - np.mean(pos, axis=0, keepdims=True)

    return pos, wq

def _rpy_to_R(yaw, pitch, roll):
    """Right-handed rotations about z (yaw), y (pitch), x (roll)."""
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx

def fov_grid(fov_h_deg=90.0, fov_v_deg=50.0, grid_w=192, grid_h=None,
             yaw0=0.0, pitch0=0.0, roll0=0.0):
    """
    Build a rectilinear FOV grid -> spherical (theta, phi) in array frame.
    Conventions:
      - theta: angle from +Z axis  (0° = +Z)
      - phi  : azimuth in XY from +X, CCW (right-handed)
    FOV maps to approximately:
      phi ∈ [-fov_h/2, +fov_h/2]
      theta ∈ [90 - fov_v/2, 90 + fov_v/2]  (i.e., around horizon)
    Returns:
      theta: (K,)
      phi:   (K,)
      dirs:  (K,3) unit vectors in array frame
      (grid_h, grid_w)
    """
    if grid_h is None:
        grid_h = max(1, int(round(grid_w * (fov_v_deg / fov_h_deg))))

    xs = np.linspace(-1, 1, grid_w)   # -1 left, +1 right
    ys = np.linspace(-1, 1, grid_h)   # -1 top,  +1 bottom
    xx, yy = np.meshgrid(xs, ys)

    # pinhole mapping in camera frame (camera looks along +X_cam)
    half_h = np.deg2rad(fov_h_deg / 2.0)
    half_v = np.deg2rad(fov_v_deg / 2.0)
    tan_h = np.tan(half_h)
    tan_v = np.tan(half_v)

    # Ray in camera coords: [1, u*tan(Fh/2), -v*tan(Fv/2)] then normalize
    rx = np.ones_like(xx)
    ry = xx * tan_h
    rz = -yy * tan_v
    r = np.stack([rx, ry, rz], axis=-1)
    r = r / np.linalg.norm(r, axis=-1, keepdims=True)  # (H,W,3)

    # Rotate camera frame into array frame (small calibration)
    R = _rpy_to_R(yaw0, pitch0, roll0)
    r_arr = r.reshape(-1, 3) @ R.T  # (K,3)

    # Convert to spherical (theta, phi)
    x, y, z = r_arr[:, 0], r_arr[:, 1], r_arr[:, 2]
    theta = np.arccos(np.clip(z, -1.0, 1.0))                  # from +Z
    phi = np.arctan2(y, x)                                    # from +X CCW

    print("phi range (deg):", np.rad2deg(phi).min(), np.rad2deg(phi).max())
    print("theta range (deg):", np.rad2deg(theta).min(), np.rad2deg(theta).max())

    return theta, phi, r_arr, (grid_h, grid_w)

# =========================
# Spherical Harmonics (N3D complex) & helpers
# =========================
def acn_index(l, m):
    return l * l + l + m

def sh_complex_n3d_matrix(order, theta, phi):
    """
    Complex SH (N3D) up to 'order' in ACN order.
    Y: (K, (order+1)^2), with columns n=l^2+l+m
    """
    K = theta.shape[0]
    C = (order + 1) ** 2
    Y = np.zeros((K, C), dtype=np.complex128)
    ct = np.cos(theta)

    for l in range(order + 1):
        P_l0 = spsp.lpmv(0, l, ct)  # (K,)
        N_l0 = np.sqrt((2 * l + 1) / (4 * np.pi))
        Y[:, acn_index(l, 0)] = (N_l0 * P_l0).astype(np.complex128)

        for m in range(1, l + 1):
            P_lm = spsp.lpmv(m, l, ct)
            N_lm = np.sqrt((2*l + 1) / (4*np.pi) * spsp.factorial(l - m) / spsp.factorial(l + m))
            base = N_lm * P_lm
            e_imphi = np.exp(1j * m * phi)
            Y_pos = base * e_imphi                     # Y_l^{+m}
            Y_neg = ((-1)**m) * np.conj(Y_pos)         # Y_l^{-m}
            Y[:, acn_index(l,  m)] = Y_pos
            Y[:, acn_index(l, -m)] = Y_neg
    return Y  # steering (no max-rE taper)

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
    """
    Real ACN/N3D -> Complex N3D (per-sample or per STFT bin).
    For m>0:
        c_{l,+m} = ( r_{l,+m}  - i r_{l,-m} ) / sqrt(2)
        c_{l,-m} = ((-1)^m) * ( r_{l,+m}  + i r_{l,-m} ) / sqrt(2)
    """
    '''
    L, C = X_block_n3d.shape
    Xc = np.zeros((L, C), dtype=np.complex128)
    for l in range(order + 1):
        i0 = acn_index(l, 0)
        Xc[:, i0] = X_block_n3d[:, i0].astype(np.complex128)
        for m in range(1, l + 1):
            ic = acn_index(l,  m)
            is_ = acn_index(l, -m)
            rc = X_block_n3d[:, ic]
            rs = X_block_n3d[:, is_]
            Xc[:, ic] = (rc - 1j * rs) / np.sqrt(2.0)
            Xc[:, is_] = ((-1)**m) * (rc + 1j * rs) / np.sqrt(2.0)
    '''
    L, C = X_block_n3d.shape
    Xc = np.zeros((L, C), dtype=np.complex128)
    for l in range(order + 1):
        i0 = acn_index(l, 0)
        Xc[:, i0] = X_block_n3d[:, i0].astype(np.complex128)
        for m in range(1, l + 1):
            ic = acn_index(l,  m)
            is_ = acn_index(l, -m)
            rp = X_block_n3d[:, ic]   # r_{l,+m}
            rn = X_block_n3d[:, is_]  # r_{l,-m}
            Xc[:, ic]  = ((-1)**m) * (rp - 1j*rn) / np.sqrt(2.0)
            Xc[:, is_] =               (rp + 1j*rn) / np.sqrt(2.0)
    return Xc

def real_to_complex_hoa_stft(X_ftc, order):
    """
    Apply real->complex mapping to an STFT cube X (F,T,C_real) in ACN/N3D.
    Returns Xc (F,T,C_complex=N3D).
    """
    F, T, C = X_ftc.shape
    Xc = np.zeros_like(X_ftc, dtype=np.complex128)
    for f in range(F):
        Xc[f] = acn_real_to_complex_n3d_block(X_ftc[f], order)
    return Xc

# =========================
# STFT utils
# =========================
def stft_multi(x, fs, map_fps=12, win_mult=4, nfft_min=1024, nfft_max=4096, force_float32=True):
    """
    x: (N,C)
    Choose hop ~ fs/map_fps for one TF slice per map frame.
    window_len = win_mult * hop (rounded to pow2 nfft).
    Returns:
      X: (F, T, C), freqs: (F,), times: (T,), hop_s
    """
    N, C = x.shape
    if force_float32 and x.dtype != np.float32:
        x = x.astype(np.float32, copy=False)

    hop = max(1, int(round(fs / float(map_fps))))
    win = max(nfft_min, 1 << int(np.ceil(np.log2(win_mult * hop))))
    if nfft_max is not None:
        win = min(win, int(nfft_max))
    nfft = win

    window = get_window("hann", win, fftbins=True)
    X_list = []
    for c in tqdm(range(C), desc="STFT per channel", unit="ch"):
        f, t, Z = stft(x[:, c], fs=fs, window=window, nperseg=win, noverlap=win - hop,
                       nfft=nfft, boundary=None, padded=False, return_onesided=True)
        X_list.append(Z)  # (F,T)
    X = np.stack(X_list, axis=-1)  # (F,T,C)
    hop_s = hop / float(fs)
    stft_t0 = float(t[0]) if len(t) else 0.0  # time of the first STFT column (sec)
    return X, f.astype(np.float32), t.astype(np.float32), hop_s, stft_t0


def bandpass_inplace(x, fs, lo=150.0, hi=8000.0, order=4):
    b, a = butter(order, [lo/(fs*0.5), hi/(fs*0.5)], btype="bandpass")
    for c in range(x.shape[1]):
        x[:, c] = filtfilt(b, a, x[:, c])

# =========================
# Beamforming: HOA (MAX-rE & MVDR) in frequency domain
# =========================
def precompute_hoa_steering(order, theta_flat, phi_flat):
    """
    Returns steering Yc (K,C) complex N3D and max-rE per-channel g (C,)
    and W_maxre = conj(Yc) * g (K,C)
    """
    Yc = sh_complex_n3d_matrix(order, theta_flat, phi_flat)  # (K,C)
    g_l = maxre_weights(order)
    g_per_ch = expand_order_weights_per_channel(g_l)
    W_maxre = np.conj(Yc) * g_per_ch[None, :]
    return Yc, g_per_ch, W_maxre

def hoa_broadband_power_maxre(Xc_ftc, W_maxre, f, f_lo=300.0, f_hi=8000.0, phat=False):
    """
    Xc_ftc: (F,T,C) complex HOA in N3D complex
    W_maxre: (K,C)
    Returns E[t,K]
    """
    F, T, C = Xc_ftc.shape
    K = W_maxre.shape[0]
    sel = np.where((f >= f_lo) & (f <= f_hi))[0]
    if len(sel) == 0:
        sel = np.arange(F)
    E = np.zeros((T, K), dtype=np.float64)
    Wt = W_maxre.T  # (C,K)
    t_iter = range(T)
    t_iter = tqdm(t_iter, desc="MAX-rE (HOA) frames", unit="frm")
    for ti in t_iter:
        X = Xc_ftc[sel, ti, :]  # (F_sel, C)
        if phat:
            # PHAT whitening across frequency per channel
            X = X / (np.abs(X) + 1e-12)
        # collapse frequency by summing energies after beamforming
        # Y_f = X_f @ Wt  => (F_sel,K)
        Y = X @ Wt
        P = np.mean(np.abs(Y)**2, axis=0)  # (K,)
        E[ti] = P.real
    return E

def hoa_broadband_power_mvdr(Xc_ftc, Yc, g_per_ch, f, map_fps, tau_s=0.25,
                             f_lo=300.0, f_hi=8000.0, lam=1e-3, use_maxre_taper=True):
    """
    MVDR in HOA domain with running SCM per frequency.
    Returns E[t,K]
    """
    F, T, C = Xc_ftc.shape
    K = Yc.shape[0]
    sel = np.where((f >= f_lo) & (f <= f_hi))[0]
    if len(sel) == 0:
        sel = np.arange(F)

    beta = (1.0 / map_fps) / max(tau_s, 1e-6)  # EMA update per frame
    beta = np.clip(beta, 0.01, 0.5)

    E = np.zeros((T, K), dtype=np.float64)

    # Initialize SCMs
    R = np.zeros((F, C, C), dtype=np.complex128)
    eyeC = np.eye(C, dtype=np.complex128)
    A = Yc.copy()  # (K,C)
    if use_maxre_taper:
        A = A * g_per_ch[None, :]
    t_iter = range(T)
    t_iter = tqdm(t_iter, desc="MVDR (HOA) frames", unit="frm")
    for ti in t_iter:
        # update SCMs
        X_ft = Xc_ftc[:, ti, :]  # (F,C)
        for fi in sel:
            x = X_ft[fi:fi+1, :]  # (1,C)
            xxH = (x.conj().T @ x)  # (C,C)
            # running average
            R[fi] = (1 - beta) * R[fi] + beta * xxH

        # beamform for each dir
        Pk = np.zeros(K, dtype=np.float64)
        for fi in sel:
            Rf = R[fi]
            tr = np.trace(Rf).real
            dl = lam * (tr / max(C, 1))
            Rf_dl = Rf + dl * eyeC
            try:
                Rinv = np.linalg.inv(Rf_dl)
            except np.linalg.LinAlgError:
                Rinv = np.linalg.pinv(Rf_dl)

            # denom_k = a_k^H R^{-1} a_k, with a_k = A[k]
            # vectorized over K: (K,C) @ (C,C) @ (C,) -> (K,)
            denom = np.einsum('kc,cd,kd->k', np.conj(A), Rinv, A).real
            Pk += 1.0 / np.clip(denom, 1e-12, None)
        Pk /= max(1, len(sel))
        E[ti] = Pk
    return E

# =========================
# Sum-and-Delay (RAW 64) in frequency domain (DAS / SRP-PHAT)
# =========================
def steering_delays(mic_pos_m, dirs, c=C_SOUND):
    """
    mic_pos_m: (M,3) meters
    dirs: (K,3) unit vectors (pointing direction from array)
    Returns tau: (K,M) delays in seconds (positive for mics in +s direction)
    """
    # plane wave delay tau_{m,k} = (r_m · s_k) / c
    return (mic_pos_m[None, :, :] @ dirs[:, :, None]).squeeze(-1) / c  # (K,M)

'''
def build_raw_steering(f, tau_km, wq=None):
    """
    f: (F,) Hz
    tau_km: (K,M) seconds
    wq: (M,) optional quadrature weights
    Returns S_fkM: (F,K,M) complex steering phases
    """
    F = f.shape[0]
    K, M = tau_km.shape
    # 2π f τ  -> (F,1,1) * (1,K,M)
    phase = -2j * np.pi * f[:, None, None] * tau_km[None, :, :]
    S = np.exp(phase)  # (F,K,M)
    if wq is not None:
        S = S * wq[None, None, :]
    return S
'''


def das_broadband_power(
    X_ftm,                 # (F, T, M) complex STFT of raw mics
    f,                     # (F,) Hz
    tau_km,                # (K, M) delays (s)
    wq=None,               # (M,) quadrature weights or None
    f_lo=300.0,
    f_hi=8000.0,
    phat=True,
    n_bands=16,            # fewer bands = faster (try 12–24)
    chunk_K=4096,          # larger chunks = fewer Python loops
):
    """
    Faster, memory-safe SRP-PHAT / DAS energy:
      - iterate per-frequency (no (B,Kc,M) tensor)
      - BLAS gemv (S_b @ x_b) per frequency, accumulate band mean(|.|^2)
      - complex64 everywhere
    Returns:
      E: (T, K) float64
    """
    # ----- setup & selections -----
    F, T, M = X_ftm.shape
    K, M2 = tau_km.shape
    assert M2 == M

    sel = np.where((f >= f_lo) & (f <= f_hi))[0]
    if sel.size == 0:
        sel = np.arange(F)
    f_sel = f[sel].astype(np.float32)

    # log bands built in the f_sel domain
    fmin, fmax = float(f_sel[0]), float(f_sel[-1])
    edges = np.exp(np.linspace(np.log(fmin), np.log(fmax), n_bands + 1))
    bands = []
    for bi in range(n_bands):
        lo, hi = edges[bi], edges[bi+1]
        m = (f_sel >= lo) & (f_sel < hi)
        idx = sel[m]
        if idx.size == 0:
            centre = 0.5 * (lo + hi)
            idx = sel[np.argmin(np.abs(f_sel - centre))][None]
        bands.append(idx)

    # weights (complex64)
    if wq is None:
        wqv = np.ones((M,), dtype=np.float32)
    else:
        wqv = wq.astype(np.float32, copy=False)
    wqv_c = (wqv.astype(np.float32)).astype(np.complex64, copy=False)

    # delays as float32 once
    tau_km = tau_km.astype(np.float32, copy=False)

    E = np.zeros((T, K), dtype=np.float64)

    two_pi = np.float32(2.0 * np.pi)

    # ----- main loop over time frames -----
    from tqdm import tqdm
    t_iter = tqdm(range(T), desc="DAS fast frames", unit="frm")
    for ti in t_iter:
        # current frame spectrum (F, M) -> cast once
        X_fm = X_ftm[:, ti, :].astype(np.complex64, copy=False)
        if phat:
            X_fm = X_fm / (np.abs(X_fm) + 1e-12).astype(np.float32, copy=False)

        acc_Pk = np.zeros((K,), dtype=np.float64)

        # per band
        for idx in bands:
            nb = idx.size
            # accumulate mean |.|^2 over nb freqs
            # do it in chunks of directions
            for k0 in range(0, K, chunk_K):
                k1 = min(K, k0 + chunk_K)
                tau_chunk = tau_km[k0:k1, :]  # (Kc, M) float32

                # per-frequency accumulation for this chunk
                # P_k = mean_b | S_b @ (wq * X_b) |^2
                # compute vector S_b @ x_b via BLAS gemv for each frequency
                P_chunk = 0.0
                for b in idx:
                    fb = np.float32(f[b])
                    # steering for this single frequency: (Kc, M) complex64
                    S_b = np.exp(-1j * two_pi * fb * tau_chunk, dtype=np.complex64)
                    # weighted mic spectrum: (M,) complex64
                    x_b = (wqv_c * X_fm[b, :].astype(np.complex64, copy=False))
                    # y = S_b @ x_b -> (Kc,)
                    y = S_b @ x_b
                    P_chunk += np.abs(y)**2

                P_chunk = (P_chunk / float(nb)).astype(np.float64, copy=False)
                acc_Pk[k0:k1] += P_chunk

        E[ti, :] = acc_Pk / float(len(bands))

    return E

'''
def das_broadband_power(X_ftm, S_fkm, f, f_lo=300.0, f_hi=8000.0, phat=True):
    """
    X_ftm: (F,T,M) raw STFT
    S_fkm: (F,K,M) steering phases
    Returns E[t,K]
    """
    F, T, M = X_ftm.shape
    _, K, _ = S_fkm.shape
    sel = np.where((f >= f_lo) & (f <= f_hi))[0]
    if len(sel) == 0:
        sel = np.arange(F)

    E = np.zeros((T, K), dtype=np.float64)
    t_iter = range(T)
    t_iter = tqdm(t_iter, desc="DAS / SRP-PHAT frames", unit="frm")
    for ti in t_iter:
        X = X_ftm[sel, ti, :]  # (F_sel,M)
        if phat:
            X = X / (np.abs(X) + 1e-12)  # classic SRP-PHAT whitening
        # Beamform: Y_fk = sum_m S_fkm * X_fm
        # (F_sel,K,M) * (F_sel,M) -> (F_sel,K)
        Y = np.einsum('fkm,fm->fk', S_fkm[sel], X)
        P = np.mean(np.abs(Y)**2, axis=0)  # (K,)
        E[ti] = P.real
    return E
'''
# =========================
# Visualization / Overlay
# =========================
def _normalize_map(E, eps=1e-12, pclip=99.5):
    vmax = np.percentile(E, pclip)
    vmin = 0.0
    return np.clip((E - vmin) / (vmax - vmin + eps), 0.0, 1.0)

def _alpha_blend(frame_rgb, overlay_rgba):
    a = (overlay_rgba[..., 3:4].astype(np.float32) / 255.0)
    o = overlay_rgba[..., :3].astype(np.float32)
    b = frame_rgb.astype(np.float32)
    out = (1 - a) * b + a * o
    return np.clip(out, 0, 255).astype(np.uint8)

def _resize_rgba(arr_rgba, new_h, new_w):
    import numpy as np
    from scipy import ndimage as ndi
    h, w, c = arr_rgba.shape
    assert c == 4
    zoom_h = new_h / float(h)
    zoom_w = new_w / float(w)
    out = np.empty((new_h, new_w, 4), dtype=np.uint8)
    for k in range(4):
        out[..., k] = np.clip(ndi.zoom(arr_rgba[..., k], (zoom_h, zoom_w), order=1), 0, 255).astype(np.uint8)
    return out

def normalize_fixed(E_frame, vmin, vmax, eps=1e-12):
    return np.clip((E_frame - vmin) / (vmax - vmin + eps), 0.0, 1.0)

def overlay_acoustic_video(energy_stack, video_path, output_path, grid_shape, colormap="inferno",
                           alpha=0.55, sigma_px=1.0, map_fps=12, time_offset_s=0.0):
    """
    energy_stack: (T_map, K)
    grid_shape: (H_map, W_map)
    """
    from tqdm import tqdm
    tqdm.write(f"[overlay] Writing: {os.path.basename(output_path)}")
    vid = VideoFileClip(video_path)
    dur = vid.duration
    vfps = vid.fps
    vw, vh = vid.size

    Hm, Wm = grid_shape
    cmap = cm.get_cmap(colormap)

    # temporal resampling: simple nearest neighbor per frame time
    T_map = energy_stack.shape[0]

    def make_frame(t):
        frame = vid.get_frame(t)
        idx = int(np.clip(np.round((t + time_offset_s) * map_fps), 0, T_map - 1))
        E = energy_stack[idx]
        Hn = _normalize_map(E).reshape(Hm, Wm)
        if sigma_px and sigma_px > 0:
            Hn = gaussian_filter(Hn, sigma=sigma_px, mode="nearest")
        heat_rgba = (cmap(Hn) * 255.0).astype(np.uint8)
        heat_rgba[..., 3] = int(round(alpha * 255))
        overlay_rgba = _resize_rgba(heat_rgba, vh, vw)
        blended = _alpha_blend(frame, overlay_rgba)
        return blended

    out = VideoClip(make_frame, duration=dur).set_audio(vid.audio)
    out.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        fps=vfps,
        preset="medium",
        threads=os.cpu_count() or 4,
        verbose=True,
        logger=None,
    )
    vid.close()
    out.close()

# =========================
# Main driver (builds 3 overlays)
# =========================
def build_acoustic_overlays(
    raw_file,
    ambisonic_file,
    video_file,
    eom_file,
    out_prefix=None,
    order=6,
    input_norm="SN3D",     # "SN3D" (AmbiX) or "N3D"
    fov_h_deg=90.0,
    fov_v_deg=50.0,
    map_w=192,
    map_h=None,
    map_fps=12,
    hoa_f_lo=300.0,
    hoa_f_hi=8000.0,
    das_f_lo=300.0,
    das_f_hi=8000.0,
    tau_s=0.25,            # MVDR SCM time constant (s)
    mvdr_lambda=1e-3,
    mvdr_use_maxre=True,
    yaw0=0.0, pitch0=0.0, roll0=0.0,
    do_phat=True,
    bandpass_pre=True,
):
    from tqdm import tqdm
    # --- I/O
    tqdm.write("[stage] Loading audio")
    raw, fs_raw = sf.read(raw_file, always_2d=True)
    hoa, fs_hoa = sf.read(ambisonic_file, always_2d=True)
    if fs_raw != fs_hoa:
        raise ValueError(f"Sample rates differ: raw={fs_raw} HOA={fs_hoa}")
    fs = fs_raw

    # --- geometry
    tqdm.write("[stage] Loading EM64 geometry")
    mic_pos, wq = load_em64_geometry(eom_file)   # (64,3), (64,)

    # --- preprocess
    if bandpass_pre:
        tqdm.write("[stage] Band-pass filtering (RAW & HOA)")
        bandpass_inplace(raw, fs, lo=150.0, hi=8000.0, order=4)
        bandpass_inplace(hoa, fs, lo=150.0, hi=8000.0, order=4)

    # --- normalize HOA to N3D if needed
    C_hoa = hoa.shape[1]
    if C_hoa != (order + 1) ** 2:
        raise ValueError(f"HOA channels {C_hoa} != {(order+1)**2} for order={order}")
    if input_norm.upper() == "SN3D":
        tqdm.write("[stage] Converting HOA SN3D -> N3D")
        sn3d_to_n3d_inplace(hoa, order)

    # --- FOV grid
    tqdm.write("[stage] Building FOV grid")
    theta, phi, dirs, hw = fov_grid(fov_h_deg, fov_v_deg, grid_w=map_w, grid_h=map_h,
                                    yaw0=yaw0, pitch0=pitch0, roll0=roll0)
    grid_h, grid_w = hw
    K = theta.size

    # --- STFTs
    tqdm.write("[stage] STFT (RAW & HOA)")
    Xraw_ftm, f, t_raw, hop_s, stft_t0 = stft_multi(raw, fs, map_fps=map_fps)
    Xhoa_ftc, f2, t_hoa, _, _ = stft_multi(hoa, fs, map_fps=map_fps)
    assert np.allclose(f, f2)

    # --- HOA: real->complex per STFT bin (already N3D)
    tqdm.write("[stage] HOA real->complex (per bin)")
    Xhoa_cplx = real_to_complex_hoa_stft(Xhoa_ftc, order)  # (F,T,C)

    # --- HOA steering / weights
    tqdm.write("[stage] Precompute HOA steering / weights")
    Yc, g_per_ch, W_maxre = precompute_hoa_steering(order, theta, phi)  # (K,C),(C,),(K,C)


    # --- HOA MAX-rE broadband energy (optionally PHAT)

    tqdm.write("[stage] MAX-rE (HOA) broadband power")
    E_maxre = hoa_broadband_power_maxre(
        Xhoa_cplx, W_maxre, f, f_lo=hoa_f_lo, f_hi=hoa_f_hi, phat= False #do_phat
    )  # (T,K)
    '''
    # --- HOA MVDR broadband energy
    tqdm.write("[stage] MVDR (HOA) broadband power")
    E_mvdr = hoa_broadband_power_mvdr(
        Xhoa_cplx, Yc, g_per_ch, f, map_fps=map_fps, tau_s=tau_s,
        f_lo=hoa_f_lo, f_hi=hoa_f_hi, lam=mvdr_lambda, use_maxre_taper=mvdr_use_maxre
    )  # (T,K)
    '''
    # --- RAW 64: DAS / SRP-PHAT broadband energy
    '''
    tau_km = steering_delays(mic_pos, dirs)  # (K, M)
    E_das = das_broadband_power(
        Xraw_ftm, f, tau_km, wq=wq,
        f_lo=das_f_lo, f_hi=das_f_hi,
        phat=do_phat,
        n_bands=16,  # try 12–24
        chunk_K=4096,  # 2048 or 4096
    )
    '''
    # --- Overlay videos
    # Decide names
    base = out_prefix if out_prefix is not None else os.path.splitext(video_file)[0]
    OUT1 = f"{base}_acoustic_overlay_maxre.mp4"
    OUT2 = f"{base}_acoustic_overlay_mvdr.mp4"
    OUT3 = f"{base}_acoustic_overlay_das.mp4"


    tqdm.write("[stage] Overlay: MAX-rE")
    overlay_acoustic_video(
        E_maxre, video_file, OUT1, (grid_h, grid_w),
        colormap="inferno", alpha=0.55, sigma_px=2.0, map_fps=map_fps,time_offset_s=stft_t0
    )
    '''
    tqdm.write("[stage] Overlay: MVDR")
    overlay_acoustic_video(
        E_mvdr, video_file, OUT2, (grid_h, grid_w),
        colormap="inferno", alpha=0.55, sigma_px=2.0, map_fps=map_fps,time_offset_s=stft_t0
    )
    '''
    '''
    tqdm.write("[stage] Overlay: DAS / SRP-PHAT")
    overlay_acoustic_video(
        E_das, video_file, OUT3, (grid_h, grid_w),
        colormap="inferno", alpha=0.55, sigma_px=1.0, map_fps=map_fps,time_offset_s=stft_t0
    )
    '''
    return {"maxre": OUT1, "mvdr": OUT2, "das": OUT3}

# =========================
# Demo driver
# =========================

def demo_overlay():

    raw_file = "/media/agjaci/Extreme SSD/anechoic_recordings/em64/ES3_20250807_170126_raw-3.wav" #"/home/agjaci-iit.local/em64_processing/ES3_20250805_111732_raw 1.wav" #"/media/agjaci/Extreme SSD/anechoic_recordings/em64/ES3_20250807_170126_raw-3.wav"
    ambisonic_file = "/media/agjaci/Extreme SSD/anechoic_recordings/em64/ES3_20250807_170126_hoa-3.wav" #"/home/agjaci-iit.local/em64_processing/ES3_20250805_111732_hoa.wav" #"/media/agjaci/Extreme SSD/anechoic_recordings/em64/ES3_20250807_170126_hoa-3.wav"
    video_file = "/media/agjaci/Extreme SSD/anechoic_recordings/camera/2025-08-13 15-15-20.mkv" #"/home/agjaci-iit.local/em64_processing/2025-08-05 15-07-09.mkv" #"/media/agjaci/Extreme SSD/anechoic_recordings/camera/2025-08-13 15-15-20.mkv" #2025-08-13 15-24-22

    out_hoa = os.path.splitext(ambisonic_file)[0] + "_aligned.wav"
    out_raw = os.path.splitext(raw_file)[0] + "_aligned.wav"
    out_vid = os.path.splitext(video_file)[0] + "_aligned.mp4"
    geom_file = "em64_geom.csv"

    '''
    info = align_hoa_to_video(
            ambisonic_wav_path=ambisonic_file,
            raw_wav_path=raw_file,
            video_path=video_file,
            output_hoa_path=out_hoa,
            output_raw_path=out_raw,
            output_video_path=out_vid,
            raw_onset_channel=0,
            hoa_onset_channel=0,
            pre_roll_s=0.5,
            search_window_s=None)

    info = align_hoa_to_video(
        ambisonic_wav_path=ambisonic_file,
        video_path=video_file,
        output_hoa_path=out_hoa,
        output_video_path=out_vid,
        hoa_onset_channel=0,     # W channel (omni) in ACN/N3D 6th-order = channel 0
        pre_roll_s=0.5,          # keep 0.5 s before the clap
        search_window_s=None,    # or e.g., 10.0 if the clap is near the start
    )
    '''

    RAW_ALIGNED = out_raw #"/media/agjaci/Extreme SSD/anechoic_recordings/em64/ES3_20250807_170126_raw-3.wav"
    HOA_ALIGNED = out_hoa #"/media/agjaci/Extreme SSD/anechoic_recordings/em64/ES3_20250807_170126_hoa-3.wav"  # /home/agjaci-iit.local/em64_processing/ES3_20250805_111732_hoa_aligned.wav"
    VID_ALIGNED = out_vid #"/media/agjaci/Extreme SSD/camera/em64/2025-08-13 15-15-20.mkv"  #/home/agjaci-iit.local/em64_processing/2025-08-05 15-07-09_aligned.mp4"

    # compute overplays
    build_acoustic_overlays(
        raw_file=RAW_ALIGNED,
        ambisonic_file=HOA_ALIGNED,
        video_file=VID_ALIGNED,
        eom_file=geom_file,
        out_prefix=os.path.splitext(video_file)[0] + "_aligned",
        order=6,
        input_norm="SN3D",
        fov_h_deg=90.0,
        fov_v_deg=50.0,
        map_w=48,
        map_h=None,
        map_fps=12,
        hoa_f_lo=300.0,
        hoa_f_hi=8000.0,
        das_f_lo=300.0,
        das_f_hi=8000.0,
        tau_s=0.25,
        mvdr_lambda=1e-3,
        mvdr_use_maxre=True,
        yaw0=0.0, pitch0=0.0, roll0=0.0,
        do_phat=True,
        bandpass_pre=True,
    )

    '''
    OUT1 = os.path.splitext(VID_ALIGNED)[0] + "_acoustic_overlay_maxre.mp4"
    overlay_acoustic_map_on_video(
        aligned_hoa_wav=HOA_ALIGNED,
        aligned_video_path=VID_ALIGNED,
        output_video_path=OUT1,
        order=6,
        input_norm="SN3D",         # set "N3D" if your file already is N3D
        fov_h_deg=90.0,
        fov_v_deg=50.0,
        map_w=192,
        window_s=0.10,
        map_fps=12,
        beamformer="maxre",
        sigma_px=1.0,
        tau_s=0.12,
        alpha=0.55,
        yaw0=0.0, pitch0=0.0, roll0=0.0,
        debug=True,
        show_bars=True,
    )
    print("Overlay (max-rE) written to:", OUT1)

    # MVDR overlay
    OUT2 = os.path.splitext(VID_ALIGNED)[0] + "_acoustic_overlay_mvdr.mp4"
    overlay_acoustic_map_on_video(
        aligned_hoa_wav=HOA_ALIGNED,
        aligned_video_path=VID_ALIGNED,
        output_video_path=OUT2,
        order=6,
        input_norm="SN3D",
        fov_h_deg=90.0,
        fov_v_deg=50.0,
        map_w=192,
        window_s=0.10,
        map_fps=12,
        beamformer="mvdr",
        mvdr_lambda=1e-3,
        mvdr_use_maxre=True,
        sigma_px=1.0,
        tau_s=0.12,
        alpha=0.55,
        yaw0=0.0, pitch0=0.0, roll0=0.0,
        debug=True,
        show_bars=True,
    )
    print("Overlay (MVDR) written to:", OUT2)
    '''

def main():
    demo_overlay()


if __name__ == "__main__":
    main()