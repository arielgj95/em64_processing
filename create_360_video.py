#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, logging, numpy as np
from moviepy.editor import VideoFileClip, AudioFileClip, clips_array
from scipy.signal import butter, filtfilt, find_peaks

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")
log = logging.getLogger("align_clap_stack")

# --- EDIT THESE ---
ACOUSTIC_MP4 = r"/home/agjaci-iit.local/em64_processing/acoustic_360_equirect_pointcloud_new.mp4"
RICOH_MP4    = r"/media/agjaci/Extreme SSD/ICO_Production_Limerick_Data_Backup/360_camera1/DCIM/100RICOH/R0010015_winter_T3_st.MP4"
OUT_MP4      = r"./stacked_aligned_clap_pointcloud.mp4"

# Optional: if a video has no/poor audio, supply a separate audio file just for detection
ACOUSTIC_AUDIO_OVERRIDE = None   # e.g. r"/path/to/acoustic_audio.wav"
RICOH_AUDIO_OVERRIDE    = None   # usually keep None

TARGET_WIDTH = 2048
TARGET_FPS   = 30
CRF          = "20"
PRESET       = "medium"

# Optional yaw seam fixes (degrees) for equirect videos
YAW_ACOUSTIC_DEG = 0
YAW_RICOH_DEG    = 180

# Pre-roll shown before the detected clap (we auto-clamp so both clips get the same pre-roll)
PREROLL_SEC       = 0.50

SEARCH_START_SEC = 3.0   # ignore the first N seconds (set 0.0 if not needed)
HP_HZ            = 500.0 # high-pass for transients
SMOOTH_MS        = 3.0   # shorter smoothing keeps the clap “spiky”
Z_THR            = 6.0   # robust z-score gate (median/MAD)
Z_PROM           = 3.0   # robust z prominence
MIN_GAP_MS       = 60.0  # min distance between candidate peaks
MIN_W_MS         = 1.0   # lower width bound (short)
MAX_W_MS         = 35.0  # upper width bound (still short)
MAX_RELAX_STEPS  = 4     # how many times to relax thresholds if nothing is found
RELAX_FACTOR     = 0.8   # multiply thresholds by this each relax step

# ------------------

def estimate_offset_gcc(x, y, sr, max_lag_s=5.0):
    """Return lag (sec) to add to y to align with x (positive => delay y)."""
    # both x,y are mono float arrays at sr, from the same time span
    n = min(len(x), len(y))
    x = x[:n]; y = y[:n]
    # GCC-PHAT
    X = np.fft.rfft(x); Y = np.fft.rfft(y)
    R = X * np.conj(Y)
    R /= (np.abs(R) + 1e-12)
    cc = np.fft.irfft(R, n=2*n-1)
    # shift zero-lag to center
    cc = np.concatenate((cc[-(n-1):], cc[:n]))
    max_lag = int(round(max_lag_s * sr))
    mid = len(cc)//2
    w0, w1 = mid - max_lag, mid + max_lag + 1
    k = np.argmax(cc[w0:w1]) + w0
    lag_samples = k - mid
    return lag_samples / float(sr)

def yaw_roll_equirect(clip, yaw_deg):
    yaw = yaw_deg % 360
    if yaw == 0:
        return clip
    w = clip.w
    dx = int(round(yaw / 360.0 * w))  # +dx shifts pixels right
    def _roll(gf, t):
        f = gf(t)
        return np.roll(f, dx, axis=1)
    return clip.fl(_roll, apply_to=["mask"])


def load_mono(aclip, sr=16000, t_end=30.0, hp=200.0):
    a = aclip.subclip(0, t_end).to_soundarray(fps=sr).mean(axis=1)
    if hp:
        b,a_co = butter(2, hp/(sr*0.5), btype="highpass")
        a = filtfilt(b, a_co, a)
    return a

def to_mono(wav):
    if wav.ndim == 1:
        return wav
    return wav.mean(axis=1)

def first_clap_time(
    audio_clip_or_path,
    sr=48000,
    search_start_sec=0.0,
    search_end_sec=None,
    hp_hz=500.0,
    smooth_ms=3.0,
    z_thr=6.0,
    z_prom=3.0,
    min_gap_ms=120.0,
    w_ms=(1.0, 25.0),
    mode="earliest",          # 'earliest' | 'strongest' | 'nearest'
    hint_sec=None,            # used if mode == 'nearest'
    near_ms=500.0,            # window around hint for 'nearest'
    debug=True,
):
    """Return time (sec) of a short & strong transient (clap)."""
    # 1) load mono at a known sr
    if isinstance(audio_clip_or_path, str):
        ac = AudioFileClip(audio_clip_or_path)
        y = ac.to_soundarray(fps=sr); ac.close()
    else:
        ac = audio_clip_or_path
        y = ac.to_soundarray(fps=sr)
    x = y if y.ndim == 1 else y.mean(axis=1)
    N = len(x)
    if N == 0:
        raise RuntimeError("Empty audio")

    # 2) restrict search
    i0 = int(sr * max(0.0, search_start_sec))
    i1 = N if search_end_sec is None else min(N, int(sr * max(0.0, search_end_sec)))
    if i0 >= i1:
        i0, i1 = 0, N
    xw = x[i0:i1]

    # 3) high-pass + short smoothing
    b, a = butter(2, hp_hz/(sr*0.5), btype="highpass")
    xf = filtfilt(b, a, xw)
    env = np.abs(xf)
    win = max(1, int(round(sr * (smooth_ms/1000.0))))
    env = np.convolve(env, np.ones(win)/win, mode="same")

    # 4) robust z
    med = np.median(env); mad = np.median(np.abs(env - med)) + 1e-12
    z = (env - med) / mad

    # 5) peak search with strict width/spacing
    dist  = int(round(sr * (min_gap_ms/1000.0)))
    wmin  = int(round(sr * (w_ms[0]/1000.0)))
    wmax  = int(round(sr * (w_ms[1]/1000.0)))
    peaks, props = find_peaks(
        z, height=z_thr, prominence=z_prom,
        distance=max(1, dist), width=(max(1, wmin), max(1, wmax))
    )

    if not len(peaks):
        # fallback to 99.5th percentile
        P = np.percentile(z, 99.5)
        peaks, props = find_peaks(z, height=P, width=(max(1,wmin), max(1,wmax)))

    if not len(peaks):
        # ultimate fallback: max z
        idx = int(np.argmax(z))
        t = (idx + i0) / float(sr)
        if debug: log.warning("No peaks; using max z at %.3fs", t)
        return t

    times = (peaks + i0) / float(sr)
    heights = props.get("peak_heights")
    promin  = props.get("prominences", heights)

    # Selection strategy
    if mode == "nearest" and hint_sec is not None:
        mask = np.abs(times - hint_sec) <= (near_ms / 1000.0)
        if np.any(mask):
            k_local = int(np.argmin(np.abs(times[mask] - hint_sec)))
            idx = np.arange(len(times))[mask][k_local]
        else:
            idx = int(np.argmin(np.abs(times - hint_sec)))  # nearest anyway
    elif mode == "strongest":
        idx = int(np.argmax(promin if promin is not None else heights))
    else:  # 'earliest'
        idx = 0

    t = float(times[idx])
    if debug:
        show = min(5, len(peaks))
        info = ", ".join([f"{times[i]:.3f}s z={heights[i]:.1f}" for i in range(show)])
        log.info("candidate peaks: %s", info)
    return t

def upsample_by_hold(clip, dst_fps):
    """Increase fps by duplicating frames (duration unchanged)."""
    src_fps = getattr(clip, "fps", None) or getattr(getattr(clip, "reader", None), "fps", None)
    if not src_fps:
        r = getattr(clip, "reader", None)
        n, d = getattr(r, "nframes", None), getattr(clip, "duration", None)
        src_fps = (n / d) if (n and d) else 12.0
    if abs(src_fps - dst_fps) < 1e-9:
        return clip.set_fps(dst_fps)
    def map_time(t):  # snap each output time to the previous source frame boundary
        return np.floor(t * src_fps) / src_fps
    return clip.fl_time(map_time).set_fps(dst_fps)

def safe_duration(clip):
    d = getattr(clip, "duration", None)
    if d is None:
        r = getattr(clip, "reader", None)
        if r is not None:
            fps, n = getattr(r, "fps", None), getattr(r, "nframes", None)
            if fps and n:
                d = n / float(fps)
    if d is None and getattr(clip, "audio", None) is not None:
        d = getattr(clip.audio, "duration", None)
    return d

def main():
    if not os.path.exists(ACOUSTIC_MP4): raise FileNotFoundError(ACOUSTIC_MP4)
    if not os.path.exists(RICOH_MP4):    raise FileNotFoundError(RICOH_MP4)

    log.info("Loading videos…")
    top0 = VideoFileClip(ACOUSTIC_MP4)   # acoustic pano
    bot0 = VideoFileClip(RICOH_MP4)      # Ricoh pano (has the audio we will keep)

    # Get audio sources to detect the clap
    top_audio = AudioFileClip(ACOUSTIC_AUDIO_OVERRIDE) if ACOUSTIC_AUDIO_OVERRIDE else top0.audio
    bot_audio = AudioFileClip(RICOH_AUDIO_OVERRIDE) if RICOH_AUDIO_OVERRIDE else bot0.audio
    if top_audio is None or bot_audio is None:
        raise RuntimeError("Both videos (or overrides) must have audio to detect the clap.")

    # Detect clap times
    t_top = first_clap_time(
        top_audio, sr=44100,
        search_start_sec=28.0, search_end_sec=34.0,  # tight window
        hp_hz=500.0, smooth_ms=3.0,
        z_thr=6.0, z_prom=3.0, min_gap_ms=150.0, w_ms=(1.0, 25.0),
        mode="nearest", hint_sec=31.711, near_ms=800.0,
    )

    # Ricoh: search early, relax thresholds, gentler HP, shorter smoothing
    t_bot = first_clap_time(
        bot_audio, sr=48000,
        search_start_sec=9.5, search_end_sec=14,  # skip startup pops
        hp_hz=500.0, smooth_ms=2.0,
        z_thr=5.0, z_prom=1.8, min_gap_ms=150.0, w_ms=(1.0, 25.0),
        mode="strongest",  # take the most prominent within the window
    )
    log.info(f"Clap @ top={t_top:.3f}s  bottom={t_bot:.3f}s  (Δ={t_bot - t_top:+.3f}s)")

    # We’ll keep the same pre-roll for both, limited by earliest clap
    pre = float(min(PREROLL_SEC, t_top, t_bot))
    start_top = max(0.0, t_top - pre)
    start_bot = max(0.0, t_bot - pre)

    # Optional yaw seam fix before trimming
    top = yaw_roll_equirect(top0, YAW_ACOUSTIC_DEG)
    bot = yaw_roll_equirect(bot0, YAW_RICOH_DEG)

    # Trim both so the clap lands at time = pre seconds from start in both clips
    top = top.subclip(start_top).set_start(0).set_duration(safe_duration(top) - start_top)
    bot = bot.subclip(start_bot).set_start(0).set_duration(safe_duration(bot) - start_bot)

    # Make both 30 fps (acoustic via frame-hold), resize to common width
    top = upsample_by_hold(top, TARGET_FPS).resize(width=TARGET_WIDTH)
    bot = bot.set_fps(TARGET_FPS).resize(width=TARGET_WIDTH)

    # Recompute common duration (trim to shortest to keep A/V lengths consistent)
    d_top = safe_duration(top)
    d_bot = safe_duration(bot)
    d = min(d_top, d_bot)
    top = top.subclip(0, d).set_duration(d)
    bot = bot.subclip(0, d).set_duration(d)

    # Stack vertically, use Ricoh audio (already aligned by the trim)
    aclip = bot.audio.subclip(0, d).set_duration(d) if bot.audio else None
    stacked = clips_array([[top], [bot]]).set_duration(d)
    if aclip is not None:
        stacked = stacked.set_audio(aclip)

    log.info(f"Writing {OUT_MP4}")
    stacked.write_videofile(
        OUT_MP4,
        codec="libx264", audio_codec="aac",
        fps=TARGET_FPS, preset=PRESET,
        ffmpeg_params=["-crf", CRF, "-pix_fmt", "yuv420p"],
        threads=os.cpu_count() or 4,
    )
    log.info("Done.")

if __name__ == "__main__":
    main()





'''
#!/usr/bin/env python3
import json, subprocess, sys, shutil
from pathlib import Path

# --- defaults so you can just press Run ---
INP  = Path("/media/agjaci/Extreme SSD/ICO_Production_Limerick_Data_Backup/360_camera1/DCIM/100RICOH/R0010015_winter_T3.MP4")
OUT  = Path("/media/agjaci/Extreme SSD/ICO_Production_Limerick_Data_Backup/360_camera1/DCIM/100RICOH/R0010015_winter_T3_full360.mp4")
FOV  = 190       # try 185–200 if the seam shows a gap/overlap
CRF  = 18
PRESET = "medium"

def need(bin_):
    if not shutil.which(bin_):
        sys.exit(f"Error: {bin_} not found in PATH.")

def ffprobe_dims(p: Path):
    j = subprocess.check_output([
        "ffprobe","-v","error","-select_streams","v:0",
        "-show_entries","stream=width,height,side_data_list",
        "-of","json", str(p)
    ])
    st = json.loads(j)["streams"][0]
    return st["width"], st["height"], st.get("side_data_list", [])

def build_vf(w,h,side_data,fov):
    # some THETA files carry 90° rotation metadata; honor it
    transpose = None
    for sd in side_data or []:
        rot = sd.get("rotation")
        if rot in (90,270): transpose = "1"  # 90° CW
        if rot == -90:      transpose = "2"  # 90° CCW
    parts = []
    if transpose: parts.append(f"transpose={transpose}")

    # 2:1 output; for 3840x1920 input SBS this is a good match
    out_w, out_h = 2*h, h

    # *** KEY BIT: side-by-side dual-fisheye layout + explicit FOV ***
    parts.append(
        f"v360=input=dfisheye:output=equirect:"
        f"in_stereo=sbs:ih_fov={int(fov)}:iv_fov={int(fov)}:"
        f"w={out_w}:h={out_h}"
    )
    # If horizon looks mirrored/upside-down, toggle these:
    # parts[-1] += ":ih_flip=1"
    # parts[-1] += ":iv_flip=1"
    # If the seam is in the wrong place, nudge yaw:
    # parts[-1] += ":yaw=180"
    return ",".join(parts)

def main(inp=INP, outp=OUT, fov=FOV):
    need("ffprobe"); need("ffmpeg")
    w,h,sd = ffprobe_dims(inp)
    vf = build_vf(w,h,sd,fov)
    outp.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg","-y","-i",str(inp),
        "-vf", vf,
        "-map","0:v:0","-map","0:a?",
        "-c:v","libx264","-crf",str(CRF),"-preset",PRESET,
        "-pix_fmt","yuv420p","-c:a","copy","-movflags","+faststart",
        str(outp)
    ]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)
    print("Done:", outp)

if __name__ == "__main__":
    # Optional CLI overrides: create_360_video.py <in> <out> [fov]
    argv = sys.argv[1:]
    if len(argv) >= 1: INP = Path(argv[0])
    if len(argv) >= 2: OUT = Path(argv[1])
    if len(argv) >= 3: FOV = int(argv[2])
    main(INP, OUT, FOV)
'''