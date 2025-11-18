#!/usr/bin/env python3
"""
Eigenmike 64 Acoustic Video System
Processes 4th-order ambisonic recordings into directional beam visualization
with binaural audio output using HRTF.
"""

import os
import time
import zipfile
import queue
import threading

import numpy as np
import pandas as pd
import soundfile as sf
from typing import Optional, Tuple


import scipy.signal
import scipy.special
from scipy.signal import stft, lfilter, get_window, butter, filtfilt, sosfiltfilt
from scipy.special import sph_harm
from scipy.spatial import cKDTree

import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from moviepy.editor import VideoFileClip, AudioFileClip
import plotly.graph_objects as go

from pysofaconventions import SOFAFile


# =========================
# Constants
# =========================
C_SOUND = 343.0


# =========================
# Helpers: HOA normalization & mappings
# =========================
def acn_order_index_ranges(order: int):
    """
    For ACN ordering, the channel indices of each order l form a contiguous block
    of length (2l+1), starting at offset l^2. This returns a list of slices.
    """
    ranges = []
    for l in range(order + 1):
        start = l * l
        end = start + (2 * l + 1)
        ranges.append(slice(start, end))
    return ranges


def sn3d_to_n3d_inplace(ch_by_samp: np.ndarray, order: int):
    """
    In-place ACN/SN3D → ACN/N3D per-order scaling for an array shaped (C, N),
    where C = (order+1)^2 are the ambisonic channels in ACN order.
    """
    assert ch_by_samp.shape[0] == (order + 1) ** 2
    for l, rng in enumerate(acn_order_index_ranges(order)):
        s = np.sqrt(2 * l + 1.0)  # per-order N3D scale
        ch_by_samp[rng, :] *= s


def acn_real_to_complex_n3d_block(X_block_n3d: np.ndarray, order: int) -> np.ndarray:
    """
    Real ACN/N3D → Complex ACN/N3D mapping applied column-wise.
    X_block_n3d: (C, N) real, returns (C, N) complex.
    For m>0:
        C_{l,+m} = ( R_{l,+m} - j R_{l,-m} ) / sqrt(2)
        C_{l,-m} = (-1)^m ( R_{l,+m} + j R_{l,-m} ) / sqrt(2)
    """
    C, N = X_block_n3d.shape
    out = np.zeros((C, N), dtype=np.complex128)
    idx = 0
    for l in range(order + 1):
        # m = 0
        out[idx, :] = X_block_n3d[idx, :]
        idx += 1
        for m in range(1, l + 1):
            i_pos = l * l + l + m
            i_neg = l * l + l - m
            rp = X_block_n3d[i_pos, :]
            rn = X_block_n3d[i_neg, :]
            out[i_pos, :] = (rp - 1j * rn) / np.sqrt(2.0)
            out[i_neg, :] = ((-1) ** m) * (rp + 1j * rn) / np.sqrt(2.0)
            idx += 2
    return out


def bandlimit_block_iir(data: np.ndarray, fs: float, band: Optional[Tuple[float, float]]):
    """
    Zero-phase bandlimit a (C,N) array using SOS filtfilt. Supports complex input.
    If band is None, returns input.
    """
    if band is None:
        return data
    f_lo, f_hi = band
    if f_lo is None and f_hi is None:
        return data
    if f_lo is None:
        wn = f_hi / (fs * 0.5)
        sos = butter(4, wn, btype='low', output='sos')
    elif f_hi is None:
        wn = f_lo / (fs * 0.5)
        sos = butter(4, wn, btype='high', output='sos')
    else:
        wn = [f_lo / (fs * 0.5), f_hi / (fs * 0.5)]
        sos = butter(4, wn, btype='band', output='sos')
    # filtfilt along axis=1; works for complex input
    return sosfiltfilt(sos, data, axis=1)


# =========================
# RAW DAS (frequency-domain) helpers
# =========================
def steering_delays(mic_pos_m: np.ndarray, dirs_cart: np.ndarray, c=C_SOUND) -> np.ndarray:
    """
    mic_pos_m: (M,3) meters
    dirs_cart: (K,3) unit vectors (pointing direction from array)
    Returns tau: (K,M) delays in seconds (positive for mics in +s direction)
    plane-wave delay tau_{m,k} = (r_m · s_k) / c
    """
    return (mic_pos_m[None, :, :] @ dirs_cart[:, :, None]).squeeze(-1) / c


def build_dirs_from_thetaphi(beam_dirs):
    """
    beam_dirs: list of (theta,phi)
      theta = colatitude [0..pi], phi = azimuth [0..2pi), +X front, +Y left, +Z up
    Returns unit vectors (K,3).
    """
    dirs = []
    for theta, phi in beam_dirs:
        s = np.array([np.sin(theta) * np.cos(phi),
                      np.sin(theta) * np.sin(phi),
                      np.cos(theta)], dtype=np.float64)
        dirs.append(s)
    return np.stack(dirs, axis=0)


def das_broadband_power_singleframe(
    X_ftm: np.ndarray,            # (F, T, M) complex STFT of current block
    f: np.ndarray,                # (F,) Hz
    tau_km: np.ndarray,           # (K, M) delays (s)
    wq: np.ndarray,               # (M,) quadrature weights
    f_lo=300.0,
    f_hi=8000.0,
    phat=True,
    n_bands=16,
    chunk_K=4096,
) -> np.ndarray:
    """
    Frequency-domain DAS / SRP-PHAT on a single audio block, averaged across the STFT time columns.
    Returns E_k: (K,) power for each look direction.
    """
    F, T, M = X_ftm.shape
    K, M2 = tau_km.shape
    assert M == M2

    sel = np.where((f >= f_lo) & (f <= f_hi))[0]
    if sel.size == 0:
        sel = np.arange(F)
    f_sel = f[sel].astype(np.float32)

    # construct logarithmic bands inside f_sel
    fmin, fmax = float(f_sel[0]), float(f_sel[-1])
    edges = np.exp(np.linspace(np.log(fmin), np.log(fmax), n_bands + 1))
    band_bins = []
    for bi in range(n_bands):
        lo, hi = edges[bi], edges[bi + 1]
        m = (f_sel >= lo) & (f_sel < hi)
        idx = sel[m]
        if idx.size == 0:
            centre = 0.5 * (lo + hi)
            idx = sel[np.argmin(np.abs(f_sel - centre))][None]
        band_bins.append(idx)

    # weights as complex
    wq_c = wq.astype(np.float32)
    wq_c = wq_c.astype(np.complex64, copy=False)

    # result accumulator over time then averaged
    E_time = np.zeros((T, K), dtype=np.float64)

    two_pi = np.float32(2.0 * np.pi)

    # loop over STFT time frames within this block
    for ti in range(T):
        X_fm = X_ftm[:, ti, :].astype(np.complex64, copy=False)
        if phat:
            X_fm = X_fm / (np.abs(X_fm) + 1e-12).astype(np.float32, copy=False)

        acc_Pk = np.zeros((K,), dtype=np.float64)

        for idx in band_bins:
            nb = idx.size
            # chunk over directions to control memory
            for k0 in range(0, K, chunk_K):
                k1 = min(K, k0 + chunk_K)
                tau_chunk = tau_km[k0:k1, :]  # (Kc,M) float32
                P_chunk = 0.0
                for b in idx:
                    fb = np.float32(f[b])
                    # steering for this single frequency and chunk of directions
                    S_b = np.exp(+1j * two_pi * fb * tau_chunk, dtype=np.complex64)  # (Kc,M)
                    x_b = (wq_c * X_fm[b, :])  # (M,)
                    y = S_b @ x_b
                    P_chunk += np.abs(y) ** 2
                P_chunk = (P_chunk / float(nb)).astype(np.float64, copy=False)
                acc_Pk[k0:k1] += P_chunk

        E_time[ti, :] = acc_Pk / float(len(band_bins))

    # average across STFT time columns
    return np.mean(E_time, axis=0)


# =========================
# Ambisonics processor
# =========================
class AmbisonicProcessor:
    """Ambisonic signal processing and HOA beamforming primitives."""

    def __init__(self, order=6, sample_rate=48000):
        self.order = order
        self.sample_rate = sample_rate
        self.num_channels = (order + 1) ** 2  # 49 for 6th order
        self._generate_sh_coefficients()
        self._setup_regularization()

    def _generate_sh_coefficients(self):
        self.sh_coeffs = {}
        self.acn_to_lm = {}
        idx = 0
        for l in range(self.order + 1):
            for m in range(-l, l + 1):
                self.sh_coeffs[(l, m)] = idx
                self.acn_to_lm[idx] = (l, m)
                idx += 1

    @staticmethod
    def _sh_real(l: int, m: int, theta, phi):
        """Real ACN/SN3D basis (cos/sin split)."""
        if m == 0:
            return sph_harm(0, l, phi, theta).real
        if m > 0:
            return np.sqrt(2.0) * sph_harm(m, l, phi, theta).real
        return np.sqrt(2.0) * sph_harm(-m, l, phi, theta).imag

    def spherical_harmonics(self, l, m, theta, phi, norm='n3d'):
        """
        Spherical harmonics Y_l^m(θ,φ).
        For 'n3d' we multiply scipy's orthonormal Y by sqrt(2l+1) to get ACN/N3D.
        """
        Y = sph_harm(m, l, phi, theta)
        if norm.lower() == 'n3d':
            Y = Y * np.sqrt(2 * l + 1.0)
        return Y

    def _setup_regularization(self):
        """
        Proper max-rE modal taper:
            g_l = P_l(cos(alpha)), alpha = pi / (2L + 2)
        """
        L = self.order
        alpha = np.pi / (2 * L + 2)
        ca = np.cos(alpha)
        self.max_re_weights = np.array(
            [scipy.special.eval_legendre(l, ca) for l in range(L + 1)],
            dtype=np.float64
        )

        # Optional in-phase weights, if you need them elsewhere
        self.in_phase_weights = np.zeros(L + 1)
        for l in range(L + 1):
            self.in_phase_weights[l] = 1.0 / (l + 1) if l <= 3 else 0.5 / (l + 1)

    def convert_acn_sn3d_real_to_complex_n3d(self, acn_real_sn3d: np.ndarray) -> np.ndarray:
        """
        Input: (C, N) real ACN/SN3D HOA.
        Output: (C, N) complex ACN/N3D HOA.
        """
        X = acn_real_sn3d.copy()
        sn3d_to_n3d_inplace(X, self.order)
        Xc = acn_real_to_complex_n3d_block(X, self.order)
        return Xc.astype(np.complex128, copy=False)

    def generate_beam_patterns(self, beam_dirs, beam_type='max_re_complex'):
        """
        Complex fixed beams: w(Ω) = d_l * conj(Y_lm(Ω)) in N3D normalization.
        """
        num_beams = len(beam_dirs)
        W = np.zeros((num_beams, self.num_channels), dtype=np.complex128)
        for i, (theta, phi) in enumerate(beam_dirs):
            # steering vector a = Y_n3d(Ω); weight = conj(a) * modal gain
            for l in range(self.order + 1):
                gain = 1.0
                if beam_type == 'max_re_complex':
                    gain = self.max_re_weights[l]
                elif beam_type == 'max_directivity_complex':
                    gain = 1.0
                elif beam_type == 'hypercardioid_complex':
                    if l == 0:
                        gain = 0.25
                    elif l == 1:
                        gain = 0.75
                    else:
                        gain = 0.2 * self.max_re_weights[l]
                for m in range(-l, l + 1):
                    idx = self.sh_coeffs[(l, m)]
                    Ylm = self.spherical_harmonics(l, m, theta, phi, norm='n3d')
                    W[i, idx] = np.conj(Ylm) * gain
            # normalize each beam vector
            nrm = np.linalg.norm(W[i])
            if nrm > 0:
                W[i] /= nrm
        return W

    def beamform(self, ambisonic_signals_cplx_n3d: np.ndarray, beam_weights: np.ndarray):
        """
        ambisonic_signals_cplx_n3d: (C, N) complex ACN/N3D
        beam_weights: (K, C) with entries conj(Y) * d_l
        y_k = sum_n w_{k,n} x_n  (this equals w^H x because w = conj(a))
        """
        y = beam_weights @ ambisonic_signals_cplx_n3d
        p = np.mean(np.abs(y) ** 2, axis=1)
        return y, p

    def compute_directivity_pattern(self, beam_weight_vec: np.ndarray, resolution=360):
        """
        1D cut through the beampattern for diagnostics.
        Returns magnitude response, -3 dB beamwidth (deg), 2D approx directivity index.
        """
        if beam_weight_vec.ndim != 1 or beam_weight_vec.shape[0] != self.num_channels:
            raise ValueError(f"beam_weight_vec must be (C,) with C={(self.order + 1) ** 2}")
        test_angles = np.linspace(0, np.pi, resolution)
        resp = np.zeros(resolution, dtype=np.complex128)
        for i, theta in enumerate(test_angles):
            a = []
            for l in range(self.order + 1):
                for m in range(-l, l + 1):
                    a.append(self.spherical_harmonics(l, m, theta, 0.0, norm='n3d'))
            a = np.array(a, dtype=np.complex128)
            resp[i] = np.vdot(beam_weight_vec, a)  # w^H a
        mag = np.abs(resp)
        mx = np.max(mag)
        if mx < 1e-12:
            return mag, 360.0, 1.0
        th = mx / np.sqrt(2.0)
        idx = np.where(mag >= th)[0]
        if idx.size > 1:
            step = 180.0 / (resolution - 1)
            bw = (idx[-1] - idx[0]) * step
        else:
            bw = 0.0
        bw_rad = np.deg2rad(bw)
        di_approx = np.inf if bw_rad <= 0 else 2.0 / (1.0 - np.cos(bw_rad / 2.0))
        return mag, bw, di_approx

    def compute_mvdr_beamformer(self, ambisonic_signals_cplx_n3d: np.ndarray, beam_dirs,
                                alpha=1e-3):
        """
        Broadband MVDR pseudo-spectrum in HOA domain for a single short block.
        a(Ω) = conj(Y_n3d(Ω)).
        Returns normalized power (K,).
        """
        X = ambisonic_signals_cplx_n3d.astype(np.complex128, copy=False)
        R = (X @ X.conj().T) / X.shape[1]
        R += alpha * np.trace(R).real * np.eye(R.shape[0]) / R.shape[0]
        try:
            R_inv = np.linalg.inv(R)
        except np.linalg.LinAlgError:
            R_inv = np.linalg.pinv(R)

        K = len(beam_dirs)
        denom = np.zeros(K, dtype=np.float64)
        for i, (theta, phi) in enumerate(beam_dirs):
            a = []
            for l in range(self.order + 1):
                for m in range(-l, l + 1):
                    a.append(np.conj(self.spherical_harmonics(l, m, theta, phi, norm='n3d')))
            a = np.array(a, dtype=np.complex128)
            d = np.real(np.vdot(a, R_inv @ a))  # a^H R^{-1} a
            denom[i] = max(d, 1e-12)
        p = 1.0 / denom
        p /= np.max(p) if np.max(p) > 0 else 1.0
        return p


# =========================
# HRTF processor (expects real SN3D HOA)
# =========================
class HRTFProcessor:
    """Handles HRTF load and binaural processing from real SN3D HOA."""

    def __init__(self, sample_rate=48000, hoa_order=6):
        self.sample_rate = sample_rate
        self.hrtf_loaded = False
        self.order = hoa_order
        self.n_ambi = (hoa_order + 1) ** 2

        self.hrir = None
        self.hrir_sr = None
        self.hrir_az_rad = None
        self.hrir_el_rad = None
        self._sh_decode = None
        self._dir_kdtree = None

        self.fir_L = None
        self.fir_R = None
        self.zi_L = None
        self.zi_R = None

    @staticmethod
    def _sph_to_cart(az, el):
        return np.column_stack((
            np.cos(el) * np.cos(az),
            np.cos(el) * np.sin(az),
            np.sin(el)
        ))

    @staticmethod
    def _sh_real(l: int, m: int, theta, phi):
        if m == 0:
            return sph_harm(0, l, phi, theta).real
        if m > 0:
            return np.sqrt(2.0) * sph_harm(m, l, phi, theta).real
        return np.sqrt(2.0) * sph_harm(-m, l, phi, theta).imag

    def _build_loudspeaker_decoder(self, hoa_order):
        n_sh = (hoa_order + 1) ** 2
        n_pos = self.hrir_az_rad.size
        D = np.zeros((n_pos, n_sh), dtype=np.float32)
        col = 0
        theta = np.pi / 2 - self.hrir_el_rad  # colatitude
        phi = self.hrir_az_rad
        for l in range(hoa_order + 1):
            for m in range(-l, l + 1):
                Ylm = self._sh_real(l, m, theta, phi)
                if l != 0:
                    Ylm *= np.sqrt(1.0 + (m == 0))  # SN3D scale
                D[:, col] = Ylm
                col += 1
        return D.astype(np.float32)

    def load_sofa_hrtf(self, sofa_path, hoa_order=6):
        sof = SOFAFile(sofa_path, 'r')
        raw_hrir = sof.getDataIR()
        self.hrir = np.array(raw_hrir.data, dtype=np.float32)
        self.hrir_sr = int(sof.getSamplingRate())
        if self.hrir_sr != self.sample_rate:
            raise ValueError("HRIR sampling rate doesn't match processing rate.")
        pos_deg = sof.getVariableValue('SourcePosition')
        self.hrir_az_rad = np.deg2rad(pos_deg[:, 0])
        self.hrir_el_rad = np.deg2rad(pos_deg[:, 1])
        cart = self._sph_to_cart(self.hrir_az_rad, self.hrir_el_rad)
        self._dir_kdtree = cKDTree(cart)
        self._sh_decode = self._build_loudspeaker_decoder(hoa_order)
        sof.close()

        # build FIR banks (L × Nsh)
        M, _, L = self.hrir.shape
        Nsh = self.n_ambi

        Y = np.zeros((M, Nsh), dtype=np.float32)
        theta = np.pi / 2 - self.hrir_el_rad
        phi = self.hrir_az_rad
        col = 0
        for l in range(self.order + 1):
            for m in range(-l, l + 1):
                if m == 0:
                    Y[:, col] = sph_harm(0, l, phi, theta).real
                elif m > 0:
                    Y[:, col] = np.sqrt(2.0) * sph_harm(m, l, phi, theta).real
                else:
                    Y[:, col] = np.sqrt(2.0) * sph_harm(-m, l, phi, theta).imag
                col += 1

        G = np.linalg.pinv(Y, rcond=1e-3)  # Nsh × M decode
        self.fir_L = (G @ self.hrir[:, 0, :]).T  # (L × Nsh)
        self.fir_R = (G @ self.hrir[:, 1, :]).T

        self.zi_L = [np.zeros(L - 1, dtype=np.float32) for _ in range(Nsh)]
        self.zi_R = [np.zeros(L - 1, dtype=np.float32) for _ in range(Nsh)]

        self.hrtf_loaded = True

    @staticmethod
    def rotate_hoa_block(hoa_block, yaw, pitch):
        """Very simple 1st-order rotation example."""
        if hoa_block.shape[0] < 4:
            return hoa_block
        W, X, Y, Z = hoa_block[:4]
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        Xr = X * cy + Y * sy
        Yr = -X * sy + Y * cy
        Zr = Z * cp - (Xr * sp * cy + Yr * sp * sy)
        hoa_block[:4] = np.stack([W, Xr, Yr, Zr], axis=0)
        return hoa_block

    def apply_binaural_processing_real(self, hoa_block_real_sn3d: np.ndarray, head_orientation=(0., 0.)):
        if not self.hrtf_loaded:
            raise RuntimeError("HRTF not loaded")
        yaw, pitch = head_orientation
        hoa_rot = self.rotate_hoa_block(hoa_block_real_sn3d.copy(), yaw, pitch)
        Nsh, N = hoa_rot.shape
        outL = np.zeros(N, dtype=np.float32)
        outR = np.zeros(N, dtype=np.float32)
        for n in range(Nsh):
            x = hoa_rot[n].astype(np.float32, copy=False)
            bL = self.fir_L[:, n]
            bR = self.fir_R[:, n]
            yL, self.zi_L[n] = lfilter(bL, 1, x, zi=self.zi_L[n])
            yR, self.zi_R[n] = lfilter(bR, 1, x, zi=self.zi_R[n])
            outL += yL
            outR += yR
        return np.vstack([outL, outR]).astype(np.float32)


# =========================
# Visualization
# =========================
class VisualizationEngine:
    """Pixel-grid visualization of a spherical scan."""

    def __init__(self, width=1080, height=1080, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.grid_size = 0
        self.grid_indices = {}
        self.colormap = LinearSegmentedColormap.from_list(
            'pressure', ['blue', 'cyan', 'green', 'yellow', 'orange', 'red'], N=256
        )

    def generate_beam_directions(self, grid_size=32):
        self.grid_size = grid_size
        elevations = np.linspace(-np.pi / 2 + 0.01, np.pi / 2 - 0.01, grid_size)
        azimuths = np.linspace(-np.pi, np.pi, grid_size, endpoint=False)

        beam_dirs = []
        self.grid_indices = {}
        for i, elev in enumerate(elevations):
            for j, azim in enumerate(azimuths):
                theta = np.pi / 2 - elev
                phi = azim if azim >= 0 else azim + 2 * np.pi
                beam_dirs.append((theta, phi))
                self.grid_indices[len(beam_dirs) - 1] = (i, j)
        return beam_dirs

    def create_pixelated_grid_visualization(self, beam_powers):
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        if self.grid_size == 0:
            return img
        max_power = np.max(beam_powers)
        if max_power <= 0:
            return img
        beam_powers_norm = beam_powers / max_power
        cell_w = self.width / self.grid_size
        cell_h = self.height / self.grid_size
        for beam_idx, (gi, gj) in self.grid_indices.items():
            p = float(beam_powers_norm[beam_idx])
            color = self.colormap(p)[:3]
            color_bgr = (np.array(color) * 255)[::-1].astype(np.uint8).tolist()
            x1 = int(gj * cell_w)
            y1 = int(gi * cell_h)
            x2 = int((gj + 1) * cell_w)
            y2 = int((gi + 1) * cell_h)
            cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, -1)
        return img

    def plot_interactive_sphere(self, ambi_proc: AmbisonicProcessor, beam_weights, beam_dir_idx, filename):
        n = 64
        theta = np.linspace(0, np.pi, n)
        phi = np.linspace(0, 2 * np.pi, n)
        T, P = np.meshgrid(theta, phi)
        Wv = beam_weights[beam_dir_idx]  # (C,)

        # build response G(T,P) = |sum w* Y_n3d(T,P)|
        G = np.zeros_like(T, dtype=np.complex128)
        for l in range(ambi_proc.order + 1):
            for m in range(-l, l + 1):
                idx = ambi_proc.sh_coeffs[(l, m)]
                Ylm = ambi_proc.spherical_harmonics(l, m, T, P, norm='n3d')
                G += Wv[idx] * Ylm
        G = np.abs(G)

        X = np.sin(T) * np.cos(P)
        Y = np.sin(T) * np.sin(P)
        Z = np.cos(T)

        fig = go.Figure(data=[
            go.Surface(
                x=X, y=Y, z=Z,
                surfacecolor=G,
                colorscale='Jet',
                cmin=G.min(), cmax=G.max(),
                showscale=True
            )
        ])
        fig.update_layout(
            scene=dict(
                xaxis_title='X (front)',
                yaxis_title='Y (left)',
                zaxis_title='Z (up)',
                aspectmode='data'
            )
        )
        fig.write_html(filename)


# =========================
# Audio + Video main processor
# =========================
def _build_A_conjY(order: int, beam_dirs):
    """Build A(K,C) where rows a_k = conj(Y_n3d(theta_k,phi_k))."""
    C = (order + 1) ** 2
    K = len(beam_dirs)
    A = np.zeros((K, C), dtype=np.complex128)
    idxs = []
    col = 0
    # Precompute for speed
    for k, (theta, phi) in enumerate(beam_dirs):
        col = 0
        for l in range(order + 1):
            for m in range(-l, l + 1):
                Ylm = sph_harm(m, l, phi, theta) * np.sqrt(2 * l + 1.0)  # N3D
                A[k, col] = np.conj(Ylm)
                col += 1
    return A


class AudioVideoProcessor:
    """Main processor combining all components"""

    def __init__(self,
                 ambisonic_file,
                 raw_file,
                 geom_file,
                 input_norm='SN3D',
                 map_mode='DAS',              # 'DAS' | 'MAXRE' | 'MVDR'
                 das_band=(300.0, 8000.0),
                 hoa_band=None,               # e.g. (300.,8000.) if you want HOA-banded maps
                 mvdr_alpha=1e-3,
                 grid_size=220,
                 ):
        self.ambisonic_file = ambisonic_file
        self.raw_file = raw_file
        self.geom_file = geom_file
        self.input_norm = input_norm.upper()
        self.map_mode = map_mode.upper()
        self.das_band = das_band
        self.hoa_band = hoa_band
        self.mvdr_alpha = mvdr_alpha

        self.ambisonic_proc = AmbisonicProcessor(order=6, sample_rate=48000)
        self.hrtf_proc = HRTFProcessor(sample_rate=48000, hoa_order=6)
        self.hrtf_proc.load_sofa_hrtf("/home/agjaci-iit.local/em64_analysis/hrtf_data/HRIRs/AKO536081622_1_processed.sofa")
        self.viz_engine = VisualizationEngine(fps=30)

        self.sample_rate = 48000
        self.video_fps = 30
        self.hop_size = self.sample_rate // self.video_fps  # 1600 samples for 30 fps

        # Load data
        self.load_data()

        # Precompute scan grid and steering
        self.grid_size = grid_size
        self.beam_dirs = self.viz_engine.generate_beam_directions(grid_size=self.grid_size)

        # HOA fixed beams (for MAXRE)
        self.beam_weights_maxre = self.ambisonic_proc.generate_beam_patterns(
            self.beam_dirs, beam_type='max_re_complex'
        )

        # MVDR steering (pure conj(Y), unweighted)
        self.A_conjY = _build_A_conjY(self.ambisonic_proc.order, self.beam_dirs)  # (K,C)

        # For RAW-DAS: precompute delays and weights
        self.dirs_cart = build_dirs_from_thetaphi(self.beam_dirs)
        self.geom_data = pd.read_csv(self.geom_file)
        self.mic_positions = self.geom_data[['mic X (m)', 'mic Y (m)', 'mic Z (m)']].values.astype(np.float32)
        if 'Quad. Weight' in self.geom_data.columns:
            self.wq = self.geom_data['Quad. Weight'].values.astype(np.float32)
        else:
            self.wq = np.ones(self.mic_positions.shape[0], dtype=np.float32)
        self.tau_km = steering_delays(self.mic_positions, self.dirs_cart).astype(np.float32)

        # Diagnostics for one beam
        resp, bw_deg, di_approx = self.ambisonic_proc.compute_directivity_pattern(self.beam_weights_maxre[0])
        print(f"max-rE beam: -3 dB width ≈ {bw_deg:.1f}°,  2D DI≈{di_approx:.1f}")
        print(f"Map mode: {self.map_mode}")

    def load_data(self):
        print("Loading ambisonic recording...")
        hoa, self.sample_rate = sf.read(self.ambisonic_file)
        if hoa.ndim == 1:
            hoa = hoa[:, None]
        hoa = hoa.T  # (C, N)

        C_expected = (self.ambisonic_proc.order + 1) ** 2
        if hoa.shape[0] != C_expected:
            raise ValueError(f"HOA channels {hoa.shape[0]} != {C_expected} for order={self.ambisonic_proc.order}")

        # Keep a real SN3D copy for HRTF decoding
        if self.input_norm == 'N3D':
            hoa_sn3d = hoa.copy()
            for l, rng in enumerate(acn_order_index_ranges(self.ambisonic_proc.order)):
                hoa_sn3d[rng, :] /= np.sqrt(2 * l + 1.0)
        else:
            hoa_sn3d = hoa.copy()
        self.hoa_real_sn3d = hoa_sn3d.astype(np.float32, copy=False)

        # Build complex ACN/N3D for HOA-domain beamforming
        if self.input_norm == 'SN3D':
            hoa_cplx_n3d = self.ambisonic_proc.convert_acn_sn3d_real_to_complex_n3d(hoa)
        else:
            hoa_cplx_n3d = acn_real_to_complex_n3d_block(hoa, self.ambisonic_proc.order)
        self.hoa_cplx_n3d = hoa_cplx_n3d

        print("Loading raw recording...")
        raw, fs_raw = sf.read(self.raw_file, always_2d=True)
        if fs_raw != self.sample_rate:
            raise ValueError("Raw and HOA sample rates differ.")
        self.raw_data = raw.astype(np.float32, copy=False)  # shape (N, M) time-major

    # ---- Map calculators ----
    def _das_power_for_block(self, raw_block_TxM: np.ndarray) -> np.ndarray:
        """
        Compute a DAS map for a block, using frequency-domain steering,
        quadrature weights and PHAT whitening, averaged across STFT columns.
        """
        window = get_window("hann", 1024, fftbins=True)
        f, t, Z = stft(raw_block_TxM.T, fs=self.sample_rate,
                       window=window, nperseg=1024, noverlap=512, nfft=1024,
                       axis=-1, boundary=None, padded=False, return_onesided=True)
        X_ftm = np.transpose(Z, (1, 2, 0))  # (F, T, M)
        E_k = das_broadband_power_singleframe(
            X_ftm, f, self.tau_km, self.wq,
            f_lo=self.das_band[0] if self.das_band else 0.0,
            f_hi=self.das_band[1] if self.das_band else float(self.sample_rate/2),
            phat=True, n_bands=16, chunk_K=4096
        )
        return E_k / (np.max(E_k) if np.max(E_k) > 0 else 1.0)

    def _maxre_power_for_block(self, hoa_block_cplx_n3d: np.ndarray) -> np.ndarray:
        """
        HOA fixed-beam map via quadratic form p_k = w_k^H R w_k.
        Optional band-limiting applied before forming R.
        """
        X = hoa_block_cplx_n3d
        if self.hoa_band is not None:
            X = bandlimit_block_iir(X, self.sample_rate, self.hoa_band)

        R = (X @ X.conj().T) / X.shape[1]  # (C,C)
        # p = diag(W R W^H)
        # Vectorized: p_k = w_k^H R w_k
        W = self.beam_weights_maxre  # (K,C)
        p = np.einsum('kc,cd,kd->k', W.conj(), R, W, optimize=True).real
        p = np.maximum(p, 0.0)
        return p / (np.max(p) if np.max(p) > 0 else 1.0)

    def _mvdr_power_for_block(self, hoa_block_cplx_n3d: np.ndarray) -> np.ndarray:
        """
        Frequency-domain MVDR pseudo-spectrum averaged over a band:
            P_k = mean_f 1 / ( a_k^H R_f^{-1} a_k )
        where a_k = conj(Y_n3d(theta_k,phi_k)). Uses chunking over directions.
        """
        # STFT of HOA channels (C,N) → (F,T,C)
        window = get_window("hann", 1024, fftbins=True)
        f, t, Z = stft(hoa_block_cplx_n3d, fs=self.sample_rate,
                       window=window, nperseg=1024, noverlap=512, nfft=1024,
                       axis=-1, boundary=None, padded=False, return_onesided=True)
        # Z: (C, F, T) → (F, T, C)
        X_ftc = np.transpose(Z, (1, 2, 0))
        F, T, C = X_ftc.shape

        # Band select
        f_lo = self.das_band[0] if self.das_band else 0.0
        f_hi = self.das_band[1] if self.das_band else float(self.sample_rate/2)
        sel = np.where((f >= f_lo) & (f <= f_hi))[0]
        if sel.size == 0:
            sel = np.arange(F)

        K = len(self.beam_dirs)
        A = self.A_conjY  # (K,C)   rows are a_k

        P_acc = np.zeros(K, dtype=np.float64)
        chunk_K = 4096

        for b in sel:
            X_tc = X_ftc[b, :, :]  # (T,C)
            # Spatial covariance at bin b
            Rf = (X_tc.conj().T @ X_tc) / T  # (C,C)
            # Diagonal loading
            Rf += self.mvdr_alpha * np.trace(Rf).real * np.eye(C) / C
            try:
                Rinv = np.linalg.inv(Rf)
            except np.linalg.LinAlgError:
                Rinv = np.linalg.pinv(Rf)
            # chunk over directions
            denom = np.zeros(K, dtype=np.float64)
            for k0 in range(0, K, chunk_K):
                k1 = min(K, k0 + chunk_K)
                A_chunk = A[k0:k1, :]           # (Kc,C)
                B = (Rinv @ A_chunk.T)          # (C,Kc)
                # a^H R^-1 a = sum_c conj(a_c) * (Rinv a)_c  → since A rows are a, conj(A) used on left
                # but A already = conj(Y). So conj(A) = Y.
                denom_chunk = np.sum(np.conj(A_chunk) * B.T, axis=1).real
                denom[k0:k1] = np.clip(denom_chunk, 1e-12, None)
            P_acc += 1.0 / denom

        P_acc /= float(sel.size)
        return P_acc / (np.max(P_acc) if np.max(P_acc) > 0 else 1.0)

    # ---- Frame processing ----
    def process_frame(self, frame_idx):
        start = frame_idx * self.hop_size
        end = start + self.hop_size
        if end > self.hoa_cplx_n3d.shape[1] or end > self.raw_data.shape[0]:
            return None, None

        # HOA blocks
        hoa_block_cplx_n3d = self.hoa_cplx_n3d[:, start:end]
        hoa_block_real_sn3d = self.hoa_real_sn3d[:, start:end]

        # RAW block
        raw_block = self.raw_data[start:end, :]  # (N, M)

        # Choose map algorithm
        if self.map_mode == 'DAS':
            beam_powers = self._das_power_for_block(raw_block)
        elif self.map_mode == 'MAXRE':
            beam_powers = self._maxre_power_for_block(hoa_block_cplx_n3d)
        elif self.map_mode == 'MVDR':
            beam_powers = self._mvdr_power_for_block(hoa_block_cplx_n3d)
        else:
            raise ValueError("map_mode must be 'DAS', 'MAXRE', or 'MVDR'.")

        # Binaural audio from real SN3D HOA (independent of map)
        binaural_audio = self.hrtf_proc.apply_binaural_processing_real(hoa_block_real_sn3d, (0.0, 0.0))

        viz_frame = self.viz_engine.create_pixelated_grid_visualization(beam_powers)
        return viz_frame, binaural_audio

    def run(self, output_file=None):
        if output_file is None:
            output_file = f"eigenmike_acoustic_video_{self.map_mode.lower()}_71025.mp4"

        print("Starting acoustic video processing...")
        temp_video_file = "temp_video_only.mp4"
        temp_audio_file = "temp_audio.wav"

        total_samples = min(self.hoa_cplx_n3d.shape[1], self.raw_data.shape[0])
        total_frames = total_samples // self.hop_size
        print(f"Processing {total_frames} frames...")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(temp_video_file, fourcc, self.viz_engine.fps,
                                       (self.viz_engine.width, self.viz_engine.height))

        all_audio_chunks = []
        try:
            for frame_idx in range(total_frames):
                if frame_idx % 30 == 0:
                    print(f"Frame {frame_idx}/{total_frames} "
                          f"({100.0 * frame_idx / max(1, total_frames):.1f}%)")

                viz_frame, binaural_audio = self.process_frame(frame_idx)
                if viz_frame is None:
                    break
                video_writer.write(viz_frame)
                if binaural_audio is not None:
                    all_audio_chunks.append(binaural_audio)
        finally:
            video_writer.release()
            print("Video frame generation complete.")

        if not all_audio_chunks:
            print("No audio was generated. The output video will be silent.")
            if os.path.exists(temp_video_file):
                os.rename(temp_video_file, output_file)
            return

        print("Concatenating full audio track...")
        full_audio = np.concatenate(all_audio_chunks, axis=1).real
        max_val = np.max(np.abs(full_audio))
        if max_val > 0:
            full_audio *= (0.98 / max_val)
        sf.write(temp_audio_file, full_audio.T, self.sample_rate)
        print(f"Audio track saved to {temp_audio_file}")

        print("Muxing video and audio into final file...")
        try:
            video_clip = VideoFileClip(temp_video_file)
            audio_clip = AudioFileClip(temp_audio_file)
            final_clip = video_clip.set_audio(audio_clip)
            final_clip.write_videofile(output_file, codec='libx264', audio_codec='aac',
                                       temp_audiofile='temp-audio.m4a', remove_temp=True,
                                       logger='bar')
            print(f"Final video saved as {output_file}")
        except Exception as e:
            print(f"Error combining video and audio with moviepy: {e}")
            print(f"Temporary files are available: '{temp_video_file}' and '{temp_audio_file}'")
        finally:
            if os.path.exists(temp_video_file):
                os.remove(temp_video_file)
            if os.path.exists(temp_audio_file):
                os.remove(temp_audio_file)


# =========================
# Utility: 3D surface plot of a power map
# =========================
def plot_directivity_on_sphere(beam_dirs, beam_powers):
    N = int(np.sqrt(len(beam_dirs)))
    thetas = np.linspace(0, np.pi, N)
    phis = np.linspace(0, 2 * np.pi, N)
    Θ, Φ = np.meshgrid(thetas, phis, indexing='ij')

    P = np.zeros_like(Θ)
    for (θ, φ), p in zip(beam_dirs, beam_powers):
        i = np.argmin(np.abs(thetas - θ))
        j = np.argmin(np.abs(phis - φ))
        P[i, j] = p

    X = np.sin(Θ) * np.cos(Φ)
    Y = np.sin(Θ) * np.sin(Φ)
    Z = np.cos(Θ)

    fig = go.Figure(go.Surface(
        x=X, y=Y, z=Z,
        surfacecolor=P,
        colorscale='Jet',
        cmin=0, cmax=1,
    ))
    fig.update_layout(
        scene=dict(
            aspectmode='data',
            xaxis_title='X (front)',
            yaxis_title='Y (left)',
            zaxis_title='Z (up)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0))
        )
    )
    fig.show()


# =========================
# Entrypoint
# =========================
def main():
    #ambisonic_file = "/media/agjaci/Extreme SSD/em64_rec/ES3_20250522_101457_hoa-4.wav"
    #raw_file = "/media/agjaci/Extreme SSD/em64_rec/ES3_20250522_101457_raw-4.wav"
    #geom_file = "/media/agjaci/Extreme SSD/em64_rec/em64_geom.csv"
    ambisonic_file = "/media/agjaci/Extreme SSD/anechoic_recordings/em64/ES3_20250807_170126_hoa-7_aligned.wav"
    raw_file = "/media/agjaci/Extreme SSD/anechoic_recordings/em64/ES3_20250807_170126_raw-7_aligned.wav"
    geom_file = "/home/agjaci-iit.local/em64_analysis/em64_geom.csv"

    for fp in [ambisonic_file, raw_file, geom_file]:
        if not os.path.exists(fp):
            print(f"Error: File not found: {fp}")
            return

    try:
        # Choose one: 'DAS', 'MAXRE', or 'MVDR'
        processor = AudioVideoProcessor(
            ambisonic_file, raw_file, geom_file,
            input_norm='SN3D',
            map_mode='MAXRE',          # change to 'DAS', 'MAXRE' or 'MVDR'
            das_band=(300., 8000.),
            hoa_band=None,           # e.g. (300., 8000.) to band-limit MAXRE quadratic map
            mvdr_alpha=1e-3,
            grid_size=64
        )
        processor.run()  # filename auto-tagged by map_mode
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()