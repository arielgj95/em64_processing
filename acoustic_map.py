#!/usr/bin/env python3
"""
Eigenmike 64 Acoustic Video System
Processes 4th-order ambisonic recordings into directional beam visualization
with binaural audio output using HRTF.
"""

import numpy as np
import scipy.signal
import scipy.special
import soundfile as sf
import pandas as pd
import cv2
import pygame
import requests
import zipfile
import os
import threading
import queue
from scipy.spatial.transform import Rotation
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import time
from moviepy.editor import VideoFileClip, AudioFileClip
import plotly.graph_objects as go
from scipy.signal import stft, istft
from scipy.special import sph_harm
from scipy.spatial.distance import cdist
from scipy.interpolate import griddata
from scipy.signal import fftconvolve
from pysofaconventions import SOFAFile
from scipy.spatial import cKDTree
from scipy.signal import lfilter

from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial.transform import Rotation as R_scipy

'''
def spherical_harmonics(self, l, m, theta, phi):
    """Compute real spherical harmonics Y_l^m(theta, phi) with proper normalization"""
    # Use SN3D normalization (Standard N3D) for ambisonic compatibility
    if m == 0:
        # Zonal harmonics
        return scipy.special.sph_harm(0, l, phi, theta).real
    elif m > 0:
        # Positive sectorial/tesseral harmonics
        return np.sqrt(2) * scipy.special.sph_harm(m, l, phi, theta).real
    else:
        # Negative sectorial/tesseral harmonics
        return np.sqrt(2) * scipy.special.sph_harm(-m, l, phi, theta).imag

def generate_beam_patterns(self, beam_dirs, beam_type='max_directivity'):
    """Generate ultra-directional beam patterns using full 6th order"""
    num_beams = len(beam_dirs)
    beam_weights = np.zeros((num_beams, self.num_channels))

    for i, (theta, phi) in enumerate(beam_dirs):
        if beam_type == 'max_directivity':
            # Ultra-sharp beams using all orders with optimized weighting
            for l in range(self.order + 1):
                for m in range(-l, l + 1):
                    idx = self.sh_coeffs[(l, m)]

                    # Compute spherical harmonic
                    Y_lm = self.spherical_harmonics(l, m, theta, phi)

                    # Apply sophisticated weighting for maximum directivity
                    if l == 0:
                        # Omnidirectional component
                        weight = Y_lm
                    elif l <= 2:
                        # Strong contribution from lower orders for stability
                        weight = Y_lm * 2.0
                    elif l <= 4:
                        # Medium contribution from mid orders
                        weight = Y_lm * 1.5 * self.max_re_weights[l]
                    else:
                        # High orders for ultra-sharp directivity
                        # Use aggressive weighting for maximum sharpness
                        weight = Y_lm * (l + 1) * 0.8 * self.max_re_weights[l]

                    beam_weights[i, idx] = weight

        elif beam_type == 'super_directional':
            # Even more aggressive directivity using 6th order
            for l in range(self.order + 1):
                for m in range(-l, l + 1):
                    idx = self.sh_coeffs[(l, m)]
                    Y_lm = self.spherical_harmonics(l, m, theta, phi)

                    # Exponential weighting for extreme directivity
                    weight = Y_lm * np.exp(l * 0.3) * self.max_re_weights[l]
                    beam_weights[i, idx] = weight

        elif beam_type == 'adaptive_directional':
            # Adaptive weighting based on order
            for l in range(self.order + 1):
                for m in range(-l, l + 1):
                    idx = self.sh_coeffs[(l, m)]
                    Y_lm = self.spherical_harmonics(l, m, theta, phi)

                    # Adaptive weighting: stronger high-order contribution
                    if l <= 1:
                        weight = Y_lm * 1.0
                    elif l <= 3:
                        weight = Y_lm * 1.8
                    else:
                        # Use full 6th order potential
                        weight = Y_lm * (2.0 + l * 0.5) * self.max_re_weights[l]

                    beam_weights[i, idx] = weight

        # Normalize beam weights for stability
        norm = np.linalg.norm(beam_weights[i, :])
        if norm > 0:
            beam_weights[i, :] /= norm

    return beam_weights
'''

'''
class DOALocalizer:
    """High-resolution DOA using MUSIC in the spherical-harmonic domain"""
    def __init__(self, geometry, order=6, fs=48000, nfft=1024, alpha=1e-3):
        # geometry: DataFrame with 'Theta (degrees)', 'Phi (degrees)'
        self.fs = fs
        self.order = order
        self.nfft = nfft
        self.alpha = alpha
        # build spherical-harmonic encoding matrix
        theta = np.radians(geometry['Theta (degrees)'].values)
        phi   = np.radians(geometry['Phi (degrees)'].values)
        self.Y = np.stack([
            sph_harm(m, l, phi, theta)
            for l in range(order+1)
            for m in range(-l, l+1)
        ], axis=1)

    def estimate_covariance(self, X):
        """
        X: STFT data shape (freq_bins, n_sensors, frames)
        returns: covariance matrices shape (freq_bins, n_sensors, n_sensors)
        """
        F, M, T = X.shape
        R = np.zeros((F, M, M), dtype=complex)
        for f in range(F):
            # spatial covariance at freq f
            R[f] = (X[f] @ X[f].conj().T) / T
            # regularize
            R[f] += self.alpha * np.eye(M)
        return R

    def music_spectrum(self, R, scan_dirs):
        """
        R: covariance (nch, nch)
        scan_dirs: array (D,3) of cartesian direction vectors
        returns: spectrum grid of length D
        """
        # eigen-decompose
        eigvals, eigvecs = np.linalg.eigh(R)
        # assume 1 source → noise subspace = eigvecs[:, :-1]
        En = eigvecs[:, :-1]
        # build steering for each dir
        D = len(scan_dirs)
        P = np.zeros(D)
        # compute complex steering in SH domain
        th = np.arccos(np.clip(scan_dirs[:,2],-1,1))
        ph = np.arctan2(scan_dirs[:,1], scan_dirs[:,0])
        A = np.stack([sph_harm(m, l, ph, th)
                      for l in range(self.order+1)
                      for m in range(-l, l+1)], axis=1)  # (D, M)
        for d in range(D):
            a = A[d][:,None]
            P[d] = 1.0 / np.real((a.conj().T @ (En @ En.conj().T) @ a))
        return P

    def localize(self, frame_data, scan_dirs):
        """Estimate DOA for a block of time-domain signals"""
        # frame_data: (n_sensors, n_samples)
        # STFT
        f, t, X = stft(frame_data, fs=self.fs, nperseg=self.nfft, axis=-1)
        # pick freq range
        valid = (f>30)&(f<20000)
        Xf = X[valid]  # (F, M, T)
        # covariance per freq
        Rf = self.estimate_covariance(Xf)
        # average MUSIC spectrum across freqs
        D = len(scan_dirs)
        P_acc = np.zeros(D)
        for R in Rf:
            P_acc += self.music_spectrum(R, scan_dirs)
        P_acc /= len(Rf)
        # find peak
        idx = np.argmax(P_acc)
        return scan_dirs[idx], P_acc
'''
class AmbisonicProcessor:
    """Handles ambisonic signal processing and beamforming"""

    def __init__(self, order=6, sample_rate=48000):
        self.order = order
        self.sample_rate = sample_rate
        self.num_channels = (order + 1) ** 2  # 49 channels for 6th order

        # Generate spherical harmonics coefficients
        self._generate_sh_coefficients()

        # Precompute regularization matrix for robust inversion
        self._setup_regularization()

    def convert_acn_to_complex_n3d(self, acn_signals):
        """
        Converts a real-valued ACN signal block to a complex-valued N3D signal block.

        acn_signals: (num_channels, num_samples) array of real ACN signals.
        """
        if not np.isrealobj(acn_signals):
            print("Warning: Input signal for conversion is already complex. Skipping.")
            return acn_signals.astype(complex)

        num_channels, num_samples = acn_signals.shape
        complex_signals = np.zeros((num_channels, num_samples), dtype=complex)

        # A mapping from (l, |m|) to the ACN channel indices for cos and sin parts
        # This helps us find the +/- m pairs efficiently.
        m_pairs = {}
        for acn_idx, (l, m) in self.acn_to_lm.items():
            if m >= 0:
                if (l, m) not in m_pairs:
                    m_pairs[(l, m)] = {'cos': None, 'sin': None}
                m_pairs[(l, m)]['cos'] = acn_idx
            else:  # m < 0
                if (l, -m) not in m_pairs:
                    m_pairs[(l, -m)] = {'cos': None, 'sin': None}
                m_pairs[(l, -m)]['sin'] = acn_idx

        for l in range(self.order + 1):
            # Handle m = 0 (zonal harmonics)
            # Y_l^0 is real. ACN_l^0 = Y_l^0. N3D needs normalization.
            # However, your spherical_harmonics function applies this, so we just copy.
            acn_idx_m0 = m_pairs.get((l, 0), {}).get('cos')
            if acn_idx_m0 is not None:
                sh_idx = self.sh_coeffs[(l, 0)]
                complex_signals[sh_idx, :] = acn_signals[acn_idx_m0, :]

            # Handle m > 0 (tesseral/sectoral harmonics)
            for m in range(1, l + 1):
                pair = m_pairs.get((l, m))
                if pair is None or pair['cos'] is None or pair['sin'] is None:
                    continue

                acn_cos_idx = pair['cos']  # Corresponds to R_l^m
                acn_sin_idx = pair['sin']  # Corresponds to R_l^-m

                # Reconstruct the complex SH from the real ones (ACN/SN3D convention)
                # Y_l^m  = (R_l^m + i * R_l^-m) / sqrt(2)
                # Y_l^-m = (-1)^m * (R_l^m - i * R_l^-m) / sqrt(2)

                # Note: We can ignore the 1/sqrt(2) because beamforming is ratio-metric
                # and steering vectors/beam weights will have the same factor.
                # Or, more robustly, let's keep it.

                real_part = acn_signals[acn_cos_idx, :]
                imag_part = acn_signals[acn_sin_idx, :]

                # For +m
                sh_pos_m_idx = self.sh_coeffs[(l, m)]
                complex_signals[sh_pos_m_idx, :] = (real_part + 1j * imag_part) / np.sqrt(2.0)

                # For -m
                sh_neg_m_idx = self.sh_coeffs[(l, -m)]
                complex_signals[sh_neg_m_idx, :] = ((-1) ** m * (real_part - 1j * imag_part)) / np.sqrt(2.0)

        return complex_signals


    def spherical_harmonics_matrix(self, max_order=None):
        """
        For HOA recordings in ACN ordering, the ambisonic channels *are* the
        SH coefficients, so this is just an identity (Nsh×Nsh).
        """
        order = self.order if max_order is None else max_order
        Nsh   = (order + 1)**2
        return np.eye(Nsh, dtype=complex)

    def _generate_sh_coefficients(self):
        """Generate spherical harmonics coefficients for ambisonic decoding"""
        self.sh_coeffs = {}
        self.acn_to_lm = {}  # ACN (Ambisonic Channel Number) to (l,m) mapping

        idx = 0
        for l in range(self.order + 1):
            for m in range(-l, l + 1):
                self.sh_coeffs[(l, m)] = idx
                self.acn_to_lm[idx] = (l, m)
                idx += 1

    @staticmethod
    def _sh_complex(l: int, m: int, theta, phi):
        """Complex N3D‑normalised spherical harmonics (scipy convention)."""
        return sph_harm(m, l, phi, theta)  # already complex N3D

    @staticmethod
    def _sh_real(l: int, m: int, theta, phi):
        """Real ACN/SN3D basis (cos/sin split)."""
        if m == 0:
            return sph_harm(0, l, phi, theta).real
        if m > 0:
            return np.sqrt(2.0) * sph_harm(m, l, phi, theta).real
        return np.sqrt(2.0) * sph_harm(-m, l, phi, theta).imag

    def _setup_regularization(self):
        """Setup regularization for high-order ambisonic processing"""
        # Max-rE weights for regularization (prevents spatial aliasing)
        self.max_re_weights = np.zeros(self.order + 1)
        for l in range(self.order + 1):
            # Correct max-rE weighting formula
            if l == 0:
                self.max_re_weights[l] = 1.0
            else:
                # Use Legendre polynomial at cos(0) = 1, not 0
                P_l = scipy.special.legendre(l)
                self.max_re_weights[l] = P_l(1.0)  # Fixed: was P_l(0)

        # In-phase weights for better directivity control
        self.in_phase_weights = np.zeros(self.order + 1)
        for l in range(self.order + 1):
            self.in_phase_weights[l] = 1.0 / (l + 1) if l <= 3 else 0.5 / (l + 1)

    def spherical_harmonics(self, l, m, theta, phi, norm='n3d'):
        """
        Spherical harmonics Y_l^m(θ,φ) with proper normalization.
        For ambisonic beamforming, we need to match the encoding format.

        CRITICAL: If your ambisonic recording uses real SH (ACN), use 'real_acn'.
        If it uses complex SH, use 'complex' (but this is less common).
        """
        if norm == 'real_acn':
            # Real spherical harmonics in ACN ordering (standard for ambisonics)
            if m == 0:
                # Zonal harmonics (m=0) - no azimuthal dependence
                return sph_harm(0, l, phi, theta).real
            elif m > 0:
                # Positive m: cosine component (symmetric about front-back)
                return np.sqrt(2.0) * sph_harm(m, l, phi, theta).real
            else:
                # Negative m: sine component (antisymmetric about front-back)
                return np.sqrt(2.0) * sph_harm(-m, l, phi, theta).imag
        else:
            # Complex spherical harmonics (preserves phase information)
            Y = sph_harm(m, l, phi, theta)
            if norm == 'n3d':
                Y *= np.sqrt(2 * l + 1)
            return Y

    def compute_mvdr_beamformer(self, ambisonic_signals, beam_dirs, alpha=1e-3,
                                encoding_format='complex'):
        """
        Compute MVDR beamformer with proper symmetry handling.

        For single source localization, use encoding_format='complex' to break
        front-back symmetry that occurs with real spherical harmonics.

        ambisonic_signals: (num_channels, num_samples)
        beam_dirs: List of (theta, phi) directions
        alpha: Regularization parameter
        encoding_format: 'complex' or 'real_acn'

        Returns: beam_powers (num_beams,)
        """
        ambisonic_signals = ambisonic_signals.astype(complex)
        # Compute sample covariance matrix
        ######R = ambisonic_signals @ ambisonic_signals.T
        R = ambisonic_signals @ ambisonic_signals.conj().T
        R = R / ambisonic_signals.shape[1]

        # Add diagonal loading for regularization
        R += alpha * np.trace(R) * np.eye(R.shape[0]) / R.shape[0]

        # Compute matrix inverse
        try:
            R_inv = np.linalg.inv(R)
        except np.linalg.LinAlgError:
            R_inv = np.linalg.pinv(R)

        # Compute steering vectors for all directions
        num_dirs = len(beam_dirs)
        steering_vectors = np.zeros((self.num_channels, num_dirs), dtype=complex)

        for i, (theta, phi) in enumerate(beam_dirs):
            for l in range(self.order + 1):
                for m in range(-l, l + 1):
                    idx = self.sh_coeffs[(l, m)]

                    if encoding_format == 'complex':
                        # Complex SH - breaks front-back symmetry
                        Ylm = self.spherical_harmonics(l, m, theta, phi, norm='n3d')
                        # Use conjugate for proper beamforming
                        steering_vectors[idx, i] = Ylm #######np.conj(Ylm)
                    else:
                        # Real ACN SH - standard ambisonics (has symmetry issues)
                        Ylm = self.spherical_harmonics(l, m, theta, phi, norm='real_acn')
                        steering_vectors[idx, i] = Ylm

        # MVDR power computation
        beam_powers = np.zeros(num_dirs)

        for i in range(num_dirs):
            a = steering_vectors[:, i]
            ###### denominator = np.real(a.conj().T @ R_inv @ a)
            denominator = np.real(np.dot(a.conj().T, np.dot(R_inv, a)))
            beam_powers[i] = 1.0 / max(denominator, 1e-12)

        # Normalize to [0, 1]
        if np.max(beam_powers) > 0:
            beam_powers = beam_powers / np.max(beam_powers)

        return beam_powers

    def generate_beam_patterns(self, beam_dirs, beam_type='max_directivity_complex'):
        """
        Generates complex beamforming weights for various predefined patterns.

        All patterns are designed for complex spherical harmonics to ensure proper
        source isolation and prevent front-back symmetry artifacts.

        beam_type:
          - 'max_directivity_complex': 6th-order beam with maximum theoretical
                                       directivity. Ideal for source isolation.
          - 'max_re_complex'         : 6th-order max-rE pattern. Excellent directivity
                                       with controlled side-lobes.
          - 'hypercardioid_complex'  : A focused 1st-order hypercardioid, enhanced
                                       with higher orders to sharpen the beam.
        """
        num_beams = len(beam_dirs)
        beam_weights = np.zeros((num_beams, self.num_channels), dtype=complex)

        # --- This is the core principle for all fixed beamformers ---
        # The beamforming weights 'w' for a direction should be the
        # complex conjugate of the steering vector 'a' for that direction,
        # optionally multiplied by modal weights d_l (e.g., for max_rE).
        # w_lm(theta, phi) = d_l * conj(Ylm(theta, phi))

        for i, (theta, phi) in enumerate(beam_dirs):
            # Generate the full complex steering vector for the current direction once
            steering_vector = np.zeros(self.num_channels, dtype=complex)
            for l in range(self.order + 1):
                for m in range(-l, l + 1):
                    idx = self.sh_coeffs[(l, m)]
                    # Use 'n3d' norm for consistency with beamforming literature
                    Ylm = self.spherical_harmonics(l, m, theta, phi, norm='n3d')
                    # The steering vector is the conjugate of the SH functions
                    steering_vector[idx] = Ylm  #np.conj(Ylm)  # <--- CRITICAL POINT  ###########################

            # --- Apply modal weights based on the desired beam type ---
            modal_weights = np.ones(self.order + 1)  # Default for max_directivity

            if beam_type == 'max_directivity_complex':
                # This pattern provides the highest possible directivity for a given
                # order. It's equivalent to decomposing a plane wave from the target direction.
                # All modal weights d_l are 1.0.
                pass  # Use default modal_weights = 1.0

            elif beam_type == 'max_re_complex':
                # Pure max-rE weighting
                modal_weights = self.max_re_weights

            elif beam_type == 'hypercardioid_complex':
                # 1st-order hypercardioid coefficients, with higher orders for sharpening
                # w = d_l * conj(Ylm). We define d_l here.
                d = np.zeros(self.order + 1)
                d[0] = 0.25  # l=0 weight
                d[1] = 0.75  # l=1 weight
                # Add light higher-order tapering to reduce side-lobes
                d[2:] = 0.2 * self.max_re_weights[2:]
                modal_weights = d

            else:
                raise ValueError(f"Unknown beam_type {beam_type!r}")

            # Apply the modal weights to the steering vector
            for l in range(self.order + 1):
                gain = modal_weights[l]
                for m in range(-l, l + 1):
                    idx = self.sh_coeffs[(l, m)]
                    beam_weights[i, idx] = np.conj(steering_vector[idx]) * gain

            # Normalize the final weight vector for this beam
            norm = np.linalg.norm(beam_weights[i])
            if norm > 0:
                beam_weights[i] /= norm

        return beam_weights

    def beamform(self, ambisonic_signals, beam_weights):
        """
        Apply complex beamforming weights to ambisonic signals.

        ambisonic_signals: (num_channels, num_samples), complex-valued
        beam_weights: (num_beams, num_channels), complex-valued
        """
        # The beamforming operation is y = w^H * x, where w is the weight vector
        # and x is the signal vector. In NumPy, for a matrix of weights, this
        # is equivalent to conjugating the weights matrix before multiplication.
        # Note: The weights are already conjugated in generate_beam_patterns,
        # so we perform a direct dot product here.
        beam_signals = beam_weights @ ambisonic_signals
        #########beam_signals = beam_weights.conj() @ ambisonic_signals
        #beam_signals = np.conj(beam_weights) @ ambisonic_signals


        # Calculate the power of each beamformed signal (mean square of amplitude)
        # This is more standard than RMS for power calculations.
        beam_powers = np.mean(np.abs(beam_signals) ** 2, axis=1)

        return beam_signals, beam_powers

    def compute_directivity_pattern(self, beam_weights, resolution=360):
        """
        Computes and analyzes the directivity pattern of a single beamformer.

        Args:
            beam_weights (np.ndarray): A single vector of complex beamforming
                                       weights of shape (num_channels,).
            resolution (int): The number of points to sample the pattern over 180 degrees.

        Returns:
            tuple: A tuple containing:
                - responses (np.ndarray): The magnitude of the beam response at each angle.
                - beamwidth_degrees (float): The half-power (-3dB) beamwidth in degrees.
                - approx_directivity_index (float): A simplified 2D estimation of directivity.
        """
        # Ensure beam_weights is a 1D array for analyzing a single beam
        if beam_weights.ndim != 1 or beam_weights.shape[0] != self.num_channels:
            raise ValueError(f"beam_weights must be a 1D array of shape ({self.num_channels},)")

        # Generate test directions along a great circle (e.g., from front to back)
        test_angles = np.linspace(0, np.pi, resolution)  # 0 to 180 degrees
        # Assume the beam is pointing towards theta=0 for this test
        test_dirs = [(angle, 0) for angle in test_angles]

        responses = np.zeros(resolution, dtype=complex)
        for i, (theta, phi) in enumerate(test_dirs):
            # The beampattern B is the inner product of w^H and the steering vector a.
            # B(theta, phi) = sum(conj(w_lm) * Y_lm(theta, phi))
            steering_vector_a = np.zeros(self.num_channels, dtype=complex)
            for l in range(self.order + 1):
                for m in range(-l, l + 1):
                    idx = self.sh_coeffs[(l, m)]
                    # Using 'n3d' for consistency
                    steering_vector_a[idx] = self.spherical_harmonics(l, m, theta, phi, norm='n3d')

            # <--- CHANGE 1: Correctly apply the conjugate transpose principle
            # Using dot product for a clean inner product calculation
            responses[i] = np.dot(np.conj(beam_weights.conj()), steering_vector_a)

        # We are interested in the magnitude of the response
        responses_magnitude = np.abs(responses)

        # --- Calculate Directivity Metrics ---
        max_response = np.max(responses_magnitude)
        if max_response < 1e-9:  # Handle case of a null beam
            return responses_magnitude, 360.0, 1.0

        # <--- CHANGE 2: Use correct threshold for half-power (-3dB)
        half_power_threshold = max_response / np.sqrt(2)
        above_half_power_indices = np.where(responses_magnitude >= half_power_threshold)[0]

        if len(above_half_power_indices) > 0:
            # Calculate angular width
            first_idx = above_half_power_indices[0]
            last_idx = above_half_power_indices[-1]
            angular_step = 180.0 / (resolution - 1)
            beamwidth_degrees = (last_idx - first_idx) * angular_step
        else:
            # If no points are above half power, the beam is extremely narrow
            beamwidth_degrees = 0.0

        # <--- CHANGE 3: Clarify that this is an approximation
        # This is a simplified 2D estimation, not the true directivity factor (Q)
        #####beamwidth_radians = np.deg2rad(beamwidth_degrees)
        # Clarify that this is an approximation
        # This is a simplified 2D estimation, not the true 3D directivity factor (Q)
        # The directivity index Q for a rotationally symmetric beam is 1 / (integral of |B(theta, phi)|^2 sin(theta) dtheta dphi)
        # For a 2D pattern, a common approximation related to beamwidth:
        # Directivity ~ 2 / (angular width in radians) for narrow beams, or more generally:
        # Q = 2 / integral_0_pi |B(theta)|^2 sin(theta) dtheta (for symmetric beams)
        # A simpler approximation for a very narrow beam: Q ~ 4 * pi / (solid angle of beam)
        # For a 2D pattern, Q_2D = 2 / (1 - cos(theta_hp/2)) where theta_hp is half-power beamwidth in radians.
        beamwidth_radians = np.deg2rad(beamwidth_degrees)
        if beamwidth_radians > 0:
            approx_directivity_index = 2 / (1 - np.cos(beamwidth_radians / 2))
        else:
            approx_directivity_index = np.inf # For a perfectly sharp beam
        ####approx_directivity_index = 2 / (1 - np.cos(beamwidth_radians / 2)) if beamwidth_radians > 0 else np.inf
        # (Using a slightly better 2D-to-3D approximation formula)

        return responses_magnitude, beamwidth_degrees, approx_directivity_index


class HRTFProcessor:
    """Handles HRTF download and binaural processing"""

    def __init__(self, sample_rate=48000):
        self.sample_rate = sample_rate
        self.hrtf_data = None
        self.hrtf_loaded = False
        max_order = 6
        self.order = max_order
        self.n_ambi        = (max_order + 1) ** 2
        self.hrir = None  # (Npos, 2, L)
        self.hrir_sr = None
        self.hrir_az_rad = None  # (Npos,)
        self.hrir_el_rad = None
        self._sh_decode = None  # (Npos, Nsh) loud-speaker decoder
        self._dir_kdtree = None  # fast nearest-neighbour search
        self.hrtf_loaded = False


    def load_simplified_hrtf(self):
        """Load a simplified HRTF model if download fails"""
        angles = np.linspace(0, 2 * np.pi, 72)
        elevations = np.linspace(-np.pi / 2, np.pi / 2, 37)

        self.hrtf_angles = angles
        self.hrtf_elevations = elevations

        # Simple ITD model
        head_radius = 0.0875  # meters
        sound_speed = 343  # m/s

        self.itd_values = np.zeros((len(elevations), len(angles)))
        self.ild_values = np.zeros((len(elevations), len(angles)))

        for i, elev in enumerate(elevations):
            for j, azim in enumerate(angles):
                self.itd_values[i, j] = (head_radius / sound_speed) * (azim + np.sin(azim))
                self.ild_values[i, j] = 6 * np.sin(azim) * np.cos(elev)

        self.hrtf_loaded = True

    @staticmethod
    def _sph_to_cart(az, el):
        """(az,el) in rad → xyz unit vector."""
        return np.column_stack((
            np.cos(el) * np.cos(az),
            np.cos(el) * np.sin(az),
            np.sin(el)
        ))

    def _build_loudspeaker_decoder(self, hoa_order):
        """Evaluate SN3D/ACN real SH at every HRIR direction once."""
        n_sh = (hoa_order + 1) ** 2
        n_pos = self.hrir_az_rad.size
        D = np.zeros((n_pos, n_sh), dtype=np.float32)

        idx = 0
        for l in range(hoa_order + 1):
            for m in range(-l, l + 1):
                # theta = colatitude = π/2 - elevation
                Ylm = AmbisonicProcessor._sh_real(
                    l, m,
                    theta=np.pi / 2 - self.hrir_el_rad,
                    phi=self.hrir_az_rad
                )
                if l != 0:  # SN3D normalisation
                    Ylm *= np.sqrt(1.0 + (m == 0))
                D[:, idx] = Ylm
                idx += 1
        return D.astype(np.float32)

    def load_sofa_hrtf(self, sofa_path, hoa_order=6):
        sof = SOFAFile(sofa_path, 'r')

        # Read the HRIR *data* and strip off any mask
        raw_hrir = sof.getDataIR()
        self.hrir = np.array(raw_hrir.data, dtype=np.float32)  # now a plain ndarray
        self.hrir_sr = int(sof.getSamplingRate())
        if self.hrir_sr != self.sample_rate:
            raise ValueError("...")

        # positions, decoder, etc.
        pos_deg = sof.getVariableValue('SourcePosition')
        self.hrir_az_rad = np.deg2rad(pos_deg[:, 0])
        self.hrir_el_rad = np.deg2rad(pos_deg[:, 1])
        cart = self._sph_to_cart(self.hrir_az_rad, self.hrir_el_rad)
        self._dir_kdtree = cKDTree(cart)
        self._sh_decode = self._build_loudspeaker_decoder(hoa_order)

        sof.close()
        self.hrtf_loaded = True
        self._build_binaural_decoder()  # now G @ self.hrir works cleanly
        L, Nsh = self.fir_L.shape
        self.zi_L = [np.zeros(L-1, dtype=np.float32) for _ in range(Nsh)]
        self.zi_R = [np.zeros(L-1, dtype=np.float32) for _ in range(Nsh)]

    def _build_binaural_decoder(self):
        """
        Builds two FIR banks of shape (L, Nsh):
            self.fir_L[:, n] ... HRIR for ambisonic channel n → left ear
            self.fir_R[:, n] ...                          »   → right ear
        """
        M, _, L = self.hrir.shape  # HRIR set: M directions, length L
        Nsh = self.n_ambi

        # Y  (M × Nsh): real SN3D/ACN spherical-harmonics at every HRIR direction
        Y = np.zeros((M, Nsh), dtype=np.float32)

        theta = np.pi / 2 - self.hrir_el_rad  # colatitude
        phi = self.hrir_az_rad

        col = 0
        for l in range(self.order + 1):
            for m in range(-l, l + 1):
                if m == 0:
                    Y[:, col] = sph_harm(0, l, phi, theta).real
                elif m > 0:
                    Y[:, col] = np.sqrt(2.0) * sph_harm(m, l, phi, theta).real
                else:  # m < 0
                    Y[:, col] = np.sqrt(2.0) * sph_harm(-m, l, phi, theta).imag
                col += 1

        # Decode matrix  (Nsh × M)
        G = np.linalg.pinv(Y, rcond=1e-3)  # least-squares / energy-preserving

        # Two FIR banks  (L × Nsh)
        self.fir_L = (G @ self.hrir[:, 0, :]).T
        self.fir_R = (G @ self.hrir[:, 1, :]).T

    @staticmethod
    def rotate_hoa_block(hoa_block, yaw, pitch):
        """1st–order example rotation.  For 6-th use your own HOA rotator."""
        if hoa_block.shape[0] < 4:  # W,X,Y,Z only
            return hoa_block
        W, X, Y, Z = hoa_block[:4]
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)

        Xr = X * cy + Y * sy
        Yr = -X * sy + Y * cy
        Zr = Z * cp - (Xr * sp * cy + Yr * sp * sy)
        hoa_block[:4] = np.stack([W, Xr, Yr, Zr], axis=0)
        return hoa_block

    def apply_binaural_processing_real(self, hoa_block, head_orientation=(0.,0.)):
        if not self.hrtf_loaded:
            raise RuntimeError("HRTF not loaded")

        # 1) rotate
        yaw, pitch = head_orientation
        hoa_rot = self.rotate_hoa_block(hoa_block.copy(), yaw, pitch)
        Nsh, N = hoa_rot.shape

        # 2) allocate output
        outL = np.zeros(N, dtype=np.float32)
        outR = np.zeros(N, dtype=np.float32)

        # 3) filter each SH channel in time, carrying state
        for n in range(Nsh):
            # real‐part only
            x = hoa_rot[n].real
            bL = self.fir_L[:, n]
            bR = self.fir_R[:, n]

            yL, self.zi_L[n] = lfilter(bL, 1, x, zi=self.zi_L[n])
            yR, self.zi_R[n] = lfilter(bR, 1, x, zi=self.zi_R[n])

            outL += yL
            outR += yR

        return np.vstack([outL, outR]).astype(np.float32)

    def apply_binaural_processing(self, ambisonic_signals, head_orientation=(0, 0)):
        """Fallback binaural processing method (simplified)"""
        if not self.hrtf_loaded:
            self.load_simplified_hrtf()

        # Fallback to simplified processing
        num_channels = min(ambisonic_signals.shape[0], 16)

        # Extract ambisonic components
        W = ambisonic_signals[0, :] if ambisonic_signals.shape[0] > 0 else np.zeros(ambisonic_signals.shape[1])
        X = ambisonic_signals[1, :] if ambisonic_signals.shape[0] > 1 else np.zeros_like(W)
        Y = ambisonic_signals[2, :] if ambisonic_signals.shape[0] > 2 else np.zeros_like(W)
        Z = ambisonic_signals[3, :] if ambisonic_signals.shape[0] > 3 else np.zeros_like(W)

        # Apply head rotation
        yaw, pitch = head_orientation
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        cos_pitch = np.cos(pitch)
        sin_pitch = np.sin(pitch)

        # Rotate first-order components
        X_rot = X * cos_yaw + Y * sin_yaw
        Y_rot = -X * sin_yaw + Y * cos_yaw
        Z_rot = Z * cos_pitch - (X_rot * sin_pitch * cos_yaw + Y_rot * sin_pitch * sin_yaw)

        # Binaural decode
        left = (W * 0.707 + X_rot * 0.5 + Y_rot * 0.866 + Z_rot * 0.3)
        right = (W * 0.707 + X_rot * 0.5 + Y_rot * (-0.866) + Z_rot * 0.3)

        # Apply filtering for spatial cues
        if len(left) > 100:
            b, a = scipy.signal.butter(2, 0.03, 'high')
            left_hf = scipy.signal.filtfilt(b, a, left)
            right_hf = scipy.signal.filtfilt(b, a, right)
            left = 0.7 * left + 0.3 * left_hf
            right = 0.7 * right + 0.3 * right_hf

        return np.array([left, right])


class VisualizationEngine:
    """Handles video generation and beam visualization"""

    def __init__(self, width=1080, height=1080, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.grid_size = 0
        self.grid_indices = {}

        # Create colormap for pressure visualization
        colors = ['blue', 'cyan', 'green', 'yellow', 'orange', 'red']
        self.colormap = LinearSegmentedColormap.from_list('pressure', colors, N=256)

    def generate_beam_directions(self, grid_size=32):
        """Generate squared grid of beam directions with uniform spacing"""
        self.grid_size = grid_size
        elevations = np.linspace(-np.pi / 2 + 0.01, np.pi / 2 - 0.01, grid_size)  # Avoid poles
        azimuths = np.linspace(-np.pi, np.pi, grid_size, endpoint=False) #np.linspace(0, 2 * np.pi, grid_size, endpoint=False)

        beam_dirs = []
        self.grid_indices = {} # Reset indices

        for i, elev in enumerate(elevations):
            for j, azim in enumerate(azimuths):
                theta = np.pi / 2 - elev # Convert to physics convention
                phi = azim
                phi = phi if phi >= 0 else phi + 2 * np.pi
                beam_dirs.append((theta, phi))
                self.grid_indices[len(beam_dirs) - 1] = (i, j) # Store grid position

        print(f"Generated {len(beam_dirs)} beam directions in {grid_size}x{grid_size} grid")
        return beam_dirs

    def plot_interactive_sphere(self, beam_weights, beam_dir_idx, filename):
        """
        beam_weights: (num_beams, num_channels)  from generate_beam_patterns
        beam_dir_idx:  which of those beams you want to visualize (int)
        filename:      where to save the HTML
        """
        # 1) make a fine sampling of the sphere
        n = 64
        theta = np.linspace(0, np.pi,   n)         # colatitude
        phi   = np.linspace(0, 2*np.pi, n)         # azimuth
        T,P = np.meshgrid(theta, phi)

        # 2) compute gain at each (T,P) from your weights
        W = beam_weights[beam_dir_idx]  # shape=(49,)
        G = np.zeros_like(T)
        for l in range(self.order+1):
            for m in range(-l, l+1):
                idx = self.sh_coeffs[(l,m)]
                Ylm = self.spherical_harmonics(l, m, T, P)
                G += W[idx] * Ylm
        G = np.abs(G)  # magnitude

        # 3) spherical → Cartesian
        X = np.sin(T)*np.cos(P)
        Y = np.sin(T)*np.sin(P)
        Z = np.cos(T)

        # 4) color‐map G onto the sphere
        fig = go.Figure(data=[
            go.Surface(
                x=X, y=Y, z=Z,
                surfacecolor=G,
                colorscale='Jet',
                cmin=G.min(), cmax=G.max(),
                showscale=True
            )
        ])
        # 5) align axes: X=front, Y=left, Z=up
        fig.update_layout(
            scene=dict(
                xaxis_title='X (front)',
                yaxis_title='Y (left)',
                zaxis_title='Z (up)',
                aspectmode='data'
            )
        )
        # save to standalone HTML
        fig.write_html(filename)
        print(f"Interactive sphere saved to {filename}")

    def create_pixelated_grid_visualization(self, beam_powers):
        """Create a pixelated grid visualization where each cell is a beam."""
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)


        if self.grid_size == 0:
            print("Warning: Grid size not set in VisualizationEngine.")
            return img

        # Normalize powers for coloring
        max_power = np.max(beam_powers)
        if max_power <= 0:
            print("ERROR")
            return img

        if max_power > 0:
            beam_powers_norm = beam_powers / max_power
        else:
            beam_powers_norm = beam_powers

        # Calculate the size of each grid cell to fill the screen
        cell_width = self.width / self.grid_size
        cell_height = self.height / self.grid_size

        # Draw each beam as a colored rectangle
        for beam_idx, (grid_i, grid_j) in self.grid_indices.items():
            power = beam_powers_norm[beam_idx]
            #color = self.colormap(np.clip(power, 0, 1))[:3]  # RGB from 0-1
            color = self.colormap(power)[:3]
            color_bgr = (np.array(color) * 255)[::-1].astype(np.uint8).tolist()

            # Calculate pixel coordinates for the rectangle
            # grid_j -> azimuth (x-axis), grid_i -> elevation (y-axis)
            x1 = int(grid_j * cell_width)
            y1 = int(grid_i * cell_height)
            x2 = int((grid_j + 1) * cell_width)
            y2 = int((grid_i + 1) * cell_height)

            # Draw the filled rectangle to create a solid, pixelated look
            cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, -1)

        return img

    def _draw_grid_connections(self, img, grid_points, beam_powers):
        """Draw connections between adjacent grid points"""
        if not hasattr(self, 'grid_indices'):
            return

        # Draw horizontal connections
        for beam_idx, (grid_i, grid_j) in self.grid_indices.items():
            if grid_j < self.grid_size - 1:  # Not the last column
                # Find adjacent horizontal beam
                next_beam_idx = None
                for idx, (gi, gj) in self.grid_indices.items():
                    if gi == grid_i and gj == grid_j + 1:
                        next_beam_idx = idx
                        break

                if next_beam_idx is not None:
                    # Get power-based color for connection
                    avg_power = (beam_powers[beam_idx] + beam_powers[next_beam_idx]) / 2
                    color = self.colormap(np.clip(avg_power, 0, 1))[:3]
                    color_255 = (np.array(color) * 128).astype(np.uint8)  # Dimmer for connections

                    # Draw line
                    cv2.line(img, grid_points[beam_idx], grid_points[next_beam_idx],
                             color_255[::-1].tolist(), 2)

        # Draw vertical connections
        for beam_idx, (grid_i, grid_j) in self.grid_indices.items():
            if grid_i < self.grid_size - 1:  # Not the last row
                # Find adjacent vertical beam
                next_beam_idx = None
                for idx, (gi, gj) in self.grid_indices.items():
                    if gi == grid_i + 1 and gj == grid_j:
                        next_beam_idx = idx
                        break

                if next_beam_idx is not None:
                    # Get power-based color for connection
                    avg_power = (beam_powers[beam_idx] + beam_powers[next_beam_idx]) / 2
                    color = self.colormap(np.clip(avg_power, 0, 1))[:3]
                    color_255 = (np.array(color) * 128).astype(np.uint8)  # Dimmer for connections

                    # Draw line
                    cv2.line(img, grid_points[beam_idx], grid_points[next_beam_idx],
                             color_255[::-1].tolist(), 2)

    def add_orientation_indicators(self, img):
        """Add orientation indicators (X: front, Y: left, Z: up)"""
        height, width = img.shape[:2]

        # Add text labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, 'X (Front)', (50, 50), font, 1, (255, 255, 255), 2)
        cv2.putText(img, 'Y (Left)', (50, 100), font, 1, (255, 255, 255), 2)
        cv2.putText(img, 'Z (Up)', (width // 2 - 50, 50), font, 1, (255, 255, 255), 2)

        # Add coordinate arrows
        center_x, center_y = width // 2, height // 2
        arrow_len = 100

        # X arrow (front) - pointing right
        cv2.arrowedLine(img, (center_x, center_y),
                        (center_x + arrow_len, center_y), (0, 255, 0), 3)
        cv2.putText(img, 'X', (center_x + arrow_len + 10, center_y + 10),
                    font, 0.7, (0, 255, 0), 2)

        # Y arrow (left) - pointing left
        cv2.arrowedLine(img, (center_x, center_y),
                        (center_x - arrow_len, center_y), (255, 255, 0), 3)
        cv2.putText(img, 'Y', (center_x - arrow_len - 30, center_y + 10),
                    font, 0.7, (255, 255, 0), 2)

        # Z arrow (up) - pointing up
        cv2.arrowedLine(img, (center_x, center_y),
                        (center_x, center_y - arrow_len), (0, 0, 255), 3)
        cv2.putText(img, 'Z', (center_x + 10, center_y - arrow_len - 10),
                    font, 0.7, (0, 0, 255), 2)

        return img


class AudioVideoProcessor:
    """Main processor combining all components"""

    def __init__(self, ambisonic_file, raw_file, geom_file):
        self.ambisonic_file = ambisonic_file
        self.raw_file = raw_file
        self.geom_file = geom_file

        # Initialize components
        self.ambisonic_proc = AmbisonicProcessor(order=6, sample_rate=48000)
        self.hrtf_proc = HRTFProcessor(sample_rate=48000)
        self.hrtf_proc.load_sofa_hrtf("./hrtf_data/HRIRs/AKO536081622_1_processed.sofa")
        self.viz_engine = VisualizationEngine(fps=30)

        # Audio/video parameters
        self.sample_rate = 48000
        self.video_fps = 30
        self.hop_size = self.sample_rate // self.video_fps

        # Load data
        self.load_data()

        # --- PERFORMANCE & SETUP IMPROVEMENT ---
        # Pre-calculate static beam directions and weights once
        print("Generating static beam patterns...")
        self.grid_size = 220 #540  # Define grid size here
        self.beam_dirs = self.viz_engine.generate_beam_directions(grid_size=self.grid_size)
        self.beam_weights = self.ambisonic_proc.generate_beam_patterns(
            self.beam_dirs, beam_type='max_re_complex') #'max_directivity' 'max_re'

        # Optional: Analyze directivity for the first beam (for validation)
        responses, degrees, directivity = self.ambisonic_proc.compute_directivity_pattern(self.beam_weights[0])
        print(f"Beam directivity factor: {directivity:.2f}")
        print(f"Theoretical max directivity for order 6: {(2 * 6 + 1):.1f}")

    def compute_acoustic_map_das(self, raw_frame):
        """
        Delay-and-Sum beamforming on the raw EM64 mic signals.

        Args:
            raw_frame: np.ndarray, shape (num_mics, num_samples)
              A block of time samples from each of the 64 mics.

        Returns:
            p_map: np.ndarray, shape (num_beams,)
              Normalized power map (max=1) for each direction in self.beam_dirs.
        """
        c = 343.0                                 # speed of sound (m/s)
        fs = self.sample_rate                     # sampling rate (Hz)
        positions = self.geom_data[['mic X (m)','mic Y (m)','mic Z (m)']].values  # (64,3) mic coords
        M, N = raw_frame.shape                    # M=64, N=num_samples

        p_map = np.zeros(len(self.beam_dirs))
        for i, (theta, phi) in enumerate(self.beam_dirs):
            # 1) compute unit look vector
            d = np.array([
                np.sin(theta)*np.cos(phi),
                np.sin(theta)*np.sin(phi),
                np.cos(theta)
            ])

            # 2) plane‐wave delays (s) → samples (can be fractional)
            delays_s    = -(positions @ d) / c
            delays_smpl = delays_s * fs
            delays_int  = np.round(delays_smpl).astype(int)

            # 3) apply integer shifts
            shifted = np.zeros_like(raw_frame)
            for m in range(M):
                dm = delays_int[m]
                if dm > 0:
                    shifted[m, dm:] = raw_frame[m, :N-dm]
                elif dm < 0:
                    shifted[m, :N+dm] = raw_frame[m, -dm:]
                else:
                    shifted[m, :]    = raw_frame[m, :]

            # 4) sum and compute power
            beam = shifted.sum(axis=0)
            p_map[i] = np.mean(beam**2)

        # 5) normalize
        if p_map.max() > 0:
            p_map /= p_map.max()
        return p_map

    def load_data(self):
        """Load audio files and geometry data"""
        print("Loading ambisonic recording...")
        self.ambisonic_data, self.sample_rate = sf.read(self.ambisonic_file)
        if self.ambisonic_data.ndim == 2:
            self.ambisonic_data = self.ambisonic_data.T
        else:
            # Handle 1D array case if necessary
            self.ambisonic_data = self.ambisonic_data.reshape(1, -1)
        self.ambisonic_data = self.ambisonic_proc.convert_acn_to_complex_n3d(self.ambisonic_data)
        print(f"Loaded ambisonic data: {self.ambisonic_data.shape}")

        # Note: raw_data and geom_data are loaded but not used in the current pipeline.
        # You can remove these if they are not needed for future features.
        print("Loading raw recording...")
        self.raw_data, _ = sf.read(self.raw_file)
        print(f"Loaded raw data: {self.raw_data.shape}")

        print("Loading geometry data...")
        self.geom_data = pd.read_csv(self.geom_file)
        print(f"Loaded geometry data: {self.geom_data.shape}")


        '''
        # Transpose ambisonic data to (channels, samples)
        if self.ambisonic_data.ndim == 2:
            self.ambisonic_data = self.ambisonic_data.T
        else:
            self.ambisonic_data = self.ambisonic_data.reshape(1, -1)
        '''
    def process_frame(self, frame_idx):
        """Process a single frame of audio/video"""
        start_sample = frame_idx * self.hop_size
        end_sample = start_sample + self.hop_size

        if end_sample > self.ambisonic_data.shape[1]:
            return None, None

        frame_data = self.ambisonic_data[:, start_sample:end_sample]
        frame_data = frame_data.astype(complex)

        # Apply beamforming using pre-calculated weights
        ###beam_signals, beam_powers = self.ambisonic_proc.beamform(frame_data, self.beam_weights)

        ### DAS
        raw_block = self.raw_data[start_sample:end_sample, :].T
        beam_powers = self.compute_acoustic_map_das(raw_block)

        '''
        beam_powers = self.ambisonic_proc.compute_mvdr_beamformer(
            frame_data, self.beam_dirs, alpha=1e-2,
            encoding_format='complex')
        '''

        ## Calculate beam powers (RMS)
        #beam_powers = np.sqrt(np.mean(beam_signals ** 2, axis=1))
        '''
        beam_powers = self.compute_acoustic_map_mvdr(frame_data,
                                               order=self.ambisonic_proc.order,
                                               alpha=1e-3)
        '''

        # Generate binaural audio for this frame
        #binaural_audio = self.hrtf_proc.apply_binaural_processing(frame_data)
        binaural_audio = self.hrtf_proc.apply_binaural_processing_real(frame_data, (0, 0))

        # Create the new pixelated grid visualization
        viz_frame = self.viz_engine.create_pixelated_grid_visualization(beam_powers)

        return viz_frame, binaural_audio


    def compute_acoustic_map_mvdr(self,
                                  frame_data,
                                  order=6,
                                  alpha=1e-3):
        """
        Broadband MVDR over *all* grid directions in self.beam_dirs.
        frame_data:  (Nsh, Nsamples) HOA‐ACN time-block.
        Returns p_map: (D,) normalized power map (max=1).
        """
        # 1) Covariance of ACN channels (no STFT needed for a short block)
        R = frame_data @ frame_data.T
        R = R / frame_data.shape[1]
        R += alpha * np.eye(R.shape[0])

        # 2) Pre-invert
        Rinv = np.linalg.inv(R)

        # 3) Build steering matrix V (D directions × Nsh)
        from scipy.special import sph_harm
        dirs_cart = np.array([
            [np.sin(t) * np.cos(p),
             np.sin(t) * np.sin(p),
             np.cos(t)]
            for t, p in self.beam_dirs
        ])  # (D,3)
        θ = np.arccos(np.clip(dirs_cart[:, 2], -1, 1))
        φ = np.arctan2(dirs_cart[:, 1], dirs_cart[:, 0])

        V = np.column_stack([
            sph_harm(m, l, φ, θ)
            for l in range(order + 1)
            for m in range(-l, l + 1)
        ])  # (D, Nsh)

        # 4) MVDR pseudo-power: p_i = 1 / ( v_iᴴ R⁻¹ v_i )
        denom = np.einsum('dn,nm,dm->d', V.conj(), Rinv, V).real
        p_map = 1.0 / np.clip(denom, 1e-12, None)

        # 5) normalize to [0,1]
        return p_map / np.max(p_map)

    def run(self, output_file="acoustic_video.mp4"):
        """Run the complete processing pipeline and combine audio/video."""
        print("Starting acoustic video processing...")

        temp_video_file = "temp_video_only.mp4"
        temp_audio_file = "temp_audio.wav"

        total_samples = self.ambisonic_data.shape[1]
        total_frames = total_samples // self.hop_size
        print(f"Processing {total_frames} frames...")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(temp_video_file, fourcc, self.video_fps,
                                       (self.viz_engine.width, self.viz_engine.height))

        all_audio_chunks = []

        try:
            # 1. Process all frames, generating video and audio data
            for frame_idx in range(total_frames):
                if frame_idx % 30 == 0:
                    print(f"Processing frame {frame_idx}/{total_frames} "
                          f"({100 * frame_idx / total_frames:.1f}%)")

                viz_frame, binaural_audio = self.process_frame(frame_idx)

                if viz_frame is None: break
                video_writer.write(viz_frame)

                if binaural_audio is not None:
                    all_audio_chunks.append(binaural_audio)

        except KeyboardInterrupt:
            print("\nProcessing interrupted by user.")
        finally:
            video_writer.release()
            print("Video frame generation complete.")

        if not all_audio_chunks:
            print("No audio was generated. The output video will be silent.")
            if os.path.exists(temp_video_file):
                os.rename(temp_video_file, output_file)
            return

        # 2. Concatenate and save the complete, continuous audio track
        print("Concatenating and saving full audio track...")
        full_audio = np.concatenate(all_audio_chunks, axis=1)
        full_audio= full_audio.real

        # Normalize the entire audio track at once to avoid jumps and clipping
        max_val = np.max(np.abs(full_audio))
        if max_val > 0:
            full_audio_normalized = (full_audio / max_val) * 0.98
        else:
            full_audio_normalized = full_audio

        sf.write(temp_audio_file, full_audio_normalized.T, self.sample_rate)
        print(f"Audio track saved to {temp_audio_file}")

        # 3. Mux (combine) video and audio using moviepy
        print("Muxing video and audio into final file...")
        try:
            video_clip = VideoFileClip(temp_video_file)
            audio_clip = AudioFileClip(temp_audio_file)

            final_clip = video_clip.set_audio(audio_clip)
            final_clip.write_videofile(output_file, codec='libx264', audio_codec='aac',
                                       temp_audiofile='temp-audio.m4a', remove_temp=True,
                                       logger='bar')
            print(f"\nProcessing complete! Final video saved as {output_file}")

        except Exception as e:
            print(f"\nError combining video and audio with moviepy: {e}")
            print(f"Temporary files are available: '{temp_video_file}' and '{temp_audio_file}'")
        finally:
            # 4. Clean up temporary files
            if os.path.exists(temp_video_file): os.remove(temp_video_file)
            if os.path.exists(temp_audio_file): os.remove(temp_audio_file)

def plot_directivity_on_sphere(beam_dirs, beam_powers):
    # beam_dirs: list of (θ,φ) in radians
    # beam_powers: array of length len(beam_dirs), normalized [0–1]

    # build regular grid for surface
    N = int(np.sqrt(len(beam_dirs)))
    thetas = np.linspace(0, np.pi, N)
    phis = np.linspace(0, 2 * np.pi, N)
    Θ, Φ = np.meshgrid(thetas, phis, indexing='ij')

    # map beam_powers into grid array
    P = np.zeros_like(Θ)
    for (θ, φ), p in zip(beam_dirs, beam_powers):
        # find nearest grid indices
        i = np.argmin(np.abs(thetas - θ))
        j = np.argmin(np.abs(phis - φ))
        P[i, j] = p

    # sphere xyz
    X = np.sin(Θ) * np.cos(Φ)
    Y = np.sin(Θ) * np.sin(Φ)
    Z = np.cos(Θ)

    fig = go.Figure(go.Surface(
        x=X, y=Y, z=Z,
        surfacecolor=P,
        colorscale='Jet',
        cmin=0, cmax=1,
    ))

    # camera / axes so that +X=front, +Y=left, +Z=up
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

def main():
    """Main function to run the acoustic video system"""
    # File paths (update these to match your files)
    ambisonic_file = "/media/agjaci/Extreme SSD/em64_rec/ES3_20250522_101457_hoa-4.wav"
    raw_file = "/media/agjaci/Extreme SSD/em64_rec/ES3_20250522_101457_raw-4.wav"
    geom_file = "/media/agjaci/Extreme SSD/em64_rec/em64_geom.csv"

    # Check if files exist
    for filepath in [ambisonic_file, raw_file, geom_file]:
        if not os.path.exists(filepath):
            print(f"Error: File not found: {filepath}")
            print("Please update the file paths in the main() function")
            return

    try:
        # Create processor
        processor = AudioVideoProcessor(ambisonic_file, raw_file, geom_file)

        # Run processing
        processor.run("eigenmike_acoustic_video_das.mp4")

    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()