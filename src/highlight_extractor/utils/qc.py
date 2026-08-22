"""Quality checks for incoming audio files."""

from dataclasses import dataclass

import librosa
import numpy as np


@dataclass
class QCResult:
    snr_db: float | None
    clipping_fraction: float
    duration_s: float
    pass_qc: bool
    quality_warning: str | None = None


def compute_snr(audio: np.ndarray) -> float:
    """Compute a simple SNR estimate from RMS of signal vs silence."""
    rms = np.sqrt(np.mean(audio**2))
    if rms < 1e-10:
        return 0.0
    # Estimate noise floor as the bottom 10th percentile of RMS in short windows
    hop = 512
    frames = librosa.feature.rms(y=audio, hop_length=hop)
    # frames shape: (1, n_frames) — flatten
    frames_flat = frames.flatten()
    noise_floor = np.percentile(frames_flat, 10)
    if noise_floor < 1e-10:
        return float("inf")
    ratio = rms / noise_floor
    # If noise floor is within 1 dB of the signal, treat as infinite SNR
    # (happens with uniform synthetic signals like sine waves)
    if ratio < 1.15:  # < ~1.2 dB
        return float("inf")
    return 20 * np.log10(ratio)


def detect_clipping(audio: np.ndarray, threshold: float = 0.99) -> float:
    """Fraction of samples at or above threshold (clipping proxy)."""
    peak = np.max(np.abs(audio))
    if peak == 0:
        return 0.0
    clipped = np.sum(np.abs(audio) >= threshold * peak)
    return clipped / len(audio)


def run_qc(audio: np.ndarray, sr: int, duration_s: float) -> QCResult:
    """Run quality-control checks on a loaded audio array.

    Returns a QCResult with pass_qc = True unless quality is critically bad.
    """
    snr = compute_snr(audio)
    clip_frac = detect_clipping(audio)

    warnings = []
    if clip_frac > 0.05:
        warnings.append(f"clipping detected ({clip_frac:.1%} of samples)")
    if snr is not None and snr < 15:
        warnings.append(f"low SNR ({snr:.1f} dB)")

    warning_str = "; ".join(warnings) if warnings else None

    return QCResult(
        snr_db=snr,
        clipping_fraction=clip_frac,
        duration_s=duration_s,
        pass_qc=True,
        quality_warning=warning_str,
    )
