#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Librosa Mel Spectrogram Extractor
================================
Extract Mel Spectrogram using Librosa.

Parameters:
- n_fft: FFT window size
- hop_length: hop size
- n_mels: number of MEL filters
- fmin/fmax: frequency range

Usage:
    python 02_librosa_melspectrogram.py <audio_file> [-o output] [options]

Output:
- Mel spectrogram numpy array (.npy)
- Optional: visualization image (.png)

Author: Floating Fu @ MCUD
Date: 2026-05-26
"""

import argparse
import os
import sys
import numpy as np


def load_audio_librosa(audio_path: str, sr: int = None, mono: bool = True) -> tuple:
    """
    Load audio using librosa.

    Parameters
    ----------
    audio_path : str
        Path to audio file.
    sr : int, optional
        Target sample rate. If None, use original sample rate.
    mono : bool
        Whether to convert to mono.

    Returns
    -------
    y : np.ndarray
        Audio signal.
    sr : int
        Sample rate.
    """
    try:
        import librosa
    except ImportError:
        raise ImportError(
            "Please install librosa: pip install librosa\n"
            "Also required: pip install soundfile"
        )

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    y, sr = librosa.load(audio_path, sr=sr, mono=mono)
    return y, sr


def compute_mel_spectrogram(
    y: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    fmin: float = 0,
    fmax: float = None,
    power: float = 2.0
) -> np.ndarray:
    """
    Compute Mel spectrogram.

    Parameters
    ----------
    y : np.ndarray
        Audio signal.
    sr : int
        Sample rate.
    n_fft : int
        FFT window size.
    hop_length : int
        Hop size.
    n_mels : int
        Number of MEL filters.
    fmin : float
        Minimum frequency.
    fmax : float
        Maximum frequency (default: sr/2).
    power : float
        Power exponent (1.0 = amplitude, 2.0 = power).

    Returns
    -------
    mel_spec : np.ndarray
        Mel spectrogram with shape (n_mels, time_frames).
    """
    import librosa

    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax or sr / 2,
        power=power
    )

    # Convert to dB scale (optional)
    # mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    return mel_spec


def compute_mel_spectrogram_db(
    y: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    fmin: float = 0,
    fmax: float = None
) -> np.ndarray:
    """
    Compute Mel spectrogram in decibel scale.

    Parameters
    ----------
    Same as compute_mel_spectrogram.

    Returns
    -------
    mel_spec_db : np.ndarray
        Mel spectrogram in dB scale.
    """
    import librosa

    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax or sr / 2
    )

    # Convert to dB
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    return mel_spec_db


def visualize_mel_spectrogram(
    mel_spec: np.ndarray,
    sr: int,
    hop_length: int,
    output_path: str = None,
    title: str = "Mel Spectrogram",
    y_axis: str = "mel",
    x_axis: str = "time"
):
    """
    Visualize Mel spectrogram.

    Parameters
    ----------
    mel_spec : np.ndarray
        Mel spectrogram.
    sr : int
        Sample rate.
    hop_length : int
        Hop size.
    output_path : str, optional
        Output image path.
    title : str
        Plot title.
    y_axis : str
        Y-axis label.
    x_axis : str
        X-axis label.
    """
    import librosa.display
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    # Set font for Chinese characters
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(10, 6))

    # Display mel spectrogram
    img = librosa.display.specshow(
        mel_spec,
        sr=sr,
        hop_length=hop_length,
        x_axis=x_axis,
        y_axis=y_axis,
        ax=ax,
        cmap='viridis'
    )

    # Add colorbar
    fig.colorbar(img, ax=ax, format='%+2.0f dB')

    ax.set_title(title)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
    else:
        plt.show()

    plt.close()


def save_mel_spectrogram(mel_spec: np.ndarray, output_path: str, as_db: bool = False):
    """
    Save Mel spectrogram.

    Parameters
    ----------
    mel_spec : np.ndarray
        Mel spectrogram.
    output_path : str
        Output path.
    as_db : bool
        Whether to save in dB scale.
    """
    if as_db:
        import librosa
        # Convert to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        np.save(output_path, mel_spec_db)
    else:
        np.save(output_path, mel_spec)

    print(f"Mel spectrogram saved to: {output_path}")
    print(f"  Shape: {mel_spec.shape} (n_mels x time_frames)")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Extract Mel Spectrogram using Librosa',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Input/Output
    parser.add_argument('input_audio', help='Input audio file path')
    parser.add_argument('-o', '--output', help='Output file path', default=None)
    parser.add_argument('--img', '--image', help='Output visualization image path',
                     default=None)

    # Parameter options
    parser.add_argument('--n-fft', type=int, default=2048,
                     help='FFT window size (default: 2048)')
    parser.add_argument('--hop-length', type=int, default=512,
                     help='Hop size (default: 512)')
    parser.add_argument('--n-mels', type=int, default=128,
                     help='Number of MEL filters (default: 128)')
    parser.add_argument('--fmin', type=float, default=0,
                     help='Minimum frequency (default: 0)')
    parser.add_argument('--fmax', type=float, default=None,
                     help='Maximum frequency (default: sr/2)')
    parser.add_argument('--power', type=float, default=2.0,
                     help='Power exponent (default: 2.0 = power spectrum)')

    # Output options
    parser.add_argument('--db', '--decibel', action='store_true',
                       help='Output in dB scale instead of linear power')
    parser.add_argument('--sr', '--sample-rate', type=int, default=None,
                       help='Sample rate (default: use original file sample rate)')

    # Visualization options
    parser.add_argument('--no-display', action='store_true',
                       help='Do not display plot')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Show detailed information')

    args = parser.parse_args()

    # ==================== Handle input/output paths ====================
    input_name = os.path.splitext(os.path.basename(args.input_audio))[0]

    if args.output is None:
        suffix = '_melspec_db.npy' if args.db else '_melspec.npy'
        args.output = f"{input_name}{suffix}"

    if args.img is None and not args.no_display:
        args.img = f"{input_name}_melspec.png"

    # ==================== Load audio ====================
    print("Loading audio...")

    try:
        import librosa
    except ImportError:
        print("Error: Please install librosa", file=sys.stderr)
        print("  pip install librosa", file=sys.stderr)
        sys.exit(1)

    y, sr = load_audio_librosa(args.input_audio, sr=args.sr)

    print(f"  Sample rate: {sr} Hz")
    print(f"  Audio duration: {len(y)/sr:.2f} seconds")

    # ==================== Compute Mel spectrogram ====================
    print("Computing Mel spectrogram...")

    mel_spec = compute_mel_spectrogram(
        y=y,
        sr=sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        fmin=args.fmin,
        fmax=args.fmax,
        power=args.power
    )

    if args.verbose:
        print(f"  n_fft: {args.n_fft}")
        print(f"  hop_length: {args.hop_length}")
        print(f"  n_mels: {args.n_mels}")
        print(f"  Shape: {mel_spec.shape}")

    # ==================== Save results ====================
    save_mel_spectrogram(mel_spec, args.output, as_db=args.db)

    # ==================== Visualization ====================
    if args.img and not args.no_display:
        print("Generating visualization...")
        visualize_mel_spectrogram(
            mel_spec,
            sr=sr,
            hop_length=args.hop_length,
            output_path=args.img,
            title=f"Mel Spectrogram - {input_name}"
        )

    # Print summary
    print("\n" + "="*50)
    print("Summary:")
    print(f"  Input: {args.input_audio}")
    print(f"  Output: {args.output}")
    print(f"  Shape: {mel_spec.shape}")
    print("="*50)


if __name__ == '__main__':
    main()