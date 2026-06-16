#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OpenL3 Music Embedding Generator
================================
Generate music embedding vectors using OpenL3.

OpenL3 is a self-supervised audio embedding model optimized for music.
It captures high-level semantic features like harmony, melody, rhythm, etc.

Parameters:
- model: 'music' (music model) or 'audio' (generic audio model)
- embedding_size: 256 or 512
- frame_size: time frame size (default 0.1 sec)
- hop_size: hop size (default 0.1 sec)

Usage:
    python 03_openl3_embeddings.py <audio_file> [-o output] [options]

Output:
- Music embedding vector (.npy), default 512 dims
- Aggregation: mean pooling

Author: Floating Fu @ MCUD
Date: 2026-05-26
"""

import argparse
import os
import sys
import numpy as np


def load_audio(audio_path: str, sr: int = None, mono: bool = True) -> tuple:
    """
    Load audio.

    Parameters
    ----------
    audio_path : str
        Path to audio file.
    sr : int, optional
        Target sample rate (default: 22050, OpenL3 default).
    mono : bool
        Whether to convert to mono.

    Returns
    -------
    audio : np.ndarray
        Audio signal.
    sr : int
        Sample rate.
    """
    try:
        import librosa
    except ImportError:
        raise ImportError(
            "Please install librosa: pip install librosa"
        )

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # OpenL3 default sample rate is 22050 Hz
    if sr is None:
        sr = 22050

    audio, sr = librosa.load(audio_path, sr=sr, mono=mono)
    return audio, sr


def load_model(model_type: str = 'music', embedding_size: int = 512):
    """
    Load OpenL3 model.

    Parameters
    ----------
    model_type : str
        Model type: 'music' or 'audio'.
    embedding_size : int
        Embedding dimension: 256 or 512.

    Returns
    -------
    model : object
        OpenL3 model.
    """
    try:
        import openl3
    except ImportError:
        raise ImportError(
            "Please install openl3: pip install openl3\n"
            "Note: openl3 requires tensorflow or pytorch backend"
        )

    # Load pretrained model
    # Keras or torch backend will be auto-selected
    model = openl3.get_model_input(
        model_type=model_type,
        embedding_size=embedding_size,
        verbose=False
    )

    return model


def extract_embedding_openl3(
    audio: np.ndarray,
    sr: int,
    model_type: str = 'music',
    embedding_size: int = 512,
    frame_size: float = 0.1,
    hop_size: float = 0.1,
    chunk_size: int = None
) -> np.ndarray:
    """
    Extract embedding using OpenL3.

    Parameters
    ----------
    audio : np.ndarray
        Audio signal.
    sr : int
        Sample rate.
    model_type : str
        Model type: 'music' or 'audio'.
    embedding_size : int
        Embedding dimension.
    frame_size : float
        Time frame size (seconds).
    hop_size : float
        Hop size (seconds).
    chunk_size : int, optional
        Chunk size for memory optimization.

    Returns
    -------
    embeddings : np.ndarray
        Per-frame embeddings with shape (num_frames, embedding_size).
    """
    import openl3

    # Convert time parameters to sample counts
    frame_size_samples = int(frame_size * sr)
    hop_size_samples = int(hop_size * sr)

    # Extract embedding
    # openl3.process_audio returns (embeddings, timestamps)
    embeddings, timestamps = openl3.process_audio(
        audio,
        sr,
        model_type=model_type,
        embedding_size=embedding_size,
        frame_size=frame_size,
        hop_size=hop_size,
        chunk_size=chunk_size,
        verbose=False
    )

    return embeddings


def mean_pool(embeddings: np.ndarray) -> np.ndarray:
    """
    Mean pooling of embeddings.

    Parameters
    ----------
    embeddings : np.ndarray
        Per-frame embeddings with shape (num_frames, embedding_dim).

    Returns
    -------
    pooled : np.ndarray
        Pooled embedding with shape (embedding_dim,).
    """
    return np.mean(embeddings, axis=0)


def max_pool(embeddings: np.ndarray) -> np.ndarray:
    """
    Max pooling of embeddings.

    Parameters
    ----------
    embeddings : np.ndarray
        Per-frame embeddings.

    Returns
    -------
    pooled : np.ndarray
        Pooled embedding.
    """
    return np.max(embeddings, axis=0)


def std_pool(embeddings: np.ndarray) -> np.ndarray:
    """
    Standard deviation pooling of embeddings.

    Parameters
    ----------
    embeddings : np.ndarray
        Per-frame embeddings.

    Returns
    -------
    pooled : np.ndarray
        Pooled embedding.
    """
    return np.std(embeddings, axis=0)


def statistics_pool(embeddings: np.ndarray) -> dict:
    """
    Generate statistics pool.

    Parameters
    ----------
    embeddings : np.ndarray
        Per-frame embeddings.

    Returns
    -------
    stats : dict
        Dictionary containing mean, std, max, min.
    """
    return {
        'mean': mean_pool(embeddings),
        'std': std_pool(embeddings),
        'max': max_pool(embeddings),
        'min': np.min(embeddings, axis=0),
    }


def save_embedding(embedding: np.ndarray, output_path: str):
    """
    Save embedding.

    Parameters
    ----------
    embedding : np.ndarray
        Embedding vector.
    output_path : str
        Output path.
    """
    np.save(output_path, embedding)
    print(f"Embedding saved to: {output_path}")
    print(f"  Shape: {embedding.shape}")


def save_detailed_results(embeddings: np.ndarray, pooled: np.ndarray,
                      output_path: str, stats: dict = None):
    """
    Save detailed results.

    Parameters
    ----------
    embeddings : np.ndarray
        Raw per-frame embeddings.
    pooled : np.ndarray
        Pooled embedding.
    output_path : str
        Base output path.
    stats : dict, optional
        Statistics features.
    """
    # Save raw embeddings (may be large)
    raw_path = output_path.replace('.npy', '_frames.npy')
    np.save(raw_path, embeddings)
    print(f"Raw frame embeddings saved to: {raw_path}")
    print(f"  Shape: {embeddings.shape}")

    # Save pooled embedding
    np.save(output_path, pooled)
    print(f"Pooled embedding saved to: {output_path}")
    print(f"  Shape: {pooled.shape}")

    # Save statistics
    if stats is not None:
        stats_path = output_path.replace('.npy', '_stats.npy')
        # Convert dict to storable format
        combined = np.concatenate([
            stats['mean'],
            stats['std'],
            stats['max'],
            stats['min']
        ])
        np.save(stats_path, combined)
        print(f"Statistics saved to: {stats_path}")
        print(f"  Shape: {combined.shape}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Generate music embeddings using OpenL3',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Input/Output
    parser.add_argument('input_audio', help='Input audio file path')
    parser.add_argument('-o', '--output', help='Output file path', default=None)

    # Model parameters
    parser.add_argument('--model', choices=['music', 'audio'], default='music',
                       help='Model type: music or audio (default: music)')
    parser.add_argument('--embedding-size', type=int, choices=[256, 512],
                       default=512,
                       help='Embedding dimension (default: 512)')

    # Time parameters
    parser.add_argument('--frame-size', type=float, default=0.1,
                       help='Time frame size in seconds (default: 0.1)')
    parser.add_argument('--hop-size', type=float, default=0.1,
                       help='Hop size in seconds (default: 0.1)')
    parser.add_argument('--sr', '--sample-rate', type=int, default=None,
                       help='Sample rate (default: 22050)')

    # Aggregation options
    parser.add_argument('--pooling', choices=['mean', 'max', 'mean+std', 'none'],
                     default='mean',
                     help='Aggregation method (default: mean)')
    parser.add_argument('--save-frames', action='store_true',
                       help='Save per-frame raw embeddings as well')

    # Other options
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Show detailed information')

    args = parser.parse_args()

    # ==================== Handle input/output paths ====================
    input_name = os.path.splitext(os.path.basename(args.input_audio))[0]

    if args.output is None:
        suffix = '_embedding.npy'
        args.output = f"{input_name}{suffix}"

    # ==================== Load audio ====================
    print("Loading audio...")

    try:
        import librosa
    except ImportError:
        print("Error: Please install librosa", file=sys.stderr)
        sys.exit(1)

    audio, sr = load_audio(args.input_audio, sr=args.sr)

    if args.sr is None:
        args.sr = sr

    print(f"  Sample rate: {sr} Hz")
    print(f"  Audio duration: {len(audio)/sr:.2f} seconds")

    # ==================== Extract Embedding ====================
    print("Extracting OpenL3 embedding...")

    try:
        import openl3
    except ImportError:
        print("Error: Please install openl3", file=sys.stderr)
        print("  pip install openl3", file=sys.stderr)
        print("  (May require tensorflow or pytorch backend)", file=sys.stderr)
        sys.exit(1)

    embeddings = extract_embedding_openl3(
        audio=audio,
        sr=sr,
        model_type=args.model,
        embedding_size=args.embedding_size,
        frame_size=args.frame_size,
        hop_size=args.hop_size
    )

    if args.verbose:
        print(f"  Extracted {embeddings.shape[0]} frame embeddings")
        print(f"  Per-frame dimension: {embeddings.shape[1]}")

    # ==================== Pooling ====================
    print("Pooling...")

    if args.pooling == 'mean':
        pooled = mean_pool(embeddings)
        print(f"  Pooling method: mean")
    elif args.pooling == 'max':
        pooled = max_pool(embeddings)
        print(f"  Pooling method: max")
    elif args.pooling == 'mean+std':
        # Concatenate mean and std
        pooled = np.concatenate([
            mean_pool(embeddings),
            std_pool(embeddings)
        ])
        print(f"  Pooling method: mean + std")
    elif args.pooling == 'none':
        pooled = embeddings  # No pooling
        print(f"  No pooling, keeping {embeddings.shape[0]} frames")

    # ==================== Save results ====================
    if args.save_frames:
        # Save detailed results
        stats = statistics_pool(embeddings) if args.pooling != 'none' else None
        save_detailed_results(embeddings, pooled, args.output, stats)
    else:
        # Only save pooled embedding
        save_embedding(pooled, args.output)

    # Print summary
    print("\n" + "="*50)
    print("Summary:")
    print(f"  Input: {args.input_audio}")
    print(f"  Model: {args.model}")
    print(f"  Embedding dimension: {args.embedding_size}")
    print(f"  Pooling method: {args.pooling}")
    print(f"  Final output shape: {pooled.shape}")
    print("="*50)


if __name__ == '__main__':
    main()