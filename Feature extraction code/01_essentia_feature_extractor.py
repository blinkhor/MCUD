#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Essentia Audio Feature Extractor
================================
Extract audio features using Essentia library with Librosa fallback.

Features include:
- MFCC (13 dims + delta + delta-delta = 39 dims)
- Spectral features (centroid, flux, entropy, bandwidth, rolloff)
- Temporal features (ZCR, RMS energy)
- Chroma features
- Rhythm features (Tempo, Beat)

Usage:
    python 01_essentia_feature_extractor.py <audio_file> [-o output_file]

Author: Floating Fu @ MCUD
Date: 2026-05-26
"""

import argparse
import os
import sys
import json
import numpy as np

# ============================================================================
# Try to import essentia. If not available, use librosa as fallback.
# ============================================================================
try:
    import essentia
    # Recommended: use built-in algorithm factory
    from essentia import (
        EqloudLoader,
        Windowing,
        FFT,
        CartesianToPolar,
        MFCC,
        SpectralCentroid,
        SpectralFlux,
        SpectralEntropy,
        ZeroCrossingRate,
        RMS,
        DemuxInertia,
        Pool,
        runsStandard as pool,
    )
    from essentia.standard import (
        MonoLoader,
        FrameGenerator,
        Welch,
        Key,
        Danceability,
        BpmHistogramDescriptors,
        BpmEvents,
        Rhythm2013,
        BeatsLoudness,
    )
    ESSENTIA_AVAILABLE = True
except ImportError:
    ESSENTIA_AVAILABLE = False
    # If essentia is not available, try to use librosa as fallback
    try:
        import librosa
        LIBROSA_AVAILABLE = True
    except ImportError:
        LIBROSA_AVAILABLE = False


def load_audio(audio_path: str, sample_rate: int = 44100) -> np.ndarray:
    """
    Load audio file.

    Parameters
    ----------
    audio_path : str
        Path to audio file.
    sample_rate : int
        Target sample rate.

    Returns
    -------
    audio : np.ndarray
        Audio signal (mono).
    sr : int
        Actual sample rate.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if ESSENTIA_AVAILABLE:
        # Use essentia loader
        loader = MonoLoader(filename=audio_path, sampleRate=sample_rate)
        audio = loader()
        return audio, sample_rate
    elif LIBROSA_AVAILABLE:
        # Use librosa loader
        audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
        return audio, sr
    else:
        # Try soundfile + scipy
        try:
            import soundfile as sf
            audio, sr = sf.read(audio_path)
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            # Resample if needed
            if sr != sample_rate:
                from scipy.signal import resample
                num_samples = int(len(audio) * sample_rate / sr)
                audio = resample(audio, num_samples)
            return audio, sample_rate
        except ImportError:
            raise ImportError(
                "Failed to load audio! Please install one of:\n"
                "  - essentia: pip install essentia\n"
                "  - librosa: pip install librosa soundfile\n"
            )


def extract_features_essentia(audio: np.ndarray, sr: int) -> dict:
    """
    Extract audio features using Essentia.

    Parameters
    ----------
    audio : np.ndarray
        Audio signal.
    sr : int
        Sample rate.

    Returns
    -------
    features : dict
        Feature dictionary.
    """
    pool = Pool()

    # Configure algorithm parameters
    frame_size = 2048
    hop_size = 512

    # Create frame generator
    for frame in FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size):
        # Apply window
        windowed = Windowing(type='hann', zeroPhase=True)(frame)

        # FFT
        fft = FFT()(windowed)
        fft_mag, _ = CartesianToPolar()(fft)

        # MFCC (13 dims + delta)
        mfcc = MFCC(numberCoefficients=13, numberBuckets=36, highFrequencyBound=sr//2)(fft_mag)
        # mfcc[0] is log-energy, mfcc[1:] is 13 coefficients

        # Spectral features
        sc = SpectralCentroid()(fft_mag)  # spectral centroid
        sf = SpectralFlux()(fft_mag)   # spectral flux
        se = SpectralEntropy()(fft_mag)  # spectral entropy

        # Temporal features
        zcr = ZeroCrossingRate()(frame)  # zero crossing rate
        rms = RMS()(frame)             # RMS energy

        # Add to pool
        pool.add('mfcc', mfcc)
        pool.add('spectral_centroid', sc)
        pool.add('spectral_flux', sf)
        pool.add('spectral_entropy', se)
        pool.add('zcr', zcr)
        pool.add('rms', rms)

    # Compute global statistics
    features = {
        # MFCC statistics
        'mfcc_mean': np.mean(pool['mfcc'], axis=0).tolist(),
        'mfcc_std': np.std(pool['mfcc'], axis=0).tolist(),

        # Spectral statistics
        'spectral_centroid_mean': float(np.mean(pool['spectral_centroid'])),
        'spectral_centroid_std': float(np.std(pool['spectral_centroid'])),
        'spectral_flux_mean': float(np.mean(pool['spectral_flux'])),
        'spectral_flux_std': float(np.std(pool['spectral_flux'])),
        'spectral_entropy_mean': float(np.mean(pool['spectral_entropy'])),
        'spectral_entropy_std': float(np.std(pool['spectral_entropy'])),

        # Temporal statistics
        'zcr_mean': float(np.mean(pool['zcr'])),
        'zcr_std': float(np.std(pool['zcr'])),
        'rms_mean': float(np.mean(pool['rms'])),
        'rms_std': float(np.std(pool['rms'])),
    }

    # Add key and rhythm features (based on full audio)
    try:
        key = Key()+ Danceable() # requires input of raw audio
        # Simplified - these features need more complex processing
    except Exception:
        pass

    # Try to extract rhythm features
    try:
        rhythm = Rhythm2013()(audio)
        # rhythm[0] is tempo
        # rhythm[1] is beats info
        if hasattr(rhythm, '__len__') and len(rhythm) >= 2:
            features['tempo'] = float(rhythm[0]) if rhythm[0] is not None else 0.0
    except Exception:
        features['tempo'] = 0.0

    return features


def extract_features_librosa_fallback(audio: np.ndarray, sr: int) -> dict:
    """
    Extract features using librosa as fallback.

    Parameters
    ----------
    audio : np.ndarray
        Audio signal.
    sr : int
        Sample rate.

    Returns
    -------
    features : dict
        Feature dictionary.
    """
    import librosa

    # ==================== MFCC ====================
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    features = {
        # MFCC features (13 + 13 delta + 13 delta2 = 39 dims)
        'mfcc': np.vstack([mfcc, mfcc_delta, mfcc_delta2]),  # (39, frames)
        'mfcc_mean': np.mean(mfcc, axis=1).tolist(),
        'mfcc_std': np.std(mfcc, axis=1).tolist(),

        # Delta MFCC statistics
        'mfcc_delta_mean': np.mean(mfcc_delta, axis=1).tolist(),
        'mfcc_delta_std': np.std(mfcc_delta, axis=1).tolist(),
    }

    # ==================== Spectral features ====================
    # Spectral centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    features['spectral_centroid_mean'] = float(np.mean(spectral_centroid))
    features['spectral_centroid_std'] = float(np.std(spectral_centroid))

    # Spectral bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    features['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))
    features['spectral_bandwidth_std'] = float(np.std(spectral_bandwidth))

    # Spectral rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
    features['spectral_rolloff_std'] = float(np.std(spectral_rolloff))

    # Spectral entropy (approximated with flatness)
    spectral_flatness = librosa.feature.spectral_flatness(y=audio)
    features['spectral_flatness_mean'] = float(np.mean(spectral_flatness))
    features['spectral_flatness_std'] = float(np.std(spectral_flatness))

    # ==================== Temporal features ====================
    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y=audio)
    features['zcr_mean'] = float(np.mean(zcr))
    features['zcr_std'] = float(np.std(zcr))

    # RMS energy
    rms = librosa.feature.rms(y=audio)
    features['rms_mean'] = float(np.mean(rms))
    features['rms_std'] = float(np.std(rms))

    # ==================== Chroma features ====================
    chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
    features['chroma_mean'] = np.mean(chroma, axis=1).tolist()
    features['chroma_std'] = np.std(chroma, axis=1).tolist()

    # ==================== Rhythm features ====================
    try:
        tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
        features['tempo'] = float(tempo)
    except Exception:
        features['tempo'] = 0.0

    return features


def extract_features(audio: np.ndarray, sr: int) -> dict:
    """
    Main feature extraction function. Auto-selects the best backend.

    Parameters
    ----------
    audio : np.ndarray
        Audio signal.
    sr : int
        Sample rate.

    Returns
    -------
    features : dict
        Feature dictionary.
    """
    if ESSENTIA_AVAILABLE:
        return extract_features_essentia(audio, sr)
    else:
        return extract_features_librosa_fallback(audio, sr)


def features_to_array(features: dict) -> np.ndarray:
    """
    Convert feature dictionary to flattened numpy array.

    Parameters
    ----------
    features : dict
        Feature dictionary.

    Returns
    -------
    arr : np.ndarray
        Flattened feature vector.
    """
    flat_features = []

    for key, value in features.items():
        if isinstance(value, list):
            flat_features.extend(value)
        elif isinstance(value, (int, float)):
            flat_features.append(value)
        elif isinstance(value, np.ndarray):
            flat_features.extend(value.flatten().tolist())

    return np.array(flat_features)


def save_features(features: dict, output_path: str, as_json: bool = False):
    """
    Save features to file.

    Parameters
    ----------
    features : dict
        Feature dictionary.
    output_path : str
        Output path.
    as_json : bool
        Whether to save as JSON format.
    """
    if as_json:
        # JSON only supports serializable data
        serializable = {}
        for key, value in features.items():
            if isinstance(value, np.ndarray):
                serializable[key] = value.tolist()
            elif isinstance(value, list):
                # Check if numeric list
                if value and isinstance(value[0], (int, float)):
                    serializable[key] = value
                else:
                    serializable[key] = str(value)
            elif isinstance(value, (int, float, str)):
                serializable[key] = value

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
    else:
        # Default: save as numpy dictionary
        np.save(output_path, features, allow_pickle=True)

    print(f"Features saved to: {output_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Extract audio features using Essentia/Librosa',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('input_audio', help='Input audio file path')
    parser.add_argument('-o', '--output', help='Output file path (default: {input}_features.npy)',
                     default=None)
    parser.add_argument('--json', action='store_true',
                     help='Output JSON format instead of numpy')
    parser.add_argument('--sr', '--sample-rate', type=int, default=44100,
                     help='Sample rate (default: 44100)')
    parser.add_argument('-v', '--verbose', action='store_true',
                     help='Show detailed information')

    args = parser.parse_args()

    # Determine output path
    if args.output is None:
        input_name = os.path.splitext(os.path.basename(args.input_audio))[0]
        ext = '.json' if args.json else '.npy'
        args.output = f"{input_name}_features{ext}"

    if args.verbose:
        print(f"Input audio: {args.input_audio}")
        print(f"Output file: {args.output}")
        print(f"Sample rate: {args.sr}")

    # Load audio
    print("Loading audio...")
    audio, sr = load_audio(args.input_audio, args.sr)
    print(f"Audio duration: {len(audio)/sr:.2f} seconds")

    # Extract features
    print("Extracting features...")
    features = extract_features(audio, sr)

    if args.verbose:
        print(f"Extracted {len(features)} feature groups")

        # Show feature dimensions
        for key, value in features.items():
            if isinstance(value, list):
                print(f"  - {key}: {len(value)} dims")
            elif isinstance(value, np.ndarray):
                print(f"  - {key}: {value.shape}")
            else:
                print(f"  - {key}: {value}")

    # Save features
    save_features(features, args.output, as_json=args.json)

    # Also save flattened array version (for ML use)
    array_output = args.output.replace('.npy', '_vector.npy').replace('.json', '_vector.npy')
    arr = features_to_array(features)
    np.save(array_output, arr)
    print(f"Feature vector saved to: {array_output} (shape: {arr.shape})")


if __name__ == '__main__':
    main()