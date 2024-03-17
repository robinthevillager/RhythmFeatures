import numpy as np
from scipy.signal import get_window
import librosa

""" DSP Functions for Rhythm Feature Extraction """

DB_RANGE = 80.0
eps = np.finfo(float).eps


def load_audio_data(path: str = None, fs: int = 44100) -> (np.ndarray, int):
    """ Load & Prepare Audio Data"""
    # Load Audio Data
    audio_data, fs = librosa.load(path, sr=fs)
    # Sum to Mono (if multichannel file)
    if len(audio_data.shape) > 1:
        audio_data = np.sum(audio_data, axis=1) * 1/len(audio_data.shape[1])
    # Normalise Audio Data
    audio_data = normalise(audio_data)
    # Pad Audio Data (add 10ms silence for last downbeat)
    audio_data = pad_audio(audio_data, [0, int(fs*0.01)])

    return audio_data, fs


def pad_audio(audio: np.ndarray, padding: list = [100, 100]) -> np.ndarray:
    """ Pad Audio File for better Beat Estimation"""
    return np.pad(audio, padding, mode='constant')


def normalise(x: np.ndarray) -> np.ndarray:
    """ Normalise Data"""
    return x / max(np.abs(x))


def scale(x: np.ndarray, min_val: float = 0, max_val: float = 1) -> np.ndarray:
    """ Scale Data
    Args:
        x (np.ndarray): Data
        min_val (float): Minimum Value
        max_val (float): Maximum Value
    Returns:
        x (np.ndarray): Scaled
    """
    x = (x - np.min(x)) / (np.max(x) - np.min(x))  # Scale to [0, 1]
    x = x * (max_val - min_val) + min_val          # Scale to [min_val, max_val]
    return x


def center_of_mass(audio: np.ndarray) -> float:
    """ Center of Mass Function"""
    # Compute RMS Energy
    rms = rms_energy(audio, 128)**4
    # Compute Center of Mass
    com = np.sum(np.arange(len(rms)) * rms) / np.sum(rms)
    return com


def power_to_db(power: float, ref_db: float = 0.0, range_db: float = DB_RANGE) -> float:
    """Converts power from linear scale to decibels."""
    # Convert to decibels.
    pmin = 10**-(range_db / 10.0)
    power = np.maximum(pmin, power)
    db = 10.0 * np.log10(power)

    # Set dynamic range.
    db -= ref_db
    db = np.maximum(db, -range_db)
    return db


def amplitude_to_db(amplitude: float, ref_db: float = 0.0, range_db: float = DB_RANGE) -> float:
    """Converts amplitude in linear scale to power in decibels."""
    power = amplitude**2.0
    return power_to_db(power, ref_db=ref_db, range_db=range_db)


def compute_loudness(audio: np.ndarray, sample_rate: int = 44100) -> float:
    """Perceptual loudness (weighted power) in dB.

    Parameters:
        audio: 1D np.array of audio samples.
        sample_rate: Sample rate in Hz.

    Returns:
        Loudness in decibels. Shape [batch_size, n_frames] or [n_frames,].
    """

    # Take STFT.
    s = np.fft.rfft(audio)

    # Compute power.
    magnitude = np.abs(s) / s.size
    power = magnitude**2

    # Perceptual weighting.
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=audio.size)
    a_weighting = librosa.A_weighting(frequencies + eps)

    # Perform weighting in linear scale, a_weighting given in decibels.
    weighting = 10**(a_weighting/10)
    power = power * weighting

    loudness = np.sum(power, axis=0)

    return loudness


def spectral_novelty(audio: np.ndarray) -> np.ndarray:
    """ Spectral Novelty Function"""
    # Compute Spectrogram
    S = np.abs(librosa.stft(audio, n_fft=128, hop_length=64))
    # Compute Spectral Novelty
    spectral_novelty = np.mean(np.diff(S, axis=1)**2, axis=0)
    # Half Wave Rectification
    spectral_novelty = np.maximum(0, spectral_novelty)
    # Convert to Time Domain
    spectral_novelty = librosa.frames_to_time(spectral_novelty, hop_length=64, n_fft=128, sr=44100)
    return spectral_novelty


def get_peaks(x: np.ndarray, thresh: float = 0.0) -> np.ndarray:
    """ Peak Picking Algorithm"""
    local_maxima = [p for p in range(1, len(x)-1) if x[p] > x[p-1] and x[p] > x[p+1]]
    peaks = [p for p in local_maxima if x[p] > thresh]
    return np.array(peaks)


def rms_energy(x: np.ndarray, win: int) -> np.ndarray:
    """ Compute RMS Energy of Signal"""
    return np.sqrt(np.convolve(x**2, np.ones(win) / win, mode='same'))


def conv_smoothing(x: np.ndarray, M: int, win='hann') -> np.ndarray:
    """ Smoothing Function using Convolution"""
    w = get_window(win, M)
    if len(x.shape) > 1:
        y = np.array([np.convolve(x[b], w / np.sum(w), mode='same') for b in range(x.shape[0])])
    else:
        y = np.convolve(x, w / sum(w), mode='same')
    return y


def timestamps_to_samples(timestamps: np.ndarray, sr: int = 44100) -> np.ndarray:
    """ Convert timestamps to samples.

    Parameters
    ----------
    timestamps : np.ndarray
        Array of time stamps in seconds.
    sr : int
        Sampling rate.

    Returns
    -------
    samples : np.ndarray
        Array of sample indices.
    """
    return (np.array(timestamps) * sr).astype(int)


def generate_click_sound(click_length: float = 0.02, fs: int = 44100) -> np.ndarray:
    """ Generate a click sound for sonification.

    Parameters
    ----------
    click_length : float
        Length of the click sound in seconds.
    fs : int
        Sampling rate.

    Returns
    -------
    click : np.ndarray
        Array of audio samples of the click sound.

    """
    # Generate Click Sound
    click_samps = int(click_length * fs)
    envelope = np.hanning(click_samps * 2)[click_samps:]
    noise = np.random.uniform(-1, 1, click_samps)
    ping = (2 * np.sin(2 * np.pi * 880 * np.arange(click_samps) / fs)
            + 1 * np.sin(2 * np.pi * 1760 * np.arange(click_samps) / fs)
            + 0.808 * np.sin(2 * np.pi * 2640 * np.arange(click_samps) / fs)
            + 0.707 * np.sin(2 * np.pi * 3520 * np.arange(click_samps) / fs))

    # Combine Noise and Ping
    click = noise * envelope ** 2 + ping * envelope ** 0.25
    click = normalise(click)

    return click

