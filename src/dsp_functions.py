import numpy as np
from scipy.signal import get_window
import librosa


def load_audio_data(path, fs=44100):
    """ Load & Prepare Audio Data"""
    # Load Audio Data
    audio_data, fs = librosa.load(path, sr=fs)
    # Sum to Mono (if multichannel file)
    if len(audio_data.shape) > 1:
        audio_data = np.sum(audio_data, axis=1) * 1/len(audio_data.shape[1])

    # Pad Audio File for better Beat Estimation
    # audio_data = np.pad(audio_data, (fs//2, fs//2), mode='constant')

    return normalise(audio_data), fs


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


def get_peaks(x: np.ndarray, thresh: float = 0.0) -> np.ndarray:
    """ Peak Picking Algorithm"""
    local_maxima = [p for p in range(1, len(x)-1) if x[p] > x[p-1] and x[p] > x[p+1]]
    peaks = [p for p in local_maxima if x[p] > thresh]
    return np.array(peaks)


def rms_energy(x: np.ndarray, win: int) -> np.ndarray:
    """ Compute RMS Energy of Signal"""
    return np.sqrt(np.convolve(x**2, np.ones(win) / win, mode='same'))


def conv_smoothing(x: np.ndarray, M: int, win='hann'):
    """ Smoothing Function using Convolution"""
    w = get_window(win, M)
    if len(x.shape) > 1:
        y = np.array([np.convolve(x[b], w / np.sum(w), mode='same') for b in range(x.shape[0])])
    else:
        y = np.convolve(x, w / sum(w), mode='same')
    return y