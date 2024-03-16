import matplotlib.pyplot as plt
import numpy as np
import dsp_functions as dsp

""" Plotting Functions """


def plot_beats_downbeats(x: np.array, sr: int, beat_times: list, downbeat_times: list):
    """
    Plot Beats and Downbeats on top of Audio Signal

    Parameters
    ----------
    x : np.array
        Audio Signal
    sr : int
        Sampling Rate
    beat_times : list
        List of Beat Times in Seconds
    downbeat_times : list
        List of Downbeat Times in Seconds

    """
    plt.figure(figsize=(12, 3))
    plt.plot(np.arange(x.size/sr), x)
    plt.vlines(beat_times, -1, 1, label='Beats', color='red', linewidths=1, linestyle = ':')
    plt.vlines(downbeat_times, -1, 1, label='DownBeats', color='black', linewidths=1, linestyle = '--')
    plt.title("Estimated Beats and Downbeats")
    plt.ylabel('Amplitude')
    plt.xlabel('Time (sec)')
    plt.legend(frameon=True, framealpha=1.0, edgecolor='black')


def plot_onset_stats_on_frame(audio_segment: np.ndarray,
                              win: np.ndarray,
                              center_of_mass: float,
                              spectral_onset: float,
                              offset_left: float) -> None:

    plt.figure(figsize=(17, 8))
    plt.plot(audio_segment)
    plt.plot(win, color='dimgray', label='Focus Window')
    plt.axvline(center_of_mass, color='red', label='Center of Mass')
    plt.axvline(spectral_onset, color='blue', label='Spectral Onset')
    plt.axvline(offset_left, color='black', label='Current Reference Beat')
    plt.plot(dsp.rms_energy(audio_segment, 128), label="RMS Energy")
    plt.legend()
    plt.tight_layout()


def create_analysis_plot(audio: np.ndarray,
                         time: np.ndarray,
                         note_times: np.ndarray,
                         time_signature: list,
                         beats_per_bar: int,
                         beat_times: np.ndarray,
                         downbeat_times: np.ndarray,
                         onset_times: np.ndarray,
                         onset_loudnesses: np.ndarray,
                         file_name: str) -> plt:
    """ Plot the estimated rhythm features."""
    plt.figure(figsize=(17, 8))
    plt.plot(time, audio, label='Audio Signal', color='teal', alpha=0.75)
    plt.vlines(note_times, -1, 1, label=f'{beats_per_bar}th Notes', color='dimgray', linestyle='--',
               alpha=0.75)
    plt.vlines(beat_times, -1, 1, label=f'{time_signature[1]}th Notes', color='dimgray', alpha=0.85,
               linewidths=2, linestyle='-')
    plt.vlines(downbeat_times, -1, 1, label='Downbeats', color='black', linewidths=4, linestyle='-')
    plt.vlines(onset_times, -1, 1, label='Onsets', color='red', alpha=0.75, linewidths=1, linestyle='-')
    plt.plot(onset_times, onset_loudnesses,
             label='Onset Strength',
             color='red',
             linestyle='None',
             marker='o',
             markersize=5)
    plt.title(f"Estimated Rhythm Features: {file_name}")
    plt.xlabel('Time (sec)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.tight_layout()

    return plt

