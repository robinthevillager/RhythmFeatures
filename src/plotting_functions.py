import matplotlib.pyplot as plt
import numpy as np

""" Plotting Functions """


def plot_beats_downbeats(x: np.array(), sr: int, beat_times: list, downbeat_times: list):
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
