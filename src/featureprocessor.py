import numpy as np
import os, sys
import madmom
import librosa
import mir_eval

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

import carat as cr
import dsp_functions as dsp


class FeatureProcessor:
    def __init__(self, base_path, file_name):
        # Get File Attributes
        self.base_path = base_path
        self.file_name = file_name
        self.audio_path = f'{self.base_path}/{self.file_name}'
        self.audio, self.fs = dsp.load_audio_data(self.audio_path)
        self.time = np.arange(self.audio.size) / self.fs

        # Define Time Signature
        self.beats_per_bar = 8
        self.time_signature = [4, 4]

        # Rhythm Features
        self.onset_strength: np.ndarray = None
        self.onset_times: np.ndarray = None
        self.downbeat_times: np.ndarray = None
        self.note_times: np.ndarray = None
        self.beat_times: np.ndarray = None
        self.tempo_bpm: int = None

    def extract_rhythm_features(self):
        """ Extract Rhythm Features from Audio Data"""

        # ---- Rhythm Features ----
        # Beat and Downbeat Times
        self.beat_times, self.downbeat_times = self._estimate_beats_downbeats_madmom()
        # Note Grid
        self.note_times = self._get_note_times(self.downbeat_times, beats_per_bar=self.beats_per_bar)
        # Onsets
        self.onset_times = self.estimate_onset_times()
        # TODO: Derive Onset Times and Strength using Carat / Madmom whatever
        self.onset_strength = self._get_onset_strength(self.onset_times)

        # Derive Tempo in BPM
        # TODO: Whole Process Should be Done Bar by Bar not for the whole file
        average_bar_time = np.average(np.diff(self.downbeat_times))
        average_beat_time = average_bar_time/2
        self.tempo_bpm = int(60 / average_beat_time)

        # ---- Collect Features ----
        features = self.collect_features()

        return features

    def _estimate_beats_downbeats_madmom(self):
        """ Compute beat positions using a machine learned onset novelty function with madmom. """

        # Beat estimation followed by downbeat detection in post

        proc1 = madmom.features.beats.RNNBeatProcessor()
        act1 = proc1(self.audio)

        # use DBNBeatTrackingProcessor from madmom to obtain the beat time stamps
        dbn = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
        beats = dbn(act1)

        # Subsequent analysis of detected beats to detect bar
        act2 = madmom.features.downbeats.RNNBarProcessor()((self.audio_path, beats))
        proc2 = madmom.features.downbeats.DBNBarTrackingProcessor(beats_per_bar=self.time_signature, fps=100)
        downbeats = proc2(act2)

        # separate beat and downbeat times
        downbeat_times = downbeats[downbeats[:, 1] == 1][:, 0]
        beat_times = downbeats[downbeats[:, 1] != 1][:, 0]

        return beat_times, downbeat_times

    def _get_note_times(self, downbeats: np.ndarray, beats_per_bar: int = 8):
        """ Get 16th Note Grid from Tempo.

        Parameters
        ----------
        tempo : float
            Estimated tempo in BPM.

        Returns
        -------
        grid : 1-d np.array
            Array of time stamps of the 16th note grid in seconds.
        """

        # Compute grid from downbeat to downbeat
        note_times = [np.linspace(downbeats[i-1], downbeats[i], beats_per_bar, endpoint=False)
                      for i in range(1, len(downbeats))]

        return np.array(note_times).flatten()

    def estimate_onset_times(self):
        """ Estimate the onsets of an audio signal using madmom.

        Parameters
        ----------
        audio : np.ndarray
            Audio signal.

        Returns
        -------
        onset_times : 1-d np.array
            Array of time stamps of the estimated onsets in seconds.
        """

        # Estimate Onsets
        onset_proc = madmom.features.onsets.RNNOnsetProcessor()
        onsets = onset_proc(self.audio)

        # Get Onset Times
        onset_peaks = dsp.get_peaks(onsets, 0.25)

        # Convert to Seconds
        onset_times = onset_peaks / 100

        return onset_times

    def _get_onset_strength(self, onset_times: np.ndarray):
        """ Compute the onset strength of an audio signal using RMS and Onset Times."""
        # Get Onset Strength
        audio_rms = dsp.rms_energy(self.audio, int(self.fs/100))
        onset_strength = dsp.normalise(audio_rms[[int(i*100) for i in onset_times]])
        # TODO: Should be Replaced by Carat or Madmom

        return onset_strength

    def collect_features(self):
        """ Make a dictionary of rhythm features.
        Return
        ------
            features : dict
                Dictionary of rhythm features.
        """
        # Construct Dictionary (no ndarrays)
        features = {"tempo": self.tempo_bpm,
                    "onset_times": list(self.onset_times),
                    "onset_strength": list(self.onset_strength)
                    }

        return features

    def _onset_strength_madmom(self, onset_times: np.ndarray):
        """ Compute the onset strength of an audio signal using madmom.

        Parameters
        ----------
        audio : np.ndarray
            Audio signal.
        onset_times : np.ndarray
            Array of time stamps of the estimated onsets in seconds.

        Returns
        -------
        onset_strength : np.ndarray
            Array of onset strength values.
        """

        # Compute Onset Strength
        onset_strength = madmom.features.onsets.onset_strength(self.audio, onset_times)

        return onset_strength

    def _onset_microtiming(self, onset_times: np.ndarray, note_times: np.ndarray):
        """ Compute microtiming statistics of onsets.

        Parameters
        ----------
        onset_times : np.ndarray
            Array of onset times in seconds.
        note_times : np.ndarray
            Array of note times in seconds.

        Returns
        -------
        onset_timing : np.ndarray
            Array of microtiming values in seconds.
        """

        # Compute Microtiming
        onset_timing = onset_times - note_times

        return onset_timing

    def estimate_tempo_madmom(self, audio: np.ndarray):
        """ Estimate the tempo of an audio signal using madmom.

        Parameters
        ----------
        audio : np.ndarray
            Audio signal.

        Returns
        -------
        tempo : float
            Estimated tempo in BPM.
        """

        proc = madmom.features.tempo.TempoEstimationProcessor(fps=100)
        tempo = proc(audio)

        return tempo

    def _timestamps_to_samples(self, timestamps, sr):
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
        samples = (timestamps * sr).astype(int)

        return samples

    def write_rhythm_features_plot(self, output_path: str = None):
        """ Plot the estimated rhythm features."""
        # Derive File Name without datatype
        file_name = self.file_name.split(".")[0]
        # ---- Plotting ----
        plt.figure(figsize=(17, 8))
        plt.plot(self.time, self.audio)
        plt.vlines(self.note_times, -1, 1, label=f'{self.beats_per_bar} - Note Grid', color='grey', linestyle='--')
        # plt.vlines(beat_times, -1, 1, label='Beats', color='orange', linewidths=1, linestyle='-')
        plt.vlines(self.downbeat_times, -1, 1, label='Downbeats', color='black', linewidths=2, linestyle='-')
        plt.vlines(self.onset_times, -1, 1, label='Onsets', color='coral', linewidths=2, linestyle='-')
        plt.plot(self.onset_times, self.onset_strength,
                 label='Onset Strength',
                 color='red',
                 linestyle='None',
                 marker='o',
                 markersize=5)
        plt.title(f"Estimated Rhythm Features: {file_name}")
        plt.legend()

        # Write fig to file
        plt.savefig(f'{output_path}/{file_name}_rhythm_features.png')
        plt.close()
        plt.clf()
