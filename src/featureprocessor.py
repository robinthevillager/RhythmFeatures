import numpy as np
import madmom
import re
import soundfile as sf
import plotting_functions as pf
import dsp_functions as dsp
import midi_functions as midi


""" Feature Processor for Rhythm Feature Extraction """


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
        self.rhythm_features: dict = None
        self.rhythm_midi: object = None
        self.note_times: np.ndarray = None
        self.beat_times: np.ndarray = None
        self.downbeat_times: np.ndarray = None
        self.onset_times: np.ndarray = None
        self.onset_loudnesses: np.ndarray = None
        self.microtimings: np.ndarray = None
        self.bpm: int = None

    def extract_rhythm_features(self):
        """ Extract Rhythm Features from Audio Data"""

        # ---- Rhythm Features ----

        # Beat and Downbeat Times
        self.beat_times, self.downbeat_times = self._calculate_beats_downbeats()

        # Note Grid
        self.note_times = self._get_note_times()

        # Onset Statistics
        self.onset_times, self.microtimings, self.onset_loudnesses = self._estimate_onset_stats()

        # ---- Collect Features ----
        self.rhythm_features = self._collect_features()

        return self.rhythm_features

    def construct_midi(self):
        """ Constructs Midi File from Rhythm Features"""
        self.rhythm_midi = midi.construct_midi(self.rhythm_features)

        return self.rhythm_midi

    def write_analysis_data(self, output_path: str = None):
        """ Plot the estimated rhythm features."""
        # Derive File Name without datatype
        file_name = self.file_name.split(".")[0]

        # ---- Plotting ----
        fig = pf.create_analysis_plot(self.audio,
                                      self.time,
                                      self.note_times,
                                      self.time_signature,
                                      self.beats_per_bar,
                                      self.beat_times,
                                      self.downbeat_times,
                                      self.onset_times,
                                      self.onset_loudnesses,
                                      file_name)

        # Write fig to file
        fig.savefig(f'{output_path}/{file_name}_rhythm_features.png')
        fig.clf()

        # --- Onset Sonification ---
        onsets_sonified = self._sonify_onsets()
        sf.write(f'{output_path}/{file_name}_onset_sonification.wav', onsets_sonified, samplerate=self.fs)

    def _calculate_beats_downbeats(self):
        """ Calculate beat and downbeat times using a simple BPM calculation. """
        # find number value in the file name
        self.bpm = int(re.search(r'\d+', self.file_name).group())
        file_length = self.audio.size / self.fs

        # Calculate the time for each beat
        beat_time = 60 / self.bpm
        beat_times = np.arange(0, file_length, beat_time)

        # Separate Downbeats and Beats
        downbeat_times = beat_times[::self.time_signature[0]]
        beat_times = beat_times[np.where(np.isin(beat_times, downbeat_times) is False)]

        return beat_times, downbeat_times

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

    def _get_note_times(self):
        """ Get 16th Note Grid from Tempo.

        Parameters
        ----------
        self.downbeat_times : np.ndarray
            Array of downbeat times in seconds.
        self.beats_per_bar : int
            Number of beats per bar.

        Returns
        --------
        note_times : np.ndarray
            Array of note times in seconds.
        """
        # Compute note grid from downbeat to downbeat
        note_times = [
            np.linspace(self.downbeat_times[i - 1], self.downbeat_times[i], self.beats_per_bar, endpoint=False)
            for i in range(1, len(self.downbeat_times))
        ]

        return np.array(note_times).flatten()

    def _estimate_onset_stats(self):
        """ Estimate the onset timings and strengths of an audio signal.

        Returns
        -------
        onset_times : 1-d np.array
            Array of time stamps of the estimated onsets in seconds.

        micro_timings : 1-d np.array
            Array of microtiming deviations from grid in seconds.

        onset_loudnesses : 1-d np.array
            Array of onset loudness values.
            A-Weighted Spectral Power (Normalised for File)

        """
        # Compute Onset Statistics
        onset_times = []
        micro_timings = []
        onset_loudnesses = []

        # Get Onsets close to Note Times
        for i in range(self.note_times.size):
            # Get Adjacent Beat Markers
            current_beat_pos = int(np.clip(self.note_times[i], 0, self.audio.size) * self.fs)
            # Get Previous Beat Marker
            if i != 0:
                prev_beat_marker = np.clip(self.note_times[i - 1], 0, self.audio.size) * self.fs
            else:
                prev_beat_marker = 0
            # Get Next Beat Marker
            if i < self.note_times.size - 1:
                next_beat = np.clip(self.note_times[i + 1], 0, self.audio.size) * self.fs
            else:
                next_beat = current_beat_pos

            # Get Region around Beat Marker
            offset_left = int((current_beat_pos - prev_beat_marker) // 2)
            offset_right = int((next_beat - current_beat_pos) // 2)

            # Get Audio Segment
            audio_segment = self.audio[current_beat_pos - offset_left: current_beat_pos + offset_right]

            # Compute Beat Focus Window
            win_left = np.sin(0.5 * np.pi * np.arange(offset_left) / offset_left)
            win_right = np.sin(0.5 * np.pi * np.arange(offset_right) / offset_right + (0.5 * np.pi))
            win = np.concatenate((win_left, win_right))

            # Apply Window to Focus on Proximity to current Beat
            audio_segment = audio_segment * win
            onset_loudness = dsp.compute_loudness(audio_segment, self.fs)

            # Get Spectral Novelty Function
            spectral_novelty = dsp.spectral_novelty(audio_segment)
            spectral_onset = int(np.argmax(spectral_novelty) / spectral_novelty.size * audio_segment.size)
            center_of_mass = int(dsp.center_of_mass(audio_segment))

            # Define Onset as the average of Center of Mass and Spectral Novelty Max
            onset = int(np.average([spectral_onset, center_of_mass]))

            # Compute difference between onset and current note time
            microtiming = onset - offset_left
            microtiming_seconds = microtiming / self.fs
            # Convert onset sample index to time in seconds (and difference to note time)
            onset_time = microtiming_seconds + self.note_times[i]

            # Collect Statistics
            onset_times.append(onset_time)
            micro_timings.append(microtiming_seconds)
            onset_loudnesses.append(onset_loudness)

            # Plot Frame
            # pf.plot_onset_stats_on_frame(audio_segment, win, center_of_mass, spectral_onset, offset_left)

        # Normalise Loudness for file
        onset_loudnesses = dsp.normalise(onset_loudnesses)

        return onset_times, micro_timings, onset_loudnesses

    def _collect_features(self):
        """ Make a dictionary of rhythm features.
        Return
        ------
            features : dict
                Dictionary of rhythm features.
        """
        # Construct Dictionary (no ndarrays)
        features = {"tempo": self.bpm,
                    "downbeat_times": list(np.round(self.downbeat_times, decimals=4)),
                    "beat_times": list(np.round(self.beat_times, decimals=4)),
                    "onset_times": list(self.onset_times),
                    "microtiming": list(self.microtimings),
                    "onset_loudness": list(self.onset_loudnesses)
                    }

        return features

    def _get_onset_energy(self, onset_times: np.ndarray):
        """ Compute the onset strength of an audio signal using RMS and Onset Times."""
        # Get Onset Strength
        audio_rms = dsp.rms_energy(self.audio, int(self.fs / 100))
        onset_energy = dsp.normalise(audio_rms[[int(i * 100) for i in onset_times]])
        return onset_energy

    def _onset_strength_madmom(self, onset_times: np.ndarray) -> np.ndarray:
        """ Compute the onset strength of an audio signal using madmom.

        Parameters
        ----------
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

    def _estimate_tempo_madmom(self) -> float:
        """ Estimate the tempo of an audio signal using madmom.

        Returns
        -------
        tempo : float
            Estimated tempo in BPM.
        """
        proc = madmom.features.tempo.TempoEstimationProcessor(fps=100)
        tempo = proc(self.audio)
        return tempo

    def _sonify_onsets(self):
        """ Sonify the onsets of an audio signal.

        Returns
        -------
        onsets : np.ndarray
            Array of audio samples with sonified onsets.
        """
        # Generate Click Sound
        click_length = 0.02  # 20ms
        click = dsp.generate_click_sound(click_length=click_length)

        # Create Onset Signal
        onsets = np.zeros_like(self.audio)
        onset_samples = dsp.timestamps_to_samples(self.onset_times, self.fs)
        for i, onset in enumerate(onset_samples):
            onsets[onset:onset + len(click)] = self.onset_loudnesses[i] * click

        # Mix with Original Audio
        sonification = onsets + self.audio * 0.5
        # Normalise
        sonification = dsp.normalise(sonification)

        return sonification


if __name__ == '__main__':
    import main
    main.main()

