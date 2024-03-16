import music21
import json
import numpy as np

""" Rhythm Feature to Midi Conversion Functions """


def construct_midi(features):
    """ Construct MIDI File from Rhythm Features"""
    # Convert Features to MIDI Data
    midi_times, midi_velocities, midi_pitches = convert_features_to_midi_data(features)

    # Create new MIDI Object
    midi_obj = create_midi_object(midi_times, midi_velocities, midi_pitches, features["tempo"])

    return midi_obj


def write_midi_file(output_path: str, midi_obj: object = None):
    """
    Write MIDI Object to File
    Parameters
    ----------
        output_path (str): Path to File to Write
        midi (object): Music21 Midi Object to Write
    Returns
    -------
        None: Function writes MIDI File to Path
    """

    # Save new MIDI file
    midi_obj.write('midi', output_path)


def convert_features_to_midi_data(features):
    """ Generate MIDI Data based on Features

    Parameters
    ----------
    features : dict
        Dictionary with Rhythm Features
    Returns
    -------
    midi_note_times : list
        List of Note Times
    midi_velocities : list
        List of Note Velocities
    new_note_pitches : list
        List of Note Pitches

    """

    # Derive Downbeats and Bar Length
    downbeats = features['downbeat_times']

    # Convert Downbeats to Beat Measures
    n_bars = len(downbeats)
    len_bar = np.average(np.diff(downbeats))
    converting_factor = 4 / len_bar

    # Convert Microtiming in seconds to Beat-Times
    microtiming = features['microtiming']
    microtiming_beat = [time * converting_factor for time in microtiming]

    # Generate Quantized Notes
    quantized_notes = np.arange(len(microtiming_beat) * 0.5, step=0.5)

    # Displace Note Times by microtiming offset
    midi_note_times = [
        np.clip(time + microtiming_beat[i], a_min=0, a_max=None) for i, time in enumerate(quantized_notes)
    ]

    # Determine Velocities by Loudness
    midi_velocities = [convert_loudness_to_velocity(loudness) for loudness in features['onset_loudness']]

    # Define Pitch and Repeat 8 times for each bar
    midi_pitches = [60] * 8 * (n_bars - 1)

    return midi_note_times, midi_velocities, midi_pitches


def create_midi_object(note_times: list, note_velocities: list, note_pitches: list, bpm: int = 120):
    """ Create MIDI File

    Parameters
    ----------
    note_times : list
        List of Note Times
    note_velocities : list
        List of Note Velocities
    note_pitches : list
        List of Note Pitches
    bpm : int
        Beats per Minute

    Returns
    -------
    new_midi : music21.stream.Stream
        New MIDI File
    """
    # Create new MIDI Object
    new_midi = music21.stream.Stream()

    # Add tempo information
    tempo_bpm = music21.tempo.MetronomeMark(number=bpm)
    new_midi.insert(0, tempo_bpm)

    # Iterate through Note Information
    for time, velocity, pitch in zip(note_times, note_velocities, note_pitches):
        # Create new note
        new_note = music21.note.Note(pitch=pitch, quarterLength=0.25)
        # Change note velocity
        new_note.volume.velocity = velocity
        # Add note to new MIDI file
        new_midi.insert(time, new_note)

    return new_midi


def convert_loudness_to_velocity(loudness):
    """ Convert Loudness to Velocity"""
    return int(np.clip((loudness**0.45) * 127, 0, 127))


def load_feautres_from_json(file_path):
    """ Load Features from File """
    with open(file_path, 'r') as file:
        features = json.load(file)
    return features


def apply_features_to_midi(features_path, midi_input_path, midi_output_path):
    """ Apply Features to given Midi File

    Parameters
    ----------
    features_path : str
        Path to Features File
    midi_input_path : str
        Path to Input MIDI File
    midi_output_path : str
        Path to Output MIDI File

    Returns
    -------
    None: Function writes MIDI File to Path
    """

    # Load the MIDI file
    file_name = f'{midi_input_path.split(".")[0]}'
    midi = music21.converter.parse(midi_input_path)

    # Get Note Times and Velocities and pitches
    new_note_times = []
    note_velocities = []
    note_pitches = []
    for element in midi.recurse():
        # Split Barline into Channels for each note
        if isinstance(element, music21.note.Note):
            new_note_times.append(element.offset)
            note_velocities.append(element.volume.velocity)
            note_pitches.append(element.pitch.midi)

    # Load Features from File
    features = load_feautres_from_json(features_path)

    # Construct Midi
    new_midi = construct_midi(features)

    # Write Midi File
    output_path = f'{midi_output_path}/{file_name}.mid'
    write_midi_file(output_path, new_midi)
    print(f"MIDI File written to: {output_path}")
