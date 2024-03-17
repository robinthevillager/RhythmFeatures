import utils as u
from featureprocessor import FeatureProcessor


"""
--------------------------- PREPROCESSING SCRIPT ---------------------------

Author:     Robin Doerfler
Project:    MIR Course 2024 - Rhythm Style Transfer 

----------------------------------------------------------------------------
"""


def main():
    """ ---- Main Process ---- """
    # Get Path and Files
    base_path, file_names = u.get_base_path()

    # Preprocess each file
    for file_name in file_names:

        # ---- Create Output Directory ----
        output_dir = u.create_output_directories(file_name)

        # ---- Extraction ----
        features_proc = FeatureProcessor(base_path, file_name)
        rhythm_features = features_proc.extract_rhythm_features()
        rhythm_midi = features_proc.construct_midi()

        # ---- Write Data ----
        features_proc.write_analysis_data(output_dir)
        u.write_midi(output_dir, rhythm_midi, file_name)
        u.write_json(output_dir, rhythm_features, file_name)

    print("Preprocessing Complete.")
    exit()


if __name__ == '__main__':
    main()
