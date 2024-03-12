from featureprocessor import FeatureProcessor

import utils as u

"""
------------------------------ PREPROCESSING SCRIPT -----------------------------

Author:     Robin Doerfler
Project:    MIR Course 2024 - Rhythm Style Transfer 

-----------------------------------------------------------------------------------
"""


def main():
    """
    ---- Main Process ----
    Get Path and File
    Iterate through Files
        - Load Audio Data
        - Extract Rhythm Features
        - Encode Rhythm Features
    """
    # Get Path and Files
    base_path, file_names = u.get_base_path()

    # Preprocess each file
    for file_name in file_names:

        # ---- Create Output Directory ----
        output_dir = u.create_output_directories(file_name)

        # ---- Extraction ----
        features_proc = FeatureProcessor(base_path, file_name)
        features = features_proc.extract_rhythm_features()
        features_proc.write_rhythm_features_plot(output_dir)

        # ---- Save Data ----
        u.write_json(features, file_name, output_dir)

    print("Preprocessing Complete")
    exit()


if __name__ == '__main__':
    main()
