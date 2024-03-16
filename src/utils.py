import os
import json

""" Helper Functions """


def get_base_path():
    """
    Get Path to File from User Input
    Return:
        base_path (str): Path to File
        files (list): List of Files in Path
    """

    # Get Input Path
    # base_path = input("1. Copy File / Folder Path"
    #                   "\n"
    #                   "[option + cmnd + c to copy path of highlightet item on MAC]\n"
    #                   "2. Paste here: ") or "../audiofiles/drumloops"

    # Define Base Path from new Path
    base_path = "../audiofiles/drumloops"

    if not os.path.isfile(base_path):
        # If Path is a Folder
        files = [file for file in os.listdir(base_path) if file.endswith(".wav")]
        base_path = f"{base_path}/"
        print(f"Files in Process: {files}")
    else:
        # If Path is a File
        file = os.path.basename(base_path)
        base_path = f"{base_path[:-len(file)]}"
        files = [file]
        print(f"File in Process: {files}")

    return base_path, files


def define_session_parameters(base_path, file_name):
    # Define Session Parameters:
    session_params = {
        # Define Session Settings
        "base_path": base_path,
        "file_name": file_name,
        # Define Audio Parameters
        "fs": 44100
    }

    return session_params


def increase_number_recursively(name, index=1):
    new_name = f"{name}_{index:02d}"
    # Check if File already exists
    if os.path.exists(new_name):
        # Index of Session
        index += 1
        # Call Function again with new index
        new_name = increase_number_recursively(name, index)
    return new_name


def create_output_directories(file_name):
    """
    Create Output Directories
    Args:
        file_name (str): File Name
    """
    # Create Output Directories
    file_name = file_name.split(".")[0]
    file_name = increase_number_recursively(file_name)
    output_path = f"../output/{file_name}"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    print(f"Output Directory Created: {output_path}")

    return output_path


def write_json(path: str, data: dict, file_name: str):
    """
    Write Data to JSON File
    Args:
        path (str): Path to File
        data (dict): Data to Write
        file_name (str): File Name
    """
    # Define Output Path
    output_path = f'{path}/{file_name.split(".")[0]}.json'

    # Write Data to JSON
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)


def write_midi(path: str, midi_obj: object, file_name: str):
    """ Write Midi Object to Directory """
    from midi_functions import write_midi_file
    # Define Output Path
    output_path = f'{path}/{file_name.split(".")[0]}.mid'
    write_midi_file(output_path, midi_obj)
