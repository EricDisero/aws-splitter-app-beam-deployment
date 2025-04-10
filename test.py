from beam import Image, Output, endpoint, QueueDepthAutoscaler
import os
import requests
import json
import base64
import zipfile
import tempfile
from pathlib import Path
import numpy as np
import soundfile as sf
import shutil
import demucs.separate
import boto3
import io

autoscaler = QueueDepthAutoscaler(
  tasks_per_container=1,
  max_containers=5
)

def convert_to_44100hz(input_file, output_file=None):
    """
    Convert any audio file to 44.1kHz sample rate using soundfile

    Args:
        input_file (str): Path to input audio file
        output_file (str, optional): Path to output file. If None, creates a new file.

    Returns:
        str: Path to the converted file
    """
    try:
        # Check if file needs conversion
        info = sf.info(input_file)

        # If already 44.1kHz, just return the original file path
        if abs(info.samplerate - 44100) < 100:  # Allow small tolerance
            print(f"File {input_file} already at 44.1kHz, skipping conversion")
            return input_file

        if output_file is None:
            # Create new file in same directory
            dir_name = os.path.dirname(input_file)
            base_name = os.path.basename(input_file)
            output_file = os.path.join(dir_name, f"44k_{base_name}")

        print(f"Converting {input_file} from {info.samplerate}Hz to 44.1kHz")

        # Read the audio data
        data, samplerate = sf.read(input_file)

        # Write with new samplerate (soundfile does the resampling automatically)
        sf.write(output_file, data, 44100, subtype='PCM_24')

        print(f"Successfully converted to {output_file}")
        return output_file

    except Exception as e:
        print(f"Error converting sample rate: {str(e)}")
        # Return original file if conversion fails
        return input_file


def invert_phase_and_mix(input_file_path, stems_files, output_file_path):
    """
    Inverts the phase of summed stems and mixes them with the original input file.
    Handles potential size mismatches between original and processed files.
    """
    # Read the original input file
    original_data, samplerate = sf.read(input_file_path)

    # Initialize a numpy array for the sum of selected stems
    summed_stems_data = None

    # Sum the selected stems
    for stem_file in stems_files:
        data, _ = sf.read(stem_file)
        if summed_stems_data is None:
            summed_stems_data = np.zeros_like(data)
        summed_stems_data += data

    # Invert phase of the summed stems
    inverted_stems_data = summed_stems_data * -1

    # Check for shape mismatch and handle it
    if original_data.shape != inverted_stems_data.shape:
        print(f"Shape mismatch detected: original={original_data.shape}, stems={inverted_stems_data.shape}")

        # Option 1: Resize to the smaller of the two lengths
        min_length = min(original_data.shape[0], inverted_stems_data.shape[0])
        original_data = original_data[:min_length]
        inverted_stems_data = inverted_stems_data[:min_length]
        print(f"Resized both arrays to length {min_length}")

    # Mix inverted phase stems with the original track
    try:
        mixed_data = original_data + inverted_stems_data
        # Write the result to the output file
        sf.write(output_file_path, mixed_data, samplerate, subtype='FLOAT')
        print(f"Mixed and saved EE track to {output_file_path}")
        error_occurred = False
    except ValueError as e:
        print(f"Error mixing audio: {str(e)}")
        print(f"Stems files: {stems_files}")
        print(f"Inverted stems shape: {inverted_stems_data.shape}")
        print(f"Original data shape: {original_data.shape}")
        error_occurred = True

    return error_occurred


def adjust_volume_and_save(input_file, db_change, output_file):
    data, samplerate = sf.read(input_file)
    factor = np.power(10, db_change / 20)
    adjusted_data = data * factor
    sf.write(output_file, adjusted_data, samplerate, subtype='FLOAT')
    print(f"Audio volume adjusted by {db_change}dB and saved to {output_file} in 32-bit float format.")


def run_additional_function(input_file_name, input_file, temp_dir_path, final_output_dir):
    # This will hold paths to the drums, bass, and vocals stems
    selected_stems_files = [] 
        
    # Adjusted for loop to append selected stems paths
    for stem_file in temp_dir_path.glob("**/*.wav"):
        stem_type = stem_file.stem.split('_')[-1]
        if "temp" in stem_file.stem:
            continue
        new_filename = f"{input_file_name} {stem_type.capitalize()}.wav"
        new_file_path = Path(final_output_dir) / new_filename
        adjust_volume_and_save(stem_file, 10, new_file_path)
        if stem_type in ['drums', 'bass', 'vocals']:
            selected_stems_files.append(new_file_path)      

    # Create "EE" track
    ee_output_file_path = Path(final_output_dir) / f"{input_file_name} EE.wav"
    invert_phase_and_mix(input_file, selected_stems_files, ee_output_file_path)

    # Additional step: Delete the "other" stem, if it exists
    other_stem_path = Path(final_output_dir) / f"{input_file_name} Other.wav"
    if other_stem_path.exists():
        other_stem_path.unlink()
        print(f"Deleted 'Other' stem: {other_stem_path}")
    


    try:
        # Code that may raise a PermissionError
        shutil.rmtree(temp_dir_path)  
    except PermissionError as e:
        # Handle PermissionError
        print("PermissionError:", e)
        # Optionally, perform error handling actions such as logging, notifying the user, etc.    

def deserialize_zip_file(json_data):
    # Extract filename and base64 encoded content from JSON data
    zip_file_path = json_data['file_name']

    current_dir = "/tmp"
    media_path = os.path.join(current_dir, 'uploads')
    if not os.path.exists(media_path):
        os.makedirs(media_path)
    input_file_path = os.path.join(media_path, zip_file_path)
    base64_encoded_content = json_data['base64_encoded_content']

    # Decode the base64 encoded content
    decoded_content = base64.b64decode(base64_encoded_content)

    # Write the decoded content to a new file
    with open(input_file_path, 'wb') as f:
        f.write(decoded_content)


def split_audio(uploaded_file_name) -> str:
    current_dir = "/tmp"
    media_path = os.path.join(current_dir, 'uploads')
    input_file_path = os.path.join(media_path, uploaded_file_name)
    final_output_dir = os.path.join(current_dir, "downloads", f"{uploaded_file_name}-Stems")

    # Check output already exists
    try:
        os.makedirs(final_output_dir)
    except FileExistsError:
        pass

        # First convert to 44.1kHz if needed
    try:
        print(f"Checking sample rate of {input_file_path}")
        converted_input_path = convert_to_44100hz(input_file_path)
        print(f"Using file {converted_input_path} for processing")
    except Exception as e:
        print(f"Error during sample rate conversion: {str(e)}")
        # Fall back to original file
        converted_input_path = input_file_path

    temp_dir_path = Path(tempfile.mkdtemp(dir=final_output_dir))
    temp_input_file = os.path.join(temp_dir_path, f"{os.path.basename(converted_input_path)}_temp.wav")

    # Adjust volume on the converted file
    adjust_volume_and_save(converted_input_path, -10, temp_input_file)
    print(temp_input_file)

    # Run your subprocess command here
    demucs.separate.main(["--float32", "-d", "cuda", "--out", str(temp_dir_path), str(temp_input_file)])

    # Simulating the work done by the thread
    print(f"Processing {temp_input_file} in {temp_dir_path}")

    # Call the callback function if it exists
    start_next_function(uploaded_file_name, converted_input_path, temp_dir_path, final_output_dir)

    print("Finished")

    return final_output_dir

def start_next_function(input_file_name, input_file, temp_dir_path, final_output_dir):
    print("Next function started")
    run_additional_function(input_file_name, input_file, temp_dir_path, final_output_dir)


def serialize_zip_file(zip_file_path):

    base64_zip_contents = base64.b64encode(zip_file_path).decode('utf-8')

    json_data = {
        'zip_file_name': 'zip_file_name',
        'base64_encoded_content': base64_zip_contents
    }

    return json_data

@endpoint(
    name="demucs-analysis",
    autoscaler=autoscaler,
    cpu=1,
    memory="8Gi",
    gpu="RTX4090",  # Changed from "T4" to "RTX_4090"
    keep_warm_seconds=0,  # ✅ Shut down container immediately after task finishes
    secrets=['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY'],
    image=Image(
        python_version="python3.9",
        python_packages=[
            "antlr4-python3-runtime",
            "asgiref",
            "cffi",
            "cloudpickle",
            "colorama",
            "dora_search",
            "einops",
            "filelock",
            "fsspec",
            "julius",
            "lameenc",
            "MarkupSafe",
            "mpmath",
            "networkx",
            "numpy",
            "omegaconf",
            "openunmix",
            "pathlib",
            "pycparser",
            "PyYAML",
            "retrying",
            "six",
            "soundfile",
            "sqlparse",
            "submitit",
            "sympy",
            "torch",
            "torchaudio",
            "tqdm",
            "treetable",
            "typing_extensions",
            "tzdata",
            "boto3"
        ],  # You can also add a path to a requirements.txt instead
    ),
)
def predict(**inputs):

    file_name = inputs['file_name']
    
    current_dir = "/tmp"
    media_path = os.path.join(current_dir, 'uploads')
    if not os.path.exists(media_path):
        os.makedirs(media_path)
    input_file_path = os.path.join(media_path, file_name)

    session = boto3.Session(
                aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
                aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY']
            )
    s3 = session.client('s3')

    s3.download_file('us-audio-bucket-2', file_name, input_file_path)

    s3.delete_object(Bucket='us-audio-bucket-2', Key=file_name)

    processed_files_path = split_audio(uploaded_file_name=file_name)

    print(processed_files_path)

    files = os.listdir(processed_files_path)
    name, extension = os.path.splitext(file_name)
    
    # Create a BytesIO buffer
    buffer = io.BytesIO()
    
    with zipfile.ZipFile(buffer, 'w') as zip_file:
        for file_name in files:
            print(file_name)
            file_path = os.path.join(processed_files_path, file_name)
            zip_file.write(file_path, arcname=file_name)

    buffer.seek(0)

    s3.put_object(Bucket='us-audio-bucket-2', Key=f"{name}.zip", Body=buffer.getvalue())


    json_data = {
            'file_name': f"{name}.zip",
        }

    print('Finished')

    return json_data

