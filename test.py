import os
import json
import base64
import zipfile
import tempfile
from pathlib import Path
import numpy as np
import soundfile as sf
import shutil
import boto3
import io
import torch
import traceback
from datetime import datetime

# Demucs imports
from demucs.pretrained import get_model
from demucs.audio import AudioFile
from demucs.apply import apply_model

# Global variable to hold the model - will be loaded just once per container
DEMUCS_MODEL = None


def get_cached_model():
    """Get the globally cached model or load it if not available"""
    global DEMUCS_MODEL

    if DEMUCS_MODEL is not None:
        print("Using already loaded model from global variable")
        return DEMUCS_MODEL

    print("Loading model for the first time in this container")
    # Set expected model path - should be pre-loaded in the Docker image
    model_path = os.environ.get("MODEL_CACHE_DIR", "/app/model_cache")

    # Try to use the pre-loaded model
    try:
        # Inform demucs where to find the model
        torch.hub.set_dir(model_path)
        model = get_model("htdemucs")
        model.eval()

        # Move to CUDA if available
        if torch.cuda.is_available():
            print("CUDA available - moving model to GPU")
            model.cuda()

        # Cache the model in the global variable
        DEMUCS_MODEL = model
        print("Model loaded successfully and cached in global variable")
        return model
    except Exception as e:
        print(f"Error loading pre-cached model: {str(e)}")
        # Fall back to standard loading
        print("Falling back to standard model loading")
        model = get_model("htdemucs")
        model.eval()
        if torch.cuda.is_available():
            model.cuda()
        DEMUCS_MODEL = model
        return model


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


def process_audio(temp_input_file, temp_dir_path):
    """Process the audio with the cached model"""
    try:
        print(f"Processing {temp_input_file} with cached model")

        # Get the pre-loaded model
        model = get_cached_model()

        # Load audio
        wav = AudioFile(temp_input_file).read(streams=0, samplerate=44100, channels=2)
        ref = wav.mean(0)
        wav = (wav - ref.mean()) / ref.std()

        # Apply the model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Processing audio on {device}")
        sources = apply_model(model, wav[None], device=device)[0]
        sources = sources * ref.std() + ref.mean()

        # Get the list of sources
        source_names = model.sources

        # Create directory for each source
        stem_dir = temp_dir_path / "htdemucs" / os.path.basename(temp_input_file).replace(".wav", "")
        stem_dir.mkdir(parents=True, exist_ok=True)

        # Save each source
        for i, source_name in enumerate(source_names):
            source_path = stem_dir / f"{source_name}.wav"
            sf.write(source_path, sources[i].cpu().numpy().T, 44100, subtype='FLOAT')
            print(f"Saved {source_name} to {source_path}")

        print("Model processing complete")
        return True
    except Exception as e:
        print(f"Error processing with cached model: {str(e)}")
        traceback_str = traceback.format_exc()
        print(f"Traceback: {traceback_str}")
        return False


def run_additional_function(input_file_name, input_file, temp_dir_path, final_output_dir):
    # This will hold paths to the drums, bass, and vocals stems
    selected_stems_files = []

    # Adjusted for loop to append selected stems paths
    for stem_file in temp_dir_path.glob("**/*.wav"):
        if "temp" in stem_file.stem:
            continue

        # Get stem type based on the filename
        stem_type = ""
        if stem_file.name.endswith('.wav'):
            # Try to extract stem type from filename
            stem_name = stem_file.name[:-4]  # Remove .wav extension
            # Check common stem names
            if any(keyword in stem_name.lower() for keyword in ['drum', 'drums']):
                stem_type = 'drums'
            elif any(keyword in stem_name.lower() for keyword in ['bass']):
                stem_type = 'bass'
            elif any(keyword in stem_name.lower() for keyword in ['vocal', 'vocals']):
                stem_type = 'vocals'
            elif any(keyword in stem_name.lower() for keyword in ['other']):
                stem_type = 'other'
            else:
                # Try checking parent directory
                parent_dir = stem_file.parent.name.lower()
                if parent_dir in ['drums', 'bass', 'vocals', 'other']:
                    stem_type = parent_dir

        if not stem_type:
            print(f"Warning: Could not determine stem type for {stem_file}, skipping")
            continue

        print(f"Processing stem: {stem_file}, identified type: {stem_type}")

        # Create new filename based on the stem type
        new_filename = f"{input_file_name} {stem_type.capitalize()}.wav"
        new_file_path = Path(final_output_dir) / new_filename

        adjust_volume_and_save(stem_file, 10, new_file_path)
        if stem_type.lower() in ['drums', 'bass', 'vocals']:
            selected_stems_files.append(new_file_path)

            # Create "EE" track if we have stems to work with
    if selected_stems_files:
        ee_output_file_path = Path(final_output_dir) / f"{input_file_name} EE.wav"
        invert_phase_and_mix(input_file, selected_stems_files, ee_output_file_path)

        # Additional step: Delete the "other" stem, if it exists
        other_stem_path = Path(final_output_dir) / f"{input_file_name} Other.wav"
        if other_stem_path.exists():
            other_stem_path.unlink()
            print(f"Deleted 'Other' stem: {other_stem_path}")
    else:
        print("Warning: No stems found for EE track creation")

    try:
        # Clean up temp directory
        shutil.rmtree(temp_dir_path)
    except PermissionError as e:
        # Handle PermissionError
        print("PermissionError:", e)


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

    # Adjust volume before processing
    adjust_volume_and_save(converted_input_path, -10, temp_input_file)
    print(f"Prepared temporary file at {temp_input_file}")

    # Process with our cached model
    process_success = process_audio(temp_input_file, temp_dir_path)

    if not process_success:
        print("Warning: Model processing failed. Falling back to demucs.separate.main")
        # Fall back to original method if needed
        import demucs.separate
        import traceback
        try:
            device_arg = "cuda" if torch.cuda.is_available() else "cpu"
            demucs.separate.main(["--float32", "-d", device_arg, "--out", str(temp_dir_path), str(temp_input_file)])
        except Exception as e:
            print(f"Error in fallback processing: {str(e)}")
            traceback_str = traceback.format_exc()
            print(f"Traceback: {traceback_str}")

    # Process the stems
    start_next_function(uploaded_file_name, converted_input_path, temp_dir_path, final_output_dir)

    print("Finished processing")
    return final_output_dir


def start_next_function(input_file_name, input_file, temp_dir_path, final_output_dir):
    print("Next function started")
    run_additional_function(input_file_name, input_file, temp_dir_path, final_output_dir)


def predict(**inputs):
    """The main entry point function that processes an audio file"""

    # Import traceback for better error reporting
    import traceback

    # Start timing the process
    start_time = datetime.now()
    print(f"Starting processing at {start_time}")

    try:
        # Get the filename from inputs
        file_name = inputs['file_name']

        # Setup directories
        current_dir = "/tmp"
        media_path = os.path.join(current_dir, 'uploads')
        if not os.path.exists(media_path):
            os.makedirs(media_path)
        input_file_path = os.path.join(media_path, file_name)

        # Download file from S3
        print(f"Downloading file {file_name} from S3")
        session = boto3.Session(
            aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
            aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY']
        )
        s3 = session.client('s3')

        s3.download_file('us-audio-bucket-2', file_name, input_file_path)
        s3.delete_object(Bucket='us-audio-bucket-2', Key=file_name)
        print("File downloaded and deleted from S3")

        # Process the audio file
        processed_files_path = split_audio(uploaded_file_name=file_name)
        print(f"Processing complete, files saved to {processed_files_path}")

        # Create zip file
        name, extension = os.path.splitext(file_name)
        print(f"Creating zip file for {name}")

        # Create a BytesIO buffer
        buffer = io.BytesIO()

        with zipfile.ZipFile(buffer, 'w') as zip_file:
            for output_file_name in os.listdir(processed_files_path):
                if output_file_name.endswith('.wav'):  # Only include WAV files
                    file_path = os.path.join(processed_files_path, output_file_name)
                    print(f"Adding to zip: {output_file_name}")
                    zip_file.write(file_path, arcname=output_file_name)

        buffer.seek(0)

        # Upload zip to S3
        print(f"Uploading zip file to S3 as {name}.zip")
        s3.put_object(Bucket='us-audio-bucket-2', Key=f"{name}.zip", Body=buffer.getvalue())

        # Finish timing
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        print(f"Processing completed at {end_time}, took {duration} seconds")

        # Return result
        json_data = {
            'file_name': f"{name}.zip",
        }

        print('All tasks finished successfully')
        return json_data

    except Exception as e:
        print(f"Error in predict function: {str(e)}")
        traceback_str = traceback.format_exc()
        print(f"Traceback: {traceback_str}")

        # Return error response
        return {
            'file_name': f"{inputs.get('file_name', 'unknown').split('.')[0]}.zip",
            'error': str(e)
        }


# This enables the file to be run locally for testing if needed
if __name__ == "__main__":
    # Test code could go here
    print("This module is designed to be imported by deploy.py, not run directly.")