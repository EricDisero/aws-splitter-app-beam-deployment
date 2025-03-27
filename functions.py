import numpy as np
import soundfile as sf
import subprocess
import shutil
from pathlib import Path


def run_demucs(input_file, output_dir):
    command = ['python', '-m', 'demucs.separate', '--float32', '-d', 'cpu', '--out', str(output_dir), str(input_file)]
    subprocess.run(command, check=True)
    print("Demucs processing completed with 32-bit float output.")




