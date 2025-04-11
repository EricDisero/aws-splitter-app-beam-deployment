FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir \
    antlr4-python3-runtime \
    asgiref \
    cffi \
    cloudpickle \
    colorama \
    dora_search \
    einops \
    filelock \
    fsspec \
    julius \
    lameenc \
    MarkupSafe \
    mpmath \
    networkx \
    numpy \
    omegaconf \
    openunmix \
    pathlib \
    pycparser \
    PyYAML \
    retrying \
    six \
    soundfile \
    sqlparse \
    submitit \
    sympy \
    torch \
    torchaudio \
    tqdm \
    treetable \
    typing_extensions \
    tzdata \
    boto3 \
    demucs \
    requests

# Pre-download the model
RUN mkdir -p /app/model_cache && \
    python -c "from demucs.pretrained import get_model; model = get_model('htdemucs')"

# Copy your application code
COPY . .

# Set environment variable to use the pre-cached model
ENV MODEL_CACHE_DIR=/app/model_cache

# Command to run your app
CMD ["python", "deploy.py"]