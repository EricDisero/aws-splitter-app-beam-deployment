from beam import Image, endpoint, QueueDepthAutoscaler
import traceback

autoscaler = QueueDepthAutoscaler(
    tasks_per_container=1,
    max_containers=5
)


@endpoint(
    name="demucs-analysis",
    autoscaler=autoscaler,
    cpu=1,
    memory="8Gi",
    gpu="RTX4090",
    keep_warm_seconds=0,  # Shut down container immediately after task finishes
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
            "boto3",
            "demucs",
            "requests"
        ],
    ),
)
def predict(**inputs):
    """
    Proxy function that calls the actual implementation in test.py
    """
    import importlib.util
    import sys
    import os

    try:
        # Set up environment variable for model caching
        os.environ["MODEL_CACHE_DIR"] = "/tmp/model_cache"
        os.makedirs("/tmp/model_cache", exist_ok=True)

        # Log some diagnostic information
        print(f"Current working directory: {os.getcwd()}")
        print(f"Files in directory: {os.listdir()}")

        # Import the test.py module
        spec = importlib.util.spec_from_file_location("test", "test.py")
        test_module = importlib.util.module_from_spec(spec)
        sys.modules["test"] = test_module
        spec.loader.exec_module(test_module)

        print("Successfully imported test.py")

        # Call the predict function in test.py
        result = test_module.predict(**inputs)
        print(f"Result from test.py: {result}")
        return result

    except Exception as e:
        print(f"Error in proxy function: {str(e)}")
        trace = traceback.format_exc()
        print(f"Traceback: {trace}")

        # Provide a fallback that matches the expected output format
        return {
            "file_name": f"{inputs.get('file_name', '').split('.')[0]}.zip",
            "error": str(e)
        }