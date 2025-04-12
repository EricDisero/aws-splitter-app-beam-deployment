from beam import Image, endpoint, QueueDepthAutoscaler

autoscaler = QueueDepthAutoscaler(
    tasks_per_container=1,
    max_containers=5
)


@endpoint(
    name="demucs-analysis",
    autoscaler=autoscaler,
    cpu=1,
    memory="8Gi",
    gpu="A10G",  # Changed from "T4" to "RTX_4090 to A100-40"
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
            "boto3",
            "demucs",
            "requests"
        ],
    ),
)
def predict(**inputs):
    # This is a proxy function that will call the actual implementation
    # Import here to avoid local dependency requirements
    import importlib.util
    import sys
    import os

    # Dynamically import the test module
    try:
        # Log the current working directory and available files
        print(f"Current working directory: {os.getcwd()}")
        print(f"Files in directory: {os.listdir()}")

        # Import the test.py module and call its predict function
        spec = importlib.util.spec_from_file_location("test", "test.py")
        test_module = importlib.util.module_from_spec(spec)
        sys.modules["test"] = test_module
        spec.loader.exec_module(test_module)

        print("Successfully imported test.py")

        # Call the actual implementation
        result = test_module.predict(**inputs)
        print(f"Result from test.py: {result}")
        return result

    except Exception as e:
        print(f"Error in proxy function: {str(e)}")
        # Provide a fallback that matches the expected output format
        # This helps with debugging while still allowing the request to "succeed"
        return {
            "file_name": f"{inputs.get('file_name', '').split('.')[0]}.zip",
            "error": str(e)
        }