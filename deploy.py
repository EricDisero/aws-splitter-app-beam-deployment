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
    gpu="T4",
    secrets=['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY'],
    image=Image(
        python_version="python3.9",
        python_packages=[
            "numpy",
            "soundfile",
            "boto3",
            "demucs",
            "torch",
            "torchaudio",
            "requests",
        ],
    ),
)
def predict(**inputs):
    # Will be replaced by actual implementation at runtime
    return {"message": "deployment placeholder"}