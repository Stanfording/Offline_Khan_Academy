import os, sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.session import Session
from sagemaker.amazon.amazon_estimator import get_image_uri

# AWS config
ROLE_ARN = os.getenv("SAGEMAKER_ROLE_ARN")  # e.g., "arn:aws:iam::123456789012:role/SageMakerRole"
BUCKET = os.getenv("SAGEMAKER_BUCKET")      # your S3 bucket
REGION = os.getenv("AWS_REGION", "us-east-1")
INSTANCE_TYPE = os.getenv("INSTANCE_TYPE", "ml.p4d.24xlarge")  # 8xA100
INSTANCE_COUNT = int(os.getenv("INSTANCE_COUNT", "1"))

session = sagemaker.Session()
account = session.boto_session.client("sts").get_caller_identity()["Account"]
print(f"Using account {account} in {REGION}")

# Upload code to S3
code_path = "train_unsloth_sft.py"
s3_code = sagemaker.s3.S3Uploader.upload(code_path, f"s3://{BUCKET}/mr_delight/code/")

estimator = PyTorch(
    entry_point="train_unsloth_sft.py",
    source_dir=".",
    role=ROLE_ARN,
    framework_version="2.3.0",
    py_version="py310",
    instance_type=INSTANCE_TYPE,
    instance_count=INSTANCE_COUNT,
    hyperparameters={
        "BASE_MODEL": "google/gemma-2-9b-it",   # or your gemma3n:e4b path
        "OUTPUT_DIR": "/opt/ml/model",
        "TRAIN_PATH": "mn_dataset_out_v6_trust/clean/train_chains.jsonl",
        "DEV_PATH": "mn_dataset_out_v6_trust/clean/dev_chains.jsonl",
        "EPOCHS": "2.0",
        "BATCH_SIZE": "1",
        "GRAD_ACCUM": "16",
        "LR": "2e-5",
        "USE_QLORA": "true",
    },
    disable_profiler=True,
    debugger_hook_config=False,
)

# Set volume size and environment
estimator.fit(wait=True)
# Model artifacts saved automatically to S3 under your training job.
print("SFT completed. To run RL, create a similar job with rlhf_gppo_gemma3n.py as entry_point.")