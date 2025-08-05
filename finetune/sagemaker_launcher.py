import os
import sagemaker
from sagemaker.pytorch import PyTorch
from dotenv import load_dotenv
load_dotenv()

ROLE_ARN = os.getenv("SAGEMAKER_ROLE_ARN")
BUCKET = os.getenv("SAGEMAKER_BUCKET")
REGION = os.getenv("AWS_REGION", "us-east-1")
INSTANCE_TYPE = os.getenv("INSTANCE_TYPE", "ml.p4d.24xlarge")
INSTANCE_COUNT = int(os.getenv("INSTANCE_COUNT", "1"))

if not all([ROLE_ARN, BUCKET]):
    raise RuntimeError("Set SAGEMAKER_ROLE_ARN and SAGEMAKER_BUCKET in your environment or .env")

session = sagemaker.Session()
print(f"Using bucket: {BUCKET}, region: {REGION}")

# This assumes you run from the finetune directory.
source_dir = "."
entry_point = "train_unsloth_sft.py"

# We’ll run a small bootstrap at the start of training to pip install requirements.
# You can add this snippet at the top of train_unsloth_sft.py. If not, use a wrapper.
# Simpler approach: use "dependencies" feature or pass a shell cmd via training script.
# For PyTorch estimator, we'll pass requirements.txt as part of source_dir (same folder).

estimator = PyTorch(
    entry_point=entry_point,
    source_dir=source_dir,
    role=ROLE_ARN,
    framework_version="2.3.0",
    py_version="py311",
    instance_type=INSTANCE_TYPE,
    instance_count=INSTANCE_COUNT,
    hyperparameters={
        "BASE_MODEL": "google/gemma3n:e4b",   # replace with gemma3n:e4b repo path if available
        "OUTPUT_DIR": "/opt/ml/model",
        "TRAIN_PATH": "mn_dataset_out_v6_trust/clean/train_chains.jsonl",
        "DEV_PATH": "mn_dataset_out_v6_trust/clean/dev_chains.jsonl",
        "EPOCHS": "2.0",
        "BATCH_SIZE": "1",
        "GRAD_ACCUM": "16",
        "LR": "2e-5",
        "USE_QLORA": "true",
        "MAX_SEQ_LEN": "2048",
        "INSTALL_REQS": "true"  # custom flag to pip install -r requirements.txt
    },
    disable_profiler=True,
    debugger_hook_config=False,
    environment={
        "AWS_REGION": REGION,
        "SAGEMAKER_BUCKET": BUCKET,
        # Optional keys for RL later
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY", ""),
        "WANDB_API_KEY": os.getenv("WANDB_API_KEY", ""),
        "WANDB_PROJECT": os.getenv("WANDB_PROJECT", "mr_delight_rl"),

        "TE_DISABLE": "1",
        "ACCELERATE_DISABLE_TE": "1",
        # optional: set to "1" if you bake a custom image later and want to skip bootstrap
        # "BOOTSTRAPPED_UNSLOTH": "1",
    },
)

estimator.fit(wait=True)
print("SFT completed. Model artifacts saved to S3 in the job output path.")