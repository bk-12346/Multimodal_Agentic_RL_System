from sagemaker.train import ModelTrainer
from sagemaker.train.configs import SourceCode, Compute
from sagemaker.core import image_uris

ROLE_ARN = "arn:aws:iam::555984984896:role/sagemaker-multiagent-rl-role"
BUCKET = "bk-multiagent-rl-2026"
REGION = "us-east-1"

training_image = image_uris.retrieve(
    framework="pytorch",
    region=REGION,
    version="2.1",
    py_version="py310",
    image_scope="training",
    instance_type="ml.m5.large",
)

source_code = SourceCode(
    source_dir="sagemaker_training",
    entry_script="train_script.py",
    requirements="requirements.txt",
)

model_trainer = ModelTrainer(
    training_image=training_image,
    source_code=source_code,
    role=ROLE_ARN,
    compute=Compute(instance_type="ml.m5.large", instance_count=1),
    base_job_name="multiagent-rl-demo",
    hyperparameters={"total-timesteps": "50000", "learning-rate": "0.0003"},
)

model_trainer.train(wait=True)