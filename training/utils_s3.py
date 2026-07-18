import boto3
import json
from pathlib import Path
from datetime import datetime, timezone

BUCKET_NAME = "bk-multiagent-rl-2026"

_s3 = boto3.client("s3")


def upload_model(local_model_path: str, run_name: str) -> str:
    """Uploads a saved SB3 model (.zip) to S3 under models/<run_name>/.
    Returns the S3 key it was uploaded to.
    """
    local_path = Path(local_model_path)
    if local_path.suffix != ".zip":
        local_path = local_path.with_suffix(".zip")  # SB3's model.save() appends .zip automatically

    s3_key = f"models/{run_name}/{local_path.name}"
    _s3.upload_file(str(local_path), BUCKET_NAME, s3_key)
    print(f"Uploaded model to s3://{BUCKET_NAME}/{s3_key}")
    return s3_key


def upload_metrics(metrics: dict, run_name: str) -> str:
    """Uploads a metrics dict (reward, success rate, etc.) as JSON to
    S3 under logs/<run_name>/metrics.json, tagged with a UTC timestamp.
    """
    metrics_with_meta = {
        **metrics,
        "run_name": run_name,
        "uploaded_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    s3_key = f"logs/{run_name}/metrics.json"
    _s3.put_object(
        Bucket=BUCKET_NAME,
        Key=s3_key,
        Body=json.dumps(metrics_with_meta, indent=2).encode("utf-8"),
    )
    print(f"Uploaded metrics to s3://{BUCKET_NAME}/{s3_key}")
    return s3_key