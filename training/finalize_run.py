import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from training.utils_s3 import upload_model, upload_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Local path to the .zip model file")
    parser.add_argument("--run_name", required=True, help="Name for this run, e.g. 'ppo_curriculum_n2_finetuned'")
    parser.add_argument("--success_rate", type=float, required=True)
    parser.add_argument("--mean_reward", type=float, required=True)
    parser.add_argument("--notes", type=str, default="")
    args = parser.parse_args()

    upload_model(args.model_path, args.run_name)
    upload_metrics(
        {
            "success_rate": args.success_rate,
            "mean_reward": args.mean_reward,
            "notes": args.notes,
        },
        args.run_name,
    )


if __name__ == "__main__":
    main()