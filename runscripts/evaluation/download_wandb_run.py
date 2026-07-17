import wandb
import argparse


def download_run_history(run_path: str, output_path: str):
    api = wandb.Api()
    run = api.run(run_path)
    df = run.history()
    df = df.rename(columns={"_step": "step"})
    df = df.drop(columns=["_runtime", "_timestamp"], errors="ignore")
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_path", help="WandB run path (entity/project/run_id)")
    parser.add_argument("--output", "-o", default="run_history.csv", help="Output CSV path")
    args = parser.parse_args()

    download_run_history(args.run_path, args.output)
