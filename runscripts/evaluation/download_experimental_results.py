from huggingface_hub import snapshot_download

import fluidgym

fluidgym.config.update("local_data_path", "./local_data")

if __name__ == "__main__":
    snapshot_download(
        repo_id="safe-autonomous-systems/fluidgym-experiments",
        repo_type="dataset",
        local_dir="output",
    )
