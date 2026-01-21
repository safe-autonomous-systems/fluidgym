from huggingface_hub import HfApi

import fluidgym

fluidgym.config.update("local_data_path", "./local_data")

api = HfApi()

api.upload_large_folder(
    folder_path="output",
    repo_id="anonymous-authors/anonymous-experiments",
    repo_type="dataset",
    ignore_patterns=[
        "*.gif",
        "*.pkl",
        "*.npz",
        "*.png",
        "*.pdf",
        "*.log",
        "**/preparation/*",
        "**/wandb/*",
        "hydra.yaml",
        "multirun.yaml",
        "overrides.yaml",
        "**/local_data/**",
        "**/preparation/**",
        "**/tests/**",
    ],
)
