from huggingface_hub import HfApi

import fluidgym

fluidgym.config.update("local_data_path", "./local_data")

api = HfApi()

api.upload_large_folder(
    folder_path=fluidgym.config.local_data_path,
    repo_id=fluidgym.config.hf_intial_domains_repo_id,
    repo_type="dataset",
    ignore_patterns=["*.png"],
)
