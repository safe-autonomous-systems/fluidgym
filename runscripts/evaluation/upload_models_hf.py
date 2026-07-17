import fluidgym
from huggingface_hub import HfApi, create_repo, upload_large_folder, create_collection
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
from moviepy import VideoFileClip
from PIL import Image, ImageSequence
from pathlib import Path

def count_gif_frames_pillow(path: str | Path) -> int:
    with Image.open(path) as im:
        return getattr(im, "n_frames", 1)

def prune_gif_if_long(input_gif_path, max_frames=500, stride=5):
    """
    Checks GIF frame count. If it exceeds max_frames, saves a new version
    taking every 'stride'-th frame to reduce size and complexity.
    """
    gif_path = Path(input_gif_path)
    
    with Image.open(gif_path) as img:
        # Check total frames
        frame_count = count_gif_frames_pillow(gif_path)
        
        if frame_count <= max_frames:
            stride = 1

        # Extract frames based on stride
        frames = [f.copy() for i, f in enumerate(ImageSequence.Iterator(img)) if i % stride == 0]
        
        # Save the pruned version (overwriting or saving as a temp file)
        pruned_path = gif_path.with_name(f"pruned_{gif_path.name}")

        target_fps = 24
        duration = int(1000 / target_fps)  # duration per frame in ms
        
        frames[0].save(
            pruned_path,
            save_all=True,
            append_images=frames[1:],
            optimize=True,
            duration=duration,
            loop=0
        )
        
    return pruned_path

def get_gif_key_name(env_name: str) -> str:
    if "CylinderJet2D" in env_name or "CylinderRot2D" in env_name or "Airfoil2D" in env_name:
        return "vorticity"
    elif "CylinderJet3D" in env_name or "Airfoil3D" in env_name:
        return "3d_vorticity"
    elif "RBC2D" in env_name:
        return "temperature"
    elif "RBC3D" in env_name:
        return "3d_temperature"
    elif "TCF" in env_name:
        return "3d_q_criterion"
    else:
        raise ValueError(f"Unknown env name {env_name} for GIF key name.")


def generate_sb3_readme(env_name: str, algo_name: str, base_path: Path) -> str:
    """
    Generates a professional README for a specific Env/Algo pair.
    Parses CSVs from seed folders and finds a preview GIF.
    """
    stats_rows = []
    all_means = []
    n_seeds = 0
    max_reward = -np.inf
    best_seed_folder = None
    best_gif_name = None
    
    # 1. Process each seed to collect metrics
    for seed in range(5):
        seed_dir = base_path / str(seed)
        test_dir = seed_dir / "test"
        csv_path = test_dir / "test_eval_sequences.csv"

        if not csv_path.exists():
            continue
        n_seeds += 1
        
        df = pd.read_csv(csv_path)
        mean_rew = df["reward"].mean()
        std_rew = df["reward"].std()
        
        stats_rows.append(f"| Seed {seed} | {mean_rew:.2f} | {std_rew:.2f} |")
        all_means.append(mean_rew)

        # 2. Look for the first available GIF for the given seed
        if mean_rew > max_reward:
            gif_key = get_gif_key_name(env_name)
            gif_file_path = test_dir / f"{gif_key}_test_eval_episode_0.gif"
            if gif_file_path.exists():
                max_reward = mean_rew
                best_seed_folder = test_dir
                best_gif_name = gif_file_path.name

    preview_file_name = None
    dest_mp4 = base_path / "replay.mp4"
    if best_seed_folder and best_gif_name and not dest_mp4.exists():
        original_gif = best_seed_folder / best_gif_name        
        processed_gif = prune_gif_if_long(original_gif)
        
        try:
            # Load GIF and write as MP4 using the new API
            clip = VideoFileClip(str(processed_gif))
            
            # Note: v2.0 uses 'codec' but we also ensure no audio
            clip.write_videofile(
                str(dest_mp4), 
                codec="libx264", 
                audio=False
            )
            clip.close()
            
            preview_file_name = "replay.mp4"
            
            # Clean up the pruned temp GIF
            if processed_gif.name.startswith("processed_"):
                processed_gif.unlink()

        except Exception as e:
            print(f"Warning: MP4 conversion failed: {e}. Falling back to GIF.")
            # Fallback logic
            shutil.copy2(processed_gif, base_path / "replay.gif")
            preview_file_name = "replay.gif"

    table_body = "\n".join(stats_rows)

    # Calculate global performance
    global_avg = np.mean(all_means) if all_means else 0
    global_std = np.std(all_means) if all_means else 0

    # 3. Build the Markdown string
    readme_text = f"""---
library_name: stable-baselines3
tags:
- reinforcement-learning
- stable-baselines3
- deep-reinforcement-learning
- fluidgym
- active-flow-control
- fluid-dynamics
- simulation
- {env_name}
model-index:
- name: {algo_name}-{env_name}
  results:
  - task:
      type: reinforcement-learning
      name: reinforcement-learning
    dataset:
      name: FluidGym-{env_name}
      type: fluidgym
    metrics:
    - type: mean_reward
      value: {global_avg:.2f}
      name: mean_reward
{"predict_config:" if preview_file_name else ""}
{"  preview_file: " + preview_file_name if preview_file_name else ""}
---

# {algo_name} on {env_name} (FluidGym)

This repository is part of the **FluidGym** benchmark results. It contains trained Stable Baselines3 agents for the specialized **{env_name}** environment.

## Evaluation Results

### Global Performance (Aggregated across {n_seeds} seeds)
**Mean Reward:** {global_avg:.2f} ± {global_std:.2f}

### Per-Seed Statistics
| Run | Mean Reward | Std Dev |
| --- | --- | --- |
{table_body}

## About FluidGym
FluidGym is a benchmark for reinforcement learning in active flow control.

## Usage
Each seed is contained in its own subdirectory. You can load a model using:
```python
from stable_baselines3 import {algo_name}
model = {algo_name}.load("0/ckpt_latest.zip")

**Important:** The models were trained using ```fluidgym==0.0.2```. In order to use
them with newer versions of FluidGym, you need to wrap the environment with a
`FlattenObservation` wrapper as shown below:
```python
import fluidgym
from fluidgym.wrappers import FlattenObservation
from stable_baselines3 import {algo_name}

env = fluidgym.make("{env_name}")
env = FlattenObservation(env)
model = {algo_name}.load("path_to_model/ckpt_latest.zip")

obs, info = env.reset(seed=42)

action, _ = model.predict(obs, deterministic=True)
obs, reward, terminated, truncated, info = env.step(action)
```

## References

* [Plug-and-Play Benchmarking of Reinforcement Learning Algorithms for Large-Scale Flow Control](http://arxiv.org/abs/2601.15015)
* [FluidGym GitHub Repository](https://github.com/safe-autonomous-systems/fluidgym)
"""
    
    return readme_text

api = HfApi()
username = "safe-autonomous-systems"
envs = list(fluidgym.registry.registry.ids)
algos = ["PPO", "SAC"]
rl_modes = ["sarl", "marl"]
training_path = Path("./output/training")

all_repo_ids = []

for rl_mode in rl_modes:
    for env in envs:
        for algo in algos:
            local_path = training_path / rl_mode / env / algo 
            if not local_path.exists():
                continue

            algo_name = algo
            if rl_mode == "marl":
                algo_name = "MA-" + algo_name

            repo_name = f"{algo_name.lower()}-{env}"
            repo_id = f"{username}/{repo_name}"
            
            # 1. Create the repository for this Algo-Env pair
            create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
            all_repo_ids.append(repo_id)

            # Generate and save the README
            readme_str = generate_sb3_readme(env, algo, local_path)
            with open(local_path / "README.md", "w") as f:
                f.write(readme_str)

            upload_large_folder(
                repo_type="model",
                folder_path=local_path,
                repo_id=repo_id,
                ignore_patterns=[
                    "*00.gif",
                    "*.pdf",
                    "*.npz",
                    "*.png",
                    "*.pdf",
                    "*.log",
                    "**/preparation/*",
                    "**/wandb/*",
                    "hydra.yaml",
                    "multirun.yaml",      
                    "*.pkl",
                ]
            )

# 3. Group everything into a nice Collection
# collection = api.create_collection(
#     title="FluidGym Benchmark Models",
#     description="Plug-and-Play Benchmarking of Reinforcement Learning Algorithms for Large-Scale Flow Control",
#     namespace=username
# )

# for r_id in all_repo_ids:
#     api.add_collection_item(collection.slug, item_id=r_id, item_type="model")