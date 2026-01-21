# Reproducing Experimental Results

## Baseline Training

### Cylinder

<details>
<summary>CylinderJet2D</summary>
    <br>

    python -m runscripts/train/train_sb3.py \
        env_id=CylinderJet2D-easy-v0,CylinderJet2D-medium-v0,CylinderJet2D-hard-v0 \ 
        algorithm=sb3_ppo,sb3_sac \
        seed=range(5) 

</details>

<details>
<summary>CylinderRot2D</summary>
    <br>

    python -m runscripts/train/train_sb3.py \
        env_id=CylinderRot2D-easy-v0,CylinderRot2D-medium-v0,CylinderRot2D-hard-v0 \ 
        algorithm=sb3_ppo,sb3_sac \
        seed=range(5) 
    
</details>

<details>
<summary>CylinderJet3D SARL</summary>
    <br>

    python -m runscripts/train/train_sb3.py \
        env_id=CylinderJet3D-easy-v0,CylinderJet3D-medium-v0,CylinderJet3D-hard-v0 \ 
        algorithm=sb3_ppo,sb3_sac \
        seed=range(3) \
        rl_mode=sarl

</details>

<details>
<summary>CylinderJet3D MARL</summary>
    <br>

    python -m runscripts/train/train_sb3.py \
        env_id=CylinderJet3D-easy-v0,CylinderJet3D-medium-v0,CylinderJet3D-hard-v0 \ 
        algorithm=sb3_ppo,sb3_sac \
        seed=range(3) \
        rl_mode=marl

</details>

### RBC

<details>
<summary>RBC2D SARL</summary>
    <br>

    python -m runscripts/train/train_sb3.py \
        env_id=RBC2D-easy-v0,RBC2D-medium-v0,RBC2D-hard-v0 \ 
        algorithm=sb3_ppo,sb3_sac \
        seed=range(5) \
        rl_mode=sarl

</details>

<details>
<summary>RBC2D MARL</summary>
    <br>

    python -m runscripts/train/train_sb3.py \
        env_id=RBC2D-easy-v0,RBC2D-medium-v0,RBC2D-hard-v0 \ 
        algorithm=sb3_ppo,sb3_sac \
        seed=range(5) \
        rl_mode=marl
    
</details>

<details>
<summary>RBC3D MARL</summary>
    <br>

    python -m runscripts/train/train_sb3.py \
        env_id=RBC3D-easy-v0,RBC3D-medium-v0,RBC3D-hard-v0 \ 
        algorithm=sb3_ppo,sb3_sac \
        seed=range(5) \
        rl_mode=marl
    
</details>

### Airfoil

<details>
<summary>Airfoil2D</summary>
    <br>

    python -m runscripts/train/train_sb3.py \
        env_id=Airfoil2D-easy-v0,Airfoil2D-medium-v0,Airfoil2D-hard-v0 \ 
        algorithm=sb3_ppo,sb3_sac \
        seed=range(5) \
        total_timesteps=20000
    
</details>

<details>
<summary>Airfoil3D SARL</summary>
    <br>

    python -m runscripts/train/train_sb3.py \
        env_id=Airfoil3D-easy-v0 \ 
        algorithm=sb3_ppo,sb3_sac \
        seed=range(3) \
        total_timesteps=20000
    
</details>

<details>
<summary>Airfoil3D MARL</summary>
    <br>

    python -m runscripts/train/train_sb3.py \
        env_id=Airfoil3D-easy-v0 \ 
        algorithm=sb3_ppo,sb3_sac \
        seed=range(3) \
        total_timesteps=20000 \
        rl_mode=marl
    
</details>

### TCF

<details>
<summary>TCFSmall3D</summary>
    <br>

    python -m runscripts/train/train_sb3.py \
        env_id=TCFSmall3D-both-easy-v0 \ 
        algorithm=sb3_ppo \
        seed=range(5) \
        total_timesteps=100000 \
        rl_mode=marl

    python -m runscripts/train/train_sb3.py \
        env_id=TCFSmall3D-both-easy-v0 \ 
        algorithm=sb3_sac \
        seed=range(5) \
        total_timesteps=100000 \
        rl_mode=marl \
        algorithm.obj.train_freq=1 \
        algorithm.obj.gradient_steps=1
    
</details>

<details>
<summary>TCFLarge3D</summary>
    <br>

    python -m runscripts/train/train_sb3.py \
        env_id=TCFLarge3D-both-easy-v0 \ 
        algorithm=sb3_ppo \
        seed=range(5) \
        total_timesteps=100000 \
        rl_mode=marl

    python -m runscripts/train/train_sb3.py \
        env_id=TCFLarge3D-both-easy-v0 \ 
        algorithm=sb3_sac \
        seed=range(5) \
        total_timesteps=100000 \
        rl_mode=marl \
        algorithm.obj.train_freq=1 \
        algorithm.obj.gradient_steps=1
        
</details>

## Baseline Testing

### Cylinder

<details>
<summary>CylinderJet2D</summary>
    <br>

    python -m runscripts/train/test_sb3.py \
        env_id=CylinderJet2D-easy-v0,CylinderJet2D-medium-v0,CylinderJet2D-hard-v0 \ 
        algorithm=sb3_ppo,sb3_sac \
        seed=range(5) 

</details>

<details>
<summary>CylinderRot2D</summary>
    <br>

    python -m runscripts/train/test_sb3.py \
        env_id=CylinderRot2D-easy-v0,CylinderRot2D-medium-v0,CylinderRot2D-hard-v0 \ 
        algorithm=sb3_ppo,sb3_sac \
        seed=range(5) 
    
</details>

<details>
<summary>CylinderJet3D SARL</summary>
    <br>

    python -m runscripts/train/test_sb3.py \
        env_id=CylinderJet3D-easy-v0,CylinderJet3D-medium-v0,CylinderJet3D-hard-v0 \ 
        algorithm=sb3_ppo,sb3_sac \
        seed=range(3) \
        rl_mode=sarl

</details>

<details>
<summary>CylinderJet3D MARL</summary>
    <br>

    python -m runscripts/train/test_sb3.py \
        env_id=CylinderJet3D-easy-v0,CylinderJet3D-medium-v0,CylinderJet3D-hard-v0 \ 
        algorithm=sb3_ppo,sb3_sac \
        seed=range(3) \
        rl_mode=marl

</details>

### RBC

<details>
<summary>RBC2D SARL</summary>
    <br>

    python -m runscripts/train/test_sb3.py \
        env_id=RBC2D-easy-v0,RBC2D-medium-v0,RBC2D-hard-v0 \ 
        algorithm=sb3_ppo,sb3_sac \
        seed=range(5) \
        rl_mode=sarl

</details>

<details>
<summary>RBC2D MARL</summary>
    <br>

    python -m runscripts/train/test_sb3.py \
        env_id=RBC2D-easy-v0,RBC2D-medium-v0,RBC2D-hard-v0 \ 
        algorithm=sb3_ppo,sb3_sac \
        seed=range(5) \
        rl_mode=marl
    
</details>

<details>
<summary>RBC3D MARL</summary>
    <br>

    python -m runscripts/train/test_sb3.py \
        env_id=RBC3D-easy-v0,RBC3D-medium-v0,RBC3D-hard-v0 \ 
        algorithm=sb3_ppo,sb3_sac \
        seed=range(5) \
        rl_mode=marl
        
</details>

### Airfoil

<details>
<summary>Airfoil2D</summary>
    <br>

    python -m runscripts/train/test_sb3.py \
        env_id=Airfoil2D-easy-v0,Airfoil2D-medium-v0,Airfoil2D-hard-v0 \ 
        algorithm=sb3_ppo,sb3_sac \
        seed=range(5)
        
</details>

<details>
<summary>Airfoil3D SARL</summary>
    <br>

    python -m runscripts/train/test_sb3.py \
        env_id=Airfoil3D-easy-v0 \ 
        algorithm=sb3_ppo,sb3_sac \
        seed=range(3) 
    
</details>

<details>
<summary>Airfoil3D MARL</summary>
    <br>

    python -m runscripts/train/test_sb3.py \
        env_id=Airfoil3D-easy-v0 \ 
        algorithm=sb3_ppo,sb3_sac \
        seed=range(3) \
        rl_mode=marl
        
</details>

### TCF

<details>
<summary>TCFSmall3D</summary>
    <br>

    python -m runscripts/train/test_sb3.py \
        env_id=TCFSmall3D-both-easy-v0 \ 
        algorithm=sb3_ppo \
        seed=range(5) \
        rl_mode=marl

    python -m runscripts/train/test_sb3.py \
        env_id=TCFSmall3D-both-easy-v0 \ 
        algorithm=sb3_sac \
        seed=range(5) \
        rl_mode=marl \
    
</details>

<details>
<summary>TCFLarge3D</summary>
    <br>

    python -m runscripts/train/test_sb3.py \
        env_id=TCFLarge3D-both-easy-v0 \ 
        algorithm=sb3_ppo \
        seed=range(5) \
        rl_mode=marl

    python -m runscripts/train/test_sb3.py \
        env_id=TCFLarge3D-both-easy-v0 \ 
        algorithm=sb3_sac \
        seed=range(5) \
        rl_mode=marl 
    
</details>

## Transfer

<details>
<summary>Cylinder 2D -> 3D</summary>
    <br>

    python runscripts/train/test_sb3.py -m \
        env_id=CylinderJet2D-easy-v0 \
        algorithm=sb3_ppo,sb3_sac \
        seed=range(5) \
        rl_mode=sarl
        test_env_id=CylinderJet3D-easy-v0
        test_rl_mode=marl 
        +test_env_kwargs.local_2d_obs=True

    python runscripts/train/test_sb3.py -m \
        env_id=CylinderJet2D-medium-v0 \
        algorithm=sb3_ppo,sb3_sac \
        seed=range(5) \
        rl_mode=sarl
        test_env_id=CylinderJet3D-medium-v0
        test_rl_mode=marl 
        +test_env_kwargs.local_2d_obs=True

    python runscripts/train/test_sb3.py -m \
        env_id=CylinderJet2D-hard-v0 \
        algorithm=sb3_ppo,sb3_sac \
        seed=range(5) \
        rl_mode=sarl
        test_env_id=CylinderJet3D-hard-v0
        test_rl_mode=marl 
        +test_env_kwargs.local_2d_obs=True
    
</details>

<details>
<summary>TCF Small -> Large</summary>
    <br>

    python runscripts/train/test_sb3.py -m \
        env_id=TCFSmall3D-easy-v0 \
        algorithm=sb3_ppo,sb3_sac \
        seed=range(5) \
        rl_mode=marl
        test_env_id=TCFLarge3D-easy-v0

    python runscripts/train/test_sb3.py -m \
        env_id=TCFSmall3D-medium-v0 \
        algorithm=sb3_ppo,sb3_sac \
        seed=range(5) \
        rl_mode=marl
        test_env_id=TCFLarge3D-medium-v0

    python runscripts/train/test_sb3.py -m \
        env_id=TCFSmall3D-hard-v0 \
        algorithm=sb3_ppo,sb3_sac \
        seed=range(5) \
        rl_mode=marl
        test_env_id=TCFLarge3D-hard-v0
    
</details>

## Differentiable MPC (D-MPC)

<details>
<summary>CylinderJet2D-easy-v0</summary>
    <br>

    python runscripts/train/run_d-mpc.py -m \
        env_id=CylinderJet2D-easy-v0 \
        seed=range(10)

</details>

<details>
<summary>CylinderJet2D-medium-v0</summary>
    <br>

    python runscripts/train/run_d-mpc.py -m \
        env_id=CylinderJet2D-medium-v0 \
        seed=range(10)

</details>

<details>
<summary>CylinderJet2D-hard-v0</summary>
    <br>

    python runscripts/train/run_d-mpc.py -m \
        env_id=CylinderJet2D-hard-v0 \
        seed=range(10)

</details>

## Runtime Benchmarks

<details>
<summary>Command</summary>
    Insert all environment IDs here:
    <br>

    python runscripts/benchmark/benchmark.py -m \
        env_id=<env_id>

</details>

## Evaluation

<details>
<summary>Download Experimental Results</summary>
    <br>

    python runscripts/evaluation/download_experimental_results.py

</details>

<details>
<summary>Plots and Tables</summary>
    <br>

    python runscripts/evaluation/generate_plots_and_tables.py

</details>

## Additional Solver Validation

<details>
<summary>Rayleigh-Bénard Convection</summary>
    <br>

    python runscripts/preparation/validate.py env_id=RBC2D-easy-v0 +resolutions=[8,12,16]

</details>
