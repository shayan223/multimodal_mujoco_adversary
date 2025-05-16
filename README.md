# Deep Diffusion Policy Gradient (DDiffPG)
This repository provides a PyTorch implementation of DDiffPG.

---

- [:gear: Installation](#installation)
    - [Install DDiffPG](#install_ddiffpg)
- [:scroll: Usage](#usage)
    - [:pencil2: Logging](#usage_logging)
    - [:bulb: Train with DDiffPG](#usage_ddiffpg)
    - [:bookmark: Baselines](#usage_baselines)
    - [:floppy_disk: Saving and Loading](#usage_saving_loading)


## :gear: Installation

### Install DDiffPG <a name="install_ddiffpg"></a>

1. Clone the package:

    ```bash
    git clone git@github.com:ddiffpg.git
    cd ddiffpg
    ```

2. Create Conda environment and install dependencies:

    ```bash
    conda env create -f environment.yml
    conda activate ddiffpg
    ```

3. Install MuJoCo 210. 

4. Install [D4RL](https://github.com/Farama-Foundation/D4RL). 

5. Install [panda-gym](https://github.com/qgallouedec/panda-gym)

5. Install DDiffPG. 

     ```bash
    pip install -e .
    ```


## :scroll: Usage

### :pencil2: Logging <a name="usage_logging"></a>

We use Weights & Biases (W&B) for logging. 

1. Get a W&B account from https://wandb.ai/site

2. Get your API key from https://wandb.ai/authorize

3. set up your account in terminal
    ```bash
    export WANDB_API_KEY=$API Key$
    ```

### :bulb: Train with DDiffPG <a name="usage_ddiffpg"></a>

Run DDiffPG on AntMaze tasks.

```bash
python scripts/ddiffpg_main.py algo=ddiffpg_algo env.name=antmaze-v1
```

### :bookmark: Baselines <a name="usage_baselines"></a>

Run DIPO baseline

```bash
python scripts/baselines_main.py algo=dipo_algo env.name=antmaze-v1
```

Run SAC baseline

```bash
python scripts/baselines_main.py algo=sac_algo env.name=antmaze-v1
```

### :floppy_disk: Saving and Loading <a name="usage_saving_loading"></a>

Checkpoints are automatically saved as W&B [Artifacts](https://docs.wandb.ai/ref/python/artifact).