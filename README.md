# Multimodal Agent Test Bed
**For the paper: "Adversarial Robustness of Multimodal RL Agents"**

This repository builds upon the original [DDiffPG](https://github.com/chenfh5/ddiffpg) algorithm to create a test bed for evaluating the adversarial robustness of multimodal reinforcement learning (RL) agents.

> :heart: **Credit**: We gratefully acknowledge the original authors of [DDiffPG](https://github.com/chenfh5/ddiffpg) for their foundational work. This repository extends their implementation to support multimodal RL research.

---

- [:gear: Installation](#installation)
    - [Install Environment](#install-environment)
- [:scroll: Usage](#usage)
    - [:pencil2: Logging](#logging)
    - [:bulb: Training](#training)
    - [:bookmark: Baselines](#baselines)
    - [:floppy_disk: Saving and Loading](#saving-and-loading)
- [:wrench: Troubleshooting](#troubleshooting)

---

## :gear: Installation

### Install Environment <a name="install-environment"></a>

1. Clone the repository:

    ```bash
    git clone git@github.com:ddiffpg.git
    cd ddiffpg
    ```

2. Create the Conda environment and install dependencies:

    ```bash
    conda env create -f environment.yml
    conda activate ddiffpg
    ```

3. Install MuJoCo 210.

4. Install [D4RL](https://github.com/Farama-Foundation/D4RL)

5. Install [panda-gym](https://github.com/qgallouedec/panda-gym)

6. Install this package:

    ```bash
    pip install -e .
    ```

---

## :scroll: Usage

---

## :scroll: Usage

### :pencil2: Logging <a name="logging"></a>

We use Weights & Biases (W&B) for experiment tracking.

1. Sign up at [wandb.ai](https://wandb.ai)
2. Get your API key: [wandb.ai/authorize](https://wandb.ai/authorize)
3. Set the API key in your terminal:

    ```bash
    export WANDB_API_KEY=your_key_here
    ```

---

### :bulb: Training <a name="training"></a>

The default and recommended training setup uses the **SAC (Soft Actor-Critic)** algorithm:

```bash
python scripts/baselines_main.py algo=sac_algo env.name=antmaze-v1
```

---

### :bookmark: Alternative Algorithms <a name="baselines"></a>

You may also run experiments using algorithms derived from the original DDiffPG repository:

- **DIPO** (Diffusion In Policy Optimization):

    ```bash
    python scripts/baselines_main.py algo=dipo_algo env.name=antmaze-v1
    ```

- **DDiffPG** (Deep Diffusion Policy Gradient):

    ```bash
    python scripts/ddiffpg_main.py algo=ddiffpg_algo env.name=antmaze-v1
    ```

These can be helpful for benchmarking and ablation studies.

---

### :floppy_disk: Saving and Loading <a name="saving-and-loading"></a>

Checkpoints are automatically saved using W&B [Artifacts](https://docs.wandb.ai/ref/python/artifact).


## :wrench: Troubleshooting

If you encounter installation or runtime issues, try the following fixes:

**Package Fixes**:

```bash
pip install --upgrade diffusers
pip install "cython<3"
```

**System Dependencies**:

```bash
sudo apt update
sudo apt install build-essential python3-dev
```

To avoid build errors (e.g., related to `bin/gcc`), you may also need:

```bash
sudo apt-get install libglew-dev
```

**Environment Variables**:  
Add these lines to your `~/.bashrc` (or equivalent shell config) if MuJoCo or GPU libraries are not found:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/shayan/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
```

Then run:

```bash
source ~/.bashrc
```

---

Happy experimenting! Feel free to open issues or contribute improvements.


---

## :hammer_and_wrench: Customizing and Extending the Codebase

The main script for running baseline models and defenses is located in [`scripts/baselines_main.py`](scripts/baselines_main.py). Below are guidelines for understanding and modifying the implementation.

### 🔐 Implemented Defenses

The current defense mechanisms supported in the codebase include:

- **Gaussian**: Applies Gaussian noise as a simple filter against adversarial perturbations.
- **VAE**: Uses a Variational Autoencoder (VAE) to filter and reconstruct observations, providing robustness against adversarial inputs.
- **VAE_3d**: An experimental defense model that reshapes 1D observation vectors into a 6×6 2D format and processes them via a 2D VAE. This version is not fully functional or robust.

These defenses are defined and invoked via the `defender` function.

### ➕ Adding a New Defense

To add a new defense model:

1. Navigate to the `defender` function.
2. Add a new condition that returns a function which:
   - **Input**: takes an `np.ndarray` observation.
   - **Output**: returns a modified `np.ndarray` observation.

**Example Skeleton:**

```python
def custom_defense(obs: np.ndarray) -> np.ndarray:
    # Apply transformation to obs
    return modified_obs

# In defender():
if defense_type == "custom":
    return custom_defense
```

### ⚔️ Adding a New Attack

To implement a new adversarial attack:

1. Review the `fgsm_attack` function as a reference.
2. Define a new function that perturbs observations based on your attack strategy.

Ensure it:
- Receives the observation, model, and other required context.
- Returns the perturbed observation.

---

