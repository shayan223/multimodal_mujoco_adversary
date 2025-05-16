import wandb
import torch
from loguru import logger
from pathlib import Path
import ddiffpg


def load_model(model, model_type, cfg):
    artifact = wandb.Api().artifact(cfg.artifact)
    artifact.download(ddiffpg.LIB_PATH)
    logger.warning(f'Load {model_type}')
    weights = torch.load(Path(ddiffpg.LIB_PATH, "model.pth"))

    if model_type in ["actor", "critic", "obs_rms"]:
        if model_type == "obs_rms" and weights[model_type] is None:
            logger.warning('Observation normalization is enabled, but loaded weight contains no normalization info.')
            return
        model.load_state_dict(weights[model_type])
    else:
        logger.warning(f'Invalid model type:{model_type}')


def save_model(path, actor, critic, rms, wandb_run, ret_max, embedding, coverage):
    checkpoint = {'obs_rms': rms,
            'critic': critic,
            'actor': actor,
            'embedding': embedding,
            'coverage': coverage,
            }
    torch.save(checkpoint, path)  # save policy network in *.pth

    model_artifact = wandb.Artifact(wandb_run.id, type="model", description=f"return: {ret_max}")
    model_artifact.add_file(path)
    wandb.save(path, base_path=wandb_run.dir)
    wandb_run.log_artifact(model_artifact)
