import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run baselines_main.py in adversarial dataset-collection mode.")
    parser.add_argument("--adv-preset", default="default", help="ADV_PRESET base preset to start from.")
    parser.add_argument("--algo", default="sac_algo", help="Hydra algo override.")
    parser.add_argument("--env-name", default="antmaze-v1", help="Hydra env.name override.")
    parser.add_argument("--attack-choice", default="FGSM", help="Attack choice.")
    parser.add_argument("--target-modality", default="None", help="Target modality: None, velocity, or angular.")
    parser.add_argument("--fgsm-magnitude", type=float, default=0.015, help="FGSM epsilon.")
    parser.add_argument("--defense-method", default="DDPM", help="Defense method or None.")
    parser.add_argument("--train-on-defense", action="store_true", help="Enable defended observations for SAC training.")
    parser.add_argument("--defense-mode", default="stochastic_light", help="Defense runtime mode.")
    parser.add_argument("--ddpm-experiment-name", default="diffusion_defense_1", help="DDPM experiment/checkpoint name.")
    parser.add_argument("--dataset-prefix", default=None, help="Output dataset prefix. Defaults to a name derived from attack settings.")
    parser.add_argument("--save-path", default=None, help="Absolute repository/output path used by adversarial config.")
    parser.add_argument("--max-step", type=int, default=None, help="Optional MAX_STEPS_OVERRIDE value.")
    parser.add_argument("hydra_overrides", nargs="*", help="Additional Hydra overrides.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    dataset_prefix = args.dataset_prefix or (
        f"{args.attack_choice.lower()}_{str(args.fgsm_magnitude).replace('.', '')}_"
        f"{args.target_modality.lower() if args.target_modality != 'None' else 'both'}_"
        f"{args.defense_method.lower() if args.defense_method != 'None' else 'nodef'}"
    )

    env = os.environ.copy()
    env["ADV_PRESET"] = args.adv_preset
    env["ADV_GENERATE_DATASET"] = "true"
    env["ADV_ENABLE_ATTACK"] = "true"
    env["ADV_ATTACK_CHOICE"] = args.attack_choice
    env["ADV_TARGET_MODALITY"] = args.target_modality
    env["ADV_FGSM_MAGNITUDE"] = str(args.fgsm_magnitude)
    env["ADV_DEF_METHOD"] = args.defense_method
    env["ADV_TRAIN_ON_DEF"] = "true" if args.train_on_defense else "false"
    env["ADV_DEFENSE_MODE"] = args.defense_mode
    env["ADV_DDPM_EXPERIMENT_NAME"] = args.ddpm_experiment_name
    env["ADV_DATA_PREFIX"] = dataset_prefix
    if args.save_path is not None:
        env["ADV_SAVE_PATH"] = args.save_path
    if args.max_step is not None:
        env["ADV_MAX_STEPS_OVERRIDE"] = str(args.max_step)

    command = [
        sys.executable,
        str(repo_root / "scripts" / "baselines_main.py"),
        f"algo={args.algo}",
        f"env.name={args.env_name}",
        *args.hydra_overrides,
    ]
    print("Running:", " ".join(command))
    print("Dataset prefix:", dataset_prefix)
    return subprocess.call(command, cwd=str(repo_root), env=env)


if __name__ == "__main__":
    raise SystemExit(main())
