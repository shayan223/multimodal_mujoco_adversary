import argparse
import json
from pathlib import Path

from diffusion import Diffusion_model


def main():
    parser = argparse.ArgumentParser(description="Evaluate DDPM purifier reconstruction metrics on train/val/test splits.")
    parser.add_argument("--experiment-name", default="diffusion_defense_1")
    parser.add_argument("--mode", default="stochastic_light", choices=["deterministic", "stochastic_light", "stochastic_heavy"])
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--renoise-strength", type=float, default=1.0)
    parser.add_argument("--benign-csv", default=None)
    parser.add_argument("--adversarial-csv", default=None)
    args = parser.parse_args()

    model = Diffusion_model(
        experiment_name=args.experiment_name,
        inference_mode=args.mode,
        inference_steps=args.steps,
        renoise_strength=args.renoise_strength,
        benign_csv=args.benign_csv,
        adversarial_csv=args.adversarial_csv,
    )
    model.load_params()
    model._build_split_loaders()

    metrics = {
        "train": model.evaluate_loader(model.training_data_loader),
        "validation": model.evaluate_loader(model.validation_data_loader),
        "test": model.evaluate_loader(model.test_data_loader),
        "split_metadata": model.dataset_split_metadata,
        "mode": args.mode,
        "steps": args.steps,
        "renoise_strength": args.renoise_strength,
    }

    print(json.dumps(metrics, indent=2))
    output_path = Path(model.experiment_path) / f"{args.experiment_name}_evaluation_metrics.json"
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    print(f"Wrote evaluation metrics to {output_path}")


if __name__ == "__main__":
    main()
