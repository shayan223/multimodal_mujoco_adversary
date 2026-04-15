import json
from pathlib import Path

import torch

class _FakeActor(torch.nn.Module):
    def forward(self, x):
        return x.sum(dim=-1, keepdim=True)


class _FakeModel:
    def __init__(self):
        self.actor = _FakeActor()

    def update_actor(self, input_vals, skip_weight_update=True):
        loss = (input_vals ** 2).sum()
        loss.backward()
        return loss


def fgsm_attack(model, input_vals, eps=0.015, target_modality=None, outputs=None):
    input_vals.requires_grad = True

    if outputs is None:
        outputs = model.actor(input_vals)

    model.actor.zero_grad()
    _ = model.update_actor(input_vals, skip_weight_update=True)

    if target_modality == 'velocity':
        targeted_features = input_vals.clone()
        targeted_features[..., 13:19] += eps * input_vals.grad[..., 13:19].sign()
        perturbed_out = targeted_features
    elif target_modality == 'angular':
        targeted_features = input_vals.clone()
        targeted_features[..., 0:13] += eps * input_vals.grad[..., 0:13].sign()
        targeted_features[..., 19:] += eps * input_vals.grad[..., 19:].sign()
        perturbed_out = targeted_features
    else:
        perturbed_out = input_vals + eps * input_vals.grad.sign()

    return perturbed_out


def validate_attack_targeting(original_obs: torch.Tensor, perturbed_obs: torch.Tensor, target_modality=None):
    changed = (perturbed_obs - original_obs).abs() > 1e-8
    changed_any = changed.any(dim=0) if changed.dim() > 1 else changed
    changed_indices = torch.where(changed_any)[0].detach().cpu().tolist()

    if target_modality == 'velocity':
        expected = set(range(13, 19))
    elif target_modality == 'angular':
        expected = set(list(range(13)) + list(range(19, original_obs.shape[-1])))
    else:
        expected = set(range(original_obs.shape[-1]))

    actual = set(changed_indices)
    return {
        "target_modality": target_modality,
        "changed_indices": changed_indices,
        "unexpected_indices": sorted(actual - expected),
        "missing_expected_indices": sorted(expected - actual) if target_modality is not None else [],
        "targeting_valid": len(actual - expected) == 0,
    }


def main():
    model = _FakeModel()
    obs = torch.linspace(0.1, 2.9, steps=29, dtype=torch.float32).unsqueeze(0)

    results = {}
    for target_modality in [None, "velocity", "angular"]:
        perturbed = fgsm_attack(model, obs.clone(), eps=0.015, target_modality=target_modality)
        result = validate_attack_targeting(obs, perturbed, target_modality=target_modality)
        key = "both" if target_modality is None else target_modality
        results[key] = result
        print(f"{key}: {result}")

    output_path = Path(__file__).resolve().parents[1] / "attack_validation.json"
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    print(f"Wrote attack validation report to {output_path}")


if __name__ == "__main__":
    main()
