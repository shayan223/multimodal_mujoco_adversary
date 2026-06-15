import json
from pathlib import Path

import torch

class _FakeActor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return self.weight * x.sum(dim=-1, keepdim=True)


class _FakeModel:
    def __init__(self):
        self.actor = _FakeActor()
        self.critic = torch.nn.Linear(1, 1, bias=False)
        self.log_alpha = torch.nn.Parameter(torch.tensor([0.0]))

    def get_alpha(self):
        return self.log_alpha.exp().detach()

    def get_attack_input_gradient(self, input_vals):
        logits = self.actor(input_vals)
        q_value = self.critic(logits)
        attack_loss = (self.get_alpha() * logits - q_value).mean()
        (input_grad,) = torch.autograd.grad(
            attack_loss,
            input_vals,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )
        return input_grad.detach()


def fgsm_attack(model, input_vals, eps=0.015, target_modality=None, outputs=None):
    attack_inputs = input_vals.detach().clone().requires_grad_(True)
    input_grad = model.get_attack_input_gradient(attack_inputs)

    if target_modality == 'velocity':
        targeted_features = attack_inputs.detach().clone()
        targeted_features[..., 13:19] += eps * input_grad[..., 13:19].sign()
        perturbed_out = targeted_features
    elif target_modality == 'angular':
        targeted_features = attack_inputs.detach().clone()
        targeted_features[..., 0:13] += eps * input_grad[..., 0:13].sign()
        targeted_features[..., 19:] += eps * input_grad[..., 19:].sign()
        perturbed_out = targeted_features
    else:
        perturbed_out = attack_inputs.detach() + eps * input_grad.sign()

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


def snapshot_model_state(model):
    return {
        "actor_weight": model.actor.weight.detach().clone(),
        "critic_weight": model.critic.weight.detach().clone(),
        "log_alpha": model.log_alpha.detach().clone(),
        "actor_grad": None if model.actor.weight.grad is None else model.actor.weight.grad.detach().clone(),
        "critic_grad": None if model.critic.weight.grad is None else model.critic.weight.grad.detach().clone(),
        "alpha_grad": None if model.log_alpha.grad is None else model.log_alpha.grad.detach().clone(),
    }


def validate_state_unchanged(before, after):
    return {
        "actor_weight_unchanged": torch.equal(before["actor_weight"], after["actor_weight"]),
        "critic_weight_unchanged": torch.equal(before["critic_weight"], after["critic_weight"]),
        "log_alpha_unchanged": torch.equal(before["log_alpha"], after["log_alpha"]),
        "actor_grad_unchanged": (
            before["actor_grad"] is None and after["actor_grad"] is None
        ) or (
            before["actor_grad"] is not None
            and after["actor_grad"] is not None
            and torch.equal(before["actor_grad"], after["actor_grad"])
        ),
        "critic_grad_unchanged": (
            before["critic_grad"] is None and after["critic_grad"] is None
        ) or (
            before["critic_grad"] is not None
            and after["critic_grad"] is not None
            and torch.equal(before["critic_grad"], after["critic_grad"])
        ),
        "alpha_grad_unchanged": (
            before["alpha_grad"] is None and after["alpha_grad"] is None
        ) or (
            before["alpha_grad"] is not None
            and after["alpha_grad"] is not None
            and torch.equal(before["alpha_grad"], after["alpha_grad"])
        ),
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

    before_state = snapshot_model_state(model)
    _ = fgsm_attack(model, obs.clone(), eps=0.015, target_modality=None)
    after_state = snapshot_model_state(model)
    state_result = validate_state_unchanged(before_state, after_state)
    results["state_preservation"] = state_result
    print(f"state_preservation: {state_result}")

    output_path = Path(__file__).resolve().parents[1] / "attack_validation.json"
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    print(f"Wrote attack validation report to {output_path}")


if __name__ == "__main__":
    main()
