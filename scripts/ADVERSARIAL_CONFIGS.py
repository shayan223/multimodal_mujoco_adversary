import itertools
import os
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple


# Single source of truth; same values as the previous hardcoded __init__.
_DEFAULT_ADV: Dict[str, Any] = {
    # Whether or not to use attack in testbed
    "ENABLE_ATTACK": True,
    # Which attack to use: 'FGSM', 'ZeroOut', 'RandomZeroOut', 'ModalityZeroOut'
    "ATTACK_CHOICE": "FGSM",
    # Dataset collection mode
    "GENERATE_DATASET": False,
    "DATA_PREFIX": "fgsm015",
    # Optional comma-separated collection specs, e.g. "FGSM:0.007,FGSM:0.015".
    # When unset, collection uses ATTACK_CHOICE and FGSM_MAGNITUDE.
    "COLLECTION_ATTACKS": None,
    # Dataset save policy:
    # episode_mean_gate = keep only eval buffers whose mean return clears the threshold
    # save_all_with_quality = keep every eval buffer and annotate quality in sidecar CSVs
    "COLLECTION_SAVE_MODE": "episode_mean_gate",
    "COLLECTION_MEAN_RETURN_THRESHOLD": 0.0,
    # 'VAE', 'VAE_3d', 'Gaussian', 'DDPM', or None
    "DEF_METHOD": "DDPM",
    "TRAIN_ON_DEF": False,
    # Defense runtime modes:
    # deterministic, stochastic_light, stochastic_heavy
    "DEFENSE_MODE": "stochastic_light",
    # Whether to apply the defense path during training/eval loops when a defense exists.
    "ENABLE_DEFENSE_TRAIN": True,
    "ENABLE_DEFENSE_EVAL": True,
    # DDPM-specific runtime controls
    "DDPM_EXPERIMENT_NAME": "diffusion_defense_1",
    "DDPM_RENOISE_STRENGTH": 1.0,
    "DDPM_INFERENCE_STEPS": 3,
    # None (both), 'velocity', or 'angular'
    "TARGET_MODALITY": None,
    "MAX_STEPS_OVERRIDE": 3000000,
    "LEARNING_RATE": 1e-3,
    "FGSM_MAGNITUDE": 0.015,
    "SAVE_PATH": "/home/shayan/github/multimodal_mujoco_adversary/",
}

# --- Full factorial grid over magnitude, attack, modality, and defense ---
# FGSM / ZeroOut / RandomZeroOut sweep both + targeted modalities.
# ModalityZeroOut only sweeps targeted modalities because "both" is not a valid mode there.
# Names look like: m007_FGSM_velocity_none, m015_ModalityZeroOut_angular_DDPM
FGSM_MAGNITUDE_GRID: Tuple[float, ...] = (0.007, )# (0.015, 0.007)
ATTACK_CHOICE_GRID: Tuple[str, ...] = (
    "FGSM",
    "ZeroOut",
    "RandomZeroOut",
    "ModalityZeroOut",
)
ATTACK_MODALITY_GRID: Dict[str, Tuple[Optional[str], ...]] = {
    "FGSM": (None, "velocity", "angular"),
    "ZeroOut": (None, "velocity", "angular"),
    "RandomZeroOut": (None, "velocity", "angular"),
    "ModalityZeroOut": ("velocity", "angular"),
}
# None = no defense; strings match DefenceObsWrapper / loader branches in baselines_main.
DEF_METHOD_GRID: Tuple[Optional[str], ...] = (
    None,
    "VAE",
    #"VAE_3d",
    "Gaussian",
    "DDPM",
)
TRAIN_ON_DEF_GRID: Tuple[bool, ...] = (
    # False,
    True,
)


def _grid_preset_key(
    fgsm: float,
    attack: str,
    target_modality: Optional[str],
    def_method: Optional[str],
) -> str:
    eps = "007" if abs(fgsm - 0.007) < 1e-12 else "015"
    modality_slug = "both" if target_modality is None else target_modality
    def_slug = "none" if def_method is None else def_method
    return f"m{eps}_{attack}_{modality_slug}_{def_slug}"


def _build_grid_presets() -> Dict[str, Dict[str, Any]]:
    presets: Dict[str, Dict[str, Any]] = {}
    for fgsm, attack in itertools.product(FGSM_MAGNITUDE_GRID, ATTACK_CHOICE_GRID):
        for target_modality, def_method, train_on_def in itertools.product(
            ATTACK_MODALITY_GRID[attack], DEF_METHOD_GRID, TRAIN_ON_DEF_GRID
        ):
            if def_method is None and train_on_def:
                continue
            name = _grid_preset_key(fgsm, attack, target_modality, def_method)
            if train_on_def:
                name = f"{name}_trainDef"
            entry: Dict[str, Any] = {
                "FGSM_MAGNITUDE": fgsm,
                "ATTACK_CHOICE": attack,
                "DEF_METHOD": def_method,
                "TARGET_MODALITY": target_modality,
                "TRAIN_ON_DEF": train_on_def,
            }
            presets[name] = entry
    return presets


_GRID_PRESETS = _build_grid_presets()
GRID_PRESET_NAMES: Tuple[str, ...] = tuple(sorted(_GRID_PRESETS.keys()))


def _build_no_attack_presets() -> Dict[str, Dict[str, Any]]:
    presets: Dict[str, Dict[str, Any]] = {}
    for def_method, train_on_def in itertools.product(DEF_METHOD_GRID, TRAIN_ON_DEF_GRID):
        if def_method is None and train_on_def:
            continue

        if def_method is None:
            name = "no_attack_none"
        else:
            name = f"no_attack_{def_method}"
            if train_on_def:
                name = f"{name}_trainDef"

        presets[name] = {
            "ENABLE_ATTACK": False,
            "DEF_METHOD": def_method,
            "TRAIN_ON_DEF": train_on_def,
            "TARGET_MODALITY": None,
        }

    return presets


_NO_ATTACK_PRESETS = _build_no_attack_presets()
NO_ATTACK_PRESET_NAMES: Tuple[str, ...] = tuple(sorted(_NO_ATTACK_PRESETS.keys()))

# Partial overrides. Pick with: ADV_PRESET=name or adversarial_cfg(preset="name").
ADV_PRESETS: Dict[str, Dict[str, Any]] = {
    "default": {},
    "no_attack": deepcopy(_NO_ATTACK_PRESETS["no_attack_DDPM_trainDef"]),
    **_NO_ATTACK_PRESETS,
    **_GRID_PRESETS,
}


def _resolve_preset_name(preset: Optional[str]) -> str:
    if preset is not None:
        return preset.strip()
    return os.environ.get("ADV_PRESET", "default").strip()


def _merged_adv(preset_name: str) -> Dict[str, Any]:
    if preset_name not in ADV_PRESETS:
        valid = ", ".join(sorted(ADV_PRESETS.keys()))
        raise ValueError(
            f"Unknown adversarial preset {preset_name!r}. Valid presets: {valid}"
        )
    merged = deepcopy(_DEFAULT_ADV)
    merged.update(ADV_PRESETS[preset_name])
    return merged


def _parse_bool(raw: str) -> bool:
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _coerce_override(raw: str, current: Any) -> Any:
    if current is None:
        stripped = raw.strip()
        if stripped.lower() == "none":
            return None
        return stripped
    if isinstance(current, bool):
        return _parse_bool(raw)
    if isinstance(current, int) and not isinstance(current, bool):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    return raw.strip()


def _apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(config)
    for key, value in list(merged.items()):
        env_key = f"ADV_{key}"
        if env_key in os.environ:
            merged[key] = _coerce_override(os.environ[env_key], value)
    return merged


class adversarial_cfg:
    def __init__(self, preset: Optional[str] = None):
        name = _resolve_preset_name(preset)
        merged = _merged_adv(name)
        merged = _apply_env_overrides(merged)
        self.PRESET_NAME = name
        for key, value in merged.items():
            setattr(self, key, value)

        if self.SAVE_PATH == "/path/to/github/multimodal_mujoco_adversary/":
            print("#####################")
            print(
                "WARNING: SAVE_PATH not set. Set it to absolute path to the repository "
                "directory. Please update _DEFAULT_ADV in scripts/ADVERSARIAL_CONFIGS.py"
            )
            print("#####################")
        if self.SAVE_PATH[-1] != "/":
            print("#####################")
            print(
                'WARNING: File Path does not end with "/". adding one for you. '
                "Please double check its correctness."
            )
            self.SAVE_PATH += "/"
            print("#####################")


def build_adv_wandb_metadata(adv_cfg: adversarial_cfg, seed: Any) -> Dict[str, Any]:
    target_modality = "both" if adv_cfg.TARGET_MODALITY is None else str(adv_cfg.TARGET_MODALITY)
    attack_type = str(adv_cfg.ATTACK_CHOICE)
    defense_type = "none" if adv_cfg.DEF_METHOD is None else str(adv_cfg.DEF_METHOD)
    fgsm_magnitude = float(adv_cfg.FGSM_MAGNITUDE)
    fgsm_magnitude_display = f"{fgsm_magnitude:.3f}".rstrip("0").rstrip(".")
    run_name_base = f"{adv_cfg.PRESET_NAME}_{target_modality}_seed{seed}"

    return {
        "adv_preset": adv_cfg.PRESET_NAME,
        "attack_type": attack_type,
        "defense_type": defense_type,
        "target_modality": target_modality,
        "target_modality_display": target_modality,
        "fgsm_magnitude": fgsm_magnitude,
        "fgsm_magnitude_display": fgsm_magnitude_display,
        "enable_attack": bool(adv_cfg.ENABLE_ATTACK),
        "train_on_defense": bool(adv_cfg.TRAIN_ON_DEF),
        "defense_mode": str(adv_cfg.DEFENSE_MODE),
        "enable_defense_train": bool(adv_cfg.ENABLE_DEFENSE_TRAIN),
        "enable_defense_eval": bool(adv_cfg.ENABLE_DEFENSE_EVAL),
        "ddpm_renoise_strength": float(adv_cfg.DDPM_RENOISE_STRENGTH),
        "ddpm_inference_steps": int(adv_cfg.DDPM_INFERENCE_STEPS),
        "collection_save_mode": str(adv_cfg.COLLECTION_SAVE_MODE),
        "collection_mean_return_threshold": float(adv_cfg.COLLECTION_MEAN_RETURN_THRESHOLD),
        "run_name_base": run_name_base,
    }
