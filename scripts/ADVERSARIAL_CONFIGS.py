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
    # 'VAE', 'VAE_3d', 'Gaussian', 'DDPM', or None
    "DEF_METHOD": "DDPM",
    "TRAIN_ON_DEF": False,
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
FGSM_MAGNITUDE_GRID: Tuple[float, ...] = (0.007, 0.015)
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
        for target_modality, def_method in itertools.product(
            ATTACK_MODALITY_GRID[attack], DEF_METHOD_GRID
        ):
            name = _grid_preset_key(fgsm, attack, target_modality, def_method)
            entry: Dict[str, Any] = {
                "FGSM_MAGNITUDE": fgsm,
                "ATTACK_CHOICE": attack,
                "DEF_METHOD": def_method,
                "TARGET_MODALITY": target_modality,
            }
            presets[name] = entry
    return presets


_GRID_PRESETS = _build_grid_presets()
GRID_PRESET_NAMES: Tuple[str, ...] = tuple(sorted(_GRID_PRESETS.keys()))

# Partial overrides. Pick with: ADV_PRESET=name or adversarial_cfg(preset="name").
ADV_PRESETS: Dict[str, Dict[str, Any]] = {
    "default": {},
    "no_attack": {"ENABLE_ATTACK": False},
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


class adversarial_cfg:
    def __init__(self, preset: Optional[str] = None):
        name = _resolve_preset_name(preset)
        merged = _merged_adv(name)
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
        "run_name_base": run_name_base,
    }
