"""
Experiment registry: swap VLA backends and datasets without mixing artifacts.

Current ICRA stack targets Unitree G1 **29-DoF @ 100 Hz**. Only backends with
``embodiment == "g1_29dof"`` (or compatible) may drive Steps 1–5 / ESN.

Xiaomi-Robotics-1 is registered as ``dual_arm_ee_60`` (30×60 relative EE + base
vel) — **not** a G1 humanoid drop-in. Keep it listed for future work; do not
wire it into the live G1 wipe loop until a real embodiment adapter exists.

Results for new runs:
  research/results/experiments/<vla_id>/<step>/...
Legacy UnifoLM folders (``step1_profiling_unifolm_vla0``, …) stay as-is.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import yaml

from src.paths import RESEARCH_ROOT, experiment_results_path

# Embodiment tags used for compatibility checks.
EMBODIMENT_G1_29DOF = "g1_29dof"
EMBODIMENT_DUAL_ARM_EE_60 = "dual_arm_ee_60"  # Xiaomi XR-1 style
EMBODIMENT_OPENVLA_GENERIC = "openvla_7d"  # legacy OpenVLA arm actions


@dataclass(frozen=True)
class VLABackendSpec:
    """One pluggable VLA backend."""

    id: str
    display_name: str
    embodiment: str
    hf_model_id: str
    action_dim: int
    action_layout: str
    compatible_with_g1_esn: bool
    notes: str = ""
    # Optional: transformers pin / separate env requirement
    requires_separate_env: bool = False
    default_unnorm_key: Optional[str] = None


@dataclass(frozen=True)
class DatasetSpec:
    """One demonstration / eval dataset."""

    id: str
    display_name: str
    embodiment: str
    hf_dataset_id: Optional[str] = None
    local_path: Optional[str] = None
    train_episodes: str = ""
    heldout_episodes: str = ""
    control_hz: float = 100.0
    notes: str = ""


@dataclass
class ExperimentConfig:
    """Resolved VLA + dataset pair for a run."""

    vla: VLABackendSpec
    dataset: DatasetSpec
    results_root: Path
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vla": asdict(self.vla),
            "dataset": asdict(self.dataset),
            "results_root": str(self.results_root),
            "extras": self.extras,
        }


# ── Built-in registry (extend via YAML under configs/experiments/) ──────────

VLA_BACKENDS: Dict[str, VLABackendSpec] = {
    "unifolm": VLABackendSpec(
        id="unifolm",
        display_name="Unitree UnifoLM-VLA-Base",
        embodiment=EMBODIMENT_G1_29DOF,
        hf_model_id="unitreerobotics/UnifoLM-VLA-Base",
        action_dim=23,
        action_layout="g1_ee_6d_23",
        compatible_with_g1_esn=True,
        default_unnorm_key="g1_wipe_table",
        notes="Primary ICRA 2027 VLA. EE→29-DoF via vla_ee_bridge.",
    ),
    "openvla": VLABackendSpec(
        id="openvla",
        display_name="OpenVLA-7B",
        embodiment=EMBODIMENT_OPENVLA_GENERIC,
        hf_model_id="openvla/openvla-7b",
        action_dim=7,
        action_layout="delta_ee_7d",
        compatible_with_g1_esn=False,
        notes="Profiling / reference only — not G1 29-DoF native.",
    ),
    "xiaomi_xr1": VLABackendSpec(
        id="xiaomi_xr1",
        display_name="Xiaomi-Robotics-1",
        embodiment=EMBODIMENT_DUAL_ARM_EE_60,
        hf_model_id="XiaomiRobotics/Xiaomi-Robotics-1-5B",
        action_dim=60,
        action_layout="relative_dual_arm_ee_30x60",
        compatible_with_g1_esn=False,
        requires_separate_env=True,
        notes=(
            "NOT a Unitree G1 29-DoF humanoid policy. Actions are packed "
            "(30, 60) dual-arm EE deltas + grippers + waist + base velocity "
            "(see Xiaomi xr1/docs/data_format.md). Do not drop into the G1 "
            "wipe/ESN loop without a dedicated embodiment adapter + post-train."
        ),
    ),
}

def _datasets_from_unifolm_tasks() -> Dict[str, DatasetSpec]:
    """Register all 12 UnifoLM G1 tasks (+ Dex1 wipe alias for legacy configs)."""
    from src.unifolm_tasks import UNIFOLM_TASKS

    out: Dict[str, DatasetSpec] = {}
    for t in UNIFOLM_TASKS.values():
        out[t.id] = DatasetSpec(
            id=t.id,
            display_name=t.display_name,
            embodiment=EMBODIMENT_G1_29DOF,
            hf_dataset_id=t.primary_dataset_id,
            train_episodes="train",
            heldout_episodes="heldout",
            control_hz=100.0,
            notes=t.notes or f"UnifoLM unnorm_key={t.unnorm_key}",
        )
        # Also index by unnorm key for YAML convenience.
        out[t.unnorm_key] = DatasetSpec(
            id=t.unnorm_key,
            display_name=f"{t.display_name} ({t.unnorm_key})",
            embodiment=EMBODIMENT_G1_29DOF,
            hf_dataset_id=t.hf_dataset_id,
            train_episodes="train",
            heldout_episodes="heldout",
            control_hz=100.0,
            notes=f"Alias for task id={t.id}",
        )
    # Legacy ids used by existing YAML / resolve_experiment defaults.
    wipe = UNIFOLM_TASKS["wipe_table"]
    out["g1_dex1_wipe_table"] = DatasetSpec(
        id="g1_dex1_wipe_table",
        display_name="G1 Dex1 Wipe Table",
        embodiment=EMBODIMENT_G1_29DOF,
        hf_dataset_id=wipe.alt_hf_dataset_id or wipe.hf_dataset_id,
        train_episodes="0-159",
        heldout_episodes="160-199",
        control_hz=100.0,
        notes="Canonical ESN / oracle / live-wipe demos (alias of wipe_table).",
    )
    out["g1_wipe_table"] = DatasetSpec(
        id="g1_wipe_table",
        display_name="G1 Wipe Table (official HF)",
        embodiment=EMBODIMENT_G1_29DOF,
        hf_dataset_id=wipe.hf_dataset_id,
        train_episodes="train",
        heldout_episodes="heldout",
        control_hz=100.0,
        notes="Official UnifoLM wipe corpus (non-Dex1).",
    )
    return out


DATASETS: Dict[str, DatasetSpec] = _datasets_from_unifolm_tasks()


def list_vlas() -> List[VLABackendSpec]:
    return list(VLA_BACKENDS.values())


def list_datasets() -> List[DatasetSpec]:
    return list(DATASETS.values())


def get_vla(vla_id: str) -> VLABackendSpec:
    key = str(vla_id).strip().lower()
    if key not in VLA_BACKENDS:
        known = ", ".join(sorted(VLA_BACKENDS))
        raise KeyError(f"Unknown VLA {vla_id!r}. Registered: {known}")
    return VLA_BACKENDS[key]


def get_dataset(dataset_id: str) -> DatasetSpec:
    key = str(dataset_id).strip().lower()
    if key not in DATASETS:
        known = ", ".join(sorted(DATASETS))
        raise KeyError(f"Unknown dataset {dataset_id!r}. Registered: {known}")
    return DATASETS[key]


def assert_g1_compatible(vla: VLABackendSpec, dataset: DatasetSpec) -> None:
    """Raise if this pair cannot drive the G1 29-DoF ESN / wipe stack."""
    if not vla.compatible_with_g1_esn:
        raise ValueError(
            f"VLA {vla.id!r} ({vla.display_name}) is embodiment={vla.embodiment!r} "
            f"and is NOT compatible with the G1 29-DoF ESN stack.\n{vla.notes}"
        )
    if dataset.embodiment != EMBODIMENT_G1_29DOF:
        raise ValueError(
            f"Dataset {dataset.id!r} embodiment={dataset.embodiment!r}; "
            f"G1 experiments require {EMBODIMENT_G1_29DOF!r}."
        )
    if vla.embodiment != dataset.embodiment and vla.embodiment != EMBODIMENT_G1_29DOF:
        raise ValueError(
            f"VLA embodiment {vla.embodiment!r} incompatible with dataset "
            f"{dataset.embodiment!r}."
        )


def resolve_experiment(
    vla_id: str = "unifolm",
    dataset_id: str = "g1_dex1_wipe_table",
    *,
    step: str = "run",
    require_g1: bool = True,
    extras: Optional[Mapping[str, Any]] = None,
) -> ExperimentConfig:
    """Pick VLA + dataset and a distinct results directory."""
    vla = get_vla(vla_id)
    dataset = get_dataset(dataset_id)
    if require_g1:
        assert_g1_compatible(vla, dataset)
    root = experiment_results_path(vla.id, step)
    return ExperimentConfig(vla=vla, dataset=dataset, results_root=root, extras=dict(extras or {}))


def load_experiment_yaml(path: Path) -> ExperimentConfig:
    """
    Load ``configs/experiments/*.yaml``::

        vla: unifolm
        dataset: g1_dex1_wipe_table
        step: step3_live_wipe
        require_g1: true
        extras: {}
    """
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return resolve_experiment(
        vla_id=str(raw.get("vla", "unifolm")),
        dataset_id=str(raw.get("dataset", "g1_dex1_wipe_table")),
        step=str(raw.get("step", "run")),
        require_g1=bool(raw.get("require_g1", True)),
        extras=raw.get("extras") or {},
    )


def register_vla(spec: VLABackendSpec) -> None:
    """Runtime registration for a custom / future backend."""
    VLA_BACKENDS[spec.id] = spec


def register_dataset(spec: DatasetSpec) -> None:
    DATASETS[spec.id] = spec


def configs_dir() -> Path:
    return RESEARCH_ROOT / "configs" / "experiments"


def print_registry() -> None:
    print("VLA backends:")
    for v in list_vlas():
        g1 = "yes" if v.compatible_with_g1_esn else "NO"
        print(f"  - {v.id:16s}  g1_esn={g1:3s}  {v.display_name}  [{v.embodiment}]")
    print("Datasets:")
    for d in list_datasets():
        print(f"  - {d.id:16s}  {d.display_name}  [{d.embodiment}]")


if __name__ == "__main__":
    print_registry()
    print()
    cfg = resolve_experiment("unifolm", "g1_dex1_wipe_table", step="demo")
    print("Default G1 experiment →", cfg.results_root)
    try:
        resolve_experiment("xiaomi_xr1", "g1_dex1_wipe_table", step="demo")
    except ValueError as exc:
        print("XR-1 blocked (expected):", str(exc).split("\n")[0])
