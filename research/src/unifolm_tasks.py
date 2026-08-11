"""
UnifoLM-VLA-0 / UnifoLM-VLA-Base — all 12 Unitree G1 benchmark tasks.

Source: https://github.com/unitreerobotics/unifolm-vla (Dataset table)
Norm keys verified against UnifoLM-VLA-Base ``dataset_statistics.json`` (all 23-D).

Default for the ICRA wipe paper path remains ``wipe_table`` (Dex1 corpus).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# All keys present in UnifoLM-VLA-Base dataset_statistics.json
UNIFOLM_NORM_KEYS: Tuple[str, ...] = (
    "g1_bag_insert",
    "g1_clean_table",
    "g1_dual_clean_table_left",
    "g1_dual_clean_table_right",
    "g1_erase_board",
    "g1_fold_towel",
    "g1_organize_tools",
    "g1_pack_pencilbox",
    "g1_pack_pingpong",
    "g1_pour_medicine",
    "g1_prepare_fruit",
    "g1_stack_block",
    "g1_wipe_table",
)


@dataclass(frozen=True)
class UnifoLMTask:
    """One of the 12 UnifoLM G1 real-robot benchmark tasks."""

    id: str
    display_name: str
    unnorm_key: str
    hf_dataset_id: str
    instruction: str
    # Optional Dex1 / alternate corpus used by this research stack.
    alt_hf_dataset_id: Optional[str] = None
    dual_robot: bool = False
    # Dual-robot policies expose left/right unnorm keys.
    unnorm_key_secondary: Optional[str] = None
    # Interactive mocap-cloth wipe metrics only apply to wipe_table.
    supports_wipe_cloth_metrics: bool = False
    notes: str = ""

    @property
    def primary_dataset_id(self) -> str:
        """Preferred HF dataset for ESN / oracle (Dex1 wipe when available)."""
        return self.alt_hf_dataset_id or self.hf_dataset_id

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["primary_dataset_id"] = self.primary_dataset_id
        return d


# Official 12-task suite (UnifoLM README). Dual clean table = one task, two keys.
UNIFOLM_TASKS: Dict[str, UnifoLMTask] = {
    "stack_block": UnifoLMTask(
        id="stack_block",
        display_name="Stack Block",
        unnorm_key="g1_stack_block",
        hf_dataset_id="unitreerobotics/G1_Stack_Block",
        instruction="Stack the blocks.",
    ),
    "bag_insert": UnifoLMTask(
        id="bag_insert",
        display_name="Bag Insert",
        unnorm_key="g1_bag_insert",
        hf_dataset_id="unitreerobotics/G1_Bag_Insert",
        instruction="Insert the object into the bag.",
    ),
    "erase_board": UnifoLMTask(
        id="erase_board",
        display_name="Erase Board",
        unnorm_key="g1_erase_board",
        hf_dataset_id="unitreerobotics/G1_Erase_Board",
        instruction="Erase the board.",
    ),
    "clean_table": UnifoLMTask(
        id="clean_table",
        display_name="Clean Table",
        unnorm_key="g1_clean_table",
        hf_dataset_id="unitreerobotics/G1_Clean_Table",
        instruction="Clean the table.",
    ),
    "pack_pencilbox": UnifoLMTask(
        id="pack_pencilbox",
        display_name="Pack Pencil Box",
        unnorm_key="g1_pack_pencilbox",
        hf_dataset_id="unitreerobotics/G1_Pack_PencilBox",
        instruction="Pack the pencil box.",
    ),
    "pour_medicine": UnifoLMTask(
        id="pour_medicine",
        display_name="Pour Medicine",
        unnorm_key="g1_pour_medicine",
        hf_dataset_id="unitreerobotics/G1_Pour_Medicine",
        instruction="Pour the medicine.",
    ),
    "pack_pingpong": UnifoLMTask(
        id="pack_pingpong",
        display_name="Pack Ping Pong",
        unnorm_key="g1_pack_pingpong",
        hf_dataset_id="unitreerobotics/G1_Pack_PingPong",
        instruction="Pack the ping pong balls.",
    ),
    "prepare_fruit": UnifoLMTask(
        id="prepare_fruit",
        display_name="Prepare Fruit",
        unnorm_key="g1_prepare_fruit",
        hf_dataset_id="unitreerobotics/G1_Prepare_Fruit",
        instruction="Prepare the fruit.",
    ),
    "organize_tools": UnifoLMTask(
        id="organize_tools",
        display_name="Organize Tools",
        unnorm_key="g1_organize_tools",
        hf_dataset_id="unitreerobotics/G1_Organize_Tools",
        instruction="Organize the tools.",
    ),
    "fold_towel": UnifoLMTask(
        id="fold_towel",
        display_name="Fold Towel",
        unnorm_key="g1_fold_towel",
        hf_dataset_id="unitreerobotics/G1_Fold_Towel",
        instruction="Fold the towel.",
    ),
    "wipe_table": UnifoLMTask(
        id="wipe_table",
        display_name="Wipe Table",
        unnorm_key="g1_wipe_table",
        hf_dataset_id="unitreerobotics/G1_Wipe_Table",
        # ICRA stack trains/evals on the Dex1 wipe corpus (same unnorm key).
        alt_hf_dataset_id="unitreerobotics/G1_Dex1_Wipe_Table",
        instruction="Wipe the table with the cloth.",
        supports_wipe_cloth_metrics=True,
        notes="Default ICRA task. Cloth metrics / live wipe use Dex1 demos.",
    ),
    "dual_clean_table": UnifoLMTask(
        id="dual_clean_table",
        display_name="Dual-Robot Clean Table",
        unnorm_key="g1_dual_clean_table_left",
        unnorm_key_secondary="g1_dual_clean_table_right",
        hf_dataset_id="unitreerobotics/G1_DualRobot_Clean_Table",
        instruction="Clean the table (dual robot).",
        dual_robot=True,
        notes="Two unnorm keys (left/right). Pass --unnorm_key to pick a side.",
    ),
}

DEFAULT_TASK_ID = "wipe_table"


def esn_checkpoint_basename(task_id: str = DEFAULT_TASK_ID) -> str:
    """Step-2 checkpoint folder under ``models/``. Wipe keeps legacy name."""
    tid = get_task(task_id).id
    if tid == DEFAULT_TASK_ID:
        return "esn_cuda_ridge"
    return f"esn_cuda_ridge_{tid}"


def list_tasks() -> List[UnifoLMTask]:
    return list(UNIFOLM_TASKS.values())


def get_task(task_id: str) -> UnifoLMTask:
    key = str(task_id).strip().lower().replace("-", "_").replace(" ", "_")
    # Allow unnorm_key or HF suffix aliases.
    if key in UNIFOLM_TASKS:
        return UNIFOLM_TASKS[key]
    for t in UNIFOLM_TASKS.values():
        if key == t.unnorm_key or key == f"g1_{t.id}":
            return t
        if t.unnorm_key_secondary and key == t.unnorm_key_secondary:
            return t
        if key.replace("g1_", "") == t.id:
            return t
        # HF repo short names
        short = t.hf_dataset_id.rsplit("/", 1)[-1].lower()
        if key in {short, short.replace("g1_", "")}:
            return t
        if t.alt_hf_dataset_id:
            alt = t.alt_hf_dataset_id.rsplit("/", 1)[-1].lower()
            if key in {alt, alt.replace("g1_", "").replace("dex1_", "")}:
                return t
    known = ", ".join(sorted(UNIFOLM_TASKS))
    raise KeyError(f"Unknown UnifoLM task {task_id!r}. Choose one of: {known}")


def resolve_unnorm_key(task: UnifoLMTask, unnorm_key: Optional[str] = None) -> str:
    if unnorm_key:
        key = str(unnorm_key).strip()
        if key not in UNIFOLM_NORM_KEYS:
            raise KeyError(
                f"unnorm_key={key!r} not in UnifoLM-VLA-Base stats. "
                f"Known: {', '.join(UNIFOLM_NORM_KEYS)}"
            )
        return key
    return task.unnorm_key


def add_task_arg(parser, *, default: str = DEFAULT_TASK_ID) -> None:
    """Attach ``--task`` / ``--list_tasks`` to an argparse parser."""
    parser.add_argument(
        "--task",
        type=str,
        default=default,
        help=(
            "UnifoLM G1 benchmark task id (one of 12). "
            f"Default: {default}. Use --list_tasks to print the suite."
        ),
    )
    parser.add_argument(
        "--list_tasks",
        action="store_true",
        help="Print the 12 UnifoLM tasks and exit",
    )


def maybe_print_tasks_and_exit(args) -> None:
    if getattr(args, "list_tasks", False):
        print_task_table()
        raise SystemExit(0)


def print_task_table() -> None:
    print(f"{'id':18s} {'unnorm_key':28s} {'dataset':42s} cloth")
    print("-" * 100)
    for t in list_tasks():
        cloth = "yes" if t.supports_wipe_cloth_metrics else "-"
        ds = t.primary_dataset_id.replace("unitreerobotics/", "")
        print(f"{t.id:18s} {t.unnorm_key:28s} {ds:42s} {cloth}")
    print()
    print(f"Norm keys in UnifoLM-VLA-Base ({len(UNIFOLM_NORM_KEYS)}):")
    print(" ", ", ".join(UNIFOLM_NORM_KEYS))


if __name__ == "__main__":
    print_task_table()
