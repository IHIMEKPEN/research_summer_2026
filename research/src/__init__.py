"""
VLA + ESN research code for Unitree G1 / OpenVLA profiling.

Import from submodules, e.g.:
    from src.step1_profile_openvla import profile_openvla, main
"""

from .paths import RESEARCH_ROOT, models_path, result_file, results_path

__all__ = [
    "RESEARCH_ROOT",
    "results_path",
    "models_path",
    "result_file",
]
