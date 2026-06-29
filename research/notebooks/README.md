# Notebooks

| Notebook | Purpose | Where to run |
|----------|---------|--------------|
| `step1_mock_profiling.ipynb` | Dry-run logging pipeline (no model download) | Mac / any machine |
| `step1_openvla_profiling.ipynb` | **Real** OpenVLA-7B latency profiling | Lab GPU (NVIDIA CUDA) |
| `step2_esn_cuda_ridge.ipynb` | **CUDA ESN** ridge regression on G1_Dex1_Wipe_Table | Lab GPU (NVIDIA CUDA) |
| `step3_dual_thread_mujoco.ipynb` | **Dual-process** VLA + ESN (GIL-free, latency) | Lab GPU |
| `step4_mujoco_evaluation.ipynb` | **MuJoCo eval** dataset oracle + cloth grasp + video | Lab GPU |

Run the setup cell first in either notebook so `import src` works.
