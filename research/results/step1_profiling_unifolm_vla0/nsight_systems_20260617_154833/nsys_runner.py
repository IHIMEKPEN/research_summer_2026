
import os
import sys
from pathlib import Path

RESEARCH_DIR = Path('/home/aihimekpen/research_summer_2026/research')
if str(RESEARCH_DIR) not in sys.path:
    sys.path.insert(0, str(RESEARCH_DIR))
os.chdir(RESEARCH_DIR)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from src.step1_profile_unifolm_vla0 import MockG1CleanTableEnv, UnifoLMVLAWrapper

TASK_LABEL = 'G1_Clean_Table'
INSTRUCTION = 'Clean the table by moving all clutter items into the bin.'
MODEL_ID = 'unitreerobotics/UnifoLM-VLA-Base'
VLM_BACKBONE_ID = 'unitreerobotics/UnifoLM-VLM-Base'
UNNORM_KEY = 'g1_clean_table'
MAX_NEW_TOKENS = 64
CUDA_GRAPH_WARMUP = 2
WARMUP_STEPS = 1
PROFILE_STEPS = 3

env = MockG1CleanTableEnv(image_size=(224, 224))
model = UnifoLMVLAWrapper(
    model_id=MODEL_ID,
    vlm_backbone_id=VLM_BACKBONE_ID,
    unnorm_key=UNNORM_KEY,
    use_int4=False,
    action_dim=29,
    allow_mock_fallback=False,
    max_new_tokens=MAX_NEW_TOKENS,
    cuda_graph_warmup=CUDA_GRAPH_WARMUP,
)

obs = env.reset()
for _ in range(WARMUP_STEPS):
    torch.cuda.nvtx.range_push('vla_action_generation_warmup')
    action, _, _ = model.infer(obs, INSTRUCTION)
    torch.cuda.nvtx.range_pop()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    obs, _, done, _ = env.step(action)
    if done:
        obs = env.reset()

for _ in range(PROFILE_STEPS):
    torch.cuda.nvtx.range_push('vla_action_generation_profiled')
    action, _, _ = model.infer(obs, INSTRUCTION)
    torch.cuda.nvtx.range_pop()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    obs, _, done, _ = env.step(action)
    if done:
        obs = env.reset()

print('NSYS target run complete')
