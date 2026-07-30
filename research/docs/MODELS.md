# Open-source models in R2S

This pipeline uses **pluggable open-source models** with mock fallbacks so you can develop without GPUs, then flip to real weights on Tesla V100 / Jetson Thor / G1 Edu.

## Stack

| Role | Default OSS model | Node | Rate |
|---|---|---|---|
| Object detection | **Ultralytics YOLOv8n** (`yolov8n.pt`) | `vision` | ~5–15 Hz |
| Scene VLM | **Qwen2.5-VL-3B-Instruct** | `vision` (every N frames) | ~1 Hz |
| Mission reasoning | **Qwen2.5-3B-Instruct** | `reasoning` | ~4–5 Hz |
| Action tokens | Mock / OpenVLA / π0 / GR00T hook | `vla` | ~2 Hz |
| Dynamics upsample | **Echo State Network** | `esn` | 50–100+ Hz |

## Why this split?

- **YOLO** keeps fast closed-loop object presence (pen/table/person).
- **Qwen2.5-VL** adds language-grounded scene understanding (“pen on table, graspable?”).
- **Qwen2.5 LLM** produces structured JSON decisions (intent/risk/allow_motion).
- **ESN** absorbs sparse VLA tokens into smooth high-rate joint commands without GPU.

## Enable real models

### Local transformers (V100 workstation)

```bash
pip install -e ".[models]"
# edit config/default.yaml
models:
  detector: { mock: false, model_id: yolov8n.pt, device: cuda }
  vlm: { mock: false, model_id: Qwen/Qwen2.5-VL-3B-Instruct, device: cuda }
  reasoner: { mock: false, model_id: Qwen/Qwen2.5-3B-Instruct, device: cuda }
```

### vLLM / OpenAI-compatible server (recommended on Jetson Thor)

```bash
# Terminal A — reasoner
vllm serve Qwen/Qwen2.5-3B-Instruct --host 0.0.0.0 --port 8000

# Terminal B — VLM
vllm serve Qwen/Qwen2.5-VL-3B-Instruct \
  --host 0.0.0.0 --port 9000 \
  --trust-remote-code \
  --gpu-memory-utilization 0.75
```

```yaml
models:
  reasoner:
    mock: false
    api_base: http://127.0.0.1:8000/v1
    model_id: Qwen/Qwen2.5-3B-Instruct
  vlm:
    mock: false
    api_base: http://127.0.0.1:9000/v1
    model_id: Qwen/Qwen2.5-VL-3B-Instruct
```

### Quantization tips

| GPU | Prefer |
|---|---|
| Tesla V100 | FP16, AWQ/GPTQ INT4 for 7B if VRAM tight |
| Jetson Thor | FP8/FP4, AWQ, vLLM; start with **3B** VL + **3B** LLM |
| G1 Edu onboard | Same as Thor/Orin class: serve 3B models, YOLO-nano always on-device |

## Mission JSON contract (reasoner)

```json
{
  "intent": "explore|locate_pen|approach_table|grasp_pen|return_to_user|handoff|hold",
  "reason": "...",
  "risk": 0.2,
  "allow_motion": true,
  "tags": ["bring_pen"]
}
```

## Swapping models

Edit only `config/*.yaml` — factories live in `src/s2r/models/registry.py`.

Suggested upgrades later:
- Detector: YOLO11n / RF-DETR
- VLM: Qwen2.5-VL-7B or Qwen3-VL when runtime supports it
- Reasoner: Qwen2.5-7B-Instruct / Qwen3-4B
- VLA: OpenVLA, π0, Isaac GR00T N1.5
