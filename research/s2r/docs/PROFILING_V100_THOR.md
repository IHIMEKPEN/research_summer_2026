# Profiling on NVIDIA Tesla V100 & Jetson Thor

Profile the open-source stack (YOLO, Qwen2.5 reasoner, Qwen2.5-VL, ESN) before deploying onto Unitree G1 Edu.

## Quick start

```bash
cd s2r
source .venv/bin/activate
pip install -e ".[models]"   # for real GPU runs

# Always-safe mock timing (no weights download)
python scripts/profile_models.py --backend mock --platform generic \
  --out data/processed/profile_mock.json

# Tesla V100 host with local weights
python scripts/profile_models.py --platform v100 --backend real --device cuda \
  --yolo yolov8n.pt \
  --reasoner Qwen/Qwen2.5-3B-Instruct \
  --vlm Qwen/Qwen2.5-VL-3B-Instruct \
  --out data/processed/profile_v100.json

# Jetson Thor with vLLM servers
python scripts/profile_models.py --platform thor --backend real \
  --reasoner-api http://127.0.0.1:8000/v1 \
  --vlm-api http://127.0.0.1:9000/v1 \
  --out data/processed/profile_thor.json
```

The script writes JSON with:

- GPU identity / VRAM
- mean / p50 / p95 latency
- estimated Hz
- reference latency envelopes for V100 vs Thor

## Platform notes

### Tesla V100 (Volta)

| Item | Guidance |
|---|---|
| Memory | 16GB or 32GB |
| Best precision | FP16 (no FP8/FP4 tensor cores) |
| Good fit | Training ESN + offline VLM, server-side reasoning |
| YOLO | YOLOv8n/s FP16 easily real-time |
| Qwen2.5-3B | Comfortable in FP16 on 16GB |
| Qwen2.5-VL-3B | Prefer 32GB or AWQ; batch=1 |
| Qwen 7B | Use AWQ/GPTQ or offload |

**Reference envelopes (planning, not a warranty):**

| Model | Target latency |
|---|---|
| YOLOv8n | 3–8 ms |
| Qwen2.5-3B reason | 40–120 ms |
| Qwen2.5-VL-3B | 80–250 ms |
| ESN step | << 1–2 ms (budget 10 ms @100Hz) |

### Jetson Thor / AGX Thor (Blackwell)

| Item | Guidance |
|---|---|
| Memory | up to **128GB** LPDDR5X on AGX Thor |
| AI perf | up to ~**2070 FP4 TFLOPS** (T5000 class) |
| Best precision | FP4/FP8 + AWQ via vLLM |
| Good fit | On-robot VLM + LLM + YOLO concurrently |
| Serving | vLLM OpenAI-compatible HTTP |

NVIDIA public Thor materials show strong Qwen2.5-VL-3B token/s vs Orin; use those as upper-bound server metrics, then measure **your** end-to-end camera→decision latency with this script + GUI metrics.

**Reference envelopes:**

| Model | Target latency |
|---|---|
| YOLOv8n | 2–6 ms |
| Qwen2.5-3B reason | 25–80 ms |
| Qwen2.5-VL-3B | 40–150 ms |
| ESN step | << 1–2 ms |

## Realtime budget for bring-pen

| Loop | Rate | Budget |
|---|---|---|
| Camera | 15 Hz | 66 ms |
| YOLO detect | 5–15 Hz | 66–200 ms |
| Qwen-VL caption | ~1 Hz | 1000 ms |
| Qwen reasoner | 4–5 Hz | 200–250 ms |
| VLA token | 2 Hz | 500 ms |
| ESN joint cmd | 50–100 Hz | 10–20 ms |
| G1 high-level loco | 50 Hz | 20 ms |

**Rule:** never put Qwen-VL on the 100Hz path. Keep it async; let YOLO + ESN carry the tight loops.

## Interpreting results

1. If ESN p95 > 2 ms → unexpected; check Python overhead / debug prints.  
2. If reasoner p95 > 200 ms on Thor → enable quantization / shorter `max_new_tokens`.  
3. If VLM p95 > 300 ms → reduce image size (640px), use 3B AWQ, or lower `vlm_every_n`.  
4. End-to-end instruction→first-motion is **not** only model latency — include ZMQ + bridge. Use GUI node metrics.

## Suggested MIG / process layout on Thor

```text
GPU instance A: vLLM Qwen2.5-3B reasoner (:8000)
GPU instance B: vLLM Qwen2.5-VL-3B (:9000)
CPU/GPU small : YOLO + ESN + ZMQ nodes
```

## CI-friendly smoke

```bash
python scripts/profile_models.py --backend mock --skip-vlm --iters 10 \
  --out data/processed/profile_ci.json
```
