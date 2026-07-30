# ESN ablation study (with vs without dynamic engine)

This guide supports research comparing the full S2R pipeline **with ESN** as a buffer/dynamic upsampler versus **without ESN**.

## Why this matters

VLA models typically emit sparse action tokens (~1–5 Hz). Real robots need denser joint commands (50–100 Hz+). Candidates:

| Engine | Behavior |
|---|---|
| `esn` | Learned/reservoir dynamics upsample tokens → smooth high-rate cmds |
| `passthrough_zoh` | Zero-order hold: repeat last token at high rate (buffer, no dynamics) |
| `passthrough_linear` | Linear interpolate between tokens |
| `passthrough_raw` | No buffer: publish cmds only when tokens arrive (~2 Hz) |

## Quick runs

```bash
# WITH ESN (GUI :8080, logs in data/raw/ablation_with_esn)
python -m s2r.cli deploy -c config/ablation_with_esn.yaml

# WITHOUT ESN — ZOH (GUI :8081)
python -m s2r.cli deploy -c config/ablation_no_esn_zoh.yaml

# WITHOUT ESN — raw 2Hz (GUI :8082)
python -m s2r.cli deploy -c config/ablation_no_esn_raw.yaml
```

Or flip any config:

```yaml
pipeline:
  control_engine: esn          # or: passthrough | zoh | linear | raw
passthrough:
  mode: zoh                    # used when not esn
```

## Inspect + compare

```bash
# Robotics distribution / fitness of any logs
python -m s2r.cli inspect-data -s data/raw

# Compare ablation folders automatically
python -m s2r.cli compare-ablation
```

Notebooks:

- `notebooks/05_inspect_data_distributions.ipynb`
- `notebooks/06_esn_ablation_compare.ipynb`

## Metrics reported

- token / command / state Hz
- upsample ratio (`cmd_hz / token_hz`)
- command jerk proxy (smoothness)
- latency distribution
- tracking error (state vs command)

## Suggested paper plot set

1. Trajectory overlay of `joint_cmd[0]` for ESN vs ZOH vs raw  
2. Bar chart: upsample ratio  
3. Bar chart: jerk p95  
4. Latency table for ESN inference vs VLA token period  
