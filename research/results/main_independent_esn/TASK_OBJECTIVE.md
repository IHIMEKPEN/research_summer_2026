# Wipe-task objective and validation policy

## Objective used by the experiment

The simulator reports five independently interpretable, dimensionless terms:

1. grasp failure (binary);
2. normalized wipe-path shortfall;
3. normalized table-contact shortfall;
4. normalized target-area coverage shortfall;
5. small smoothness and joint-limit penalties.

The first four terms have equal unit weight after normalization. This avoids
claiming that an arbitrary numerical coefficient is scientifically meaningful.
The sparse frozen-UnifoLM MSE is a separate term and is evaluated only every
570 ms.

Success requires all of:

- a proximity-gated cloth grasp;
- at least **0.768 m** cloth path while grasped (the empirical 5th percentile
  of right-EE closed-gripper path length across all 200
  `G1_Dex1_Wipe_Table` demonstrations; measured locally on 2026-08-26);
- at least **90%** table-contact ratio during the grasped phase;
- coverage of at least **90%** of the declared 0.2016 m² table top; coverage
  outside its XY bounds is discarded;
- no joint-limit failure and no per-tick cloth jump above **5 cm**.

## Evidence and limits

- Surface coverage is a standard cleaning metric. A robotic swabbing study
  defines coverage as activated cells divided by total cells and reports
  90–100% coverage for trained humans and 99.5% for its robot. It also uses a
  contact-pressure activation threshold. This supports a percentage-based
  coverage gate, not the prior unnormalized footprint-area score:
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC12523492/>.
- Wiping is contact-rich and requires force adaptation; loss of contact means
  the robot is not effectively wiping. This supports treating contact as a
  mandatory gate rather than a small optional bonus:
  <https://arxiv.org/abs/2505.06451>.
- Robotic wiping reward design is explicitly quality-critical: naive sums of
  dense quality reward and sparse completion reward can be poorly behaved.
  The experiment therefore reports every component and uses bounded,
  normalized shortfalls:
  <https://arxiv.org/abs/2502.12599>.
- Contact distribution and smooth control inputs are relevant to wiping across
  different surface materials:
  <https://arxiv.org/abs/2403.11198>.

## Important simulation limitation

The current cloth is mocap-driven and has no calibrated force/torque sensor.
`table_contact_ratio` is therefore a geometric contact proxy based on cloth
height, not a Newton-valued force measurement. The experiment must not claim
validated force control. A later hardware or compliant-contact experiment must
add a calibrated force sensor and define a task-specific safe force band.

The 5 cm jump gate is a simulator-integrity check that rejects policies which
exploit the mocap attachment instead of wiping. **Stabilized accounting (2026-08-27):**
jumps above 5 cm are excluded from wipe-path and coverage accumulation before
they can inflate scores; rollouts terminate immediately on NaN/QACC instability;
every reported loss component is bounded to `[0, 1]` and logged separately
(`L_grasp`, `L_path`, `L_contact`, `L_coverage`, `L_smooth`, `L_limits`,
`L_teacher`). Held-out / `teacher_weight=0` evaluations must report
`teacher_source="none"` with `L_teacher=0` — never demonstration-proxy loss
disguised as a real teacher.

The 90% contact and coverage gates are preregistered engineering criteria,
motivated by the sources above. They are not universal standards. Results must
include sensitivity analysis at 80%, 90%, and 95%.

**Go/no-go before further ESN optimizer work:** demonstration replay must achieve
stable contact and a plausible path on ≥90% of evaluated episodes with zero NaNs
(`notebooks/run_oracle_benchmark_gate.py`). See `BENCHMARK_STABILIZATION.md`.
