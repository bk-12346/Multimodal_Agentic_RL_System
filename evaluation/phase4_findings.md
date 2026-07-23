# Phase 4 — Deployment Findings

## Experiment tracking: S3-based logging chosen over MLflow

For this project's scale (~15 training runs), a lightweight custom pattern
(training/finalize_run.py -> S3, storing model + JSON metrics + notes per
run) was used instead of standing up MLflow. This was a deliberate scope
decision: MLflow's comparison UI and experiment-versioning features add
real value at a larger scale (dozens-hundreds of runs, team collaboration),
but for this project's run count, a simpler S3 convention was more
proportionate engineering effort. MLflow experience is demonstrated in
prior professional work rather than this project specifically.