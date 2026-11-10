# FSDP2 temporal training

Last updated: 2026-11-10
Status: smoke test passed remotely; integration pending

## Sources and handoff

[Task](thread://01a158d4-423f-7630-b7d0-e2c4050c02b1?hostId=local) ·
[Runtime](../../../DINOv3_BBCH/distributed_runtime.py) ·
[Smoke test](../../../DINOv3_BBCH/tests/fsdp2_smoke.py)

- The two-rank synthetic smoke test was reported to pass sharding,
  forward/backward, optimizer updates, and evaluation. The local tree also
  has a DTensor-aware gate reader, trainer readers, and smoke-test gate
  checks; this handoff does not verify a two-rank run of the gate checks.
- Next: run the gate smoke on two GPUs and confirm every rank participates
  in gate reads in identical order, with output written only by rank zero.
  Then test a model/optimizer checkpoint round-trip and integrate sharding
  before optimizer creation. Full multi-GPU trainer, resume, and metric
  parity are not yet established.

## Immediate next action

Run the gate-logging smoke test on two CUDA ranks.

## Preflight and blockers

Needs two usable CUDA GPUs and a compatible PyTorch distributed runtime. The earlier two-rank pass did not verify the new gate reads.

## Validation and completion criteria

Confirm all ranks participate in gate reads, then verify a model/optimizer checkpoint round-trip before integrating sharding into the trainer and checking resume and metric parity.
