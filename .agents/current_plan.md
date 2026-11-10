# Current plan

Last updated: 2026-11-10

Read [AGENTS.md](AGENTS.md), then the relevant ExecPlan. For historical results, see [summary.md](summary.md).

- [Magnification sorter](exec_plans/active/magnification-sorter.md): Smoke-test one fold on the machine with the saved weights, using a visually verified image.
- [DINOv3 PEFT capacity and loss](exec_plans/active/adapter-capacity-loss.md): Prepare the width 128/32 × hybrid/soft-target CE ablation on the frozen baseline.
- [Visual developmental progression](exec_plans/active/visual-progression.md): Specify season-bounded training pairs and the matched softmax/ordinal controls.
- [FSDP2 temporal training](exec_plans/active/fsdp2-temporal.md): Run the gate-logging smoke test on two CUDA ranks.
- [Backbone/config compatibility](exec_plans/active/backbone-compatibility.md): Dry-run the local vit-base, deits, and beit config directories; select one fold for a smoke test.
