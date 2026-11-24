# Loss BBCH implementation record

Date: 2026-11-12. Status: implemented; remote execution pending.

The repository planning directory `.agents` is read-only in this session;
this local workstream record preserves state without modifying protected files.
Read `.agents/AGENTS.md`, current_plan.md, visual-progression.md and
adapter-capacity-loss.md. `.agents/PLANS.md` is absent.

Scope: frozen original DINOv3 cache, fixed spatial pooling, identical small
image-only heads. Compare hard CE, soft CE, squared CDF ordinal loss, and
soft CE plus marginal likelihood of monotone season paths constrained by
selected-label support. No transformer/ViT fine-tuning, adapters or temporal
inference. No claim of novelty or established improvement.

Preflight: Python 3.11.9 available; repository cache/label/fold APIs inspected.
No model runtime is needed for editing. Dataset/cache/runtime checks are
implemented for the user's remote run. No local training or tests authorized.

Work allocation: Luna max agents own losses/tests, runner/config, and a
read-only season/leakage review. Primary owns documentation, configurations,
final integration review. Git is read-only.

Validation plan: static review only here; provide remote unit tests for
path enumeration, gradients and infeasible constraints. Runtime must fail on
invalid cache, overlapping folds, impossible anchors and stale output folders.
No assertions of passing tests until the user runs them remotely.

Completed: four loss functions, train-only season masks, strict YAML/CLI,
matched full-batch MLP runner, 25/50/100% configs, source/data provenance,
validation-only checkpoint selection and held-out predictions. Added mathematical,
configuration and synthetic training integration tests for remote execution.
Static review fixed an invalid class-encounter-order check, empty index handling,
and a mistakenly monotone negative-test fixture. Added a guard against merged
planting cycles. AST syntax parsing succeeded for all seven Python files;
no model imports, pytest runs or training were executed by the primary agent.

Remote next action: edit paths in YAML, run --dry-run and pytest, then one LOSO
fold before the complete experiment. Sequence loss uses existing soft-target
support, not measured transition-date uncertainty; reduced-label results are
semi-supervised and need an equal-unlabeled-access control for a stronger claim.
