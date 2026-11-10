# Execution Environment Gate

Before substantial execution, determine required runtimes, services, credentials, and setup from repository instructions, manifests, and scripts, then run a bounded preflight.

* Verify required executables/versions and services once.
* Do not begin implementation, debugging, testing campaigns, or security work until required dependencies are usable, except for preflight diagnostics.
* On failure, report the exact blocker and stop for user action instead of repeatedly retrying, elevating, or globally installing.
* After remediation, re-run only relevant failed checks.
* Persist concise preflight blockers when planning state exists.
# Version Control Safety

Git is read-only unless the user explicitly authorizes mutation for the current task.

Allowed: git status, git diff, git log, git show, and equivalent non-mutating inspection.

Do not run mutating Git operations including add, commit, push, pull, merge, rebase, checkout/switch, reset, restore, clean, stash, or branch/tag creation/deletion without explicit authorization.

# Persistent Planning

For substantial multi-step engineering work, treat planning state as a repository invariant.

Use an existing repository convention when present. Otherwise maintain:

* .agents/current_plan.md — concise immediate execution state.
* .agents/exec_plans/active/ — durable ExecPlans for substantial active workstreams.
* .agents/exec_plans/closed/ — completed, abandoned, or superseded ExecPlans.

If PLANS.md exists, read it before planning and treat it as repository-specific policy rather than duplicating it globally.

Each active ExecPlan represents one coherent workstream and preserves durable context: goals, scope, architecture, discoveries, decisions, blockers, validation, and remaining work.

current_plan.md contains only current work and immediate next tasks for each active plan, normally within ~10 lines per workstream. Keep tasks concrete and short-horizon; broader strategy belongs in the ExecPlan. Remove completed tasks instead of accumulating history.

The primary orchestrator owns planning state unless ownership is explicitly delegated.

During substantial work:

1. Read current_plan.md, the relevant ExecPlan, and PLANS.md if present before resuming.
2. Keep current_plan.md aligned with actual current/next work.
3. Update the ExecPlan after material discoveries, decisions, blockers, scope changes, milestones, or verification.
4. Before compaction, handoff, ownership transfer, or ending unfinished work, persist enough state for a fresh agent to resume without chat history.
5. When a workstream ends, remove it from current_plan.md, record its outcome, and move its ExecPlan to exec_plans/closed/.

Do not create plans for trivial single-step work. For more detail, inspect [PLANS.md](PLANS.md)

# Context Hygiene and Subagents

Treat the primary agent's context as a scarce orchestration resource.

For bounded, context-heavy work, prefer subagents by default, especially for:
- repository search and code discovery;
- dependency/call-path tracing;
- documentation discovery or reconciliation;
- test execution and large test shards;
- test-failure analysis;
- independent code review;
- module/package-specific investigation;
- large log/output analysis;
- API/library usage discovery.

Keep cross-cutting decisions, integration, plan state, and final completion claims with the primary agent. Do not spawn subagents for trivial lookups or tightly coupled edits.

Give subagents a narrow question/scope and require compact handoff evidence: files/symbols inspected, commands/results, conclusions, uncertainties, and artifact/log paths. Do not return raw retrieval dumps, full documentation, or large logs to the primary unless required as proof.