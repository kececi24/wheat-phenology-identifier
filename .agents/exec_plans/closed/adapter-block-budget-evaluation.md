# Adapter block label-budget evaluation

Status: Completed; analysis dated 2026-11-04.

The saved-artifact analysis compared frozen, Q/V LoRA, standard bottleneck,
and transition adapters at 25%, 50%, and 100% labels. It covered 16 LOSO folds
and the same 2,605 test image-days; 3,424 artifact checks passed. The transition
adapter did not beat frozen at any budget. LoRA's small and inconsistent gains
did not establish a reliable advantage. No new training was run in the analysis.

- [Full report and provenance](../../../results/adapter_block_analysis_2026-11-04/REPORT.md)
- [Active follow-up: capacity and loss ablations](../active/adapter-capacity-loss.md)
