# Visual developmental progression

Last updated: 2026-11-10
Status: proposed, not implemented

## Sources and handoff

[Task](thread://01a1a647-df94-7893-a7a0-36b47cc6992e?hostId=local)

- Start with frozen features, fixed spatial pooling, and a small image head.
  Predict a scalar progression score and use six learned ordered thresholds
  for the seven BBCH stages. Sparse soft BBCH labels anchor the scale;
  within-season chronological pairs penalize reversed order. Dates form
  training pairs only, so inference remains single-image.
- Next: compare a small softmax head, a matched ordinal head, ordinal plus
  true chronological pairs, and then station-balanced loss. Include a
  shuffled-order control; exclude harvest/reset crossings and use unlabeled
  images only from training stations with matched access for relevant
  controls. Assess pooled and mean-station macro-F1 and difficult stages
  alongside reversals. Accuracy and novelty are unestablished.

## Immediate next action

Specify season-bounded training pairs and the matched softmax/ordinal controls.

## Preflight and blockers

No implementation or training run is confirmed. Verify usable season boundaries and frozen image features before implementing the pilot.

## Validation and completion criteria

Evaluate true versus shuffled chronology and station balancing on held-out stations. Report both stage performance and reversals; fewer reversals alone do not establish a gain.
