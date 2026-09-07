# Chart Runtime 0.4.0

## Star control

`starTargetRatio` is now the direct target-Star ratio against the calibrated official-chart reference. On controlled EXPERT, MASTER, and Re:MASTER charts, the CUDA Harness uses the resulting integer as both lower and upper Star bounds; the Generator adds or removes Star candidates through the normal feedback loop.

## WHAT / WHERE separation

EXPERT, MASTER, and Re:MASTER now use `JointEventPlanModel` to choose geometry-free `EventIntent` values before V4 realization. Candidate SOFT findings only rerank or resample WHERE for the current WHAT; a lower-ranked WHAT is considered only when every realization of the current WHAT is HARD.

## Hand-transition preference

Two train-derived SOFT WHERE preferences were added without changing HARD legality. During Hold or Touch Hold occupancy, the free hand is ranked by displacement and direction-change cost while smooth adjacent same-direction rotation is explicitly preserved. A narrow Slide-entry preference also catches abrupt Slide pickup after a three-Tap adjacent directional run.

The Home Street Re:MASTER regression keeps the Slide at tick 27072 while reducing local `motionSpeed` from 14.13 to 7.07 and `motionChange` from 21.2 to 0. High-motion Slide coverage is unchanged at 26/109 before and after the final narrow policy.

## Validation

- Native five-difficulty A/B at `starTargetRatio=0.5`: controlled Stars 58/94/86 against references 117/189/172.
- Native five-difficulty A/B at `starTargetRatio=1.0`: controlled Stars 117/189/172.
- Final Home Street Re:MASTER end-to-end run: accepted, HARD=0, quality=0.
- Intent contract tests: 3/3 passed; Star policy tests: 2/2 passed; Python compilation passed.
- Simai re-read digests match accepted CUDA IR before publication.