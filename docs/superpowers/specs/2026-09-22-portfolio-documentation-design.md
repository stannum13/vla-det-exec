# Deterministic VLA portfolio documentation design

## Objective

Explain why deterministic inference is a robotics execution problem rather than
a seed-setting trick. The README should make the implemented harness, unfinished
checkpoint benchmark, and next executor experiment obvious without presenting
stand-in tests as model evidence.

## Narrative

The repository will lead with the executable-semantics question: given a fixed
observation history, checkpoint, metadata, decoding rule, and controller state,
can the software produce one auditable command sequence?

The existing harness is presented as finished engineering infrastructure.
The real-checkpoint benchmark remains the immediate empirical milestone. The
README will not lead with "no result"; it will lead with the working comparison
system and the decision it is ready to make.

## README structure

1. A concise thesis and status callout.
2. A visual flow from fixed observation through repeated inference to metrics.
3. A model-condition matrix for ACT, VQ-BeT, and MolmoAct2.
4. A clear explanation of exact agreement, RMS drift, entropy, latency tails,
   and deadline misses.
5. An implemented-versus-next table.
6. An accurate repository map and runnable commands.
7. A "current frontier" section that connects inference repeatability to a
   phase-gated executor and perturbation test.
8. Boundaries: no physical-robot, checkpoint, or capability claim yet.

## Visual design

Add two lightweight, accessible explanatory SVGs under docs/media:

- deterministic-execution-pipeline.svg
- experiment-zero-matrix.svg

They will use a restrained graphite/blue/amber palette and embedded text, with no
robot photograph or synthetic result curve. Captions will say "experiment design"
or "execution model," never "result."

If real Experiment 0 artifacts become available later, the README reserves a
clearly named measured-results section rather than mixing them with diagrams.

## Current frontier and next experiments

Phase 1 — complete the real-checkpoint determinism benchmark:

1. freeze 50 LIBERO observations;
2. run 100 calls per model and condition;
3. publish exact/tolerance agreement, drift, entropy, latency, and deadline data;
4. create the determinism-versus-capability frontier only when capability results
   exist.

Phase 2 — test deterministic response rules:

1. compile phase predicates and action limits into an execution boundary;
2. compare fixed and predicate-bounded chunk horizons;
3. introduce a scripted object displacement and failed grasp;
4. score stale continuation, replan, retry, safe abort, unsafe behavior, and
   intervention burden.

Teacher-to-student transfer remains gated on a stable executor baseline.

## Scope

Documentation and explanatory media only:

- README.md
- docs/media/deterministic-execution-pipeline.svg
- docs/media/experiment-zero-matrix.svg
- no checkpoint downloads, training runs, or result generation

## Verification

- Confirm all commands match the implemented CLI.
- Confirm all repository links and paths exist.
- Render both SVGs and inspect them at README display width.
- Run the model-independent test suite.
- Scan the final copy for language that implies the real checkpoint experiment
  has already run.

