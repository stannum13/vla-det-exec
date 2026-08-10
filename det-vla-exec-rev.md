# Can VLA-Deployed Robot Tasks Be Made Deterministic?

## Literature review and a low-infrastructure Franka Panda proof of concept

**Date:** 9 August 2026  
**Scope:** Action Chunking Transformer (ACT) baseline, VQ-BeT, MolmoAct2, deterministic execution, action abstraction, runtime monitoring, and a real-tabletop experiment that avoids building a custom simulator.

## Executive conclusion

Yes, but the useful target is **deterministic executable semantics**, not identical physical trajectories.

A deployed robot system can be made deterministic in the following sense:

> Given a fixed observation history, task instruction, checkpoint, action normalization metadata, random seed or fixed latent, numerical backend, replanning rule, and controller state, the software produces one auditable sequence of authorized commands.

That does **not** imply that repeated physical runs follow exactly the same path. Camera noise, timing, friction, backlash, object pose, grasp contact, and controller state change the observations. A good closed-loop robot should react differently when the world differs. Requiring one invariant open-loop trajectory would often reduce robustness.

The most promising architecture is therefore:

1. a VLA or imitation policy proposes an action chunk, discrete skill, phase, or sparse waypoint;
2. deterministic decoding selects one proposal;
3. a deterministic executor projects it into a bounded action set;
4. phase predicates and a runtime monitor decide whether execution may continue, replan, retry, or abort;
5. a conventional Panda controller tracks the authorized setpoints.

This is closer to **compiling a learned policy into a guarded reactive executable** than compiling it into one fixed trajectory.

For a fast proof of concept, do not build a new MuJoCo scene. Use:

- the real Franka Panda and its existing controller;
- one fixed third-person camera;
- a printed placement grid;
- a green object and green bin;
- 40–60 successful demonstrations;
- recorded-observation replay for cheap determinism tests;
- 6–12 repeatable physical start configurations;
- simple color, geometry, robot-state, and timeout predicates.

Train ACT first. Add VQ-BeT through the same dataset/action adapter. Treat MolmoAct2-DROID initially as a pretrained Franka policy/teacher and semantic baseline, not as infrastructure that must be modified before the central hypothesis can be tested.

---

## 1. What “deterministic” can mean

The word hides several different research questions. They should be measured separately.

| Level | Desired property | Achievable? | Appropriate test |
|---|---|---:|---|
| Decoder repeatability | Same tensors in produce the same action tensors out | Mostly yes | Re-run identical recorded inputs 100 times and hash/compare outputs |
| Timing determinism | Commands arrive within a fixed deadline and execution schedule | Bounded, not automatic | Measure mean, p95, p99, maximum latency and deadline misses |
| Mode determinism | A multimodal policy always selects the same behavioral mode | Yes, by fixed latent/argmax/canonical selection | Compare fixed decoding with stochastic sampling |
| Executable equivalence | Checkpoint, normalization, action convention, controller mapping, and camera order define the same physical policy | Yes, if explicitly specified | Version and test an executable-policy manifest |
| Trajectory repeatability | Same nominal initial condition yields a narrow distribution of physical trajectories | Approximately | Repeat fixture-defined starts; measure endpoint/path covariance |
| Constraint determinism | Joint, velocity, workspace, phase-order, and abort constraints are never violated | Often enforceable within modeled limits | Runtime shield plus trace checking |
| Outcome determinism | The task always succeeds | No general guarantee | Report empirical success with confidence intervals |

The important conceptual distinction is:

\[
\text{deterministic policy mapping} \neq \text{deterministic world trajectory} \neq \text{guaranteed success}.
\]

For this project, the strongest defensible claim is likely:

> The learned proposal distribution was converted into a repeatable, deadline-aware, phase-gated executable whose authorized command trace is deterministic for a fixed sensor history, while preserving closed-loop correction under changed observations.

---

## 2. What the three model families contribute

### 2.1 ACT: the best first baseline

ACT predicts a future action chunk rather than one action at a time. Its original formulation uses a conditional VAE to represent variation in demonstrations and temporally ensembles overlapping chunk predictions. It achieved 80–90% success on several fine-manipulation tasks from roughly 50 demonstrations, which is why it remains a strong low-data baseline [Zhao et al., 2023](https://arxiv.org/abs/2304.13705). The official implementation is compact, and LeRobot recommends ACT as a lightweight starting policy because of its training speed and low compute requirements ([official ACT repository](https://github.com/tonyzhaozh/act), [LeRobot ACT guide](https://huggingface.co/docs/lerobot/en/act)).

ACT is useful here for three reasons:

1. **It isolates the deployment question.** There is no large VLM, language decoder, or iterative flow sampler to confuse policy capability with execution semantics.
2. **Its chunk interface is easy to intercept.** One can change latent selection, chunk commitment, temporal aggregation, projection, and phase gating without retraining the vision encoder.
3. **It is already close to deterministic at inference.** A fixed latent/prior sample, evaluation mode, fixed preprocessing, and deterministic numerical path turn it into a single-valued function of observations. The official code separates training-time latent inference from inference-time generation ([ACT policy implementation](https://github.com/tonyzhaozh/act/blob/main/policy.py)).

ACT is not a guarantee by itself. Temporal ensembling can make motion smooth, but it also makes a command depend on an implicit history of prior chunk predictions. The contents and indexing of that history are part of the executable specification.

### 2.2 VQ-BeT: discrete modes are useful, but not automatically safe

VQ-BeT learns a hierarchical vector-quantized representation of continuous actions and predicts discrete latent actions plus continuous corrections. It was designed to preserve multimodal behaviors without the scaling weaknesses of k-means action clusters, and the ICML paper reports roughly 5× faster inference than the compared diffusion policies across its test domains [Lee et al., 2024](https://proceedings.mlr.press/v235/lee24y.html). Official code is available in the [VQ-BeT repository](https://github.com/jayLEE0301/vq_bet_official).

VQ-BeT supplies a natural experimental handle:

- **stochastic mode:** sample a code/token;
- **repeatable mode:** choose the argmax token;
- **canonical mode:** choose the highest-probability token that passes a verifier;
- **auditable mode:** log the token sequence, codebook version, residual, and decoded action chunk.

This is closer to a discrete instruction trace than ACT's raw continuous chunk. However, discreteness alone does not provide determinism or guarantees:

- sampled tokens remain stochastic;
- codebook boundaries can cause discontinuous action changes;
- a deterministic residual decoder may still issue unsafe continuous actions;
- an argmax choice can consistently select the wrong mode;
- token identity is not necessarily semantically stable across retraining.

The right hypothesis is not “VQ-BeT is deterministic.” It is:

> A discrete latent interface makes behavioral mode selection easier to canonicalize, inspect, cache, and constrain than a directly sampled continuous trajectory.

### 2.3 MolmoAct2: strongest spatial teacher, hardest direct determinism baseline

MolmoAct2 combines a spatially specialized VLM with a continuous flow-matching action expert. At inference, its action expert starts from Gaussian noise and integrates a learned velocity field into a continuous trajectory [MolmoAct2 paper](https://arxiv.org/html/2605.02881v1). Therefore its native direct-control path is stochastic unless the initial noise and all subsequent execution details are fixed.

The release is unusually relevant to this experiment:

- it includes a DROID/Franka checkpoint with absolute joint-pose control;
- it includes real-world Franka deployment guidance;
- it is integrated with LeRobot workflows;
- its code, weights, tokenizer, and training data are open ([official MolmoAct2 repository](https://github.com/allenai/molmoact2)).

MolmoAct2 also pretrains a discrete action-token interface and then attaches a continuous flow expert. This makes it a useful example of a broader pattern: discrete supervision can stabilize representation learning while a continuous action expert handles precise control.

For repeatable inference, fix:

- Gaussian initialization;
- flow solver and number of steps;
- prompt template and decoding settings;
- camera order, crop, calibration, and image preprocessing;
- state/action normalization key;
- chunk horizon and replanning schedule;
- CUDA/PyTorch deterministic settings where supported.

This gives **numerical repeatability within a tolerance**, not necessarily portable bitwise equality across GPU models and software versions.

MolmoAct2 should enter the proof of concept in three roles, in this order:

1. **frozen-log inference baseline** using fixed versus random flow noise;
2. **teacher/labeler** for phase, object anchor, and difficult spatially shifted states;
3. **direct Panda policy** initialized from MolmoAct2-DROID if the existing camera/action conventions are sufficiently close.

Starting by reproducing its entire direct-control stack would test integration effort more than the determinism hypothesis.

---

## 3. Literature strands that matter for “compilation”

### 3.1 Action tokenization and temporal abstraction

Action tokenization makes a continuous trajectory addressable through a shorter symbolic sequence. This is useful for caching, constrained selection, trace inspection, and hierarchical control, but it is not itself a formal program.

- **VQ-BeT** uses hierarchical vector quantization to model multimodal continuous behavior [Lee et al., 2024](https://proceedings.mlr.press/v235/lee24y.html).
- **PRISE** combines continuous action quantization with byte-pair-style sequence compression to discover variable-length temporal action abstractions [Zheng et al., 2024](https://arxiv.org/html/2402.10450v1).
- **QueST** learns compressed latent action sequences as reusable skills and reports benefits for few-shot and multitask imitation learning [Sontakke et al., 2024](https://arxiv.org/html/2407.15840v2).
- **FAST** applies a discrete cosine transform and compression-based tokenization to action chunks, addressing the poor scaling of per-dimension, per-timestep discretization for high-frequency dexterous actions [Pertsch et al., 2025](https://arxiv.org/abs/2501.09747).

These methods support an important design principle: **compile at a temporal abstraction boundary**. Do not try to formalize every 1 kHz torque. Formalize a small vocabulary of phases, modes, sparse waypoints, or action chunks, then delegate tracking to a conventional controller.

### 3.2 Sparse semantic interfaces and deterministic rendering

The closest recent work to the proposed architecture is NoTVLA. It predicts sparse, semantically meaningful waypoints around contact events and uses a deterministic detokenizer to reconstruct high-frequency motion [NoTVLA, 2026](https://arxiv.org/html/2510.03895v2). Its core separation—semantic decision making versus transparent motion rendering—is exactly the direction worth testing.

Behavior trees are another executable intermediate. Recent work uses VLMs to synthesize behavior trees from images, instructions, and system specifications [Learning Structured Robot Policies, 2026](https://arxiv.org/html/2604.02812v1). Behavior trees provide explicit sequencing, fallback, and recovery, but generated trees are only useful if leaf skills and predicates are grounded and tested. For the green-object task, a full learned behavior-tree generator is unnecessary; a hand-specified eight-state machine is a cleaner verifier of the central idea.

### 3.3 Chunk execution is part of the policy

The same predicted chunk can behave very differently depending on how much is executed before replanning.

- **BID** studies closed-loop resampling and the trade-off between long-chunk temporal coherence and stale actions under environmental change [Improving Action Chunking via Closed-Loop Resampling, 2024](https://arxiv.org/html/2408.17355v1).
- **RTC** asynchronously generates a new flow-policy chunk while preserving the prefix already committed to execution [Real-Time Chunking, 2025](https://arxiv.org/html/2506.07339v1).
- **PACE** selects execution horizons from low-speed phase boundaries in the predicted chunk. It reports improvements on both ALOHA and single-arm Franka experiments without retraining [PACE, 2026](https://arxiv.org/abs/2606.00537).
- **DEHP** learns a lightweight horizon predictor while freezing the chunk policy, using shorter horizons around precision phases and longer horizons in free-space motion [DEHP, 2026](https://arxiv.org/abs/2606.11408).

For a minimal proof, do not implement these systems wholesale. Compare two explicit schedules:

1. **fixed receding horizon:** execute the first \(h\) actions, then re-query;
2. **predicate-bounded horizon:** stop the chunk early when a phase predicate changes, a low-speed boundary appears, or a verifier rejects the next command.

Log the schedule. Otherwise “same model” does not mean “same executable.”

### 3.4 Distilling a large VLA into a small deployed policy

Recent work increasingly treats a large VLA or VLM as an offline supervisor rather than a component that must remain in the control loop.

- **VLA-AD** distills phase and directional semantics from a VLM/VLA teacher into a lightweight closed-loop student, removing the semantic supervisor at deployment [VLA-AD, 2026](https://arxiv.org/html/2605.16241v1).
- **ActDistill** transfers action capability from a full VLA to a smaller model using action-guided self-distillation [ActDistill, 2026](https://arxiv.org/html/2511.18082v3).
- **XS-VLA** separately distills coarse task-conditioned spatial grounding and learns coherent latent-flow actions at a 0.25B scale [XS-VLA, 2026](https://arxiv.org/html/2607.04171v3).

These results support the user's earlier hypothesis: at fixed deployed student size, a stronger teacher may improve spatial generalization. They do **not** show that a larger teacher automatically produces a more deterministic student. The teacher must be queried on states the student actually visits, and the student's action/embodiment interface must be able to represent the teacher's behavior.

For this proof, MolmoAct2 teacher signals should be cheap and structured:

- task phase;
- 2D object anchor or mask;
- target region;
- whether to replan;
- optional preferred action token/chunk among a small candidate set.

Avoid distilling free-form chain-of-thought. It is difficult to verify and unnecessary for a one-object task.

### 3.5 Runtime monitoring, shields, and formal methods

Formal verification of an end-to-end high-dimensional VLA is currently not a practical first milestone. Formal and semi-formal methods become tractable when applied to a low-dimensional execution boundary.

- **FAIL-Detect** treats deployment failures as sequential out-of-distribution events and uses scalar policy/observation scores plus conformal thresholds. It is notable because it can calibrate from successful trajectories without requiring a complete failure dataset [Xu et al., 2025](https://arxiv.org/html/2503.08558v3).
- **SafeDec** incorporates Signal Temporal Logic constraints into action decoding rather than only filtering after generation [SafeDec, 2026](https://arxiv.org/html/2509.01728v4).
- Programmatic runtime shields such as **Aegis** synthesize a lightweight correction layer that prevents specified unsafe commands in studied control systems [Aegis, 2025](https://arxiv.org/html/2410.05641v3).
- A 2026 survey organizes formal methods for robot policy learning and verification, while emphasizing the continuing scalability gap for realistic learned systems [Manganaris et al., 2026](https://arxiv.org/abs/2602.06971).

The key practical lesson is to verify **predicates and authorization**, not the entire visual network. For example:

- every commanded joint is within bounds;
- forward-kinematic end-effector position remains in a tabletop workspace;
- Cartesian/joint velocity and acceleration are bounded;
- CLOSE is permitted only near the object;
- LIFT is permitted only after gripper closure/contact evidence;
- RELEASE is permitted only above the bin region;
- timeout, missing object, or repeated rejection leads to ABORT rather than improvisation.

### 3.6 The executable includes more than model weights

A particularly important deployment result shows that changing action unnormalization metadata while keeping the checkpoint fixed can destroy policy success. It formalizes an executable policy as weights **plus** action representation, normalization metadata, and controller conventions [Same Weights, Different Robot, 2026](https://arxiv.org/abs/2606.03724).

This motivates an `ExecSpec` manifest for every run:

- model hash;
- preprocessing hash;
- camera names and ordering;
- prompt/instruction;
- observation and action schema;
- action normalization statistics and key;
- action frequency and chunk length;
- decoder mode, seed/latent, and solver steps;
- replanning/aggregation rule;
- safety limits and state-machine version;
- controller mode and gains;
- software, CUDA, and hardware versions.

Without this manifest, claims about determinism are not portfolio-safe or reproducible.

---

## 4. Minimal Panda proof of concept with no custom simulator

### 4.1 Task

**Instruction:** “Pick up the green object and place it in the green bin.”

Use a rigid object rather than a deformable ball for the first experiment. A cube, short cylinder, or firm foam ball reduces grasp-state ambiguity. Deformability can be introduced as a later distribution shift.

### 4.2 Physical fixture

- Franka Panda with existing low-level joint or Cartesian impedance controller.
- One fixed third-person RGB camera. A wrist camera is optional and should not be added until the single-camera baseline works.
- Printed mat with 6–12 marked start cells.
- Bin at one fixed marked pose.
- Fixed lighting for baseline; one alternative lighting condition for stress testing.
- Emergency stop and conservative velocity/workspace limits.

The printed grid substitutes for scene authoring. It provides enumerable initial conditions and lets a human reset the world in seconds.

### 4.3 One shared action interface

Do not let each repository define a different robot executable. Create one thin Panda adapter and freeze it before model comparison.

Recommended first action schema, chosen to align with MolmoAct2-DROID:

\[
a_t = [q^{\mathrm{target}}_{1:7}, g^{\mathrm{target}}]
\]

at a fixed learned-policy rate, with the existing Panda controller tracking targets at its native high rate. The adapter performs:

1. denormalization from versioned statistics;
2. absolute-versus-delta convention checking;
3. joint/velocity/acceleration saturation;
4. forward-kinematic workspace checking;
5. interpolation to the controller rate;
6. timestamping and command logging.

If the current teleoperation stack already records Cartesian end-effector targets reliably, that is also acceptable. The important point is to use **one** representation across ACT and VQ-BeT and to make the MolmoAct2 mapping explicit.

### 4.4 Cheap phase predicates

Use an explicit state machine:

`APPROACH → DESCEND → CLOSE → LIFT → TRANSFER → LOWER → RELEASE → RETREAT → DONE`

with `ABORT` reachable from every state.

Predicates can be built without a learned scene model:

- green-object centroid from HSV thresholding;
- bin region from a fixed image polygon or fiducial;
- end-effector pose from Panda forward kinematics;
- gripper width/current from robot state;
- fixed table-plane homography or one-time camera calibration;
- timeouts and rejected-action counts.

Examples:

- `APPROACH → DESCEND` only if the end effector is within an XY radius of the object;
- `CLOSE → LIFT` only if gripper closure lies in an expected object-holding interval;
- `TRANSFER → LOWER` only if the object/gripper is over the bin polygon;
- `RELEASE → DONE` only if the green object is detected inside the bin after retreat;
- otherwise retry once or abort.

This state machine is not claimed as a universal task planner. It is a cheap experimental instrument for testing whether a structured execution boundary reduces variance and unsafe improvisation.

### 4.5 Dataset

Collect **40–60 successful demonstrations**:

- at least 4–5 per placement cell;
- small variation in approach direction and execution speed;
- consistent instruction and camera setup;
- robot state, RGB, raw teleop command, authorized command, timestamps, and phase labels.

Phase labels can be generated mostly from gripper state and geometry, then corrected manually. This is faster and more consistent than free-form annotation.

Split by placement cell, not random frames:

- training: interior/easy cells;
- in-distribution test: held-out repeats in those cells;
- spatial shift: edge/corner cells;
- perturbation: object moved manually during approach or after a failed grasp.

### 4.6 Experiment 0 — recorded-input determinism (hours, no robot rollouts)

Run each model on exactly the same stored observation/state snapshots.

For each of 50 snapshots, perform 100 inference repetitions under:

| Model | Stochastic condition | Deterministic condition |
|---|---|---|
| ACT | sampled latent/prior | fixed zero or fixed stored latent |
| VQ-BeT | sample latent-action tokens | argmax tokens, deterministic residual decoder |
| MolmoAct2 | fresh Gaussian flow noise | stored/fixed Gaussian noise and fixed solver |

Measure:

- exact action-array hash equality;
- maximum and RMS action difference;
- selected token/mode entropy;
- chunk-to-chunk disagreement;
- mean, p95, p99, maximum latency;
- deadline-miss count;
- sensitivity to batch size, precision, and restart.

This separates model stochasticity and systems nondeterminism from physical-world variation. It is the cheapest publishable first figure.

### 4.7 Experiment 1 — ACT execution ablation (first physical proof)

Use one trained ACT checkpoint. Do not compare architectures yet.

| Variant | Decoder | Chunk schedule | Guard |
|---|---|---|---|
| A: native | fixed latent | standard temporal ensemble/receding horizon | joint/workspace limits only |
| B: committed | fixed latent | fixed first-\(h\) prefix | joint/workspace limits only |
| C: compiled | fixed latent | predicate-bounded prefix | limits + phase state machine + retry/abort |

Run 6 start cells × 5 repeats = **30 trials per variant** if time permits. A cheaper screening stage can use 3 repeats and advance only the best two variants.

This experiment answers the central question more directly than an immediate three-model bake-off: does the deterministic execution contract reduce physical dispersion and failure severity for the same learned policy?

### 4.8 Experiment 2 — action representation comparison

Train ACT and VQ-BeT on the same demonstrations and action schema.

Compare:

1. ACT fixed latent;
2. VQ-BeT argmax token;
3. each model behind the identical compiled executor.

The critical interaction is:

\[
\text{model family} \times \text{native/compiled execution}.
\]

If VQ-BeT is more repeatable only because it chooses one mode but still fails phase predicates, the discrete representation is insufficient. If the compiled executor helps both models similarly, the main contribution lies at the execution boundary. If VQ-BeT plus the executor produces the narrowest outcome distribution without losing success, discrete modes are genuinely useful for compilation.

### 4.9 Experiment 3 — MolmoAct2 as teacher and direct policy

Proceed in two steps.

**Teacher experiment:**

1. collect ACT/VQ-BeT failures and edge-cell observations;
2. ask MolmoAct2 for phase/spatial features or use its internal spatial/action outputs;
3. add teacher-derived phase/anchor supervision to a fixed-size ACT or VQ-BeT student;
4. keep deployment free of MolmoAct2;
5. re-evaluate the exact held-out fixture cells.

**Direct-policy experiment:**

1. initialize from MolmoAct2-DROID;
2. verify camera ordering, absolute joint-pose semantics, normalization, and control rate;
3. compare random-noise, fixed-noise, and verifier-selected candidate chunks;
4. use the same Panda adapter and compiled executor.

This tests the earlier hypothesis fairly: does a stronger spatial teacher improve a small deployed policy at fixed student size and fixed execution semantics?

### 4.10 Perturbation test

Outcome repeatability on static starts is not enough. Add one scripted human perturbation:

- move the object by 3–5 cm during `APPROACH`, or
- induce one failed grasp by shifting it just before `CLOSE`.

Measure whether each system:

- continues stale open-loop motion;
- replans;
- retries through an allowed transition;
- safely aborts;
- improvises an unapproved behavior.

A deterministic but non-reactive policy will look good on replay and fail here. The compiled reactive executor should produce a deterministic **response rule**, not an identical action trace.

---

## 5. Metrics and plots

### 5.1 Primary metrics

1. **Task success:** object fully inside bin after retreat.
2. **Authorized execution rate:** fraction of proposed actions accepted without projection/rejection.
3. **Constraint violations:** before and after the guard.
4. **Endpoint dispersion:** covariance or median absolute deviation of final object pose.
5. **Trajectory dispersion:** time-normalized Fréchet/DTW distance between end-effector paths from the same fixture start.
6. **Phase timing variation:** coefficient of variation of transition times.
7. **Recovery outcome:** recover, retry, abort safely, or unsafe continuation.
8. **Inference timing:** p50/p95/p99/max and missed deadlines.
9. **Determinism score on replay:** fraction of repeated calls with identical token trace and action hash; use tolerance-based equality for cross-device floating point.
10. **Intervention burden:** projections, rejections, retries, and aborts per successful task.

### 5.2 The most useful plot

Plot each variant on a determinism–capability frontier:

- x-axis: physical trajectory dispersion or replay action variance;
- y-axis: task success under static and perturbed starts;
- color: unsafe violations or abort rate;
- marker size: p99 latency.

This prevents a trivial “deterministic” policy that always aborts or repeats one brittle trajectory from appearing successful.

### 5.3 Statistical reporting

With small physical trial counts, report Wilson intervals for success proportions and bootstrap intervals for dispersion metrics. Do not overclaim differences from 10–15 trials. Use the inexpensive replay experiment for high-\(n\) numerical evidence and reserve robot time for hypothesis-discriminating conditions.

---

## 6. Expected hypotheses

### H1 — Most decoder nondeterminism is removable

Fixed latent/argmax/stored flow noise should make all three models highly repeatable on identical recorded inputs. ACT and VQ-BeT are likely easier to stabilize than a large flow-matching VLA across software/hardware versions.

### H2 — Deterministic decoding alone will not stabilize physical outcomes

The remaining variance will come from sensor history, contact, chunk scheduling, action metadata, and control timing. This is why the executor and predicates are first-class experimental variables.

### H3 — A phase-gated executor will reduce severe failures

It should reduce commands that close away from the object, lift without a grasp, release outside the bin, or continue after stale observations. It may increase safe aborts. That trade is acceptable and should be shown explicitly.

### H4 — VQ-BeT will make mode selection more auditable

Argmax code selection should produce a stable latent trace and may reduce mode switching. Quantization may hurt precision around grasp/contact unless continuous residuals and replanning remain available.

### H5 — MolmoAct2 helps most under spatial shift

Its value should appear on held-out object positions, camera/lighting changes, or semantically varied objects—not necessarily on the easiest in-distribution placement. As a direct controller, fixed flow noise may reduce diversity without improving success.

### H6 — Predicate-bounded chunking beats a single fixed horizon

Longer commitment should help smooth free-space transfer; shorter commitment should help approach, grasp, and placement. This is consistent with the recent PACE and DEHP results.

---

## 7. What not to build

For the first proof, avoid:

- a custom MuJoCo/Isaac scene;
- photorealistic assets or domain randomization;
- a learned world model;
- full behavior-tree synthesis;
- formal verification of the vision backbone;
- RL fine-tuning;
- a torque-output learned policy;
- a universal factor graph for all skills;
- a multi-task language benchmark;
- a second camera before the single-camera pipeline is stable.

Each can become useful later, but each introduces a larger debugging surface than the hypothesis requires.

Existing environments can be used only for a **no-modification smoke test**. If the official environment and checkpoint do not run quickly, skip them rather than turning environment repair into the project.

---

## 8. Recommended implementation order

### Stage 1 — one week-equivalent proof

1. Freeze the Panda action adapter and `ExecSpec`.
2. Collect 40–60 fixture-based demonstrations.
3. Train ACT.
4. Run recorded-input determinism tests.
5. Run ACT native versus compiled execution on 3–6 start cells.

**Exit criterion:** the compiled variant preserves at least roughly the baseline success rate while reducing severe phase/constraint failures and trajectory dispersion, or produces a clear negative result explaining why.

### Stage 2 — representation test

1. Train VQ-BeT on the same data.
2. Compare sampled versus argmax codes offline.
3. Compare ACT and VQ-BeT under the identical executor.

**Exit criterion:** establish whether discrete latent actions add value beyond the executor itself.

### Stage 3 — foundation-model transfer

1. Run MolmoAct2-DROID on stored observations.
2. Audit direct action compatibility before robot execution.
3. Distill phase/spatial signals into the fixed-size student.
4. Test spatial-shift and perturbation cells.

**Exit criterion:** determine whether a large spatial VLA adds capability that survives compilation into the smaller deterministic executable.

---

## 9. Novelty and evidence boundary

None of the individual ingredients is novel by itself:

- fixed seeds and argmax decoding are standard;
- ACT and VQ-BeT are established policy families;
- action chunking and adaptive horizons are active research areas;
- state machines, safety filters, and runtime monitors are established robotics tools;
- VLA-to-small-policy distillation is emerging rapidly.

The potentially novel contribution is a **controlled compilation study** that keeps the task, data, embodiment, controller, action schema, and verifier fixed while separating:

1. generative model stochasticity;
2. discrete versus continuous action representation;
3. chunk scheduling;
4. executable metadata;
5. runtime authorization;
6. physical outcome dispersion;
7. teacher capability transfer.

The strongest claim should remain empirical:

> Under a fixed Panda task and executable specification, structured decoding and predicate-gated execution improve command repeatability, trace auditability, and bounded failure behavior, with measured effects on task success and perturbation recovery.

Do not claim formal end-to-end safety, guaranteed task success, or universal VLA compilation from this experiment.

---

## 10. Core reading list

1. Zhao et al., **Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (ACT)**, 2023 — action chunks, CVAE, temporal ensembling. <https://arxiv.org/abs/2304.13705>
2. Lee et al., **Behavior Generation with Latent Actions (VQ-BeT)**, ICML 2024 — hierarchical VQ action modes. <https://proceedings.mlr.press/v235/lee24y.html>
3. Fang et al., **MolmoAct2: Action Reasoning Models for Real-World Deployment**, 2026 — spatial VLM, discrete action pretraining, flow-matching expert, Franka release. <https://arxiv.org/html/2605.02881v1>
4. Pertsch et al., **FAST: Efficient Action Tokenization for Vision-Language-Action Models**, 2025 — compression-based action tokens. <https://arxiv.org/abs/2501.09747>
5. Zheng et al., **PRISE**, ICML 2024 — learned variable-length temporal abstractions. <https://arxiv.org/html/2402.10450v1>
6. Sontakke et al., **QueST**, 2024 — self-supervised skill abstractions. <https://arxiv.org/html/2407.15840v2>
7. Chi et al., **Improving Action Chunking via Closed-Loop Resampling**, 2024 — chunk length versus reactivity. <https://arxiv.org/html/2408.17355v1>
8. Black et al., **Real-Time Execution of Action Chunking Flow Policies**, 2025 — asynchronous committed-prefix execution. <https://arxiv.org/html/2506.07339v1>
9. Nie et al., **PACE**, 2026 — phase-aware execution horizons with Franka results. <https://arxiv.org/abs/2606.00537>
10. Zhao et al., **Dynamic Execution Horizon Prediction**, 2026 — learned horizon selection around precision phases. <https://arxiv.org/abs/2606.11408>
11. **NoTVLA**, 2026 — sparse semantic waypoints and deterministic motion rendering. <https://arxiv.org/html/2510.03895v2>
12. Xu et al., **FAIL-Detect**, 2025 — conformal, time-varying runtime failure detection from successful data. <https://arxiv.org/html/2503.08558v3>
13. **SafeDec**, 2026 — temporal-logic-constrained action decoding. <https://arxiv.org/html/2509.01728v4>
14. Manganaris et al., **Formal Methods in Robot Policy Learning and Verification**, 2026 — survey and limitations. <https://arxiv.org/abs/2602.06971>
15. **Same Weights, Different Robot**, 2026 — action metadata as part of the executable policy. <https://arxiv.org/abs/2606.03724>
16. **VLA-AD**, 2026 — offline semantic supervision for lightweight deployed students. <https://arxiv.org/html/2605.16241v1>
17. **From Imitation to Refinement**, 2024 — small closed-loop residual correction over frozen chunked behavior cloning. <https://arxiv.org/html/2407.16677v2>

## Bottom line

The fastest high-signal experiment is not “build three policies in a simulator.” It is:

1. build one stable Panda action adapter;
2. train ACT on a fixture-defined task;
3. prove decoder repeatability on stored inputs;
4. compare native ACT against a deterministic, phase-gated executor;
5. add VQ-BeT to isolate whether discrete action modes help;
6. add MolmoAct2 as a Franka-native teacher/direct policy only after the execution boundary is already fixed.

This yields a credible proof of concept even if the answer is negative: it will identify whether nondeterminism lives in the generative decoder, discrete/continuous representation, chunk scheduler, executable metadata, contact dynamics, or missing runtime predicates.
