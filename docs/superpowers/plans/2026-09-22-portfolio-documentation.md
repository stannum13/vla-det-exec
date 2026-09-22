# Deterministic VLA Portfolio Documentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the deterministic VLA repository visually explain its implemented benchmark, metrics, and next phase-gated execution experiment without implying real-checkpoint results already exist.

**Architecture:** Add two self-contained accessible SVG diagrams and rewrite the README around executable semantics, Experiment 0, implemented infrastructure, and the next perturbation study. Diagrams explain design only; future measured output remains a separate section.

**Tech Stack:** GitHub Markdown, SVG 1.1, Python/pytest

## Global Constraints

- Modify only `README.md` and create two files under `docs/media/`.
- Do not modify or stage the untracked `exp0/snapshots/*.npz` files.
- Use graphite, blue, and amber with WCAG-readable text.
- Put `role="img"`, `aria-labelledby`, `<title>`, and `<desc>` in each SVG.
- Label both diagrams as explanatory, not measured.
- State clearly that the full checkpoint benchmark has not run.

---

### Task 1: Create the explanatory media

**Files:**
- Create: `docs/media/deterministic-execution-pipeline.svg`
- Create: `docs/media/experiment-zero-matrix.svg`

**Interfaces:**
- Produces: two 1200-pixel-wide SVGs linked by the README

- [ ] **Step 1: Verify the media files do not already exist**

```bash
test ! -e docs/media/deterministic-execution-pipeline.svg
test ! -e docs/media/experiment-zero-matrix.svg
```

Expected: both commands pass.

- [ ] **Step 2: Create the pipeline SVG**

Use a 1200×360 viewBox with five rounded boxes connected left to right:

```text
Fixed observation
instruction + metadata
        →
Versioned preprocessing
        →
ACT / VQ-BeT / MolmoAct2
stochastic and fixed modes
        →
Repeated inference ×100
        →
agreement · drift · entropy
p50/p95/p99 · deadline misses
```

Add a bottom amber band reading: “Design diagram — no checkpoint result is encoded in this figure.”

- [ ] **Step 3: Create the experiment matrix SVG**

Use a 1200×430 viewBox with rows for ACT, VQ-BeT, and MolmoAct2 and columns for stochastic path, deterministic path, and diagnostic:

```text
ACT       sampled latent       zero latent             action drift
VQ-BeT    sampled token        argmax token            token entropy
MolmoAct2 fresh flow noise     fixed-seed flow noise   trajectory drift
```

Add a footer: “Same stored inputs, 100 calls per condition, 100 ms policy deadline.”

- [ ] **Step 4: Validate and render**

```bash
xmllint --noout docs/media/deterministic-execution-pipeline.svg
xmllint --noout docs/media/experiment-zero-matrix.svg
```

Expected: both exit zero. Render each SVG to PNG with an available SVG renderer and inspect at README width.

- [ ] **Step 5: Commit the media**

```bash
git add docs/media/deterministic-execution-pipeline.svg docs/media/experiment-zero-matrix.svg
git commit -m "docs: add deterministic execution diagrams"
```

Expected: one commit containing only the two SVGs.

### Task 2: Rewrite the README as an experiment case study

**Files:**
- Modify: `README.md`
- Reference: `det-vla-exec-rev.md`
- Reference: `exp0/run_exp0.py`

**Interfaces:**
- Consumes: diagrams from Task 1 and implemented CLI behavior
- Produces: recruiter-readable project documentation

- [ ] **Step 1: Record current CLI behavior**

```bash
python exp0/run_exp0.py --help
python -m pytest -q tests --ignore=tests/test_extract_snapshots.py
```

Expected: help exits zero and the model-independent suite passes.

- [ ] **Step 2: Rewrite with this exact section order**

```markdown
# Deterministic VLA Execution

[thesis, status, and one-sentence current frontier]

![Deterministic VLA execution pipeline](docs/media/deterministic-execution-pipeline.svg)

## Why repeatable inference is not enough
## Experiment 0
![Experiment 0 condition matrix](docs/media/experiment-zero-matrix.svg)
## What the harness measures
## What is implemented
## What remains to run
## Current frontier: deterministic response rules
## Repository map
## Setup and reproduction
## Scope and limitations
```

- [ ] **Step 3: Explain metrics in plain language**

Include exact agreement, maximum/RMS drift, token entropy, latency percentiles, and 100 ms deadline misses. Explain that fixed decoding can stabilize software output but cannot force identical physical trajectories under changed observations and contact.

- [ ] **Step 4: Add the active next-step sequence**

Specify the real-checkpoint run on 50 observations × 100 calls, followed by fixed versus predicate-bounded chunk execution, scripted object displacement, induced failed grasp, and measurements of stale continuation/replan/retry/safe abort/unsafe behavior. Gate teacher-to-student transfer on a stable executor.

- [ ] **Step 5: Commit the README**

```bash
git add README.md
git commit -m "docs: present VLA determinism as active execution research"
```

Expected: one commit containing only `README.md`.

### Task 3: Verify documentation and tests

- [ ] **Step 1: Validate paths and XML**

Run relative-link validation plus:

```bash
xmllint --noout docs/media/*.svg
git diff HEAD~2 --check
```

Expected: no missing links, invalid XML, or whitespace errors.

- [ ] **Step 2: Run the model-independent suite**

```bash
python -m pytest -q tests --ignore=tests/test_extract_snapshots.py
```

Expected: all available model-independent tests pass.

- [ ] **Step 3: Verify scope**

```bash
git status --short
git show --stat --oneline HEAD~1
git show --stat --oneline HEAD
```

Expected: only the two planned commits are new; the two pre-existing snapshot files remain untracked and unstaged.

