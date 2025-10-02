# Shipmind Git Workflow

This project has been developed locally without version control so far. We are adopting Git now with a lightweight, low‑friction workflow that supports our session‑based handoffs and preserves a clean, linear history until we add a remote.

## Goals
- Keep history readable and linear during rapid iteration
- Capture session milestones and performance baselines
- Enable feature isolation when needed (sensors, gossip, etc.)
- Avoid committing secrets, large binaries, or transient artifacts

## Repository Setup (local)

1. Initialize and make the first commit
```
# from project root
git init
git add .gitignore README.md project/ .
git commit -m "chore: initialize git and add .gitignore"
```

2. Add a session tag (optional but recommended)
```
git tag -a session-YYYY-MM-DD-aquarium-bootstrap -m "Session milestone: aquarium foundation"
```

3. Keep working on `main` (or `master`) with small, logical commits.

## Branching Strategy (local-first)
- Default: single branch `main` with linear history (no merges)
- Use short‑lived feature branches only when you need to isolate larger work:
  - `feat/sensors-cone`
  - `feat/sensors-thermal`
  - `feat/gossip-exchange`
  - `perf/avoidance-hash`
- Rebase/squash before merging back to keep history clean.

When a remote is introduced (GitHub/GitLab):
- Continue feature branches via PRs
- Protect `main` with required checks once CI is added

## Commit Conventions
Use Conventional Commits with concise scope and meaningful body lines when helpful.
```
feat(sensors): add cone query DTO and flags
perf(avoidance): obstacle-centric vectorization with spatial hash
fix(thermal): guard divide-by-zero when influence <= 0
docs(sensor-api): document flags, fields, and omission rules
chore(git): add .gitignore and workflow docs
```
General guidance:
- Commit one logical change per commit
- Keep titles ≤ 72 chars; use bullet points in body when useful
- Reference tags or follow‑ups in the body (e.g., “prep for C4b checkpoint”)

## Milestone Tags
Use annotated tags to capture major slice checkpoints and performance baselines.

Examples:
```
# C1: vectorized batch via subtrees
git tag -a c1-spatial-batch -m "C1: spatial batch complete, 1000=22.3ms"

# C2: avoidance optimized within budget
git tag -a c2-avoidance -m "C2: avoidance 2.8ms, 1000=19.5ms"

# C4a: sensor channels Phase A
git tag -a c4a-sensors -m "C4a: acoustic+biolum+optical alias, 1000=1.16ms"

# C4b: thermal channel
git tag -a c4b-thermal -m "C4b: thermal vectorized, 1000=1.53ms"
```

## Session Tags
Optionally tag handoff boundaries to align with mission briefs:
```
git tag session-2025-10-01-avoidance-optimized
```

## Pre‑commit Checklist
```
# Review diff and file list
git status && git diff --stat

# No secrets or keys (breadth search)
git grep -nE "(KEY|SECRET|TOKEN|PASSWORD)" -- :!**/*.md

# Python temp and caches should be ignored
git check-ignore -v aquarium/__pycache__
```

## Files to Avoid Committing
- Private keys/secrets, environment files
- Large binaries, generated assets (consider Git LFS if needed later)
- Transient reports (keep in status/ but ignore heavy or per‑run outputs)

## Suggested First Commit Set (retroactive)
Make atomic commits by domain so history is meaningful:
- `feat(sim): minimal tick + behavior engine`
- `perf(spatial): cKDTree adapter + batch`
- `perf(avoidance): spatial hash + obstacle-centric`
- `feat(sensors): cone DTO + flags`
- `feat(sensors): acoustic+biolum+optical alias`
- `feat(sensors): thermal nearest-vent`
- `docs(progress): update PROGRESS.md for C4b`

> Tip: You can `git add -p` to stage hunks per commit and `git rebase -i` to squash/rename before tagging.

## Remote Adoption (when ready)
1. Create a private repo
2. `git remote add origin <url>`
3. `git push -u origin main --tags`
4. Add a minimal CI (lint/tests) later; protect `main`

---
Last updated: 2025‑10‑01
