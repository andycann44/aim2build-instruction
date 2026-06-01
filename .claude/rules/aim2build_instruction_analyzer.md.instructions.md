---
description: Aim2Build Instruction Analyzer global rules
paths:
  - "**/*"
  - "**/*.py"
  - "**/*.md"
  - "clean/**"
  - "debug/**"
---

# Aim2Build Instruction Analyzer

Before making any code changes, read:

- docs/PROJECT_BRAIN/ARCHITECTURE.md
- docs/PROJECT_BRAIN/SOURCE_OF_TRUTH.md
- docs/PROJECT_BRAIN/ROUTES.md
- docs/PROJECT_BRAIN/DATA_FLOW.md
- docs/PROJECT_BRAIN/DECISIONS.md
- docs/PROJECT_BRAIN/DO_NOT_TOUCH.md

## Core Goal

This project is building a LEGO instruction analysis system.

The current priority is:

1. Human-reviewed Bag Review workflow.
2. Correct crop extraction.
3. Correct slot assignment.
4. Training data quality.
5. Building a reliable dataset.
6. AI automation later.

Do not redesign the workflow around full AI automation.

## Source Of Truth

Authoritative review state:

debug/training_labels/{set_num}_bag{bag}.json

This is the ONLY source of truth for Bag Review.

## Derived Systems

These are derived only:

- clip_memory.json
- catalog embeddings
- crop cache
- step_segmented_cutouts
- part_cutouts
- candidate rankings
- memory boosts

Do not treat derived systems as authoritative.

## Before Any Change

Report:

- Files to modify
- What reads data
- What writes data
- Source-of-truth impact
- Risks

before implementing.

## Never Do These Without Approval

- Create a second source of truth
- Create duplicate review databases
- Create duplicate review JSON files
- Rewrite ranking systems
- Replace human review workflow
- Remove Bag Review workflow
- Add major architecture changes

## Current Review Workflow

Bag 1 has been manually reviewed and verified.

Workflow:

Crop Detection
→ Crop Cache
→ Bag Review UI
→ Save Label / Unknown / Ignored
→ training_labels JSON
→ Derived Clip Memory
→ Future Training

Always preserve this workflow.

## Human Review Rules

Human decisions are authoritative.

If a human-reviewed label conflicts with AI output:

Human review wins.

## Preferred Behaviour

Small isolated changes.

Do not refactor unrelated systems.

Do not rewrite working code.

Fix the smallest thing that solves the problem.

## Commit Policy

Never commit automatically.

Always show:

- files changed
- diff summary
- risks

before commit.