# AI Override Recommendation Plan

## Purpose

AI override is a recommendation layer only. It exists to help a human reviewer inspect uncertain matches before any final review label is saved.

The AI must never write `training_labels`, must never auto-accept labels, and must never promote a recommendation into source-of-truth review state.

## Scope

Only rows in `instruction-v2/indexes/11_manual_match_config.json` with:

```json
"status": "needs_ai_check"
```

are eligible for AI override recommendation.

Rows with `pending`, `accepted`, or `rejected` status are not processed by the AI override stage.

## Future Output

The later AI recommendation stage will write:

```text
instruction-v2/indexes/12_ai_override_recommendations.json
```

This file is derived only. It is not a review label source of truth.

## Inputs

For each eligible row, the AI recommendation stage may read:

- cutout image
- overlay image
- top catalog candidate image
- optional manual override candidate
- set context from `instruction-v2/indexes/00_set_context.json`

## Output Fields

Each recommendation entry must include:

```json
{
  "crop_id": "string",
  "segment_index": "integer",
  "recommended_part_num": "string or null",
  "recommended_color_id": "integer or null",
  "confidence": "number",
  "reasoning": "string",
  "review_required": true
}
```

`review_required` must always be `true`.

## Failure Handling

If required images are missing, the recommendation entry should not invent a match. It should record low confidence, explain which image is missing, and keep `review_required` as `true`.

If confidence is low, the recommendation should keep `recommended_part_num` and `recommended_color_id` as `null` unless there is a clear best candidate. The reasoning should explain the uncertainty.

If the AI disagrees with a manual override candidate, it must not replace the override. It should record the disagreement in `reasoning`, include its recommendation if confidence is sufficient, and keep `review_required` as `true`.

## Authority

Human review remains the final authority.

AI recommendations can assist review, but they cannot save labels, accept labels, reject labels, or update training state.
