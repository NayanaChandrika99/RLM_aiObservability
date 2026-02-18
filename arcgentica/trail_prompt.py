# ABOUTME: Defines TRAIL-specific prompts for recursive planning and chunk-level error investigation.
# ABOUTME: Keeps agent outputs constrained to JSON so downstream semantic checks can validate evidence grounding.

TRAIL_ROOT_PLAN_PROMPT = """
You are analyzing a long agent execution trace for TRAIL error taxonomy labeling.

Inputs are embedded below in JSON sections:
- Trace summary.
- Chunk catalog.
- Allowed taxonomy.
- Max chunk budget.

Task:
1. Select the most promising chunk ids for error discovery.
2. Prefer diverse chunks with likely failures (errors, timeouts, auth, retries, malformed outputs).
3. Respect budget limits and avoid selecting redundant chunks.

Return strict JSON:
{
  "chunk_ids": [0, 2, 4],
  "reason": "short explanation"
}

Rules:
- `chunk_ids` must contain integers present in `chunk_catalog`.
- Do not return markdown.
- Do not include fields other than `chunk_ids` and `reason`.
"""


TRAIL_CHUNK_ANALYSIS_PROMPT = """
You are investigating a subset of a long trace for TRAIL benchmark labeling.

Inputs are embedded below:
- Trace id.
- Chunk id.
- Chunk span ids.
- Chunk payload snippets.
- Allowed taxonomy.

Task:
1. Identify concrete errors in this chunk only.
2. Use only categories in `taxonomy`.
3. Use a real span id from `chunk_span_ids` for `location`.
4. Quote evidence from `chunk_payload` directly.
5. Keep evidence concise and specific.

Return strict JSON:
{
  "errors": [
    {
      "category": "Timeout Issues",
      "location": "span_123",
      "evidence": "timed out while waiting for tool response",
      "description": "Tool call exceeded response deadline.",
      "impact": "HIGH"
    }
  ]
}

Rules:
- If no errors are found, return `{"errors":[]}`.
- `impact` must be one of `LOW`, `MEDIUM`, `HIGH`.
- Do not return markdown.
- Do not include score fields.
"""
