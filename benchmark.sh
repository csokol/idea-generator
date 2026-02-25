#!/usr/bin/env bash
set -euo pipefail

DRY_RUN="${DRY_RUN:-}"

run() {
  if [ -n "$DRY_RUN" ]; then
    echo "[dry-run] uv run $*"
  else
    uv run "$@"
  fi
}

RUN_ID="benchmark_$(date -u +%Y%m%d_%H%M%S)"
GOAL='Find real complaints, frustrations, and unmet needs from Snowflake Data Cloud users on Reddit. Focus on: cost and billing surprises, credit consumption issues, performance bottlenecks, query optimization struggles, migration difficulties, missing features, tooling gaps, governance challenges, workarounds people build, and alternatives they consider.'
DOMAINS="reddit.com"

COMMON=(--goal "$GOAL" --domains "$DOMAINS" --run-id "$RUN_ID" --fetch-pages -v)
RESEARCH=(--research-provider groq --research-model "qwen/qwen3-32b")

# Array of: label|provider|model
MODELS=(
  "openai-gpt52|openai|gpt-5.2"
  "gemini-3pro|gemini|gemini-3.1-pro-preview"
  "gemini-flashlite|gemini|gemini-3-flash-preview"
  "groq-qwen3|groq|qwen/qwen3-32b"
  "groq-openai-oss-120b|groq|openai/gpt-oss-120b"
)

# First run: full pipeline (research + index + insights)
IFS='|' read -r label provider model <<< "${MODELS[0]}"
echo "=== Run 1/4: $label (full pipeline) ==="
run pipeline "${COMMON[@]}" "${RESEARCH[@]}" \
  --insight-provider "$provider" --insight-model "$model" \
  --out-report "data/$RUN_ID/report-${label}.md"

# Remaining runs: insights only (reuse corpus + index)
for i in 1 2 3; do
  IFS='|' read -r label provider model <<< "${MODELS[$i]}"
  echo "=== Run $((i+1))/4: $label (insights only) ==="
  run pipeline "${COMMON[@]}" \
    --start-from insights \
    --insight-provider "$provider" --insight-model "$model" \
    --out-report "data/$RUN_ID/report-${label}.md"
done

echo ""
echo "All reports in data/$RUN_ID/"
ls -la "data/$RUN_ID"/report-*.md
