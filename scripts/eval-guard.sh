#!/usr/bin/env bash
# Eval guard — runs retrieval evaluation and exits 0/1 based on pass/fail.
# Usage: ./scripts/eval-guard.sh [--repo /path/to/docs] [--test-cases path.json]
#
# Exit codes:
#   0 — evaluation passed (P@k >= 0.50, MRR >= 0.50)
#   1 — evaluation failed or error occurred

set -euo pipefail

REPO="${1:---repo}"
REPO_PATH="${2:-}"
TEST_CASES="${3:---test-cases}"
TEST_CASES_PATH="${4:-doc_qa/eval/test_cases.json}"

# Allow named arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo)
            REPO_PATH="$2"
            shift 2
            ;;
        --test-cases)
            TEST_CASES_PATH="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

if [ -z "$REPO_PATH" ]; then
    echo "Error: --repo is required"
    echo "Usage: $0 --repo /path/to/docs [--test-cases path.json]"
    exit 1
fi

echo "Running retrieval evaluation..."
echo "  Repo: $REPO_PATH"
echo "  Test cases: $TEST_CASES_PATH"
echo ""

python -m doc_qa eval --repo "$REPO_PATH" --test-cases "$TEST_CASES_PATH"
exit $?
