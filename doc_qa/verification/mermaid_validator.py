"""Two-tier Mermaid diagram validation: regex pre-check + Node.js subprocess.

Tier 1 (regex) catches common syntax errors cheaply without requiring Node.js.
Tier 2 (Node.js) uses the official ``mermaid`` parser for full validation when
a Node.js script is available.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Diagram type recognition
# ---------------------------------------------------------------------------

# Matches the diagram type declaration at the start of a Mermaid diagram.
# Handles optional %%{directives}%% before the type keyword.
_DIAGRAM_TYPE_RE = re.compile(
    r"^\s*"
    r"(?:%%\{.*?%%\s*)?"  # optional %%{...}%% directive block
    r"("
    r"graph(?:\s+(?:TD|TB|BT|RL|LR))?"
    r"|flowchart(?:\s+(?:TD|TB|BT|RL|LR))?"
    r"|sequenceDiagram"
    r"|classDiagram"
    r"|stateDiagram(?:-v2)?"
    r"|erDiagram"
    r"|journey"
    r"|gantt"
    r"|pie"
    r"|gitGraph"
    r"|mindmap"
    r"|timeline"
    r"|sankey-beta"
    r"|xychart-beta"
    r"|block-beta"
    r"|quadrantChart"
    r"|requirementDiagram"
    r"|C4Context"
    r"|C4Container"
    r"|C4Component"
    r"|C4Dynamic"
    r"|C4Deployment"
    r")",
    re.IGNORECASE | re.MULTILINE,
)

# Bracket pairs for balance checking.
_OPEN_BRACKETS = {"(": ")", "[": "]", "{": "}"}
_CLOSE_BRACKETS = set(_OPEN_BRACKETS.values())


class MermaidValidator:
    """Validate Mermaid diagram syntax using a two-tier approach.

    Args:
        node_script_path: Path to a Node.js validation script that reads
            Mermaid text from stdin and writes a JSON result to stdout.
        timeout: Maximum seconds to wait for the Node.js subprocess.
        mode: Validation mode -- ``"node"``, ``"regex"``, ``"auto"``,
            or ``"none"``.
    """

    def __init__(
        self,
        node_script_path: str = "scripts/validate_mermaid.mjs",
        timeout: float = 5.0,
        mode: str = "auto",
    ) -> None:
        self._node_script_path = node_script_path
        self._timeout = timeout
        self._mode = mode

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, diagram_text: str) -> dict:
        """Validate a Mermaid diagram.

        Returns:
            A dict with keys ``valid`` (bool), ``diagram_type`` (str | None),
            and ``error`` (str | None).
        """
        if self._mode == "none":
            return {"valid": True, "diagram_type": None, "error": None}

        # -- Tier 1: regex pre-validation --
        pre_error = self._pre_validate(diagram_text)
        if pre_error is not None:
            dtype = self._detect_diagram_type(diagram_text)
            return {"valid": False, "diagram_type": dtype, "error": pre_error}

        dtype = self._detect_diagram_type(diagram_text)

        # -- Tier 2: Node.js validation --
        if self._mode in ("node", "auto"):
            node_result = self._node_validate(diagram_text)
            if node_result is not None:
                # Merge detected type if node didn't report one
                if node_result.get("diagram_type") is None:
                    node_result["diagram_type"] = dtype
                return node_result

            # node_validate returned None => Node.js unavailable
            if self._mode == "node":
                logger.warning(
                    "Node.js validation requested but unavailable; "
                    "falling back to regex-only result."
                )

        # Regex-only result (tier 1 passed)
        return {"valid": True, "diagram_type": dtype, "error": None}

    # ------------------------------------------------------------------
    # Tier 1: regex pre-validation
    # ------------------------------------------------------------------

    def _pre_validate(self, text: str) -> str | None:
        """Quick regex checks. Returns an error string or None if OK."""
        stripped = text.strip()

        # Empty diagram
        if not stripped:
            return "Empty diagram"

        # Must start with a recognised diagram type
        if not _DIAGRAM_TYPE_RE.match(stripped):
            return (
                "Diagram does not start with a recognised Mermaid diagram "
                "type (e.g. graph TD, sequenceDiagram, classDiagram, ...)"
            )

        # Bracket balance
        balance_error = self._check_bracket_balance(stripped)
        if balance_error:
            return balance_error

        return None

    @staticmethod
    def _check_bracket_balance(text: str) -> str | None:
        """Check that brackets are balanced. Returns error or None."""
        stack: list[str] = []
        in_string = False
        string_char: str | None = None

        for ch in text:
            # Simple string tracking (single/double quotes)
            if ch in ('"', "'"):
                if in_string and ch == string_char:
                    in_string = False
                    string_char = None
                elif not in_string:
                    in_string = True
                    string_char = ch
                continue

            if in_string:
                continue

            if ch in _OPEN_BRACKETS:
                stack.append(_OPEN_BRACKETS[ch])
            elif ch in _CLOSE_BRACKETS:
                if not stack:
                    return f"Unexpected closing bracket '{ch}'"
                expected = stack.pop()
                if ch != expected:
                    return f"Mismatched bracket: expected '{expected}', found '{ch}'"

        if stack:
            return f"Unclosed bracket(s): expected {', '.join(repr(b) for b in reversed(stack))}"

        return None

    @staticmethod
    def _detect_diagram_type(text: str) -> str | None:
        """Extract the diagram type keyword from the text."""
        m = _DIAGRAM_TYPE_RE.match(text.strip())
        if m:
            return m.group(1).strip()
        return None

    # ------------------------------------------------------------------
    # Tier 2: Node.js validation
    # ------------------------------------------------------------------

    def _node_validate(self, text: str) -> dict | None:
        """Run the Node.js Mermaid validator. Returns result dict or None if unavailable."""
        try:
            result = subprocess.run(
                ["node", self._node_script_path],
                input=text,
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )
        except FileNotFoundError:
            logger.debug("Node.js not found on PATH; skipping tier-2 validation.")
            return None
        except subprocess.TimeoutExpired:
            logger.warning(
                "Node.js Mermaid validation timed out after %.1fs.", self._timeout
            )
            return {"valid": False, "diagram_type": None, "error": "Validation timed out"}

        stdout = result.stdout.strip()
        if not stdout:
            if result.returncode != 0:
                stderr_msg = result.stderr.strip()[:200] if result.stderr else "unknown error"
                logger.debug("Node.js validation script failed: %s", stderr_msg)
                return None
            return None

        try:
            parsed = json.loads(stdout)
        except json.JSONDecodeError:
            logger.debug(
                "Node.js validation returned non-JSON: %.100s", stdout
            )
            return None

        # Normalise the result to our expected shape
        return {
            "valid": bool(parsed.get("valid", False)),
            "diagram_type": parsed.get("diagram_type") or parsed.get("diagramType"),
            "error": parsed.get("error"),
        }
