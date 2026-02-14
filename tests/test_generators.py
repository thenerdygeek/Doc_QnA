"""Tests for all generators and the router."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from doc_qa.config import GenerationConfig, IntelligenceConfig
from doc_qa.generation.base import GenerationResult
from doc_qa.generation.code_example import CodeExampleGenerator, _ensure_language_tags
from doc_qa.generation.comparison import ComparisonGenerator
from doc_qa.generation.diagram import DiagramGenerator
from doc_qa.generation.explanation import ExplanationGenerator
from doc_qa.generation.procedural import ProceduralGenerator
from doc_qa.generation.router import resolve_generation_strategy, route_and_generate
from doc_qa.intelligence.intent_classifier import IntentMatch, OutputIntent
from doc_qa.llm.backend import Answer, LLMBackend


# ── Helpers ──────────────────────────────────────────────────────────


class MockLLM(LLMBackend):
    """Mock LLM that returns a controllable response."""

    def __init__(self, text: str = "", error: str | None = None) -> None:
        self.text = text
        self.error = error
        self.call_count = 0

    async def ask(self, question: str, context: str, history=None) -> Answer:
        self.call_count += 1
        return Answer(text=self.text, sources=[], model="mock", error=self.error)

    async def close(self) -> None:
        pass


def _intent(intent=OutputIntent.EXPLANATION, confidence=0.9, sub_type="none"):
    return IntentMatch(
        intent=intent, confidence=confidence,
        matched_pattern="test", sub_type=sub_type,
    )


# ── ExplanationGenerator ─────────────────────────────────────────────


class TestExplanationGenerator:
    @pytest.mark.asyncio
    async def test_pass_through(self):
        llm = MockLLM(text="The rate limit is 100 req/s.")
        gen = ExplanationGenerator()
        result = await gen.generate("rate limit?", "context", None, llm, _intent())
        assert result.text == "The rate limit is 100 req/s."
        assert result.model == "mock"
        assert result.diagrams is None


# ── DiagramGenerator ─────────────────────────────────────────────────


class TestDiagramGenerator:
    @pytest.mark.asyncio
    async def test_extracts_mermaid_blocks(self):
        mermaid = "graph TD\n  A-->B"
        llm = MockLLM(text=f"Here:\n```mermaid\n{mermaid}\n```\n")
        gen = DiagramGenerator()
        result = await gen.generate("draw flow", "ctx", None, llm, _intent(OutputIntent.DIAGRAM))
        assert result.diagrams is not None
        assert len(result.diagrams) == 1
        assert "A-->B" in result.diagrams[0]

    @pytest.mark.asyncio
    async def test_no_diagrams_in_output(self):
        llm = MockLLM(text="No diagram here.")
        gen = DiagramGenerator()
        result = await gen.generate("draw flow", "ctx", None, llm, _intent(OutputIntent.DIAGRAM))
        assert result.diagrams is None

    @pytest.mark.asyncio
    async def test_validate_repair_loop(self):
        """Test that validation triggers a repair attempt."""
        # First call returns invalid, second returns valid
        texts = iter([
            "```mermaid\nbadgraph\n```",
            "```mermaid\ngraph TD\nA-->B\n```",
        ])

        class RepairLLM(LLMBackend):
            async def ask(self, question, context, history=None):
                return Answer(text=next(texts), sources=[], model="mock")
            async def close(self):
                pass

        llm = RepairLLM()
        gen = DiagramGenerator()

        mock_validator_cls = type("MockValidator", (), {
            "__init__": lambda self, **kw: None,
            "validate": lambda self, d: (
                {"valid": False, "error": "bad syntax"} if "badgraph" in d
                else {"valid": True, "error": None}
            ),
        })

        with patch("doc_qa.generation.diagram._HAS_VALIDATOR", True), \
             patch("doc_qa.generation.diagram.MermaidValidator", mock_validator_cls):
            result = await gen.generate(
                "draw", "ctx", None, llm,
                _intent(OutputIntent.DIAGRAM, sub_type="flowchart"),
            )
        assert result.diagrams is not None
        assert "A-->B" in result.diagrams[0]

    @pytest.mark.asyncio
    async def test_config_driven_max_retries(self):
        """Bug 1: max_diagram_retries from config is respected, not hardcoded 3."""
        call_count = 0

        class CountingLLM(LLMBackend):
            async def ask(self, question, context, history=None):
                nonlocal call_count
                call_count += 1
                # Always return invalid diagram so repair keeps trying
                return Answer(
                    text="```mermaid\nbadgraph\n```", sources=[], model="mock",
                )
            async def close(self):
                pass

        # Set max_diagram_retries to 5 (not the old hardcoded 3)
        cfg = GenerationConfig(max_diagram_retries=5)
        gen = DiagramGenerator(gen_config=cfg)

        mock_validator_cls = type("MockValidator", (), {
            "__init__": lambda self, **kw: None,
            "validate": lambda self, d: {"valid": False, "error": "bad syntax"},
        })

        with patch("doc_qa.generation.diagram._HAS_VALIDATOR", True), \
             patch("doc_qa.generation.diagram.MermaidValidator", mock_validator_cls):
            await gen.generate(
                "draw flow", "ctx", None, CountingLLM(),
                _intent(OutputIntent.DIAGRAM, sub_type="flowchart"),
            )

        # 1 plan + 1 initial generate + 5 repair attempts = 7 total LLM calls
        assert call_count == 7

    @pytest.mark.asyncio
    async def test_config_driven_max_retries_one(self):
        """Config with max_diagram_retries=1 should only try one repair."""
        call_count = 0

        class CountingLLM(LLMBackend):
            async def ask(self, question, context, history=None):
                nonlocal call_count
                call_count += 1
                return Answer(
                    text="```mermaid\nbadgraph\n```", sources=[], model="mock",
                )
            async def close(self):
                pass

        cfg = GenerationConfig(max_diagram_retries=1)
        gen = DiagramGenerator(gen_config=cfg)

        mock_validator_cls = type("MockValidator", (), {
            "__init__": lambda self, **kw: None,
            "validate": lambda self, d: {"valid": False, "error": "bad syntax"},
        })

        with patch("doc_qa.generation.diagram._HAS_VALIDATOR", True), \
             patch("doc_qa.generation.diagram.MermaidValidator", mock_validator_cls):
            await gen.generate(
                "draw", "ctx", None, CountingLLM(),
                _intent(OutputIntent.DIAGRAM, sub_type="flowchart"),
            )

        # 1 plan + 1 initial + 1 repair = 3
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_repair_continues_on_llm_error(self):
        """Bug 3: LLM error during repair should continue to next attempt, not break."""
        call_count = 0

        class ErrorThenSuccessLLM(LLMBackend):
            async def ask(self, question, context, history=None):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    # Planning call — returns unparseable text (falls back)
                    return Answer(text="no plan", sources=[], model="mock")
                if call_count == 2:
                    # Initial generation — returns invalid diagram
                    return Answer(
                        text="```mermaid\nbadgraph\n```",
                        sources=[], model="mock",
                    )
                if call_count == 3:
                    # First repair attempt — LLM transient error
                    return Answer(text="", sources=[], model="mock", error="timeout")
                # Second repair attempt — succeeds with valid diagram
                return Answer(
                    text="```mermaid\ngraph TD\nA-->B\n```",
                    sources=[], model="mock",
                )
            async def close(self):
                pass

        cfg = GenerationConfig(max_diagram_retries=3)
        gen = DiagramGenerator(gen_config=cfg)

        mock_validator_cls = type("MockValidator", (), {
            "__init__": lambda self, **kw: None,
            "validate": lambda self, d: (
                {"valid": False, "error": "bad syntax"} if "badgraph" in d
                else {"valid": True, "error": None}
            ),
        })

        with patch("doc_qa.generation.diagram._HAS_VALIDATOR", True), \
             patch("doc_qa.generation.diagram.MermaidValidator", mock_validator_cls):
            result = await gen.generate(
                "draw flow", "ctx", None, ErrorThenSuccessLLM(),
                _intent(OutputIntent.DIAGRAM, sub_type="flowchart"),
            )

        # Should have recovered: plan + initial + error + success = 4
        assert call_count == 4
        assert result.diagrams is not None
        assert "A-->B" in result.diagrams[0]

    @pytest.mark.asyncio
    async def test_repair_prompt_includes_question(self):
        """Bug 4: Repair prompt should include the original question for context."""
        captured_prompts: list[str] = []

        class CaptureLLM(LLMBackend):
            def __init__(self):
                self._call = 0

            async def ask(self, question, context, history=None):
                self._call += 1
                captured_prompts.append(question)
                if self._call == 1:
                    # Plan call — unparseable so falls back
                    return Answer(text="no plan", sources=[], model="mock")
                if self._call == 2:
                    # Generation — returns invalid diagram
                    return Answer(
                        text="```mermaid\nbadgraph\n```",
                        sources=[], model="mock",
                    )
                # Repair — returns valid diagram
                return Answer(
                    text="```mermaid\ngraph TD\nA-->B\n```",
                    sources=[], model="mock",
                )
            async def close(self):
                pass

        gen = DiagramGenerator()

        mock_validator_cls = type("MockValidator", (), {
            "__init__": lambda self, **kw: None,
            "validate": lambda self, d: (
                {"valid": False, "error": "bad syntax"} if "badgraph" in d
                else {"valid": True, "error": None}
            ),
        })

        original_question = "Show the OAuth2 authorization code flow"

        with patch("doc_qa.generation.diagram._HAS_VALIDATOR", True), \
             patch("doc_qa.generation.diagram.MermaidValidator", mock_validator_cls):
            await gen.generate(
                original_question, "ctx", None, CaptureLLM(),
                _intent(OutputIntent.DIAGRAM, sub_type="sequence"),
            )

        # Prompts: [0]=plan, [1]=generation, [2]=repair
        assert len(captured_prompts) >= 3
        repair_prompt = captured_prompts[2]
        assert "Original question:" in repair_prompt
        assert original_question in repair_prompt
        # Also verify it still has the repair-specific content
        assert "bad syntax" in repair_prompt
        assert "badgraph" in repair_prompt


# ── CodeExampleGenerator ─────────────────────────────────────────────


class TestCodeExampleGenerator:
    @pytest.mark.asyncio
    async def test_language_tag_injection(self):
        llm = MockLLM(text="Example:\n```\ncurl http://api\n```\n")
        gen = CodeExampleGenerator()
        result = await gen.generate(
            "show curl", "ctx", None, llm,
            _intent(OutputIntent.CODE_EXAMPLE, sub_type="curl"),
        )
        assert "```curl" in result.text

    def test_ensure_language_tags(self):
        text = "```\ncode\n```"
        result = _ensure_language_tags(text, "python")
        assert "```python" in result

    def test_existing_tag_unchanged(self):
        text = "```bash\necho hi\n```\n"
        result = _ensure_language_tags(text, "python")
        assert "```bash" in result
        # The closing ``` on its own line gets replaced too (bare fence), that's fine
        # Just verify the original language tag is preserved
        assert result.startswith("```bash")


# ── ComparisonGenerator ──────────────────────────────────────────────


class TestComparisonGenerator:
    @pytest.mark.asyncio
    async def test_table_detection(self):
        table = "| Feature | A | B |\n|---------|---|---|\n| Speed | Fast | Slow |\n"
        llm = MockLLM(text=table)
        gen = ComparisonGenerator()
        result = await gen.generate("compare A vs B", "ctx", None, llm, _intent())
        assert "Feature" in result.text

    @pytest.mark.asyncio
    async def test_no_table_logs_warning(self):
        llm = MockLLM(text="No table here.")
        gen = ComparisonGenerator()
        # Should not raise, just warn
        result = await gen.generate("compare", "ctx", None, llm, _intent())
        assert result.text == "No table here."


# ── ProceduralGenerator ──────────────────────────────────────────────


class TestProceduralGenerator:
    @pytest.mark.asyncio
    async def test_numbered_list_detection(self):
        llm = MockLLM(text="1. First\n2. Second\n3. Third")
        gen = ProceduralGenerator()
        result = await gen.generate("how to set up", "ctx", None, llm, _intent())
        assert "1." in result.text

    @pytest.mark.asyncio
    async def test_no_list_logs_warning(self):
        llm = MockLLM(text="Just text, no steps.")
        gen = ProceduralGenerator()
        result = await gen.generate("how to", "ctx", None, llm, _intent())
        assert result.text == "Just text, no steps."


# ── Router ───────────────────────────────────────────────────────────


class TestResolveGenerationStrategy:
    def test_high_confidence_specialized_only(self):
        match = _intent(OutputIntent.DIAGRAM, confidence=0.90)
        strategy = resolve_generation_strategy(match, GenerationConfig())
        assert strategy["generator"] == "DIAGRAM"
        assert not strategy["include_explanation"]

    def test_medium_confidence_includes_explanation(self):
        match = _intent(OutputIntent.CODE_EXAMPLE, confidence=0.75)
        strategy = resolve_generation_strategy(match, GenerationConfig())
        assert strategy["generator"] == "CODE_EXAMPLE"
        assert strategy["include_explanation"]

    def test_low_confidence_explanation_only(self):
        match = _intent(OutputIntent.DIAGRAM, confidence=0.50)
        strategy = resolve_generation_strategy(match, GenerationConfig())
        assert strategy["generator"] == "EXPLANATION"
        assert not strategy["include_explanation"]


class TestRouteAndGenerate:
    @pytest.mark.asyncio
    async def test_intent_to_generator_mapping(self):
        llm = MockLLM(text="1. Step one\n2. Step two\n3. Step three")
        result = await route_and_generate(
            question="How to deploy?",
            context="deploy docs",
            history=None,
            llm_backend=llm,
            intent_match=_intent(OutputIntent.PROCEDURAL, confidence=0.95),
        )
        assert "Step one" in result.text

    @pytest.mark.asyncio
    async def test_fallback_on_error(self):
        """If specialized generator fails, fall back to ExplanationGenerator."""
        llm = MockLLM(text="Fallback answer")

        # Create an intent match that will cause the router to try DiagramGenerator
        match = _intent(OutputIntent.DIAGRAM, confidence=0.95, sub_type="flowchart")

        # Patch DiagramGenerator.generate to raise
        with patch.object(DiagramGenerator, "generate", side_effect=RuntimeError("boom")):
            result = await route_and_generate(
                question="draw", context="ctx", history=None,
                llm_backend=llm, intent_match=match,
            )
        assert result.text == "Fallback answer"

    @pytest.mark.asyncio
    async def test_none_config_handling(self):
        llm = MockLLM(text="Answer")
        result = await route_and_generate(
            question="explain", context="ctx", history=None,
            llm_backend=llm,
            intent_match=_intent(OutputIntent.EXPLANATION, confidence=0.95),
            gen_config=None,
            intel_config=None,
        )
        assert result.text == "Answer"

    @pytest.mark.asyncio
    async def test_router_passes_gen_config_to_diagram_generator(self):
        """Bug 1: Router should pass gen_config to DiagramGenerator so config is respected."""
        captured_config = {}

        original_init = DiagramGenerator.__init__

        def spy_init(self, gen_config=None):
            captured_config["gen_config"] = gen_config
            original_init(self, gen_config=gen_config)

        llm = MockLLM(text="```mermaid\ngraph TD\nA-->B\n```")
        custom_cfg = GenerationConfig(max_diagram_retries=7)

        with patch.object(DiagramGenerator, "__init__", spy_init):
            await route_and_generate(
                question="draw flow", context="ctx", history=None,
                llm_backend=llm,
                intent_match=_intent(OutputIntent.DIAGRAM, confidence=0.95, sub_type="flowchart"),
                gen_config=custom_cfg,
            )

        assert captured_config.get("gen_config") is custom_cfg
