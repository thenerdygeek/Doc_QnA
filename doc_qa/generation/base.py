"""Base classes for the generation layer."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class GenerationResult:
    """Result from a specialized output generator."""

    text: str
    diagrams: list[str] | None = None
    format_instruction: str = ""
    model: str = ""


class OutputGenerator(ABC):
    """Abstract base class for specialized output generators."""

    @abstractmethod
    async def generate(
        self,
        question: str,
        context: str,
        history: list[dict] | None,
        llm_backend,
        intent_match,
    ) -> GenerationResult:
        """Generate a response in the specialized format."""
