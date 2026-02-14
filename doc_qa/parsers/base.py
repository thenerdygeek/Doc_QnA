"""Base parser interface for document formats."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ParsedSection:
    """A section extracted from a document.

    Represents a logical unit of content (heading + body) that will
    be passed to the chunker for splitting into embeddable chunks.
    """

    title: str
    content: str
    level: int = 1  # heading level (1=top, 2=sub, etc.)
    file_path: str = ""
    file_type: str = ""
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        """Title + content combined for embedding."""
        if self.title:
            return f"{self.title}\n\n{self.content}"
        return self.content

    def estimate_tokens(self) -> int:
        """Approximate token count (~4 chars per token for English)."""
        return len(self.full_text) // 4


class Parser(ABC):
    """Abstract base class for document parsers.

    Each parser handles one file format and returns a list of
    ParsedSection objects representing the document's structure.
    """

    @property
    @abstractmethod
    def supported_extensions(self) -> list[str]:
        """File extensions this parser handles (e.g., ['.adoc'])."""
        ...

    @abstractmethod
    def parse(self, file_path: Path) -> list[ParsedSection]:
        """Parse a document file into sections.

        Args:
            file_path: Absolute path to the document file.

        Returns:
            List of ParsedSection objects. Empty list if parsing fails.
        """
        ...

    def can_parse(self, file_path: Path) -> bool:
        """Check if this parser can handle the given file."""
        return file_path.suffix.lower() in self.supported_extensions
