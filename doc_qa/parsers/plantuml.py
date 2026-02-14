"""PlantUML parser — regex-based extraction of diagram elements."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from doc_qa.parsers.base import ParsedSection, Parser

logger = logging.getLogger(__name__)

# Participant declarations
_PARTICIPANT_RE = re.compile(
    r"^\s*(participant|actor|boundary|control|entity|database|queue|collections)\s+"
    r'(?:"([^"]+)"|(\S+))(?:\s+as\s+(\S+))?',
    re.IGNORECASE,
)

# Messages: A -> B : message text
_MESSAGE_RE = re.compile(
    r"^\s*(\S+)\s*(<?-+>?)\s*(\S+)\s*:\s*(.+)$"
)

# Notes: note left of A : text  OR  note over A,B : text
_NOTE_INLINE_RE = re.compile(
    r'^\s*note\s+(?:left|right|over)\s+(?:of\s+)?[\w,\s]+\s*:\s*(.+)$',
    re.IGNORECASE,
)

# Title: title My Diagram
_TITLE_RE = re.compile(r"^\s*title\s+(.+)$", re.IGNORECASE)

# Group blocks: alt, opt, loop, group, par, break, critical
_GROUP_START_RE = re.compile(
    r"^\s*(alt|else|opt|loop|group|par|break|critical)\s*(.*)?$",
    re.IGNORECASE,
)
_GROUP_END_RE = re.compile(r"^\s*end\s*$", re.IGNORECASE)

# Component/class declarations
_COMPONENT_RE = re.compile(
    r'^\s*(?:component|class|interface|package|node|folder|frame|cloud|database)\s+'
    r'(?:"([^"]+)"|(\S+))(?:\s+as\s+(\S+))?',
    re.IGNORECASE,
)

# Relationship: A --> B : label
_RELATIONSHIP_RE = re.compile(
    r"^\s*(\S+)\s*([-.<>|*o#+]+)\s*(\S+)\s*(?::\s*(.+))?$"
)


class PlantUMLParser(Parser):
    """Parse PlantUML diagram files into natural language summaries."""

    @property
    def supported_extensions(self) -> list[str]:
        return [".puml", ".plantuml", ".pu"]

    def parse(self, file_path: Path) -> list[ParsedSection]:
        try:
            return self._parse_diagram(file_path)
        except Exception:
            logger.exception("Failed to parse %s", file_path)
            return []

    def _parse_diagram(self, file_path: Path) -> list[ParsedSection]:
        with open(file_path, encoding="utf-8") as f:
            lines = f.readlines()

        # Pre-scan: detect diagram type by looking for type-specific keywords
        raw_text = "".join(lines)
        has_sequence_kw = bool(
            re.search(r"^\s*(participant|actor|boundary|control|entity)\s+", raw_text, re.MULTILINE)
        )
        has_component_kw = bool(
            re.search(
                r"^\s*(component|class|interface|package|node|folder|frame|cloud)\s+",
                raw_text,
                re.MULTILINE,
            )
        )
        # Sequence if it has participant/actor keywords and no component/class keywords
        # Otherwise treat as structural/component diagram
        is_sequence = has_sequence_kw or (not has_component_kw and not has_sequence_kw)

        title = file_path.stem
        participants: dict[str, dict[str, str]] = {}  # alias → {name, kind}
        messages: list[dict[str, str]] = []
        notes: list[str] = []
        components: dict[str, dict[str, str]] = {}  # alias → {name, kind}
        relationships: list[dict[str, str]] = []
        groups: list[str] = []  # active group labels

        for line in lines:
            stripped = line.strip()

            # Skip empty lines, comments, and start/end markers
            if not stripped or stripped.startswith("'") or stripped.startswith("@"):
                continue

            # Title
            m = _TITLE_RE.match(stripped)
            if m:
                title = m.group(1).strip()
                continue

            if is_sequence:
                # Participants (sequence diagrams)
                m = _PARTICIPANT_RE.match(stripped)
                if m:
                    kind = m.group(1).lower()
                    name = m.group(2) or m.group(3)
                    alias = m.group(4) or name
                    participants[alias] = {"name": name, "kind": kind}
                    continue

                # Messages (sequence diagrams)
                m = _MESSAGE_RE.match(stripped)
                if m:
                    messages.append({
                        "from": m.group(1),
                        "to": m.group(3),
                        "arrow": m.group(2),
                        "message": m.group(4).strip(),
                    })
                    continue
            else:
                # Components (class/component diagrams) — try before participant
                m = _COMPONENT_RE.match(stripped)
                if m:
                    name = m.group(1) or m.group(2)
                    alias = m.group(3) or name
                    components[alias] = {"name": name}
                    continue

            # Inline notes
            m = _NOTE_INLINE_RE.match(stripped)
            if m:
                notes.append(m.group(1).strip())
                continue

            # Group blocks
            m = _GROUP_START_RE.match(stripped)
            if m:
                label = m.group(1)
                condition = m.group(2).strip() if m.group(2) else ""
                if condition:
                    groups.append(f"{label}: {condition}")
                continue

            # Relationships
            m = _RELATIONSHIP_RE.match(stripped)
            if m:
                rel = {
                    "from": m.group(1),
                    "to": m.group(3),
                }
                if m.group(4):
                    rel["label"] = m.group(4).strip()
                relationships.append(rel)

        # Build natural language summary
        summary_parts: list[str] = []

        if is_sequence:
            summary_parts.append(f"Sequence diagram: {title}")

            if participants:
                names = [f"{v['name']} ({v['kind']})" for v in participants.values()]
                summary_parts.append(f"Participants: {', '.join(names)}")

            if messages:
                summary_parts.append("Flow:")
                for i, msg in enumerate(messages, 1):
                    sender = participants.get(msg["from"], {}).get("name", msg["from"])
                    receiver = participants.get(msg["to"], {}).get("name", msg["to"])
                    summary_parts.append(f"  {i}. {sender} → {receiver}: {msg['message']}")

            if groups:
                summary_parts.append(f"Conditional blocks: {', '.join(groups)}")
        else:
            summary_parts.append(f"Diagram: {title}")

            if components:
                names = [v["name"] for v in components.values()]
                summary_parts.append(f"Components: {', '.join(names)}")

            if relationships:
                summary_parts.append("Relationships:")
                for rel in relationships:
                    label = rel.get("label", "")
                    line = f"  {rel['from']} → {rel['to']}"
                    if label:
                        line += f": {label}"
                    summary_parts.append(line)

        if notes:
            summary_parts.append("Notes:")
            for note in notes:
                summary_parts.append(f"  - {note}")

        content = "\n".join(summary_parts)

        if not content.strip():
            return []

        section = ParsedSection(
            title=title,
            content=content,
            level=1,
            file_path=str(file_path),
            file_type="puml",
        )

        logger.info("Parsed %s: %s diagram", file_path.name, "sequence" if is_sequence else "component")
        return [section]
