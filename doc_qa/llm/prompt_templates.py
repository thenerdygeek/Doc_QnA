"""Centralized prompt templates for all LLM interactions.

All prompts use {placeholders} for runtime values.  Use .format() (not f-strings)
to avoid accidental injection from user content.
"""

from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a documentation assistant. Answer the user's question "
    "based ONLY on the provided context. If the context does not contain "
    "enough information, say so clearly. Always cite sources using "
    "[Source: filename] format."
)

# ── Intent Classification ────────────────────────────────────────────

INTENT_CLASSIFICATION = """\
You are a query intent classifier for a documentation Q&A system.

Given a user's documentation question, classify what OUTPUT FORMAT would best \
answer their question. Think step-by-step about what the user actually wants \
to SEE in the response.

## Output format categories

- DIAGRAM: The user wants a visual representation (flowchart, sequence diagram, \
architecture diagram, ER diagram, state machine). They want to SEE relationships \
or flows between components.
- CODE_EXAMPLE: The user wants runnable code, API call examples, curl commands, \
JSON/YAML samples, or configuration snippets. They want something they can COPY \
and USE.
- COMPARISON_TABLE: The user wants a structured comparison of two or more things. \
They want to SEE differences side by side.
- PROCEDURAL: The user wants step-by-step instructions to accomplish a task. They \
want an ORDERED list of actions.
- EXPLANATION: The user wants a conceptual explanation, description, or answer to \
a factual question. This is the DEFAULT when nothing else fits.

## Examples

Query: "How does the OAuth2 authorization code flow work between the client, \
auth server, and resource server?"
Reasoning: The user asks about a FLOW between multiple COMPONENTS. They want to \
see how messages pass between parties. This is best shown as a sequence diagram.
Intent: DIAGRAM
Sub-type: sequence

Query: "Show me how to call the /users endpoint with authentication headers"
Reasoning: The user says "show me how to call" an endpoint. They want a concrete \
API call example they can copy. This is a code example request.
Intent: CODE_EXAMPLE
Sub-type: curl

Query: "What are the differences between JWT and session-based authentication?"
Reasoning: The user asks about "differences between" two approaches. They want \
a structured comparison. This is a comparison request.
Intent: COMPARISON_TABLE
Sub-type: none

Query: "How do I set up SSO with our SAML provider?"
Reasoning: The user says "how do I set up" which is action-oriented. They want \
step-by-step instructions to accomplish a configuration task.
Intent: PROCEDURAL
Sub-type: none

Query: "What is the rate limit for the API?"
Reasoning: The user asks a factual question. They want a direct answer, not a \
diagram, code, table, or procedure.
Intent: EXPLANATION
Sub-type: none

Query: "Explain the retry mechanism in the message queue"
Reasoning: The user says "explain" a mechanism. Even though this involves a \
process, they want understanding, not a diagram or steps to follow.
Intent: EXPLANATION
Sub-type: none

## Your task

Classify this query:
Query: "{query}"
Reasoning:"""

# ── Query Decomposition ──────────────────────────────────────────────

QUERY_DECOMPOSITION = """\
A user asked a documentation question that may require multiple different \
output formats (e.g., a diagram AND a code example). Your job is to split \
this into separate, self-contained sub-queries. Each sub-query should be \
answerable independently.

Rules:
1. If the query only needs ONE output format, return it as a single sub-query.
2. Each sub-query must include enough context to be understood alone.
3. Preserve the original topic/subject in each sub-query.
4. Maximum {max_sub_queries} sub-queries.
5. Return exactly this format (one sub-query per line):

SUB-QUERY 1: <text>
SUB-QUERY 2: <text>

User query: "{query}"

Sub-queries:"""

# ── CRAG: Document Grading ───────────────────────────────────────────

DOCUMENT_GRADING = """\
You are evaluating the relevance of retrieved document chunks for answering \
a user's question. Grade each chunk as RELEVANT, PARTIAL, or IRRELEVANT.

RELEVANT: The chunk directly answers or strongly supports answering the question.
PARTIAL: The chunk contains some useful information but is not sufficient alone.
IRRELEVANT: The chunk has no useful information for answering the question.

User question: "{query}"

Retrieved chunks:
{chunks}

Grade each chunk on a separate line in this exact format:
Chunk 1: RELEVANT — <brief reason>
Chunk 2: PARTIAL — <brief reason>
...

Grades:"""

# ── CRAG: Query Rewriting ────────────────────────────────────────────

QUERY_REWRITE = """\
The user asked a documentation question, but the initial retrieval returned \
mostly irrelevant results. Rewrite the query to better match the documentation.

Original query: "{original_query}"

What was found (partial matches):
{partial_context}

Rewrite the query to be more specific and likely to match relevant documentation. \
Use terminology from the partial matches if helpful. Return ONLY the rewritten query, \
nothing else.

Rewritten query:"""

# ── Answer Verification ──────────────────────────────────────────────

VERIFICATION = """\
You are a fact-checker for a documentation Q&A system. Given the source \
documents and an answer, verify that the answer is accurate and well-supported.

Check for:
1. Claims that are NOT supported by the source documents (hallucinations).
2. Important information from the sources that was omitted.
3. Factual accuracy of any specific details (names, numbers, commands).

Source documents:
{sources}

Answer to verify:
{answer}

Original question: "{question}"

Respond in this exact format:
Verdict: PASS or FAIL
Confidence: <0.0 to 1.0>
Issues: <comma-separated list of issues, or "none">
Suggested fix: <brief fix suggestion, or "none">"""

# ── Diagram Planning ─────────────────────────────────────────────────

DIAGRAM_PLAN = """\
You are a diagram architect. Given a user's question and relevant documentation, \
plan a clear, focused diagram.

Decide:
1. What diagram TYPE best represents this? Choose ONE:
   - flowchart: processes, pipelines, decision flows, build steps
   - sequence: interactions between actors/services/components over time
   - classDiagram: class hierarchies, interfaces, inheritance
   - erDiagram: data models, entity relationships
   - stateDiagram: state machines, lifecycle transitions
2. What are the KEY ENTITIES (max 8 nodes/actors)?
3. What are the KEY RELATIONSHIPS between them?

Keep it focused — fewer nodes with clear relationships is better than a cluttered \
diagram with everything.

User question: {question}

Respond in EXACTLY this format (no extra text):
DIAGRAM_TYPE: <type>
TITLE: <short title>
ENTITIES:
- <entity1>: <brief role>
- <entity2>: <brief role>
RELATIONSHIPS:
- <entity1> -> <entity2>: <label>
- <entity2> -> <entity3>: <label>"""

# ── Diagram Generation ───────────────────────────────────────────────

DIAGRAM_GENERATION = """\
Generate a Mermaid diagram to answer the user's question.

Diagram type to use: {diagram_type}

Important Mermaid syntax rules:
- Start with the diagram type declaration (e.g., "graph TD", "sequenceDiagram")
- For flowcharts: use --> for arrows, square brackets for process nodes, \
curly braces for decision nodes
- For sequence diagrams: use ->> for async, -> for sync arrows
- Escape semicolons as #59; in label text
- Never use a bare "end" as a node name — use "End" or "END" instead
- Keep node labels concise (NO parentheses, quotes, or special chars inside brackets)
- Use subgraph blocks for grouping related nodes

Wrap the diagram in a ```mermaid code fence.
Also provide a brief textual explanation of what the diagram shows."""

DIAGRAM_GENERATION_PLANNED = """\
Generate a Mermaid diagram based on the following plan.

Diagram type: {diagram_type}
Title: {title}

Entities to include:
{entities}

Relationships to show:
{relationships}

Important Mermaid syntax rules:
- Start with the diagram type declaration (e.g., "graph TD", "sequenceDiagram")
- For flowcharts: use --> for arrows, use square brackets [] for labels
- For sequence diagrams: use ->> for async, -> for sync arrows
- Escape semicolons as #59; in label text
- Never use a bare "end" as a node name — use "End" or "END" instead
- Keep node labels SHORT and CLEAN — no parentheses, quotes, or special \
characters inside square brackets
- Use subgraph blocks for grouping related nodes
- For node IDs use simple camelCase identifiers (e.g., buildStep, deployPhase)

Wrap the diagram in a ```mermaid code fence.
Also provide a brief textual explanation of what the diagram shows."""

# ── Proactive Diagram Suggestion (Phase 2) ──────────────────────────

DIAGRAM_SUGGEST = """\
You just generated a text explanation for a user's question. Evaluate whether \
adding a Mermaid diagram would SIGNIFICANTLY improve comprehension.

A diagram HELPS when the answer describes:
- A process or workflow with 3+ steps
- Interactions between 3+ components or services
- A hierarchy or inheritance structure
- State transitions or lifecycle
- Data flow or architecture

A diagram does NOT help when:
- The answer is a simple factual statement
- There are fewer than 3 entities or steps
- The content is purely conceptual without concrete relationships

User question: {question}

Answer given (first 500 chars):
{answer_preview}

Respond in EXACTLY this format (no extra text):
SUGGEST_DIAGRAM: YES or NO
DIAGRAM_TYPE: <flowchart|sequence|classDiagram|erDiagram|stateDiagram|none>
REASON: <one sentence>"""

# ── Diagram Repair ───────────────────────────────────────────────────

DIAGRAM_REPAIR = """\
The following Mermaid diagram has a syntax error. Fix the syntax while \
preserving the intended meaning.

Error: {error}

Broken diagram:
```mermaid
{diagram}
```

Return ONLY the corrected diagram in a ```mermaid code fence. Do not \
explain the changes."""

# ── Code Example Generation ──────────────────────────────────────────

CODE_EXAMPLE_GENERATION = """\
Include a concrete, runnable code example in your response.
Format: {code_format}
Use realistic values, not placeholders like <YOUR_TOKEN>.
If showing an API call, include the full request with headers and body.
Wrap code in a fenced code block with the appropriate language tag."""

# ── Comparison Table Generation ──────────────────────────────────────

COMPARISON_TABLE_GENERATION = """\
Present the comparison as a Markdown table with clear column headers.
Include at least 3 comparison dimensions (rows).
Add a brief summary after the table highlighting the key differences."""

# ── Procedural Generation ────────────────────────────────────────────

PROCEDURAL_GENERATION = """\
Present the answer as numbered steps. Each step should be actionable.
Include any prerequisite steps at the beginning.
Include expected outcomes or verification steps where relevant."""
