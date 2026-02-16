"""Query pipeline — orchestrates retrieval, reranking, and LLM answering."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from doc_qa.retrieval.retriever import HybridRetriever, RetrievedChunk

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Complete result of a query pipeline execution."""

    answer: str
    sources: list[SourceReference]
    chunks_retrieved: int
    chunks_after_rerank: int
    model: str
    error: str | None = None
    attributions: list | None = None
    # Intelligence fields
    intent: str | None = None
    intent_confidence: float = 0.0
    confidence_score: float = 0.0
    is_abstained: bool = False
    verification_passed: bool | None = None
    detected_formats: dict | None = None
    diagrams: list[str] | None = None
    sub_results: list[QueryResult] | None = None


@dataclass
class SourceReference:
    """A source citation from a retrieved chunk."""

    file_path: str
    section_title: str
    chunk_id: str
    score: float


class QueryPipeline:
    """Orchestrates the full query flow with optional intelligence features.

    Core flow (always active):
        retrieve → rerank → diversify → build context → LLM → answer

    Optional features (each independently toggleable via config):
        - Intent classification (routes to specialized generators)
        - Multi-intent decomposition (compound queries)
        - CRAG (corrective retrieval with document grading)
        - Specialized generation (diagrams, code, tables, procedures)
        - Answer verification (generate-then-verify)
        - Confidence scoring with abstention
        - Output format detection
        - Source attribution
    """

    def __init__(
        self,
        table: Any,
        llm_backend: Any,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        search_mode: str = "hybrid",
        candidate_pool: int = 20,
        top_k: int = 5,
        min_score: float = 0.3,
        max_chunks_per_file: int = 2,
        rerank: bool = True,
        max_history_turns: int = 10,
        intelligence_config: Any | None = None,
        generation_config: Any | None = None,
        verification_config: Any | None = None,
    ) -> None:
        self._retriever = HybridRetriever(
            table=table,
            embedding_model=embedding_model,
            mode=search_mode,
        )
        self._llm = llm_backend
        self._candidate_pool = candidate_pool
        self._top_k = top_k
        self._min_score = min_score
        self._max_per_file = max_chunks_per_file
        self._rerank = rerank
        self._history: list[dict] = []
        self._max_history = max_history_turns * 2  # each turn = question + answer
        self._intel_config = intelligence_config
        self._gen_config = generation_config
        self._verify_config = verification_config

    async def query(self, question: str) -> QueryResult:
        """Execute the full query pipeline with optional intelligence features."""
        # Step 1: Intent classification (if enabled)
        intent_match = await self._classify_intent(question)

        # Step 2: Query decomposition (if multi-intent detected)
        sub_queries = await self._decompose_if_needed(question, intent_match)

        # Step 3: Process sub-queries
        if len(sub_queries) > 1:
            sub_results = []
            for sq_text, sq_intent in sub_queries:
                result = await self._process_single(sq_text, sq_intent)
                sub_results.append(result)
            return self._merge_results(sub_results, intent_match)

        sq_text, sq_intent = sub_queries[0]
        return await self._process_single(sq_text, sq_intent)

    async def _classify_intent(self, question: str) -> Any | None:
        """Classify query intent if intelligence features are enabled."""
        if not self._intel_config or not self._intel_config.enable_intent_classification:
            return None
        try:
            from doc_qa.intelligence.intent_classifier import classify_intent
            return await classify_intent(question, self._llm)
        except Exception as exc:
            logger.warning("Intent classification failed: %s", exc)
            return None

    async def _decompose_if_needed(
        self, question: str, intent_match: Any | None,
    ) -> list[tuple[str, Any | None]]:
        """Decompose query if multi-intent is detected."""
        if not self._intel_config or not self._intel_config.enable_multi_intent:
            return [(question, intent_match)]
        try:
            from doc_qa.intelligence.query_analyzer import decompose_query
            decomposed = await decompose_query(
                question, self._llm, max_sub_queries=self._intel_config.max_sub_queries,
            )
            if decomposed.is_multi_intent:
                logger.info("Decomposed into %d sub-queries.", len(decomposed.sub_queries))
                return [(sq.query_text, sq.intent) for sq in decomposed.sub_queries]
        except Exception as exc:
            logger.warning("Query decomposition failed: %s", exc)
        return [(question, intent_match)]

    async def _process_single(
        self, question: str, intent_match: Any | None,
    ) -> QueryResult:
        """Process a single query through the full pipeline."""
        # ── Retrieve ──────────────────────────────────────────────
        candidates = self._retriever.search(
            query=question,
            top_k=self._candidate_pool,
            min_score=self._min_score,
        )
        logger.info("Retrieved %d candidates.", len(candidates))

        if not candidates:
            return QueryResult(
                answer="I couldn't find any relevant information in the indexed documents.",
                sources=[],
                chunks_retrieved=0,
                chunks_after_rerank=0,
                model="none",
            )

        # ── Near-duplicate dedup (version-aware) ─────────────────
        # Remove near-duplicate chunks from different file versions,
        # keeping the one from the most recently dated file.
        from doc_qa.retrieval.dedup import deduplicate_near_duplicates
        candidates = deduplicate_near_duplicates(candidates)

        # ── Rerank ────────────────────────────────────────────────
        reranked = candidates
        if self._rerank and len(candidates) > 1:
            from doc_qa.retrieval.reranker import rerank
            reranked = rerank(query=question, chunks=candidates, top_k=None)
            logger.info("Reranked to %d chunks.", len(reranked))

        # ── File diversity cap ────────────────────────────────────
        diverse = self._apply_file_diversity(reranked)
        top_chunks = diverse[: self._top_k]

        # ── CRAG: Grade + rewrite if needed ───────────────────────
        was_rewritten = False
        if self._verify_config and self._verify_config.enable_crag:
            try:
                from doc_qa.retrieval.corrective import corrective_retrieve
                top_chunks, was_rewritten = await corrective_retrieve(
                    query=question,
                    initial_chunks=top_chunks,
                    llm_backend=self._llm,
                    retriever=self._retriever,
                    max_rewrites=self._verify_config.max_crag_rewrites,
                    candidate_pool=self._candidate_pool,
                    min_score=self._min_score,
                )
                if was_rewritten:
                    logger.info("CRAG rewrote query; %d chunks after re-retrieval.", len(top_chunks))
            except Exception as exc:
                logger.warning("CRAG failed, using original chunks: %s", exc)

        # ── Build context ─────────────────────────────────────────
        context = self._build_context(top_chunks)
        sources = [
            SourceReference(
                file_path=c.file_path,
                section_title=c.section_title,
                chunk_id=c.chunk_id,
                score=c.score,
            )
            for c in top_chunks
        ]

        # ── Generate (specialized or standard) ────────────────────
        answer_text = ""
        answer_model = ""
        diagrams = None

        if intent_match is not None:
            try:
                from doc_qa.generation.router import route_and_generate
                gen_result = await route_and_generate(
                    question=question,
                    context=context,
                    history=self._history if self._history else None,
                    llm_backend=self._llm,
                    intent_match=intent_match,
                    gen_config=self._gen_config,
                    intel_config=self._intel_config,
                )
                answer_text = gen_result.text
                answer_model = gen_result.model
                diagrams = gen_result.diagrams
            except Exception as exc:
                logger.warning("Specialized generation failed, falling back: %s", exc)

        if not answer_text:
            answer = await self._llm.ask(
                question=question,
                context=context,
                history=self._history if self._history else None,
            )
            answer_text = answer.text
            answer_model = answer.model
            if answer.error:
                return QueryResult(
                    answer=answer_text,
                    sources=sources,
                    chunks_retrieved=len(candidates),
                    chunks_after_rerank=len(top_chunks),
                    model=answer_model,
                    error=answer.error,
                )

        # ── Detect output formats ─────────────────────────────────
        detected_formats = None
        try:
            from doc_qa.intelligence.output_detector import detect_response_formats
            fmt = detect_response_formats(answer_text)
            detected_formats = {
                "has_mermaid": fmt.has_mermaid,
                "has_code_blocks": fmt.has_code_blocks,
                "has_table": fmt.has_table,
                "has_numbered_list": fmt.has_numbered_list,
            }
            if fmt.has_mermaid and not diagrams:
                diagrams = fmt.mermaid_blocks
        except Exception:
            pass

        # ── Validate diagrams (if mermaid detected) ───────────────
        # Filter out invalid diagrams so the frontend never receives
        # broken Mermaid syntax that would fail to render.
        if diagrams and self._gen_config and self._gen_config.enable_diagrams:
            try:
                from doc_qa.verification.mermaid_validator import MermaidValidator
                validator = MermaidValidator(
                    node_script_path=self._gen_config.node_script_path,
                    mode=self._gen_config.mermaid_validation,
                )
                valid_diagrams: list[str] = []
                for i, diagram in enumerate(diagrams):
                    result = validator.validate(diagram)
                    if result["valid"]:
                        valid_diagrams.append(diagram)
                    else:
                        logger.warning(
                            "Mermaid diagram %d invalid (filtered out): %s",
                            i + 1,
                            result.get("error"),
                        )
                diagrams = valid_diagrams or None
            except Exception as exc:
                logger.debug("Mermaid validation unavailable: %s", exc)

        # ── Verify answer ─────────────────────────────────────────
        verification = None
        if self._verify_config and self._verify_config.enable_verification:
            try:
                from doc_qa.verification.verifier import verify_answer
                verification = await verify_answer(
                    question=question,
                    answer=answer_text,
                    source_texts=[c.text for c in top_chunks],
                    llm_backend=self._llm,
                )
            except Exception as exc:
                logger.warning("Verification failed: %s", exc)

        # ── Confidence scoring ────────────────────────────────────
        confidence_score = 0.0
        is_abstained = False
        if self._verify_config:
            try:
                from doc_qa.intelligence.confidence import compute_confidence
                assessment = compute_confidence(
                    retrieval_scores=[c.score for c in top_chunks],
                    verification=verification,
                    config=self._verify_config,
                )
                confidence_score = assessment.score
                is_abstained = assessment.should_abstain
                if is_abstained:
                    answer_text = (
                        "I don't have enough information to answer this accurately. "
                        + (assessment.abstain_reason or "")
                    ).strip()
                    logger.info("Abstaining: confidence=%.2f", confidence_score)
            except Exception as exc:
                logger.debug("Confidence scoring unavailable: %s", exc)

        # ── Source attribution ──────────────────────────────────────
        attributions = None
        if answer_text and not is_abstained:
            try:
                from doc_qa.verification.source_attributor import attribute_sources

                attrs = attribute_sources(answer_text, top_chunks)
                attributions = [
                    {"sentence": a.sentence, "source_index": a.source_index, "similarity": round(a.similarity, 4)}
                    for a in attrs
                ] if attrs else None
            except Exception as exc:
                logger.debug("Source attribution unavailable: %s", exc)

        # ── Update history (only on success, non-abstained) ───────
        if not is_abstained:
            self._history.append({"role": "user", "text": question})
            self._history.append({"role": "assistant", "text": answer_text})
            max_entries = self._max_history
            if len(self._history) > max_entries:
                del self._history[:-max_entries]

        intent_name = None
        intent_conf = 0.0
        if intent_match is not None:
            try:
                intent_name = intent_match.intent.value
                intent_conf = intent_match.confidence
            except AttributeError:
                pass

        return QueryResult(
            answer=answer_text,
            sources=sources,
            chunks_retrieved=len(candidates),
            chunks_after_rerank=len(top_chunks),
            model=answer_model,
            attributions=attributions,
            intent=intent_name,
            intent_confidence=intent_conf,
            confidence_score=confidence_score,
            is_abstained=is_abstained,
            verification_passed=verification.passed if verification else None,
            detected_formats=detected_formats,
            diagrams=diagrams,
        )

    def _merge_results(
        self, sub_results: list[QueryResult], intent_match: Any | None,
    ) -> QueryResult:
        """Merge multiple sub-query results into a single response."""
        if not sub_results:
            return QueryResult(
                answer="No results.", sources=[], chunks_retrieved=0,
                chunks_after_rerank=0, model="none",
            )

        parts: list[str] = []
        all_sources: list[SourceReference] = []
        seen_chunks: set[str] = set()
        total_retrieved = 0
        total_reranked = 0
        all_diagrams: list[str] = []

        for i, r in enumerate(sub_results):
            if len(sub_results) > 1:
                parts.append(f"### Part {i + 1}\n")
            parts.append(r.answer)
            parts.append("")
            total_retrieved += r.chunks_retrieved
            total_reranked += r.chunks_after_rerank
            for s in r.sources:
                if s.chunk_id not in seen_chunks:
                    all_sources.append(s)
                    seen_chunks.add(s.chunk_id)
            if r.diagrams:
                all_diagrams.extend(r.diagrams)

        # Use first result's model and confidence as representative
        first = sub_results[0]
        return QueryResult(
            answer="\n".join(parts).strip(),
            sources=all_sources,
            chunks_retrieved=total_retrieved,
            chunks_after_rerank=total_reranked,
            model=first.model,
            intent=first.intent,
            intent_confidence=first.intent_confidence,
            confidence_score=min(r.confidence_score for r in sub_results),
            is_abstained=any(r.is_abstained for r in sub_results),
            verification_passed=all(
                r.verification_passed for r in sub_results if r.verification_passed is not None
            ) if any(r.verification_passed is not None for r in sub_results) else None,
            diagrams=all_diagrams or None,
            sub_results=sub_results,
        )

    def reset_history(self) -> None:
        """Clear conversation history for a fresh session."""
        self._history.clear()

    def _apply_file_diversity(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Limit chunks per file to ensure source diversity."""
        file_counts: dict[str, int] = defaultdict(int)
        result: list[RetrievedChunk] = []

        for chunk in chunks:
            if file_counts[chunk.file_path] < self._max_per_file:
                result.append(chunk)
                file_counts[chunk.file_path] += 1

        return result

    @staticmethod
    def _build_context(chunks: list[RetrievedChunk]) -> str:
        """Format retrieved chunks into a context string for the LLM.

        Includes document dates when available so the LLM can prefer
        more recent sources when information conflicts.
        """
        if not chunks:
            return ""

        from datetime import datetime, timezone

        parts: list[str] = []
        for i, chunk in enumerate(chunks, 1):
            filename = Path(chunk.file_path).name
            header = f"[Source {i}: {filename}"
            # Include document date if available
            if chunk.doc_date > 0:
                date_str = datetime.fromtimestamp(
                    chunk.doc_date, tz=timezone.utc
                ).strftime("%Y-%m-%d")
                header += f" ({date_str})"
            if chunk.section_title:
                header += f" > {chunk.section_title}"
            header += f"] (score: {chunk.score:.3f})"

            parts.append(header)
            parts.append(chunk.text)
            parts.append("")  # blank line separator

        return "\n".join(parts)
