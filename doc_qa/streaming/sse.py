"""SSE streaming for phased query responses."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import AsyncGenerator

from fastapi import Request
from sse_starlette.sse import ServerSentEvent

logger = logging.getLogger(__name__)


async def _persist_exchange(
    conversation_id: str,
    question: str,
    answer: str,
    metadata: dict | None = None,
) -> None:
    """Fire-and-forget: persist Q&A exchange to the database."""
    try:
        from doc_qa.db.repository import (
            add_message,
            get_conversation_with_messages,
            update_conversation_title,
        )

        # Auto-title from first question if conversation has no messages yet
        conv = await get_conversation_with_messages(conversation_id)
        if conv and not conv.get("messages"):
            title = question[:100].strip()
            if len(question) > 100:
                title += "..."
            await update_conversation_title(conversation_id, title)

        await add_message(conversation_id, "user", question)
        await add_message(conversation_id, "assistant", answer, metadata=metadata)
    except Exception as exc:
        logger.warning("Failed to persist exchange to DB: %s", exc)


async def streaming_query(
    question: str,
    pipeline,
    request: Request,
    session_history: list[dict],
    config,
    session_id: str = "",
    conversation_id: str | None = None,
) -> AsyncGenerator[ServerSentEvent, None]:
    """Phased SSE stream that emits events as each query stage completes.

    Event sequence:
        status(classifying) -> intent -> status(retrieving) -> sources ->
        status(grading) -> status(generating) -> answer ->
        status(verifying) -> verified -> done

    Optional phases (intent classification, CRAG grading, verification) are
    skipped gracefully when the corresponding feature is not configured or
    its module is not yet available.

    Args:
        question: The user's query text.
        pipeline: A ``QueryPipeline`` instance whose internals are accessed
            directly for staged execution.
        request: The FastAPI ``Request`` -- used to detect client disconnect.
        session_history: Mutable list of conversation turns for multi-turn.
        config: Application configuration object.
        session_id: Opaque session identifier echoed back to the client.

    Yields:
        ``ServerSentEvent`` instances with typed ``event`` fields.
    """
    t_start = time.monotonic()

    try:
        # ------------------------------------------------------------------
        # Phase 1: Intent classification (optional)
        # ------------------------------------------------------------------
        intent_match = None
        if getattr(config, "intelligence", None) and config.intelligence.enable_intent_classification:
            yield ServerSentEvent(
                data=json.dumps({"status": "classifying", "session_id": session_id}),
                event="status",
            )
            try:
                from doc_qa.intelligence.intent_classifier import classify_intent

                intent_match = await classify_intent(question, pipeline._llm)
                yield ServerSentEvent(
                    data=json.dumps({
                        "intent": intent_match.intent.value,
                        "confidence": round(intent_match.confidence, 3),
                    }),
                    event="intent",
                )
            except Exception as exc:
                logger.warning("Intent classification failed: %s", exc)

        # ------------------------------------------------------------------
        # Phase 2: Retrieval
        # ------------------------------------------------------------------
        if await request.is_disconnected():
            return

        yield ServerSentEvent(
            data=json.dumps({"status": "retrieving"}),
            event="status",
        )

        candidates = pipeline._retriever.search(
            query=question,
            top_k=pipeline._candidate_pool,
            min_score=pipeline._min_score,
        )

        if not candidates:
            yield ServerSentEvent(
                data=json.dumps({
                    "answer": "I couldn't find any relevant information.",
                    "sources": [],
                    "model": "none",
                    "session_id": session_id,
                }),
                event="answer",
            )
            yield ServerSentEvent(
                data=json.dumps({
                    "status": "complete",
                    "elapsed": round(time.monotonic() - t_start, 2),
                }),
                event="done",
            )
            return

        # Rerank
        reranked = candidates
        if pipeline._rerank and len(candidates) > 1:
            from doc_qa.retrieval.reranker import rerank

            reranked = rerank(query=question, chunks=candidates, top_k=None)

        diverse = pipeline._apply_file_diversity(reranked)
        top_chunks = diverse[: pipeline._top_k]

        # Emit sources
        sources_payload = [
            {
                "file_path": c.file_path,
                "section_title": c.section_title,
                "score": round(c.score, 4),
            }
            for c in top_chunks
        ]
        yield ServerSentEvent(
            data=json.dumps({"sources": sources_payload, "chunks_retrieved": len(candidates)}),
            event="sources",
        )

        if await request.is_disconnected():
            return

        # ------------------------------------------------------------------
        # Phase 3: CRAG grading (optional)
        # ------------------------------------------------------------------
        if getattr(config, "verification", None) and config.verification.enable_crag:
            yield ServerSentEvent(
                data=json.dumps({"status": "grading"}),
                event="status",
            )
            try:
                from doc_qa.retrieval.corrective import corrective_retrieve

                top_chunks, was_rewritten = await corrective_retrieve(
                    query=question,
                    initial_chunks=top_chunks,
                    llm_backend=pipeline._llm,
                    retriever=pipeline._retriever,
                    max_rewrites=config.verification.max_crag_rewrites,
                    candidate_pool=pipeline._candidate_pool,
                    min_score=pipeline._min_score,
                )
            except Exception as exc:
                logger.warning("CRAG failed: %s", exc)

        if await request.is_disconnected():
            return

        # ------------------------------------------------------------------
        # Phase 4: Generation
        # ------------------------------------------------------------------
        yield ServerSentEvent(
            data=json.dumps({"status": "generating"}),
            event="status",
        )

        context = pipeline._build_context(top_chunks)

        gen_result_text = None
        gen_model = ""
        diagrams = None

        # Try specialized generation if intent was classified
        if intent_match is not None:
            try:
                from doc_qa.generation.router import route_and_generate

                gen_result = await route_and_generate(
                    question=question,
                    context=context,
                    history=session_history or None,
                    llm_backend=pipeline._llm,
                    intent_match=intent_match,
                    gen_config=config.generation if hasattr(config, "generation") else None,
                    intel_config=config.intelligence if hasattr(config, "intelligence") else None,
                )
                gen_result_text = gen_result.text
                gen_model = gen_result.model
                diagrams = gen_result.diagrams
            except Exception as exc:
                logger.warning("Specialized generation failed: %s", exc)

        # Fallback to standard LLM generation (with optional token streaming)
        if gen_result_text is None:
            if hasattr(pipeline._llm, 'ask_streaming'):
                try:
                    token_queue: asyncio.Queue[str | None] = asyncio.Queue()
                    prev_text = ""

                    def _on_token(cumulative: str) -> None:
                        nonlocal prev_text
                        delta = cumulative[len(prev_text):]
                        prev_text = cumulative
                        if delta:
                            token_queue.put_nowait(delta)

                    async def _run_streaming():
                        return await pipeline._llm.ask_streaming(
                            question=question, context=context,
                            history=session_history or None,
                            on_token=_on_token,
                        )

                    stream_task = asyncio.create_task(_run_streaming())

                    # Emit token events as they arrive
                    while not stream_task.done():
                        try:
                            delta = await asyncio.wait_for(token_queue.get(), timeout=0.1)
                            yield ServerSentEvent(
                                data=json.dumps({"token": delta}),
                                event="answer_token",
                            )
                        except asyncio.TimeoutError:
                            continue

                    # Drain remaining tokens
                    while not token_queue.empty():
                        delta = token_queue.get_nowait()
                        yield ServerSentEvent(
                            data=json.dumps({"token": delta}),
                            event="answer_token",
                        )

                    stream_answer = stream_task.result()
                    gen_result_text = stream_answer.text
                    gen_model = stream_answer.model
                except Exception as exc:
                    logger.warning("Streaming generation failed, falling back: %s", exc)
                    # Fall through to non-streaming below

            if gen_result_text is None:
                answer = await pipeline._llm.ask(
                    question=question,
                    context=context,
                    history=session_history or None,
                )
                gen_result_text = answer.text
                gen_model = answer.model

        # Emit answer
        answer_payload: dict = {
            "answer": gen_result_text,
            "model": gen_model,
            "session_id": session_id,
        }
        if diagrams:
            answer_payload["diagrams"] = diagrams
        yield ServerSentEvent(data=json.dumps(answer_payload), event="answer")

        # ------------------------------------------------------------------
        # Phase 4b: Source attribution (optional)
        # ------------------------------------------------------------------
        if gen_result_text and top_chunks:
            try:
                from doc_qa.verification.source_attributor import attribute_sources

                attrs = attribute_sources(gen_result_text, top_chunks)
                if attrs:
                    yield ServerSentEvent(
                        data=json.dumps({"attributions": [
                            {"sentence": a.sentence, "source_index": a.source_index, "similarity": round(a.similarity, 4)}
                            for a in attrs
                        ]}),
                        event="attribution",
                    )
            except Exception as exc:
                logger.debug("Attribution failed: %s", exc)

        if await request.is_disconnected():
            return

        # ------------------------------------------------------------------
        # Phase 5: Verification (optional)
        # ------------------------------------------------------------------
        verification_result = None
        if getattr(config, "verification", None) and config.verification.enable_verification:
            yield ServerSentEvent(
                data=json.dumps({"status": "verifying"}),
                event="status",
            )
            try:
                from doc_qa.verification.verifier import verify_answer

                verification_result = await verify_answer(
                    question=question,
                    answer=gen_result_text,
                    source_texts=[c.text for c in top_chunks],
                    llm_backend=pipeline._llm,
                )
                yield ServerSentEvent(
                    data=json.dumps({
                        "passed": verification_result.passed,
                        "confidence": round(verification_result.confidence, 3),
                    }),
                    event="verified",
                )
            except Exception as exc:
                logger.warning("Verification failed: %s", exc)

        # ------------------------------------------------------------------
        # Update session history
        # ------------------------------------------------------------------
        if gen_result_text:
            session_history.append({"role": "user", "text": question})
            session_history.append({"role": "assistant", "text": gen_result_text})
            max_entries = pipeline._max_history
            if len(session_history) > max_entries:
                del session_history[:-max_entries]

            # Persist to DB (fire-and-forget)
            if conversation_id:
                persist_meta = {}
                if sources_payload:
                    persist_meta["sources"] = sources_payload
                if diagrams:
                    persist_meta["diagrams"] = diagrams
                if verification_result is not None:
                    persist_meta["verification"] = {
                        "passed": verification_result.passed,
                        "confidence": round(verification_result.confidence, 3),
                    }
                asyncio.create_task(
                    _persist_exchange(
                        conversation_id=conversation_id,
                        question=question,
                        answer=gen_result_text,
                        metadata=persist_meta or None,
                    )
                )

        elapsed = round(time.monotonic() - t_start, 2)
        yield ServerSentEvent(
            data=json.dumps({"status": "complete", "elapsed": elapsed}),
            event="done",
        )

    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.exception("SSE query failed")
        yield ServerSentEvent(
            data=json.dumps({"error": str(exc), "type": type(exc).__name__}),
            event="error",
        )
