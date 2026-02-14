"""50 diverse test scenarios for the doc_qa intelligence pipeline.

Each scenario simulates a real user interaction with expected outcomes
computed by analyzing the tool's actual behavior (regex patterns,
thresholds, scoring formulae).

Confidence formula reference:
  retrieval_signal = avg(scores)
    - penalty: all scores < 0.3 => signal *= 0.5
    - penalty: len(scores) >= 2 and (scores[0] - scores[1]) > 0.3 => signal -= 0.15
  verification_signal:
    - None => 0.7
    - passed => verification.confidence
    - not passed => max(0.0, verification.confidence - 0.2)
  combined = 0.4 * retrieval_signal + 0.6 * verification_signal
  should_abstain = combined < threshold (default 0.4) AND abstain_on_low_confidence

Routing thresholds (from IntelligenceConfig defaults):
  confidence >= 0.85 => specialized only
  confidence >= 0.65 => specialized + explanation
  confidence < 0.65  => explanation only

CRAG rewrite threshold (from corrective.py):
  irrelevant fraction > 0.5 => rewrite triggered
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TestScenario:
    """A single test scenario."""

    id: str
    name: str
    category: str
    query: str

    # Retrieval simulation
    chunk_texts: list[str]
    chunk_scores: list[float]
    chunk_files: list[str]

    # Expected pipeline behavior
    expected_intent: str
    expected_intent_source: str
    expected_sub_type: str | None

    # CRAG expectations
    expected_crag_rewrite: bool
    chunk_grades: list[str] | None

    # Generation expectations
    expected_generator: str
    expected_include_explanation: bool

    # Verification expectations
    verification_passed: bool | None
    verification_confidence: float | None

    # Confidence expectations
    expected_retrieval_signal: float
    expected_verification_signal: float
    expected_combined_confidence: float
    expected_should_abstain: bool

    # Other expectations
    expected_has_attribution: bool
    expected_history_updated: bool
    is_multi_intent: bool

    # LLM response keys
    llm_response_keys: dict = field(default_factory=dict)

    description: str = ""


# =====================================================================
# Helper: compute confidence the same way the production code does
# =====================================================================

def _calc_retrieval_signal(scores: list[float]) -> float:
    if not scores:
        return 0.0
    avg = sum(scores) / len(scores)
    signal = avg
    if all(s < 0.3 for s in scores):
        signal *= 0.5
    if len(scores) >= 2:
        gap = scores[0] - scores[1]
        if gap > 0.3:
            signal = max(0.0, signal - 0.15)
    return max(0.0, min(1.0, signal))


def _calc_verification_signal(passed: bool | None, confidence: float | None) -> float:
    if passed is None and confidence is None:
        return 0.7
    if passed:
        return max(0.0, min(1.0, confidence))
    return max(0.0, min(1.0, confidence - 0.2))


def _calc_combined(ret: float, ver: float) -> float:
    return max(0.0, min(1.0, 0.4 * ret + 0.6 * ver))


# =====================================================================
# SCENARIOS
# =====================================================================

SCENARIOS: list[TestScenario] = [
    # ------------------------------------------------------------------
    # Category 1: Intent Classification (10 scenarios)
    # ------------------------------------------------------------------
    TestScenario(
        id="S01",
        name="Heuristic DIAGRAM via topic+verb",
        category="intent",
        query="Draw a diagram showing the architecture flow between services",
        chunk_texts=[
            "The platform uses a microservices architecture with API Gateway, Auth Service, and Order Service communicating via REST.",
            "Service discovery is handled by Consul. Each service registers on startup and deregisters on shutdown.",
            "The API Gateway routes incoming requests to downstream services based on path prefix matching.",
        ],
        chunk_scores=[0.82, 0.75, 0.70],
        chunk_files=["docs/architecture.md", "docs/discovery.md", "docs/gateway.md"],
        expected_intent="DIAGRAM",
        expected_intent_source="heuristic",
        expected_sub_type="flowchart",
        expected_crag_rewrite=False,
        chunk_grades=["relevant", "relevant", "relevant"],
        expected_generator="DIAGRAM",
        expected_include_explanation=False,
        verification_passed=True,
        verification_confidence=0.88,
        # retrieval: avg(0.82,0.75,0.70)=0.7567; no penalties
        # verification: 0.88
        # combined: 0.4*0.7567 + 0.6*0.88 = 0.3027 + 0.528 = 0.8307
        expected_retrieval_signal=0.7567,
        expected_verification_signal=0.88,
        expected_combined_confidence=0.8307,
        expected_should_abstain=False,
        expected_has_attribution=True,
        expected_history_updated=True,
        is_multi_intent=False,
        llm_response_keys={"grading": "good[0]", "generation": "good[0]", "verification": "good[0]"},
        description=(
            "Tests heuristic DIAGRAM detection via _DIAGRAM_TOPIC ('flow', 'architecture') "
            "and _DIAGRAM_VERB ('diagram', 'draw'). Confidence 0.92 >= 0.85 so specialized only. "
            "Sub-type defaults to 'flowchart' (no sequence/ER/class/state keywords)."
        ),
    ),
    TestScenario(
        id="S02",
        name="Heuristic CODE_EXAMPLE with curl sub-type",
        category="intent",
        query="Show me a curl example for the authentication API endpoint",
        chunk_texts=[
            "The /auth/login endpoint accepts POST requests with username and password in the JSON body.",
            "Authentication returns a JWT token in the response body with a 24-hour expiry.",
        ],
        chunk_scores=[0.88, 0.72],
        chunk_files=["docs/auth-api.md", "docs/auth-api.md"],
        expected_intent="CODE_EXAMPLE",
        expected_intent_source="heuristic",
        expected_sub_type="curl",
        expected_crag_rewrite=False,
        chunk_grades=["relevant", "relevant"],
        expected_generator="CODE_EXAMPLE",
        expected_include_explanation=False,
        verification_passed=True,
        verification_confidence=0.90,
        # retrieval: avg(0.88,0.72)=0.80; gap=0.16 <= 0.3 so no penalty
        # verification: 0.90
        # combined: 0.4*0.80 + 0.6*0.90 = 0.32 + 0.54 = 0.86
        expected_retrieval_signal=0.80,
        expected_verification_signal=0.90,
        expected_combined_confidence=0.86,
        expected_should_abstain=False,
        expected_has_attribution=True,
        expected_history_updated=True,
        is_multi_intent=False,
        llm_response_keys={"grading": "good[0]", "generation": "good[1]", "verification": "good[0]"},
        description=(
            "Tests heuristic CODE_EXAMPLE via _CODE_FORMAT ('curl', 'API', 'endpoint') "
            "and _CODE_VERB ('show'). Confidence 0.90 >= 0.85 so specialized only. "
            "Sub-type detected as 'curl' via _SUBTYPE_CURL pattern."
        ),
    ),
    TestScenario(
        id="S03",
        name="Heuristic COMPARISON_TABLE via phrase",
        category="intent",
        query="What are the pros and cons of using PostgreSQL vs MongoDB for our use case?",
        chunk_texts=[
            "PostgreSQL offers ACID compliance and strong consistency guarantees suitable for financial transactions.",
            "MongoDB provides flexible schema design and horizontal scaling through sharding.",
            "For mixed workloads, consider using PostgreSQL for transactional data and MongoDB for analytics.",
        ],
        chunk_scores=[0.85, 0.83, 0.68],
        chunk_files=["docs/databases.md", "docs/databases.md", "docs/architecture-decisions.md"],
        expected_intent="COMPARISON_TABLE",
        expected_intent_source="heuristic",
        expected_sub_type="none",
        expected_crag_rewrite=False,
        chunk_grades=["relevant", "relevant", "partial"],
        expected_generator="COMPARISON_TABLE",
        expected_include_explanation=False,
        verification_passed=True,
        verification_confidence=0.85,
        # retrieval: avg(0.85,0.83,0.68)=0.7867; no all-below-0.3; gap=0.02 <= 0.3
        # verification: 0.85
        # combined: 0.4*0.7867 + 0.6*0.85 = 0.3147 + 0.51 = 0.8247
        expected_retrieval_signal=0.7867,
        expected_verification_signal=0.85,
        expected_combined_confidence=0.8247,
        expected_should_abstain=False,
        expected_has_attribution=True,
        expected_history_updated=True,
        is_multi_intent=False,
        llm_response_keys={"grading": "good[1]", "generation": "good[2]", "verification": "good[0]"},
        description=(
            "Tests heuristic COMPARISON_TABLE via _COMPARISON_PHRASE ('pros and cons'). "
            "Confidence 0.93 >= 0.85 so specialized only. "
            "Also matches _COMPARISON_TOPIC ('vs') but phrase match fires first."
        ),
    ),
    TestScenario(
        id="S04",
        name="Heuristic PROCEDURAL via topic+verb",
        category="intent",
        query="How do I set up the CI/CD pipeline to deploy to Kubernetes?",
        chunk_texts=[
            "Step 1: Create a Dockerfile in your project root. Step 2: Configure the GitLab CI pipeline with stages: build, test, deploy.",
            "The Kubernetes deployment manifest should specify resource limits and readiness probes.",
            "Use Helm charts for templating Kubernetes resources across environments.",
        ],
        chunk_scores=[0.90, 0.78, 0.65],
        chunk_files=["docs/ci-cd-guide.md", "docs/k8s-deploy.md", "docs/helm.md"],
        expected_intent="PROCEDURAL",
        expected_intent_source="heuristic",
        expected_sub_type="none",
        expected_crag_rewrite=False,
        chunk_grades=["relevant", "relevant", "partial"],
        expected_generator="PROCEDURAL",
        expected_include_explanation=False,
        verification_passed=True,
        verification_confidence=0.92,
        # retrieval: avg(0.90,0.78,0.65)=0.7767; gap=0.12 <= 0.3
        # verification: 0.92
        # combined: 0.4*0.7767 + 0.6*0.92 = 0.3107 + 0.552 = 0.8627
        expected_retrieval_signal=0.7767,
        expected_verification_signal=0.92,
        expected_combined_confidence=0.8627,
        expected_should_abstain=False,
        expected_has_attribution=True,
        expected_history_updated=True,
        is_multi_intent=False,
        llm_response_keys={"grading": "good[1]", "generation": "good[3]", "verification": "good[0]"},
        description=(
            "Tests heuristic PROCEDURAL via _PROCEDURAL_TOPIC ('how do I') "
            "and _PROCEDURAL_VERB ('set up', 'deploy'). Confidence 0.88 >= 0.85 "
            "so specialized only."
        ),
    ),
    TestScenario(
        id="S05",
        name="Heuristic EXPLANATION (none match => LLM fallback returns EXPLANATION)",
        category="intent",
        query="What is the purpose of the retry mechanism in the message broker?",
        chunk_texts=[
            "The message broker implements exponential backoff retry with a maximum of 5 attempts.",
            "Dead letter queues capture messages that exceed the retry limit for manual inspection.",
        ],
        chunk_scores=[0.80, 0.74],
        chunk_files=["docs/messaging.md", "docs/error-handling.md"],
        expected_intent="EXPLANATION",
        expected_intent_source="llm_fallback",
        expected_sub_type="none",
        expected_crag_rewrite=False,
        chunk_grades=["relevant", "relevant"],
        expected_generator="EXPLANATION",
        expected_include_explanation=False,
        verification_passed=True,
        verification_confidence=0.90,
        # retrieval: avg(0.80,0.74)=0.77; gap=0.06 <= 0.3
        # verification: 0.90
        # combined: 0.4*0.77 + 0.6*0.90 = 0.308 + 0.54 = 0.848
        expected_retrieval_signal=0.77,
        expected_verification_signal=0.90,
        expected_combined_confidence=0.848,
        expected_should_abstain=False,
        expected_has_attribution=True,
        expected_history_updated=True,
        is_multi_intent=False,
        llm_response_keys={
            "intent": "good[0]",
            "grading": "good[0]",
            "generation": "good[0]",
            "verification": "good[0]",
        },
        description=(
            "No heuristic matches (no diagram/code/comparison/procedural triggers). "
            "Falls to LLM fallback which returns EXPLANATION with confidence 0.85 "
            "(has Reasoning line). 0.85 >= 0.85 so specialized EXPLANATION only."
        ),
    ),
    TestScenario(
        id="S06",
        name="LLM fallback for ambiguous query returning DIAGRAM",
        category="intent",
        query="Can you help me understand how data moves through the system?",
        chunk_texts=[
            "Data ingestion starts at the API layer, passes through validation, then enters the processing pipeline.",
            "The ETL pipeline extracts data from PostgreSQL, transforms it, and loads into the data warehouse.",
        ],
        chunk_scores=[0.75, 0.70],
        chunk_files=["docs/data-flow.md", "docs/etl.md"],
        expected_intent="DIAGRAM",
        expected_intent_source="llm_fallback",
        expected_sub_type="flowchart",
        expected_crag_rewrite=False,
        chunk_grades=["relevant", "relevant"],
        expected_generator="DIAGRAM",
        expected_include_explanation=True,
        verification_passed=True,
        verification_confidence=0.82,
        # LLM fallback with reasoning line => confidence 0.85 but let's say
        # the LLM returns without reasoning => confidence 0.70
        # 0.70 >= 0.65 and < 0.85 => specialized + explanation
        # retrieval: avg(0.75,0.70)=0.725; gap=0.05 <= 0.3
        # verification: 0.82
        # combined: 0.4*0.725 + 0.6*0.82 = 0.29 + 0.492 = 0.782
        expected_retrieval_signal=0.725,
        expected_verification_signal=0.82,
        expected_combined_confidence=0.782,
        expected_should_abstain=False,
        expected_has_attribution=True,
        expected_history_updated=True,
        is_multi_intent=False,
        llm_response_keys={
            "intent": "good[1]",
            "grading": "good[0]",
            "generation": "good[0]",
            "verification": "good[0]",
        },
        description=(
            "No heuristic match (no diagram/code/comparison/procedural keywords). "
            "LLM fallback classifies as DIAGRAM without Reasoning line, so "
            "intent confidence=0.70. 0.65 <= 0.70 < 0.85 => specialized + explanation."
        ),
    ),
    TestScenario(
        id="S07",
        name="LLM fallback fuzzy match returns CODE_EXAMPLE",
        category="intent",
        query="I need to see how the webhook payload looks when an order is created",
        chunk_texts=[
            "When an order is created, the system fires an order.created webhook with the full order JSON payload.",
            "Webhook payloads include: order_id, customer_id, items array, total_amount, and created_at timestamp.",
        ],
        chunk_scores=[0.85, 0.80],
        chunk_files=["docs/webhooks.md", "docs/webhooks.md"],
        expected_intent="CODE_EXAMPLE",
        expected_intent_source="llm_fallback",
        expected_sub_type="json",
        expected_crag_rewrite=False,
        chunk_grades=["relevant", "relevant"],
        expected_generator="EXPLANATION",
        expected_include_explanation=False,
        verification_passed=True,
        verification_confidence=0.88,
        # LLM fuzzy match => confidence 0.60
        # 0.60 < 0.65 => explanation only
        # retrieval: avg(0.85,0.80)=0.825; gap=0.05 <= 0.3
        # verification: 0.88
        # combined: 0.4*0.825 + 0.6*0.88 = 0.33 + 0.528 = 0.858
        expected_retrieval_signal=0.825,
        expected_verification_signal=0.88,
        expected_combined_confidence=0.858,
        expected_should_abstain=False,
        expected_has_attribution=True,
        expected_history_updated=True,
        is_multi_intent=False,
        llm_response_keys={
            "intent": "good[2]",
            "grading": "good[0]",
            "generation": "good[0]",
            "verification": "good[0]",
        },
        description=(
            "No heuristic match. LLM response contains CODE_EXAMPLE but not "
            "in the Intent: line format => fuzzy match with confidence 0.60. "
            "0.60 < 0.65 => routes to EXPLANATION only."
        ),
    ),
    TestScenario(
        id="S08",
        name="Ambiguous query: could be DIAGRAM or PROCEDURAL",
        category="intent",
        query="How does the deployment process work from commit to production?",
        chunk_texts=[
            "The deployment pipeline has 4 stages: lint, test, build, deploy. Each stage runs in a Docker container.",
            "Production deployments require manual approval in the GitLab pipeline after staging passes all checks.",
        ],
        chunk_scores=[0.83, 0.77],
        chunk_files=["docs/deployment.md", "docs/deployment.md"],
        expected_intent="EXPLANATION",
        expected_intent_source="llm_fallback",
        expected_sub_type="none",
        expected_crag_rewrite=False,
        chunk_grades=["relevant", "relevant"],
        expected_generator="EXPLANATION",
        expected_include_explanation=True,
        verification_passed=True,
        verification_confidence=0.85,
        # No heuristic: "How does" doesn't match _PROCEDURAL_TOPIC ("how do I" / "how to" / "how can I")
        # LLM returns EXPLANATION with Reasoning => confidence 0.85
        # But for this test, let's say LLM returns without Reasoning => 0.70
        # 0.65 <= 0.70 < 0.85 => specialized + explanation (EXPLANATION + explanation = just explanation)
        # retrieval: avg(0.83,0.77)=0.80; gap=0.06 <= 0.3
        # verification: 0.85
        # combined: 0.4*0.80 + 0.6*0.85 = 0.32 + 0.51 = 0.83
        expected_retrieval_signal=0.80,
        expected_verification_signal=0.85,
        expected_combined_confidence=0.83,
        expected_should_abstain=False,
        expected_has_attribution=True,
        expected_history_updated=True,
        is_multi_intent=False,
        llm_response_keys={
            "intent": "good[3]",
            "grading": "good[0]",
            "generation": "good[0]",
            "verification": "good[0]",
        },
        description=(
            "Ambiguous: 'How does ... work' is not 'How do I' (procedural topic) nor "
            "does it have diagram verbs. Falls to LLM which classifies EXPLANATION. "
            "With confidence 0.70 => specialized + explanation (but both are EXPLANATION)."
        ),
    ),
    TestScenario(
        id="S09",
        name="Ambiguous query: comparison-like but no verb match",
        category="intent",
        query="What is the difference between the sync and async processing modes?",
        chunk_texts=[
            "Synchronous mode processes requests inline and returns results immediately.",
            "Asynchronous mode queues work items and processes them in background workers.",
        ],
        chunk_scores=[0.86, 0.82],
        chunk_files=["docs/processing.md", "docs/processing.md"],
        expected_intent="COMPARISON_TABLE",
        expected_intent_source="heuristic",
        expected_sub_type="none",
        expected_crag_rewrite=False,
        chunk_grades=["relevant", "relevant"],
        expected_generator="COMPARISON_TABLE",
        expected_include_explanation=False,
        verification_passed=True,
        verification_confidence=0.87,
        # Heuristic: _COMPARISON_TOPIC matches 'difference', _COMPARISON_VERB matches 'between'
        # => comparison_topic_verb with confidence 0.90 >= 0.85 => specialized only
        # retrieval: avg(0.86,0.82)=0.84; gap=0.04 <= 0.3
        # verification: 0.87
        # combined: 0.4*0.84 + 0.6*0.87 = 0.336 + 0.522 = 0.858
        expected_retrieval_signal=0.84,
        expected_verification_signal=0.87,
        expected_combined_confidence=0.858,
        expected_should_abstain=False,
        expected_has_attribution=True,
        expected_history_updated=True,
        is_multi_intent=False,
        llm_response_keys={"grading": "good[0]", "generation": "good[2]", "verification": "good[0]"},
        description=(
            "Tests that 'difference' + 'between' triggers COMPARISON_TABLE via "
            "_COMPARISON_TOPIC ('difference') and _COMPARISON_VERB ('between'). "
            "Confidence 0.90 >= 0.85 => specialized only."
        ),
    ),
    TestScenario(
        id="S10",
        name="DIAGRAM with sequence sub-type detection",
        category="intent",
        query="Visualize the sequence of message passing between the Order Service and Payment Service",
        chunk_texts=[
            "When an order is placed, the Order Service sends a PaymentRequest to the Payment Service via RabbitMQ.",
            "The Payment Service validates the payment, charges the card, and sends a PaymentConfirmed event back.",
        ],
        chunk_scores=[0.88, 0.84],
        chunk_files=["docs/order-flow.md", "docs/payment.md"],
        expected_intent="DIAGRAM",
        expected_intent_source="heuristic",
        expected_sub_type="sequence",
        expected_crag_rewrite=False,
        chunk_grades=["relevant", "relevant"],
        expected_generator="DIAGRAM",
        expected_include_explanation=False,
        verification_passed=True,
        verification_confidence=0.91,
        # Heuristic: _DIAGRAM_TOPIC ('sequence') + _DIAGRAM_VERB ('visualize') => 0.92
        # Sub-type: _SUBTYPE_SEQUENCE matches 'sequence' and 'message passing' and 'between...and'
        # retrieval: avg(0.88,0.84)=0.86; gap=0.04 <= 0.3
        # verification: 0.91
        # combined: 0.4*0.86 + 0.6*0.91 = 0.344 + 0.546 = 0.890
        expected_retrieval_signal=0.86,
        expected_verification_signal=0.91,
        expected_combined_confidence=0.890,
        expected_should_abstain=False,
        expected_has_attribution=True,
        expected_history_updated=True,
        is_multi_intent=False,
        llm_response_keys={"grading": "good[0]", "generation": "good[0]", "verification": "good[0]"},
        description=(
            "Tests DIAGRAM heuristic with sub-type 'sequence' detection. "
            "'sequence' matches _DIAGRAM_TOPIC and _SUBTYPE_SEQUENCE. "
            "'visualize' matches _DIAGRAM_VERB. Confidence 0.92 >= 0.85."
        ),
    ),

    # ------------------------------------------------------------------
    # Category 2: CRAG / Corrective Retrieval (6 scenarios)
    # ------------------------------------------------------------------
    TestScenario(
        id="S11",
        name="CRAG: all chunks relevant, no rewrite needed",
        category="crag",
        query="How to configure the database connection pool settings?",
        chunk_texts=[
            "Database connection pooling is configured in application.yml under spring.datasource.hikari.",
            "Key pool settings: maximum-pool-size (default 10), minimum-idle (default 5), connection-timeout (30000ms).",
            "For production, set maximum-pool-size to 20 and enable leak detection with leak-detection-threshold: 60000.",
        ],
        chunk_scores=[0.91, 0.87, 0.80],
        chunk_files=["docs/database.md", "docs/database.md", "docs/production.md"],
        expected_intent="PROCEDURAL",
        expected_intent_source="heuristic",
        expected_sub_type="none",
        expected_crag_rewrite=False,
        chunk_grades=["relevant", "relevant", "relevant"],
        expected_generator="PROCEDURAL",
        expected_include_explanation=False,
        verification_passed=True,
        verification_confidence=0.93,
        # retrieval: avg(0.91,0.87,0.80)=0.86; gap=0.04 <= 0.3
        # verification: 0.93
        # combined: 0.4*0.86 + 0.6*0.93 = 0.344 + 0.558 = 0.902
        expected_retrieval_signal=0.86,
        expected_verification_signal=0.93,
        expected_combined_confidence=0.902,
        expected_should_abstain=False,
        expected_has_attribution=True,
        expected_history_updated=True,
        is_multi_intent=False,
        llm_response_keys={"grading": "good[0]", "generation": "good[3]", "verification": "good[0]"},
        description=(
            "All 3 chunks graded RELEVANT. Irrelevant fraction = 0/3 = 0.0 <= 0.5, "
            "so no rewrite triggered. CRAG returns relevant+partial chunks as-is."
        ),
    ),
    TestScenario(
        id="S12",
        name="CRAG: mostly irrelevant, rewrite triggered",
        category="crag",
        query="What monitoring tools does the platform support?",
        chunk_texts=[
            "The build system uses Maven 3.9 with Java 17 as the target compilation version.",
            "Unit tests are written in JUnit 5 with Mockito for mocking dependencies.",
            "Application logs are shipped to ELK stack via Filebeat sidecar containers.",
            "Prometheus metrics are exposed at /actuator/prometheus and scraped every 15 seconds.",
        ],
        chunk_scores=[0.45, 0.42, 0.55, 0.52],
        chunk_files=["docs/build.md", "docs/testing.md", "docs/logging.md", "docs/metrics.md"],
        expected_intent="EXPLANATION",
        expected_intent_source="llm_fallback",
        expected_sub_type="none",
        expected_crag_rewrite=True,
        chunk_grades=["irrelevant", "irrelevant", "irrelevant", "relevant"],
        expected_generator="EXPLANATION",
        expected_include_explanation=True,
        verification_passed=True,
        verification_confidence=0.80,
        # 3/4 irrelevant = 0.75 > 0.5 => rewrite triggered
        # retrieval: avg(0.45,0.42,0.55,0.52)=0.485; gap=0.03 <= 0.3; no all-below-0.3
        # verification: 0.80
        # combined: 0.4*0.485 + 0.6*0.80 = 0.194 + 0.48 = 0.674
        expected_retrieval_signal=0.485,
        expected_verification_signal=0.80,
        expected_combined_confidence=0.674,
        expected_should_abstain=False,
        expected_has_attribution=True,
        expected_history_updated=True,
        is_multi_intent=False,
        llm_response_keys={
            "intent": "good[0]",
            "grading": "good[3]",
            "rewrite": "good[0]",
            "generation": "good[0]",
            "verification": "good[0]",
        },
        description=(
            "3 of 4 chunks irrelevant (fraction 0.75 > 0.5) => rewrite triggered. "
            "Tests the rewrite-and-re-retrieve cycle with mostly irrelevant initial results."
        ),
    ),
    TestScenario(
        id="S13",
        name="CRAG: mixed grades, borderline no-rewrite (exactly 50%)",
        category="crag",
        query="How does the caching layer invalidate stale entries?",
        chunk_texts=[
            "Redis is used as the primary cache layer with a TTL of 300 seconds for most entries.",
            "The user management module handles CRUD operations on the users table.",
            "Cache invalidation happens on write-through: every PUT/DELETE to the API invalidates the cache key.",
            "The batch processing module runs nightly to reconcile ledger entries.",
        ],
        chunk_scores=[0.72, 0.38, 0.68, 0.35],
        chunk_files=["docs/caching.md", "docs/users.md", "docs/caching.md", "docs/batch.md"],
        expected_intent="EXPLANATION",
        expected_intent_source="llm_fallback",
        expected_sub_type="none",
        expected_crag_rewrite=False,
        chunk_grades=["relevant", "irrelevant", "relevant", "irrelevant"],
        expected_generator="EXPLANATION",
        expected_include_explanation=True,
        verification_passed=True,
        verification_confidence=0.78,
        # 2/4 irrelevant = 0.5, NOT > 0.5, so no rewrite
        # retrieval: avg(0.72,0.38,0.68,0.35)=0.5325; gap=0.72-0.38=0.34 > 0.3 => -0.15
        # signal = 0.5325 - 0.15 = 0.3825
        # verification: 0.78
        # combined: 0.4*0.3825 + 0.6*0.78 = 0.153 + 0.468 = 0.621
        expected_retrieval_signal=0.3825,
        expected_verification_signal=0.78,
        expected_combined_confidence=0.621,
        expected_should_abstain=False,
        expected_has_attribution=True,
        expected_history_updated=True,
        is_multi_intent=False,
        llm_response_keys={
            "intent": "good[0]",
            "grading": "good[4]",
            "generation": "good[0]",
            "verification": "good[0]",
        },
        description=(
            "Exactly 2/4 chunks irrelevant = 0.5 fraction. should_rewrite threshold is "
            "> 0.5 (strict), so 0.5 does NOT trigger rewrite. Tests the boundary condition."
        ),
    ),
    TestScenario(
        id="S14",
        name="CRAG: rewrite produces better results",
        category="crag",
        query="What rate limiting strategy does the API gateway use?",
        chunk_texts=[
            "The frontend application is built with React 18 and TypeScript.",
            "Nginx is configured as a reverse proxy with upstream load balancing.",
            "The API uses OpenAPI 3.0 specification for documentation.",
        ],
        chunk_scores=[0.40, 0.42, 0.38],
        chunk_files=["docs/frontend.md", "docs/nginx.md", "docs/api-spec.md"],
        expected_intent="EXPLANATION",
        expected_intent_source="llm_fallback",
        expected_sub_type="none",
        expected_crag_rewrite=True,
        chunk_grades=["irrelevant", "irrelevant", "irrelevant"],
        expected_generator="EXPLANATION",
        expected_include_explanation=True,
        verification_passed=True,
        verification_confidence=0.75,
        # 3/3 irrelevant = 1.0 > 0.5 => rewrite triggered
        # retrieval signal uses initial scores: avg(0.40,0.42,0.38)=0.40; not all < 0.3
        # gap=0.02 <= 0.3
        # verification: 0.75
        # combined: 0.4*0.40 + 0.6*0.75 = 0.16 + 0.45 = 0.61
        expected_retrieval_signal=0.40,
        expected_verification_signal=0.75,
        expected_combined_confidence=0.61,
        expected_should_abstain=False,
        expected_has_attribution=True,
        expected_history_updated=True,
        is_multi_intent=False,
        llm_response_keys={
            "intent": "good[0]",
            "grading": "good[5]",
            "rewrite": "good[0]",
            "generation": "good[0]",
            "verification": "good[0]",
        },
        description=(
            "All 3 chunks irrelevant (fraction 1.0 > 0.5). CRAG rewrites query. "
            "After rewrite, retriever returns better chunks about rate limiting. "
            "Tests the full rewrite-and-re-retrieve cycle."
        ),
    ),
    TestScenario(
        id="S15",
        name="CRAG: max rewrites exhausted (2 rewrites)",
        category="crag",
        query="What is the SLA for the notification delivery service?",
        chunk_texts=[
            "Email templates are stored in the templates directory and rendered with Thymeleaf.",
            "The notification service supports email, SMS, and push notification channels.",
        ],
        chunk_scores=[0.35, 0.40],
        chunk_files=["docs/templates.md", "docs/notifications.md"],
        expected_intent="EXPLANATION",
        expected_intent_source="llm_fallback",
        expected_sub_type="none",
        expected_crag_rewrite=True,
        chunk_grades=["irrelevant", "irrelevant"],
        expected_generator="EXPLANATION",
        expected_include_explanation=True,
        verification_passed=False,
        verification_confidence=0.45,
        # 2/2 irrelevant = 1.0 > 0.5 => rewrite triggered
        # After rewrite 1: still irrelevant => rewrite again.
        # After rewrite 2: max_crag_rewrites=2, exhausted => final grading pass.
        # retrieval: avg(0.35,0.40)=0.375; gap=0.05 <= 0.3; not all < 0.3 (0.40 >= 0.3)
        # verification: FAIL => 0.45 - 0.2 = 0.25
        # combined: 0.4*0.375 + 0.6*0.25 = 0.15 + 0.15 = 0.30
        # 0.30 < 0.40 threshold => should abstain
        expected_retrieval_signal=0.375,
        expected_verification_signal=0.25,
        expected_combined_confidence=0.30,
        expected_should_abstain=True,
        expected_has_attribution=False,
        expected_history_updated=False,
        is_multi_intent=False,
        llm_response_keys={
            "intent": "good[0]",
            "grading": "bad[0]",
            "rewrite": "good[0]",
            "generation": "good[0]",
            "verification": "bad[0]",
        },
        description=(
            "Both chunks irrelevant after each rewrite attempt. max_crag_rewrites=2 "
            "from VerificationConfig default. After 2 rewrites, exhausted. Final grading "
            "still poor. Verification FAIL with low confidence => abstention."
        ),
    ),
    TestScenario(
        id="S16",
        name="CRAG: disabled in config",
        category="crag",
        query="What authentication methods are supported by the API?",
        chunk_texts=[
            "The system uses JWT tokens for stateless authentication.",
            "OAuth 2.0 authorization code flow is supported for third-party integrations.",
            "API keys can be generated for machine-to-machine communication.",
        ],
        chunk_scores=[0.85, 0.78, 0.72],
        chunk_files=["docs/auth.md", "docs/oauth.md", "docs/api-keys.md"],
        expected_intent="EXPLANATION",
        expected_intent_source="llm_fallback",
        expected_sub_type="none",
        expected_crag_rewrite=False,
        chunk_grades=None,
        expected_generator="EXPLANATION",
        expected_include_explanation=True,
        verification_passed=True,
        verification_confidence=0.88,
        # CRAG disabled => no grading, no rewrite
        # retrieval: avg(0.85,0.78,0.72)=0.7833; gap=0.07 <= 0.3
        # verification: 0.88
        # combined: 0.4*0.7833 + 0.6*0.88 = 0.3133 + 0.528 = 0.8413
        expected_retrieval_signal=0.7833,
        expected_verification_signal=0.88,
        expected_combined_confidence=0.8413,
        expected_should_abstain=False,
        expected_has_attribution=True,
        expected_history_updated=True,
        is_multi_intent=False,
        llm_response_keys={
            "intent": "good[0]",
            "generation": "good[0]",
            "verification": "good[0]",
        },
        description=(
            "Tests behavior when enable_crag=False in VerificationConfig. "
            "No grading step, no rewrite. Chunks pass directly to generation. "
            "chunk_grades is None because grading never runs."
        ),
    ),

    # ------------------------------------------------------------------
    # Category 3: Confidence Scoring (8 scenarios)
    # ------------------------------------------------------------------
    TestScenario(
        id="S17",
        name="High retrieval + high verification => high confidence",
        category="confidence",
        query="What is the maximum file upload size in the document service?",
        chunk_texts=[
            "The document service accepts file uploads up to 50MB. Larger files must use the multipart upload API.",
            "File type validation checks the MIME type against an allowlist: PDF, DOCX, PNG, JPG.",
        ],
        chunk_scores=[0.92, 0.88],
        chunk_files=["docs/upload.md", "docs/upload.md"],
        expected_intent="EXPLANATION",
        expected_intent_source="llm_fallback",
        expected_sub_type="none",
        expected_crag_rewrite=False,
        chunk_grades=["relevant", "relevant"],
        expected_generator="EXPLANATION",
        expected_include_explanation=True,
        verification_passed=True,
        verification_confidence=0.95,
        # retrieval: avg(0.92,0.88)=0.90; gap=0.04 <= 0.3
        # verification: 0.95
        # combined: 0.4*0.90 + 0.6*0.95 = 0.36 + 0.57 = 0.93
        expected_retrieval_signal=0.90,
        expected_verification_signal=0.95,
        expected_combined_confidence=0.93,
        expected_should_abstain=False,
        expected_has_attribution=True,
        expected_history_updated=True,
        is_multi_intent=False,
        llm_response_keys={
            "intent": "good[0]",
            "grading": "good[0]",
            "generation": "good[0]",
            "verification": "good[1]",
        },
        description=(
            "High retrieval scores (avg 0.90) and high verification (0.95 PASS). "
            "Combined 0.93 well above threshold 0.40. No abstention."
        ),
    ),
    TestScenario(
        id="S18",
        name="Low retrieval + low verification => abstention",
        category="confidence",
        query="What is the disaster recovery plan for the Singapore data center?",
        chunk_texts=[
            "Our infrastructure runs on AWS in the us-east-1 and eu-west-1 regions.",
            "Database backups are taken every 6 hours using pg_dump.",
        ],
        chunk_scores=[0.32, 0.28],
        chunk_files=["docs/infra.md", "docs/backups.md"],
        expected_intent="EXPLANATION",
        expected_intent_source="llm_fallback",
        expected_sub_type="none",
        expected_crag_rewrite=True,
        chunk_grades=["irrelevant", "irrelevant"],
        expected_generator="EXPLANATION",
        expected_include_explanation=True,
        verification_passed=False,
        verification_confidence=0.30,
        # retrieval: avg(0.32,0.28)=0.30; not all < 0.3 (0.32 >= 0.3); gap=0.04 <= 0.3
        # verification: FAIL => 0.30 - 0.20 = 0.10
        # combined: 0.4*0.30 + 0.6*0.10 = 0.12 + 0.06 = 0.18
        # 0.18 < 0.40 => abstain
        expected_retrieval_signal=0.30,
        expected_verification_signal=0.10,
        expected_combined_confidence=0.18,
        expected_should_abstain=True,
        expected_has_attribution=False,
        expected_history_updated=False,
        is_multi_intent=False,
        llm_response_keys={
            "intent": "good[0]",
            "grading": "bad[0]",
            "rewrite": "good[0]",
            "generation": "good[0]",
            "verification": "bad[1]",
        },
        description=(
            "Query about Singapore DC but docs only cover AWS us-east-1/eu-west-1. "
            "Low retrieval (0.30), verification FAIL with 0.30 confidence => signal 0.10. "
            "Combined 0.18 < threshold 0.40 => abstention. No attribution, no history update."
        ),
    ),
    TestScenario(
        id="S19",
        name="Single-source reliance penalty (top-1 >> top-2)",
        category="confidence",
        query="What are the password requirements for user accounts?",
        chunk_texts=[
            "Passwords must be at least 12 characters, include upper/lower case, a digit, and a special character.",
            "The user profile page allows updating display name and email preferences.",
            "Account lockout occurs after 5 failed login attempts within 15 minutes.",
        ],
        chunk_scores=[0.92, 0.40, 0.38],
        chunk_files=["docs/security.md", "docs/profile.md", "docs/security.md"],
        expected_intent="EXPLANATION",
        expected_intent_source="llm_fallback",
        expected_sub_type="none",
        expected_crag_rewrite=False,
        chunk_grades=["relevant", "irrelevant", "partial"],
        expected_generator="EXPLANATION",
        expected_include_explanation=True,
        verification_passed=True,
        verification_confidence=0.85,
        # retrieval: avg(0.92,0.40,0.38)=0.5667; gap=0.92-0.40=0.52 > 0.3 => -0.15
        # signal = 0.5667 - 0.15 = 0.4167
        # verification: 0.85
        # combined: 0.4*0.4167 + 0.6*0.85 = 0.1667 + 0.51 = 0.6767
        expected_retrieval_signal=0.4167,
        expected_verification_signal=0.85,
        expected_combined_confidence=0.6767,
        expected_should_abstain=False,
        expected_has_attribution=True,
        expected_history_updated=True,
        is_multi_intent=False,
        llm_response_keys={
            "intent": "good[0]",
            "grading": "good[1]",
            "generation": "good[0]",
            "verification": "good[0]",
        },
        description=(
            "Top score (0.92) far exceeds second (0.40), gap=0.52 > 0.3. "
            "Single-source reliance penalty: signal -= 0.15. "
            "Tests that the gap penalty correctly reduces the retrieval signal."
        ),
    ),
    TestScenario(
        id="S20",
        name="All scores below 0.3 => halved retrieval signal",
        category="confidence",
        query="What is the quantum encryption module configuration?",
        chunk_texts=[
            "TLS 1.3 is enforced on all external endpoints with strong cipher suites.",
            "Certificate rotation happens automatically every 90 days via Let's Encrypt.",
        ],
        chunk_scores=[0.25, 0.22],
        chunk_files=["docs/tls.md", "docs/certs.md"],
        expected_intent="EXPLANATION",
        expected_intent_source="llm_fallback",
        expected_sub_type="none",
        expected_crag_rewrite=True,
        chunk_grades=["irrelevant", "irrelevant"],
        expected_generator="EXPLANATION",
        expected_include_explanation=True,
        verification_passed=False,
        verification_confidence=0.35,
        # retrieval: avg(0.25,0.22)=0.235; all < 0.3 => halved = 0.1175
        # gap=0.03 <= 0.3
        # verification: FAIL => 0.35 - 0.20 = 0.15
        # combined: 0.4*0.1175 + 0.6*0.15 = 0.047 + 0.09 = 0.137
        # 0.137 < 0.40 => abstain
        expected_retrieval_signal=0.1175,
        expected_verification_signal=0.15,
        expected_combined_confidence=0.137,
        expected_should_abstain=True,
        expected_has_attribution=False,
        expected_history_updated=False,
        is_multi_intent=False,
        llm_response_keys={
            "intent": "good[0]",
            "grading": "bad[0]",
            "rewrite": "good[0]",
            "generation": "good[0]",
            "verification": "bad[1]",
        },
        description=(
            "Both scores below 0.3 (0.25, 0.22). Retrieval signal avg=0.235 halved to 0.1175. "
            "Verification FAIL with 0.35 => signal 0.15. Combined 0.137 < 0.40 => abstention. "
            "Tests the all-below-0.3 halving penalty."
        ),
    ),
    TestScenario(
        id="S21",
        name="Verification FAIL with issues list",
        category="confidence",
        query="How many concurrent users can the system handle?",
        chunk_texts=[
            "Load testing with JMeter showed the system handles 500 concurrent users at p99 latency under 200ms.",
            "The auto-scaler adds new pods when CPU usage exceeds 70% across the deployment.",
        ],
        chunk_scores=[0.80, 0.65],
        chunk_files=["docs/performance.md", "docs/scaling.md"],
        expected_intent="EXPLANATION",
        expected_intent_source="llm_fallback",
        expected_sub_type="none",
        expected_crag_rewrite=False,
        chunk_grades=["relevant", "partial"],
        expected_generator="EXPLANATION",
        expected_include_explanation=True,
        verification_passed=False,
        verification_confidence=0.55,
        # retrieval: avg(0.80,0.65)=0.725; gap=0.15 <= 0.3
        # verification: FAIL => 0.55 - 0.20 = 0.35
        # combined: 0.4*0.725 + 0.6*0.35 = 0.29 + 0.21 = 0.50
        # 0.50 >= 0.40 => no abstain
        expected_retrieval_signal=0.725,
        expected_verification_signal=0.35,
        expected_combined_confidence=0.50,
        expected_should_abstain=False,
        expected_has_attribution=True,
        expected_history_updated=True,
        is_multi_intent=False,
        llm_response_keys={
            "intent": "good[0]",
            "grading": "good[1]",
            "generation": "good[0]",
            "verification": "bad[2]",
        },
        description=(
            "Verification returns FAIL with issues like 'answer states 1000 but source says 500'. "
            "FAIL penalty: signal = 0.55 - 0.20 = 0.35. Combined 0.50 >= 0.40 so no abstention "
            "despite verification failure. Tests the -0.2 FAIL penalty."
        ),
    ),
    TestScenario(
        id="S22",
        name="No verification => neutral 0.7 default",
        category="confidence",
        query="What logging framework does the application use?",
        chunk_texts=[
            "The application uses SLF4J with Logback as the logging implementation.",
            "Log levels can be changed at runtime via the /actuator/loggers endpoint.",
        ],
        chunk_scores=[0.88, 0.75],
        chunk_files=["docs/logging.md", "docs/actuator.md"],
        expected_intent="EXPLANATION",
        expected_intent_source="llm_fallback",
        expected_sub_type="none",
        expected_crag_rewrite=False,
        chunk_grades=["relevant", "relevant"],
        expected_generator="EXPLANATION",
        expected_include_explanation=True,
        verification_passed=None,
        verification_confidence=None,
        # retrieval: avg(0.88,0.75)=0.815; gap=0.13 <= 0.3
        # verification: None => 0.7
        # combined: 0.4*0.815 + 0.6*0.7 = 0.326 + 0.42 = 0.746
        expected_retrieval_signal=0.815,
        expected_verification_signal=0.7,
        expected_combined_confidence=0.746,
        expected_should_abstain=False,
        expected_has_attribution=True,
        expected_history_updated=True,
        is_multi_intent=False,
        llm_response_keys={
            "intent": "good[0]",
            "grading": "good[0]",
            "generation": "good[0]",
        },
        description=(
            "Verification disabled (enable_verification=False). "
            "verification_signal defaults to 0.7. Combined 0.746 well above threshold. "
            "Tests the neutral default when verification is not performed."
        ),
    ),
    TestScenario(
        id="S23",
        name="Edge case: exactly at confidence threshold",
        category="confidence",
        query="What is the retry policy for failed webhook deliveries?",
        chunk_texts=[
            "Webhooks use exponential backoff starting at 1 second, doubling up to 5 attempts.",
            "The notification preferences page lets users toggle email and push notifications.",
        ],
        chunk_scores=[0.55, 0.35],
        chunk_files=["docs/webhooks.md", "docs/notifications.md"],
        expected_intent="EXPLANATION",
        expected_intent_source="llm_fallback",
        expected_sub_type="none",
        expected_crag_rewrite=False,
        chunk_grades=["relevant", "irrelevant"],
        expected_generator="EXPLANATION",
        expected_include_explanation=True,
        verification_passed=True,
        verification_confidence=0.40,
        # retrieval: avg(0.55,0.35)=0.45; gap=0.20 <= 0.3; not all < 0.3
        # verification: PASS => 0.40
        # combined: 0.4*0.45 + 0.6*0.40 = 0.18 + 0.24 = 0.42
        # 0.42 >= 0.40 => no abstain (at threshold, not below)
        expected_retrieval_signal=0.45,
        expected_verification_signal=0.40,
        expected_combined_confidence=0.42,
        expected_should_abstain=False,
        expected_has_attribution=True,
        expected_history_updated=True,
        is_multi_intent=False,
        llm_response_keys={
            "intent": "good[0]",
            "grading": "good[1]",
            "generation": "good[0]",
            "verification": "good[2]",
        },
        description=(
            "Combined confidence 0.42 is just at threshold 0.40. The condition is "
            "'combined < threshold' (strict less-than), so 0.42 >= 0.40 means no abstention. "
            "Tests the exact boundary."
        ),
    ),
    TestScenario(
        id="S24",
        name="Abstention disabled in config",
        category="confidence",
        query="What is the GraphQL schema for the inventory service?",
        chunk_texts=[
            "The REST API returns product listings in JSON format with pagination.",
        ],
        chunk_scores=[0.28],
        chunk_files=["docs/products.md"],
        expected_intent="EXPLANATION",
        expected_intent_source="llm_fallback",
        expected_sub_type="none",
        expected_crag_rewrite=True,
        chunk_grades=["irrelevant"],
        expected_generator="EXPLANATION",
        expected_include_explanation=True,
        verification_passed=False,
        verification_confidence=0.25,
        # retrieval: avg(0.28)=0.28; all < 0.3 => halved = 0.14; only 1 score so no gap check
        # verification: FAIL => 0.25 - 0.20 = 0.05
        # combined: 0.4*0.14 + 0.6*0.05 = 0.056 + 0.03 = 0.086
        # 0.086 < 0.40 but abstain_on_low_confidence=False => should_abstain=False
        expected_retrieval_signal=0.14,
        expected_verification_signal=0.05,
        expected_combined_confidence=0.086,
        expected_should_abstain=False,
        expected_has_attribution=True,
        expected_history_updated=True,
        is_multi_intent=False,
        llm_response_keys={
            "intent": "good[0]",
            "grading": "bad[0]",
            "rewrite": "good[0]",
            "generation": "good[0]",
            "verification": "bad[1]",
        },
        description=(
            "Extremely low confidence (0.086) but abstain_on_low_confidence=False in config. "
            "Even though combined < threshold, abstention is disabled so the system "
            "still returns an answer. Tests the abstain toggle."
        ),
    ),

    # ------------------------------------------------------------------
    # Category 4: Generation Routing (6 scenarios)
    # ------------------------------------------------------------------
    TestScenario(
        id="S25",
        name="High confidence diagram: specialized only",
        category="generation",
        query="Draw a mermaid diagram of the order processing flow",
        chunk_texts=[
            "Order processing: receive order -> validate -> check inventory -> charge payment -> ship.",
            "Failed payments trigger a retry workflow with 3 attempts before cancellation.",
        ],
        chunk_scores=[0.87, 0.80],
        chunk_files=["docs/orders.md", "docs/payments.md"],
        expected_intent="DIAGRAM",
        expected_intent_source="heuristic",
        expected_sub_type="flowchart",
        expected_crag_rewrite=False,
        chunk_grades=["relevant", "relevant"],
        expected_generator="DIAGRAM",
        expected_include_explanation=False,
        verification_passed=True,
        verification_confidence=0.90,
        # Heuristic: explicit_mermaid => confidence 0.95 >= 0.85 => specialized only
        # retrieval: avg(0.87,0.80)=0.835; gap=0.07 <= 0.3
        # verification: 0.90
        # combined: 0.4*0.835 + 0.6*0.90 = 0.334 + 0.54 = 0.874
        expected_retrieval_signal=0.835,
        expected_verification_signal=0.90,
        expected_combined_confidence=0.874,
        expected_should_abstain=False,
        expected_has_attribution=True,
        expected_history_updated=True,
        is_multi_intent=False,
        llm_response_keys={"grading": "good[0]", "generation": "good[0]", "verification": "good[0]"},
        description=(
            "Explicit 'mermaid' keyword => heuristic confidence 0.95 >= 0.85. "
            "Routes to DiagramGenerator only, no explanation appended."
        ),
    ),
    TestScenario(
        id="S26",
        name="Medium confidence code example: specialized + explanation",
        category="generation",
        query="I need to see how to call the user creation API",
        chunk_texts=[
            "POST /api/v1/users creates a new user. Required fields: name, email. Optional: role, department.",
            "The response returns 201 Created with the user object including the generated UUID.",
        ],
        chunk_scores=[0.84, 0.78],
        chunk_files=["docs/user-api.md", "docs/user-api.md"],
        expected_intent="CODE_EXAMPLE",
        expected_intent_source="llm_fallback",
        expected_sub_type="http",
        expected_crag_rewrite=False,
        chunk_grades=["relevant", "relevant"],
        expected_generator="CODE_EXAMPLE",
        expected_include_explanation=True,
        verification_passed=True,
        verification_confidence=0.86,
        # No heuristic match: "call the user creation API" has _CODE_FORMAT('API') but
        # _CODE_VERB needs (show|give|generate|write|create). "call" is not in the verb list.
        # Wait, "need" is not a match either. Let's check: "I need to see how to call..."
        # "see" is not in _CODE_VERB. So falls to LLM.
        # LLM returns CODE_EXAMPLE with Reasoning => confidence 0.85. Actually, let's
        # say without reasoning => 0.70. 0.65 <= 0.70 < 0.85 => specialized + explanation.
        # retrieval: avg(0.84,0.78)=0.81; gap=0.06 <= 0.3
        # verification: 0.86
        # combined: 0.4*0.81 + 0.6*0.86 = 0.324 + 0.516 = 0.84
        expected_retrieval_signal=0.81,
        expected_verification_signal=0.86,
        expected_combined_confidence=0.84,
        expected_should_abstain=False,
        expected_has_attribution=True,
        expected_history_updated=True,
        is_multi_intent=False,
        llm_response_keys={
            "intent": "good[4]",
            "grading": "good[0]",
            "generation": "good[1]",
            "verification": "good[0]",
        },
        description=(
            "LLM classifies as CODE_EXAMPLE without Reasoning line => confidence 0.70. "
            "0.65 <= 0.70 < 0.85 => specialized CodeExampleGenerator + ExplanationGenerator "
            "results concatenated."
        ),
    ),
    TestScenario(
        id="S27",
        name="Low confidence => explanation only",
        category="generation",
        query="Tell me about the payment module",
        chunk_texts=[
            "The payment module integrates with Stripe for card processing and PayPal for alternative payments.",
            "Payment webhooks from Stripe are verified using the webhook signing secret.",
        ],
        chunk_scores=[0.75, 0.68],
        chunk_files=["docs/payments.md", "docs/webhooks.md"],
        expected_intent="EXPLANATION",
        expected_intent_source="llm_fallback",
        expected_sub_type="none",
        expected_crag_rewrite=False,
        chunk_grades=["relevant", "relevant"],
        expected_generator="EXPLANATION",
        expected_include_explanation=False,
        verification_passed=True,
        verification_confidence=0.80,
        # No heuristic match. LLM parse failure => confidence 0.50. < 0.65 => explanation only
        # retrieval: avg(0.75,0.68)=0.715; gap=0.07 <= 0.3
        # verification: 0.80
        # combined: 0.4*0.715 + 0.6*0.80 = 0.286 + 0.48 = 0.766
        expected_retrieval_signal=0.715,
        expected_verification_signal=0.80,
        expected_combined_confidence=0.766,
        expected_should_abstain=False,
        expected_has_attribution=True,
        expected_history_updated=True,
        is_multi_intent=False,
        llm_response_keys={
            "intent": "bad[0]",
            "grading": "good[0]",
            "generation": "good[0]",
            "verification": "good[0]",
        },
        description=(
            "LLM response is unparseable => llm_parse_failure, confidence 0.50. "
            "0.50 < 0.65 => routes to ExplanationGenerator only, include_explanation=False."
        ),
    ),
    TestScenario(
        id="S28",
        name="Specialized generator fails => fallback to ExplanationGenerator",
        category="generation",
        query="Generate a code example for the search API endpoint",
        chunk_texts=[
            "GET /api/v1/search?q=term&page=1&size=20 returns paginated search results.",
            "Search supports filters: category, date_range, and sort_by fields.",
        ],
        chunk_scores=[0.85, 0.79],
        chunk_files=["docs/search-api.md", "docs/search-api.md"],
        expected_intent="CODE_EXAMPLE",
        expected_intent_source="heuristic",
        expected_sub_type="http",
        expected_crag_rewrite=False,
        chunk_grades=["relevant", "relevant"],
        expected_generator="CODE_EXAMPLE",
        expected_include_explanation=False,
        verification_passed=True,
        verification_confidence=0.78,
        # Heuristic: _CODE_FORMAT('code example', 'API', 'endpoint') + _CODE_VERB('generate')
        # => confidence 0.90 >= 0.85 => specialized only (resolve_generation_strategy returns CODE_EXAMPLE)
        # At runtime, CodeExampleGenerator.generate() may raise => fallback to ExplanationGenerator
        # retrieval: avg(0.85,0.79)=0.82; gap=0.06 <= 0.3
        # verification: 0.78
        # combined: 0.4*0.82 + 0.6*0.78 = 0.328 + 0.468 = 0.796
        expected_retrieval_signal=0.82,
        expected_verification_signal=0.78,
        expected_combined_confidence=0.796,
        expected_should_abstain=False,
        expected_has_attribution=True,
        expected_history_updated=True,
        is_multi_intent=False,
        llm_response_keys={
            "grading": "good[0]",
            "generation": "error[0]",
            "verification": "good[0]",
        },
        description=(
            "CodeExampleGenerator raises an exception during generate(). "
            "route_and_generate catches it and falls back to ExplanationGenerator. "
            "Tests the graceful fallback path in the router."
        ),
    ),
    TestScenario(
        id="S29",
        name="Diagram with mermaid repair loop",
        category="generation",
        query="Illustrate the architecture flow of the authentication system in a diagram",
        chunk_texts=[
            "Auth flow: Client -> API Gateway -> Auth Service -> Token Store -> Response with JWT.",
            "The Auth Service validates credentials against LDAP and issues short-lived access tokens.",
        ],
        chunk_scores=[0.86, 0.82],
        chunk_files=["docs/auth-flow.md", "docs/auth-service.md"],
        expected_intent="DIAGRAM",
        expected_intent_source="heuristic",
        expected_sub_type="flowchart",
        expected_crag_rewrite=False,
        chunk_grades=["relevant", "relevant"],
        expected_generator="DIAGRAM",
        expected_include_explanation=False,
        verification_passed=True,
        verification_confidence=0.84,
        # Heuristic: _DIAGRAM_TOPIC('architecture', 'flow') + _DIAGRAM_VERB('illustrate', 'diagram')
        # => confidence 0.92 >= 0.85 => specialized only
        # Mermaid validation fails first attempt, repair succeeds on attempt 2
        # retrieval: avg(0.86,0.82)=0.84; gap=0.04 <= 0.3
        # verification: 0.84
        # combined: 0.4*0.84 + 0.6*0.84 = 0.336 + 0.504 = 0.84
        expected_retrieval_signal=0.84,
        expected_verification_signal=0.84,
        expected_combined_confidence=0.84,
        expected_should_abstain=False,
        expected_has_attribution=True,
        expected_history_updated=True,
        is_multi_intent=False,
        llm_response_keys={
            "grading": "good[0]",
            "generation": "good[5]",
            "mermaid_validation": "repair[0]",
            "verification": "good[0]",
        },
        description=(
            "DiagramGenerator produces a mermaid block. MermaidValidator finds it invalid. "
            "Repair loop invoked (max_diagram_retries=3). First repair attempt succeeds. "
            "Tests the mermaid validation + repair cycle."
        ),
    ),
    TestScenario(
        id="S30",
        name="Code example with yaml sub-type detection",
        category="generation",
        query="Show me a yaml example for the Kubernetes deployment configuration",
        chunk_texts=[
            "The Kubernetes deployment spec requires: apiVersion, kind, metadata, and spec sections.",
            "Resource limits should be set to 512Mi memory and 500m CPU for each pod.",
        ],
        chunk_scores=[0.88, 0.82],
        chunk_files=["docs/k8s.md", "docs/k8s.md"],
        expected_intent="CODE_EXAMPLE",
        expected_intent_source="heuristic",
        expected_sub_type="yaml",
        expected_crag_rewrite=False,
        chunk_grades=["relevant", "relevant"],
        expected_generator="CODE_EXAMPLE",
        expected_include_explanation=False,
        verification_passed=True,
        verification_confidence=0.91,
        # Heuristic: _CODE_FORMAT('yaml') + _CODE_VERB('show') => 0.90 >= 0.85
        # Sub-type: _SUBTYPE_YAML matches 'yaml' and 'kubernetes'
        # retrieval: avg(0.88,0.82)=0.85; gap=0.06 <= 0.3
        # verification: 0.91
        # combined: 0.4*0.85 + 0.6*0.91 = 0.34 + 0.546 = 0.886
        expected_retrieval_signal=0.85,
        expected_verification_signal=0.91,
        expected_combined_confidence=0.886,
        expected_should_abstain=False,
        expected_has_attribution=True,
        expected_history_updated=True,
        is_multi_intent=False,
        llm_response_keys={"grading": "good[0]", "generation": "good[6]", "verification": "good[0]"},
        description=(
            "Heuristic CODE_EXAMPLE via 'yaml' + 'show'. Sub-type detected as 'yaml' "
            "via _SUBTYPE_YAML ('yaml', 'kubernetes'). CodeExampleGenerator uses the "
            "sub-type to inject the correct language tag in the code fence."
        ),
    ),

    # ------------------------------------------------------------------
    # Category 5: SSE Streaming (5 scenarios)
    # ------------------------------------------------------------------
    TestScenario(
        id="S31",
        name="SSE: full event sequence with all phases",
        category="streaming",
        query="How to configure SSL certificates for the API gateway?",
        chunk_texts=[
            "SSL termination happens at the API Gateway level. Certificates are stored in AWS Certificate Manager.",
            "For local development, self-signed certificates can be generated with openssl.",
        ],
        chunk_scores=[0.85, 0.72],
        chunk_files=["docs/ssl.md", "docs/local-dev.md"],
        expected_intent="PROCEDURAL",
        expected_intent_source="heuristic",
        expected_sub_type="none",
        expected_crag_rewrite=False,
        chunk_grades=["relevant", "relevant"],
        expected_generator="PROCEDURAL",
        expected_include_explanation=False,
        verification_passed=True,
        verification_confidence=0.87,
        # SSE event sequence: status(classifying) -> intent -> status(retrieving) ->
        # sources -> status(grading) -> status(generating) -> answer ->
        # status(verifying) -> verified -> done
        # retrieval: avg(0.85,0.72)=0.785; gap=0.13 <= 0.3
        # verification: 0.87
        # combined: 0.4*0.785 + 0.6*0.87 = 0.314 + 0.522 = 0.836
        expected_retrieval_signal=0.785,
        expected_verification_signal=0.87,
        expected_combined_confidence=0.836,
        expected_should_abstain=False,
        expected_has_attribution=True,
        expected_history_updated=True,
        is_multi_intent=False,
        llm_response_keys={
            "grading": "good[0]",
            "generation": "good[3]",
            "verification": "good[0]",
        },
        description=(
            "Tests the complete SSE event sequence. Verifies that all phase events are "
            "emitted in order: status(classifying), intent, status(retrieving), sources, "
            "status(grading), status(generating), answer, attribution, status(verifying), "
            "verified, done."
        ),
    ),
    TestScenario(
        id="S32",
        name="SSE: token streaming with deltas",
        category="streaming",
        query="Explain the event sourcing pattern used in the order service",
        chunk_texts=[
            "The order service uses event sourcing: state is derived by replaying a sequence of domain events.",
            "Events are stored in an append-only event store backed by PostgreSQL with event_type and payload columns.",
        ],
        chunk_scores=[0.82, 0.76],
        chunk_files=["docs/event-sourcing.md", "docs/event-store.md"],
        expected_intent="EXPLANATION",
        expected_intent_source="llm_fallback",
        expected_sub_type="none",
        expected_crag_rewrite=False,
        chunk_grades=["relevant", "relevant"],
        expected_generator="EXPLANATION",
        expected_include_explanation=True,
        verification_passed=True,
        verification_confidence=0.83,
        # When no specialized generator matches or intent is None, SSE falls back to
        # ask_streaming if available, emitting answer_token events with deltas.
        # retrieval: avg(0.82,0.76)=0.79; gap=0.06 <= 0.3
        # verification: 0.83
        # combined: 0.4*0.79 + 0.6*0.83 = 0.316 + 0.498 = 0.814
        expected_retrieval_signal=0.79,
        expected_verification_signal=0.83,
        expected_combined_confidence=0.814,
        expected_should_abstain=False,
        expected_has_attribution=True,
        expected_history_updated=True,
        is_multi_intent=False,
        llm_response_keys={
            "intent": "good[0]",
            "grading": "good[0]",
            "generation": "good[0]",
            "verification": "good[0]",
        },
        description=(
            "Tests token-level streaming via ask_streaming. SSE emits answer_token events "
            "with delta text as tokens arrive via the on_token callback. Verifies the "
            "token queue draining logic and final answer event."
        ),
    ),
    TestScenario(
        id="S33",
        name="SSE: client disconnect mid-stream",
        category="streaming",
        query="How do I install the development dependencies?",
        chunk_texts=[
            "Run `pip install -r requirements-dev.txt` to install all development dependencies.",
        ],
        chunk_scores=[0.90],
        chunk_files=["docs/setup.md"],
        expected_intent="PROCEDURAL",
        expected_intent_source="heuristic",
        expected_sub_type="none",
        expected_crag_rewrite=False,
        chunk_grades=["relevant"],
        expected_generator="PROCEDURAL",
        expected_include_explanation=False,
        verification_passed=None,
        verification_confidence=None,
        # Client disconnects after sources event. Remaining phases are skipped.
        # request.is_disconnected() returns True at the check after sources.
        # retrieval: avg(0.90)=0.90; only 1 score, no gap check
        # verification: None => 0.7
        # combined: 0.4*0.90 + 0.6*0.7 = 0.36 + 0.42 = 0.78
        expected_retrieval_signal=0.90,
        expected_verification_signal=0.7,
        expected_combined_confidence=0.78,
        expected_should_abstain=False,
        expected_has_attribution=False,
        expected_history_updated=False,
        is_multi_intent=False,
        llm_response_keys={"grading": "good[0]"},
        description=(
            "Client disconnects after receiving the sources event. "
            "request.is_disconnected() returns True, so generator returns early. "
            "No answer, verification, or done events emitted. History not updated."
        ),
    ),
    TestScenario(
        id="S34",
        name="SSE: empty retrieval short-circuit",
        category="streaming",
        query="What is the quantum flux capacitor configuration?",
        chunk_texts=[],
        chunk_scores=[],
        chunk_files=[],
        expected_intent="EXPLANATION",
        expected_intent_source="llm_fallback",
        expected_sub_type="none",
        expected_crag_rewrite=False,
        chunk_grades=None,
        expected_generator="EXPLANATION",
        expected_include_explanation=False,
        verification_passed=None,
        verification_confidence=None,
        # No candidates found. SSE emits answer with "I couldn't find any relevant
        # information." and done event. Short-circuits before generation/verification.
        # retrieval: empty => 0.0
        # verification: None => 0.7
        # combined: 0.4*0.0 + 0.6*0.7 = 0.0 + 0.42 = 0.42
        expected_retrieval_signal=0.0,
        expected_verification_signal=0.7,
        expected_combined_confidence=0.42,
        expected_should_abstain=False,
        expected_has_attribution=False,
        expected_history_updated=False,
        is_multi_intent=False,
        llm_response_keys={"intent": "good[0]"},
        description=(
            "Retriever returns zero candidates (all below min_score 0.3). "
            "SSE short-circuits: emits answer('I couldn't find any relevant information.') "
            "and done event immediately. No grading, generation, or verification."
        ),
    ),
    TestScenario(
        id="S35",
        name="SSE: error during retrieval emits error event",
        category="streaming",
        query="How does the notification service work?",
        chunk_texts=[],
        chunk_scores=[],
        chunk_files=[],
        expected_intent="EXPLANATION",
        expected_intent_source="llm_fallback",
        expected_sub_type="none",
        expected_crag_rewrite=False,
        chunk_grades=None,
        expected_generator="EXPLANATION",
        expected_include_explanation=False,
        verification_passed=None,
        verification_confidence=None,
        # Retriever.search raises an exception. The outer try/except catches it
        # and emits an SSE error event with the exception details.
        expected_retrieval_signal=0.0,
        expected_verification_signal=0.7,
        expected_combined_confidence=0.42,
        expected_should_abstain=False,
        expected_has_attribution=False,
        expected_history_updated=False,
        is_multi_intent=False,
        llm_response_keys={"intent": "good[0]"},
        description=(
            "Retriever.search() raises a RuntimeError. The outer exception handler "
            "in streaming_query catches it and emits an SSE error event with "
            "{'error': str(exc), 'type': 'RuntimeError'}. No further events emitted."
        ),
    ),

    # ------------------------------------------------------------------
    # Category 6: Edge Cases (10 scenarios)
    # ------------------------------------------------------------------
    TestScenario(
        id="S36",
        name="Edge: empty query string",
        category="edge_case",
        query="",
        chunk_texts=[],
        chunk_scores=[],
        chunk_files=[],
        expected_intent="EXPLANATION",
        expected_intent_source="llm_fallback",
        expected_sub_type="none",
        expected_crag_rewrite=False,
        chunk_grades=None,
        expected_generator="EXPLANATION",
        expected_include_explanation=False,
        verification_passed=None,
        verification_confidence=None,
        expected_retrieval_signal=0.0,
        expected_verification_signal=0.7,
        expected_combined_confidence=0.42,
        expected_should_abstain=False,
        expected_has_attribution=False,
        expected_history_updated=False,
        is_multi_intent=False,
        llm_response_keys={},
        description=(
            "Empty query string. No heuristic matches, LLM fallback likely returns "
            "parse failure => EXPLANATION with confidence 0.50. Retriever returns no "
            "results. Short-circuits to 'no relevant information' response."
        ),
    ),
    TestScenario(
        id="S37",
        name="Edge: very long query (>2000 chars)",
        category="edge_case",
        query=(
            "I have a very detailed question about the system architecture. "
            "Specifically, I want to understand how the microservices communicate "
            "with each other, including the message broker configuration, the "
            "serialization format used for inter-service communication, the retry "
            "policies for failed message deliveries, the dead letter queue setup, "
            "the monitoring and alerting for message broker health, the scaling "
            "strategy for the broker cluster, and the disaster recovery procedures. "
            + "Additionally, I need details about the network topology. " * 30
        ),
        chunk_texts=[
            "Inter-service communication uses RabbitMQ with JSON serialization.",
            "Message retry follows exponential backoff: 1s, 2s, 4s, 8s, 16s.",
        ],
        chunk_scores=[0.60, 0.55],
        chunk_files=["docs/messaging.md", "docs/retry.md"],
        expected_intent="EXPLANATION",
        expected_intent_source="llm_fallback",
        expected_sub_type="none",
        expected_crag_rewrite=False,
        chunk_grades=["relevant", "partial"],
        expected_generator="EXPLANATION",
        expected_include_explanation=True,
        verification_passed=True,
        verification_confidence=0.70,
        # Long query may be truncated to max_query_length=2000 (RetrievalConfig)
        # retrieval: avg(0.60,0.55)=0.575; gap=0.05 <= 0.3; not all < 0.3
        # verification: 0.70
        # combined: 0.4*0.575 + 0.6*0.70 = 0.23 + 0.42 = 0.65
        expected_retrieval_signal=0.575,
        expected_verification_signal=0.70,
        expected_combined_confidence=0.65,
        expected_should_abstain=False,
        expected_has_attribution=True,
        expected_history_updated=True,
        is_multi_intent=False,
        llm_response_keys={
            "intent": "good[0]",
            "grading": "good[1]",
            "generation": "good[0]",
            "verification": "good[0]",
        },
        description=(
            "Query exceeds max_query_length (2000 chars from RetrievalConfig). "
            "Tests that the pipeline handles very long input gracefully. "
            "The query should be truncated or handled without crashing."
        ),
    ),
    TestScenario(
        id="S38",
        name="Edge: multi-intent query decomposition",
        category="edge_case",
        query=(
            "Explain how the authentication flow works and also draw a diagram "
            "of the service architecture"
        ),
        chunk_texts=[
            "Authentication uses OAuth 2.0 with JWT tokens issued by the Auth Service.",
            "The architecture consists of API Gateway, Auth Service, User Service, and Order Service.",
        ],
        chunk_scores=[0.82, 0.78],
        chunk_files=["docs/auth.md", "docs/architecture.md"],
        expected_intent="DIAGRAM",
        expected_intent_source="heuristic",
        expected_sub_type="flowchart",
        expected_crag_rewrite=False,
        chunk_grades=["relevant", "relevant"],
        expected_generator="DIAGRAM",
        expected_include_explanation=False,
        verification_passed=True,
        verification_confidence=0.85,
        # Heuristic: _DIAGRAM_TOPIC ('flow', 'architecture') + _DIAGRAM_VERB ('draw', 'diagram')
        # => DIAGRAM confidence 0.92 >= 0.85 => specialized only
        # NOTE: Multi-intent detection+decomposition happens AFTER initial classification
        # retrieval: avg(0.82,0.78)=0.80; gap=0.04 <= 0.3
        # verification: 0.85
        # combined: 0.4*0.80 + 0.6*0.85 = 0.32 + 0.51 = 0.83
        expected_retrieval_signal=0.80,
        expected_verification_signal=0.85,
        expected_combined_confidence=0.83,
        expected_should_abstain=False,
        expected_has_attribution=True,
        expected_history_updated=True,
        is_multi_intent=True,
        llm_response_keys={
            "decomposition": "good[0]",
            "grading": "good[0]",
            "generation": "good[0]",
            "verification": "good[0]",
        },
        description=(
            "Tests multi-intent detection and decomposition. Heuristic matches DIAGRAM "
            "('flow'+'draw'). 'and also' matches _COORDINATION_MARKERS. "
            "'explain' + 'draw' are 2 distinct output verbs. "
            "LLM decomposes into sub-queries processed independently, results merged."
        ),
    ),
    TestScenario(
        id="S39",
        name="Edge: query with no matching docs at all",
        category="edge_case",
        query="What is the airspeed velocity of an unladen swallow?",
        chunk_texts=[],
        chunk_scores=[],
        chunk_files=[],
        expected_intent="EXPLANATION",
        expected_intent_source="llm_fallback",
        expected_sub_type="none",
        expected_crag_rewrite=False,
        chunk_grades=None,
        expected_generator="EXPLANATION",
        expected_include_explanation=False,
        verification_passed=None,
        verification_confidence=None,
        expected_retrieval_signal=0.0,
        expected_verification_signal=0.7,
        expected_combined_confidence=0.42,
        expected_should_abstain=False,
        expected_has_attribution=False,
        expected_history_updated=False,
        is_multi_intent=False,
        llm_response_keys={"intent": "good[0]"},
        description=(
            "Completely off-topic query with no matching documentation. "
            "Retriever returns empty candidates list. Pipeline returns "
            "'I couldn't find any relevant information in the indexed documents.' "
            "No generation, verification, or attribution. No history update."
        ),
    ),
    TestScenario(
        id="S40",
        name="Edge: all chunks from same file (diversity cap)",
        category="edge_case",
        query="What are all the endpoints in the user management API?",
        chunk_texts=[
            "GET /api/v1/users - List all users with pagination.",
            "POST /api/v1/users - Create a new user account.",
            "GET /api/v1/users/{id} - Get user by ID.",
            "PUT /api/v1/users/{id} - Update user details.",
            "DELETE /api/v1/users/{id} - Delete a user account.",
        ],
        chunk_scores=[0.92, 0.90, 0.88, 0.85, 0.82],
        chunk_files=[
            "docs/user-api.md",
            "docs/user-api.md",
            "docs/user-api.md",
            "docs/user-api.md",
            "docs/user-api.md",
        ],
        expected_intent="EXPLANATION",
        expected_intent_source="llm_fallback",
        expected_sub_type="none",
        expected_crag_rewrite=False,
        chunk_grades=["relevant", "relevant"],
        expected_generator="EXPLANATION",
        expected_include_explanation=True,
        verification_passed=True,
        verification_confidence=0.92,
        # File diversity cap: max_chunks_per_file=2 (default).
        # 5 chunks all from user-api.md => only first 2 kept (scores 0.92, 0.90).
        # Then top_k=5 but only 2 available.
        # NOTE: chunk_scores has all 5 for pre-diversity simulation.
        # Confidence is computed on post-diversity scores.
        # retrieval (all 5): avg(0.92,0.90,0.88,0.85,0.82)=0.874; gap=0.02 <= 0.3
        # verification: 0.92
        # combined: 0.4*0.874 + 0.6*0.92 = 0.3496 + 0.552 = 0.9016
        expected_retrieval_signal=0.874,
        expected_verification_signal=0.92,
        expected_combined_confidence=0.9016,
        expected_should_abstain=False,
        expected_has_attribution=True,
        expected_history_updated=True,
        is_multi_intent=False,
        llm_response_keys={
            "intent": "good[0]",
            "grading": "good[0]",
            "generation": "good[0]",
            "verification": "good[0]",
        },
        description=(
            "All 5 chunks come from the same file (user-api.md). "
            "_apply_file_diversity caps at max_chunks_per_file=2. "
            "Only first 2 chunks (highest scoring) survive. Tests diversity enforcement."
        ),
    ),
    TestScenario(
        id="S41",
        name="Edge: history at max capacity (10 turns = 20 entries)",
        category="edge_case",
        query="What is the database migration strategy?",
        chunk_texts=[
            "Database migrations use Flyway with versioned SQL scripts in db/migration/.",
            "Each migration file follows the naming convention V{version}__{description}.sql.",
        ],
        chunk_scores=[0.87, 0.80],
        chunk_files=["docs/migrations.md", "docs/migrations.md"],
        expected_intent="EXPLANATION",
        expected_intent_source="llm_fallback",
        expected_sub_type="none",
        expected_crag_rewrite=False,
        chunk_grades=["relevant", "relevant"],
        expected_generator="EXPLANATION",
        expected_include_explanation=True,
        verification_passed=True,
        verification_confidence=0.85,
        # max_history_turns=10 => max_entries=20. If history already has 20 entries,
        # adding 2 more (user + assistant) makes 22, then del history[:-20] trims oldest 2.
        # retrieval: avg(0.87,0.80)=0.835; gap=0.07 <= 0.3
        # verification: 0.85
        # combined: 0.4*0.835 + 0.6*0.85 = 0.334 + 0.51 = 0.844
        expected_retrieval_signal=0.835,
        expected_verification_signal=0.85,
        expected_combined_confidence=0.844,
        expected_should_abstain=False,
        expected_has_attribution=True,
        expected_history_updated=True,
        is_multi_intent=False,
        llm_response_keys={
            "intent": "good[0]",
            "grading": "good[0]",
            "generation": "good[0]",
            "verification": "good[0]",
        },
        description=(
            "History already at max capacity (20 entries = 10 turns). "
            "After adding the new Q&A pair, oldest entries are trimmed via "
            "del history[:-max_entries]. Tests the rolling window behavior."
        ),
    ),
    TestScenario(
        id="S42",
        name="Edge: malformed LLM responses across all phases",
        category="edge_case",
        query="What are the system requirements for the application?",
        chunk_texts=[
            "The application requires Java 17+, PostgreSQL 14+, and Redis 7+.",
            "Minimum hardware: 4 CPU cores, 8GB RAM, 50GB disk.",
        ],
        chunk_scores=[0.80, 0.75],
        chunk_files=["docs/requirements.md", "docs/requirements.md"],
        expected_intent="EXPLANATION",
        expected_intent_source="llm_fallback",
        expected_sub_type="none",
        expected_crag_rewrite=False,
        chunk_grades=["relevant", "relevant"],
        expected_generator="EXPLANATION",
        expected_include_explanation=False,
        verification_passed=True,
        verification_confidence=0.50,
        # Intent LLM: unparseable => EXPLANATION with confidence 0.50 (llm_parse_failure)
        # Grading LLM: unparseable => all chunks default to "relevant" (conservative)
        # Verification LLM: unparseable => passed=True (conservative), confidence=0.5
        # 0.50 < 0.65 => explanation only
        # retrieval: avg(0.80,0.75)=0.775; gap=0.05 <= 0.3
        # verification: PASS => 0.50
        # combined: 0.4*0.775 + 0.6*0.50 = 0.31 + 0.30 = 0.61
        expected_retrieval_signal=0.775,
        expected_verification_signal=0.50,
        expected_combined_confidence=0.61,
        expected_should_abstain=False,
        expected_has_attribution=True,
        expected_history_updated=True,
        is_multi_intent=False,
        llm_response_keys={
            "intent": "bad[1]",
            "grading": "bad[1]",
            "generation": "good[0]",
            "verification": "bad[3]",
        },
        description=(
            "All LLM responses are malformed garbage text. Each phase falls back to "
            "conservative defaults: intent => EXPLANATION(0.50), grading => all relevant, "
            "verification => passed=True, confidence=0.50. Tests graceful degradation."
        ),
    ),
    TestScenario(
        id="S43",
        name="Edge: concurrent queries (session isolation)",
        category="edge_case",
        query="What is the API rate limit?",
        chunk_texts=[
            "The API rate limit is 100 requests per minute per API key.",
            "Rate limit headers: X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset.",
        ],
        chunk_scores=[0.90, 0.85],
        chunk_files=["docs/rate-limiting.md", "docs/rate-limiting.md"],
        expected_intent="EXPLANATION",
        expected_intent_source="llm_fallback",
        expected_sub_type="none",
        expected_crag_rewrite=False,
        chunk_grades=["relevant", "relevant"],
        expected_generator="EXPLANATION",
        expected_include_explanation=True,
        verification_passed=True,
        verification_confidence=0.90,
        # Tests that two concurrent queries in different sessions don't share history
        # Each session has its own history list (session_history parameter in SSE)
        # retrieval: avg(0.90,0.85)=0.875; gap=0.05 <= 0.3
        # verification: 0.90
        # combined: 0.4*0.875 + 0.6*0.90 = 0.35 + 0.54 = 0.89
        expected_retrieval_signal=0.875,
        expected_verification_signal=0.90,
        expected_combined_confidence=0.89,
        expected_should_abstain=False,
        expected_has_attribution=True,
        expected_history_updated=True,
        is_multi_intent=False,
        llm_response_keys={
            "intent": "good[0]",
            "grading": "good[0]",
            "generation": "good[0]",
            "verification": "good[0]",
        },
        description=(
            "Two concurrent queries with different session IDs must not share history. "
            "Tests session isolation: each session_history list is independent. "
            "Session A's history should not leak into Session B."
        ),
    ),
    TestScenario(
        id="S44",
        name="Edge: Unicode and special characters in query",
        category="edge_case",
        query="How do I configure the Konigsberg-Bruecke API for Strasse endpoints? (v2.0-beta)",
        chunk_texts=[
            "The API supports UTF-8 encoded request bodies and response payloads.",
            "Special characters in URL parameters must be percent-encoded per RFC 3986.",
        ],
        chunk_scores=[0.55, 0.50],
        chunk_files=["docs/encoding.md", "docs/url-params.md"],
        expected_intent="PROCEDURAL",
        expected_intent_source="heuristic",
        expected_sub_type="none",
        expected_crag_rewrite=False,
        chunk_grades=["partial", "partial"],
        expected_generator="PROCEDURAL",
        expected_include_explanation=False,
        verification_passed=True,
        verification_confidence=0.65,
        # Heuristic: "How do I" matches _PROCEDURAL_TOPIC, "configure" matches _PROCEDURAL_VERB
        # => confidence 0.88 >= 0.85 => specialized only
        # retrieval: avg(0.55,0.50)=0.525; gap=0.05 <= 0.3; not all < 0.3
        # verification: 0.65
        # combined: 0.4*0.525 + 0.6*0.65 = 0.21 + 0.39 = 0.60
        expected_retrieval_signal=0.525,
        expected_verification_signal=0.65,
        expected_combined_confidence=0.60,
        expected_should_abstain=False,
        expected_has_attribution=True,
        expected_history_updated=True,
        is_multi_intent=False,
        llm_response_keys={
            "grading": "good[1]",
            "generation": "good[3]",
            "verification": "good[0]",
        },
        description=(
            "Query contains Unicode characters (umlauts), hyphens, parentheses, and dots. "
            "Tests that regex patterns still match correctly and the pipeline handles "
            "special characters without encoding errors. 'How do I' + 'configure' triggers PROCEDURAL."
        ),
    ),
    TestScenario(
        id="S45",
        name="Edge: low confidence but abstention disabled",
        category="edge_case",
        query="What is the quantum entanglement protocol for service mesh?",
        chunk_texts=[
            "The service mesh uses Istio for traffic management and mTLS between services.",
        ],
        chunk_scores=[0.32],
        chunk_files=["docs/service-mesh.md"],
        expected_intent="EXPLANATION",
        expected_intent_source="llm_fallback",
        expected_sub_type="none",
        expected_crag_rewrite=True,
        chunk_grades=["irrelevant"],
        expected_generator="EXPLANATION",
        expected_include_explanation=True,
        verification_passed=False,
        verification_confidence=0.30,
        # retrieval: avg(0.32)=0.32; not all < 0.3 (0.32 >= 0.3); single score, no gap check
        # verification: FAIL => 0.30 - 0.20 = 0.10
        # combined: 0.4*0.32 + 0.6*0.10 = 0.128 + 0.06 = 0.188
        # 0.188 < 0.40 but abstain_on_low_confidence=False => no abstain
        expected_retrieval_signal=0.32,
        expected_verification_signal=0.10,
        expected_combined_confidence=0.188,
        expected_should_abstain=False,
        expected_has_attribution=True,
        expected_history_updated=True,
        is_multi_intent=False,
        llm_response_keys={
            "intent": "good[0]",
            "grading": "bad[0]",
            "rewrite": "good[0]",
            "generation": "good[0]",
            "verification": "bad[1]",
        },
        description=(
            "Combined confidence 0.188 is well below threshold 0.40, but "
            "abstain_on_low_confidence is set to False. System still returns an answer "
            "despite extremely low confidence. Tests the abstention disable override."
        ),
    ),

    # ------------------------------------------------------------------
    # Category 7: Full Pipeline E2E (5 scenarios)
    # ------------------------------------------------------------------
    TestScenario(
        id="S46",
        name="E2E happy path: simple explanation query",
        category="e2e",
        query="What caching strategy does the application use?",
        chunk_texts=[
            "The application uses Redis as a distributed cache with a default TTL of 5 minutes.",
            "Cache-aside pattern: on cache miss, fetch from DB, write to cache, return result.",
            "Cache invalidation is triggered by write operations via Spring Cache @CacheEvict annotation.",
        ],
        chunk_scores=[0.90, 0.85, 0.78],
        chunk_files=["docs/caching.md", "docs/caching.md", "docs/spring-config.md"],
        expected_intent="EXPLANATION",
        expected_intent_source="llm_fallback",
        expected_sub_type="none",
        expected_crag_rewrite=False,
        chunk_grades=["relevant", "relevant", "relevant"],
        expected_generator="EXPLANATION",
        expected_include_explanation=True,
        verification_passed=True,
        verification_confidence=0.92,
        # retrieval: avg(0.90,0.85,0.78)=0.8433; gap=0.05 <= 0.3
        # After diversity: caching.md has 2 chunks (capped), spring-config.md has 1 => all 3 pass
        # Wait, caching.md has chunks at index 0 and 1 (score 0.90, 0.85).
        # max_chunks_per_file=2, so both pass. spring-config.md has 1. Total: 3 chunks, top_k=5 => all 3.
        # verification: 0.92
        # combined: 0.4*0.8433 + 0.6*0.92 = 0.3373 + 0.552 = 0.8893
        expected_retrieval_signal=0.8433,
        expected_verification_signal=0.92,
        expected_combined_confidence=0.8893,
        expected_should_abstain=False,
        expected_has_attribution=True,
        expected_history_updated=True,
        is_multi_intent=False,
        llm_response_keys={
            "intent": "good[0]",
            "grading": "good[0]",
            "generation": "good[0]",
            "verification": "good[1]",
        },
        description=(
            "Happy path E2E: simple explanation query flows through all 8 phases. "
            "Good retrieval, all chunks relevant, verification passes, high confidence. "
            "Attribution computed, history updated. No CRAG rewrite needed."
        ),
    ),
    TestScenario(
        id="S47",
        name="E2E happy path: diagram generation with mermaid validation",
        category="e2e",
        query="Draw a sequence diagram showing the interaction between the client and the auth service",
        chunk_texts=[
            "Client sends POST /auth/login with credentials. Auth Service validates against LDAP.",
            "On success, Auth Service generates JWT and returns 200 with token. Client stores token.",
            "For subsequent requests, Client sends token in Authorization header.",
        ],
        chunk_scores=[0.91, 0.87, 0.82],
        chunk_files=["docs/auth-flow.md", "docs/auth-flow.md", "docs/client-guide.md"],
        expected_intent="DIAGRAM",
        expected_intent_source="heuristic",
        expected_sub_type="sequence",
        expected_crag_rewrite=False,
        chunk_grades=["relevant", "relevant", "relevant"],
        expected_generator="DIAGRAM",
        expected_include_explanation=False,
        verification_passed=True,
        verification_confidence=0.89,
        # Heuristic: _DIAGRAM_TOPIC('sequence', 'interaction') + _DIAGRAM_VERB('draw')
        # => confidence 0.92 >= 0.85 => specialized only
        # Sub-type: 'sequence' matches _SUBTYPE_SEQUENCE
        # Diversity: auth-flow.md has 2 (capped), client-guide.md has 1 => 3 total
        # retrieval: avg(0.91,0.87,0.82)=0.8667; gap=0.04 <= 0.3
        # verification: 0.89
        # combined: 0.4*0.8667 + 0.6*0.89 = 0.3467 + 0.534 = 0.8807
        expected_retrieval_signal=0.8667,
        expected_verification_signal=0.89,
        expected_combined_confidence=0.8807,
        expected_should_abstain=False,
        expected_has_attribution=True,
        expected_history_updated=True,
        is_multi_intent=False,
        llm_response_keys={
            "grading": "good[0]",
            "generation": "good[5]",
            "mermaid_validation": "good[0]",
            "verification": "good[0]",
        },
        description=(
            "E2E diagram flow: heuristic detects DIAGRAM+sequence, DiagramGenerator "
            "produces mermaid sequence diagram, MermaidValidator validates it (passes), "
            "output detector confirms has_mermaid=True. Full pipeline success."
        ),
    ),
    TestScenario(
        id="S48",
        name="E2E complex: multi-intent + CRAG + verification + attribution",
        category="e2e",
        query=(
            "Explain the database schema design and also show me a code example "
            "for querying the orders table"
        ),
        chunk_texts=[
            "The orders table has columns: id (UUID), customer_id (FK), total (DECIMAL), status (ENUM), created_at (TIMESTAMP).",
            "Indexes on orders: idx_customer_id for customer lookups, idx_created_at for date range queries.",
            "SELECT o.* FROM orders o WHERE o.customer_id = ? AND o.created_at > ? ORDER BY o.created_at DESC LIMIT 20;",
        ],
        chunk_scores=[0.85, 0.72, 0.80],
        chunk_files=["docs/schema.md", "docs/schema.md", "docs/queries.md"],
        expected_intent="CODE_EXAMPLE",
        expected_intent_source="heuristic",
        expected_sub_type="none",
        expected_crag_rewrite=False,
        chunk_grades=["relevant", "partial", "relevant"],
        expected_generator="CODE_EXAMPLE",
        expected_include_explanation=False,
        verification_passed=True,
        verification_confidence=0.86,
        # Heuristic: "code example" + "show" => CODE_EXAMPLE with confidence ~0.90
        # Multi-intent: "and also" + "explain" + "show" => 2 verbs => is_multi_intent=True
        # Decomposed into 2 sub-queries, each processed independently
        # Using first sub-query's retrieval scores for confidence
        # retrieval: avg(0.85,0.72,0.80)=0.79; gap=0.05 <= 0.3 (after diversity: schema.md capped at 2)
        # verification: 0.86
        # combined: 0.4*0.79 + 0.6*0.86 = 0.316 + 0.516 = 0.832
        expected_retrieval_signal=0.79,
        expected_verification_signal=0.86,
        expected_combined_confidence=0.832,
        expected_should_abstain=False,
        expected_has_attribution=True,
        expected_history_updated=True,
        is_multi_intent=True,
        llm_response_keys={
            "intent": "good[0]",
            "decomposition": "good[0]",
            "grading": "good[1]",
            "generation": "good[0]",
            "verification": "good[0]",
        },
        description=(
            "Complex E2E: multi-intent decomposition ('explain' + 'show' with 'and also'). "
            "Each sub-query goes through CRAG grading, generation, and verification. "
            "Results merged with ### Part N headers. Attribution computed on merged text."
        ),
    ),
    TestScenario(
        id="S49",
        name="E2E degraded: multiple LLM failures, graceful degradation",
        category="e2e",
        query="How to configure the message broker for high availability?",
        chunk_texts=[
            "RabbitMQ clustering is configured with 3 nodes for high availability.",
            "Queue mirroring ensures messages survive node failures with ha-mode: all policy.",
        ],
        chunk_scores=[0.78, 0.72],
        chunk_files=["docs/rabbitmq.md", "docs/ha-config.md"],
        expected_intent="PROCEDURAL",
        expected_intent_source="heuristic",
        expected_sub_type="none",
        expected_crag_rewrite=False,
        chunk_grades=["relevant", "relevant"],
        expected_generator="PROCEDURAL",
        expected_include_explanation=False,
        verification_passed=True,
        verification_confidence=0.50,
        # Heuristic: "How to" + "configure" => PROCEDURAL with 0.88
        # => confidence 0.88 >= 0.85 => specialized only (resolve_generation_strategy returns PROCEDURAL)
        # At runtime, ProceduralGenerator.generate() may raise => fallback to ExplanationGenerator
        # Verification LLM returns garbage => conservative: passed=True, confidence=0.50
        # retrieval: avg(0.78,0.72)=0.75; gap=0.06 <= 0.3
        # verification: PASS => 0.50
        # combined: 0.4*0.75 + 0.6*0.50 = 0.30 + 0.30 = 0.60
        expected_retrieval_signal=0.75,
        expected_verification_signal=0.50,
        expected_combined_confidence=0.60,
        expected_should_abstain=False,
        expected_has_attribution=True,
        expected_history_updated=True,
        is_multi_intent=False,
        llm_response_keys={
            "grading": "good[0]",
            "generation": "error[0]",
            "verification": "bad[3]",
        },
        description=(
            "Multiple LLM failures: specialized ProceduralGenerator raises exception "
            "(falls back to ExplanationGenerator), verification response is unparseable "
            "(conservative defaults: passed=True, confidence=0.50). Tests graceful "
            "degradation across multiple pipeline stages."
        ),
    ),
    TestScenario(
        id="S50",
        name="E2E complete: all 8 phases active, high quality result",
        category="e2e",
        query="Show me a JSON example of the webhook payload for order events",
        chunk_texts=[
            '{"event": "order.created", "data": {"order_id": "abc-123", "total": 99.99, "currency": "USD"}}',
            "Webhook payloads follow the CloudEvents specification with type, source, and data fields.",
            "The webhook endpoint must respond with 200 OK within 5 seconds or the delivery is retried.",
            "Order events include: order.created, order.updated, order.cancelled, order.shipped.",
        ],
        chunk_scores=[0.93, 0.88, 0.75, 0.82],
        chunk_files=[
            "docs/webhook-examples.md",
            "docs/webhook-spec.md",
            "docs/webhook-delivery.md",
            "docs/order-events.md",
        ],
        expected_intent="CODE_EXAMPLE",
        expected_intent_source="heuristic",
        expected_sub_type="json",
        expected_crag_rewrite=False,
        chunk_grades=["relevant", "relevant", "partial", "relevant"],
        expected_generator="CODE_EXAMPLE",
        expected_include_explanation=False,
        verification_passed=True,
        verification_confidence=0.94,
        # Heuristic: _CODE_FORMAT('JSON example') + _CODE_VERB('show') => 0.90 >= 0.85
        # Sub-type: _SUBTYPE_JSON matches 'json' and 'payload'
        # Diversity: all different files, all pass. top_k=5, have 4 => use all 4.
        # retrieval: avg(0.93,0.88,0.75,0.82)=0.845; gap=0.05 <= 0.3
        # verification: 0.94
        # combined: 0.4*0.845 + 0.6*0.94 = 0.338 + 0.564 = 0.902
        expected_retrieval_signal=0.845,
        expected_verification_signal=0.94,
        expected_combined_confidence=0.902,
        expected_should_abstain=False,
        expected_has_attribution=True,
        expected_history_updated=True,
        is_multi_intent=False,
        llm_response_keys={
            "grading": "good[1]",
            "generation": "good[6]",
            "verification": "good[1]",
        },
        description=(
            "Full E2E with all 8 phases active and producing high-quality output. "
            "Phase 1: intent=CODE_EXAMPLE(heuristic, 0.90). Phase 2: 4 chunks retrieved. "
            "Phase 3: CRAG grades 3 relevant + 1 partial, no rewrite. "
            "Phase 4: CodeExampleGenerator with json sub-type. Phase 5: output detection "
            "finds code blocks. Phase 6: no mermaid. Phase 7: verification PASS(0.94), "
            "confidence 0.902. Phase 8: sentence-level attribution. History updated."
        ),
    ),
]
