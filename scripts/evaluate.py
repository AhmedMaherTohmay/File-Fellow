"""
LLM-as-Judge Evaluation Pipeline for File Fellow.

Pipeline (per sample):
  1. Sample two random document chunks from the ingested vector store.
  2. Feed both chunks to a generator LLM → synthetic (question, ground_truth_answer) pair.
  3. Feed the same question to the live RAG agent → candidate answer + retrieved context.
  4. Feed (question, ground_truth, retrieved_context, candidate_answer) to a judge LLM.
  5. Judge scores the candidate on Faithfulness, Relevance, and Correctness (each 0–10)
     and flags hallucinations.

Usage:
    # Ensure Fraud_Detection_System_Design.pdf is already ingested, then:
    python scripts/evaluate.py

    # Custom options:
    python scripts/evaluate.py --num_samples 10 --output reports/eval_out.json

Output:
    JSON report with per-sample details and aggregate averages.
    Pretty summary printed to stdout.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

# Make sure project root is on the path when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────

def _parse_json_response(text: str) -> Optional[Dict]:
    """Extract the first JSON object from an LLM response string.

    The model may wrap the JSON in markdown fences or add preamble text.
    We strip those defensively before parsing.
    """
    # Remove markdown code fences if present
    text = re.sub(r"```(?:json)?", "", text).strip()

    # Try to find the first { ... } block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError as e:
            logger.warning("JSON decode failed: %s\nRaw text: %s", e, text[:300])
    return None


def _parse_json_array(text: str) -> Optional[List]:
    """Extract the first JSON array from an LLM response string."""
    text = re.sub(r"```(?:json)?", "", text).strip()
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError as e:
            logger.warning("JSON array decode failed: %s", e)
    return None


# ── Step 1: Sample chunks from the vector store ────────────────────────────

def sample_chunks(doc_name: Optional[str] = None, n: int = 2) -> List[Any]:
    """Sample ``n`` random document chunks from the Chroma vector store.

    Args:
        doc_name: If given, restrict sampling to this document's store.
                  If None, sample from the global store (all documents).
        n: Number of chunks to sample.

    Returns:
        List of LangChain Document objects.

    Raises:
        RuntimeError: If the vector store is not ready or has too few chunks.
    """
    from src.ingestion.vector_store import (
        get_global_store,
        get_store_for_doc,
        store_is_ready,
    )

    if not store_is_ready():
        raise RuntimeError(
            "No documents are ingested. "
            "Please ingest Fraud_Detection_System_Design.pdf first."
        )

    store = get_store_for_doc(doc_name) if doc_name else get_global_store()

    # Chroma exposes the underlying collection via ._collection
    try:
        collection = store._collection
        total = collection.count()
    except AttributeError:
        # Fallback: run a broad similarity search and sample from those results
        logger.warning("Cannot access _collection directly; using similarity search fallback.")
        results = store.similarity_search("fraud detection", k=max(n * 5, 20))
        if len(results) < n:
            raise RuntimeError(f"Not enough chunks to sample {n} items (got {len(results)}).")
        return random.sample(results, n)

    if total < n:
        raise RuntimeError(
            f"Only {total} chunks available; need at least {n} to sample."
        )

    # Sample random IDs then fetch those documents
    all_ids = collection.get(include=[])["ids"]
    sampled_ids = random.sample(all_ids, n)
    raw = collection.get(ids=sampled_ids, include=["documents", "metadatas"])

    # Convert to LangChain Document objects for consistent downstream use
    from langchain_core.documents import Document
    docs = []
    for text, meta in zip(raw["documents"], raw["metadatas"]):
        docs.append(Document(page_content=text, metadata=meta or {}))

    logger.info("Sampled %d chunks from the vector store.", n)
    return docs


# ── Step 2: Generate synthetic baseline Q&A pair ──────────────────────────

def generate_baseline_qa(
    chunk_1: str,
    chunk_2: str,
    num_questions: int = 1,
) -> List[Dict[str, str]]:
    """Use the LLM to create synthetic (question, ground_truth_answer) pairs.

    Both chunks are shown to the model so the question can span information
    from either excerpt, increasing question diversity.

    Args:
        chunk_1: Text of the first sampled chunk.
        chunk_2: Text of the second sampled chunk.
        num_questions: How many QA pairs to generate from these chunks.

    Returns:
        List of {"question": ..., "answer": ...} dicts.
        Returns an empty list if generation or parsing fails.
    """
    from src.llm.llm_factory import get_llm
    from src.llm.prompts import QUESTION_GEN_PROMPT

    llm = get_llm()
    chain = QUESTION_GEN_PROMPT | llm

    try:
        response = chain.invoke(
            {
                "chunk_1": chunk_1,
                "chunk_2": chunk_2,
                "num_questions": num_questions,
            }
        )
        raw_text = response.content if hasattr(response, "content") else str(response)
        parsed = _parse_json_array(raw_text)

        if not parsed:
            logger.warning("Could not parse QA array from generator response.")
            return []

        # Normalise — each item must have "question" and "answer"
        pairs = []
        for item in parsed:
            if isinstance(item, dict) and "question" in item and "answer" in item:
                pairs.append({"question": item["question"], "answer": item["answer"]})

        logger.info("Generated %d baseline QA pairs from chunks.", len(pairs))
        return pairs

    except Exception as e:
        logger.error("Baseline QA generation failed: %s", e)
        return []


# ── Step 3: Run the RAG agent on the same question ─────────────────────────

def run_rag_agent(question: str, doc_name: Optional[str] = None) -> Dict[str, Any]:
    """Query the live RAG pipeline and return the answer with its sources.

    Args:
        question: The synthetic question generated in Step 2.
        doc_name: Optionally restrict retrieval to one document.

    Returns:
        Dict with keys ``answer``, ``sources``, and ``retrieved_chunks``.
    """
    from src.llm.qa_chain import answer_question

    # Use a dedicated eval session ID so eval turns don't pollute user history
    return answer_question(
        question=question,
        history=[],
        doc_name=doc_name,
        session_id="__eval__",
    )


# ── Step 4: Judge the candidate answer ────────────────────────────────────

def judge_answer(
    question: str,
    ground_truth: str,
    candidate_answer: str,
    retrieved_context: str,
) -> Dict[str, Any]:
    """Use the judge LLM to score the RAG candidate answer.

    Compares the candidate against the ground-truth answer and the
    retrieved context simultaneously.

    Args:
        question: The original synthetic question.
        ground_truth: The baseline answer produced from the source chunks.
        candidate_answer: The answer generated by the RAG pipeline.
        retrieved_context: The context text the RAG agent retrieved.

    Returns:
        Dict with keys: faithfulness, relevance, correctness,
        hallucination_flag, reasoning.
        Returns a zeroed-out dict on parse failure.
    """
    from src.llm.llm_factory import get_llm_for_eval
    from src.llm.prompts import JUDGE_PROMPT

    judge_llm = get_llm_for_eval()
    chain = JUDGE_PROMPT | judge_llm

    try:
        response = chain.invoke(
            {
                "question": question,
                "ground_truth": ground_truth,
                "context": retrieved_context,
                "candidate_answer": candidate_answer,
            }
        )
        raw_text = response.content if hasattr(response, "content") else str(response)
        parsed = _parse_json_response(raw_text)

        if parsed:
            return {
                "faithfulness": float(parsed.get("faithfulness", 0)),
                "relevance": float(parsed.get("relevance", 0)),
                "correctness": float(parsed.get("correctness", 0)),
                "hallucination_flag": bool(parsed.get("hallucination_flag", False)),
                "reasoning": parsed.get("reasoning", ""),
            }

        logger.warning("Could not parse judge response; defaulting to zeros.")
        return {
            "faithfulness": 0.0,
            "relevance": 0.0,
            "correctness": 0.0,
            "hallucination_flag": True,
            "reasoning": "Parse error — raw response: " + raw_text[:200],
        }

    except Exception as e:
        logger.error("Judge LLM call failed: %s", e)
        return {
            "faithfulness": 0.0,
            "relevance": 0.0,
            "correctness": 0.0,
            "hallucination_flag": True,
            "reasoning": f"Judge error: {e}",
        }


# ── Full evaluation pipeline ───────────────────────────────────────────────

def run_evaluation(
    doc_name: Optional[str],
    num_samples: int,
    qa_per_sample: int = 1,
) -> Dict[str, Any]:
    """Execute the full LLM-as-judge evaluation pipeline.

    For each sample:
      1. Draw 2 random chunks from the store.
      2. Generate ``qa_per_sample`` synthetic (Q, A) pairs from those chunks.
      3. Run each question through the RAG agent.
      4. Use the judge LLM to score the RAG answer vs. the ground truth.

    Args:
        doc_name: Document to target, or None for all documents.
        num_samples: Number of chunk-pair samples to draw.
        qa_per_sample: QA pairs to generate per chunk sample (default 1).

    Returns:
        Dict with ``summary`` (aggregate metrics) and ``details`` (per-question rows).
    """
    all_results: List[Dict[str, Any]] = []
    sample_idx = 0

    for i in range(num_samples):
        logger.info("Sample %d/%d — drawing chunks...", i + 1, num_samples)

        # Step 1: Sample two chunks
        try:
            chunks = sample_chunks(doc_name=doc_name, n=2)
        except RuntimeError as e:
            logger.error("Chunk sampling failed: %s", e)
            break

        chunk_1_text = chunks[0].page_content
        chunk_2_text = chunks[1].page_content
        chunk_source = chunks[0].metadata.get("source", "unknown")

        # Step 2: Generate synthetic baseline Q&A
        qa_pairs = generate_baseline_qa(chunk_1_text, chunk_2_text, num_questions=qa_per_sample)
        if not qa_pairs:
            logger.warning("Skipping sample %d — no QA pairs generated.", i + 1)
            continue

        for qa in qa_pairs:
            question = qa["question"]
            ground_truth = qa["answer"]
            sample_idx += 1
            logger.info("  [%d] Evaluating: %s", sample_idx, question[:80])

            # Step 3: Run the RAG agent
            rag_result = run_rag_agent(question, doc_name=doc_name)
            candidate_answer = rag_result.get("answer", "")

            # Build the retrieved context string for the judge
            retrieved_chunks = rag_result.get("retrieved_chunks", [])
            if retrieved_chunks:
                retrieved_context = "\n\n---\n\n".join(
                    doc.page_content for doc, _ in retrieved_chunks
                )
            else:
                retrieved_context = "No context retrieved."

            # Step 4: Judge
            scores = judge_answer(
                question=question,
                ground_truth=ground_truth,
                candidate_answer=candidate_answer,
                retrieved_context=retrieved_context,
            )

            all_results.append(
                {
                    "sample": sample_idx,
                    "source_document": chunk_source,
                    "question": question,
                    "ground_truth": ground_truth,
                    "candidate_answer": candidate_answer,
                    "retrieved_context_preview": retrieved_context[:400],
                    "num_chunks_retrieved": len(retrieved_chunks),
                    **scores,
                }
            )

    if not all_results:
        logger.error("No evaluation results collected.")
        return {"summary": {}, "details": []}

    # ── Aggregate metrics ──────────────────────────────────────────────────
    n = len(all_results)

    def _avg(key: str) -> float:
        return round(sum(r[key] for r in all_results) / n, 3)

    hallucination_rate = round(
        sum(1 for r in all_results if r["hallucination_flag"]) / n, 3
    )

    summary = {
        "num_samples_evaluated": n,
        "avg_faithfulness": _avg("faithfulness"),
        "avg_relevance": _avg("relevance"),
        "avg_correctness": _avg("correctness"),
        "hallucination_rate": hallucination_rate,
        "avg_composite_score": round(
            (_avg("faithfulness") + _avg("relevance") + _avg("correctness")) / 3, 3
        ),
    }

    return {"summary": summary, "details": all_results}


# ── CLI entrypoint ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "LLM-as-Judge evaluation pipeline for File Fellow. "
            "Requires Fraud_Detection_System_Design.pdf to be ingested."
        )
    )
    parser.add_argument(
        "--doc_name",
        default="Fraud_Detection_System_Design.pdf",
        help="Name of the ingested document to evaluate against "
             "(default: Fraud_Detection_System_Design.pdf). "
             "Pass 'all' to use the global store.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of chunk-pair samples to draw (each produces 1+ QA pairs).",
    )
    parser.add_argument(
        "--qa_per_sample",
        type=int,
        default=1,
        help="Number of QA pairs to generate per chunk sample (default: 1).",
    )
    parser.add_argument(
        "--output",
        default="eval_report.json",
        help="Path to write the JSON evaluation report.",
    )
    args = parser.parse_args()

    # Resolve "all" to None (global store)
    doc_name = None if args.doc_name.lower() == "all" else args.doc_name

    logger.info(
        "Starting evaluation | doc=%s | samples=%d | qa_per_sample=%d",
        doc_name or "ALL",
        args.num_samples,
        args.qa_per_sample,
    )

    report = run_evaluation(
        doc_name=doc_name,
        num_samples=args.num_samples,
        qa_per_sample=args.qa_per_sample,
    )

    # Pretty-print summary to stdout
    print("\n" + "=" * 55)
    print("  LLM-AS-JUDGE EVALUATION REPORT")
    print("=" * 55)
    if report["summary"]:
        for key, val in report["summary"].items():
            label = key.replace("_", " ").title()
            print(f"  {label:<30} {val}")
    else:
        print("  No results collected.")
    print("=" * 55)

    # Save full report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Full report saved to '%s'.", output_path)


if __name__ == "__main__":
    main()
