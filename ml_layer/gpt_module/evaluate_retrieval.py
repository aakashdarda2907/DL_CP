"""
Retrieval Evaluation Script for WeightCoachAI Knowledge Base (RAG).

Evaluates the retriever (get_relevant_context from knowledge_base.py)
against a labeled dataset of user questions and expected facts.

Calculates:
- Precision@k: fraction of retrieved facts that are relevant
- Recall@k: fraction of expected facts that were retrieved
- F1@k: harmonic mean of precision and recall
- Hit Rate: fraction of queries where at least one relevant fact was retrieved
- MRR (Mean Reciprocal Rank): reciprocal rank of the first relevant fact

Edge Cases:
- Out-of-scope / empty expected facts:
    - If retriever returns 0 facts: Precision=1.0, Recall=1.0 (correct rejection).
    - If retriever returns >0 facts: Precision=0.0, Recall=0.0 (false positives / noise).
- In-scope query but retriever returns 0 facts:
    - Precision=0.0, Recall=0.0.

Usage:
    python ml_layer/gpt_module/evaluate_retrieval.py
    python ml_layer/gpt_module/evaluate_retrieval.py --top-k 3
    python ml_layer/gpt_module/evaluate_retrieval.py --output ml_layer/gpt_module/baseline_results.json
"""

import argparse
import inspect
import json
import os
import sys
from pathlib import Path

# Add project root to sys.path so ml_layer modules can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_dataset(dataset_path):
    """Load labeled evaluation dataset from JSON."""
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def calculate_metrics(retrieved_facts, expected_facts):
    """
    Calculate precision, recall, f1, hit, and reciprocal rank for a single query.
    """
    retrieved_set = set(retrieved_facts)
    expected_set = set(expected_facts)

    retrieved_count = len(retrieved_set)
    expected_count = len(expected_set)
    matched_count = len(retrieved_set.intersection(expected_set))

    # Out-of-scope / negative query (expected_facts is empty)
    if expected_count == 0:
        if retrieved_count == 0:
            # Correct rejection of irrelevant query
            precision = 1.0
            recall = 1.0
            f1 = 1.0
            hit = 1.0
            reciprocal_rank = 1.0
        else:
            # Retriever produced false positive noise
            precision = 0.0
            recall = 0.0
            f1 = 0.0
            hit = 0.0
            reciprocal_rank = 0.0
        return {
            "retrieved_count": retrieved_count,
            "expected_count": expected_count,
            "matched_count": matched_count,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "hit": hit,
            "reciprocal_rank": reciprocal_rank,
        }

    # In-scope query with positive expected facts
    if retrieved_count == 0:
        precision = 0.0
        recall = 0.0
        f1 = 0.0
        hit = 0.0
        reciprocal_rank = 0.0
    else:
        precision = matched_count / retrieved_count
        recall = matched_count / expected_count
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        hit = 1.0 if matched_count > 0 else 0.0

        # Calculate reciprocal rank of first relevant fact in ranked order
        reciprocal_rank = 0.0
        for rank, fact in enumerate(retrieved_facts, start=1):
            if fact in expected_set:
                reciprocal_rank = 1.0 / rank
                break

    return {
        "retrieved_count": retrieved_count,
        "expected_count": expected_count,
        "matched_count": matched_count,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "hit": hit,
        "reciprocal_rank": reciprocal_rank,
    }


def call_retriever(retriever_func, query, top_k=3, min_similarity=None):
    """
    Safely call retriever respecting its signature across baseline and improved versions.
    """
    sig = inspect.signature(retriever_func)
    kwargs = {}
    if "top_k" in sig.parameters:
        kwargs["top_k"] = top_k

    if min_similarity is not None and "similarity_threshold" in sig.parameters:
        kwargs["similarity_threshold"] = min_similarity
    elif min_similarity is not None and "min_similarity" in sig.parameters:
        kwargs["min_similarity"] = min_similarity

    return retriever_func(query, **kwargs)


def evaluate(dataset, top_k=3, min_similarity=None):
    """
    Runs evaluation against the existing retriever and computes aggregate metrics.
    """
    try:
        from ml_layer.gpt_module.knowledge_base import get_relevant_context, KNOWLEDGE_BASE
    except ImportError as e:
        print(f"\n[ERROR] Could not import knowledge_base: {e}")
        print("Please ensure sentence-transformers and project dependencies are available in the environment.\n")
        sys.exit(1)

    results = []
    print(f"\n{'=' * 85}")
    print(f"RETRIEVAL EVALUATION - WeightCoachAI (top_k={top_k}, min_similarity={min_similarity})")
    print(f"{'=' * 85}")
    print(f"{'ID':<9} | {'Exp':<3} | {'Ret':<3} | {'Match':<5} | {'Prec':<6} | {'Recall':<6} | {'F1':<6} | {'Hit':<4} | {'Query'}")
    print(f"{'-' * 85}")

    for item in dataset:
        qid = item["id"]
        query = item["query"]
        expected_facts = item.get("expected_facts", [])

        # Call existing retriever
        retrieved_facts = call_retriever(get_relevant_context, query, top_k=top_k, min_similarity=min_similarity)

        metrics = calculate_metrics(retrieved_facts, expected_facts)
        item_result = {
            "id": qid,
            "category": item.get("category", ""),
            "query": query,
            "expected_facts": expected_facts,
            "retrieved_facts": retrieved_facts,
            "metrics": metrics,
        }
        results.append(item_result)

        short_query = (query[:35] + "...") if len(query) > 35 else query
        print(
            f"{qid:<9} | {metrics['expected_count']:<3} | {metrics['retrieved_count']:<3} | "
            f"{metrics['matched_count']:<5} | {metrics['precision']:<6.2f} | {metrics['recall']:<6.2f} | "
            f"{metrics['f1']:<6.2f} | {int(metrics['hit']):<4} | {short_query}"
        )

    # Aggregated Summary
    total_queries = len(results)
    avg_precision = sum(r["metrics"]["precision"] for r in results) / total_queries
    avg_recall = sum(r["metrics"]["recall"] for r in results) / total_queries
    avg_f1 = sum(r["metrics"]["f1"] for r in results) / total_queries
    hit_rate = sum(r["metrics"]["hit"] for r in results) / total_queries
    mrr = sum(r["metrics"]["reciprocal_rank"] for r in results) / total_queries

    in_scope = [r for r in results if r["metrics"]["expected_count"] > 0]
    out_of_scope = [r for r in results if r["metrics"]["expected_count"] == 0]

    in_scope_prec = sum(r["metrics"]["precision"] for r in in_scope) / len(in_scope) if in_scope else 0.0
    in_scope_rec = sum(r["metrics"]["recall"] for r in in_scope) / len(in_scope) if in_scope else 0.0
    out_of_scope_prec = sum(r["metrics"]["precision"] for r in out_of_scope) / len(out_of_scope) if out_of_scope else 0.0

    summary = {
        "total_queries": total_queries,
        "in_scope_queries": len(in_scope),
        "out_of_scope_queries": len(out_of_scope),
        "top_k": top_k,
        "min_similarity": min_similarity,
        "mean_precision": round(avg_precision, 4),
        "mean_recall": round(avg_recall, 4),
        "mean_f1": round(avg_f1, 4),
        "hit_rate": round(hit_rate, 4),
        "mean_reciprocal_rank": round(mrr, 4),
        "in_scope_precision": round(in_scope_prec, 4),
        "in_scope_recall": round(in_scope_rec, 4),
        "out_of_scope_precision": round(out_of_scope_prec, 4),
    }

    print(f"{'=' * 85}")
    print("EVALUATION SUMMARY:")
    print(f"  Total Queries Evaluated:    {summary['total_queries']} ({summary['in_scope_queries']} in-scope, {summary['out_of_scope_queries']} out-of-scope)")
    print(f"  Mean Precision:             {summary['mean_precision']:.4f} ({summary['mean_precision'] * 100:.1f}%)")
    print(f"  Mean Recall:                {summary['mean_recall']:.4f} ({summary['mean_recall'] * 100:.1f}%)")
    print(f"  Mean F1 Score:              {summary['mean_f1']:.4f} ({summary['mean_f1'] * 100:.1f}%)")
    print(f"  Hit Rate:                   {summary['hit_rate']:.4f} ({summary['hit_rate'] * 100:.1f}%)")
    print(f"  Mean Reciprocal Rank (MRR): {summary['mean_reciprocal_rank']:.4f}")
    print(f"  In-Scope Precision:         {summary['in_scope_precision']:.4f}")
    print(f"  In-Scope Recall:            {summary['in_scope_recall']:.4f}")
    print(f"  Out-of-Scope Accuracy:      {summary['out_of_scope_precision']:.4f}")
    print(f"{'=' * 85}\n")

    return {"summary": summary, "results": results}


def main():
    parser = argparse.ArgumentParser(description="Evaluate WeightCoachAI Knowledge Base Retrieval")
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(Path(__file__).parent / "rag_evaluation_dataset.json"),
        help="Path to evaluation dataset JSON",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Number of facts to retrieve")
    parser.add_argument("--min-similarity", type=float, default=None, help="Optional similarity threshold")
    parser.add_argument("--output", type=str, default=None, help="Path to save evaluation results JSON")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"[ERROR] Dataset file not found at: {dataset_path}")
        sys.exit(1)

    dataset = load_dataset(dataset_path)
    eval_output = evaluate(dataset, top_k=args.top_k, min_similarity=args.min_similarity)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(eval_output, f, indent=2)
        print(f"[INFO] Results saved to: {output_path}")


if __name__ == "__main__":
    main()
