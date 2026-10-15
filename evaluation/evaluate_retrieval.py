"""Compare SuperMew retrieval strategies on the checked-in JSONL dataset.

Run from the repository root:
    .venv\\Scripts\\python.exe evaluation\\evaluate_retrieval.py --top-k 5
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from embedding import embedding_service
from milvus_client import MilvusManager
from rag_utils import RERANK_API_KEY, RERANK_BINDING_HOST, RERANK_MODEL, _rerank_documents


def load_dataset(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def retrieve(query: str, mode: str, top_k: int) -> list[dict]:
    manager = MilvusManager()
    filter_expr = "chunk_level == 3"
    if mode == "dense":
        return manager.dense_retrieve(embedding_service.get_embeddings([query])[0], top_k, filter_expr)
    if mode == "sparse":
        return manager.sparse_retrieve(embedding_service.get_sparse_embedding(query), top_k, filter_expr)
    dense = embedding_service.get_embeddings([query])[0]
    sparse = embedding_service.get_sparse_embedding(query)
    docs = manager.hybrid_retrieve(dense, sparse, max(top_k * 3, top_k), filter_expr=filter_expr)
    if mode == "hybrid_rerank":
        docs, _ = _rerank_documents(query, docs, top_k)
        return docs
    return docs[:top_k]


def score_case(case: dict, docs: list[dict], top_k: int) -> dict:
    docs = docs[:top_k]
    relevant_pages = set(case.get("relevant_pages", []))
    pages = [int(doc.get("page_number", 0)) + 1 for doc in docs]
    first_rank = next((rank for rank, page in enumerate(pages, 1) if page in relevant_pages), None)
    text = "\n".join(str(doc.get("text", "")).lower() for doc in docs)
    keywords = [str(item).lower() for item in case.get("relevant_keywords", [])]
    covered = sum(keyword in text for keyword in keywords)
    return {
        "id": case["id"],
        "hit": bool(first_rank),
        "rrf": 1 / first_rank if first_rank else 0.0,
        "keyword_coverage": covered / len(keywords) if keywords else 0.0,
        "pages": pages,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(Path(__file__).with_name("dataset.jsonl")))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", default="evaluation/results.json")
    args = parser.parse_args()

    cases = load_dataset(Path(args.dataset))
    all_results = {}
    for mode in ("dense", "sparse", "hybrid", "hybrid_rerank"):
        if mode == "hybrid_rerank" and not (RERANK_MODEL and RERANK_API_KEY and RERANK_BINDING_HOST):
            all_results[mode] = {
                "status": "not_configured",
                "message": "RERANK_MODEL/RERANK_BINDING_HOST/RERANK_API_KEY 未完整配置，未执行 Rerank。",
                "top_k": args.top_k,
                "cases": [],
            }
            continue
        started = time.perf_counter()
        case_results = []
        for case in cases:
            try:
                docs = retrieve(case["question"], mode, args.top_k)
                case_results.append(score_case(case, docs, args.top_k))
            except Exception as exc:
                case_results.append({"id": case["id"], "error": str(exc), "hit": False, "rrf": 0.0, "keyword_coverage": 0.0})
        valid = len(case_results)
        all_results[mode] = {
            "top_k": args.top_k,
            "hit_rate": sum(item["hit"] for item in case_results) / valid,
            "mrr": sum(item["rrf"] for item in case_results) / valid,
            "keyword_coverage": sum(item["keyword_coverage"] for item in case_results) / valid,
            "avg_latency_ms": (time.perf_counter() - started) * 1000 / valid,
            "cases": case_results,
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    for mode, result in all_results.items():
        if result.get("status") == "not_configured":
            print(f"{mode:15} NOT CONFIGURED - {result['message']}")
            continue
        print(f"{mode:15} Hit@{args.top_k}={result['hit_rate']:.3f}  MRR={result['mrr']:.3f}  "
              f"Keyword={result['keyword_coverage']:.3f}  Latency={result['avg_latency_ms']:.1f}ms")


if __name__ == "__main__":
    main()
