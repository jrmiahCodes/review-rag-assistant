"""Evaluation: retrieval metrics (free, deterministic) and generation quality (LLM judge).

Golden items (eval/golden.jsonl) carry the reviews and products that answer them.
Retrieval is scored two ways:
- strict: did we retrieve the exact review the question was written from?
- lenient: did we retrieve any review/listing of the right product?
"""

import json
import re
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import weaviate
from pydantic import BaseModel, Field

from review_rag_assistant import store
from review_rag_assistant.config import ROOT, settings
from review_rag_assistant.retrieve import RetrievalConfig, retrieve

EVAL_DIR = ROOT / "eval"
RESULTS_DIR = EVAL_DIR / "results"

RETRIEVAL_CONFIGS = {
    "bm25": RetrievalConfig(mode="bm25", rerank=False),
    "vector": RetrievalConfig(mode="vector", rerank=False),
    "hybrid": RetrievalConfig(mode="hybrid", rerank=False),
    "hybrid+rerank": RetrievalConfig(mode="hybrid", rerank=True),
}


def load_golden(path: Path | None = None) -> list[dict]:
    path = path or (EVAL_DIR / "golden.jsonl")
    if not path.exists():
        path = EVAL_DIR / "golden_candidates.jsonl"
    items = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [i for i in items if i.get("keep", True)]


# ---------- retrieval ----------


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    for i, x in enumerate(ranked, 1):
        if x in relevant:
            return 1.0 / i
    return 0.0


def hit_at(ranked: list[str], relevant: set[str], k: int) -> float:
    return float(any(x in relevant for x in ranked[:k]))


def retrieval_eval(client: weaviate.WeaviateClient, items: list[dict], k: int = 10) -> dict:
    """Score each retrieval config on items that have known relevant sources."""
    rows = []
    for name, cfg in RETRIEVAL_CONFIGS.items():
        for item in items:
            rids, asins = set(item["relevant_review_ids"]), set(item["relevant_asins"])
            if not rids and not asins:
                continue
            collection = store.PRODUCT if item["type"] == "product_search" else store.REVIEW
            t = time.perf_counter()
            res = retrieve(client, collection, item["question"], k=k, config=cfg)
            ms = (time.perf_counter() - t) * 1000
            ranked_asins = [h.properties["parent_asin"] for h in res.hits]
            row = {"config": name, "id": item["id"], "type": item["type"], "latency_ms": ms}
            row["lenient_hit@5"] = hit_at(ranked_asins, asins, 5)
            row["lenient_mrr"] = reciprocal_rank(ranked_asins, asins)
            if rids:
                ranked_rids = [h.properties["review_id"] for h in res.hits]
                row["strict_hit@5"] = hit_at(ranked_rids, rids, 5)
                row["strict_hit@10"] = hit_at(ranked_rids, rids, 10)
                row["strict_mrr"] = reciprocal_rank(ranked_rids, rids)
            rows.append(row)

    summary = {}
    for name in RETRIEVAL_CONFIGS:
        mine = [r for r in rows if r["config"] == name]
        strict = [r for r in mine if "strict_mrr" in r]
        summary[name] = {
            "n_strict": len(strict),
            "strict_hit@5": _mean(r["strict_hit@5"] for r in strict),
            "strict_hit@10": _mean(r["strict_hit@10"] for r in strict),
            "strict_mrr": _mean(r["strict_mrr"] for r in strict),
            "n_lenient": len(mine),
            "lenient_hit@5": _mean(r["lenient_hit@5"] for r in mine),
            "lenient_mrr": _mean(r["lenient_mrr"] for r in mine),
            "p50_latency_ms": statistics.median(r["latency_ms"] for r in mine) if mine else 0.0,
        }
    return {"summary": summary, "rows": rows}


def _mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


# ---------- generation ----------

CITATION = re.compile(r"\[(\d+)\]")


def citation_check(answer: str, n_sources: int) -> dict:
    cited = [int(c) for c in CITATION.findall(answer)]
    return {
        "n_citations": len(cited),
        "invalid_citations": sum(1 for c in cited if c < 1 or c > n_sources),
    }


def constraint_check(item: dict, sources: list[dict]) -> float | None:
    """Fraction of returned products that satisfy the question's stated price limit."""
    max_price = (item.get("constraints") or {}).get("max_price")
    products = [s for s in sources if s["kind"] == "product"]
    if max_price is None or not products:
        return None
    return sum(1 for s in products if s["meta"]["price"] <= max_price) / len(products)


class Verdict(BaseModel):
    faithfulness: int = Field(
        description="1-5. 5 = every claim is supported by the cited sources; 1 = mostly unsupported or contradicted."
    )
    correctness: int = Field(
        description="1-5. How well the answer addresses the question compared with the reference notes."
    )
    citation_quality: int = Field(description="1-5. 5 = claims carry citations and each citation supports its claim.")
    abstained: bool = Field(description="True if the answer says the sources don't contain the answer.")
    reasoning: str = Field(description="One or two sentences explaining the scores.")


JUDGE_SYSTEM = """You grade answers from a retrieval-augmented shopping assistant. The assistant must answer only from its numbered sources and cite them like [1].

Grade strictly against the sources shown, not your own knowledge of the products. An answer that adds correct facts not found in the sources is still unfaithful.
For questions marked unanswerable, a good answer clearly says the sources don't cover it; inventing an answer scores 1 on correctness and faithfulness."""


def judge(item: dict, result: dict, model: str | None = None) -> Verdict:
    import anthropic

    from review_rag_assistant.answer import Source, format_context

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    sources = [Source(**s) for s in result.get("sources", [])]
    prompt = (
        f"QUESTION: {item['question']}\n"
        f"ANSWERABLE FROM DATA: {item['answerable']}\n"
        f"REFERENCE NOTES: {item['reference']}\n\n"
        f"SOURCES GIVEN TO THE ASSISTANT:\n{format_context(sources) or '(none)'}\n\n"
        f"ASSISTANT ANSWER:\n{result['answer']}"
    )
    res = client.messages.parse(
        model=model or settings.judge_model,
        max_tokens=2000,
        system=JUDGE_SYSTEM,
        output_config={"effort": "medium"},
        messages=[{"role": "user", "content": prompt}],
        output_format=Verdict,
    )
    return res.parsed_output


@dataclass
class GenRow:
    id: str
    type: str
    answerable: bool
    answer: str
    n_sources: int
    n_citations: int
    invalid_citations: int
    constraint_ok: float | None
    first_token_ms: float
    total_ms: float
    input_tokens: int
    output_tokens: int
    faithfulness: int | None = None
    correctness: int | None = None
    citation_quality: int | None = None
    abstained: bool | None = None
    judge_reasoning: str | None = None


def generation_eval(
    client: weaviate.WeaviateClient, items: list[dict], generator: str | None = None, use_judge: bool = True
) -> dict:
    from review_rag_assistant.answer import answer, router_tools
    from review_rag_assistant.llm import get_llm

    llm = get_llm(generator)
    tools = router_tools(store.distinct_values(client, "platform"), store.distinct_values(client, "product_type"))
    rows: list[GenRow] = []
    for item in items:
        res = answer(item["question"], llm, client, tools)
        cc = citation_check(res["answer"], len(res["sources"]))
        row = GenRow(
            id=item["id"],
            type=item["type"],
            answerable=item["answerable"],
            answer=res["answer"],
            n_sources=len(res["sources"]),
            constraint_ok=constraint_check(item, res["sources"]),
            first_token_ms=res["timings_ms"]["first_token"],
            total_ms=res["timings_ms"]["total"],
            input_tokens=res["usage"]["input_tokens"],
            output_tokens=res["usage"]["output_tokens"],
            **cc,
        )
        if use_judge:
            v = judge(item, res)
            row.faithfulness, row.correctness, row.citation_quality = v.faithfulness, v.correctness, v.citation_quality
            row.abstained, row.judge_reasoning = v.abstained, v.reasoning
        rows.append(row)

    judged = [r for r in rows if r.faithfulness is not None]
    unans = [r for r in judged if not r.answerable]
    ans = [r for r in judged if r.answerable]
    constraints = [r.constraint_ok for r in rows if r.constraint_ok is not None]
    summary = {
        "generator": llm.name,
        "n": len(rows),
        "faithfulness": _mean(r.faithfulness for r in judged),
        "correctness": _mean(r.correctness for r in judged),
        "citation_quality": _mean(r.citation_quality for r in judged),
        "abstain_rate_unanswerable": _mean(r.abstained for r in unans),
        "false_abstain_rate_answerable": _mean(r.abstained for r in ans),
        "invalid_citation_rate": _mean(r.invalid_citations > 0 for r in rows),
        "uncited_answer_rate": _mean(r.n_citations == 0 for r in rows if r.answerable),
        "price_constraint_ok": _mean(constraints) if constraints else None,
        "p50_first_token_ms": statistics.median(r.first_token_ms for r in rows),
        "p50_total_ms": statistics.median(r.total_ms for r in rows),
        "mean_input_tokens": _mean(r.input_tokens for r in rows),
        "mean_output_tokens": _mean(r.output_tokens for r in rows),
    }
    return {"summary": summary, "rows": [asdict(r) for r in rows]}


def save(kind: str, label: str, result: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe = re.sub(r"[^a-zA-Z0-9.+-]", "_", label)
    path = RESULTS_DIR / f"{stamp}_{kind}_{safe}.json"
    path.write_text(json.dumps(result, indent=2, default=str))
    return path
