# Glossary

Terms as this codebase uses them. When code, docs, and evals disagree, this file wins.

**Product.** One Amazon listing, keyed by `parent_asin` (the ID shared by all variants: colors, editions). Stored in the `Product` Weaviate collection with its title, features, description, price, rating, platform, and product type.

**Review.** One customer review of a product, keyed by `review_id`, an MD5 of user, product, and timestamp. It's stable across reruns, which the evals rely on.

**Chunk.** A piece of a review sized for retrieval (about 300 tokens). Most reviews are one chunk. Keyed by `chunk_id` = `<review_id>-<index>`. Stored in the `ReviewChunk` collection with product metadata copied on.

**Search text.** What gets embedded and BM25-indexed: a short header (product title, category, star rating, review title) plus the chunk text. The header lets a review that never names the product still match a search for it.

**Platform / product type.** Filter fields derived from Amazon's category path, e.g. `PlayStation 4` / `Accessories`. Amazon sometimes mislabels these (a Switch adapter tagged `Wii U`), so filtered searches can miss correct items.

**Selection.** The DuckDB step that reduces the raw category to the indexed corpus: priced items with at least 50 ratings, top 4,000 by rating count, up to 20 reviews each.

**First stage.** The cheap, wide search in Weaviate that returns about 30 candidates: `bm25`, `vector`, or `hybrid` (both, fused by relative score with weight `alpha`).

**Rerank.** The second stage. A cross-encoder (`bge-reranker-v2-m3`) scores each (question, candidate) pair together and reorders the shortlist.

**Router.** The LLM call that turns a question into tool calls: `search_products` (structured filters and sorting) and/or `search_reviews` (opinions and experiences). It falls back to `search_reviews` with the raw question if it fails.

**Source.** A retrieved product or chunk shown to the generator as a numbered block `[n]`. Answers cite sources by number.

**Generator.** The model that writes the final answer. Spec format `<provider>:<model>`, e.g. `ollama:qwen3.5:4b`.

**Golden set.** Hand-reviewed eval questions in `eval/golden.jsonl`. Each lists the reviews and products that answer it, plus reference notes. Types: `review_fact`, `product_opinion`, `product_search`, `comparison`, `unanswerable`.

**Strict / lenient retrieval.** Strict: the exact review a question was written from was retrieved. Lenient: any review or listing of the right product was retrieved.

**Judge.** The Claude model that grades generated answers on faithfulness, correctness, and citation quality against the sources the generator saw.
