# Glossary

Terms as this codebase uses them. When code, docs, and evals disagree, this file wins.

**Product.** One Amazon listing, keyed by `parent_asin`, the ID that all variants of a listing share. Stored in the `Product` Weaviate collection with its title, features, description, price, rating, platform, and product type.

**Review.** One customer review of a product, keyed by `review_id`, an MD5 of the user, product, and timestamp. The ID stays the same across reruns, and the evals depend on that.

**Chunk.** A piece of a review sized for retrieval, about 300 tokens. Most reviews are one chunk. Keyed by `chunk_id`, which is `<review_id>-<index>`. Stored in the `ReviewChunk` collection with the product's metadata copied on.

**Search text.** What gets embedded and BM25-indexed. It's a short header with the product title, category, star rating, and review title, followed by the chunk text. The header lets a review that never names the product still match a search for it.

**Platform / product type.** Filter fields taken from Amazon's category path, e.g. `PlayStation 4` and `Accessories`. Amazon mislabels some products, such as a Switch adapter tagged `Wii U`, so a filtered search can miss correct items.

**Selection.** The DuckDB step that turns the raw category into the indexed corpus. It keeps priced items with at least 50 ratings, the top 4,000 by rating count, and up to 20 reviews each.

**First stage.** The cheap, wide search in Weaviate that returns about 30 candidates. It runs in one of three modes: `bm25`, `vector`, or `hybrid`, which fuses the other two by relative score, weighted by `alpha`.

**Rerank.** The second stage. A cross-encoder, `bge-reranker-v2-m3`, scores each question and candidate pair together and reorders the shortlist.

**Router.** The LLM call that turns a question into tool calls, `search_products` for structured filters and sorting and `search_reviews` for opinions and experiences. If it fails, the pipeline calls `search_reviews` with the raw question.

**Source.** A retrieved product or chunk, shown to the generator as a numbered block like `[3]`. Answers cite sources by number.

**Generator.** The model that writes the final answer, named as `<provider>:<model>`, e.g. `ollama:qwen3.5:4b`.

**Golden set.** The hand-reviewed eval questions in `eval/golden.jsonl`. Each lists the reviews and products that answer it, plus reference notes. The types are `review_fact`, `product_opinion`, `product_search`, `comparison`, and `unanswerable`.

**Strict / lenient retrieval.** Strict means the exact review a question came from was retrieved. Lenient means any review or listing of the right product was retrieved.

**Judge.** The Claude model that grades generated answers on faithfulness, correctness, and citation quality, against the same sources the generator saw.
