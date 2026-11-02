from review_rag_assistant.answer import PRODUCT_FILTERS, REVIEW_FILTERS, _clean_filters
from review_rag_assistant.evals import _arg_matches, citation_check, constraint_check, hit_at, reciprocal_rank
from review_rag_assistant.store import REVIEW, Filters


def test_reciprocal_rank_uses_first_relevant_position():
    assert reciprocal_rank(["a", "b", "c"], {"c"}) == 1 / 3
    assert reciprocal_rank(["a", "b"], {"z"}) == 0.0
    assert reciprocal_rank(["x", "x", "a"], {"x", "a"}) == 1.0


def test_hit_at_respects_cutoff():
    assert hit_at(["a", "b", "c"], {"c"}, 2) == 0.0
    assert hit_at(["a", "b", "c"], {"c"}, 3) == 1.0


def test_citation_check_flags_numbers_outside_sources():
    assert citation_check("Good [1][2], bad [7].", n_sources=3) == {"n_citations": 3, "invalid_citations": 1}
    assert citation_check("No citations here.", n_sources=3) == {"n_citations": 0, "invalid_citations": 0}


def test_constraint_check_counts_products_within_price():
    item = {"constraints": {"max_price": 30}}
    sources = [
        {"kind": "product", "meta": {"price": 20.0}},
        {"kind": "product", "meta": {"price": 45.0}},
        {"kind": "review", "meta": {"price": 99.0}},  # reviews don't count
    ]
    assert constraint_check(item, sources) == 0.5
    assert constraint_check({}, sources) is None


def test_arg_matches_numbers_and_case_insensitive_strings():
    assert _arg_matches(60, "60.0")
    assert not _arg_matches(60, 59)
    assert _arg_matches("PlayStation 5", "playstation 5")
    assert not _arg_matches(30, None)


def test_clean_filters_drops_unknown_empty_and_bad_values():
    f = _clean_filters(
        {"query": "x", "max_price": "45", "platform": "", "min_average_rating": "high", "sort_by": "rating"},
        PRODUCT_FILTERS,
    )
    assert f == Filters(max_price=45.0)


def test_review_filters_build_weaviate_filter_only_when_set():
    assert _clean_filters({"query": "x"}, REVIEW_FILTERS).to_weaviate(REVIEW) is None
    assert _clean_filters({"max_review_rating": 2}, REVIEW_FILTERS).to_weaviate(REVIEW) is not None
