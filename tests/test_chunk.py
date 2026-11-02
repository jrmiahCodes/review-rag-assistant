from review_rag_assistant.ingest.chunk import chunk_review, header, split_text


def words(text: str) -> int:
    return len(text.split())


def test_short_text_is_one_chunk():
    assert split_text("Great controller. Battery lasts long.", words, 50, 5) == [
        "Great controller. Battery lasts long."
    ]


def test_long_text_splits_on_sentences_within_budget():
    text = " ".join(f"Sentence number {i} has six words." for i in range(20))
    windows = split_text(text, words, max_tokens=20, overlap_tokens=6)
    assert len(windows) > 1
    assert all(words(w) <= 20 for w in windows)
    # every sentence survives somewhere
    for i in range(20):
        assert any(f"number {i} " in w for w in windows)


def test_windows_overlap():
    text = " ".join(f"S{i} a b c d e." for i in range(10))
    windows = split_text(text, words, max_tokens=18, overlap_tokens=6)
    last_sentence_of_first = windows[0].split(". ")[-1]
    assert windows[1].startswith(last_sentence_of_first.rstrip("."))


def test_oversized_sentence_becomes_its_own_window():
    huge = "word " * 100
    windows = split_text(f"Short one. {huge.strip()}. Another short.", words, max_tokens=20, overlap_tokens=5)
    assert len(windows) == 3


def test_empty_text_gives_no_chunks():
    assert split_text("   \n ", words, 50, 5) == []


def test_header_names_product_and_truncates_long_titles():
    h = header("X" * 50 + " " + "Y" * 100, "PlayStation 4", "Accessories", 4.0, "Solid")
    assert h.splitlines()[0].endswith("...")
    assert "Category: PlayStation 4 Accessories" in h
    assert "Review (4/5 stars): Solid" in h


def test_chunk_ids_are_stable_and_search_text_has_header():
    chunks = chunk_review("r1", "Meh", "It broke. Twice.", 2.0, "Pad", "PC", "Accessories", words, 50, 5)
    assert [c.chunk_id for c in chunks] == ["r1-0"]
    assert chunks[0].search_text.startswith("Product: Pad")
    assert chunks[0].text == "It broke. Twice."
