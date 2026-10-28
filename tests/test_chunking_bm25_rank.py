from app.indexing.bm25 import BM25Index
from app.indexing.hybrid import fuse_candidates, select_diverse
from app.indexing.hybrid import Candidate
from app.ingestion.chunking import chunk_fixed_window, chunk_sentence_pack
from app.textutil import hamming, simhash


def test_two_chunkers_disagree_on_a_long_passage():
    text = " ".join(f"Sentence number {i} discusses ledger-service behavior in detail." for i in range(12))
    packed = chunk_sentence_pack(text, size=180, overlap_sentences=1)
    windowed = chunk_fixed_window(text, size=90, overlap=20)
    assert len(windowed) > len(packed)
    assert packed[0].text != windowed[0].text


def test_bm25_prefers_the_document_that_mentions_the_query_terms():
    index = BM25Index()
    index.add("a", "ledger-service retry count was reduced from 5 to 1", "sentence_pack")
    index.add("b", "the bakery opened early and sold bread", "sentence_pack")
    index.add("c", "merchant notifier degraded during the payment incident", "sentence_pack")
    ranked = index.search("ledger-service retry count", k=3, strategy="sentence_pack")
    assert ranked[0][0] == "a"
    assert index.search("ledger-service retry count", k=3, strategy="fixed_window") == []


def test_reciprocal_rank_fusion_rewards_agreement():
    semantic = [("a", 0.9), ("b", 0.8)]
    lexical = [("b", 4.0), ("c", 1.0)]
    fused = fuse_candidates(semantic, lexical, rrf_k=60)
    assert fused["b"].fused > fused["a"].fused
    assert fused["c"].semantic_rank is None


def test_diversity_selection_can_prefer_a_new_document():
    first = Candidate(chunk_id="1", doc_id="doc-a", simhash=1, fused=0.05)
    second = Candidate(chunk_id="2", doc_id="doc-a", simhash=1, fused=0.049)
    third = Candidate(chunk_id="3", doc_id="doc-b", simhash=2**40, fused=0.046)
    chosen = select_diverse([first, second, third], k=2, redundancy_penalty=0.01, diversity_bonus=0.004)
    assert [item.chunk_id for item in chosen] == ["1", "3"]


def test_simhash_is_stable_and_close_for_a_one_word_edit():
    base = "ledger service payment delay " * 40
    edited = base.replace("delay", "delays", 1)
    assert simhash(base) == simhash(base)
    assert hamming(simhash(base), simhash(edited)) <= 5
