"""
test_analysis.py — Person B
Tests for modules/analysis.py
"""

import pytest
from modules.analysis import get_attention, classify_section, explain_concepts, SECTION_LABELS

SAMPLE_ABSTRACT = (
    "We propose a self-attention mechanism for sequence transduction tasks. "
    "The Transformer model relies entirely on attention, achieving state-of-the-art "
    "results on machine translation without recurrent or convolutional layers. "
    "Positional encoding is used to inject sequence order information."
)


# ---------------------------------------------------------------------------
# get_attention
# ---------------------------------------------------------------------------

def test_attention_returns_tokens_and_weights():
    result = get_attention(SAMPLE_ABSTRACT)
    assert "tokens" in result
    assert "weights" in result
    assert "note" in result


def test_attention_tokens_is_list_of_strings():
    result = get_attention(SAMPLE_ABSTRACT)
    tokens = result["tokens"]
    assert isinstance(tokens, list)
    assert all(isinstance(t, str) for t in tokens)


def test_attention_matrix_is_square():
    result = get_attention(SAMPLE_ABSTRACT)
    tokens = result["tokens"]
    weights = result["weights"]
    n = len(tokens)
    assert len(weights) == n, "weights outer length must equal token count"
    for row in weights:
        assert len(row) == n, "each row must equal token count"


def test_attention_rows_sum_to_approximately_one():
    result = get_attention(SAMPLE_ABSTRACT)
    for i, row in enumerate(result["weights"]):
        row_sum = sum(row)
        assert abs(row_sum - 1.0) < 0.05, f"Row {i} sums to {row_sum}, expected ~1.0"


def test_attention_truncated_to_max_tokens():
    from modules.analysis import MAX_TOKENS
    long_abstract = SAMPLE_ABSTRACT * 20  # force truncation
    result = get_attention(long_abstract)
    assert len(result["tokens"]) <= MAX_TOKENS


# ---------------------------------------------------------------------------
# classify_section
# ---------------------------------------------------------------------------

def test_classify_section_returns_valid_label():
    label = classify_section(SAMPLE_ABSTRACT)
    assert label in SECTION_LABELS, f"'{label}' is not a valid section label"


def test_classify_section_returns_string():
    label = classify_section(SAMPLE_ABSTRACT)
    assert isinstance(label, str)


def test_classify_methodology_abstract():
    methodology_text = (
        "In this paper, we propose a novel training procedure for neural networks. "
        "Our method introduces a new loss function and optimization strategy."
    )
    label = classify_section(methodology_text)
    assert label in SECTION_LABELS


def test_classify_results_abstract():
    results_text = (
        "Our experiments show a 5% improvement over the baseline on three benchmarks. "
        "The proposed method outperforms all prior approaches on GLUE and SQuAD."
    )
    label = classify_section(results_text)
    assert label in SECTION_LABELS


# ---------------------------------------------------------------------------
# explain_concepts
# ---------------------------------------------------------------------------

def test_explain_concepts_returns_list():
    concepts = explain_concepts(SAMPLE_ABSTRACT)
    assert isinstance(concepts, list)


def test_explain_concepts_returns_up_to_three():
    concepts = explain_concepts(SAMPLE_ABSTRACT)
    assert len(concepts) <= 3


def test_explain_concepts_each_has_term_and_explanation():
    concepts = explain_concepts(SAMPLE_ABSTRACT)
    for c in concepts:
        assert "term" in c, "Each concept must have a 'term' key"
        assert "explanation" in c, "Each concept must have an 'explanation' key"
        assert isinstance(c["term"], str)
        assert isinstance(c["explanation"], str)
        assert len(c["explanation"]) > 5, "Explanation should be non-trivial"