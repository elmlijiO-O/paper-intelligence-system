import pytest
from modules.nlp import summarize, extract_keywords

SAMPLE_ABSTRACT = (
    "We introduce a new language representation model called BERT, which stands for "
    "Bidirectional Encoder Representations from Transformers. Unlike recent language "
    "representation models, BERT is designed to pre-train deep bidirectional representations "
    "from unlabeled text by jointly conditioning on both left and right context in all layers. "
    "As a result, the pre-trained BERT model can be fine-tuned with just one additional output "
    "layer to create state-of-the-art models for a wide range of tasks."
)


def test_summarize_returns_string():
    result = summarize(SAMPLE_ABSTRACT)
    assert isinstance(result, str)
    assert len(result) > 20


def test_summarize_shorter_than_input():
    result = summarize(SAMPLE_ABSTRACT)
    assert len(result) < len(SAMPLE_ABSTRACT)


def test_extract_keywords_returns_list():
    keywords = extract_keywords(SAMPLE_ABSTRACT)
    assert isinstance(keywords, list)
    assert len(keywords) > 0


def test_extract_keywords_count():
    keywords = extract_keywords(SAMPLE_ABSTRACT, n=10)
    assert len(keywords) <= 10


def test_extract_keywords_are_strings():
    keywords = extract_keywords(SAMPLE_ABSTRACT)
    for kw in keywords:
        assert isinstance(kw, str) and len(kw) > 0


def test_summarize_short_abstract():
    short = "This paper proposes a new method."
    result = summarize(short)
    assert isinstance(result, str)