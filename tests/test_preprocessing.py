import pandas as pd

from movie_sentiment.data.preprocessing import TextPreprocessor


def test_clean_removes_punctuation_and_stopwords():
    pre = TextPreprocessor()
    text = "This movie is absolutely AMAZING, and it's not boring at all!!!"
    cleaned = pre.clean(text)

    assert "movie" in cleaned
    assert "amaz" in cleaned or "amazing" in cleaned
    assert "!" not in cleaned


def test_transform_series_returns_same_length():
    pre = TextPreprocessor()
    series = pd.Series(["good movie", "bad movie", None])
    out = pre.transform_series(series)

    assert len(out) == 3
