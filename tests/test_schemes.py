import pandas as pd

from movie_sentiment.data.schemes import apply_scheme, map_labels_for_scheme


def test_three_class_mapping_is_correct():
    y = pd.Series([0, 1, 2, 3, 4])
    mapped = map_labels_for_scheme(y, scheme="three").tolist()
    assert mapped == [0, 0, 1, 2, 2]


def test_two_class_mapping_drops_neutral():
    df = pd.DataFrame({"Sentiment": [0, 1, 2, 3, 4], "Phrase": ["a", "b", "c", "d", "e"]})
    out = apply_scheme(df, label_col="Sentiment", scheme="two")
    assert sorted(out["Label"].unique().tolist()) == [0, 1]
    assert len(out) == 4
