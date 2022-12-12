"""Test class."""
import pandas as pd

from intervals.base import IntervalPlain, IntervalSequential


def test_all_adjacent():
    """Test sequential form creation from plain."""

    data = pd.DataFrame({"a": [0, 10, 20],
                         "b": [5, 20, 30]})

    plain = IntervalPlain(data, "a", "b")
    sequential = IntervalSequential.from_plain(plain, "x", "id", "start")

    expected = pd.DataFrame({"id": [0, 0, 1, 1, 2, 2],
                             "x": [0, 5, 10, 20, 20, 30],
                             "start": [1, 0, 1, 0, 1, 0]})
    expected = expected.astype({"id": "Int64", "start": "Int64"})

    assert expected.equals(sequential.data)


def test_intersection():
    """Test intersection method in sequential form."""

    data_a = pd.DataFrame({"a": [0, 10, 20, 30],
                           "b": [5, 15, 25, 35]})
    data_x = pd.DataFrame({"x": [17, 22, 100],
                           "y": [18, 32, 110]})
    plain_a = IntervalPlain(data_a, "a", "b")
    plain_x = IntervalPlain(data_x, "x", "y")

    sequential_a = IntervalSequential.from_plain(plain_a, "a", "id")
    sequential_x = IntervalSequential.from_plain(plain_x, "x", "n")

    result = sequential_a.intersection(sequential_x)

    expected = pd.DataFrame({"id": [2, 2, 3, 3],
                             "a": [20, 25, 30, 35],
                             "start": [1, 0, 1, 0]})
    expected = expected.astype({"id": "Int64", "start": "Int64"})

    assert expected.equals(result.data)
