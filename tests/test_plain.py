"""Test class."""
import pandas as pd

from intervals.base import IntervalPlain


def test_all_adjacent():
    """Test adjacent check in plain form."""

    data = pd.DataFrame({"a": [0, 10, 20],
                         "b": [5, 20, 30]})

    intervals = IntervalPlain(data, "a", "b")
    assert intervals.all_adjacent()

    data = pd.DataFrame({"a": [0, 10, 20],
                         "b": [11, 20, 30]})

    intervals = IntervalPlain(data, "a", "b")
    assert not intervals.all_adjacent()


def test_where_intersection_cycle():
    """Test intersection in cycle in sequential form."""

    data_a = pd.DataFrame({"a": [0, 10, 20, 30],
                           "b": [5, 15, 25, 35]})
    data_x = pd.DataFrame({"x": [3, 22],
                           "y": [6, 32]})
    intervals_a = IntervalPlain(data_a, "a", "b")
    intervals_x = IntervalPlain(data_x, "x", "y")

    result = intervals_a.where_intersection_cycle(intervals_x)
    expected = pd.Series([True, False, True, True])
    assert expected.equals(result)
