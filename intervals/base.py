"""Base interval class."""
import pandas as pd
import numpy as np


class IntervalPlain:
    """
    Plain interval data.

    Interval data with columns:
    `a` - start of the intervals
    `b` - end of the intervals
    """

    def __init__(self,
                 data: pd.DataFrame,
                 a: str,  # pylint: disable=invalid-name
                 b: str):  # pylint: disable=invalid-name

        self._a = a
        self._b = b
        self._data = data.sort_values(self._a)
        self._data = self._data.reset_index(drop=True)

    @property
    def data(self) -> pd.DataFrame:
        "Return data."
        return self._data

    @property
    def start(self) -> str:
        """Start column name."""
        return self._a

    @property
    def end(self) -> str:
        """End column name."""
        return self._b

    @property
    def a(self) -> str:  # pylint: disable=invalid-name
        """Start column name."""
        return self._a

    @property
    def b(self) -> str:  # pylint: disable=invalid-name
        """End column name."""
        return self._b

    def all_adjacent(self) -> bool:
        """Check intervals are adjacent."""
        next_start = self._data[self._a].shift(-1)
        check_end_lower_next_start = self._data[self._b] <= next_start
        return check_end_lower_next_start.iloc[:-1].all()

    def where_intersection_cycle(self, other: "IntervalPlain") -> pd.Series:
        """Get boolean series where intervals intersect with other data."""
        where_intersect = pd.Series(0, dtype=bool, index=self._data.index)
        for other_a, other_b in zip(other.data[other.a], other.data[other.b]):
            where_other_lower = self._data[self.a] > other_b
            where_other_higher = self._data[self.b] < other_a
            where_current = ~(where_other_lower | where_other_higher)
            where_intersect |= where_current

        return where_intersect

    def intersect(self, other: "IntervalPlain") -> "IntervalPlain":
        """Intersect data."""
        where = self.where_intersection_cycle(other)
        return IntervalPlain(self.data[where], a=self.a, b=self.b)


class IntervalSequential:
    """
    Interval data in sequential form.

    Columns are:
    `x` - column with intervals starts and ends values
    `id` - interval identificator.
    `start` - integer value 1 and 0 where 1 is start and 0 is interval end
    """

    def __init__(self,
                 data: pd.DataFrame,
                 x: str,  # pylint: disable=invalid-name
                 id_: str,
                 start: str):

        self._x = x
        self._id = id_
        self._start = start
        self._data = data
        self._data = self._data.sort_values(self.x)
        self._data[id_] = self._data[id_].astype("Int64")
        self._data[start] = self._data[start].astype("Int64")

    @property
    def data(self) -> pd.DataFrame:
        """Return data"""
        return self._data

    @property
    def x(self) -> str:  # pylint: disable=invalid-name
        """Column name with interval endpoints values."""
        return self._x

    @property
    def id_(self) -> str:
        """Columns name with interval identificators."""
        return self._id

    @property
    def start(self) -> str:
        """Column name with start and end indicator."""
        return self._start

    @classmethod
    def from_plain(cls,
                   plain: IntervalPlain,
                   x: str = "x",  # pylint: disable=invalid-name
                   id_: str = "id",
                   start: str = "start") -> "IntervalSequential":
        """Convert interval plain form to sequential."""
        data = plain.data.sort_values(plain.a)
        data = data.reset_index(drop=True)

        a_series = data[plain.a]
        b_series = data[plain.b]
        sequence = pd.concat([a_series, b_series], axis=0)
        sequence = sequence.to_frame(x)
        sequence.index.name = id_
        sequence = sequence.reset_index()
        sequence = sequence.sort_values(by=[x, id_])
        sequence[start] = sequence[id_].diff().fillna(1)
        sequence = sequence.reset_index(drop=True)
        return cls(sequence, x=x, id_=id_, start=start)

    def to_plain(self, a: str, b: str) -> IntervalPlain:
        """Convert to plain intervals."""
        values = self.data[self.x].values.reshape(int(self.data.shape[0] / 2), 2)
        data = pd.DataFrame(values, columns=[a, b])
        return IntervalPlain(data, a=a, b=b)

    def intersection_ids(self, other: "IntervalSequential") -> np.ndarray:
        """
        Returns identificators of intervals which intersect with other data.
        """
        suffix = "_other"
        other_id = self.id_ + suffix
        other_start = self.start + suffix
        other_data = other.data[[other.x, other.id_, other.start]]
        other_data = other_data.rename(
            columns={other.x: self.x,
                     other.id_: other_id,
                     other.start: other_start}
        )

        merged = self.data.merge(other_data,
                                 on=self.x,
                                 how="outer")

        merged = merged.sort_values(by=[self._x, other_id])

        filter_series = merged[other_start].fillna(method="ffill")
        filter_series += filter_series.shift(1).fillna(0)
        start_series = merged[self.start].fillna(method="ffill")
        start_series += start_series.shift(1).fillna(0)

        merged[self._id] = merged[self._id].fillna(method="ffill")
        merged = merged[(filter_series > 0) & (start_series > 0)]
        filter_ids = merged[self.id_].unique()

        return filter_ids

    def intersection(self, other: "IntervalSequential") -> "IntervalSequential":
        """Get interval in sequential form which intersect with other data."""
        filter_ids = self.intersection_ids(other)

        filter_data = self.data.set_index(self.id_)
        filter_data = filter_data.loc[filter_ids]

        return IntervalSequential(data=filter_data.reset_index(),
                                  x=self.x,
                                  id_=self.id_,
                                  start=self.start)
