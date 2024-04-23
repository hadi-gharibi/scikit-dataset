from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Generator, Optional, cast

import numpy as np
from sklearn.utils import _safe_indexing  # type: ignore

if TYPE_CHECKING:
    from skdataset.types import IndexType, Int, SeqIndexType, SpliterType


class Dataset(dict):
    KEYWORDS = {"name", "description", "metadata"}

    def __init__(
        self,
        *,
        name: Optional[str] = "data",
        description: Optional[str] = None,
        metadata: Optional[dict] = None,
        **kwargs,
    ):
        self.name = name
        self.description = description
        self.metadata = metadata or {}
        kwargs = {k: self._fix_scalar(v) for k, v in kwargs.items()}
        super().__init__(kwargs)
        self._check_matching_sizes()

    def _fix_scalar(self, value):
        if hasattr(value, "shape") and value.shape == ():
            return value.reshape(
                1,
            )
        elif not hasattr(value, "__len__"):
            return [value]
        return value

    def _check_matching_sizes(self):
        """Check if all data variables have the same number of rows.

        Raises
        ------
        ValueError
            If the data variables have different number of rows.

        """
        sizes = [len(v) for v in self.values()]
        if len(set(sizes)) > 1:
            all_sizes = {k: len(v) for k, v in self.items()}

            raise ValueError(f"All data variables must have the same number of rows. Got {all_sizes}. {sizes}")

    def __getattr__(self, name):
        """Return the value of the attribute with the given name.

        This method is called when an attribute is accessed that does not exist in the object's dictionary.

        Parameters
        ----------
        name : str
            The name of the attribute.

        Returns
        -------
        Any
            The value of the attribute.

        Raises
        ------
        AttributeError
            If the attribute does not exist.

        """
        if name in self:
            return self.get(name)
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name not in self.KEYWORDS:
            self[name] = value
        else:
            super().__setattr__(name, value)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self._check_matching_sizes()

    def __str__(self):
        """Return a string representation of the Dataset.

        Returns
        -------
        str
            A string representation of the Dataset object, including the name, number of rows, and number of variables.
        """
        return f"""Dataset(name={self.name}, description={self.description}, metadata={self.metadata})
            - '{self.name}' dataaset has {len(self)} rows and {self.columns} variables."""

    def at(self, index: Int, column: str) -> Any:
        """Access a single value for a row/column label pair.


        Parameters
        ----------
        index : int
            The index of the row.
        column : str
            The name of the column.

        Returns
        -------
        Any
            The value at the specified index and column.
        """
        return self.__getitem__(index).__getitem__(column)

    def iloc(self, index: Int | SeqIndexType, axis: int = 0) -> Dataset:
        """
        Returns a new Dataset object containing the rows or columns specified by the given index.

        Parameters
        ----------
        index : boolean and integer array-like, integer slice, and scalar integer are supported.

        axis : int, optional
            The axis along which to select the data. Only axis=0 is supported.

        Returns
        -------
        Dataset
            A new Dataset object containing the selected rows or columns.
        """
        return self.take(index, axis=axis)

    def take(self, indices: Int | SeqIndexType, axis: int = 0) -> Dataset:
        """
        Take elements from the dataset along an axis.
        """
        if axis != 0:
            raise ValueError("Only axis=0 is supported.")
        return Dataset(**(self._get_rows(indices) | self.__dict__))

    def _get_rows(self, index: Int | SeqIndexType) -> dict[str, Any]:
        if isinstance(index, (int, np.integer)):
            if index < 0:
                index = len(self) + index
            index = int(cast(int, index))
            index = slice(index, index + 1)

        return {key: _safe_indexing(value, index, axis=0) for key, value in self.items()}

    def _get_cols(self, cols: list[str]) -> dict[str, Any]:
        """Get specified columns from the dataset.

        This method returns a dict containing the specified columns and their corresponding values from the dataset.

        Parameters
        ----------
        cols : Sequence[str]
            A sequence of column names to retrieve from the dataset.

        Returns
        -------
        dict[str, Any]
            A dictionary containing the specified columns and their corresponding values.

        Raises
        ------
        ValueError
            If any of the specified columns are not found in the dataset.

        """
        if any(col not in self for col in cols):
            raise ValueError(f"Column(s) {', '.join(col for col in cols if col not in self)} not found in the dataset.")

        return {col: self.get(col) for col in cols}

    def __getitem__(self, index: IndexType) -> Dataset | Any:
        """
        Get the item(s) at the specified index(es) from the dataset.

        Parameters
        ----------
        index : IndexType
            The index(es) to retrieve the item(s) from the dataset. The index can be of different types:
            - N2DIndexType: A tuple of two elements where the first element is an int representing the row index,
            and the second element is either a str representing the column name or a list of strs representing
            multiple column names. This is used for 2D indexing.
            E.g., (0, 'column1') or (slice(0, 5), ['column1', 'column2'])
            - FunctionIndexType: A callable object that takes the dataset as an argument and
            returns the actual index(es)
            to retrieve the item(s) from the dataset. This is useful for custom indexing operations.
            - str: A string representing a single column name. This is used to retrieve a
            single column from the dataset.
            - list[str]: A list of strings representing multiple column names.
            This is used to retrieve multiple columns
            from the dataset.


        Returns
        -------
        Dataset | Any
            The item(s) at the specified index(es) from the dataset. If you select one single column,
            it will return Any based on the type of the value. In any other case,
            it will return a new Dataset object containing the selected rows and columns.

        Raises
        ------
        ValueError
            If the index is invalid or the specified column(s) are not found in the dataset.

        ValueError
            If the second index in a 2D index is not a string or a list of strings.
        """
        if isinstance(index, tuple) and len(index) == 2 and isinstance(index[1], (str, list)):  # N2DIndexType
            if isinstance(index[0], np.integer) and isinstance(index[1], str):  # single row single column
                return self.at(cast(int, index[0]), cast(str, index[1]))
            elif hasattr(index[1], "__iter__") and all(isinstance(i, str) for i in index[1]):  # type: ignore # multiple rows and columns
                return self.__getitem__(index[0]).__getitem__(index[1])
            else:
                raise ValueError("Invalid index. 2D indexing: the 2nd index must be a str or a list of strs.")
        elif callable(index):  # FunctionIndexType
            actual_index = index(self)
            return self.__getitem__(actual_index)
        elif isinstance(index, str):  # single column
            if index not in self:
                raise ValueError(f"Column '{index}' not found in the dataset.")
            return self.get(index)

        elif isinstance(index, list) and all(isinstance(i, str) for i in index):  # multiple columns
            selected_vals = self._get_cols(cast(list[str], index))

        else:  # multiple rows
            selected_vals = self._get_rows(index)  # type: ignore
        return Dataset(**(self.__dict__ | selected_vals))

    def __len__(self):
        return len(next(iter(self.values())))

    def __eq__(self, other):
        if not isinstance(other, Dataset):
            return False
        if self.keys() != other.keys():
            return False
        return all(np.array_equal(self[k], other[k]) for k in self.keys())  # type: ignore

    @property
    def shape(self):
        """
        Returns the shape of the dataset.
        """
        return (len(self), len(self.columns))

    @property
    def columns(self) -> list[str]:
        """
        Returns a list of the variable names in the dataset.
        """
        return list(self.keys())

    def transform(self, func: Callable[[Dataset], Dataset], *args, **kwargs) -> Dataset:
        """
        transform a function to the dataset.

        Example:
        ```python
        def my_func(data):
            return data["X"] + data["y"]

        new_data = data.transform(my_func)
        ```
        """
        self_copy = self.copy()
        return func(self_copy, *args, **kwargs)

    def copy(
        self,
    ):
        """
        Returns a copy of the dataset.
        """
        return Dataset(**({k: v.copy() for k, v in self.items()} | self.__dict__))

    def filter(self, condition: Callable[[Dataset], list[bool]]) -> Dataset:
        """Filter the rows of the dataset based on a condition.

        Parameters
        ----------
        condition : Callable[[Dataset], list[bool]]
            _description_

        Returns
        -------
        Dataset
            _description_
        """
        mask = condition(self)
        return self[mask]

    def split(self, spliter: SpliterType, **kwargs) -> DatasetDict:
        """
        Split the dataset into training and testing sets.
        """
        spliter_func = partial(spliter, **kwargs) if kwargs is not None else spliter
        splits = spliter_func(self)
        if isinstance(splits, dict):
            dict_splits = splits
        elif isinstance(splits, Sequence):
            dict_splits: dict = {"train": splits[0]}
            if len(splits) == 2:
                dict_splits.update({"test": splits[1]})
            elif len(splits) > 2:
                dict_splits.update({f"test_{i}": v for i, v in enumerate(splits[1:], 1)})
        else:
            raise ValueError("Invalid split. The split function must return a tuple or a dictionary.")
        return DatasetDict(dict_splits, **self.__dict__)

    @classmethod
    def from_tuple(cls, data: tuple, **kwarg) -> Dataset:
        """
        Create a dataset from a tuple.
        """
        if len(data) == 1:
            return cls(X=data[0], **kwarg)
        elif len(data) == 2:
            return cls(X=data[0], y=data[1], **kwarg)
        else:
            return cls(**({f"var_{i}": v for i, v in enumerate(data)} | kwarg))


class DatasetDict(dict):

    def __init__(
        self,
        data: dict[str, Dataset],
        name: str = "Dict of Datasets",
        description: Optional[str] = None,
        metadata: Optional[dict] = None,
        **kwargs,
    ) -> None:
        self.name = name
        self.description = description
        self.metadata = metadata or {}
        super().__init__(**data)
        self._set_attributes_by_split()

    @property
    def splits(self) -> Generator[tuple[str, Dataset], None, None]:
        yield from self.items()

    def _set_attributes_by_split(self):
        """
        Sets attributes for each data split.

        This method iterates over the data splits in the dataset and sets attributes
        for each split. It uses the split name as the attribute name and assigns the
        corresponding data to that attribute. Additionally, it sets attributes for each
        key-value pair in the data, using the key concatenated with the split name as
        the attribute name. For example, if the dataset has a split called "train" and
        a data variable called "X", it will set an attribute called "X_train" with the
        data from the "X" variable in the "train" split.

        Args:
            None

        Returns:
            None
        """
        for data_split_name, data in self.items():
            setattr(self, data_split_name, data)
            for k, v in data.items():
                setattr(self, k + "_" + data_split_name, v)

    def __getitem__(self, name):
        """
        Return a named attributed.
        """
        return self.get(name)

    def __str__(self):
        return f"DatasetDict: '{self.name}' with splits: {list(self.keys())}."

    @property
    def columns(self) -> list[str]:
        """
        Return the columns of the dataset. This is the same as the columns of the data after split.
        """
        _, split = next(iter(self.splits))
        return split.columns

    def transform(self, func: Callable[[Dataset], Dataset], *args, **kwargs) -> DatasetDict:
        """
        transform a function to the dataset.
        """
        new_data = {k: v.transform(func, *args, **kwargs) for k, v in self.items()}
        return DatasetDict(data=new_data, name=self.name, description=self.description, metadata=self.metadata)

    def filter(self, condition: Callable[[Dataset], list[bool]]) -> DatasetDict:
        """
        Filter the dataset.
        """
        new_data = {k: v.filter(condition) for k, v in self.items()}
        return DatasetDict(data=new_data, name=self.name, description=self.description, metadata=self.metadata)

    @classmethod
    def from_dataset(cls, dataset: Dataset, split_name="train") -> DatasetDict:
        """
        Create a DatasetDict from a single dataset.
        """
        return cls(
            data={split_name: dataset},
            name=f"Auto generated from `{dataset.name}` dataset",
            description=dataset.description,
            metadata=dataset.metadata,
        )
