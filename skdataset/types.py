from collections.abc import Mapping
from typing import Any, Protocol, TypeAlias, TypeVar

import numpy as np
import numpy.typing as npt

from skdataset import Dataset

T = TypeVar("T", bound=Mapping)


class SpliterType(Protocol):
    def __call__(self, dataset: T, *args: Any, **kwargs: Any) -> dict[str, T] | tuple[T]: ...


class FunctionIndexType(Protocol):
    def __call__(self, dataset: Dataset) -> Any: ...


Int: TypeAlias = int | np.integer
SimpleIndexType: TypeAlias = str | Int
SeqIndexType: TypeAlias = list[Int] | list[bool] | npt.NDArray[np.integer] | npt.NDArray[np.bool_] | slice | list[str]
N2DIndexType: TypeAlias = tuple[Any, str] | tuple[Any, list[str]]

IndexType: TypeAlias = SimpleIndexType | SeqIndexType | FunctionIndexType | N2DIndexType
