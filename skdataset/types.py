from typing import Protocol, TypeVar, Any, Callable, TypeAlias
from collections.abc import Mapping
import numpy as np
import numpy.typing as npt
from skdataset import Dataset

T = TypeVar("T", bound=Mapping)

class SpliterType(Protocol):
    def __call__(self, dataset: T, *args: Any, **kwargs: Any) -> dict[str, T] | tuple[T]:
        ...
        
SimpleIndexType: TypeAlias = str | int
SeqIndexType: TypeAlias = list[int] | list[str] | list[bool] | tuple[int, str] | tuple[Any, str] | tuple[Any, list[str]]
NumpyIndexType: TypeAlias = npt.NDArray[np.int_] | npt.NDArray[np.bool_]

IndexType: TypeAlias = SimpleIndexType | SeqIndexType | NumpyIndexType | slice | Callable[[Dataset], Any]