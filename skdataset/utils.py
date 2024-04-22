import numpy as np
import pandas as pd
from sklearn.utils import Bunch

from skdataset import Dataset


def get_data_names_from_bunch(data: Bunch):
    array_like = (np.ndarray, pd.DataFrame, pd.Series)
    actual_data = [(k, len(v)) for k, v in data.items() if isinstance(v, array_like)]
    max_len = max(actual_data, key=lambda x: x[1])[1]
    return [k for k, v in actual_data if v == max_len]


def get_sklearn_data(func, **kwargs):
    data = func(**kwargs)
    if isinstance(data, tuple):
        return Dataset.from_tuple(data)
    elif isinstance(data, Bunch):
        data_names = get_data_names_from_bunch(data)
        metadata = {k: v for k, v in data.items() if k not in data_names}
        data = {k: v for k, v in data.items() if k in data_names}
        return Dataset(**data, metadata=metadata)
    else:
        raise ValueError(f"Unknown data type {type(data)}. Function needs to return either a tuple or a Bunch object.")
