import pytest
from skdataset.dataset import Dataset
import pandas as pd
import numpy as np
import random
from sklearn.datasets import load_iris
from itertools import chain, combinations


@pytest.fixture
def dataset():
    data = load_iris(as_frame=False)
    X = data.pop('data')
    y = data.pop('target')
    metadata = { k: v for k, v in data.items() if k not in ['X', 'y']}
    return Dataset(X=X, y=y, metadata=metadata)
    

def test_dataset_init_empty():
    dataset = Dataset(name='empty', description='Test dataset')
    assert dataset.name == 'empty'
    assert dataset.description == 'Test dataset'
    assert dataset.metadata == {}
    
    
def test_dataset_init_with_data():
    dataset = Dataset(
        name='with initial data', 
        description='Test dataset', 
        X=pd.DataFrame({'column1': [1, 2, 3], 'column2': ['a', 'b', 'c']}),
        y=pd.Series([1, 2, 3]),
        metadata={'source': 'generated'}
    )
    assert dataset.name == 'with initial data'
    assert dataset.description == 'Test dataset'
    assert dataset['X'].equals(pd.DataFrame({'column1': [1, 2, 3], 'column2': ['a', 'b', 'c']}))
    assert dataset['y'].equals(pd.Series([1, 2, 3]))
    assert dataset.metadata == {'source': 'generated'}

def test_dataset_init_unequal_data():
    with pytest.raises(ValueError):
        Dataset(
            name='with initial data', 
            description='Test dataset', 
            X=pd.DataFrame({'column1': [1, 2, 3], 'column2': ['a', 'b', 'c']}),
            y=pd.Series([1, 2, 3, 4]),
            metadata={'source': 'generated'}
        )


def test_dataset_getitem(dataset):
    assert np.array_equal(dataset.X, load_iris(as_frame=False).data)
    assert np.array_equal(dataset.y, load_iris(as_frame=False).target)

    with pytest.raises(AttributeError):
        dataset.z

def test_dataset_setattr(dataset):
    #replacing X
    new_X = np.random.random((len(dataset), 4))
    dataset.X = new_X
    assert np.array_equal(dataset.X, new_X)
    
    #adding new column
    dataset.z = np.random.random(len(dataset))
    
    #adding new column with wrong length
    with pytest.raises(ValueError):
        dataset.a = np.random.random(len(dataset)+1)
        
        
def test_dataset_setitem(dataset):
    #replacing X
    new_X = np.random.random((len(dataset), 4))
    dataset['X'] = new_X
    assert np.array_equal(dataset.X, new_X)
    
    #adding new column
    dataset['z'] = np.random.random(len(dataset))
    
    #adding new column with wrong length
    with pytest.raises(ValueError):
        dataset['a'] = np.random.random(len(dataset)+1)
        
def dataset_sync_setitem_setattr(dataset):
    new_X = np.random.random((len(dataset), 4))
    dataset['col1'] = new_X
    
    assert np.array_equal(dataset['col1'], dataset.col1)
    
    dataset.col2 = np.random.random(len(dataset))
    
    assert np.array_equal(dataset['col2'], dataset.col2)


def test_dataset_at(dataset):
    samples = 10
    indices = {
        int:random.sample(range(0, len(dataset)), samples), 
        np.integer:np.random.randint(0, len(dataset), samples),
    }  
    for t, inds in indices.items():
        for i in inds:
            assert np.array_equal(dataset.at(i, 'X'), dataset.X[slice(i, i+1)])
            assert np.array_equal(dataset.at(i, 'y'), dataset.y[slice(i, i+1)])
            
            
def test_dataset__get_rows(dataset):
    # test these types list[Int] | list[bool] | npt.NDArray[np.integer] | npt.NDArray[np.bool_] | slice | int | np.integer
    samples = 10
    indices = {
        'list[int]': random.sample(range(0, len(dataset)), samples),
        'list[bool]': random.choices([True, False], k=len(dataset)),
        'np.array[int]': np.random.randint(0, len(dataset), samples),
        'np.array[bool]': np.random.choice([True, False], len(dataset)),
        'slice': slice(0, samples),
        'int': random.randint(0, len(dataset)),
        'np.int': np.random.randint(0, len(dataset)),
    }
    for inds in indices.values():
        ds = dataset._get_rows(inds)
        for k, v in ds.items():
            inds_ = slice(inds, inds+1) if isinstance(inds, (int, np.integer)) else inds
            assert np.array_equal(v, dataset[k][inds_])
        

def test_dataset_take_and_iloc(dataset):
    samples = 10
    indices = {
        'list[int]': random.sample(range(0, len(dataset)), samples),
        'list[bool]': random.choices([True, False], k=len(dataset)),
        'np.array[int]': np.random.randint(0, len(dataset), samples),
        'np.array[bool]': np.random.choice([True, False], len(dataset)),
        'slice': slice(0, samples),
        'int': random.randint(0, len(dataset)),
        'np.int': np.random.randint(0, len(dataset)),
    }
    
    for inds in indices.values():
        assert dataset.take(inds) == dataset.iloc(inds)
        
def test_dataset__get_cols(dataset):
    cols = ['X', 'y']
    
    def powerset(iterable):
        "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))
    
    cols = list(powerset(cols))
    
    for c in cols:
        ds = dataset._get_cols(c)
        for k, v in ds.items():
            assert np.array_equal(v, dataset[k])
            
def test_dataset__getitem(dataset):
    samples = 100
    indices = {
        'int': random.randint(0, len(dataset)),
        'np.int': np.random.randint(0, len(dataset)),
        'list[int]': random.sample(range(0, len(dataset)), samples),
        'list[bool]': random.choices([True, False], k=len(dataset)),
        'np.array[int]': np.random.randint(0, len(dataset), samples),
        'np.array[bool]': np.random.choice([True, False], len(dataset)),
        'slice': slice(0, samples),
    }
    
    for inds in indices.values():
        ds = dataset.__getitem__(inds)
        for k, v in ds.items():
            _inds = slice(inds, inds+1) if isinstance(inds, (int, np.integer)) else inds
            assert np.array_equal(v, dataset[k][_inds])
    
    indices = {
        'str': 'X',
        'FunctionIndexType': lambda x: x['y'] == 0,
        'tuple_1': (slice(0, samples), 'X'),
        'tuple_2': (slice(0, samples), ['X', 'y']),
    }
    
    for k, inds in indices.items():
        ds = dataset.__getitem__(inds)
    
        if k == 'str':
            assert np.array_equal(ds, dataset.get(inds))
        elif k == 'FunctionIndexType':
            inds = inds(dataset)
            assert Dataset(**dataset[inds]) == ds
        elif k == 'tuple_1':
            assert np.array_equal(ds, dataset[inds[0]][inds[1]])
        elif k == 'tuple_2':
            test_ds = dataset[inds[1]][inds[0]]
            assert Dataset(**test_ds) == ds