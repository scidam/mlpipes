import pytest
import numpy as np
import pandas as pd
from ..preprocessing import (DropColumns, FillNASimple, MergeCategories, DataFrameLabelEncoder)

def test_drop_colums_typechecking():
    dc = DropColumns(colnames=['x', 'y'])
    with pytest.raises(Exception) as exinfo:
        dc.transform(dict())
    assert 'First argument' in str(exinfo.value)

def test_drop_columns_work():
    df = pd.DataFrame({'x': [1, 2, 3], 'y': [2, 3, 4], 'z': [6, 7, 8]})
    colnames = ['a', 'b', 'y', 'z']
    dc = DropColumns(colnames=colnames)
    assert ['x'] == dc.fit_transform(df).columns.tolist()

def test_fill_na_simple():
    df = pd.DataFrame({'x': [1, 2, 3, 5, 6, 7], 'y': [1, 2, 2, 2, np.nan, 4]})
    fna = FillNASimple(colnames=['x', 'y'])
    res = fna.fit_transform(df)
    assert res.loc[4, 'y'] == 2

def test_merge_categories():
    df = pd.DataFrame({'x': ['a', 'c', 'b'], 'y': [1, 2, 4], 'z': [6, 7, 8]})
    mc = MergeCategories(colnames=['x', 'y'], mapping={'x': {'what': ['a', 'b'],'to': 'c'},
                                                       'y': {'what': [1, 2], 'to': 3}})
    expected = pd.DataFrame({'x': ['c', 'c', 'c'], 'y': [3, 3, 4], 'z': [6, 7, 8]})
    assert expected.equals(mc.fit_transform(df))

def test_le():
    df = pd.DataFrame({'x': ['a', 'c', 'b'], 'y': [1, 2, 4], 'z': [6, 7, 8]})
    le = DataFrameLabelEncoder(colnames=('x',), prefix='')
    result = le.fit_transform(df)
    expected = pd.DataFrame({'x': [0, 2, 1], 'y': [1, 2, 4], 'z': [6, 7, 8]})
    assert all(expected.loc[:, 'x'] == result.loc[:, 'x'])
    assert set(le.data_[0]['x']) == set(df.loc[:, 'x'].tolist())

