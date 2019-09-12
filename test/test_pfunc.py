
import pytest
import numpy as np
import pandas as pd
from ..pfunc import (drop_columns, merge_categories,
                     fill_na_simple, get_le, get_ohe)


def test_type_checks():
    adict = dict(a=[1,2,3], b=4.5)
    with pytest.raises(Exception) as exinfo:
        drop_columns(adict, columns=('a',))
    assert 'First argument' in str(exinfo.value)


def test_dropcolumns():
    df = pd.DataFrame({'x': [1,2,3], 'y': [2,3,4], 'z': [6,7,8]})
    colnames = ['a', 'b', 'y', 'z']
    assert ['x'] == drop_columns(df, colnames=colnames).columns.tolist()


def test_merge_categories():
    df = pd.DataFrame({'x' :  [1,2,3,4], 'y': ['m', 'w', 'r', 'z']})
    result, _ = merge_categories(df, colnames=('x', 'y', 'phi'),
                                 mapping={'x': {'what': [1,2], 'to': 4},
                                 'y': {'what': ['w','r'], 'to': 'z'}})
    expected = pd.DataFrame({'x': [4, 4, 3, 4],
                             'y': ['m', 'z', 'z', 'z']})
    assert expected.equals(result)


def test_fill_na_simple():
    df = pd.DataFrame({'x': [1, np.nan, 2, 3, 2, 2, 2, 7],
                       'y': [1, 2, 3, 4, 5, 6, np.nan, 8]})
    result = fill_na_simple(df, colnames=('x', 'y'), methods=(np.median, np.mean))
    _ = (1 + 2 + 3 + 4 + 5 + 6 + 8) / 7.0
    expected = pd.DataFrame({'x': [1.0, 2.0, 2.0, 3.0, 2.0, 2.0, 2.0, 7.0],
                             'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, _, 8.0]})
    assert expected.equals(result)


def test_fill_na_simple_wo_methods():
    df = pd.DataFrame({'x': [1, np.nan, 2, 3, 2, 2, 2, 7],
                       'y': [1, 2, 3, 4, 5, 6, np.nan, 8]})
    result = fill_na_simple(df, colnames=('x', 'y'))
    expected = pd.DataFrame({'x': [1.0, 2.0, 2.0, 3.0, 2.0, 2.0, 2.0, 7.0],
                             'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 4.0, 8.0]})
    assert expected.equals(result)


def test_fill_na_simple_categorica():
    df = pd.DataFrame({'x': [1, np.nan, 2, 3, 2, 2, 2, 7],
                       'y': ['a', 'a', 'b', 'a', 'a', 'a', None, 'c']})

    result = fill_na_simple(df, colnames=('x', 'y'),
                            methods=(np.median, lambda x: pd.Series.mode(x)[0])
                            )
    expected = pd.DataFrame({'x': [1.0, 2.0, 2.0, 3.0, 2.0, 2.0, 2.0, 7.0],
                             'y': ['a', 'a', 'b', 'a', 'a', 'a', 'a', 'c']})
    assert expected.equals(result)


def test_get_le():
    df = pd.DataFrame({'x': ['a', 'a', 'b'],
                       'y': ['w', 'q', 'q']})
    result_df, classes_ = get_le(df, colnames=('x', 'y'), drop=True)
    assert len(result_df.columns) == 2
    assert result_df.loc[0, 'LE_x'] == classes_['x'].index('a')
    assert result_df.loc[1, 'LE_x'] == classes_['x'].index('a')
    assert result_df.loc[2, 'LE_x'] == classes_['x'].index('b')
    assert len(classes_['x']) == 2
    assert result_df.loc[0, 'LE_y'] == classes_['y'].index('w')
    assert result_df.loc[1, 'LE_y'] == classes_['y'].index('q')
    assert result_df.loc[2, 'LE_y'] == classes_['y'].index('q')
    assert len(classes_['y']) == 2

    # expected number of columns without dropping is 4!
    result_df, classes_ = get_le(df, colnames=('x', 'y'), drop=False)
    assert len(result_df.columns) == 4


def test_get_le_wo_prefix():
    df = pd.DataFrame({'x': ['a', 'b', 'b', 'c']})
    result, _ = get_le(df, colnames=('x', ), prefix='', drop=True)
    expected = pd.DataFrame({'x': [0, 1, 1, 2]})
    assert expected.equals(result)
    result, _ = get_le(df, colnames=('x',), prefix='', drop=False)
    expected = pd.DataFrame({'x': ['a', 'b', 'b', 'c'], '_x': [0, 1, 1, 2]},
                            columns=['x', '_x'])
    assert expected.equals(result)