
from inspect import signature
from collections.abc import Callable
from sklearn.base import BaseEstimator, TransformerMixin
from .pfunc import (drop_columns, get_ohe, get_le, merge_categories,
                    fill_na_simple)

# Declare all available (public) classes/functions manually
__all__ = ('DataFrameLabelEncoder', 'DataFrameOneHotEncoder',
           'MergeCategories', 'FillNASimple',
           'DropColumns')


class AbstractPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self


class BasePreprocessingMeta(type):
    def __new__(cls, clsname, bases, dct):
        return type.__new__(cls, clsname, bases, dct)


def _init_factory(varvalues=dict()):
    def __init__(self, **varvalues):
        for key, val in varvalues.items():
            setattr(self, key, val)
    return __init__


def _transform_factory(method=None):
    def transform(self, X, y=None):
        if method is not None:
            result = method(X, **self.__dict__)
            if isinstance(result, tuple):
                self.data_ = [x for x in result[1:]]
                return result[0]
            else:
                return result
        else:
            return X
    return transform


for key, obj in locals().copy().items():
    if isinstance(obj, Callable):
        if hasattr(obj, '__pipename__'):
            varvalues = dict()
            original_func = getattr(obj, '__wrapped__') if hasattr(obj, '__wrapped__') else obj
            for var, val in tuple(signature(original_func).parameters.items())[1:]:
                varvalues[var] = val.default
            methods = {'__init__': _init_factory(varvalues),
                       'transform': _transform_factory(method=obj)}
            locals()[obj.__pipename__] = BasePreprocessingMeta(obj.__pipename__,
                                                               (AbstractPreprocessor,),
                                                               methods)
