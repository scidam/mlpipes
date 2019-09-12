import itertools
from functools import wraps
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from collections.abc import Iterable, Callable

__all__ = ('drop_columns', 'get_ohe', 'get_le',
           'merge_categories', 'fill_na_simple')


def check_type(itype=pd.DataFrame):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not isinstance(args[0], itype):
                raise(Exception("First argument should be an instance of {}".format(itype)))
            return func(*args, **kwargs)
        return wrapper
    return decorator


def filter_colnames(df, colnames=None):
    if isinstance(colnames, Iterable):
        return [col for col in colnames if col in df.columns]
    return list()


@check_type()
def drop_columns(df, colnames=tuple()):
    """Drops specified columns from a DataFrame"""
    return df.drop(filter_colnames(df, colnames), axis=1)


@check_type()
def get_ohe(df, colnames=tuple(), prefix=None, drop=True):
    """Encodes specified df-columns in one-hot fashion"""
    colnames = filter_colnames(df, colnames)
    auxiliary_df = pd.get_dummies(df.loc[:, colnames], prefix=prefix, columns=colnames)
    if drop:
        return pd.concat([df.drop(colnames, axis=1), auxiliary_df], axis=1)
    else:
        return pd.concat([df, auxiliary_df], axis=1)


@check_type()
def get_le(df, colnames=tuple(), prefix=None, drop=True):
    """Label Encoder.

    Performs label encoding on specified columns of a data frame.

    **Parameters**

        :param df: input data frame
        :param colnames: a list of column names to be processed
        :param prefix: use this prefix when setting names to processed columns
        :param drop: default is True, drop original columns
        :rtype: Pandas data frame
        :returns: modified data frame
    """

    sep = '_'

    colnames = filter_colnames(df, colnames)

    if not colnames:
        return df

    enc = LabelEncoder()
    labels = dict()
    result_df = pd.DataFrame()
    if prefix is None or not isinstance(prefix, str):
        prefix = 'LE'

    for col in colnames:
        encoded = enc.fit_transform(df.loc[:, col].values)
        if drop and not prefix:
            result_df.loc[:, col] = encoded
        elif prefix:
            result_df.loc[:, prefix + sep + col] = encoded
        else:
            result_df.loc[:, sep + col] = encoded
        labels[col] = enc.classes_.tolist()

    if drop:
        return (pd.concat([df.drop(colnames, axis=1), result_df], axis=1), labels)
    else:
        return (pd.concat([df, result_df], axis=1), labels)


@check_type()
def merge_categories(df, colnames=tuple(), mapping=dict()):
    """

    Usage
    -----
    """
    colnames = filter_colnames(df, colnames)

    if not colnames or not mapping:
        return df

    if (set(colnames) - set(mapping.keys())):
        # check if mapping keys doesn't cover all colnames
        raise Exception("Mapping argument should cover all specified column names")

    _df = df.copy()
    for col in colnames:
        mask = df.loc[:, col].isin(mapping[col]['what'])
        _df.loc[mask, col] = mapping[col]['to']

    return _df, mapping


@check_type()
def fill_na_simple(df, colnames=tuple(), methods=None):

    if isinstance(methods, Iterable):
        if len(colnames) != len(methods):
            raise Exception('Colnames and methods arrays should have the same length')
    elif isinstance(methods, Callable):
        methods = (methods, )
    else:
        methods = (pd.np.median, )

    colnames = filter_colnames(df, colnames)

    if len(methods) > len(colnames):
        zipped = zip(itertools.cycle(colnames), methods)
    else:
        zipped = zip(colnames, itertools.cycle(methods))

    res_df = df.copy()
    method = None
    for col, met in zipped:
        if isinstance(met, Callable):
            method = met
        if method is not None:
            value = method(res_df.loc[:, col].dropna().values)
            res_df.loc[:, col].fillna(value, inplace=True)
    return res_df


# ---------  class names for sklearn pipelines --------
get_le.__pipename__ = 'DataFrameLabelEncoder'
get_ohe.__pipename__ = 'DataFrameOneHotEncoder'
merge_categories.__pipename__ = 'MergeCategories'
fill_na_simple.__pipename__ = 'FillNASimple'
drop_columns.__pipename__ = 'DropColumns'
# -----------------------------------------------------








# ----------------------------------------------------------------- Revision needed!

#class SelectFeatures(AbstractPreprocessor):
    
    #def __init__(self, k, n):
        #self.k = k
        #self.n = n

    #def transform(self, X, y=None):
        #_ = [int(x) for x in bin(self.k)[2:]]
        #_ = [0] * (self.n - len(_)) + _
        #return X.iloc[:, [j for j in range(self.n) if _[j]]]


#class FillNaValues(AbstractPreprocessor):

    #def __init__(self, name=None, train=None, n_features=None,
                 #clf=RandomForestRegressor()):
        #self.train = train
        #self.name = name
        #self.clf = clf
        #self.n_features = n_features
    
    #def transform(self, X, y=None):

        #if self.name is None: 
            #return X

        #if X.loc[:, self.name].isnull().sum() == 0:
            #return X
        
        #_train = self.train.copy() if self.train is not None else X.copy()
        #null_mask = _train[self.name].isnull()
        #y = _train[self.name][~null_mask]
        #_train = _train.drop(self.name, axis=1)
        
        #n_features = int(pd.np.ceil(X.shape[1] * 0.3) or self.n_features)
        
        #encoders = dict()
        #for key in _train.columns.tolist():
            #if not pd.np.issubdtype(_train[key].dtype, pd.np.number):
                #_train.loc[_train[key].isnull(), key]  = 'N-a-N'
                #le = LabelEncoder()
                #_train[key] = le.fit_transform(_train[key])
                #encoders[key] = le
            #else:
                #if any(_train[key].isnull()):
                    #_train['%s_nan' % key] = 0.0
                    #_train.loc[_train[key].isnull(), '%s_nan' % key] = 1.0
                    #_train.loc[_train[key].isnull(), key] = _train.loc[~_train[key].isnull(), key].median()

        #self.clf.fit(_train[~null_mask], y)
        
        ## dropping features
        #if hasattr(self.clf, 'feature_importances_'):
            ## drop columns and retrain classifier
            #indices = pd.np.argsort(self.clf.feature_importances_)[::-1]
            #features_to_drop = _train.columns[indices].values.tolist()[n_features:]
            #self.clf.fit(_train.drop(features_to_drop, axis=1)[~null_mask], y)
        #else:
            #features_to_drop = []
            
        #_X = X.copy()
        #for key in _train.columns:
            #if key not in _X.columns:
                #_X.loc[:, key] = 0.0
        #_X = _X[_train.columns]
        #for key in encoders.keys():
            #if not pd.np.issubdtype(_X[key].dtype, pd.np.number):
                #_X.loc[_X[key].isnull(), key]  = 'N-a-N'
                #_X[key] = encoders[key].transform(_X[key])
            #else:
                #if any(_X[key].isnull()):
                    #_X['%s_nan' % key] = 0.0
                    #_X.loc[_X[key].isnull(), '%s_nan' % key] = 1.0
                    #_X.loc[_X[key].isnull(), key] = X.loc[~_X[key].isnull(), key].median()
        
        #na_replacements = self.clf.predict(_X.drop(features_to_drop, axis=1)[null_mask])
        #result = X.copy()
        #result.loc[null_mask, self.name] = na_replacements
        #return result
