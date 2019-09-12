from collections.abc import Iterable
from .pfunc import check_type
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform

try:
    import imblearn
except ImportError:
    raise Exception("Sampling package depends on imbalanced-learning package."
                    "Install imbalanced-learning package first.")


@check_type()
def sample_like_target(target, source, using=None, autoencode=False,
                       scale=True, random_state=None, fillna=False,
                       distance=lambda x: pdist(x, 'euclidean'),
                       quantile_use=0.1):

    if isinstance(using, Iterable):
        using = list(set(using))
    else:
        using = list(set(target.columns).intersection(set(source.columns)))

    T = target.copy().loc[:, using]
    S = source.copy().loc[:, using]

    if fillna:
        T.fillna(T.median(), inplace=True)
        S.fillna(S.median(), inplace=True)
    else:
        dropped_mask = pd.isnull(S).sum(axis=1).astype(np.bool)
        leaved_indices = [j for j in range(S.shape[0]) if not dropped_mask[j]]
        S = S.loc[~dropped_mask, :]
        T = T.dropna()

    if autoencode:
        print(type(T), type(S))
        C = pd.concat([T, S]).reset_index(drop=True)
        R = pd.get_dummies(C, dummy_na=True)
        T = R.iloc[:T.shape[0], :]
        S = R.iloc[T.shape[0]:, :]

    if scale:
        scaler = StandardScaler()
        scaler.fit(T)
        tc = T.columns
        sc = S.columns
        T = pd.DataFrame(scaler.transform(T.values), columns=tc)
        S = pd.DataFrame(scaler.transform(S.values), columns=sc)
    combined = pd.concat([T, S]).reset_index(drop=True)
    distances = squareform(distance(combined))[:T.shape[0],:]
    quantiles =  np.percentile(distances,
                               quantile_use * 100 if quantle_use < 1.0\
                               else quantile_use, axis=1)
    repeated = np.repeat(quantiles[:,np.newaxis], S.shape[0], axis=1)
    filter_mask = distances[:, T.shape[0]:] <= repeated
    to_leave = np.unique(np.repeat([leaved_indices], T.shape[0], axis=0)[filter_mask])
    return source.iloc[to_leave.tolist(), :]


if __name__ == '__main__':
    target = pd.DataFrame(np.random.rand(100, 2)+0.8)
    source = pd.DataFrame(np.random.rand(400, 2))
    result = sample_like_target(target, source)
    import matplotlib.pyplot as plt
    plt.plot(target.values[:,0], target.values[:,1],'ro', source.values[:,0], source.values[:,1], 'b.',
         result.values[:,0], result.values[:,1], 'gx')
    plt.show()


