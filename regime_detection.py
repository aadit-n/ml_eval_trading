import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def compute_regimes(df, feature_cols, n_clusters=3, random_state=42):
    features = df[feature_cols].dropna()
    if features.empty:
        return pd.Series(index=df.index, data=np.nan, name='regime')

    scaler = StandardScaler()
    X = scaler.fit_transform(features.values)
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(X)

    regime = pd.Series(index=df.index, data=np.nan, name='regime')
    regime.loc[features.index] = labels
    return regime


def regime_name_map(df, regime_col='regime', vol_col='realized_vol'):
    if regime_col not in df.columns or vol_col not in df.columns:
        return {}

    grouped = df.dropna(subset=[regime_col, vol_col]).groupby(regime_col)[vol_col].mean()
    if grouped.empty:
        return {}

    ordered = grouped.sort_values().index.tolist()
    names = {}
    if len(ordered) == 1:
        names[ordered[0]] = 'Normal Vol'
        return names

    labels = [
        'Low Vol / Mean-Reverting',
        'Normal Vol / Mixed',
        'High Vol / Risk-Off',
        'Extreme Vol / Crisis',
        'Ultra Vol / Disorderly',
        'Panic Vol / Tail Risk',
    ]
    for idx, regime_id in enumerate(ordered):
        label_idx = min(idx, len(labels) - 1)
        names[regime_id] = labels[label_idx]
    return names
