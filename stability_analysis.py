import numpy as np
import pandas as pd

from sklearn.inspection import permutation_importance
from sklearn.metrics import get_scorer

from model_registry import get_models


def _rolling_windows(n_rows, train_window, test_window, step):
    windows = []
    start = 0
    while start + train_window + test_window <= n_rows:
        train_slice = slice(start, start + train_window)
        test_slice = slice(start + train_window, start + train_window + test_window)
        windows.append((train_slice, test_slice))
        start += step
    return windows


def _scorer_name(task_type, metric):
    if task_type == 'regression':
        mapping = {
            'r2': 'r2',
            'mae': 'neg_mean_absolute_error',
            'mse': 'neg_mean_squared_error',
            'rmse': 'neg_mean_squared_error',
        }
        return mapping.get(metric, 'r2')

    mapping = {
        'accuracy': 'accuracy',
        'balanced_accuracy': 'balanced_accuracy',
        'precision': 'precision_weighted',
        'recall': 'recall_weighted',
        'f1': 'f1_weighted',
        'roc_auc': 'roc_auc',
        'average_precision': 'average_precision',
    }
    return mapping.get(metric, 'f1_weighted')


def rolling_model_scores(
    X,
    y,
    train_window,
    test_window,
    step,
    task_type='classification',
    metric='f1',
    models=None,
):
    models = models or get_models(task_type=task_type)
    scorer = get_scorer(_scorer_name(task_type, metric))
    windows = _rolling_windows(len(X), train_window, test_window, step)

    results = []
    for idx, (train_slice, test_slice) in enumerate(windows):
        X_train = X.iloc[train_slice]
        y_train = y.iloc[train_slice]
        X_test = X.iloc[test_slice]
        y_test = y.iloc[test_slice]

        for name, model in models.items():
            row = {
                'window_id': idx,
                'train_start': X_train.index.min(),
                'train_end': X_train.index.max(),
                'test_start': X_test.index.min(),
                'test_end': X_test.index.max(),
                'model': name,
            }
            try:
                model.fit(X_train, y_train)
                score = scorer(model, X_test, y_test)
                if task_type == 'regression' and metric in {'mae', 'mse', 'rmse'}:
                    score = -score
                    if metric == 'rmse':
                        score = float(np.sqrt(score)) if score >= 0 else np.nan
                row['score'] = float(score)
                row['status'] = 'ok'
            except Exception as exc:
                row['score'] = np.nan
                row['status'] = f'error: {exc}'
            results.append(row)

    return pd.DataFrame(results)


def time_sliced_permutation_importance(
    X,
    y,
    train_window,
    test_window,
    step,
    task_type='classification',
    metric='f1',
    n_repeats=5,
    random_state=42,
    models=None,
):
    models = models or get_models(task_type=task_type)
    scorer = _scorer_name(task_type, metric)
    windows = _rolling_windows(len(X), train_window, test_window, step)
    results = []

    for idx, (train_slice, test_slice) in enumerate(windows):
        X_train = X.iloc[train_slice]
        y_train = y.iloc[train_slice]
        X_test = X.iloc[test_slice]
        y_test = y.iloc[test_slice]

        for name, model in models.items():
            row = {'window_id': idx, 'model': name}
            try:
                model.fit(X_train, y_train)
                perm = permutation_importance(
                    model,
                    X_test,
                    y_test,
                    scoring=scorer,
                    n_repeats=n_repeats,
                    random_state=random_state,
                    n_jobs=1,
                )
                importances = perm.importances_mean
                row.update({feat: float(val) for feat, val in zip(X_test.columns, importances)})
                row['status'] = 'ok'
            except Exception as exc:
                row['status'] = f'error: {exc}'
            results.append(row)

    df = pd.DataFrame(results)
    df = df[df['status'] == 'ok'].drop(columns=['status'])
    return df
