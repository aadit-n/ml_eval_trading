import numpy as np
import pandas as pd

from sklearn.inspection import permutation_importance

from model_registry import get_models


def _get_scorer(metric, task_type):
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


def compute_feature_importance(
    X_train,
    y_train,
    X_test,
    y_test,
    metric='f1',
    n_repeats=5,
    random_state=42,
    task_type='classification',
):
    results = []
    models = get_models(task_type=task_type)
    scorer = _get_scorer(metric, task_type)

    for name, model in models.items():
        row = {'model': name}
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
    df = df.set_index('model')
    return df
