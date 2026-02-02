import numpy as np
import pandas as pd

from sklearn.inspection import permutation_importance

from model_registry import get_models


AVAILABLE_SCORERS = {
    'accuracy': 'accuracy',
    'balanced_accuracy': 'balanced_accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc',
    'average_precision': 'average_precision',
}


def _get_scorer(metric):
    return AVAILABLE_SCORERS.get(metric, 'f1')


def compute_feature_importance(X_train, y_train, X_test, y_test, metric='f1', n_repeats=5, random_state=42):
    results = []
    models = get_models()
    scorer = _get_scorer(metric)

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
