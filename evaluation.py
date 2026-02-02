import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
    matthews_corrcoef,
    cohen_kappa_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from model_registry import get_models


def _classification_metrics(y_test, y_pred, proba=None, score_vals=None):
    average = 'binary' if y_test.nunique() == 2 else 'weighted'

    row = {}
    row['accuracy'] = accuracy_score(y_test, y_pred)
    row['balanced_accuracy'] = balanced_accuracy_score(y_test, y_pred)
    row['precision'] = precision_score(y_test, y_pred, zero_division=0, average=average)
    row['recall'] = recall_score(y_test, y_pred, zero_division=0, average=average)
    row['f1'] = f1_score(y_test, y_pred, zero_division=0, average=average)
    row['mcc'] = matthews_corrcoef(y_test, y_pred)
    row['cohen_kappa'] = cohen_kappa_score(y_test, y_pred)

    if y_test.nunique() == 2 and (proba is not None or score_vals is not None):
        scores = proba if proba is not None else score_vals
        try:
            row['roc_auc'] = roc_auc_score(y_test, scores)
        except Exception:
            row['roc_auc'] = np.nan
        try:
            row['avg_precision'] = average_precision_score(y_test, scores)
        except Exception:
            row['avg_precision'] = np.nan
    else:
        row['roc_auc'] = np.nan
        row['avg_precision'] = np.nan

    if proba is not None and y_test.nunique() == 2:
        try:
            row['log_loss'] = log_loss(y_test, proba)
        except Exception:
            row['log_loss'] = np.nan
        try:
            row['brier'] = brier_score_loss(y_test, proba)
        except Exception:
            row['brier'] = np.nan
    else:
        row['log_loss'] = np.nan
        row['brier'] = np.nan

    if y_test.nunique() == 2:
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        row['tn'] = int(tn)
        row['fp'] = int(fp)
        row['fn'] = int(fn)
        row['tp'] = int(tp)

    return row


def _regression_metrics(y_test, y_pred):
    row = {}
    row['mae'] = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    row['rmse'] = float(np.sqrt(mse))
    row['r2'] = r2_score(y_test, y_pred)
    denom = np.where(np.abs(y_test) < 1e-8, np.nan, np.abs(y_test))
    row['mape'] = np.nanmean(np.abs((y_test - y_pred) / denom))
    return row


def evaluate_models(X_train, X_test, y_train, y_test, task_type='classification'):
    results = []
    models = get_models(task_type=task_type)

    for name, model in models.items():
        row = {'model': name}
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            proba = None
            score_vals = None
            if task_type == 'classification':
                if hasattr(model, 'predict_proba'):
                    try:
                        proba = model.predict_proba(X_test)[:, 1]
                    except Exception:
                        proba = None
                if proba is None and hasattr(model, 'decision_function'):
                    try:
                        score_vals = model.decision_function(X_test)
                    except Exception:
                        score_vals = None
                row.update(_classification_metrics(y_test, y_pred, proba=proba, score_vals=score_vals))
            else:
                row.update(_regression_metrics(y_test, y_pred))
            row['status'] = 'ok'

        except Exception as exc:
            row['status'] = f'error: {exc}'

        results.append(row)

    return pd.DataFrame(results)
