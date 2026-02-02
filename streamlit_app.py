import warnings
warnings.simplefilter('ignore')

import json
import sys
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

from data_pipeline import load_data, build_features
from evaluation import evaluate_models
from feature_importance import compute_feature_importance
from indicator_registry import get_indicator_registry_with_diagnostics, INPUT_OPTIONS
from stability_analysis import rolling_model_scores, time_sliced_permutation_importance
from regime_detection import compute_regimes, regime_name_map
from model_registry import get_models

st.set_page_config(page_title='Signal Model Comparator', layout='wide')

st.title('Signal Model Comparator')
st.caption('Compare signal accuracy across multiple sklearn classifiers using a unified dataset.')

diag = get_indicator_registry_with_diagnostics()
INDICATORS = diag['registry']

def _parse_params(param_text):
    if not param_text.strip():
        return []
    parts = [p.strip() for p in param_text.split(',')]
    params = []
    for p in parts:
        if p == '':
            continue
        try:
            if '.' in p:
                params.append(float(p))
            else:
                params.append(int(p))
        except ValueError:
            params.append(float(p))
    return params


def _default_param_text(defaults):
    return ', '.join([str(d) for d in defaults])

def _parse_inputs(inputs_text):
    if not inputs_text.strip():
        return []
    return [p.strip().lower() for p in inputs_text.split(',') if p.strip()]


def _default_inputs_text(inputs):
    return ', '.join(inputs)

def _plot_price_with_signals(df, indicators, signal_idx, title):
    fig, axes = plt.subplots(
        2 if indicators else 1,
        1,
        figsize=(12, 6 if indicators else 4),
        sharex=True,
        gridspec_kw={'height_ratios': [2, 1]} if indicators else None,
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    price_ax = axes[0]
    price_ax.plot(df.index, df['close'], color='#4c78a8', label='close')
    if signal_idx is not None and len(signal_idx) > 0:
        price_ax.scatter(
            signal_idx,
            df.loc[signal_idx, 'close'],
            color='#f58518',
            s=18,
            label='predicted signal',
            zorder=3,
        )
    price_ax.set_title(title)
    price_ax.legend(loc='upper left')

    if indicators:
        ind_ax = axes[1]
        for col in indicators:
            if col in df.columns:
                ind_ax.plot(df.index, df[col], label=col)
        ind_ax.legend(loc='upper left', ncol=2)
        ind_ax.set_ylabel('indicator')

    return fig


def _plot_regimes(df, title, name_map=None):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df.index, df['close'], color='#4c78a8', label='close')

    if 'regime' in df.columns:
        regimes = df['regime'].dropna()
        if not regimes.empty:
            unique = sorted(regimes.unique())
            colors = sns.color_palette('Set2', n_colors=max(len(unique), 2))
            for regime_id, color in zip(unique, colors):
                mask = df['regime'] == regime_id
                if mask.any():
                    label = name_map.get(regime_id, f'Regime {int(regime_id)}') if name_map else f'Regime {int(regime_id)}'
                    ax.fill_between(
                        df.index,
                        df['close'].min(),
                        df['close'].max(),
                        where=mask,
                        color=color,
                        alpha=0.15,
                        label=label,
                    )
    ax.set_title(title)
    ax.legend(loc='upper left', ncol=3)
    return fig


with st.sidebar:
    st.header('Data Inputs')
    ticker = st.text_input('Ticker', value='JPM')
    period = st.text_input('Period (yfinance)', value='10y')
    interval = st.text_input('Interval (yfinance)', value='1d')

    st.header('Target')
    target_label = st.selectbox(
        'Target type',
        ['Return signal (binary)', 'Volatility (regression)', 'Volatility regime (3-class)'],
    )
    target_mode = {
        'Return signal (binary)': 'return_signal',
        'Volatility (regression)': 'volatility_regression',
        'Volatility regime (3-class)': 'volatility_regime',
    }[target_label]

    threshold = 0.0
    if target_mode == 'return_signal':
        threshold = st.number_input(
            'Signal threshold (future return > threshold)',
            value=0.0,
            step=0.0001,
            format='%.4f',
        )

    vol_window = st.number_input('Volatility window (bars)', min_value=5, max_value=252, value=20, step=1)
    vol_low_q, vol_high_q = 0.33, 0.66
    if target_mode == 'volatility_regime':
        vol_low_q = st.slider('Low regime quantile', min_value=0.1, max_value=0.45, value=0.33, step=0.01)
        vol_high_q = st.slider('High regime quantile', min_value=0.55, max_value=0.9, value=0.66, step=0.01)

    st.header('Train/Test Split')
    train_size = st.slider('Train size', min_value=0.5, max_value=0.9, value=0.8, step=0.05)

    st.header('Indicators (Tulipy)')
    indicator_count = len(INDICATORS)
    st.caption(f'Indicators found: {indicator_count}')
    indicator_names = sorted(list(INDICATORS.keys()))
    if indicator_count == 0:
        st.warning('No Tulipy indicators detected. Install tulipy in the same environment as Streamlit and reload.')
        with st.expander('Tulipy diagnostics'):
            st.write({
                'python_executable': sys.executable,
                'tulipy_spec_found': diag.get('has_spec', False),
                'tulipy_import_error': diag.get('import_error', ''),
            })
        selected_indicators = st.multiselect('Choose indicators', indicator_names, disabled=True)
    else:
        selected_indicators = st.multiselect('Choose indicators', indicator_names)

    indicator_specs = []
    for name in selected_indicators:
        meta = INDICATORS.get(name, {'inputs': [], 'param_names': [], 'defaults': []})
        with st.expander(f'{name} inputs: {", ".join(meta["inputs"]) or "none"}'):
            inputs_text = st.text_input(
                f'Inputs for {name} (options: {", ".join(INPUT_OPTIONS)})',
                value=_default_inputs_text(meta.get('inputs', [])),
                key=f'inputs_{name}',
            )
            params_text = st.text_input(
                f'Params for {name} ({", ".join(meta["param_names"]) or "none"})',
                value=_default_param_text(meta.get('defaults', [])),
                key=f'param_{name}',
            )
            inputs = _parse_inputs(inputs_text) or meta.get('inputs', [])
            params = _parse_params(params_text)
            if not params and meta.get('param_names'):
                params = meta.get('defaults', [])
            indicator_specs.append({
                'name': name,
                'inputs': inputs,
                'params': params,
                'prefix': name,
            })

    st.subheader('Custom indicators (JSON)')
    st.caption('Format: [{"name": "rsi", "inputs": ["close"], "params": [6], "prefix": "rsi_custom"}]')
    custom_json = st.text_area('Custom indicator specs', value='', height=120)

    if custom_json.strip():
        try:
            custom_specs = json.loads(custom_json)
            if isinstance(custom_specs, dict):
                custom_specs = [custom_specs]
            for spec in custom_specs:
                indicator_specs.append(spec)
        except Exception as exc:
            st.error(f'Invalid custom JSON: {exc}')

    build_btn = st.button('Build dataset')

if build_btn:
    with st.spinner('Downloading data and building features...'):
        data = load_data(ticker, period, interval)

    if data.empty:
        st.error('No data returned. Check ticker/period/interval.')
        st.stop()

    try:
        X_all, y, df = build_features(
            data,
            ticker,
            threshold,
            indicator_specs=indicator_specs,
            target_mode=target_mode,
            vol_window=vol_window,
            train_size=train_size,
            vol_regime_bins=(vol_low_q, vol_high_q),
        )
    except Exception as exc:
        st.error(f'Failed to build features: {exc}')
        st.stop()

    st.session_state['feature_df'] = X_all
    st.session_state['target'] = y
    st.session_state['full_df'] = df
    st.session_state['dataset_rows'] = len(df)
    st.session_state['last_build'] = {
        'ticker': ticker,
        'period': period,
        'interval': interval,
        'threshold': threshold,
        'target_mode': target_mode,
        'vol_window': vol_window,
        'vol_regime_bins': (vol_low_q, vol_high_q),
        'train_size': train_size,
    }

if 'feature_df' in st.session_state:
    st.write('Dataset summary')
    col1, col2, col3 = st.columns(3)
    col1.metric('Rows', st.session_state['dataset_rows'])

    feature_df = st.session_state['feature_df']
    y = st.session_state['target']
    target_mode = st.session_state['last_build']['target_mode']
    task_type = 'regression' if target_mode == 'volatility_regression' else 'classification'
    build_train_size = st.session_state['last_build']['train_size']

    col2.metric('Train rows', int(len(feature_df) * build_train_size))
    col3.metric('Test rows', int(len(feature_df) * (1 - build_train_size)))

    st.subheader('Full dataset (pre-model)')
    st.dataframe(st.session_state['full_df'], use_container_width=True)

    st.subheader('Feature selection')
    selected_columns = st.multiselect(
        'Choose columns to feed into models',
        options=list(feature_df.columns),
        default=list(feature_df.columns),
    )

    if st.button('Run comparison'):
        if not selected_columns:
            st.error('Select at least one feature column.')
            st.stop()

        X = feature_df[selected_columns].copy()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=build_train_size, shuffle=False
        )

        with st.spinner('Training models...'):
            results = evaluate_models(X_train, X_test, y_train, y_test, task_type=task_type)
        st.session_state['last_results'] = results
        st.session_state['last_X'] = X
        st.session_state['last_split'] = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
        }

    if 'last_results' in st.session_state:
        results = st.session_state['last_results']
        ok_results = results[results['status'] == 'ok'].copy()
        failed_results = results[results['status'] != 'ok'].copy()

        st.subheader('Model comparison')
        st.dataframe(ok_results.reset_index(drop=True), use_container_width=True)

        if not failed_results.empty:
            st.subheader('Models with errors')
            st.dataframe(failed_results[['model', 'status']].reset_index(drop=True), use_container_width=True)

        X_train = st.session_state['last_split']['X_train']
        X_test = st.session_state['last_split']['X_test']
        y_train = st.session_state['last_split']['y_train']
        y_test = st.session_state['last_split']['y_test']

        st.subheader('Target summary (test set)')
        if task_type == 'classification':
            st.write(y_test.value_counts(normalize=True).rename('ratio'))
        else:
            st.write(y_test.describe().to_frame(name='value'))

        st.subheader('Feature importance heatmap')
        st.caption('Permutation importance on the test set; higher means more predictive for the model.')
        if task_type == 'classification':
            metric = st.selectbox(
                'Importance metric',
                ['f1', 'accuracy', 'balanced_accuracy', 'precision', 'recall', 'roc_auc', 'average_precision'],
            )
        else:
            metric = st.selectbox('Importance metric', ['r2', 'mae', 'rmse'])

        repeats = st.slider('Permutation repeats', min_value=3, max_value=20, value=5, step=1)

        if st.button('Compute heatmap'):
            with st.spinner('Computing permutation importance...'):
                imp_df = compute_feature_importance(
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    metric=metric,
                    n_repeats=repeats,
                    task_type=task_type,
                )
            st.session_state['last_heatmap'] = imp_df
            if imp_df.empty:
                st.warning('No importance data available. Some models may have failed.')
            else:
                fig, ax = plt.subplots(figsize=(min(18, 0.6 * len(imp_df.columns) + 4), min(12, 0.4 * len(imp_df.index) + 4)))
                sns.heatmap(imp_df, cmap='viridis', ax=ax)
                ax.set_xlabel('Feature')
                ax.set_ylabel('Model')
                st.pyplot(fig, clear_figure=True)
        elif 'last_heatmap' in st.session_state:
            imp_df = st.session_state['last_heatmap']
            if not imp_df.empty:
                fig, ax = plt.subplots(figsize=(min(18, 0.6 * len(imp_df.columns) + 4), min(12, 0.4 * len(imp_df.index) + 4)))
                sns.heatmap(imp_df, cmap='viridis', ax=ax)
                ax.set_xlabel('Feature')
                ax.set_ylabel('Model')
                st.pyplot(fig, clear_figure=True)

        st.subheader('Rolling stability analysis')
        with st.expander('Rolling window settings'):
            train_window = st.number_input('Rolling train window (rows)', min_value=50, value=250, step=25)
            test_window = st.number_input('Rolling test window (rows)', min_value=10, value=50, step=10)
            step = st.number_input('Step size (rows)', min_value=5, value=25, step=5)
            if task_type == 'classification':
                roll_metric = st.selectbox(
                    'Rolling metric',
                    ['f1', 'accuracy', 'balanced_accuracy', 'precision', 'recall', 'roc_auc', 'average_precision'],
                )
            else:
                roll_metric = st.selectbox('Rolling metric', ['r2', 'mae', 'rmse'])
            roll_importance_metric = st.selectbox(
                'Rolling importance metric',
                ['f1', 'accuracy'] if task_type == 'classification' else ['r2', 'rmse'],
            )
            roll_repeats = st.slider('Rolling importance repeats', min_value=3, max_value=15, value=5, step=1)
            enable_time_sliced = st.checkbox('Compute time-sliced permutation importance', value=False)
            model_choices = ok_results['model'].tolist()
            selected_models = st.multiselect(
                'Models to include',
                options=model_choices,
                default=model_choices[:3] if model_choices else [],
            )

        if st.button('Run rolling analysis'):
            if train_window + test_window >= len(feature_df):
                st.error('Rolling windows are too large for the dataset length.')
            else:
                X_roll = st.session_state.get('last_X', feature_df[selected_columns].copy())
                model_map = get_models(task_type=task_type)
                if selected_models:
                    model_map = {name: model_map[name] for name in selected_models if name in model_map}
                with st.spinner('Running rolling window analysis...'):
                    rolling_df = rolling_model_scores(
                        X_roll,
                        y,
                        train_window=int(train_window),
                        test_window=int(test_window),
                        step=int(step),
                        task_type=task_type,
                        metric=roll_metric,
                        models=model_map,
                    )
                st.session_state['rolling_scores'] = rolling_df

                if enable_time_sliced:
                    with st.spinner('Computing time-sliced permutation importance...'):
                        sliced_imp = time_sliced_permutation_importance(
                            X_roll,
                            y,
                            train_window=int(train_window),
                            test_window=int(test_window),
                            step=int(step),
                            task_type=task_type,
                            metric=roll_importance_metric,
                            n_repeats=int(roll_repeats),
                            models=model_map,
                        )
                    st.session_state['rolling_importance'] = sliced_imp

        if 'rolling_scores' in st.session_state:
            rolling_df = st.session_state['rolling_scores']
            ok_scores = rolling_df[rolling_df['status'] == 'ok'].copy()
            if ok_scores.empty:
                st.warning('No rolling metrics available.')
            else:
                st.dataframe(ok_scores, use_container_width=True)
                summary_rows = []
                for model in sorted(ok_scores['model'].unique()):
                    subset = ok_scores[ok_scores['model'] == model].sort_values('window_id')
                    x_vals = np.arange(len(subset))
                    if len(x_vals) >= 2:
                        slope = np.polyfit(x_vals, subset['score'], 1)[0]
                    else:
                        slope = np.nan
                    summary_rows.append({
                        'model': model,
                        'mean_score': subset['score'].mean(),
                        'std_score': subset['score'].std(),
                        'decay_slope': slope,
                        'first_last_delta': subset['score'].iloc[-1] - subset['score'].iloc[0],
                    })
                st.subheader('Rolling performance summary')
                st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

        if 'rolling_importance' in st.session_state:
            imp_df = st.session_state['rolling_importance']
            if not imp_df.empty:
                st.subheader('Feature stability (rank correlation)')
                feature_cols = [c for c in imp_df.columns if c not in {'window_id', 'model'}]
                stability_rows = []
                for model in sorted(imp_df['model'].unique()):
                    model_df = imp_df[imp_df['model'] == model].sort_values('window_id')
                    corrs = []
                    for idx in range(1, len(model_df)):
                        r1 = model_df[feature_cols].iloc[idx - 1].rank()
                        r2 = model_df[feature_cols].iloc[idx].rank()
                        corrs.append(r1.corr(r2))
                    stability_rows.append({
                        'model': model,
                        'avg_rank_corr': float(np.nanmean(corrs)) if corrs else np.nan,
                        'min_rank_corr': float(np.nanmin(corrs)) if corrs else np.nan,
                        'max_rank_corr': float(np.nanmax(corrs)) if corrs else np.nan,
                    })
                st.dataframe(pd.DataFrame(stability_rows), use_container_width=True)

        st.subheader('Market regime detection')
        enable_regime = st.checkbox('Enable clustering-based regimes', value=False)
        if enable_regime:
            full_df = st.session_state['full_df']
            default_regime_cols = [c for c in ['realized_vol', 'volume', 'pct_change'] if c in full_df.columns]
            regime_cols = st.multiselect('Regime features', options=list(full_df.columns), default=default_regime_cols)
            n_clusters = st.slider('Regime clusters', min_value=2, max_value=6, value=3, step=1)
            if st.button('Compute regimes'):
                if not regime_cols:
                    st.error('Select at least one regime feature.')
                else:
                    with st.spinner('Clustering regimes...'):
                        regimes = compute_regimes(full_df, regime_cols, n_clusters=n_clusters)
                    st.session_state['regime_labels'] = regimes
                    st.session_state['full_df'] = full_df.assign(regime=regimes)

        if 'regime_labels' in st.session_state:
            regimes = st.session_state['regime_labels']
            full_df = st.session_state['full_df']
            name_map = regime_name_map(full_df, regime_col='regime', vol_col='realized_vol')
            st.write('Regime distribution')
            dist = regimes.value_counts(dropna=True).rename('count')
            if name_map:
                dist.index = [name_map.get(i, f'Regime {int(i)}') for i in dist.index]
            st.write(dist)

            st.subheader('Regime-aware metrics (test set)')
            model_choices = ok_results['model'].tolist()
            if model_choices:
                model_name = st.selectbox('Model for regime breakdown', model_choices)
                model = get_models(task_type=task_type).get(model_name)
                if model is not None:
                    model.fit(X_train, y_train)
                    rows = []
                    test_regimes = st.session_state['full_df'].loc[X_test.index, 'regime']
                    for regime_id in sorted(test_regimes.dropna().unique()):
                        mask = test_regimes == regime_id
                        X_r = X_test[mask]
                        y_r = y_test[mask]
                        if len(y_r) < 5:
                            continue
                        y_pred = model.predict(X_r)
                        label = name_map.get(regime_id, f'Regime {int(regime_id)}')
                        if task_type == 'classification':
                            rows.append({
                                'regime': label,
                                'count': len(y_r),
                                'accuracy': float((y_pred == y_r).mean()),
                                'precision': float(precision_score(y_r, y_pred, average='weighted', zero_division=0)),
                                'recall': float(recall_score(y_r, y_pred, average='weighted', zero_division=0)),
                                'f1': float(f1_score(y_r, y_pred, average='weighted', zero_division=0)),
                            })
                        else:
                            err = y_r - y_pred
                            denom = np.sum((y_r - y_r.mean()) ** 2)
                            rows.append({
                                'regime': label,
                                'count': len(y_r),
                                'mae': float(np.mean(np.abs(err))),
                                'rmse': float(np.sqrt(np.mean(err ** 2))),
                                'r2': float(1 - np.sum(err ** 2) / denom) if denom > 0 else np.nan,
                            })
                    if rows:
                        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        if task_type == 'regression':
            st.subheader('Volatility forecast diagnostics')
            model_choices = ok_results['model'].tolist()
            if model_choices:
                diag_model = st.selectbox('Model for diagnostics', model_choices, key='diag_model')
                model = get_models(task_type=task_type).get(diag_model)
                if model is not None:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    errors = y_test - y_pred
                    st.write('Error summary')
                    st.write(errors.describe().to_frame(name='error'))
                    fig, ax = plt.subplots(figsize=(8, 3))
                    ax.hist(errors, bins=30, color='#4c78a8', alpha=0.8)
                    ax.set_xlabel('Forecast error')
                    ax.set_ylabel('Count')
                    st.pyplot(fig, clear_figure=True)

        st.subheader('Model signal plots')
        if task_type != 'classification':
            st.info('Signal plots are only available for classification targets.')
        else:
            full_df = st.session_state['full_df']
            indicator_candidates = [c for c in selected_columns if c in full_df.columns and c != 'close']
            plot_indicators = st.multiselect(
                'Indicators to plot',
                options=indicator_candidates,
                default=indicator_candidates[:3],
            )

            model_choices = ok_results['model'].tolist()
            if model_choices:
                for model_name in model_choices:
                    with st.expander(f'Signals: {model_name}', expanded=False):
                        model = get_models(task_type=task_type).get(model_name)
                        if model is None:
                            st.warning('Model not found.')
                            continue
                        model.fit(X_train, y_train)
                        preds = pd.Series(model.predict(X_test), index=X_test.index)
                        signal_idx = preds[preds == 1].index
                        plot_df = full_df.loc[X_test.index, ['close'] + plot_indicators].copy()
                        fig = _plot_price_with_signals(
                            plot_df,
                            plot_indicators,
                            signal_idx,
                            f'{model_name} predicted signals (test set)',
                        )
                        st.pyplot(fig, clear_figure=True)

        st.subheader('Model regime plots')
        if 'regime' not in st.session_state['full_df'].columns:
            st.info('Run regime detection first to plot regime highlights.')
        else:
            full_df = st.session_state['full_df']
            name_map = regime_name_map(full_df, regime_col='regime', vol_col='realized_vol')
            model_choices = ok_results['model'].tolist()
            if model_choices:
                for model_name in model_choices:
                    with st.expander(f'Regimes: {model_name}', expanded=False):
                        plot_df = full_df.loc[X_test.index, ['close', 'regime']].copy()
                        fig = _plot_regimes(plot_df, f'{model_name} regimes (test set)', name_map=name_map)
                        st.pyplot(fig, clear_figure=True)

        st.caption('No backtesting performed. Metrics are on the test split only.')
else:
    st.info('Set inputs, choose indicators, then click "Build dataset".')
