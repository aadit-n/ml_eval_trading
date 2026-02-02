import numpy as np
import pandas as pd
import yfinance as yf
import tulipy as ta


def _get_series(df, field, ticker):
    if isinstance(df.columns, pd.MultiIndex):
        return df[(field, ticker)]
    return df[field]


def _pad_to_length(values, target_len):
    return np.concatenate(([np.nan] * (target_len - len(values)), values))


def _resolve_input(name, df, data, ticker):
    if name.startswith('real'):
        name = 'close'
    if name in df.columns:
        return df[name]
    if name in ['open', 'high', 'low', 'close', 'volume']:
        return _get_series(data, name.capitalize(), ticker)
    if name == 'hlc3':
        high = _get_series(data, 'High', ticker)
        low = _get_series(data, 'Low', ticker)
        close = _get_series(data, 'Close', ticker)
        return (high + low + close) / 3.0
    if name == 'ohlc4':
        open_ = _get_series(data, 'Open', ticker)
        high = _get_series(data, 'High', ticker)
        low = _get_series(data, 'Low', ticker)
        close = _get_series(data, 'Close', ticker)
        return (open_ + high + low + close) / 4.0
    raise ValueError(f'Unknown input series: {name}')


def load_data(ticker, period, interval):
    return yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)


def apply_indicators(df, data, ticker, indicator_specs):
    for spec in indicator_specs:
        name = spec.get('name')
        inputs = spec.get('inputs', [])
        params = spec.get('params', [])
        prefix = spec.get('prefix') or name

        func = getattr(ta, name, None)
        if func is None:
            raise ValueError(f'Indicator not found in tulipy: {name}')

        input_arrays = []
        for input_name in inputs:
            series = _resolve_input(input_name, df, data, ticker)
            input_arrays.append(series.to_numpy(dtype=np.float64))

        try:
            result = func(*input_arrays, *params)
        except Exception as exc:
            raise ValueError(f'Failed indicator {name}: {exc}')

        param_str = '_'.join([str(p) for p in params]) if params else ''
        base_name = f'{prefix}_{param_str}'.rstrip('_')

        if isinstance(result, (list, tuple)):
            for idx, arr in enumerate(result):
                df[f'{base_name}_{idx+1}'] = _pad_to_length(np.asarray(arr), len(df))
        else:
            df[base_name] = _pad_to_length(np.asarray(result), len(df))

    return df


def build_features(
    data,
    ticker,
    threshold,
    indicator_specs=None,
    target_mode='return_signal',
    vol_window=20,
    train_size=0.8,
    vol_regime_bins=(0.33, 0.66),
):
    indicator_specs = indicator_specs or []

    close = _get_series(data, 'Close', ticker)
    high = _get_series(data, 'High', ticker)
    low = _get_series(data, 'Low', ticker)
    open_ = _get_series(data, 'Open', ticker)
    volume = _get_series(data, 'Volume', ticker)

    df = pd.DataFrame(index=close.index)
    df['open'] = open_
    df['high'] = high
    df['low'] = low
    df['close'] = close
    df['volume'] = volume

    df['pct_change'] = close.pct_change()
    df['pct_change2'] = close.pct_change(2)
    df['pct_change5'] = close.pct_change(5)
    df['realized_vol'] = df['pct_change'].rolling(vol_window).std()

    df = apply_indicators(df, data, ticker, indicator_specs)

    df['future_return'] = df['close'].pct_change().shift(-1)
    df['future_vol'] = df['realized_vol'].shift(-1)

    if target_mode == 'volatility_regression':
        df['target'] = df['future_vol']
    elif target_mode == 'volatility_regime':
        temp = df.dropna(subset=['future_vol'])
        train_cut = max(1, int(len(temp) * train_size))
        train_slice = temp.iloc[:train_cut]
        low_q, high_q = vol_regime_bins
        low = train_slice['future_vol'].quantile(low_q)
        high = train_slice['future_vol'].quantile(high_q)
        df['target'] = np.select(
            [df['future_vol'] <= low, df['future_vol'] <= high],
            [0, 1],
            default=2,
        )
    else:
        df['target'] = np.where(df['future_return'] > threshold, 1, 0)

    df = df.dropna()

    drop_cols = ['target', 'future_return', 'future_vol']
    X = df.drop(columns=drop_cols).copy()
    y = df['target'].copy()

    return X, y, df
