import importlib.util
import inspect
import re

try:
    import tulipy as ta
except Exception as exc: 
    ta = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

INPUT_OPTIONS = ['open', 'high', 'low', 'close', 'volume', 'hlc3', 'ohlc4']

_INPUT_ALIASES = {
    'open': 'open',
    'high': 'high',
    'low': 'low',
    'close': 'close',
    'volume': 'volume',
    'real': 'close',
}

_FALLBACKS = {
    'rsi': {'inputs': ['close'], 'param_names': ['period'], 'defaults': [14]},
    'sma': {'inputs': ['close'], 'param_names': ['period'], 'defaults': [14]},
    'ema': {'inputs': ['close'], 'param_names': ['period'], 'defaults': [14]},
    'wma': {'inputs': ['close'], 'param_names': ['period'], 'defaults': [14]},
    'hma': {'inputs': ['close'], 'param_names': ['period'], 'defaults': [9]},
    'kama': {'inputs': ['close'], 'param_names': ['period'], 'defaults': [10]},
    'linreg': {'inputs': ['close'], 'param_names': ['period'], 'defaults': [10]},
    'trima': {'inputs': ['close'], 'param_names': ['period'], 'defaults': [10]},
    'tsf': {'inputs': ['close'], 'param_names': ['period'], 'defaults': [10]},
    'zlema': {'inputs': ['close'], 'param_names': ['period'], 'defaults': [10]},
    'mom': {'inputs': ['close'], 'param_names': ['period'], 'defaults': [10]},
    'roc': {'inputs': ['close'], 'param_names': ['period'], 'defaults': [10]},
    'cci': {'inputs': ['high', 'low', 'close'], 'param_names': ['period'], 'defaults': [14]},
    'adx': {'inputs': ['high', 'low', 'close'], 'param_names': ['period'], 'defaults': [14]},
    'atr': {'inputs': ['high', 'low', 'close'], 'param_names': ['period'], 'defaults': [14]},
    'willr': {'inputs': ['high', 'low', 'close'], 'param_names': ['period'], 'defaults': [14]},
    'macd': {'inputs': ['close'], 'param_names': ['fast_period', 'slow_period', 'signal_period'], 'defaults': [12, 26, 9]},
    'stoch': {'inputs': ['high', 'low', 'close'], 'param_names': ['k_period', 'k_slowing', 'd_period'], 'defaults': [14, 3, 3]},
    'stochrsi': {'inputs': ['close'], 'param_names': ['period', 'fastk_period', 'fastd_period', 'fastd_matype'], 'defaults': [14, 3, 3, 0]},
    'bbands': {'inputs': ['close'], 'param_names': ['period', 'stddev'], 'defaults': [20, 2.0]},
    'vwma': {'inputs': ['close', 'volume'], 'param_names': ['period'], 'defaults': [20]},
    'typprice': {'inputs': ['high', 'low', 'close'], 'param_names': [], 'defaults': []},
    'wcprice': {'inputs': ['high', 'low', 'close'], 'param_names': [], 'defaults': []},
    'wilders': {'inputs': ['close'], 'param_names': ['period'], 'defaults': [14]},
}


def _parse_signature(name, doc_line):
    match = re.match(rf'^{re.escape(name)}\((.*?)\)\s*->', doc_line)
    if not match:
        return [], []

    raw_args = [a.strip() for a in match.group(1).split(',') if a.strip()]
    inputs = []
    params = []

    for arg in raw_args:
        arg_l = arg.lower()
        if arg_l.startswith('real') and arg_l not in _INPUT_ALIASES:
            inputs.append('close')
            continue
        if arg_l in _INPUT_ALIASES:
            inputs.append(_INPUT_ALIASES[arg_l])
        else:
            params.append(arg)

    return inputs, params


def _split_args(raw_args):
    args = []
    for arg in raw_args:
        arg = arg.strip()
        if not arg:
            continue
        if '=' in arg:
            arg = arg.split('=')[0].strip()
        args.append(arg)
    return args


def _parse_inputs_params_from_names(names):
    inputs = []
    params = []
    for arg in names:
        arg_l = arg.lower()
        if arg_l.startswith('real'):
            inputs.append('close')
        elif arg_l in _INPUT_ALIASES:
            inputs.append(_INPUT_ALIASES[arg_l])
        else:
            params.append(arg)
    return inputs, params


def _parse_from_docstring(name, doc):
    if not doc:
        return [], []
    first_line = doc.split('\n')[0].strip()
    match = re.match(rf'^{re.escape(name)}\((.*?)\)\s*->', first_line)
    if not match:
        return [], []
    raw_args = [a.strip() for a in match.group(1).split(',') if a.strip()]
    return _parse_inputs_params_from_names(_split_args(raw_args))


def _parse_from_signature(func):
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return [], [], []

    input_names = []
    param_names = []
    defaults = []

    for param in sig.parameters.values():
        name = param.name
        name_l = name.lower()
        if name_l.startswith('real'):
            input_names.append('close')
            continue
        if name_l in _INPUT_ALIASES:
            input_names.append(_INPUT_ALIASES[name_l])
            continue

        param_names.append(name)
        if param.default is not inspect.Parameter.empty:
            defaults.append(param.default)

    return input_names, param_names, defaults


def get_indicator_registry():
    if ta is None:
        return {}

    registry = {}
    for name in dir(ta):
        if name.startswith('_'):
            continue
        func = getattr(ta, name)
        if not callable(func):
            continue

        doc = (getattr(func, '__doc__', '') or '').strip()

        inputs, params, defaults = _parse_from_signature(func)
        if not inputs and not params:
            inputs, params = _parse_from_docstring(name, doc)

        fallback = _FALLBACKS.get(name, {})
        registry[name] = {
            'inputs': inputs or fallback.get('inputs', []),
            'param_names': params or fallback.get('param_names', []),
            'defaults': defaults or fallback.get('defaults', []),
        }

    return registry


def get_indicator_registry_with_diagnostics():
    registry = get_indicator_registry()
    return {
        'count': len(registry),
        'registry': registry,
        'import_error': str(_IMPORT_ERROR) if _IMPORT_ERROR else '',
        'has_spec': importlib.util.find_spec('tulipy') is not None,
    }
