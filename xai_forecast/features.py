# Feature column definitions shared across build_features.py, train.py, xai.py

STORE = 'CA_1'
SNAP_STATE = 'CA'

LAG_WEEKS = [1, 2, 4, 8, 52]
ROLL_WINDOWS = [4, 8, 13]

EVENT_TYPE_MAP = {'National': 1, 'Cultural': 2, 'Religious': 3, 'Sporting': 4}

FEATURE_COLS = [
    'lag_1', 'lag_2', 'lag_4', 'lag_8', 'lag_52',
    'rolling_4_mean', 'rolling_8_mean', 'rolling_13_mean', 'rolling_4_std',
    'week_of_year', 'month', 'year',
    'snap', 'has_event', 'event_type_enc',
    'sell_price', 'price_change_pct',
    'dept_enc', 'cat_enc',
]
