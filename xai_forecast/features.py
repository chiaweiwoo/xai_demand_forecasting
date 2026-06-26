import warnings
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────

STORE = 'CA_1'
SNAP_STATE = 'CA'

LAG_WEEKS    = [1, 2, 4, 8, 52]
ROLL_WINDOWS = [4, 8, 13]

EVENT_TYPE_MAP = {'National': 1, 'Cultural': 2, 'Religious': 3, 'Sporting': 4}

FEATURE_COLS = [
    'lag_1', 'lag_2', 'lag_4', 'lag_8', 'lag_52',
    'rolling_4_mean', 'rolling_8_mean', 'rolling_13_mean', 'rolling_4_std',
    'week_of_year', 'month', 'year',
    'snap', 'has_event', 'event_type_enc',
    'sell_price', 'price_change_pct',
    'dept_mean_sales', 'cat_mean_sales',
]

# Extra weeks fetched before training window so lag_52 is non-NaN for first training week
HISTORY_BUFFER = 52


# ── Feature engineering ───────────────────────────────────────────────────────

def compute_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute feature matrix from a raw joined DataFrame.
    raw_df columns: week, unique_id, y, snap, has_event, event_type_enc,
                    sell_price, dept_mean_sales, cat_mean_sales

    Leakage controls:
    - lag_*: shift(n) per SKU — lag_1 at week t uses sales[t-1]
    - rolling_*: shift(1).rolling(w) — excludes current week
    - sell_price NaN: ffill within item only — no backward fill, no future data
    - dept_mean_sales / cat_mean_sales: static prior over full history (deliberate mild
      lookahead accepted for stability — 7 dept / 3 cat scalars, low signal risk)
    """
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    df = raw_df.sort_values(['unique_id', 'week']).copy()

    # Lag features
    for lag in LAG_WEEKS:
        df[f'lag_{lag}'] = df.groupby('unique_id')['y'].transform(
            lambda x: x.shift(lag)
        )

    # Rolling features (shift first — excludes current week)
    for w in ROLL_WINDOWS:
        df[f'rolling_{w}_mean'] = df.groupby('unique_id')['y'].transform(
            lambda x: x.shift(1).rolling(w, min_periods=max(1, w // 2)).mean()
        )
    df['rolling_4_std'] = df.groupby('unique_id')['y'].transform(
        lambda x: x.shift(1).rolling(4, min_periods=2).std().fillna(0)
    )

    # Calendar
    df['week_of_year']   = pd.to_datetime(df['week']).dt.isocalendar().week.astype(int)
    df['month']          = pd.to_datetime(df['week']).dt.month
    df['year']           = pd.to_datetime(df['week']).dt.year
    df['snap']           = df['snap'].fillna(0).astype(int)
    df['has_event']      = df['has_event'].fillna(0).astype(int)
    df['event_type_enc'] = df['event_type_enc'].fillna(0).astype(int)

    # Price — ffill within item only (last known price, no future data).
    # Pre-launch NaNs stay NaN and are dropped via dropna(FEATURE_COLS) at training time.
    df['sell_price'] = df.groupby('unique_id')['sell_price'].transform(
        lambda x: x.ffill()
    )
    df['price_change_pct'] = df.groupby('unique_id')['sell_price'].transform(
        lambda x: x.pct_change(fill_method=None).fillna(0).clip(-1, 2)
    )

    # dept_mean_sales / cat_mean_sales — static priors, pass through as-is
    df['dept_mean_sales'] = df['dept_mean_sales'].fillna(0)
    df['cat_mean_sales']  = df['cat_mean_sales'].fillna(0)

    return df
