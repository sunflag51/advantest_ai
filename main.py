# ============================================================ 
# main.py v4.4 
# 
# アドバンテスト AI分析システム 
# 
# 追加機能: 
# ・AIターゲット 1日 / 3日 / 5日 
# ・ai/model.py v2 対応 
# ・backtest/engine.py v2 対応 
# ・iPhone接続切れ対策 
# ・段階キャッシュ 
# ・自動復帰 
# ・本格戦略バックテスト 
# ・会計監査 
# ・コピー用テキスト 
# ・CSVダウンロード 
# ============================================================ 
 
import numpy as np 
import pandas as pd 
import streamlit as st 
 
from data.stock_data import get_stock_data 
from data.market_data import ( 
    get_all_market_data, 
    get_latest_market_values, 
) 
 
from indicators.technical import ( 
    add_all_indicators, 
) 
 
from ai.features import ( 
    build_ai_features, 
) 
 
from ai.model import ( 
    StockPredictionModel, 
) 
 
from backtest.engine import ( 
    WalkForwardBacktest, 
) 
 
from backtest.report import ( 
    BacktestReport, 
) 
 

from backtest.trading_engine import ( 
    TradingBacktestEngine, 
) 
 
from strategy.entry import ( 
    EntryStrategy, 
) 
 
from strategy.risk import ( 
    RiskManager, 
) 
 
 
# ============================================================ 
# 基本設定 
# ============================================================ 
 
STOCK_CODE = "6857.T" 
 
STOCK_NAME = "アドバンテスト" 
 
INITIAL_CAPITAL = 1_000_000 
 
DATA_PERIOD = "5y" 
 
DATA_INTERVAL = "1d" 
 
 
# ============================================================ 
# キャッシュバージョン 
# 
# AIターゲット仕様変更のためv4 
# ============================================================ 
 
CACHE_VERSION = "v4.4" 
 
 
# ============================================================ 
# Streamlit 
# ============================================================ 
 
st.set_page_config( 
    page_title="アドバンテスト AI分析", 
    page_icon="📈", 
    layout="wide", 
) 
 
 

st.title( 
    "📈 アドバンテスト AI株価分析" 
) 
 
 
st.caption( 
    "AI予測・ウォークフォワード検証・" 
    "売買バックテストを段階的に実行します。" 
) 
 
 
# ============================================================ 
# 安全なfloat 
# ============================================================ 
 
def safe_float( 
    value, 
    default=None, 
): 
 
    try: 
 
        if value is None: 
            return default 
 
 
        if isinstance( 
            value, 
            pd.DataFrame, 
        ): 
 
            if value.empty: 
                return default 
 
            value = value.iloc[ 
                -1, 
                -1, 
            ] 
 
 
        if isinstance( 
            value, 
            pd.Series, 
        ): 
 
            value = pd.to_numeric( 
                value, 
                errors="coerce", 

            ).dropna() 
 
            if value.empty: 
                return default 
 
            value = value.iloc[-1] 
 
 
        if isinstance( 
            value, 
            ( 
                np.ndarray, 
                list, 
                tuple, 
            ), 
        ): 
 
            value = np.asarray( 
                value 
            ).reshape(-1) 
 
            if len(value) == 0: 
                return default 
 
            value = value[-1] 
 
 
        number = float( 
            value 
        ) 
 
 
        if not np.isfinite( 
            number 
        ): 
 
            return default 
 
 
        return number 
 
 
    except Exception: 
 
        return default 
 
 
# ============================================================ 

# Metric用 
# ============================================================ 
 
def safe_metric_number( 
    value, 
    digits=2, 
): 
 
    number = safe_float( 
        value 
    ) 
 
    if number is None: 
        return "N/A" 
 
    return f"{number:,.{digits}f}" 
 
 
# ============================================================ 
# % 
# ============================================================ 
 
def format_percent( 
    value, 
    digits=1, 
): 
 
    number = safe_float( 
        value 
    ) 
 
    if number is None: 
        return "N/A" 
 
    return ( 
        f"{number * 100:.{digits}f}%" 
    ) 
 
 
# ============================================================ 
# CSV 
# ============================================================ 
 
def dataframe_to_csv_bytes( 
    dataframe, 
): 
 
    if dataframe is None: 

 
        dataframe = pd.DataFrame() 
 
 
    if not isinstance( 
        dataframe, 
        pd.DataFrame, 
    ): 
 
        dataframe = pd.DataFrame( 
            dataframe 
        ) 
 
 
    return dataframe.to_csv( 
        index=True 
    ).encode( 
        "utf-8-sig" 
    ) 
 
 

# ============================================================
# AI評価・確率帯分析 helper
# ============================================================

def first_available(mapping, keys, default=None):
    if not isinstance(mapping, dict):
        return default
    for key in keys:
        if key in mapping and mapping.get(key) is not None:
            return mapping.get(key)
    return default


def calculate_walk_diagnostics(walk_results):
    diagnostics = {
        "actual_up_count": 0,
        "actual_down_count": 0,
        "predicted_up_count": 0,
        "predicted_down_count": 0,
        "actual_up_rate": None,
        "predicted_up_rate": None,
        "average_return_predicted_up": None,
        "median_return_predicted_up": None,
        "average_return_predicted_down": None,
        "median_return_predicted_down": None,
    }

    probability_bands = pd.DataFrame()

    if (
        walk_results is None
        or not isinstance(walk_results, pd.DataFrame)
        or walk_results.empty
    ):
        return diagnostics, probability_bands

    data = walk_results.copy()

    actual = (
        pd.to_numeric(data["Actual"], errors="coerce")
        if "Actual" in data.columns
        else pd.Series(dtype=float)
    )
    prediction = (
        pd.to_numeric(data["Prediction"], errors="coerce")
        if "Prediction" in data.columns
        else pd.Series(dtype=float)
    )
    future_return = (
        pd.to_numeric(data["Future_Return"], errors="coerce")
        if "Future_Return" in data.columns
        else pd.Series(dtype=float)
    )

    valid_actual = actual.dropna()
    if not valid_actual.empty:
        diagnostics["actual_up_count"] = int((valid_actual == 1).sum())
        diagnostics["actual_down_count"] = int((valid_actual == 0).sum())
        diagnostics["actual_up_rate"] = float((valid_actual == 1).mean())

    valid_prediction = prediction.dropna()
    if not valid_prediction.empty:
        diagnostics["predicted_up_count"] = int((valid_prediction == 1).sum())
        diagnostics["predicted_down_count"] = int((valid_prediction == 0).sum())
        diagnostics["predicted_up_rate"] = float((valid_prediction == 1).mean())

    if not future_return.empty and not prediction.empty:
        up_returns = future_return[prediction == 1].dropna()
        down_returns = future_return[prediction == 0].dropna()

        if not up_returns.empty:
            diagnostics["average_return_predicted_up"] = float(up_returns.mean())
            diagnostics["median_return_predicted_up"] = float(up_returns.median())

        if not down_returns.empty:
            diagnostics["average_return_predicted_down"] = float(down_returns.mean())
            diagnostics["median_return_predicted_down"] = float(down_returns.median())

    if all(
        column in data.columns
        for column in ["Probability_Up", "Actual", "Future_Return"]
    ):
        analysis = pd.DataFrame(
            {
                "Probability_Up": pd.to_numeric(
                    data["Probability_Up"], errors="coerce"
                ),
                "Actual": pd.to_numeric(
                    data["Actual"], errors="coerce"
                ),
                "Future_Return": pd.to_numeric(
                    data["Future_Return"], errors="coerce"
                ),
            }
        ).dropna()

        bands = [
            ("50～55%", 0.50, 0.55),
            ("55～60%", 0.55, 0.60),
            ("60～65%", 0.60, 0.65),
            ("65%以上", 0.65, None),
        ]

        rows = []
        for label, lower, upper in bands:
            if upper is None:
                mask = analysis["Probability_Up"] >= lower
            else:
                mask = (
                    (analysis["Probability_Up"] >= lower)
                    & (analysis["Probability_Up"] < upper)
                )

            band = analysis.loc[mask]
            count = len(band)

            rows.append(
                {
                    "AI確率帯": label,
                    "件数": count,
                    "実際上昇率": (
                        float((band["Actual"] == 1).mean())
                        if count else np.nan
                    ),
                    "平均将来リターン": (
                        float(band["Future_Return"].mean())
                        if count else np.nan
                    ),
                    "中央値将来リターン": (
                        float(band["Future_Return"].median())
                        if count else np.nan
                    ),
                }
            )

        probability_bands = pd.DataFrame(rows)

    return diagnostics, probability_bands


# ============================================================ 
# ① 株価データ 
# ============================================================ 
 
@st.cache_data( 
    ttl=3600, 
    show_spinner=False, 
) 
def cached_stock_data( 
    stock_code, 
    period, 
    interval, 
    cache_version, 
): 
 
    return get_stock_data( 
        stock_code=stock_code, 
        period=period, 
        interval=interval, 
    ) 
 
 
# ============================================================ 
# ② 市場データ 
# ============================================================ 
 
@st.cache_data( 

    ttl=3600, 
    show_spinner=False, 
) 
def cached_market_data( 
    period, 
    interval, 
    cache_version, 
): 
 
    market_data = ( 
        get_all_market_data( 
            period=period, 
            interval=interval, 
        ) 
    ) 
 
 
    latest_values = ( 
        get_latest_market_values( 
            period="5d", 
            interval=interval, 
        ) 
    ) 
 
 
    return ( 
        market_data, 
        latest_values, 
    ) 
 
 
# ============================================================ 
# ③ テクニカル 
# ============================================================ 
 
@st.cache_data( 
    show_spinner=False, 
) 
def cached_technical_data( 
    stock_data, 
    cache_version, 
): 
 
    return add_all_indicators( 
        stock_data 
    ) 
 
 

# ============================================================ 
# ④ AI特徴量 
# ============================================================ 
 
@st.cache_data( 
    show_spinner=False, 
) 
def cached_ai_features( 
    technical_data, 
    market_data, 
    cache_version, 
): 
 
    return build_ai_features( 
        technical_data, 
        market_data, 
    ) 
 
 
# ============================================================ 
# ⑤ AI学習 
# 
# target_horizon がキャッシュキーになる 
# ============================================================ 
 
@st.cache_data( 
    show_spinner=False, 
) 
def cached_ai_training( 
    ai_data, 
    target_horizon, 
    target_return_threshold, 
    cache_version, 
): 
 
    model = StockPredictionModel( 
 
        target_horizon= 
            target_horizon, 
 
        target_return_threshold= 
            target_return_threshold, 
    ) 
 
 
    metrics = model.train( 
        ai_data 
    ) 

 
 
    probability = ( 
        model.predict_probability( 
            ai_data 
        ) 
    ) 
 
 
    feature_importance = ( 
        model.get_feature_importance() 
    ) 
 
 
    target_info = ( 
        model.get_target_info() 
    ) 
 
 
   

    try:
        test_results = model.get_test_results()
    except Exception:
        test_results = pd.DataFrame()

    try:
        training_log = model.get_training_log()
    except Exception:
        training_log = pd.DataFrame()

    return (
        model,
        metrics,
        probability,
        feature_importance,
        target_info,
        test_results,
        training_log,
    ) 
 
 
# ============================================================ 
# ⑧ エントリー判断 
# ============================================================ 
 
@st.cache_data( 
    show_spinner=False, 
) 
def cached_entry_result( 
    ai_data, 
    probability, 
    cache_version, 
): 
 
    strategy = EntryStrategy( 
        minimum_score=6, 
        minimum_probability=0.55, 
        strong_probability=0.60, 
    ) 
 
 

    latest_row = ( 
        ai_data.iloc[-1] 
    ) 
 
 
    try: 
 
        result = strategy.evaluate( 
            row=latest_row, 
            probability=probability, 
        ) 
 
    except TypeError: 
 
        result = strategy.evaluate( 
            latest_row, 
            probability, 
        ) 
 
 
    return result 
 
 
# ============================================================ 
# ⑨ Walk Forward 
# 
# target_horizon を統一 
# ============================================================ 
 
@st.cache_data( 
    show_spinner=False, 
) 
def cached_walk_forward( 
    ai_data, 
    target_horizon, 
    target_return_threshold, 
    cache_version, 
): 
 
    backtest = WalkForwardBacktest( 
 
        initial_train_size=500, 
 
        test_size=20, 
 
        retrain_every=20, 
 
        threshold=0.50, 

 
        target_horizon= 
            target_horizon, 
 
        target_return_threshold= 
            target_return_threshold, 
    ) 
 
 
    ( 
        results, 
        metrics, 
    ) = backtest.run( 
        ai_data 
    ) 
 
 
    training_log = ( 
        backtest.get_training_log() 
    ) 
 
 
    
    try:
        calibration_table = backtest.get_calibration_table()
    except Exception:
        calibration_table = pd.DataFrame()

    try:
        stability_table = backtest.get_stability_table()
    except Exception:
        stability_table = pd.DataFrame()

    return (
        results,
        metrics,
        training_log,
        calibration_table,
        stability_table,
    ) 
 
 
# ============================================================ 
# ⑩ 旧売買バックテスト 
# ============================================================ 
 
@st.cache_data( 
    show_spinner=False, 
) 
def cached_old_backtest( 
    stock_data, 
    walk_results, 
    target_horizon, 
    cache_version, 
): 
 
    report = BacktestReport( 
 
        initial_capital= 
            INITIAL_CAPITAL, 
 

        lot_size=100, 
 
        entry_threshold=0.60, 
 
        commission_rate=0.001, 
 
        slippage_rate=0.001, 
    ) 
 
 
    try: 
 
        result = report.run( 
            stock_data, 
            walk_results, 
        ) 
 
    except TypeError: 
 
        result = report.run( 
            stock_data=stock_data, 
            walk_results=walk_results, 
        ) 
 
 
    # -------------------------------------------------------- 
    # report.py の返却形式の違いに対応 
    # -------------------------------------------------------- 
 
    if isinstance( 
        result, 
        tuple, 
    ): 
 
        if len(result) >= 3: 
 
            trades = result[0] 
            equity = result[1] 
            metrics = result[2] 
 
        elif len(result) == 2: 
 
            trades = result[0] 
            metrics = result[1] 
 
            try: 
                equity = ( 
                    report.get_equity_curve() 

                ) 
            except Exception: 
                equity = pd.DataFrame() 
 
        else: 
 
            trades = pd.DataFrame() 
            equity = pd.DataFrame() 
            metrics = {} 
 
    elif isinstance( 
        result, 
        dict, 
    ): 
 
        metrics = result 
 
        try: 
            trades = ( 
                report.get_trades() 
            ) 
        except Exception: 
            trades = pd.DataFrame() 
 
        try: 
            equity = ( 
                report.get_equity_curve() 
            ) 
        except Exception: 
            equity = pd.DataFrame() 
 
    else: 
 
        try: 
            metrics = ( 
                report.get_metrics() 
            ) 
        except Exception: 
            metrics = {} 
 
        try: 
            trades = ( 
                report.get_trades() 
            ) 
        except Exception: 
            trades = pd.DataFrame() 
 
        try: 

            equity = ( 
                report.get_equity_curve() 
            ) 
        except Exception: 
            equity = pd.DataFrame() 
 
 
    return ( 
        trades, 
        equity, 
        metrics, 
    ) 
 
 
# ============================================================ 
# Session State 
# ============================================================ 
 
SESSION_DEFAULTS = { 
 
    "analysis_completed": 
        False, 
 
    "analysis_running": 
        False, 
 
    "current_stage": 
        "未実行", 
 
    "analysis_target_horizon": 
        None, 
 
    "analysis_target_threshold": 
        None, 
 
    "stock_data": 
        None, 
 
    "market_data": 
        None, 
 
    "latest_market_values": 
        None, 
 
    "technical_data": 
        None, 
 
    "ai_data": 

        None, 
 
    "ai_model": 
        None, 
 
    "ai_metrics": 
        None, 
 
    "ai_probability": 
        None, 
 
    "feature_importance": 
        None, 
 
    "target_info": 
        None, 
 
    
    "ai_test_results":
        None,

    "ai_training_log":
        None,

    "entry_result": 
        None, 
 
    "walk_results": 
        None, 
 
    "walk_metrics": 
        None, 
 
    "walk_training_log": 
        None, 
 
    
    "walk_calibration_table":
        None,

    "walk_stability_table":
        None,

"old_trades": 
        None, 
 
    "old_equity": 
        None, 
 
    "old_metrics": 
        None, 
} 
 
 
for ( 
    key, 
    default_value, 
) in SESSION_DEFAULTS.items(): 
 
    if key not in st.session_state: 
 
        st.session_state[ 

            key 
        ] = default_value 
 
 
# ============================================================ 
# URL query parameter 
# ============================================================ 
 
try: 
 
    query_params = ( 
        st.query_params 
    ) 
 
    resume_marker = ( 
        query_params.get( 
            "analysis" 
        ) 
        == "1" 
    ) 
 
    url_horizon = ( 
        query_params.get( 
            "horizon" 
        ) 
    ) 
 
except Exception: 
 
    resume_marker = False 
 
    url_horizon = None 
 
 
# ============================================================ 
# AIターゲット選択 
# ============================================================ 
 
st.subheader( 
    "🎯 AI予測ターゲット" 
) 
 
 
horizon_options = { 
    "1日": 1, 
    "3日": 3, 
    "5日": 5, 
} 

 
 
# ============================================================ 
# URLまたは前回Sessionから初期値 
# ============================================================ 
 
default_horizon = 3 
 
 
try: 
 
    if url_horizon is not None: 
 
        candidate = int( 
            url_horizon 
        ) 
 
        if candidate in ( 
            1, 
            3, 
            5, 
        ): 
 
            default_horizon = ( 
                candidate 
            ) 
 
    elif ( 
        st.session_state[ 
            "analysis_target_horizon" 
        ] 
        in ( 
            1, 
            3, 
            5, 
        ) 
    ): 
 
        default_horizon = ( 
            st.session_state[ 
                "analysis_target_horizon" 
            ] 
        ) 
 
except Exception: 
 
    default_horizon = 3 
 

 
default_label = { 
    1: "1日", 
    3: "3日", 
    5: "5日", 
}[default_horizon] 
 
 
selected_label = st.selectbox( 
 
    "予測期間", 
 
    options=[ 
        "1日", 
        "3日", 
        "5日", 
    ], 
 
    index=[ 
        "1日", 
        "3日", 
        "5日", 
    ].index( 
        default_label 
    ), 
) 
 
 
target_horizon = ( 
    horizon_options[ 
        selected_label 
    ] 
) 
 
 
# ============================================================ 
# 最初は0% 
# 
# 将来、0.2%などと比較可能 
# ============================================================ 
 
target_return_threshold = 0.0 
 
 
st.info( 
 
    f"現在のターゲット: " 
    f"翌営業日始値でエントリーし、" 

    f"{target_horizon}営業日後の終値までの" 
    f"リターンが " 
    f"{target_return_threshold * 100:.2f}% " 
    f"を超えるかをAIが学習します。" 
) 
 
 
# ============================================================ 
# URLへターゲット保存 
# ============================================================ 
 
try: 
 
    st.query_params[ 
        "horizon" 
    ] = str( 
        target_horizon 
    ) 
 
except Exception: 
 
    pass 
 
 
# ============================================================ 
# ターゲット変更検出 
# ============================================================ 
 
stored_horizon = ( 
    st.session_state[ 
        "analysis_target_horizon" 
    ] 
) 
 
 
target_changed = ( 
 
    stored_horizon is not None 
 
    and 
 
    stored_horizon 
    != target_horizon 
) 
 
 
if target_changed: 
 

    st.warning( 
 
        "AI予測ターゲットが変更されました。" 
        "①～④のデータは再利用し、" 
        "⑤以降は選択したターゲットで" 
        "再計算します。" 
    ) 
 
 
# ============================================================ 
# 進捗表示 
# ============================================================ 
 
status_placeholder = ( 
    st.empty() 
) 
 
 
progress_placeholder = ( 
    st.empty() 
) 
 
 
def update_status( 
    stage, 
    progress=None, 
): 
 
    st.session_state[ 
        "current_stage" 
    ] = stage 
 
 
    status_placeholder.info( 
        f"現在の処理: {stage}" 
    ) 
 
 
    if progress is not None: 
 
        progress_placeholder.progress( 
            int( 
                max( 
                    0, 
                    min( 
                        100, 
                        progress, 
                    ) 

                ) 
            ) 
        ) 
 
 
# ============================================================ 
# ①～⑫ 実行 / 復帰 
# ============================================================ 
 
def run_or_restore_base_analysis(): 
 
    st.session_state[ 
        "analysis_running" 
    ] = True 
 
 
    try: 
 
        # ==================================================== 
        # ① 
        # ==================================================== 
 
        update_status( 
            "① 株価データ", 
            5, 
        ) 
 
 
        stock_data = ( 
            cached_stock_data( 
                STOCK_CODE, 
                DATA_PERIOD, 
                DATA_INTERVAL, 
                CACHE_VERSION, 
            ) 
        ) 
 
 
        st.session_state[ 
            "stock_data" 
        ] = stock_data 
 
 
        # ==================================================== 
        # ② 
        # ==================================================== 
 
        update_status( 

            "② 市場データ", 
            12, 
        ) 
 
 
        ( 
            market_data, 
            latest_market_values, 
        ) = cached_market_data( 
 
            DATA_PERIOD, 
            DATA_INTERVAL, 
            CACHE_VERSION, 
        ) 
 
 
        st.session_state[ 
            "market_data" 
        ] = market_data 
 
 
        st.session_state[ 
            "latest_market_values" 
        ] = latest_market_values 
 
 
        # ==================================================== 
        # ③ 
        # ==================================================== 
 
        update_status( 
            "③ テクニカル分析", 
            20, 
        ) 
 
 
        technical_data = ( 
            cached_technical_data( 
                stock_data, 
                CACHE_VERSION, 
            ) 
        ) 
 
 
        st.session_state[ 
            "technical_data" 
        ] = technical_data 
 

 
        # ==================================================== 
        # ④ 
        # ==================================================== 
 
        update_status( 
            "④ AI特徴量", 
            30, 
        ) 
 
 
        ai_data = ( 
            cached_ai_features( 
                technical_data, 
                market_data, 
                CACHE_VERSION, 
            ) 
        ) 
 
 
        st.session_state[ 
            "ai_data" 
        ] = ai_data 
 
 
        # ==================================================== 
        # ⑤ 
        # ==================================================== 
 
        update_status( 
            ( 
                f"⑤ AI学習 " 
                f"({target_horizon}日ターゲット)" 
            ), 
            42, 
        ) 
 
 
        ( 
            ai_model, 
            ai_metrics, 
            ai_probability, 
            feature_importance, 
            target_info, 
                ai_test_results,
        ai_training_log,
    ) = cached_ai_training( 
 
            ai_data, 
 

            target_horizon, 
 
            target_return_threshold, 
 
            CACHE_VERSION, 
        ) 
 
 
        st.session_state[ 
            "ai_model" 
        ] = ai_model 
 
 
        st.session_state[ 
            "ai_metrics" 
        ] = ai_metrics 
 
 
        st.session_state[ 
            "ai_probability" 
        ] = ai_probability 
 
 
        st.session_state[ 
            "feature_importance" 
        ] = feature_importance 
 
 
        st.session_state[ 
            "target_info" 
        ] = target_info 
 
 
        
        st.session_state[
            "ai_test_results"
        ] = ai_test_results

        st.session_state[
            "ai_training_log"
        ] = ai_training_log

        # ==================================================== 
        # ⑥～⑧ 
        # ==================================================== 
 
        update_status( 
            "⑥～⑧ 最新AI・エントリー判断", 
            52, 
        ) 
 
 
        entry_result = ( 
            cached_entry_result( 
                ai_data, 
                ai_probability, 
                CACHE_VERSION, 

            ) 
        ) 
 
 
        st.session_state[ 
            "entry_result" 
        ] = entry_result 
 
 
        # ==================================================== 
        # ⑨ 
        # ==================================================== 
 
        update_status( 
            ( 
                f"⑨ ウォークフォワード検証 " 
                f"({target_horizon}日)" 
            ), 
            65, 
        ) 
 
 
        ( 
            walk_results, 
            walk_metrics, 
            walk_training_log, 
                    walk_calibration_table,
            walk_stability_table,
        ) = cached_walk_forward( 
 
            ai_data, 
 
            target_horizon, 
 
            target_return_threshold, 
 
            CACHE_VERSION, 
        ) 
 
 
        st.session_state[ 
            "walk_results" 
        ] = walk_results 
 
 
        st.session_state[ 
            "walk_metrics" 
        ] = walk_metrics 
 
 

        st.session_state[ 
            "walk_training_log" 
        ] = walk_training_log 
 
 
        

        st.session_state[
            "walk_calibration_table"
        ] = walk_calibration_table

        st.session_state[
            "walk_stability_table"
        ] = walk_stability_table
# ==================================================== 
        # ⑩ 
        # ==================================================== 
 
        update_status( 
            "⑩ 旧売買バックテスト", 
            82, 
        ) 
 
 
        ( 
            old_trades, 
            old_equity, 
            old_metrics, 
        ) = cached_old_backtest( 
 
            stock_data, 
 
            walk_results, 
 
            target_horizon, 
 
            CACHE_VERSION, 
        ) 
 
 
        st.session_state[ 
            "old_trades" 
        ] = old_trades 
 
 
        st.session_state[ 
            "old_equity" 
        ] = old_equity 
 
 
        st.session_state[ 
            "old_metrics" 
        ] = old_metrics 
 
 
        # ==================================================== 
        # 完了 

        # ==================================================== 
 
        st.session_state[ 
            "analysis_target_horizon" 
        ] = target_horizon 
 
 
        st.session_state[ 
            "analysis_target_threshold" 
        ] = target_return_threshold 
 
 
        st.session_state[ 
            "analysis_completed" 
        ] = True 
 
 
        st.session_state[ 
            "analysis_running" 
        ] = False 
 
 
        st.session_state[ 
            "current_stage" 
        ] = "①～⑫ 完了" 
 
 
        try: 
 
            st.query_params[ 
                "analysis" 
            ] = "1" 
 
            st.query_params[ 
                "horizon" 
            ] = str( 
                target_horizon 
            ) 
 
        except Exception: 
 
            pass 
 
 
        update_status( 
            "①～⑫ 完了", 
            100, 
        ) 

 
 
        return True 
 
 
    except Exception as error: 
 
        st.session_state[ 
            "analysis_running" 
        ] = False 
 
 
        st.session_state[ 
            "analysis_completed" 
        ] = False 
 
 
        st.session_state[ 
            "current_stage" 
        ] = ( 
            "エラー" 
        ) 
 
 
        st.error( 
            "分析中にエラーが発生しました。" 
        ) 
 
 
        st.exception( 
            error 
        ) 
 
 
        return False 
 
 
# ============================================================ 
# 操作ボタン 
# ============================================================ 
 
button_col1, button_col2 = ( 
    st.columns(2) 
) 
 
 
with button_col1: 
 

    run_analysis = st.button( 
        "🚀 ①～⑫を分析・復帰", 
        use_container_width=True, 
        type="primary", 
    ) 
 
 
with button_col2: 
 
    reset_analysis = st.button( 
        "🔄 状態をリセット", 
        use_container_width=True, 
    ) 
 
 
# ============================================================ 
# リセット 
# ============================================================ 
 
if reset_analysis: 
 
    for key in SESSION_DEFAULTS: 
 
        st.session_state[ 
            key 
        ] = SESSION_DEFAULTS[ 
            key 
        ] 
 
 
    try: 
 
        st.query_params.clear() 
 
    except Exception: 
 
        pass 
 
 
    st.cache_data.clear() 
 
 
    st.success( 
        "分析状態とキャッシュを" 
        "リセットしました。" 
    ) 
 
 

    st.rerun() 
 
 
# ============================================================ 
# 手動実行 
# ============================================================ 
 
if run_analysis: 
 
    run_or_restore_base_analysis() 
 
 
# ============================================================ 
# URLから自動復帰 
# 
# targetが一致する場合のみ 
# ============================================================ 
 
elif ( 
    resume_marker 
    and 
    not st.session_state[ 
        "analysis_completed" 
    ] 
): 
 
    st.info( 
        "前回の分析を検出しました。" 
        "保存済みキャッシュから" 
        "自動復帰します。" 
    ) 
 
 
    run_or_restore_base_analysis() 
 
 
# ============================================================ 
# ターゲット変更後 
# 
# ボタンを押さなくても誤った古い結果を 
# 表示しないようにする 
# ============================================================ 
 
if ( 
    st.session_state[ 
        "analysis_completed" 
    ] 
    and 

    st.session_state[ 
        "analysis_target_horizon" 
    ] 
    != target_horizon 
): 
 
    st.warning( 
        "現在表示されている保存結果は" 
        "別のAIターゲットです。" 
        "「①～⑫を分析・復帰」を押して" 
        "再計算してください。" 
    ) 
 
 
    st.stop() 
 
 
# ============================================================ 
# 未完了 
# ============================================================ 
 
if not st.session_state[ 
    "analysis_completed" 
]: 
 
    st.info( 
        "「①～⑫を分析・復帰」を押すと" 
        "分析を開始します。" 
    ) 
 
    st.stop() 
 
 
# ============================================================ 
# Sessionから取得 
# ============================================================ 
 
stock_data = ( 
    st.session_state[ 
        "stock_data" 
    ] 
) 
 
market_data = ( 
    st.session_state[ 
        "market_data" 
    ] 
) 

 
latest_market_values = ( 
    st.session_state[ 
        "latest_market_values" 
    ] 
) 
 
technical_data = ( 
    st.session_state[ 
        "technical_data" 
    ] 
) 
 
ai_data = ( 
    st.session_state[ 
        "ai_data" 
    ] 
) 
 
ai_metrics = ( 
    st.session_state[ 
        "ai_metrics" 
    ] 
) 
 
ai_probability = ( 
    st.session_state[ 
        "ai_probability" 
    ] 
) 
 
feature_importance = ( 
    st.session_state[ 
        "feature_importance" 
    ] 
) 
 
target_info = ( 
    st.session_state[ 
        "target_info" 
    ] 
) 
 


ai_test_results = (
    st.session_state[
        "ai_test_results"
    ]
)

ai_training_log = (
    st.session_state[
        "ai_training_log"
    ]
)


entry_result = ( 
    st.session_state[ 
        "entry_result" 
    ] 
) 

 
walk_results = ( 
    st.session_state[ 
        "walk_results" 
    ] 
) 
 
walk_metrics = ( 
    st.session_state[ 
        "walk_metrics" 
    ] 
) 
 
walk_training_log = ( 
    st.session_state[ 
        "walk_training_log" 
    ] 
) 
 


walk_calibration_table = (
    st.session_state[
        "walk_calibration_table"
    ]
)

walk_stability_table = (
    st.session_state[
        "walk_stability_table"
    ]
)

old_trades = ( 
    st.session_state[ 
        "old_trades" 
    ] 
) 
 
old_equity = ( 
    st.session_state[ 
        "old_equity" 
    ] 
) 
 
old_metrics = ( 
    st.session_state[ 
        "old_metrics" 
    ] 
) 
 
 
# ============================================================ 
# 完了表示 
# ============================================================ 
 
st.success( 
    f"全分析処理が完了しています。" 
    f" 現在のAIターゲット: " 
    f"{target_horizon}日" 
) 
 

 
# ============================================================ 
# ① 株価 
# ============================================================ 
 
st.header( 
    "① 株価データ" 
) 
 
 
if ( 
    stock_data is not None 
    and 
    not stock_data.empty 
): 
 
    latest_close = safe_float( 
        stock_data[ 
            "Close" 
        ].iloc[-1] 
    ) 
 
 
    st.metric( 
        "最新終値", 
        ( 
            f"{latest_close:,.0f} 円" 
            if latest_close is not None 
            else "N/A" 
        ), 
    ) 
 
 
    st.line_chart( 
        stock_data[ 
            ["Close"] 
        ] 
    ) 
 
 
# ============================================================ 
# ② 市場 
# ============================================================ 
 
st.header( 
    "② 市場データ" 
) 
 

 
if latest_market_values: 
 
    columns = st.columns(4) 
 
 
    market_names = [ 
        "日経平均", 
        "NASDAQ", 
        "SOXX", 
        "ドル円", 
    ] 
 
 
    for index, name in enumerate( 
        market_names 
    ): 
 
        value = safe_float( 
            latest_market_values.get( 
                name 
            ) 
        ) 
 
 
        with columns[index]: 
 
            st.metric( 
                name, 
                ( 
                    f"{value:,.2f}" 
                    if value is not None 
                    else "取得なし" 
                ), 
            ) 
 
 
# ============================================================ 
# ③ テクニカル 
# ============================================================ 
 
st.header( 
    "③ テクニカル分析" 
) 
 
 
if ( 
    technical_data is not None 

    and 
    not technical_data.empty 
): 
 
    latest = ( 
        technical_data.iloc[-1] 
    ) 
 
 
    technical_columns = ( 
        st.columns(4) 
    ) 
 
 
    with technical_columns[0]: 
 
        st.metric( 
            "SMA 5", 
            safe_metric_number( 
                latest.get( 
                    "SMA_5" 
                ), 
                0, 
            ), 
        ) 
 
 
    with technical_columns[1]: 
 
        st.metric( 
            "SMA 25", 
            safe_metric_number( 
                latest.get( 
                    "SMA_25" 
                ), 
                0, 
            ), 
        ) 
 
 
    with technical_columns[2]: 
 
        st.metric( 
            "RSI 14", 
            safe_metric_number( 
                latest.get( 
                    "RSI_14" 
                ), 

                1, 
            ), 
        ) 
 
 
    with technical_columns[3]: 
 
        st.metric( 
            "MACD", 
            safe_metric_number( 
                latest.get( 
                    "MACD" 
                ), 
                2, 
            ), 
        ) 
 
 
# ============================================================ 
# ④ AI特徴量 
# ============================================================ 
 
st.header( 
    "④ AI特徴量" 
) 
 
 
if ( 
    ai_data is not None 
    and 
    not ai_data.empty 
): 
 
    st.write( 
        f"AIデータ行数: " 
        f"{len(ai_data):,}" 
    ) 
 
 
    st.write( 
        f"AIデータ列数: " 
        f"{len(ai_data.columns):,}" 
    ) 
 
 
# ============================================================
# ⑤ AIモデル
# ============================================================

st.header("⑤ AIモデル評価")

st.write(
    f"**ターゲット期間:** {target_horizon}営業日"
)

target_formula = first_available(
    target_info,
    [
        "future_return_definition",
        "formula",
        "target_definition",
    ],
    f"Close(t+{target_horizon}) / Open(t+1) - 1",
)

st.write(
    "**ターゲット式:** "
    f"`{target_formula}`"
)

st.caption(
    f"Target = 1: 将来リターンが "
    f"{target_return_threshold * 100:.2f}% を超える場合"
)

if ai_metrics:
    ai_cols = st.columns(4)

    with ai_cols[0]:
        st.metric(
            "Accuracy",
            format_percent(ai_metrics.get("accuracy")),
        )

    with ai_cols[1]:
        st.metric(
            "Precision",
            format_percent(ai_metrics.get("precision")),
        )

    with ai_cols[2]:
        st.metric(
            "Recall",
            format_percent(ai_metrics.get("recall")),
        )

    with ai_cols[3]:
        auc_value = safe_float(ai_metrics.get("auc"))
        st.metric(
            "AUC",
            f"{auc_value:.3f}"
            if auc_value is not None
            else "N/A",
        )

    audit_cols = st.columns(3)

    with audit_cols[0]:
        st.metric(
            "Train行数",
            ai_metrics.get("train_rows", "N/A"),
        )

    with audit_cols[1]:
        st.metric(
            "Purge行数",
            ai_metrics.get(
                "purge_rows",
                ai_metrics.get(
                    "purge_days",
                    target_horizon,
                ),
            ),
        )

    with audit_cols[2]:
        st.metric(
            "Test行数",
            ai_metrics.get("test_rows", "N/A"),
        )

if (
    ai_training_log is not None
    and isinstance(ai_training_log, pd.DataFrame)
    and not ai_training_log.empty
):
    with st.expander(
        "🔎 ⑤ 学習 / Purge / Test 監査"
    ):
        st.dataframe(
            ai_training_log,
            use_container_width=True,
        )

if (
    ai_test_results is not None
    and isinstance(ai_test_results, pd.DataFrame)
    and not ai_test_results.empty
):
    with st.expander(
        "📄 ⑤ 80/20テスト結果"
    ):
        st.dataframe(
            ai_test_results.tail(30),
            use_container_width=True,
        )


# ============================================================ 
# ⑥ 最新AI 
# ============================================================ 
 
st.header( 
    "⑥ 最新AI予測" 
) 
 
 
probability = safe_float( 
    ai_probability 
) 
 
 
if probability is not None: 
 
    st.metric( 

        ( 
            f"{target_horizon}日ターゲット " 
            "上昇確率" 
        ), 
        f"{probability * 100:.1f}%", 
    ) 
 
 
    if probability >= 0.60: 
 
        st.success( 
            "AI判定: 強めの上昇予測" 
        ) 
 
    elif probability >= 0.50: 
 
        st.info( 
            "AI判定: やや上昇寄り" 
        ) 
 
    else: 
 
        st.warning( 
            "AI判定: 上昇確率50%未満" 
        ) 
 
 
# ============================================================ 
# ⑦ Feature Importance 
# ============================================================ 
 
st.header( 
    "⑦ AI特徴量重要度" 
) 
 
 
if ( 
    feature_importance is not None 
    and 
    not feature_importance.empty 
): 
 
    st.dataframe( 
        feature_importance.head( 
            20 
        ), 
        use_container_width=True, 
    ) 

 
 
# ============================================================ 
# ⑧ Entry 
# ============================================================ 
 
st.header( 
    "⑧ AIエントリー判断" 
) 
 
 
if entry_result: 
 
    entry_action = ( 
        entry_result.get( 
            "action", 
            "N/A", 
        ) 
    ) 
 
 
    entry_score = ( 
        entry_result.get( 
            "score" 
        ) 
    ) 
 
 
    entry_max_score = ( 
        entry_result.get( 
            "max_score" 
        ) 
    ) 
 
 
    entry_cols = st.columns(3) 
 
 
    with entry_cols[0]: 
 
        st.metric( 
            "判断", 
            entry_action, 
        ) 
 
 
    with entry_cols[1]: 
 

        st.metric( 
            "スコア", 
            ( 
                f"{entry_score}" 
                if entry_score is not None 
                else "N/A" 
            ), 
        ) 
 
 
    with entry_cols[2]: 
 
        st.metric( 
            "AI確率", 
            ( 
                f"{probability * 100:.1f}%" 
                if probability is not None 
                else "N/A" 
            ), 
        ) 
 
 
    reasons = ( 
        entry_result.get( 
            "reasons" 
        ) 
    ) 
 
 
    if reasons: 
 
        st.write( 
            "判定理由:" 
        ) 
 
        st.write( 
            reasons 
        ) 
 
 
# ============================================================
# ⑨ Walk Forward
# ============================================================

st.header("⑨ ウォークフォワード検証")

st.write(
    f"**検証ターゲット:** {target_horizon}営業日"
)

walk_diagnostics, probability_bands = (
    calculate_walk_diagnostics(
        walk_results
    )
)

if walk_metrics:
    walk_cols = st.columns(4)

    with walk_cols[0]:
        st.metric(
            "Accuracy",
            format_percent(
                walk_metrics.get("accuracy")
            ),
        )

    with walk_cols[1]:
        st.metric(
            "Precision",
            format_percent(
                walk_metrics.get("precision")
            ),
        )

    with walk_cols[2]:
        st.metric(
            "Recall",
            format_percent(
                walk_metrics.get("recall")
            ),
        )

    with walk_cols[3]:
        walk_auc = safe_float(
            walk_metrics.get("auc")
        )
        st.metric(
            "AUC",
            f"{walk_auc:.3f}"
            if walk_auc is not None
            else "N/A",
        )

    prediction_count = int(
        safe_float(
            first_available(
                walk_metrics,
                ["prediction_count", "predictions"],
                len(walk_results)
                if walk_results is not None
                else 0,
            ),
            0,
        )
    )

    actual_up_rate = safe_float(
        first_available(
            walk_metrics,
            ["actual_up_rate"],
            walk_diagnostics.get(
                "actual_up_rate"
            ),
        )
    )

    predicted_up_rate = safe_float(
        first_available(
            walk_metrics,
            ["predicted_up_rate"],
            walk_diagnostics.get(
                "predicted_up_rate"
            ),
        )
    )

    average_up_return = safe_float(
        first_available(
            walk_metrics,
            ["average_return_predicted_up"],
            walk_diagnostics.get(
                "average_return_predicted_up"
            ),
        )
    )

    average_down_return = safe_float(
        first_available(
            walk_metrics,
            ["average_return_predicted_down"],
            walk_diagnostics.get(
                "average_return_predicted_down"
            ),
        )
    )

    median_up_return = safe_float(
        walk_diagnostics.get(
            "median_return_predicted_up"
        )
    )

    median_down_return = safe_float(
        walk_diagnostics.get(
            "median_return_predicted_down"
        )
    )

    st.write(
        "予測件数:",
        prediction_count,
    )

    count_cols = st.columns(4)

    with count_cols[0]:
        st.metric(
            "実際の上昇",
            walk_diagnostics[
                "actual_up_count"
            ],
        )

    with count_cols[1]:
        st.metric(
            "実際の下落",
            walk_diagnostics[
                "actual_down_count"
            ],
        )

    with count_cols[2]:
        st.metric(
            "上昇予測",
            walk_diagnostics[
                "predicted_up_count"
            ],
        )

    with count_cols[3]:
        st.metric(
            "下落予測",
            walk_diagnostics[
                "predicted_down_count"
            ],
        )

    rate_cols = st.columns(2)

    with rate_cols[0]:
        st.metric(
            "実際の上昇率",
            (
                f"{actual_up_rate * 100:.1f}%"
                if actual_up_rate is not None
                else "N/A"
            ),
        )

    with rate_cols[1]:
        st.metric(
            "予測上昇率",
            (
                f"{predicted_up_rate * 100:.1f}%"
                if predicted_up_rate is not None
                else "N/A"
            ),
        )

    return_cols = st.columns(4)

    with return_cols[0]:
        st.metric(
            "上昇予測時 平均",
            (
                f"{average_up_return * 100:+.2f}%"
                if average_up_return is not None
                else "N/A"
            ),
        )

    with return_cols[1]:
        st.metric(
            "上昇予測時 中央値",
            (
                f"{median_up_return * 100:+.2f}%"
                if median_up_return is not None
                else "N/A"
            ),
        )

    with return_cols[2]:
        st.metric(
            "下落予測時 平均",
            (
                f"{average_down_return * 100:+.2f}%"
                if average_down_return is not None
                else "N/A"
            ),
        )

    with return_cols[3]:
        st.metric(
            "下落予測時 中央値",
            (
                f"{median_down_return * 100:+.2f}%"
                if median_down_return is not None
                else "N/A"
            ),
        )

if (
    probability_bands is not None
    and not probability_bands.empty
):
    st.subheader("AI確率帯別分析")

    display_bands = (
        probability_bands.copy()
    )

    for column in [
        "実際上昇率",
        "平均将来リターン",
        "中央値将来リターン",
    ]:
        display_bands[column] = (
            display_bands[column].apply(
                lambda value: (
                    f"{value * 100:+.2f}%"
                    if pd.notna(value)
                    else "N/A"
                )
            )
        )

    st.dataframe(
        display_bands,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "現在のBUY最低AI確率は55%です。"
        "確率帯別の件数・実際上昇率・"
        "将来リターンを確認できます。"
    )


# ============================================================
# v2.4 初心者向け表示
# ============================================================

st.subheader("🔰 AIの安全チェック")

leakage_ok = (
    walk_metrics.get(
        "leakage_check_all_passed"
    )
    if walk_metrics
    else None
)

if leakage_ok is True:
    st.success(
        "未来情報チェック：OK\n\n"
        "AIが、まだ分からない未来の答えを"
        "使って学習していないことを確認しました。"
    )
elif leakage_ok is False:
    st.error(
        "未来情報チェック：要確認\n\n"
        "未来の情報が学習に混ざっている可能性があります。"
        "この状態ではバックテスト結果を信用せず、"
        "プログラムを確認します。"
    )
else:
    st.info(
        "未来情報チェック：結果なし"
    )

brier = safe_float(
    walk_metrics.get("brier_score")
    if walk_metrics
    else None
)

st.write("**AIが出す確率のズレを確認**")

if brier is not None:
    st.metric(
        "Brier Score",
        f"{brier:.3f}",
    )
    st.caption(
        "0に近いほど、AIが出す確率と"
        "実際の結果が合っています。"
        "この数字だけでは判断せず、"
        "下の表と一緒に確認します。"
    )
else:
    st.write("Brier Score：N/A")


if (
    walk_calibration_table is not None
    and isinstance(
        walk_calibration_table,
        pd.DataFrame,
    )
    and not walk_calibration_table.empty
):
    st.subheader(
        "🔰 AIの『○%』は本当に当たっている？"
    )

    calibration_display = (
        walk_calibration_table.copy()
        .rename(
            columns={
                "Probability_Band": "AI確率帯",
                "Count": "件数",
                "Average_Predicted_Probability": "AI平均確率",
                "Actual_Up_Rate": "実際上昇率",
                "Average_Future_Return": "平均リターン",
                "Median_Future_Return": "中央値リターン",
            }
        )
    )

    if "Period" in calibration_display.columns:
        calibration_display = calibration_display.drop(
            columns=["Period"]
        )

    for column in [
        "AI平均確率",
        "実際上昇率",
        "平均リターン",
        "中央値リターン",
    ]:
        if column in calibration_display.columns:
            calibration_display[column] = (
                calibration_display[column].apply(
                    lambda value: (
                        f"{float(value) * 100:+.2f}%"
                        if pd.notna(value)
                        else "N/A"
                    )
                )
            )

    st.dataframe(
        calibration_display,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "例：AI平均確率が60%なのに"
        "実際上昇率が40%なら、"
        "AIの『60%』をそのまま信用しにくい、"
        "という見方をします。"
    )


if (
    walk_stability_table is not None
    and isinstance(
        walk_stability_table,
        pd.DataFrame,
    )
    and not walk_stability_table.empty
):
    st.subheader(
        "🔰 昔と最近で同じ傾向か？"
    )

    stability_display = (
        walk_stability_table.copy()
        .rename(
            columns={
                "Period": "期間",
                "Probability_Band": "AI確率帯",
                "Count": "件数",
                "Average_Predicted_Probability": "AI平均確率",
                "Actual_Up_Rate": "実際上昇率",
                "Average_Future_Return": "平均リターン",
                "Median_Future_Return": "中央値リターン",
            }
        )
    )

    for column in [
        "AI平均確率",
        "実際上昇率",
        "平均リターン",
        "中央値リターン",
    ]:
        if column in stability_display.columns:
            stability_display[column] = (
                stability_display[column].apply(
                    lambda value: (
                        f"{float(value) * 100:+.2f}%"
                        if pd.notna(value)
                        else "N/A"
                    )
                )
            )

    st.dataframe(
        stability_display,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "前半と後半の両方で似た結果なら、"
        "一時期だけ偶然うまくいった可能性が"
        "少し下がります。"
        "前半だけ良く、後半で悪くなっている場合は"
        "注意して見ます。"
    )


if (
    walk_training_log is not None
    and isinstance(
        walk_training_log,
        pd.DataFrame,
    )
    and not walk_training_log.empty
):
    with st.expander(
        "🔎 ⑨ Walk-Forward 学習監査"
    ):
        st.dataframe(
            walk_training_log,
            use_container_width=True,
        )

if (
    walk_results is not None
    and not walk_results.empty
):
    st.dataframe(
        walk_results.tail(30),
        use_container_width=True,
    )


# ============================================================ 
# ⑩ 旧バックテスト 
# ============================================================ 
 
st.header( 
    "⑩ 正式な売買バックテスト（旧方式）" 
) 
 
 
if old_metrics: 
 
    old_cols = ( 
        st.columns(4) 
    ) 
 
 
    old_final = safe_float( 
        old_metrics.get( 
            "final_capital" 
        ) 

    ) 
 
 
    old_return = safe_float( 
        old_metrics.get( 
            "total_return" 
        ) 
    ) 
 
 
    old_count = ( 
        old_metrics.get( 
            "trade_count", 
            old_metrics.get( 
                "trades", 
                0, 
            ), 
        ) 
    ) 
 
 
    old_win_rate = safe_float( 
        old_metrics.get( 
            "win_rate" 
        ) 
    ) 
 
 
    with old_cols[0]: 
 
        st.metric( 
            "最終資産", 
            ( 
                f"{old_final:,.0f} 円" 
                if old_final is not None 
                else "N/A" 
            ), 
        ) 
 
 
    with old_cols[1]: 
 
        st.metric( 
            "リターン", 
            ( 
                f"{old_return * 100:+.2f}%" 
                if old_return is not None 
                else "N/A" 

            ), 
        ) 
 
 
    with old_cols[2]: 
 
        st.metric( 
            "取引回数", 
            old_count, 
        ) 
 
 
    with old_cols[3]: 
 
        st.metric( 
            "勝率", 
            ( 
                f"{old_win_rate * 100:.1f}%" 
                if old_win_rate is not None 
                else "N/A" 
            ), 
        ) 
 
 
# ============================================================ 
# ⑪ Entry summary 
# ============================================================ 
 
st.header( 
    "⑪ エントリー判断まとめ" 
) 
 
 
st.write( 
    f"現在のAIターゲット: " 
    f"**{target_horizon}日**" 
) 
 
 
st.write( 
    f"最新AI確率: " 
    f"**{probability * 100:.1f}%**" 
    if probability is not None 
    else "最新AI確率: N/A" 
) 
 
 
if entry_result: 

 
    st.write( 
        "現在の判断:", 
        entry_result.get( 
            "action", 
            "N/A", 
        ), 
    ) 
 
 
# ============================================================ 
# ⑫ Risk 
# ============================================================ 
 
st.header( 
    "⑫ リスク・資金管理" 
) 
 
 
latest_price = None 
 
 
if ( 
    stock_data is not None 
    and 
    not stock_data.empty 
): 
 
    latest_price = safe_float( 
        stock_data[ 
            "Close" 
        ].iloc[-1] 
    ) 
 
 
if latest_price is not None: 
 
    risk_manager_preview = ( 
        RiskManager( 
            lot_size=100, 
 
            risk_per_trade=0.01, 
 
            max_position_rate=0.50, 
 
            stop_loss_rate=0.05, 
 
            take_profit_rate=0.10, 

 
            commission_rate=0.001, 
 
            slippage_rate=0.001, 
        ) 
    ) 
 
 
    preview = ( 
        risk_manager_preview 
        .evaluate_trade( 
            capital= 
                INITIAL_CAPITAL, 
 
            market_price= 
                latest_price, 
        ) 
    ) 
 
 
    st.write( 
        "基準資金:", 
        f"{INITIAL_CAPITAL:,.0f} 円", 
    ) 
 
 
    st.write( 
        "売買単位:", 
        "100 株", 
    ) 
 
 
    st.write( 
        "1回の許容リスク:", 
        "1.00%", 
    ) 
 
 
    st.write( 
        "最大投資比率:", 
        "50%", 
    ) 
 
 
    st.write( 
        "現在価格での資金管理判定:", 
        ( 
            "取引可能" 

            if preview.get( 
                "can_trade", 
                False, 
            ) 
            else "見送り" 
        ), 
    ) 
 
 
# ============================================================ 
# ⑬ 本格戦略バックテスト 
# ============================================================ 
 
st.header( 
    "⑬ 本格戦略バックテスト" 
) 
 
 
st.info( 
 
    f"⑬も⑨と同じ " 
    f"**{target_horizon}日ターゲット** " 
    f"のウォークフォワードAI予測を使用します。" 
) 
 
 
# ============================================================ 
# ⑬ 設定 
# ============================================================ 
 
capital_options = { 
 
    "100万円": 
        1_000_000, 
 
    "300万円": 
        3_000_000, 
 
    "500万円": 
        5_000_000, 
 
    "1,000万円": 
        10_000_000, 
} 
 
 
position_options = { 
 

    "25%": 
        0.25, 
 
    "50%": 
        0.50, 
 
    "75%": 
        0.75, 
 
    "100%": 
        1.00, 
} 
 
 
control_col1, control_col2 = ( 
    st.columns(2) 
) 
 
 
with control_col1: 
 
    selected_capital_label = ( 
        st.selectbox( 
            "バックテスト資金", 
 
            options=list( 
                capital_options.keys() 
            ), 
 
            index=2, 
        ) 
    ) 
 
 
with control_col2: 
 
    selected_position_label = ( 
        st.selectbox( 
            "最大投資比率", 
 
            options=list( 
                position_options.keys() 
            ), 
 
            index=1, 
        ) 
    ) 
 

 
strategy_initial_capital = ( 
    capital_options[ 
        selected_capital_label 
    ] 
) 
 
 
strategy_max_position_rate = ( 
    position_options[ 
        selected_position_label 
    ] 
) 
 
 
strategy_max_position_value = ( 
 
    strategy_initial_capital 
 
    * strategy_max_position_rate 
) 
 
 
# ============================================================ 
# Engine 
# ============================================================ 
 
strategy_engine = ( 
    TradingBacktestEngine( 
 
        initial_capital= 
            strategy_initial_capital, 
 
        lot_size=100, 
 
        entry_minimum_score=6, 
 
        entry_minimum_probability=0.55, 
 
        entry_strong_probability=0.60, 
 
        stop_loss_rate=0.05, 
 
        take_profit_rate=0.10, 
 
        ai_exit_probability=0.45, 
 
        max_holding_days=10, 

 
        minimum_exit_score=3, 
 
        risk_per_trade=0.01, 
 
        max_position_rate= 
            strategy_max_position_rate, 
 
        commission_rate=0.001, 
 
        slippage_rate=0.001, 
    ) 
) 
 
 
# ============================================================ 
# ⑬実行 
# ============================================================ 
 
try: 
 
    with st.spinner( 
        "⑬ 本格戦略バックテストを計算中..." 
    ): 
 
        ( 
            strategy_trades, 
            strategy_equity, 
            strategy_metrics, 
        ) = strategy_engine.run( 
 
            stock_data= 
                stock_data, 
 
            ai_data= 
                ai_data, 
 
            walk_results= 
                walk_results, 
        ) 
 
 
    strategy_skipped = ( 
        strategy_engine 
        .get_skipped_entries() 
    ) 
 
 

    strategy_order_log = ( 
        strategy_engine 
        .get_order_log() 
    ) 
 
 
except Exception as error: 
 
    st.error( 
        "⑬ 本格戦略バックテストで" 
        "エラーが発生しました。" 
    ) 
 
    st.exception( 
        error 
    ) 
 
    st.stop() 
 
 
# ============================================================ 
# ⑬ Metrics 
# ============================================================ 
 
final_capital = safe_float( 
    strategy_metrics.get( 
        "final_capital" 
    ), 
    strategy_initial_capital, 
) 
 
 
total_profit = safe_float( 
    strategy_metrics.get( 
        "total_profit" 
    ), 
    0.0, 
) 
 
 
total_return = safe_float( 
    strategy_metrics.get( 
        "total_return" 
    ), 
    0.0, 
) 
 
 

trade_count = int( 
    safe_float( 
        strategy_metrics.get( 
            "trade_count", 
            strategy_metrics.get( 
                "trades", 
                0, 
            ), 
        ), 
        0, 
    ) 
) 
 
 
wins = int( 
    safe_float( 
        strategy_metrics.get( 
            "wins" 
        ), 
        0, 
    ) 
) 
 
 
losses = int( 
    safe_float( 
        strategy_metrics.get( 
            "losses" 
        ), 
        0, 
    ) 
) 
 
 
flat = int( 
    safe_float( 
        strategy_metrics.get( 
            "flat" 
        ), 
        0, 
    ) 
) 
 
 
win_rate = safe_float( 
    strategy_metrics.get( 
        "win_rate" 
    ) 

) 
 
 
average_profit = safe_float( 
    strategy_metrics.get( 
        "average_profit" 
    ) 
) 
 
 
average_profit_rate = safe_float( 
    strategy_metrics.get( 
        "average_profit_rate" 
    ) 
) 
 
 
gross_profit = safe_float( 
    strategy_metrics.get( 
        "gross_profit" 
    ), 
    0.0, 
) 
 
 
gross_loss = safe_float( 
    strategy_metrics.get( 
        "gross_loss" 
    ), 
    0.0, 
) 
 
 
profit_factor = safe_float( 
    strategy_metrics.get( 
        "profit_factor" 
    ) 
) 
 
 
max_drawdown = safe_float( 
    strategy_metrics.get( 
        "max_drawdown" 
    ), 
    0.0, 
) 
 
 

average_holding_days = safe_float( 
    strategy_metrics.get( 
        "average_holding_days" 
    ) 
) 
 
 
best_trade = safe_float( 
    strategy_metrics.get( 
        "best_trade" 
    ) 
) 
 
 
worst_trade = safe_float( 
    strategy_metrics.get( 
        "worst_trade" 
    ) 
) 
 
 
total_commission = safe_float( 
    strategy_metrics.get( 
        "total_commission" 
    ), 
    0.0, 
) 
 
 
total_slippage = safe_float( 
    strategy_metrics.get( 
        "total_slippage" 
    ), 
    0.0, 
) 
 
 
total_trading_cost = safe_float( 
    strategy_metrics.get( 
        "total_trading_cost" 
    ), 
    ( 
        total_commission 
        + total_slippage 
    ), 
) 
 
 

trade_profit_sum = safe_float( 
    strategy_metrics.get( 
        "trade_profit_sum" 
    ), 
    0.0, 
) 
 
 
accounting_difference = safe_float( 
    strategy_metrics.get( 
        "accounting_difference" 
    ), 
    0.0, 
) 
 
 
accounting_ok = bool( 
    strategy_metrics.get( 
        "accounting_ok", 
        False, 
    ) 
) 
 
 
skipped_count = int( 
    safe_float( 
        strategy_metrics.get( 
            "skipped_entries" 
        ), 
        len( 
            strategy_skipped 
        ), 
    ) 
) 
 
 
# ============================================================ 
# 総合成績 
# ============================================================ 
 
st.subheader( 
    "総合成績" 
) 
 
 
summary_cols = ( 
    st.columns(4) 
) 

 
 
with summary_cols[0]: 
 
    st.metric( 
        "初期資金", 
        f"{strategy_initial_capital:,.0f} 円", 
    ) 
 
 
with summary_cols[1]: 
 
    st.metric( 
        "最終資産", 
        f"{final_capital:,.0f} 円", 
    ) 
 
 
with summary_cols[2]: 
 
    st.metric( 
        "総利益", 
        f"{total_profit:+,.0f} 円", 
    ) 
 
 
with summary_cols[3]: 
 
    st.metric( 
        "総合リターン", 
        f"{total_return * 100:+.2f}%", 
    ) 
 
 
# ============================================================ 
# 取引成績 
# ============================================================ 
 
st.subheader( 
    "取引成績" 
) 
 
 
trade_cols = ( 
    st.columns(4) 
) 
 
 

with trade_cols[0]: 
 
    st.metric( 
        "取引回数", 
        trade_count, 
    ) 
 
 
with trade_cols[1]: 
 
    st.metric( 
        "勝率", 
        ( 
            f"{win_rate * 100:.1f}%" 
            if win_rate is not None 
            else "N/A" 
        ), 
    ) 
 
 
with trade_cols[2]: 
 
    st.metric( 
        "平均損益", 
        ( 
            f"{average_profit:+,.0f} 円" 
            if average_profit is not None 
            else "N/A" 
        ), 
    ) 
 
 
with trade_cols[3]: 
 
    st.metric( 
        "平均保有日数", 
        ( 
            f"{average_holding_days:.1f} 日" 
            if average_holding_days 
            is not None 
            else "N/A" 
        ), 
    ) 
 
 
# ============================================================ 
# リスク 
# ============================================================ 

 
st.subheader( 
    "リスク・効率" 
) 
 
 
risk_cols = ( 
    st.columns(4) 
) 
 
 
with risk_cols[0]: 
 
    st.metric( 
        "PF", 
        ( 
            f"{profit_factor:.2f}" 
            if profit_factor is not None 
            else "N/A" 
        ), 
    ) 
 
 
with risk_cols[1]: 
 
    st.metric( 
        "最大DD", 
        f"{max_drawdown * 100:.2f}%", 
    ) 
 
 
with risk_cols[2]: 
 
    st.metric( 
        "見送り", 
        skipped_count, 
    ) 
 
 
with risk_cols[3]: 
 
    st.metric( 
        "取引コスト", 
        f"{total_trading_cost:,.0f} 円", 
    ) 
 
 
# ============================================================ 

# 会計チェック 
# ============================================================ 
 
st.subheader( 
    "🧮 会計チェック" 
) 
 
 
audit_cols = ( 
    st.columns(4) 
) 
 
 
with audit_cols[0]: 
 
    st.metric( 
        "全取引純損益合計", 
        f"{trade_profit_sum:+,.0f} 円", 
    ) 
 
 
with audit_cols[1]: 
 
    st.metric( 
        "最終資産との差額", 
        f"{accounting_difference:+,.2f} 円", 
    ) 
 
 
with audit_cols[2]: 
 
    st.metric( 
        "累計スリッページ", 
        f"{total_slippage:,.0f} 円", 
    ) 
 
 
with audit_cols[3]: 
 
    st.metric( 
        "会計整合性", 
        ( 
            "OK" 
            if accounting_ok 
            else "要確認" 
        ), 
    ) 
 

 
if accounting_ok: 
 
    st.success( 
        "会計監査は正常です。" 
    ) 
 
else: 
 
    st.warning( 
        "会計差額があります。" 
        "取引明細を確認してください。" 
    ) 
 
 
# ============================================================ 
# Equity 
# ============================================================ 
 
if ( 
    strategy_equity is not None 
    and 
    not strategy_equity.empty 
    and 
    "Total_Equity" 
    in strategy_equity.columns 
): 
 
    st.subheader( 
        "資産推移" 
    ) 
 
 
    st.line_chart( 
        strategy_equity[ 
            ["Total_Equity"] 
        ] 
    ) 
 
 
# ============================================================ 
# コピー用テキスト 
# ============================================================ 
 
win_rate_text = ( 
    f"{win_rate * 100:.1f}%" 
    if win_rate is not None 
    else "N/A" 

) 
 
 
pf_text = ( 
    f"{profit_factor:.2f}" 
    if profit_factor is not None 
    else "N/A" 
) 
 
 
holding_text = ( 
    f"{average_holding_days:.1f} 日" 
    if average_holding_days is not None 
    else "N/A" 
) 
 
 
average_profit_text = ( 
    f"{average_profit:+,.0f} 円" 
    if average_profit is not None 
    else "N/A" 
) 
 
 
average_profit_rate_text = ( 
    f"{average_profit_rate * 100:+.2f}%" 
    if average_profit_rate is not None 
    else "N/A" 
) 
 
 
best_trade_text = ( 
    f"{best_trade:+,.0f} 円" 
    if best_trade is not None 
    else "N/A" 
) 
 
 
worst_trade_text = ( 
    f"{worst_trade:+,.0f} 円" 
    if worst_trade is not None 
    else "N/A" 
) 
 
 
copy_text = f"""【⑬ 本格戦略バックテスト結果】 
 
銘柄: {STOCK_NAME} ({STOCK_CODE}) 

 
■ AIターゲット 
予測期間: {target_horizon} 日 
エントリー想定: 翌営業日の始値 
評価価格: {target_horizon}営業日後の終値 
Target基準: {target_return_threshold * 100:.2f}% 超 
 
■ バックテスト設定 
バックテスト資金: {strategy_initial_capital:,.0f} 円 
最大投資比率: {strategy_max_position_rate * 100:.0f}% 
1回の最大投資額: {strategy_max_position_value:,.0f} 円 
売買単位: 100 株 
1回の許容リスク: 1.00% 
損切り: 5.00% 
利益確定: 10.00% 
BUY最低AI確率: 55.0% 
BUY最低スコア: 6 
AI SELL基準: 45.0% 
最大保有日数: 10 日 
売買手数料率: 0.10% 
スリッページ率: 0.10% 
 
■ 約定ルール 
当日終値でBUY / SELL判定 
翌営業日の始値で約定 
BUY約定当日はSELL判定しない 
 
■ 総合成績 
初期資金: {strategy_initial_capital:,.0f} 円 
最終資産: {final_capital:,.0f} 円 
総利益: {total_profit:+,.0f} 円 
総合リターン: {total_return * 100:+.2f}% 
 
■ 取引成績 
取引回数: {trade_count} 回 
勝ち: {wins} 回 
負け: {losses} 回 
引き分け: {flat} 回 
勝率: {win_rate_text} 
平均損益: {average_profit_text} 
平均損益率: {average_profit_rate_text} 
総利益（勝ち取引）: {gross_profit:,.0f} 円 
総損失（負け取引）: -{gross_loss:,.0f} 円 
最高取引: {best_trade_text} 
最低取引: {worst_trade_text} 
 
■ リスク・効率 
プロフィットファクター: {pf_text} 

最大ドローダウン: {max_drawdown * 100:.2f}% 
平均保有日数: {holding_text} 
資金管理で見送り: {skipped_count} 回 
累計売買手数料: {total_commission:,.0f} 円 
累計スリッページ: {total_slippage:,.0f} 円 
総取引コスト: {total_trading_cost:,.0f} 円 
 
■ 会計チェック 
全取引純損益合計: {trade_profit_sum:+,.0f} 円 
最終資産との差額: {accounting_difference:+,.2f} 円 
会計整合性: {"OK" if accounting_ok else "要確認"} 
""" 
 
 
st.subheader( 
    "📋 コピー用バックテスト結果" 
) 
 
 
st.code( 
    copy_text, 
    language=None, 
) 
 
 
# ============================================================ 
# CSV Summary 
# ============================================================ 
 
summary_dataframe = pd.DataFrame( 
 
    [ 
        { 
            "銘柄": 
                STOCK_NAME, 
 
            "コード": 
                STOCK_CODE, 
 
            "AIターゲット日数": 
                target_horizon, 
 
            "Target基準": 
                target_return_threshold, 
 
            "初期資金": 
                strategy_initial_capital, 
 

            "最終資産": 
                final_capital, 
 
            "総利益": 
                total_profit, 
 
            "総合リターン": 
                total_return, 
 
            "取引回数": 
                trade_count, 
 
            "勝ち": 
                wins, 
 
            "負け": 
                losses, 
 
            "引き分け": 
                flat, 
 
            "勝率": 
                win_rate, 
 
            "平均損益": 
                average_profit, 
 
            "平均損益率": 
                average_profit_rate, 
 
            "総利益_勝ち": 
                gross_profit, 
 
            "総損失_負け": 
                -gross_loss, 
 
            "PF": 
                profit_factor, 
 
            "最大DD": 
                max_drawdown, 
 
            "平均保有日数": 
                average_holding_days, 
 
            "見送り回数": 
                skipped_count, 
 

            "累計手数料": 
                total_commission, 
 
            "累計スリッページ": 
                total_slippage, 
 
            "総取引コスト": 
                total_trading_cost, 
 
            "全取引純損益合計": 
                trade_profit_sum, 
 
            "会計差額": 
                accounting_difference, 
 
            "会計整合性": 
                accounting_ok, 
        } 
    ] 
) 
 
 
# ============================================================ 
# Download 
# ============================================================ 
 
st.subheader( 
    "💾 CSVダウンロード" 
) 
 
 
download_col1, download_col2 = ( 
    st.columns(2) 
) 
 
 
with download_col1: 
 
    st.download_button( 
 
        "📄 成績サマリーCSV", 
 
        data=dataframe_to_csv_bytes( 
            summary_dataframe 
        ), 
 
        file_name=( 
            f"advantest_summary_" 

            f"{target_horizon}day.csv" 
        ), 
 
        mime="text/csv", 
 
        use_container_width=True, 
    ) 
 
 
with download_col2: 
 
    st.download_button( 
 
        "📄 取引明細CSV", 
 
        data=dataframe_to_csv_bytes( 
            strategy_trades 
        ), 
 
        file_name=( 
            f"advantest_trades_" 
            f"{target_horizon}day.csv" 
        ), 
 
        mime="text/csv", 
 
        use_container_width=True, 
    ) 
 
 
download_col3, download_col4 = ( 
    st.columns(2) 
) 
 
 
with download_col3: 
 
    st.download_button( 
 
        "📄 見送り明細CSV", 
 
        data=dataframe_to_csv_bytes( 
            strategy_skipped 
        ), 
 
        file_name=( 
            f"advantest_skipped_" 
            f"{target_horizon}day.csv" 

        ), 
 
        mime="text/csv", 
 
        use_container_width=True, 
    ) 
 
 
with download_col4: 
 
    st.download_button( 
 
        "📄 注文ログCSV", 
 
        data=dataframe_to_csv_bytes( 
            strategy_order_log 
        ), 
 
        file_name=( 
            f"advantest_orders_" 
            f"{target_horizon}day.csv" 
        ), 
 
        mime="text/csv", 
 
        use_container_width=True, 
    ) 
 
 
st.download_button( 
 
    "📄 資産推移CSV", 
 
    data=dataframe_to_csv_bytes( 
        strategy_equity 
    ), 
 
    file_name=( 
        f"advantest_equity_" 
        f"{target_horizon}day.csv" 
    ), 
 
    mime="text/csv", 
 
    use_container_width=True, 
) 
 
 

# ============================================================ 
# 最後 
# ============================================================ 
 
st.success( 
    f"⑬ 本格戦略バックテスト完了 " 
    f"（AIターゲット: {target_horizon}日）" 
)

# ============================================================
# ⑭ BUY確率しきい値比較実験 v4.4
# ============================================================
st.header("⑭ BUY確率しきい値比較実験")
st.info(
    "⑬の現行BUY条件55%は変更しません。"
    "BUY最低AI確率だけを 50% / 52.5% / 55% / 57.5% / 60% に変えて比較します。"
)

EXPERIMENT_THRESHOLDS = [0.50, 0.525, 0.55, 0.575, 0.60]
CURRENT_THRESHOLD = 0.55


def run_threshold_experiment(p):
    engine = TradingBacktestEngine(
        initial_capital=strategy_initial_capital,
        lot_size=100,
        entry_minimum_score=6,
        entry_minimum_probability=p,
        entry_strong_probability=0.60,
        stop_loss_rate=0.05,
        take_profit_rate=0.10,
        ai_exit_probability=0.45,
        max_holding_days=10,
        minimum_exit_score=3,
        risk_per_trade=0.01,
        max_position_rate=strategy_max_position_rate,
        commission_rate=0.001,
        slippage_rate=0.001,
    )
    return engine.run(stock_data=stock_data, ai_data=ai_data, walk_results=walk_results)


def experiment_row(label, p, m):
    return {
        "条件": label,
        "BUY最低AI確率": f"{p*100:g}%",
        "最終資産": safe_float(m.get("final_capital"), strategy_initial_capital),
        "総利益": safe_float(m.get("total_profit"), 0.0),
        "総合リターン": safe_float(m.get("total_return"), 0.0),
        "取引回数": int(safe_float(m.get("trade_count", m.get("trades", 0)), 0)),
        "勝率": safe_float(m.get("win_rate")),
        "PF": safe_float(m.get("profit_factor")),
        "最大DD": safe_float(m.get("max_drawdown"), 0.0),
        "平均保有日数": safe_float(m.get("average_holding_days")),
        "取引コスト": safe_float(m.get("total_trading_cost"), 0.0),
        "会計整合性": bool(m.get("accounting_ok", False)),
    }


def find_trade_date_column(trades):
    if trades is None or not isinstance(trades, pd.DataFrame) or trades.empty:
        return None
    for c in ["Exit_Date", "exit_date", "Sell_Date", "sell_date", "Date", "date"]:
        if c in trades.columns:
            return c
    return None


def find_trade_profit_column(trades):
    if trades is None or not isinstance(trades, pd.DataFrame) or trades.empty:
        return None
    for c in ["Net_Profit", "net_profit", "Profit", "profit", "PnL", "pnl"]:
        if c in trades.columns:
            return c
    return None


def common_period_half_summary(trades, label, boundary_date):
    date_col = find_trade_date_column(trades)
    profit_col = find_trade_profit_column(trades)
    if date_col is None or profit_col is None:
        return pd.DataFrame()
    w = trades.copy()
    w[date_col] = pd.to_datetime(w[date_col], errors="coerce")
    w[profit_col] = pd.to_numeric(w[profit_col], errors="coerce")
    w = w.dropna(subset=[date_col, profit_col]).sort_values(date_col)
    rows = []
    for period, part in [
        ("前半", w[w[date_col] <= boundary_date]),
        ("後半", w[w[date_col] > boundary_date]),
    ]:
        profits = part[profit_col]
        count = len(part)
        rows.append({
            "条件": label,
            "期間": period,
            "共通境界日": boundary_date.strftime("%Y-%m-%d"),
            "取引回数": count,
            "勝率": (float((profits > 0).sum() / count) if count else None),
            "純損益合計": (float(profits.sum()) if count else 0.0),
            "平均損益": (float(profits.mean()) if count else None),
        })
    return pd.DataFrame(rows)


try:
    experiment_results = {}
    with st.spinner("⑭ 5つのBUY確率を同じ条件で比較中..."):
        for threshold in EXPERIMENT_THRESHOLDS:
            if abs(threshold - CURRENT_THRESHOLD) < 1e-12:
                experiment_results[threshold] = (
                    strategy_trades, strategy_equity, strategy_metrics
                )
            else:
                experiment_results[threshold] = run_threshold_experiment(threshold)

    rows = []
    for threshold in EXPERIMENT_THRESHOLDS:
        trades_x, equity_x, metrics_x = experiment_results[threshold]
        label = "現行 55%" if abs(threshold - CURRENT_THRESHOLD) < 1e-12 else f"実験 {threshold*100:g}%"
        rows.append(experiment_row(label, threshold, metrics_x))
    comparison = pd.DataFrame(rows)

    display = comparison.copy()
    display["最終資産"] = display["最終資産"].map(lambda x: f"{x:,.0f} 円")
    display["総利益"] = display["総利益"].map(lambda x: f"{x:+,.0f} 円")
    display["総合リターン"] = display["総合リターン"].map(lambda x: f"{x*100:+.2f}%")
    display["勝率"] = display["勝率"].map(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "N/A")
    display["PF"] = display["PF"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    display["最大DD"] = display["最大DD"].map(lambda x: f"{x*100:.2f}%")
    display["平均保有日数"] = display["平均保有日数"].map(lambda x: f"{x:.1f} 日" if pd.notna(x) else "N/A")
    display["取引コスト"] = display["取引コスト"].map(lambda x: f"{x:,.0f} 円")
    display["会計整合性"] = display["会計整合性"].map(lambda x: "OK" if x else "要確認")

    st.subheader("5条件の全期間比較")
    st.dataframe(display, use_container_width=True, hide_index=True)
    st.caption(
        "利益だけでなく、取引回数・PF・最大DD・取引コストも一緒に確認します。"
        "この表だけで最適なしきい値を決定するものではありません。"
    )

    # --------------------------------------------------------
    # 全条件で共通の日付境界を作る
    # walk_results の予測期間そのものを前半/後半に二分します。
    # 取引件数では分けないため、全条件が同じ期間になります。
    # --------------------------------------------------------
    common_dates = None
    if isinstance(walk_results, pd.DataFrame) and not walk_results.empty:
        if isinstance(walk_results.index, pd.DatetimeIndex):
            common_dates = pd.Series(walk_results.index)
        elif "Date" in walk_results.columns:
            common_dates = pd.to_datetime(walk_results["Date"], errors="coerce")
    if common_dates is not None:
        common_dates = pd.Series(pd.to_datetime(common_dates, errors="coerce")).dropna().sort_values().reset_index(drop=True)

    st.subheader("同じ日付で分けた前半・後半比較")
    if common_dates is not None and len(common_dates) >= 2:
        boundary_position = max(0, (len(common_dates) // 2) - 1)
        boundary_date = pd.Timestamp(common_dates.iloc[boundary_position])
        half_tables = []
        for threshold in EXPERIMENT_THRESHOLDS:
            trades_x = experiment_results[threshold][0]
            label = "現行 55%" if abs(threshold - CURRENT_THRESHOLD) < 1e-12 else f"実験 {threshold*100:g}%"
            table = common_period_half_summary(trades_x, label, boundary_date)
            if not table.empty:
                half_tables.append(table)
        if half_tables:
            half_comparison = pd.concat(half_tables, ignore_index=True)
            half_display = half_comparison.copy()
            half_display["勝率"] = half_display["勝率"].map(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "N/A")
            half_display["純損益合計"] = half_display["純損益合計"].map(lambda x: f"{x:+,.0f} 円")
            half_display["平均損益"] = half_display["平均損益"].map(lambda x: f"{x:+,.0f} 円" if pd.notna(x) else "N/A")
            st.write(f"共通の前半終了日：**{boundary_date.strftime('%Y-%m-%d')}**")
            st.dataframe(half_display, use_container_width=True, hide_index=True)
            st.caption(
                "v4.4では全条件を同じ日付で前半・後半に分けています。"
                "そのため50%と60%でも比較期間がずれません。"
            )
        else:
            st.info("取引明細の日付または損益列を確認できず、前半・後半比較を表示できませんでした。")
    else:
        st.info("共通の期間境界を作れなかったため、前半・後半比較を表示できませんでした。")

    st.subheader("📋 ⑭ コピー用結果")
    copy_text_14 = (
        "【⑭ BUY確率しきい値比較実験 v4.4】\n"
        f"AIターゲット: {target_horizon}日\n"
        "現行ルール: 55%（変更なし）\n\n"
        "■ 全期間比較\n" + display.to_csv(index=False)
    )
    if 'half_display' in locals() and isinstance(half_display, pd.DataFrame):
        copy_text_14 += "\n■ 同じ日付で分けた前半・後半比較\n" + half_display.to_csv(index=False)
    st.code(copy_text_14, language=None)

    st.download_button(
        "📄 ⑭ 5条件比較CSV",
        data=dataframe_to_csv_bytes(comparison),
        file_name=f"advantest_threshold_comparison_v4_4_{target_horizon}day.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.success("⑭ v4.4比較実験完了。⑬の現行55%条件は変更していません。")

except Exception as error:
    st.error("⑭ v4.4比較実験でエラーが発生しました。⑬の結果には影響ありません。")
    st.exception(error)

